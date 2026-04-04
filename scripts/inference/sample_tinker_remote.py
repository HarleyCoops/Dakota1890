"""Sample a published Dakota checkpoint from Thinking Machines Tinker."""

import os
import argparse
import tinker
from tinker import ServiceClient
from tinker import types

import sys
import io

# Force UTF-8 for stdout/stderr to handle Dakota characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Sample from a published Tinker model")
    parser.add_argument("--checkpoint", required=True, help="Tinker checkpoint path (tinker://...)")
    parser.add_argument("--sampler-path", help="Tinker sampler weights path (tinker://.../sampler_weights/...)")
    parser.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Base model name")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--prompt", default="Translate 'my elder brother' to Dakota.", help="Prompt to sample")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    args = parser.parse_args()

    print(f"Connecting to Tinker with checkpoint: {args.checkpoint}")
    
    service_client = ServiceClient()
    
    print("Initializing LoRA client...")
    client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank
    )
    
    # Get tokenizer
    print("Getting tokenizer...")
    tokenizer = client.get_tokenizer()
    
    # Tokenize prompt
    # Note: Using chat template if available would be better, but raw prompt for now
    # Ideally: messages = [{"role": "user", "content": args.prompt}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # But let's stick to raw prompt if simple
    
    # Check if we can use apply_chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        # Qwen usually supports chat template
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"Formatted Prompt: {text}")
        except Exception as e:
            print(f"Chat template failed, using raw: {e}")
            text = args.prompt
    else:
        text = args.prompt

    tokens = tokenizer.encode(text)
    print(f"Token count: {len(tokens)}")

    print("Loading checkpoint state...")
    try:
        client.load_state(args.checkpoint).result()
        print("Checkpoint loaded.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print("Creating sampling client...")
    try:
        # Use provided sampler path or derive from checkpoint path
        sampler_path = args.sampler_path
        if not sampler_path:
            # Try to infer: replace 'weights' with 'sampler_weights' if present
            if '/weights/' in args.checkpoint:
                sampler_path = args.checkpoint.replace('/weights/', '/sampler_weights/')
                print(f"Inferred sampler path: {sampler_path}")
            else:
                print("Warning: Could not infer sampler path and none provided. Using checkpoint path.")
                sampler_path = args.checkpoint

        # Create sampler from the loaded training state
        # Pass the sampler weights path as the model_path
        sampler = client.create_sampling_client(sampler_path)
        print("Sampling client created.")
        
        print(f"\nSampling prompt: {args.prompt}")
        print("-" * 50)
        
        # Create input
        chunk = types.EncodedTextChunk(tokens=tokens)
        model_input = types.ModelInput(chunks=[chunk])
        
        # Create sampling params
        params = types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        
        # Sample
        future = sampler.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=params
        )
        
        response = future.result()
        
        # Handle SampleResponse structure
        if hasattr(response, 'sequences'):
            seq = response.sequences[0]
            if hasattr(seq, 'tokens'):
                decoded = tokenizer.decode(seq.tokens)
                print(f"Response: {decoded}")
            else:
                print(f"Response sequence: {seq}")
        elif hasattr(response, 'candidates'):
            # Candidate likely has .token_ids or .text if decoded?
            # Usually raw tokens are returned
            cand = response.candidates[0]
            # print(f"Candidate: {cand}")
            
            if hasattr(cand, 'token_ids'):
                decoded = tokenizer.decode(cand.token_ids)
                print(f"Response: {decoded}")
            else:
                print(f"Response candidate: {cand}")
        else:
             print(f"Response object: {response}")
        
    except Exception as e:
        print(f"Error during sampling setup/execution: {e}")

if __name__ == "__main__":
    main()
