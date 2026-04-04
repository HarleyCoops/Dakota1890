#!/usr/bin/env python3
"""
Local test script that exactly matches the HuggingFace Space setup.
Use this to debug inference issues locally before deploying to Space.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

MODEL_ID = "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"

SYSTEM_PROMPT = (
    "You are a Dakota language expert specializing in the 1890 Dakota-English Dictionary grammar. "
    "Translate or explain each prompt concisely while preserving Dakota orthography exactly, "
    "including special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.) and cultural/grammatical nuance."
)

def format_chat_messages(system_prompt: str, user_message: str) -> list:
    """Format messages in chat format for Qwen models."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

def test_space_inference(prompt: str, max_tokens: int = 48, temperature: float = 0.1, top_p: float = 0.8):
    """Test inference with exact Space parameters."""
    print(f"Loading model: {MODEL_ID}")
    print("=" * 70)
    
    # Load tokenizer and model (matching Space)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Model loaded on device: {next(model.parameters()).device}")
    print(f"Dtype: {next(model.parameters()).dtype}")
    print()
    
    # Format as chat messages (matches Space)
    messages = format_chat_messages(SYSTEM_PROMPT, prompt)
    
    # Apply chat template (matches Space)
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"Formatted prompt length: {len(formatted_prompt)}")
    print(f"Formatted prompt preview: {formatted_prompt[:200]}...")
    print()
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    
    # Generate with Space parameters
    print(f"\nGenerating with:")
    print(f"  max_new_tokens: {max_tokens}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  repetition_penalty: 1.3")
    print(f"  no_repeat_ngram_size: 3")
    print()
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
    
    print(f"Output shape: {output.shape}")
    print(f"Output length: {output.shape[1]} tokens")
    
    # Decode full response
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nFull decoded response ({len(decoded)} chars):")
    print("-" * 70)
    print(decoded)
    print("-" * 70)
    
    # Extract only the generated part (matching Space logic)
    prompt_tokens = inputs['input_ids'][0]
    generated_tokens = output[0][len(prompt_tokens):]
    
    print(f"\nPrompt tokens: {len(prompt_tokens)}")
    print(f"Generated tokens: {len(generated_tokens)}")
    
    if len(generated_tokens) > 0:
        print(f"First 10 generated token IDs: {generated_tokens[:10].tolist()}")
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\nDecoded from tokens only:")
        print(f"  Raw: {repr(generated_text[:200])}")
    else:
        generated_text = ""
        print("WARNING: No new tokens generated!")
    
    # Cleanup (matching Space)
    import re
    generated_text = generated_text.split("</s>")[0].strip()
    generated_text = generated_text.split("<|im_end|>")[0].strip()
    
    # Remove reasoning tags
    reasoning_tags = [
        "<think>",
        "</think>",
        "<reasoning>",
        "</reasoning>",
    ]
    for tag in reasoning_tags:
        if tag in generated_text:
            parts = generated_text.split(tag)
            if len(parts) > 1:
                generated_text = parts[-1].strip()
            else:
                generated_text = generated_text.replace(tag, "").strip()
    
    # Remove XML-like tags
    generated_text = re.sub(r'<[^>]+>', '', generated_text)
    
    # Filter garbage characters
    lines = generated_text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip():
            dakota_chars = set('ćšŋḣṡáéíóú')
            ascii_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-')
            valid_chars = ascii_chars | dakota_chars
            char_count = len([c for c in line if c in valid_chars or c.isspace()])
            total_chars = len([c for c in line if not c.isspace()])
            if total_chars == 0 or char_count / max(total_chars, 1) > 0.5:
                cleaned_lines.append(line)
    generated_text = '\n'.join(cleaned_lines).strip()
    
    # Remove repeated characters
    generated_text = re.sub(r'(.)\1{4,}', '', generated_text)
    
    print(f"\nAfter cleanup:")
    print(f"  Length: {len(generated_text)}")
    print(f"  Text: {repr(generated_text)}")
    print()
    print("=" * 70)
    print("FINAL RESPONSE:")
    print(generated_text)
    print("=" * 70)
    
    return generated_text

if __name__ == "__main__":
    import sys
    
    # Test prompts
    test_prompts = [
        "Translate to Dakota: Hello",
        "Translate to Dakota: That is my horse",
    ]
    
    if len(sys.argv) > 1:
        # Use command line prompt
        prompt = " ".join(sys.argv[1:])
        test_space_inference(prompt)
    else:
        # Test with default prompts
        for prompt in test_prompts:
            print(f"\n{'='*70}")
            print(f"Testing: {prompt}")
            print('='*70)
            test_space_inference(prompt)
            print("\n\n")

