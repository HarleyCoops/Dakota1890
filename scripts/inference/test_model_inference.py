#!/usr/bin/env python3
"""
Improved inference test for the trained Dakota Grammar RL model.
Uses the proper chat format with system prompt as used during training.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEFAULT_SYSTEM_PROMPT = (
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

def test_inference():
    """Test inference with the trained model using proper chat format."""
    model_name = "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"
    
    print(f"Loading model: {model_name}")
    print("=" * 70)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    print(" Model loaded successfully!")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Dtype: {next(model.parameters()).dtype}")
    print()
    
    # Test prompts (using actual task formats from training)
    test_cases = [
        {
            "name": "Translation Task",
            "prompt": "Translate to Dakota: Hello"
        },
        {
            "name": "Grammar Completion",
            "prompt": "Complete: Wićaŋyaŋpi kta čha"
        },
        {
            "name": "Morphology",
            "prompt": "Add the affix -pi to: wićaŋyaŋ"
        },
        {
            "name": "Translation (Dakota to English)",
            "prompt": "Translate to English: Háu"
        }
    ]
    
    print("Running inference tests with proper chat format...")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        print("-" * 70)
        
        # Format as chat messages
        messages = format_chat_messages(DEFAULT_SYSTEM_PROMPT, test_case['prompt'])
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate with better parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Shorter for concise responses
                temperature=0.3,  # Lower temperature for more focused output
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        # The response should start after the user message
        if "assistant" in response.lower() or "</s>" in response:
            # Try to extract just the generated part
            parts = response.split("assistant")
            if len(parts) > 1:
                generated_text = parts[-1].strip()
            else:
                # Fallback: extract after the last user message
                generated_text = response.split(test_case['prompt'])[-1].strip()
        else:
            generated_text = response[len(text):].strip()
        
        # Clean up
        generated_text = generated_text.split("</s>")[0].strip()
        generated_text = generated_text.split("<|im_end|>")[0].strip()
        
        print(f"Response: {generated_text}")
        print()
    
    print("=" * 70)
    print(" Inference test completed!")
    print(f"\nModel available at: https://huggingface.co/{model_name}")

if __name__ == "__main__":
    try:
        test_inference()
    except Exception as e:
        print(f"\n Error during inference: {e}")
        import traceback
        traceback.print_exc()
