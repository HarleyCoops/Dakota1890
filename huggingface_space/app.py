# app.py

import os
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix OMP_NUM_THREADS warning
os.environ["OMP_NUM_THREADS"] = "1"

MODEL_ID = "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL"

SYSTEM_PROMPT = (
    "You are a Dakota language expert specializing in the 1890 Dakota-English Dictionary grammar. "
    "Translate or explain each prompt concisely while preserving Dakota orthography exactly, "
    "including special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.) and cultural/grammatical nuance."
)


def format_chat_messages(system_prompt: str, user_message: str) -> list:
    """Format messages in chat format for Qwen models - matches training format."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


@spaces.GPU(duration=60)
def infer(prompt, max_tokens, temperature, top_p):
    # Load weights the first time the GPU worker spins up
    if "tokenizer" not in infer.__dict__:
        infer.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        infer.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    
    tok = infer.tokenizer
    model = infer.model
    
    if not prompt.strip():
        return "Please provide a prompt."
    
    try:
        # Format as chat messages (matches training format)
        messages = format_chat_messages(SYSTEM_PROMPT, prompt)
        
        # Apply chat template (CRITICAL: This matches the format used during RL training)
        formatted_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tok(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Check debug mode early
        debug_mode = os.getenv("DEBUG_INFERENCE", "false").lower() == "true"
        debug_info = []  # Collect debug info to return
        
        # Generate with parameters matching scripts/inference/test_model_inference.py
        # Use torch.no_grad() for efficiency
        import torch
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 64),  # Match scripts/inference/test_model_inference.py
                temperature=0.3,  # Match scripts/inference/test_model_inference.py
                top_p=0.9,  # Match scripts/inference/test_model_inference.py
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                # Removed repetition_penalty and no_repeat_ngram_size to match working version
            )
        
        if debug_mode:
            debug_info.append("=== DEBUG MODE ENABLED ===")
            debug_info.append("Generation completed")
            debug_info.append(f"Output tensor shape: {output.shape}")
            print("DEBUG - Generation completed")
            print(f"DEBUG - Output tensor shape: {output.shape}")
            print(f"DEBUG - Output dtype: {output.dtype}")
        
        # Decode full response
        decoded_full = tok.decode(output[0], skip_special_tokens=False)
        decoded = tok.decode(output[0], skip_special_tokens=True)
        
        if debug_mode:
            debug_info.append("=== DEBUG MODE ENABLED ===")
            debug_info.append(f"Formatted prompt length: {len(formatted_prompt)}")
            debug_info.append(f"Output shape: {output.shape}")
            debug_info.append(f"Input length: {inputs['input_ids'].shape[1]}")
            debug_info.append(f"Output length: {output.shape[1]}")
            print(f"DEBUG - Formatted prompt length: {len(formatted_prompt)}")
            print(f"DEBUG - Full decoded (with special tokens): {decoded_full[:200]}")
            print(f"DEBUG - Decoded (no special tokens): {decoded[:200]}")
            print(f"DEBUG - Output shape: {output.shape}")
            print(f"DEBUG - Input length: {inputs['input_ids'].shape[1]}")
            print(f"DEBUG - Output length: {output.shape[1]}")
        
        # Extract only the generated part (after the prompt)
        # Use the same method as scripts/inference/test_model_inference.py
        prompt_tokens = inputs['input_ids'][0]
        generated_tokens = output[0][len(prompt_tokens):]
        
        if debug_mode:
            debug_info.append(f"Prompt tokens: {len(prompt_tokens)}")
            debug_info.append(f"Generated tokens: {len(generated_tokens)}")
            if len(generated_tokens) > 0:
                debug_info.append(f"First 10 generated token IDs: {generated_tokens[:10].tolist()}")
            print(f"DEBUG - Prompt tokens: {len(prompt_tokens)}")
            print(f"DEBUG - Generated tokens: {len(generated_tokens)}")
            if len(generated_tokens) > 0:
                print(f"DEBUG - First 10 generated token IDs: {generated_tokens[:10].tolist()}")
        
        # Method 1: Extract tokens directly (most reliable)
        if len(generated_tokens) > 0:
            generated_text = tok.decode(generated_tokens, skip_special_tokens=True)
            if debug_mode:
                debug_info.append(f"Decoded from tokens: {repr(generated_text[:200])}")
                print(f"DEBUG - Decoded from tokens: {repr(generated_text[:200])}")
        else:
            # No new tokens - model didn't generate anything
            if debug_mode:
                debug_info.append("WARNING: No new tokens generated!")
                print("DEBUG - WARNING: No new tokens generated!")
            generated_text = ""
        
        # Method 2: Fallback to string slicing used in scripts/inference/test_model_inference.py
        if not generated_text or len(generated_text.strip()) == 0:
            # Try the method from scripts/inference/test_model_inference.py
            if "assistant" in decoded.lower() or "</s>" in decoded:
                parts = decoded.split("assistant")
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
                else:
                    # Fallback: extract after the prompt text
                    generated_text = decoded[len(formatted_prompt):].strip()
            else:
                # Simple slice after prompt
                generated_text = decoded[len(formatted_prompt):].strip()
            
            if debug_mode:
                debug_info.append(f"Fallback extraction: {repr(generated_text[:200])}")
                print(f"DEBUG - Fallback extraction: {repr(generated_text[:200])}")
        
        # Clean up special tokens and reasoning tags
        generated_text = generated_text.split("</s>")[0].strip()
        generated_text = generated_text.split("<|im_end|>")[0].strip()
        
        # Remove reasoning tags that the model might generate (more aggressive)
        reasoning_tags = [
            "<think>",
            "</think>",
            "<think>",
            "</think>",
            "<reasoning>",
            "</reasoning>",
        ]
        for tag in reasoning_tags:
            if tag in generated_text:
                # Split on tag and take everything after the last occurrence
                parts = generated_text.split(tag)
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
                else:
                    generated_text = generated_text.replace(tag, "").strip()
        
        # Remove any remaining XML-like tags
        import re
        generated_text = re.sub(r'<[^>]+>', '', generated_text)
        
        # Don't remove too aggressively - keep Dakota special characters
        # Only remove if they're clearly artifacts
        if generated_text.startswith("<|im_start|>"):
            generated_text = generated_text[13:].strip()
        if generated_text.startswith("<|endoftext|>"):
            generated_text = generated_text[13:].strip()
        
        # Remove prompt artifacts but be careful
        if generated_text.startswith("User:"):
            generated_text = generated_text[5:].strip()
        if generated_text.startswith("Assistant:"):
            generated_text = generated_text[10:].strip()
        
        # Filter out garbage Unicode characters (full-width, weird symbols)
        # Keep valid Dakota characters but remove obvious garbage
        import re
        # Remove full-width characters that aren't Dakota
        # Keep: ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú and normal ASCII
        # Remove: full-width N (Ｎ), weird Chinese chars, etc.
        # This is a heuristic - be careful not to remove valid Dakota
        lines = generated_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that are mostly garbage characters
            if line.strip():
                # Count non-ASCII, non-Dakota characters
                dakota_chars = set('ćšŋḣṡáéíóú')
                ascii_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-')
                valid_chars = ascii_chars | dakota_chars
                char_count = len([c for c in line if c in valid_chars or c.isspace()])
                total_chars = len([c for c in line if not c.isspace()])
                if total_chars == 0 or char_count / max(total_chars, 1) > 0.5:  # At least 50% valid chars
                    cleaned_lines.append(line)
        generated_text = '\n'.join(cleaned_lines).strip()
        
        # Remove repeated single characters (like "ＮＮＮＮ")
        generated_text = re.sub(r'(.)\1{4,}', '', generated_text)  # Remove 5+ repeats
        
        if debug_mode:
            debug_info.append(f"After cleanup: {repr(generated_text[:200])}")
            debug_info.append(f"Final length: {len(generated_text)}")
            debug_info.append(f"Full decoded (first 500 chars): {decoded[:500]}")
            print(f"DEBUG - After cleanup: {repr(generated_text[:200])}")
            print(f"DEBUG - Final length: {len(generated_text)}")
        
        # Return result
        if not generated_text or len(generated_text.strip()) == 0:
            # Return debug info if enabled
            if debug_mode:
                debug_output = "\n".join(debug_info)
                return f"[DEBUG: Empty result]\n\n{debug_output}\n\nFull decoded text:\n{decoded[:1000]}"
            return "[Model returned an empty string - enable DEBUG_INFERENCE=true to diagnose]"
        
        # If debug mode, prepend debug info to response
        if debug_mode:
            debug_output = "\n".join(debug_info)
            return f"[DEBUG INFO]\n{debug_output}\n\n[RESPONSE]\n{generated_text}"
        
        return generated_text
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # Always log errors for debugging
        print(f"ERROR during generation: {str(e)}")
        print(f"Traceback: {error_details}")
        return f"Error during generation: {str(e)}\n\nEnable DEBUG_INFERENCE=true for more details."


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(lines=4, label="Prompt", placeholder="Translate into Dakota: ..."),
        gr.Slider(32, 128, value=60, step=8, label="Max new tokens"),
        gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p"),
    ],
    outputs=gr.Textbox(lines=6, label="Model output"),
    title="Dakota Grammar RL – Qwen3‑0.6B",
    description="Composite-reward RL model trained on Dakota grammar tasks. "
                "Use concise prompts (translations, grammar explanations, etc.).",
)


if __name__ == "__main__":
    demo.launch()
