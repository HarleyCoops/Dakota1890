"""Local inference entrypoint for the published Dakota LoRA adapter."""

from __future__ import annotations

import argparse

BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ADAPTER_NAME = "HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890"
DEFAULT_SYSTEM_PROMPT = "You are a Dakota language expert."
DEFAULT_PROMPT = "Translate 'my elder brother' to Dakota using the correct possessive suffix."


def main() -> None:
    """Run a single local generation against the Dakota adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Run local inference for the Dakota adapter.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the model.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of generated tokens.")
    args = parser.parse_args()

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading tokenizer: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print(f"Loading adapter: {ADAPTER_NAME}")
    model = PeftModel.from_pretrained(model, ADAPTER_NAME)

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]

    print("Generating response...")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
