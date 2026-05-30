"""Remote Thinking Machines inference for the latest Dakota1890 sampler."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

ADAPTER_NAME = "HarleyCooper/Qwen3.6-35B-A3B-Dakota1890-GRPO"
BASE_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_MODEL_PATH = (
    "tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/"
    "sampler_weights/final"
)
DEFAULT_SYSTEM_PROMPT = "Answer Dakota grammar tasks concisely. Return only the answer."
DEFAULT_PROMPT = "Translate 'my elder brother' to Dakota. Return only the answer."


def build_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool,
) -> str:
    """Format a prompt with the tokenizer's native chat template when available."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"


def main() -> int:
    """Run one remote generation against a Tinker sampler checkpoint."""
    import tinker

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run remote inference against the Dakota1890 Tinker sampler."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Tinker sampler URI.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send to the sampler.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling value.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to request.")
    parser.add_argument("--enable-thinking", action="store_true", help="Allow Qwen thinking tokens.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON output.")
    args = parser.parse_args()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=args.model_path)
    tokenizer = sampling_client.get_tokenizer()

    formatted_prompt = build_chat_prompt(
        tokenizer,
        args.system_prompt,
        args.prompt,
        enable_thinking=args.enable_thinking,
    )
    prompt_tokens = tokenizer.encode(formatted_prompt)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    response = sampling_client.sample(
        prompt=model_input,
        num_samples=args.num_samples,
        sampling_params=sampling_params,
    ).result()

    decoded = [
        tokenizer.decode(sequence.tokens, skip_special_tokens=True).strip()
        for sequence in response.sequences
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "backend": "tinker",
                    "model_path": args.model_path,
                    "prompt": args.prompt,
                    "responses": decoded,
                    "stop_reasons": [
                        str(sequence.stop_reason) for sequence in response.sequences
                    ],
                    "prompt_tokens": len(prompt_tokens),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for text in decoded:
            print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
