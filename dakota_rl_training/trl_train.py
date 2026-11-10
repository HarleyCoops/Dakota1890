#!/usr/bin/env python3
"""Dakota1890 GRPO training via Hugging Face TRL (single-GPU, open-source).

This is the HF/TRL counterpart to ``tinker_train.py``. It trains a LoRA adapter with
``trl.GRPOTrainer`` using the same deterministic Dakota grammar reward (Reward v2) that
the packaged verifiers environment uses, then optionally pushes the adapter to the Hub.

The reward is deliberately dependency-free here so it runs with just ``trl`` +
``transformers`` + ``peft`` (no ``verifiers``/``tinker``). It mirrors
``DakotaGrammarRubric`` in ``environments/dakota_grammar_translation`` — keep the two in
sync when changing reward weights.

Example:
    python dakota_rl_training/trl_train.py \
        --model Qwen/Qwen3-0.6B \
        --dataset dakota_rl_training/datasets/grammar_tasks_complete.jsonl \
        --output-dir dakota_rl_training/outputs/trl_qwen06b \
        --max-steps 200 --learning-rate 4e-5 --lora-rank 32 \
        --push-to-hub HarleyCooper/Qwen3-0.6B-Dakota-TRL-pilot
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Deterministic Dakota reward (Reward v2) — mirrors DakotaGrammarRubric.
# ---------------------------------------------------------------------------

SPECIAL_CHARS = set("ćšŋḣṡáéíóúķśṅźėčžʼ")

DIFFICULTY_MULTIPLIERS = {
    "easy": 1.0,
    "basic": 1.0,
    "medium": 1.2,
    "intermediate": 1.2,
    "advanced": 1.5,
    "hard": 1.5,
    "expert": 2.0,
}

# Reward v2 component weights (sum to 1.0):
#   semantic answer-presence 0.35, char overlap 0.25, pattern 0.20, affix 0.15, length 0.05
W_SEMANTIC = 0.35
W_CHAR = 0.25
W_PATTERN = 0.20
W_AFFIX = 0.15
W_LENGTH = 0.05


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _char_f1(prediction: str, target: str) -> float:
    pred_chars = Counter(_normalize(prediction).replace(" ", ""))
    target_chars = Counter(_normalize(target).replace(" ", ""))
    if not target_chars:
        return 0.0
    overlap = sum(min(pred_chars[ch], target_chars[ch]) for ch in target_chars)
    precision = overlap / max(sum(pred_chars.values()), 1)
    recall = overlap / max(sum(target_chars.values()), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _semantic_presence(prediction: str, answer: str) -> float:
    pred_norm = _normalize(prediction)
    ans_norm = _normalize(answer)
    if pred_norm == ans_norm:
        return 1.0
    if ans_norm and ans_norm in pred_norm:
        return 1.0
    ans_words = set(ans_norm.split())
    if not ans_words:
        return 1.0
    pred_words = set(pred_norm.split())
    return len(ans_words & pred_words) / len(ans_words)


def _affix_accuracy(prediction: str, required_affixes: list[str]) -> float:
    if not required_affixes:
        return 1.0
    correct = 0
    for affix in required_affixes:
        affix_clean = affix.strip("-")
        if affix.startswith("-") and not affix.endswith("-"):
            if re.search(rf"\w+{re.escape(affix_clean)}\b", prediction):
                correct += 1
        elif affix.endswith("-") and not affix.startswith("-"):
            if re.search(rf"\b{re.escape(affix_clean)}\w+", prediction):
                correct += 1
        elif affix_clean in prediction:
            correct += 1
    return correct / len(required_affixes)


def _pattern_match(prediction: str, info: dict) -> float:
    score = 0.0
    pattern = (info or {}).get("verification_pattern") or (info or {}).get("pattern")
    if pattern:
        pattern_text = str(pattern)
        literal_candidates = [pattern_text]
        if ":" in pattern_text:
            literal_candidates.append(pattern_text.split(":", 1)[1].strip())
        try:
            if re.search(pattern_text, prediction, flags=re.IGNORECASE):
                score = 1.0
        except re.error:
            pass
        if score < 1.0:
            normalized_prediction = _normalize(prediction)
            if any(
                _normalize(c) and _normalize(c) in normalized_prediction
                for c in literal_candidates
            ):
                score = 1.0
    hints = (info or {}).get("hints", []) or []
    if score < 1.0 and hints:
        covered = sum(1 for h in hints if h.lower() in prediction.lower())
        score = max(score, covered / len(hints))
    # Neutral 1.0 when no pattern/hint constraint applies (mirrors affix N/A behavior).
    if not pattern and not hints:
        return 1.0
    return float(score)


def _length_quality(prediction: str, answer: str, max_ratio: float = 8.0, min_chars: int = 2) -> float:
    n_pred = len(prediction.strip())
    if n_pred < min_chars:
        return 0.0
    n_ans = len(answer.strip())
    if n_ans > 0 and n_pred > max_ratio * n_ans:
        return max(0.0, (max_ratio * n_ans) / n_pred)
    return 1.0


def score_dakota(response: str, answer: str, info: dict | None) -> float:
    """Return the deterministic composite reward for one (response, answer, info) triple."""
    info = info or {}
    semantic = _semantic_presence(response, answer)
    char = _char_f1(response, answer)
    pattern = _pattern_match(response, info)
    affix = _affix_accuracy(response, info.get("required_affixes", []) or [])
    length = _length_quality(response, answer)
    composite = (
        W_SEMANTIC * semantic
        + W_CHAR * char
        + W_PATTERN * pattern
        + W_AFFIX * affix
        + W_LENGTH * length
    )
    difficulty = str(info.get("difficulty", "medium")).lower()
    return composite * DIFFICULTY_MULTIPLIERS.get(difficulty, 1.0)


def dakota_reward(completions: list[str], **kwargs) -> list[float]:
    """TRL reward function: reads dataset ``answer`` and ``info`` columns from kwargs."""
    answers = kwargs.get("answer")
    if answers is None:
        raise ValueError("The TRL dataset must include an 'answer' column.")
    infos = kwargs.get("info") or [None] * len(completions)
    return [score_dakota(c, a, i) for c, a, i in zip(completions, answers, infos)]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dakota_dataset(path: str | Path, max_examples: int = -1) -> Dataset:
    """Load a Dakota grammar_tasks_*.jsonl file into a TRL-compatible Dataset.

    Keeps ``prompt`` (the user instruction), ``answer`` (reference), and ``info``
    (per-task metadata: task_type, difficulty, pattern, required_affixes, hints).
    """
    ds = load_dataset("json", data_files=str(path), split="train")
    if max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Dakota grammar RL policy with TRL GRPO.")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model (small for a single-GPU pilot).")
    p.add_argument("--dataset", default="dakota_rl_training/datasets/grammar_tasks_complete.jsonl")
    p.add_argument("--output-dir", default="dakota_rl_training/outputs/trl_qwen06b")
    p.add_argument("--max-examples", type=int, default=-1, help="Cap on training examples (-1 = all).")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=8, help="Rollouts per prompt (GRPO group size).")
    p.add_argument("--max-completion-length", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient.")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=0, help="Eval every N steps (0 = disabled).")
    p.add_argument("--seed", type=int, default=1890)
    p.add_argument("--push-to-hub", default=None, help="Optional HF repo id to push the adapter to.")
    p.add_argument("--hub-token", default=None, help="HF token (defaults to HF_TOKEN env).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dakota_dataset(args.dataset, args.max_examples)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=None,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        beta=args.beta,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=True,
        report_to=["none"],
        remove_unused_columns=False,  # keep 'answer'/'info' for the reward function
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[dakota_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub(args.push_to_hub, token=args.hub_token)
        print(f"Adapter pushed to https://huggingface.co/{args.push_to_hub}")

    print(f"Training complete. Adapter saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
