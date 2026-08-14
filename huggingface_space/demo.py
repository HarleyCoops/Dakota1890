"""CPU-safe helpers for the grant-clean Dakota1890 Hugging Face Space.

Live generation is optional and goes to a remote Tinker sampler. This module
does not load a 30B model, transformers, or ``spaces.GPU``.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import tinker
except ImportError:  # optional: example bank still works without Tinker
    tinker = None


# Grant-clean 30B Tinker session (W&B cebp9acs). In-repo logs do not include a
# cebp9acs checkpoints.jsonl, so sampler_weights/final is the same convention
# used by other Dakota Tinker runs. Override with TINKER_SAMPLER_PATH.
GRANT_CLEAN_TINKER_SESSION = "dc44ca83-ce9e-5c91-a38d-0e866549f397:train:0"
CONVENTION_SAMPLER_PATH = (
    f"tinker://{GRANT_CLEAN_TINKER_SESSION}/sampler_weights/final"
)
BASE_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
WANDB_RUN_URL = (
    "https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/cebp9acs"
)
REPO_URL = "https://github.com/HarleyCoops/Dakota1890"

# Same last-line / boxed instruction used by the grant-clean train env.
DEFAULT_SYSTEM_PROMPT = (
    "You are a Dakota language expert specializing in the 1890 Dakota-English "
    "Dictionary grammar. Translate or explain each prompt concisely while "
    "preserving Dakota orthography exactly, including special characters "
    "(ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.) and cultural/grammatical nuance. "
    "Put the final answer alone on the last line, or wrap it in \\boxed{...}."
)

DEFAULT_MAX_TOKENS = 64
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 42

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer\s*(?:is|:))\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

TASK_TYPE_LABELS = {
    "reverse_translation": "EN→Dakota",
    "word_translation": "Dakota→English",
    "sentence_translation": "Dakota→English",
    "morphology": "Morphology",
    "identify_pattern": "Pattern",
    "syntax": "Syntax",
    "affix_insertion": "Affix",
    "multi_step_morphology": "Morphology",
}

PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLES_PATH = PACKAGE_DIR / "examples.jsonl"


def extract_final_answer(text: str) -> str:
    """Return the span that should be shown as the model's answer.

    Priority matches the grant-clean scorer: last ``\\boxed{...}``, then last
    ``final answer is/`` line, then the last non-empty line.
    """
    if not text or not str(text).strip():
        return ""
    boxed = list(BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip()
    finals = list(FINAL_ANSWER_RE.finditer(text))
    if finals:
        return finals[-1].group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) > 1:
        return lines[-1]
    return lines[0]


def resolve_sampler_path() -> str:
    """Prefer TINKER_SAMPLER_PATH; otherwise the grant-clean convention URI."""
    override = os.environ.get("TINKER_SAMPLER_PATH", "").strip()
    if override:
        return override
    return CONVENTION_SAMPLER_PATH


def tinker_key_configured() -> bool:
    return bool(os.environ.get("TINKER_API_KEY", "").strip())


def live_inference_available() -> bool:
    return tinker_key_configured() and tinker is not None


def live_status_message() -> str:
    if live_inference_available():
        return (
            "Live Tinker sampling is on. Prompts go to the remote grant-clean "
            f"sampler (`{resolve_sampler_path()}`), not a model hosted in this Space."
        )
    if tinker_key_configured() and tinker is None:
        return (
            "TINKER_API_KEY is set, but the `tinker` package is not installed. "
            "The example bank and gold answers still work."
        )
    return (
        "Live Tinker sampling is off (no TINKER_API_KEY). Browse the frozen "
        "holdout examples and gold answers below. Set TINKER_API_KEY and "
        "optionally TINKER_SAMPLER_PATH as Space secrets to enable remote generation."
    )


def load_examples(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the curated holdout v1 example bank shipped with the Space."""
    source = path or EXAMPLES_PATH
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def example_label(row: dict[str, Any]) -> str:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    task_type = str(info.get("task_type") or "task")
    difficulty = str(info.get("difficulty") or "unspecified")
    kind = TASK_TYPE_LABELS.get(task_type, task_type)
    preview = _prompt_preview(str(row.get("prompt") or ""))
    return f"{kind} · {difficulty} · {preview}"


def _prompt_preview(prompt: str) -> str:
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if not lines:
        return "(empty prompt)"
    # Skip the instruction line when a short target line follows.
    if len(lines) >= 2 and lines[0].lower().startswith("translate"):
        return lines[1][:80]
    if lines[-1].lower().startswith("transform or analyze:"):
        return lines[-1].split(":", 1)[-1].strip()[:80] or lines[0][:80]
    return lines[0][:80]


def gold_framing_note(row: dict[str, Any]) -> str:
    """Honest note when reverse-translation gold is an English glossary term."""
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    if info.get("task_type") != "reverse_translation":
        return ""
    gold = str(row.get("answer") or "").strip()
    if gold and gold.isascii() and gold.replace(" ", "").replace("-", "").isalpha():
        return (
            "Holdout gold is an English glossary term from Riggs 1890, "
            "not a Dakota surface form. This is the class of item where "
            "eval exact match rose; EN→Dakota remains weak."
        )
    return (
        "EN→Dakota is still weak on the grant-clean run. Treat the gold as "
        "an 1890 Riggs scaffold for speakers to correct, not fluent Dakota."
    )


def format_example_card(row: dict[str, Any]) -> str:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    task_type = str(info.get("task_type") or "")
    difficulty = str(info.get("difficulty") or "")
    kind = TASK_TYPE_LABELS.get(task_type, task_type)
    note = gold_framing_note(row)
    note_block = f"\n\n*{note}*" if note else ""
    return (
        f"**Task type:** {kind} (`{task_type}`)  \n"
        f"**Difficulty:** {difficulty}  \n"
        f"**Task id:** `{row.get('task_id', '')}`  \n"
        f"**Source:** frozen holdout v1 (`grammar_tasks_heldout.jsonl`, "
        f"cebp9acs eval){note_block}"
    )


def build_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    enable_thinking: bool = False,
) -> str:
    """Format a prompt with the sampler tokenizer chat template when available."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"


@lru_cache(maxsize=4)
def get_sampling_client(model_path: str) -> Any:
    if tinker is None:
        raise RuntimeError("The tinker package is not installed.")
    service_client = tinker.ServiceClient()
    return service_client.create_sampling_client(model_path=model_path)


def sample_remote(
    prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_path: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Call the remote Tinker sampler. Raises if Tinker is unavailable."""
    if tinker is None:
        raise RuntimeError("The tinker package is not installed.")
    if not tinker_key_configured():
        raise RuntimeError("TINKER_API_KEY is not set.")
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    sampler_path = model_path or resolve_sampler_path()
    sampling_client = get_sampling_client(sampler_path)
    tokenizer = sampling_client.get_tokenizer()
    formatted_prompt = build_chat_prompt(
        tokenizer,
        system_prompt,
        prompt,
        enable_thinking=False,
    )
    prompt_tokens = tokenizer.encode(formatted_prompt)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampling_params = tinker.SamplingParams(
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=-1,
        seed=int(seed),
    )
    response = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()
    raw = tokenizer.decode(
        response.sequences[0].tokens,
        skip_special_tokens=True,
    ).strip()
    extracted = extract_final_answer(raw)
    return {
        "ok": True,
        "raw": raw,
        "extracted": extracted,
        "model_path": sampler_path,
        "prompt_tokens": len(prompt_tokens),
        "stop_reason": str(response.sequences[0].stop_reason),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "base_model": BASE_MODEL_NAME,
    }


def run_live_or_explain(
    prompt: str,
    gold: str = "",
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[str, str, str]:
    """UI helper: live sample when configured, otherwise an honest offline note."""
    if not live_inference_available():
        extracted = ""
        raw = live_status_message()
        comparison = (
            f"Gold (holdout v1): {gold}" if gold else "No gold for a custom prompt."
        )
        return extracted, raw, comparison
    try:
        result = sample_remote(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        extracted = result["extracted"]
        raw = result["raw"]
        if gold:
            match = extracted.strip() == gold.strip()
            comparison = (
                f"Gold (holdout v1): {gold}\n"
                f"Extracted span matches gold: {'yes' if match else 'no'}\n"
                f"Sampler: {result['model_path']}"
            )
        else:
            comparison = (
                f"No gold for a custom prompt.\nSampler: {result['model_path']}"
            )
        return extracted, raw, comparison
    except Exception as exc:  # UI must surface remote failures
        return (
            "",
            f"Live inference failed ({type(exc).__name__}): {exc}",
            f"Gold (holdout v1): {gold}" if gold else "No gold for a custom prompt.",
        )
