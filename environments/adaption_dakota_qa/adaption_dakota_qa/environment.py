"""Verifiers environment for HarleyCooper/adaption-dakota-english-qa.

This is intentionally a new environment rather than a light edit of the older
Dakota grammar environment because the Adaption Labs dataset has a different
schema: question/answer plus enhanced_prompt/enhanced_completion and source
metadata.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.request import urlopen

import verifiers as vf
from datasets import Dataset

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGED_DATASET = PACKAGE_ROOT / "data" / "train_0.jsonl"
HF_RAW_DATASET_URL = (
    "https://huggingface.co/datasets/HarleyCooper/adaption-dakota-english-qa/"
    "resolve/main/data/train_0.jsonl"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful Dakota language assistant working from Stephen Return "
    "Riggs' 1890 Dakota-English Dictionary. Answer the user's question directly. "
    "Preserve Dakota orthography exactly, including characters such as ć, š, ŋ, "
    "ḣ, ṡ, ǧ, á, é, í, ó, ú. If a Dakota form is requested, include the form and "
    "a concise grammatical explanation. Do not invent unsupported paradigms."
)

DAKOTA_CHARS = set("ćšŋḣṡǧáéíóúķśṅźėčžʼ’")
STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "they", "their", "there",
    "would", "could", "should", "which", "what", "when", "where", "using",
    "into", "about", "also", "because", "within", "while", "your", "you",
    "are", "for", "of", "to", "in", "on", "or", "is", "be", "a", "an", "it",
    "as", "by", "its", "not", "than", "then", "these", "those", "has", "have",
}


def _normalize(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().replace("`", " ").replace("*", " ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compact(text: Any) -> str:
    return re.sub(r"[^0-9a-zA-Zćšŋḣṡǧáéíóúķśṅźėčž'ʼ-]+", "", _normalize(text))


def _char_f1(prediction: str, target: str) -> float:
    pred_chars = Counter(_compact(prediction))
    target_chars = Counter(_compact(target))
    if not target_chars:
        return 0.0
    overlap = sum(min(pred_chars[ch], target_chars[ch]) for ch in target_chars)
    precision = overlap / max(sum(pred_chars.values()), 1)
    recall = overlap / max(sum(target_chars.values()), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _word_set(text: str) -> set[str]:
    words = re.findall(r"[0-9a-zA-Zćšŋḣṡǧáéíóúķśṅźėčž'ʼ-]{3,}", _normalize(text))
    return {w.strip("-'ʼ") for w in words if w.strip("-'ʼ") and w not in STOPWORDS}


def _dakota_terms(answer: str) -> list[str]:
    terms: list[str] = []
    # Prefer explicitly marked forms first.
    for pattern in [r"`([^`]{2,80})`", r"\*\*([^*]{2,80})\*\*"]:
        for match in re.findall(pattern, answer):
            clean = match.strip().strip(".,;:()[]{}")
            if clean:
                terms.append(clean)
    # Add tokens that look Dakota-specific or segmented dictionary forms.
    for token in re.findall(r"[A-Za-zćšŋḣṡǧáéíóúķśṅźėčž'ʼ-]{3,}", answer):
        if any(ch in DAKOTA_CHARS for ch in token.lower()) or "-" in token or "'" in token or "ʼ" in token:
            terms.append(token.strip(".,;:()[]{}"))
    # Stable de-dup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = _compact(term)
        if key and key not in seen:
            seen.add(key)
            out.append(term)
    return out[:12]


def _extract_response(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "")
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content") or "")
    return str(completion or "")


def _read_jsonl(path_or_url: str | Path) -> list[dict[str, Any]]:
    if isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url, timeout=60) as response:
            lines = response.read().decode("utf-8").splitlines()
    else:
        lines = Path(path_or_url).expanduser().read_text(encoding="utf-8").splitlines()

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSONL line %s: %s", line_no, exc)
            continue
        if obj.get("question") and obj.get("answer"):
            records.append(obj)
    return records


def _build_records(
    raw_rows: Sequence[dict[str, Any]],
    prompt_field: str = "enhanced_prompt",
    max_examples: int = -1,
    source_language: str | None = "english",
    include_enhanced_completion: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        if source_language and str(row.get("source_language") or "").lower() != source_language.lower():
            continue
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        enhanced_prompt = str(row.get("enhanced_prompt") or "").strip()
        prompt_text = enhanced_prompt if prompt_field == "enhanced_prompt" and enhanced_prompt else question
        terms = _dakota_terms(answer)
        info: dict[str, Any] = {
            "pair_id": row.get("pair_id", idx),
            "source_files": row.get("source_files"),
            "source_pages": row.get("source_pages"),
            "source_language": row.get("source_language"),
            "question": question,
            "prompt_field": prompt_field,
            "dakota_terms": terms,
            "target_keywords": sorted(_word_set(answer), key=len, reverse=True)[:20],
        }
        if include_enhanced_completion:
            info["enhanced_completion"] = row.get("enhanced_completion")
        records.append(
            {
                "id": f"adaption_dakota_{int(row.get('pair_id', idx)):05d}",
                "question": prompt_text,
                "answer": answer,
                "info": info,
            }
        )
    if max_examples and max_examples > 0:
        records = records[:max_examples]
    return records


class AdaptionDakotaRubric(vf.Rubric):
    """Continuous reward for Dakota-English QA answers.

    The reward intentionally avoids an all-or-nothing exact match because the
    target answers are natural language. RL needs intra-group variance, so the
    rubric combines exactness with term, character, keyword, and length signals.
    """

    def __init__(self) -> None:
        super().__init__(
            funcs=[
                self.exact_reward,
                self.dakota_term_reward,
                self.char_f1_reward,
                self.keyword_reward,
                self.length_reward,
            ],
            weights=[0.20, 0.30, 0.25, 0.15, 0.10],
        )

    async def exact_reward(self, completion: Any, answer: str, **_: Any) -> float:
        return float(_normalize(_extract_response(completion)) == _normalize(answer))

    async def dakota_term_reward(self, completion: Any, answer: str, info: dict[str, Any] | None = None, **_: Any) -> float:
        response = _normalize(_extract_response(completion))
        terms = (info or {}).get("dakota_terms") or _dakota_terms(answer)
        if not terms:
            return 1.0
        hits = 0
        for term in terms:
            if _compact(term) and _compact(term) in _compact(response):
                hits += 1
        return hits / len(terms)

    async def char_f1_reward(self, completion: Any, answer: str, **_: Any) -> float:
        return float(_char_f1(_extract_response(completion), answer))

    async def keyword_reward(self, completion: Any, answer: str, info: dict[str, Any] | None = None, **_: Any) -> float:
        response_words = _word_set(_extract_response(completion))
        keywords = set((info or {}).get("target_keywords") or sorted(_word_set(answer), key=len, reverse=True)[:20])
        if not keywords:
            return 1.0
        return len(response_words & keywords) / len(keywords)

    async def length_reward(self, completion: Any, answer: str, **_: Any) -> float:
        response = _extract_response(completion).strip()
        if not response:
            return 0.0
        answer_len = max(len(answer), 1)
        ratio = len(response) / answer_len
        # Smoothly reward responses within roughly 0.35x-2.5x the target length.
        if 0.35 <= ratio <= 2.5:
            return 1.0
        distance = min(abs(math.log(max(ratio, 1e-6) / 0.35)), abs(math.log(max(ratio, 1e-6) / 2.5)))
        return max(0.0, 1.0 - distance / 2.0)


def build_dataset(
    dataset_path: str | Path | None = None,
    prompt_field: str = "enhanced_prompt",
    max_examples: int = -1,
    source_language: str | None = "english",
    seed: int = 42,
    shuffle: bool = True,
    include_enhanced_completion: bool = False,
) -> Dataset:
    source = dataset_path or PACKAGED_DATASET
    if not Path(source).exists() if not isinstance(source, str) or not source.startswith(("http://", "https://")) else False:
        source = HF_RAW_DATASET_URL
    records = _build_records(
        _read_jsonl(source),
        prompt_field=prompt_field,
        max_examples=max_examples,
        source_language=source_language,
        include_enhanced_completion=include_enhanced_completion,
    )
    if not records:
        raise ValueError("No usable Adaption Dakota QA records found")
    dataset = Dataset.from_list(records)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset


def _split_dataset(dataset: Dataset, eval_fraction: float, eval_examples: int, seed: int) -> tuple[Dataset, Dataset | None]:
    if eval_fraction <= 0 or len(dataset) < 2:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    eval_dataset = split["test"]
    if eval_examples and eval_examples > 0:
        eval_dataset = eval_dataset.select(range(min(eval_examples, len(eval_dataset))))
    return split["train"], eval_dataset


def load_environment(
    dataset_path: str | Path | None = None,
    prompt_field: str = "enhanced_prompt",
    max_examples: int = -1,
    eval_fraction: float = 0.10,
    eval_examples: int = 64,
    source_language: str | None = "english",
    system_prompt: str | None = None,
    sampling_args: dict[str, Any] | None = None,
    seed: int = 42,
    shuffle: bool = True,
    include_enhanced_completion: bool = False,
) -> vf.Environment:
    dataset = build_dataset(
        dataset_path=dataset_path,
        prompt_field=prompt_field,
        max_examples=max_examples,
        source_language=source_language,
        seed=seed,
        shuffle=shuffle,
        include_enhanced_completion=include_enhanced_completion,
    )
    train_dataset, eval_dataset = _split_dataset(dataset, eval_fraction, eval_examples, seed)
    rubric = AdaptionDakotaRubric()
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        rubric=rubric,
        sampling_args=sampling_args,
        message_type="chat",
    )
