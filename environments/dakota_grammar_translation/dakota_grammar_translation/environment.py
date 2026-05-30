"""PrimeIntellect verifiers-compatible environment for Dakota grammar and translation tasks.

Version 0.1.8 includes verbose penalty reward to prevent overly long responses.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.request import urlopen

import verifiers as vf
from datasets import Dataset
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a Dakota language expert specializing in the 1890 Dakota-English Dictionary grammar. "
    "Translate or explain each prompt concisely while preserving Dakota orthography exactly, "
    "including special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.) and cultural/grammatical nuance."
)

PACKAGE_ROOT = Path(__file__).resolve().parent
# Default dataset packaged with environment
DEFAULT_DATASET = PACKAGE_ROOT / "data" / "grammar_tasks_complete.jsonl"
# Fallback sample dataset (for development/testing)
FALLBACK_DATASET = PACKAGE_ROOT / "data" / "sample_tasks.jsonl"
# Legacy repo path (for local development)
REPO_DATASET = Path(__file__).resolve().parents[3] / "dakota_rl_training" / "datasets" / "grammar_tasks_complete.jsonl"
# GitHub URL for hosted evals
GITHUB_DATASET_URL = "https://raw.githubusercontent.com/HarleyCoops/Dakota1890/main/dakota_rl_training/datasets/grammar_tasks_complete.jsonl"


def _normalize(text: str) -> str:
    """Normalize text for comparison while preserving Dakota special characters."""
    return " ".join(text.strip().lower().split())


def _char_f1(prediction: str, target: str) -> float:
    """Compute character-level F1 score, preserving Dakota special characters."""
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


def _load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load entries from a JSONL file or URL."""
    # Handle URL paths
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
        logger.info("Downloading dataset from URL: %s", path)
        try:
            with urlopen(path) as response:
                content = response.read().decode("utf-8")
                # Write to temp file for processing
                with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                try:
                    return _load_jsonl_from_file(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error("Failed to download dataset from URL %s: %s", path, e)
            return []
    
    # Handle Path objects
    path_obj = Path(path) if isinstance(path, str) else path
    if not path_obj.exists() or path_obj.stat().st_size == 0:
        logger.info("Dataset file %s missing or empty.", path_obj)
        return []
    
    return _load_jsonl_from_file(path_obj)


def _load_jsonl_from_file(path: Path) -> list[dict[str, Any]]:
    """Load entries from a local JSONL file."""
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d of %s (%s).", line_no, path, exc)
                continue
            if "prompt" not in payload or "answer" not in payload:
                # Try alternative field names
                if "prompt" not in payload and "question" in payload:
                    payload["prompt"] = payload["question"]
                if "answer" not in payload and "ideal_answer" in payload:
                    payload["answer"] = payload["ideal_answer"]
                elif "answer" not in payload:
                    logger.debug(
                        "Skipping entry without prompt or answer on line %d of %s.",
                        line_no,
                        path,
                    )
                    continue
            entries.append(payload)
    return entries


def _prepare_records(
    entries: Sequence[dict[str, Any]],
    difficulties: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    include_hints: bool = True,
) -> list[dict[str, Any]]:
    """Prepare records for dataset creation with filtering."""
    allowed_difficulties = {d.lower() for d in difficulties or []}
    allowed_tasks = {t.lower() for t in task_types or []}
    records: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        entry_info = entry.get("info") if isinstance(entry.get("info"), dict) else {}
        difficulty = str(
            entry.get("difficulty")
            or entry_info.get("difficulty")
            or "medium"
        )
        task_type = str(
            entry.get("task_type")
            or entry_info.get("task_type")
            or "default"
        )
        if allowed_difficulties and difficulty.lower() not in allowed_difficulties:
            continue
        if allowed_tasks and task_type.lower() not in allowed_tasks:
            continue
        prompt = str(entry.get("prompt", "")).strip()
        answer = str(entry.get("answer") or entry.get("ideal_answer", "")).strip()
        if not prompt or not answer:
            continue
        info = {
            "rule_id": entry.get("rule_id") or entry_info.get("rule_id"),
            "verification_pattern": (
                entry.get("verification_pattern")
                or entry_info.get("verification_pattern")
                or entry_info.get("pattern")
            ),
            "difficulty": difficulty,
            "task_type": task_type,
            "special_chars": entry_info.get("special_chars", []),
            "required_affixes": entry_info.get("required_affixes", []),
        }
        if include_hints:
            info["hints"] = entry.get("hints") or entry_info.get("hints") or []
        records.append(
            {
                "id": entry.get("task_id") or f"dakota_task_{idx:05d}",
                "question": prompt,
                "answer": answer,
                "task": task_type,
                "info": info,
            }
        )
    return records


def _ensure_dataset(records: list[dict[str, Any]]) -> Dataset:
    """Ensure we have a valid dataset."""
    if not records:
        raise ValueError(
            "No usable Dakota grammar tasks were found. "
            "Ensure the RL dataset has been generated or provide `dataset_path`."
        )
    return Dataset.from_list(records)


@dataclass
class DatasetBundle:
    train: Dataset
    eval: Dataset | None = None


def _build_dataset_bundle(
    dataset_path: Optional[Path | str],
    eval_path: Optional[Path | str],
    max_examples: int,
    eval_examples: int,
    eval_fraction: float,
    seed: Optional[int],
    difficulty_filter: Optional[Sequence[str]],
    task_filter: Optional[Sequence[str]],
    include_hints: bool,
) -> DatasetBundle:
    """Build train/eval dataset bundle from JSONL files."""
    candidates: list[dict[str, Any]] = []
    
    # Try multiple dataset sources in order:
    # 1. Explicitly provided path (URL or local path)
    # 2. Default packaged dataset
    # 3. Legacy repo path (for local development)
    # 4. GitHub URL (for hosted evals)
    # 5. Fallback sample
    
    if dataset_path:
        explicit_path = dataset_path
        logger.info("Using provided dataset path: %s", explicit_path)
        candidates.extend(_prepare_records(_load_jsonl(explicit_path), difficulty_filter, task_filter, include_hints))
    
    if not candidates:
        if DEFAULT_DATASET.exists():
            logger.info("Using packaged default dataset: %s", DEFAULT_DATASET)
            candidates.extend(_prepare_records(_load_jsonl(DEFAULT_DATASET), difficulty_filter, task_filter, include_hints))
    
    if not candidates:
        if REPO_DATASET.exists():
            logger.info("Using repo dataset (local development): %s", REPO_DATASET)
            candidates.extend(_prepare_records(_load_jsonl(REPO_DATASET), difficulty_filter, task_filter, include_hints))
    
    if not candidates:
        logger.info("Attempting to download dataset from GitHub: %s", GITHUB_DATASET_URL)
        github_entries = _load_jsonl(GITHUB_DATASET_URL)
        if github_entries:
            logger.info("Successfully downloaded dataset from GitHub")
            candidates.extend(_prepare_records(github_entries, difficulty_filter, task_filter, include_hints))
    
    if not candidates:
        fallback_entries = _load_jsonl(FALLBACK_DATASET)
        if fallback_entries:
            logger.warning(
                "Using bundled fallback tasks because no generated dataset was found.",
            )
            candidates = _prepare_records(
                fallback_entries, difficulty_filter, task_filter, include_hints
            )
    
    if not candidates:
        raise ValueError(
            "Unable to locate any Dakota grammar tasks. "
            "Please provide dataset_path or ensure the dataset is accessible."
        )

    if max_examples > 0:
        candidates = candidates[: max_examples]

    train_dataset = _ensure_dataset(candidates)

    if eval_path:
        eval_candidates = _prepare_records(
            _load_jsonl(eval_path), difficulty_filter, task_filter, include_hints
        )
        if eval_examples > 0:
            eval_candidates = eval_candidates[:eval_examples]
        eval_dataset = _ensure_dataset(eval_candidates) if eval_candidates else None
        return DatasetBundle(train=train_dataset, eval=eval_dataset)

    if eval_fraction > 0 and len(train_dataset) > 1:
        split = train_dataset.train_test_split(test_size=eval_fraction, seed=seed or 42)
        eval_dataset = split["test"]
        if eval_examples > 0:
            eval_dataset = eval_dataset.select(range(min(eval_examples, len(eval_dataset))))
        train_dataset = split["train"]
    else:
        eval_dataset = None
    return DatasetBundle(train=train_dataset, eval=eval_dataset)


def build_dataset_bundle(
    dataset_path: Optional[Path | str] = None,
    eval_path: Optional[Path | str] = None,
    max_examples: int = -1,
    eval_examples: int = -1,
    eval_fraction: float = 0.1,
    seed: Optional[int] = None,
    difficulty_filter: Optional[Sequence[str]] = None,
    task_filter: Optional[Sequence[str]] = None,
    include_hints: bool = True,
) -> DatasetBundle:
    """Public wrapper for constructing Dakota grammar datasets."""

    return _build_dataset_bundle(
        dataset_path=dataset_path,
        eval_path=eval_path,
        max_examples=max_examples,
        eval_examples=eval_examples,
        eval_fraction=eval_fraction,
        seed=seed,
        difficulty_filter=difficulty_filter,
        task_filter=task_filter,
        include_hints=include_hints,
    )


class DakotaTranslationParser(Parser):
    """Parser that preserves Dakota orthography while normalizing whitespace."""

    def parse(self, text: str) -> str:
        """Parse and normalize text, preserving Dakota special characters."""
        return text.strip()

    def parse_answer(self, completion: Messages) -> str:
        """Extract answer from completion messages."""
        parsed = super().parse_answer(completion) or ""
        return parsed.strip()


class DakotaGrammarRubric(Rubric):
    """Reward rubric for Dakota grammar tasks with character preservation focus."""

    def __init__(self, parser: Optional[Parser] = None):
        parser = parser or DakotaTranslationParser()
        funcs = [
            self.exact_match_reward,
            self.char_overlap_reward,
            self.pattern_reward,
            self.affix_reward,
            self.length_penalty_reward,
        ]
        weights = [0.4, 0.2, 0.15, 0.1, 0.15]  # Added length penalty weight
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")
        self._last_ledger: Optional[Dict[str, float]] = None

    def _prediction(self, completion: vf.types.Messages, parser: Parser) -> str:
        """Extract prediction from completion."""
        response = parser.parse_answer(completion) or ""
        return response.strip()

    def exact_match_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        **_: Any,
    ) -> float:
        """Reward for exact match (normalized)."""
        prediction = self._prediction(completion, parser)
        return float(_normalize(prediction) == _normalize(answer))

    def char_overlap_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        **_: Any,
    ) -> float:
        """Reward for character-level overlap (critical for Dakota orthography)."""
        prediction = self._prediction(completion, parser)
        return float(_char_f1(prediction, answer))

    def affix_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        info: Dict[str, Any],
        **_: Any,
    ) -> float:
        """Reward for correct affix application."""
        prediction = self._prediction(completion, parser)
        required_affixes = (info or {}).get("required_affixes", []) or []
        if not required_affixes:
            return 1.0
        
        correct_count = 0
        for affix in required_affixes:
            affix_clean = affix.strip("-")
            if affix.startswith("-") and not affix.endswith("-"):
                # Suffix
                if re.search(rf'\w+{re.escape(affix_clean)}\b', prediction):
                    correct_count += 1
            elif affix.endswith("-") and not affix.startswith("-"):
                # Prefix
                if re.search(rf'\b{re.escape(affix_clean)}\w+', prediction):
                    correct_count += 1
            else:
                if affix_clean in prediction:
                    correct_count += 1
        
        return correct_count / len(required_affixes) if required_affixes else 1.0

    def pattern_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        info: Dict[str, Any],
        **_: Any,
    ) -> float:
        """Reward for pattern matching using verification patterns and hints."""
        prediction = self._prediction(completion, parser)
        score = 0.0
        pattern = (info or {}).get("verification_pattern")
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
                    _normalize(candidate) and _normalize(candidate) in normalized_prediction
                    for candidate in literal_candidates
                ):
                    score = 1.0
        if score < 1.0:
            hints = (info or {}).get("hints", []) or []
            if hints:
                covered = sum(1 for hint in hints if hint.lower() in prediction.lower())
                score = max(score, covered / len(hints))
        return float(score)

    def length_penalty_reward(
        self,
        completion: Messages,
        answer: str,
        parser: Parser,
        max_length_ratio: float = 3.0,
        **_: Any,
    ) -> float:
        """
        Length penalty disabled for the small-model Tinker configuration (always 1.0).
        """
        return 1.0

    def score(
        self,
        completion: Messages,
        answer: str,
        info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute reward score and build detailed ledger.
        
        Overrides parent to compute all components explicitly and build ledger.
        Returns the scalar reward (same as parent behavior).
        """
        info = info or {}
        parser = self.parser
        
        # Compute raw component scores
        exact_match_raw = float(self.exact_match_reward(completion, answer, parser, **kwargs))
        char_overlap_raw = float(self.char_overlap_reward(completion, answer, parser, **kwargs))
        pattern_raw = float(self.pattern_reward(completion, answer, parser, info=info, **kwargs))
        affix_raw = float(self.affix_reward(completion, answer, parser, info=info, **kwargs))
        length_penalty_raw = float(self.length_penalty_reward(completion, answer, parser, **kwargs))
        
        # Normalized values (identity for now, but can add transforms)
        exact_match_norm = exact_match_raw
        char_overlap_norm = char_overlap_raw
        pattern_norm = pattern_raw
        affix_norm = affix_raw
        length_penalty_norm = length_penalty_raw
        
        # Weights from __init__
        weights = [0.4, 0.2, 0.15, 0.1, 0.15]
        w_exact = weights[0]
        w_char = weights[1]
        w_pattern = weights[2]
        w_affix = weights[3]
        w_length = weights[4]
        
        # Difficulty multiplier (from info if available)
        difficulty = info.get("difficulty", "intermediate")
        difficulty_multipliers = {
            "easy": 1.0,
            "basic": 1.0,
            "medium": 1.2,
            "intermediate": 1.2,
            "advanced": 1.5,
            "hard": 1.5,
            "expert": 2.0,
        }
        difficulty_mult = difficulty_multipliers.get(difficulty.lower(), 1.0)
        
        # Compute additive component contributions.
        # Note: length_penalty_reward returns a multiplier (1.0 = no penalty, <1.0 = penalty),
        # so length is tracked separately and applied after the weighted sum.
        contrib_exact = w_exact * exact_match_norm
        contrib_char = w_char * char_overlap_norm
        contrib_pattern = w_pattern * pattern_norm
        contrib_affix = w_affix * affix_norm
        composite_pre = contrib_exact + contrib_char + contrib_pattern + contrib_affix
        
        # Apply length penalty as multiplier
        composite_with_length = composite_pre * length_penalty_norm
        
        # Apply difficulty multiplier
        composite_with_diff = composite_with_length * difficulty_mult
        
        # Final reward scalar
        reward_scalar = composite_with_diff
        
        # Build ledger
        ledger = {
            # Raw components
            "exact_match_raw": exact_match_raw,
            "char_overlap_raw": char_overlap_raw,
            "pattern_raw": pattern_raw,
            "affix_raw": affix_raw,
            "length_penalty_raw": length_penalty_raw,
            
            # Normalized components (identity for now)
            "exact_match_norm": exact_match_norm,
            "char_overlap_norm": char_overlap_norm,
            "pattern_norm": pattern_norm,
            "affix_norm": affix_norm,
            "length_penalty_norm": length_penalty_norm,
            
            # Weights
            "w_exact": w_exact,
            "w_char": w_char,
            "w_pattern": w_pattern,
            "w_affix": w_affix,
            "w_length": w_length,
            
            # Difficulty
            "difficulty_multiplier": difficulty_mult,

            # Weighted component contributions
            "contrib_exact": contrib_exact,
            "contrib_char": contrib_char,
            "contrib_pattern": contrib_pattern,
            "contrib_affix": contrib_affix,
            
            # Composites
            "composite_pre": composite_pre,
            "composite_with_length": composite_with_length,
            "composite_with_difficulty": composite_with_diff,
            "composite_predicted": composite_with_diff,
            
            # Final reward
            "reward_scalar": reward_scalar,
        }
        
        # Consistency check
        diff = abs(reward_scalar - composite_with_diff)
        ledger["composite_diff"] = float(diff)
        
        # Store ledger for retrieval
        self._last_ledger = ledger
        
        return reward_scalar
    
    def get_last_ledger(self) -> Optional[Dict[str, float]]:
        """Get the ledger from the last score() call."""
        return self._last_ledger


class DakotaGrammarEnv(SingleTurnEnv):
    """Single-turn chat environment for Dakota grammar and translation tasks."""

    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset],
        system_prompt: str,
        rubric: DakotaGrammarRubric,
        sampling_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=rubric.parser,
            rubric=rubric,
            sampling_args=sampling_args,
            message_type="chat",
            **kwargs,
        )
        self.rubric = rubric  # Store reference for ledger access
    
    def get_reward_ledger(self) -> Optional[Dict[str, float]]:
        """Get the reward ledger from the last scoring operation."""
        if isinstance(self.rubric, DakotaGrammarRubric):
            return self.rubric.get_last_ledger()
        return None


def load_environment(
    dataset_path: str | Path | None = None,
    eval_path: str | Path | None = None,
    max_examples: int = -1,
    eval_examples: int = -1,
    eval_fraction: float = 0.1,
    difficulty_filter: Optional[Sequence[str]] = None,
    task_filter: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
    sampling_args: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    include_hints: bool = True,
) -> vf.Environment:
    """
    Load the Dakota grammar translation environment.

    Args:
        dataset_path: Path to a JSONL file produced by the RL grammar pipeline.
        eval_path: Optional path to a JSONL evaluation split.
        max_examples: Cap on training examples (-1 uses full dataset).
        eval_examples: Cap on evaluation examples (-1 uses full split).
        eval_fraction: Fraction of training split reserved for evaluation if `eval_path` is not provided.
        difficulty_filter: Optional iterable of difficulty labels to keep (easy/medium/hard).
        task_filter: Optional iterable of task types to keep (morphology/translation/etc).
        system_prompt: Override the default system prompt.
        sampling_args: Optional sampling overrides passed to verifiers.
        seed: RNG seed used when performing internal splits.
        include_hints: Keep or drop hint metadata in the info payload.

    Returns:
        A configured verifiers `Environment` instance.
    """

    # Handle string paths (including URLs) - don't convert to Path if it's a URL
    if dataset_path:
        if isinstance(dataset_path, str) and (dataset_path.startswith("http://") or dataset_path.startswith("https://")):
            resolved_dataset = dataset_path  # Keep as string for URL
        else:
            resolved_dataset = Path(dataset_path).expanduser().resolve()
    else:
        resolved_dataset = None
    
    # Handle eval_path similarly
    if eval_path:
        if isinstance(eval_path, str) and (eval_path.startswith("http://") or eval_path.startswith("https://")):
            resolved_eval = eval_path  # Keep as string for URL
        else:
            resolved_eval = Path(eval_path).expanduser().resolve()
    else:
        resolved_eval = None

    bundle = _build_dataset_bundle(
        dataset_path=resolved_dataset,
        eval_path=resolved_eval,
        max_examples=max_examples,
        eval_examples=eval_examples,
        eval_fraction=eval_fraction,
        seed=seed,
        difficulty_filter=difficulty_filter,
        task_filter=task_filter,
        include_hints=include_hints,
    )

    rubric = DakotaGrammarRubric()
    env = DakotaGrammarEnv(
        dataset=bundle.train,
        eval_dataset=bundle.eval,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        rubric=rubric,
        sampling_args=sampling_args,
    )

    logger.info(
        "Loaded Dakota grammar environment with %d train and %s eval examples.",
        len(env.dataset) if env.dataset else 0,
        len(env.eval_dataset) if env.eval_dataset else 0,
    )
    return env
