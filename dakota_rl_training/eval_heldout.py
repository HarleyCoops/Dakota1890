#!/usr/bin/env python3
"""Held-out / hack-probe eval that is not part of the Tinker GRPO loop.

Optional judge is used only when QWEN_JUDGE_BASE_URL or OPENAI_BASE_URL is set.
This module is not imported by tinker_train.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_PKG = ROOT / "environments" / "dakota_grammar_translation"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))

from dakota_grammar_translation.judge import judge_from_env
from dakota_grammar_translation.train_reward import score_train_reward


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_row(
    row: dict[str, Any],
    output_field: str = "model_output",
    judge_fn: Any | None = None,
) -> dict[str, Any]:
    output = str(row.get(output_field) or row.get("completion") or row.get("prediction") or "")
    gold = str(row.get("answer") or "")
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    prompt = str(row.get("prompt") or row.get("question") or "")
    scored = score_train_reward(output, gold, info)
    record = {
        "id": row.get("probe_id") or row.get("task_id") or row.get("id"),
        "prompt": prompt,
        "gold": gold,
        "model_output": output,
        "answer_span": scored["answer_span"],
        "semantic": scored["semantic"],
        "char": scored["char"],
        "special_char": scored["special_char"],
        "affix": scored["affix"],
        "pattern": scored["pattern"],
        "length_penalty": scored["length_penalty"],
        "composite_unweighted": scored["composite_unweighted"],
        "difficulty_multiplier": scored["difficulty_multiplier"],
        "composite_with_difficulty": scored["composite_with_difficulty"],
        "reward_scalar": scored["reward_scalar"],
        "passed": scored["passed"],
        "judge_correct": None,
        "judge_morphology_ok": None,
        "judge_meaning_ok": None,
        "judge_orthography_ok": None,
        "judge_rationale": None,
    }
    if judge_fn is not None:
        judged = judge_fn(
            prompt=prompt,
            gold=gold,
            model_output=output,
            rule_snippet=str(info.get("verification_pattern") or ""),
        )
        payload = judged.to_json()
        record["judge_correct"] = payload["correct"]
        record["judge_morphology_ok"] = payload["morphology_ok"]
        record["judge_meaning_ok"] = payload["meaning_ok"]
        record["judge_orthography_ok"] = payload["orthography_ok"]
        record["judge_rationale"] = payload["rationale"]
    return record


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    keys = ["semantic", "char", "affix", "special_char", "reward_scalar"]
    summary: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        summary[f"mean_{key}"] = sum(values) / len(values)
    judged = [row for row in rows if row.get("judge_correct") is not None]
    if judged:
        summary["mean_judge_correct"] = sum(float(row["judge_correct"]) for row in judged) / len(judged)
        summary["n_judged"] = len(judged)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Dakota held-out or hack-probe outputs.")
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "dakota_rl_training" / "datasets" / "hack_probes.jsonl"),
        help="JSONL with gold answers and model_output (hack probes by default).",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "dakota_rl_training" / "outputs" / "grant_clean_eval.jsonl"),
        help="Where to write per-row scores.",
    )
    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help="Call the optional env-configured judge. No-op if no endpoint is set.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_jsonl(Path(args.predictions))
    judge_fn = judge_from_env() if args.enable_judge else None
    scored = [score_row(row, judge_fn=judge_fn) for row in rows]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summarize(scored), ensure_ascii=False, indent=2))
    print(f"Wrote {len(scored)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
