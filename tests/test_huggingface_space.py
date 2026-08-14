"""Grant-clean Hugging Face Space: holdout gold stays frozen; CPU path works offline."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_space.demo import (
    CONVENTION_SAMPLER_PATH,
    extract_final_answer,
    live_inference_available,
    load_examples,
    resolve_sampler_path,
    run_live_or_explain,
)

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = ROOT / "dakota_rl_training" / "datasets" / "grammar_tasks_heldout.jsonl"
SPACE_APP = ROOT / "huggingface_space" / "app.py"
SPACE_REQ = ROOT / "huggingface_space" / "requirements.txt"
SPACE_README = ROOT / "huggingface_space" / "README.md"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_examples_match_frozen_holdout_gold() -> None:
    heldout = {row["task_id"]: row for row in _load_jsonl(HELD_OUT)}
    examples = load_examples()
    assert len(examples) == 12
    assert examples[0]["info"]["task_type"] == "reverse_translation"
    reverse = [row for row in examples if row["info"]["task_type"] == "reverse_translation"]
    other = [row for row in examples if row["info"]["task_type"] != "reverse_translation"]
    assert len(reverse) >= 6
    assert {row["info"]["task_type"] for row in other} >= {
        "word_translation",
        "morphology",
    }
    for row in examples:
        source = heldout[row["task_id"]]
        assert row["prompt"] == source["prompt"]
        assert row["answer"] == source["answer"]
        assert row["info"]["task_type"] == source["info"]["task_type"]
        assert row["info"]["difficulty"] == source["info"]["difficulty"]


def test_extract_final_answer_matches_grant_clean_priority() -> None:
    assert extract_final_answer("foo\n\\boxed{Dawid suŋkaku}\n") == "Dawid suŋkaku"
    assert extract_final_answer("Let me think.\nFinal answer is kašká") == "kašká"
    assert extract_final_answer("line one\nline two") == "line two"
    assert extract_final_answer("hehan") == "hehan"
    assert extract_final_answer("") == ""


def test_offline_live_path_explains_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_SAMPLER_PATH", raising=False)
    assert live_inference_available() is False
    assert resolve_sampler_path() == CONVENTION_SAMPLER_PATH
    assert "owf98569" not in CONVENTION_SAMPLER_PATH
    assert "1f23df9c" not in CONVENTION_SAMPLER_PATH
    extracted, raw, comparison = run_live_or_explain(
        "Translate this English sentence to Dakota:\n\nthen",
        "hehan",
    )
    assert extracted == ""
    assert "TINKER_API_KEY" in raw
    assert "hehan" in comparison


def test_space_bundle_is_cpu_only() -> None:
    app_text = SPACE_APP.read_text(encoding="utf-8")
    req_text = SPACE_REQ.read_text(encoding="utf-8")
    readme = SPACE_README.read_text(encoding="utf-8")
    assert "spaces.GPU" not in app_text
    assert "transformers" not in app_text
    assert "transformers" not in req_text
    assert "torch" not in req_text
    assert "Qwen3-0.6B-Dakota-Grammar-RL" not in readme
    assert "HarleyCooper/Dakota1890-Grant-Clean" in readme
    assert "cebp9acs" in readme
    assert "owf98569" in readme  # named only as a different, unused run
    assert "cpu-basic" in readme
    assert "Qwen3-0.6B-Dakota-Grammar-RL" not in app_text
