"""Optional Dakota judge hook stays off the Tinker train path."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV_PKG = ROOT / "environments" / "dakota_grammar_translation"
if str(ENV_PKG) not in sys.path:
    sys.path.insert(0, str(ENV_PKG))

from dakota_grammar_translation.judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    JudgeResult,
    judge_dakota,
    judge_from_env,
)


def test_judge_returns_required_json_fields() -> None:
    def fake_complete(prompt: str) -> str:
        assert "Dawid suŋkaku" in prompt
        return json.dumps(
            {
                "correct": 1,
                "morphology_ok": 1,
                "meaning_ok": 1,
                "orthography_ok": 1,
                "rationale": "Exact gold form.",
            }
        )

    result = judge_dakota(
        prompt="Apply -ku",
        gold="Dawid suŋkaku",
        model_output="Dawid suŋkaku",
        rule_snippet="relationship nouns take -ku",
        complete_fn=fake_complete,
    )
    assert isinstance(result, JudgeResult)
    payload = result.to_json()
    assert payload == {
        "correct": 1,
        "morphology_ok": 1,
        "meaning_ok": 1,
        "orthography_ok": 1,
        "rationale": "Exact gold form.",
    }


def test_judge_coerces_messy_model_json() -> None:
    def fake_complete(_prompt: str) -> str:
        return (
            "Here is my verdict:\n"
            '{"correct": 0, "morphology_ok": 0, "meaning_ok": 1, '
            '"orthography_ok": 0, "rationale": "Wrong stem."}\n'
        )

    result = judge_dakota(
        prompt="p",
        gold="Dawid suŋkaku",
        model_output="wicaštaku",
        complete_fn=fake_complete,
    )
    assert result.correct == 0
    assert result.morphology_ok == 0
    assert result.meaning_ok == 1
    assert result.orthography_ok == 0


def test_judge_from_env_is_disabled_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert judge_from_env() is None


def test_default_judge_model_name() -> None:
    assert DEFAULT_JUDGE_MODEL in {"Qwen/Qwen3-8B", "Qwen/Qwen3.8-Max"}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
            names.add(node.module)
    return names


def test_tinker_trainer_does_not_import_judge() -> None:
    train_imports = _imported_modules(ROOT / "dakota_rl_training" / "tinker_train.py")
    env_imports = _imported_modules(ROOT / "dakota_rl_training" / "tinker_integration" / "env.py")
    assert "judge" not in train_imports
    assert "dakota_grammar_translation.judge" not in train_imports
    assert "judge" not in env_imports
    assert "dakota_grammar_translation.judge" not in env_imports
    train_src = (ROOT / "dakota_rl_training" / "tinker_train.py").read_text(encoding="utf-8")
    env_src = (ROOT / "dakota_rl_training" / "tinker_integration" / "env.py").read_text(encoding="utf-8")
    assert "judge_dakota" not in train_src
    assert "judge_dakota" not in env_src
