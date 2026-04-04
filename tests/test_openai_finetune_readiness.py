"""Readiness smoke for the Dakota OpenAI SFT baseline launcher."""

from __future__ import annotations

from scripts.rl.dakota_openai_finetune import DakotaOpenAIFineTuner


def test_openai_finetune_readiness_report_uses_repo_assets() -> None:
    """The baseline launcher should resolve the maintained repo splits without an API key."""
    tuner = DakotaOpenAIFineTuner(require_api_key=False)
    report = tuner.readiness_report()

    assert report["train_exists"] is True
    assert report["valid_exists"] is True
    assert report["train_examples"] == 980
    assert report["valid_examples"] == 245
    assert report["base_model"] == "gpt-4.1-mini-2025-04-14"
    assert report["epochs"] == 3
    assert report["train_token_estimate"] > 0
    assert report["estimated_training_tokens"] >= report["train_token_estimate"]
