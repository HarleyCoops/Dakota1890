"""Configuration checks for the maintained Dakota inference surfaces."""

from __future__ import annotations

import json
from pathlib import Path

from hf_inference_standalone import MODEL_ID
from run_inference import ADAPTER_NAME, BASE_MODEL_NAME


def test_local_inference_base_matches_saved_adapter_config() -> None:
    """The local inference base model should match the saved adapter metadata."""
    adapter_config = json.loads(
        Path("Qwen3-30B-ThinkingMachines-Dakota1890/adapter_config.json").read_text(encoding="utf-8")
    )

    assert BASE_MODEL_NAME == adapter_config["base_model_name_or_path"]


def test_inference_surfaces_use_same_adapter_id() -> None:
    """The local and Hugging Face inference entrypoints should point at the same adapter."""
    assert ADAPTER_NAME == MODEL_ID
