"""OpenAI SFT configuration for the Dakota baseline pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv


SUPPORTED_FINETUNE_MODELS = (
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
)

DEFAULT_OPENAI_FINETUNE_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_OPENAI_FINETUNE_EPOCHS = 3


class ConfigError(ValueError):
    """Raised when Dakota OpenAI fine-tune configuration is invalid."""


@dataclass(frozen=True)
class DakotaOpenAIConfig:
    """Resolved model and hyperparameter configuration for Dakota OpenAI SFT."""

    openai_finetune_model: str
    openai_finetune_epochs: int
    legacy_openai_model_present: bool = False

    def diagnostics(self) -> str:
        """Return safe configuration diagnostics without exposing secrets."""

        lines = [
            "Dakota OpenAI SFT configuration:",
            f"- OPENAI_FINETUNE_MODEL: {self.openai_finetune_model}",
            f"- OPENAI_FINETUNE_EPOCHS: {self.openai_finetune_epochs}",
            "- Supported fine-tune models: " + ", ".join(SUPPORTED_FINETUNE_MODELS),
        ]
        if self.legacy_openai_model_present:
            lines.append("- OPENAI_MODEL: set but ignored when OPENAI_FINETUNE_MODEL is present")
        return "\n".join(lines)


def validate_finetune_model(model: str) -> None:
    """Raise when the selected supervised fine-tune model is unsupported."""

    if model not in SUPPORTED_FINETUNE_MODELS:
        allowed = ", ".join(SUPPORTED_FINETUNE_MODELS)
        raise ConfigError(
            f"Unsupported OPENAI_FINETUNE_MODEL '{model}'. "
            f"Choose one of: {allowed}. Update dakota_openai_config.py if OpenAI support changes."
        )


def _parse_epochs(raw: str) -> int:
    """Parse and validate the epoch count."""

    try:
        epochs = int(raw)
    except ValueError as exc:
        raise ConfigError(f"OPENAI_FINETUNE_EPOCHS must be an integer, got {raw!r}") from exc
    if epochs <= 0:
        raise ConfigError("OPENAI_FINETUNE_EPOCHS must be greater than zero")
    return epochs


def load_dakota_openai_config(
    env: Mapping[str, str] | None = None,
    *,
    load_dotenv_file: bool = True,
    validate_finetune: bool = True,
) -> DakotaOpenAIConfig:
    """Resolve Dakota OpenAI SFT configuration from environment variables."""

    if load_dotenv_file:
        load_dotenv(override=True)
    values = os.environ if env is None else env
    model = values.get(
        "OPENAI_FINETUNE_MODEL",
        values.get("OPENAI_MODEL", DEFAULT_OPENAI_FINETUNE_MODEL),
    )
    config = DakotaOpenAIConfig(
        openai_finetune_model=model,
        openai_finetune_epochs=_parse_epochs(
            values.get("OPENAI_FINETUNE_EPOCHS", str(DEFAULT_OPENAI_FINETUNE_EPOCHS))
        ),
        legacy_openai_model_present=bool(values.get("OPENAI_MODEL")),
    )
    if validate_finetune:
        validate_finetune_model(config.openai_finetune_model)
    return config
