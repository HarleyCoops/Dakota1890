"""Secret hygiene checks for tracked repository files."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WANDB_API_KEY_ASSIGNMENT = (
    r"\bWANDB_API_KEY\b[[:space:]]*=[[:space:]]*[\"']?[[:xdigit:]]{40}[\"']?"
)


def test_tracked_files_do_not_hardcode_wandb_api_keys() -> None:
    """Tracked files should not assign literal W&B API keys."""
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-I",
            "-E",
            WANDB_API_KEY_ASSIGNMENT,
            "--",
            ".",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode in {0, 1}, result.stderr
    assert result.returncode == 1, result.stdout
