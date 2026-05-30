from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import requests
import tinker


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint log not found: {path}")
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if raw_line:
                entries.append(json.loads(raw_line))
    return entries


def select_checkpoint(log_dir: Path, name: str | None = None) -> dict:
    entries = _read_jsonl(log_dir / "checkpoints.jsonl")
    candidates = entries
    if name:
        candidates = [entry for entry in entries if entry.get("name") == name]
        if not candidates:
            raise ValueError(f"No checkpoint named '{name}' found in {log_dir}")
    if not candidates:
        raise ValueError(f"No checkpoints available in {log_dir}")
    return candidates[-1]


def publish_checkpoint(tinker_path: str) -> None:
    """Publish a checkpoint via the Tinker REST client."""
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    future = rest_client.publish_checkpoint_from_tinker_path(tinker_path)
    if future is not None:
        future.result()


def unpublish_checkpoint(tinker_path: str) -> None:
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    future = rest_client.unpublish_checkpoint_from_tinker_path(tinker_path)
    if future is not None:
        future.result()


def download_checkpoint_archive(tinker_path: str, output_path: Path) -> Path:
    """Download a checkpoint archive for local packaging or W&B artifacts."""

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    signed_url_future = rest_client.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    signed_url_result = signed_url_future.result()
    url = getattr(signed_url_result, "url", signed_url_result)
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return output_path


def read_latest_metrics(log_dir: Path) -> dict[str, float]:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    last_entry: dict[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if raw_line:
                last_entry = json.loads(raw_line)
    if not last_entry:
        return {}
    return {
        key: value
        for key, value in last_entry.items()
        if isinstance(value, (int, float)) and (key.startswith("reward") or key.startswith("ledger/"))
    }


def build_metadata(
    checkpoint: dict,
    log_dir: Path,
    wandb_run_url: str | None = None,
) -> dict[str, Any]:
    metrics = read_latest_metrics(log_dir)
    metadata = {
        "checkpoint_name": checkpoint.get("name"),
        "step": checkpoint.get("batch"),
        "tinker_path": checkpoint.get("sampler_path"),
        "state_path": checkpoint.get("state_path"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "wandb_run": wandb_run_url,
        "metrics": metrics,
    }
    return metadata


def write_metadata(metadata: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path
