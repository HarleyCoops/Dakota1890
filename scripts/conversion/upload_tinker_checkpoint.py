#!/usr/bin/env python3
"""
Upload a single Tinker checkpoint archive/file to a Hugging Face repo.

Usage (PowerShell):
  $env:HF_TOKEN='...'
  python scripts/conversion/upload_tinker_checkpoint.py `
    --file path/to/final_sampler.ckpt `
    --repo-id HarleyCooper/Qwen3-4B-RailRoadEngineer1959 `
    --repo-path checkpoint/final_sampler.ckpt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a Tinker checkpoint file to Hugging Face.")
    parser.add_argument("--file", required=True, help="Local checkpoint file to upload.")
    parser.add_argument("--repo-id", required=True, help="Target HF repo (e.g., username/model).")
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path inside the repo (default: use the local filename).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Tinker checkpoint",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = HfApi(token=args.token)

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Local file not found: {file_path}")

    repo_path = args.repo_path or file_path.name

    # Ensure repo exists
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=repo_path,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )

    print(f"Uploaded {file_path} to https://huggingface.co/{args.repo_id}/blob/main/{repo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
