#!/usr/bin/env python3
"""
Sample Dakota bilingual QA examples and query the public HF endpoint.

This script:
- Randomly samples questions from HarleyCooper/dakota-bilingual-qa.
- Sends each question to the deployed endpoint.
- Records question, reference answer, model response, and metadata in JSONL.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from datasets import load_dataset


# Use the live inference URL (not the management API URL).
DEFAULT_ENDPOINT = "https://sodh94mt9pt8xuzf.us-east4.gcp.endpoints.huggingface.cloud"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample Dakota QA examples and query the HF endpoint."
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Endpoint URL to query (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of questions to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/benchmarks/endpoint_samples.jsonl",
        help="Path to write JSONL results.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation max_new_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Generation top_p.",
    )
    return parser.parse_args()


def sample_dataset(split: str, sample_size: int, seed: int):
    ds = load_dataset("HarleyCooper/dakota-bilingual-qa", split=split)
    if sample_size > len(ds):
        sample_size = len(ds)
    sampled = ds.shuffle(seed=seed).select(range(sample_size))
    return sampled


def call_endpoint(
    url: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, timeout: int = 120
) -> Dict[str, Any]:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        status = resp.status_code
        data: Any
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return {"status_code": status, "data": data, "error": None}
    except Exception as e:
        return {"status_code": None, "data": None, "error": str(e)}


def extract_generated_text(response_payload: Any) -> Optional[str]:
    """
    Endpoint returns [{"generated_text": "..."}].
    If the shape differs, fall back to string formatting.
    """
    if isinstance(response_payload, list) and response_payload:
        item = response_payload[0]
        if isinstance(item, dict) and "generated_text" in item:
            return str(item["generated_text"])
    if isinstance(response_payload, dict) and "generated_text" in response_payload:
        return str(response_payload["generated_text"])
    if isinstance(response_payload, str):
        return response_payload
    return None


def main():
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = sample_dataset(args.split, args.sample_size, args.seed)

    results: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"]
        meta = {
            "pair_id": row.get("pair_id"),
            "source_language": row.get("source_language"),
            "source_pages": row.get("source_pages"),
            "source_files": row.get("source_files"),
            "generated_at": row.get("generated_at"),
        }

        resp = call_endpoint(
            url=args.endpoint_url,
            prompt=question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generated_text = extract_generated_text(resp["data"])

        results.append(
            {
                "question": question,
                "reference_answer": answer,
                "model_response": generated_text,
                "raw_response": resp["data"],
                "status_code": resp["status_code"],
                "error": resp["error"],
                "endpoint_url": args.endpoint_url,
                **meta,
            }
        )
        print(f"[{i+1}/{len(ds)}] status={resp['status_code']} question={question[:50]!r}")

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
