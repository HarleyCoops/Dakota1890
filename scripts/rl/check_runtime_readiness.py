#!/usr/bin/env python3
"""Check the live Dakota extraction/RL runtime before paid runs."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_TINKER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "warn"}


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_tuple(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    match = re.match(r"(\d+(?:\.\d+)*)", version)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def add_result(results: list[CheckResult], name: str, status: str, detail: str) -> None:
    results.append(CheckResult(name=name, status=status, detail=detail))


def load_dotenv_if_available(results: list[CheckResult]) -> None:
    try:
        from dotenv import load_dotenv
    except Exception as exc:  # pragma: no cover - defensive runtime check
        add_result(results, "dotenv", "warn", f"python-dotenv is not importable: {exc}")
        return

    loaded = load_dotenv()
    add_result(results, "dotenv", "ok", f".env loaded={loaded}")


def check_distribution(results: list[CheckResult], distribution: str, minimum: str | None = None) -> None:
    found = package_version(distribution)
    if found is None:
        add_result(results, distribution, "fail", "not installed")
        return
    if minimum and version_tuple(found) < version_tuple(minimum):
        add_result(results, distribution, "fail", f"installed {found}, expected >= {minimum}")
        return
    add_result(results, distribution, "ok", f"installed {found}")


def check_protobuf(results: list[CheckResult]) -> None:
    found = package_version("protobuf")
    if found is None:
        add_result(results, "protobuf", "fail", "not installed")
        return
    parsed = version_tuple(found)
    if parsed < (6, 31, 1) or parsed >= (7,):
        add_result(results, "protobuf", "fail", f"installed {found}, expected >= 6.31.1 and < 7")
        return
    add_result(results, "protobuf", "ok", f"installed {found}")


def model_name(model: Any) -> str | None:
    if isinstance(model, str):
        return model
    if isinstance(model, dict):
        return model.get("model_name") or model.get("name")
    return getattr(model, "model_name", None) or getattr(model, "name", None)


def check_tinker_api(results: list[CheckResult], required_models: list[str]) -> None:
    try:
        import tinker
    except Exception as exc:
        add_result(results, "tinker-api", "fail", f"could not import tinker: {exc}")
        return

    try:
        client = tinker.ServiceClient()
        capabilities = client.get_server_capabilities()
    except Exception as exc:
        add_result(results, "tinker-api", "fail", f"server capability check failed: {exc}")
        return

    supported = getattr(capabilities, "supported_models", None) or getattr(capabilities, "models", None) or []
    names = sorted(name for name in (model_name(model) for model in supported) if name)
    missing = [name for name in required_models if name not in names]
    if missing:
        add_result(
            results,
            "tinker-api",
            "fail",
            f"{len(names)} models visible, missing required models: {', '.join(missing)}",
        )
        return
    add_result(results, "tinker-api", "ok", f"{len(names)} models visible; required models present")


def check_tokenizers(results: list[CheckResult], required_models: list[str]) -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        add_result(results, "tokenizer", "fail", f"could not import transformers: {exc}")
        return

    for name in required_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name, fast=True)
        except Exception as exc:
            add_result(results, "tokenizer", "fail", f"could not load tokenizer for {name}: {exc}")
            return
        add_result(results, "tokenizer", "ok", f"{name} -> {type(tokenizer).__name__}")


def normalize_gemini_model(name: str) -> str:
    return name if name.startswith("models/") else f"models/{name}"


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))
    return "".join(parts)


def check_gemini_api(results: list[CheckResult], model: str, generate_smoke: bool) -> None:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        add_result(results, "gemini-api", "fail", f"could not import google-genai: {exc}")
        return

    try:
        client = genai.Client()
        available = [getattr(item, "name", "") for item in client.models.list()]
    except Exception as exc:
        add_result(results, "gemini-api", "fail", f"model list failed: {exc}")
        return

    expected = normalize_gemini_model(model)
    if expected not in available:
        add_result(results, "gemini-api", "fail", f"{expected} not present in {len(available)} listed models")
        return

    if not generate_smoke:
        add_result(results, "gemini-api", "ok", f"{expected} listed")
        return

    try:
        response = client.models.generate_content(
            model=model,
            contents='Return only JSON: {"ok": true}',
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=200,
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        parsed = json.loads(response_text(response))
    except Exception as exc:
        add_result(results, "gemini-api", "fail", f"generation smoke failed: {exc}")
        return

    if parsed.get("ok") is True:
        add_result(results, "gemini-api", "ok", f"{expected} generated valid JSON")
    else:
        add_result(results, "gemini-api", "fail", f"generation smoke returned unexpected JSON: {parsed}")


def check_wandb(results: list[CheckResult], check_api: bool) -> None:
    try:
        import wandb
    except Exception as exc:
        add_result(results, "wandb", "fail", f"could not import wandb: {exc}")
        return

    version = getattr(wandb, "__version__", "unknown")
    if not check_api:
        add_result(results, "wandb", "ok", f"imported {version}")
        return

    try:
        api = wandb.Api()
        viewer = api.viewer
    except Exception as exc:
        add_result(results, "wandb", "fail", f"API check failed: {exc}")
        return
    username = getattr(viewer, "username", None) or getattr(viewer, "entity", None) or str(viewer)
    add_result(results, "wandb", "ok", f"API authenticated as {username}")


def check_legacy_google_package(results: list[CheckResult]) -> None:
    found = package_version("google-generativeai")
    if found is None:
        add_result(results, "legacy-google-generativeai", "ok", "not installed")
        return
    add_result(results, "legacy-google-generativeai", "warn", f"installed {found}; active code should not import it")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tinker-model",
        action="append",
        default=[DEFAULT_TINKER_MODEL],
        help="Required Tinker model name. May be passed more than once.",
    )
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model required for Q/A generation.")
    parser.add_argument("--skip-tinker-api", action="store_true", help="Skip Tinker server capability check.")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip local Hugging Face tokenizer load check.")
    parser.add_argument("--skip-gemini-api", action="store_true", help="Skip Gemini model/API check.")
    parser.add_argument("--gemini-generate-smoke", action="store_true", help="Run one tiny Gemini JSON generation check.")
    parser.add_argument("--check-wandb-api", action="store_true", help="Call the W&B API instead of import-only validation.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results: list[CheckResult] = []

    load_dotenv_if_available(results)
    check_distribution(results, "tinker", minimum="0.22.1")
    check_distribution(results, "tinker-cookbook", minimum="0.4.1")
    check_protobuf(results)
    check_distribution(results, "google-genai", minimum="2.6.0")
    check_legacy_google_package(results)
    check_wandb(results, check_api=args.check_wandb_api)

    if args.skip_tinker_api:
        add_result(results, "tinker-api", "warn", "skipped")
    else:
        check_tinker_api(results, args.tinker_model)

    if args.skip_tokenizer:
        add_result(results, "tokenizer", "warn", "skipped")
    else:
        check_tokenizers(results, args.tinker_model)

    if args.skip_gemini_api:
        add_result(results, "gemini-api", "warn", "skipped")
    else:
        check_gemini_api(results, args.gemini_model, generate_smoke=args.gemini_generate_smoke)

    width = max(len(item.name) for item in results)
    for item in results:
        print(f"{item.name:<{width}}  {item.status.upper():<5}  {item.detail}")

    failed = [item for item in results if item.status == "fail"]
    if failed:
        print(f"\nRuntime readiness failed: {len(failed)} blocking check(s).", file=sys.stderr)
        return 1

    print("\nRuntime readiness passed. Warnings are informational; investigate them before changing active code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
