"""Optional external-judge eval hook. Not imported by the Tinker trainer."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-8B"

CompleteFn = Callable[[str], str]


@dataclass(frozen=True)
class JudgeResult:
    correct: int
    morphology_ok: int
    meaning_ok: int
    orthography_ok: int
    rationale: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _as_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value >= 0.5 else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "ok"}:
        return 1
    return 0


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge output did not contain a JSON object.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Judge JSON was not an object.")
    return payload


def build_judge_prompt(
    prompt: str,
    gold: str,
    model_output: str,
    rule_snippet: str | None = None,
) -> str:
    rule_block = rule_snippet.strip() if rule_snippet else "(none provided)"
    return (
        "You are checking a Dakota language grammar/morphology answer from "
        "Stephen Return Riggs' 1890 Dakota-English Dictionary.\n"
        "Score only the model's final answer, not whether gold text appears in reasoning.\n"
        "Return JSON only with keys: correct, morphology_ok, meaning_ok, "
        "orthography_ok, rationale.\n"
        "Each of the first four keys must be 0 or 1.\n\n"
        f"Task prompt:\n{prompt}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Model output:\n{model_output}\n\n"
        f"Optional rule snippet:\n{rule_block}\n"
    )


def judge_dakota(
    prompt: str,
    gold: str,
    model_output: str,
    rule_snippet: str | None = None,
    complete_fn: CompleteFn | None = None,
) -> JudgeResult:
    if complete_fn is None:
        raise ValueError(
            "judge_dakota requires complete_fn or a configured judge endpoint."
        )
    raw = complete_fn(build_judge_prompt(prompt, gold, model_output, rule_snippet))
    payload = _extract_json_object(raw)
    return JudgeResult(
        correct=_as_flag(payload.get("correct")),
        morphology_ok=_as_flag(payload.get("morphology_ok")),
        meaning_ok=_as_flag(payload.get("meaning_ok")),
        orthography_ok=_as_flag(payload.get("orthography_ok")),
        rationale=str(payload.get("rationale") or "").strip(),
    )


def _post_chat_completion(base_url: str, api_key: str, model: str, prompt: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Judge request failed: {exc}") from exc
    return str(payload["choices"][0]["message"]["content"])


def judge_from_env() -> Optional[Callable[..., JudgeResult]]:
    """Return a judge callable if an OpenAI-compatible endpoint is configured.

    Environment variables:
    - QWEN_JUDGE_MODEL (default Qwen/Qwen3-8B)
    - QWEN_JUDGE_BASE_URL or OPENAI_BASE_URL
    - QWEN_JUDGE_API_KEY or OPENAI_API_KEY
    """
    base_url = os.environ.get("QWEN_JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return None
    api_key = os.environ.get("QWEN_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    model = os.environ.get("QWEN_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL

    def _judge(
        prompt: str,
        gold: str,
        model_output: str,
        rule_snippet: str | None = None,
    ) -> JudgeResult:
        return judge_dakota(
            prompt=prompt,
            gold=gold,
            model_output=model_output,
            rule_snippet=rule_snippet,
            complete_fn=lambda text: _post_chat_completion(base_url, api_key, model, text),
        )

    return _judge
