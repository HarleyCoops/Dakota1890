"""Strip leaked gold / pattern supervision from Dakota task prompts."""

from __future__ import annotations

import re

_EXAMPLES_RE = re.compile(r"\n+Examples:\n(?:[ \t]*-[ \t]*.+\n?)+", re.MULTILINE)


def strip_leaked_supervision(
    prompt: str,
    gold: str = "",
    pattern: str | None = None,
) -> str:
    """Remove example/pattern lines that copy the gold form into the prompt.

    Single-letter patterns that appear in ordinary English (``b`` in Dakota)
    are not deleted from the rule text. Only ``Examples:`` blocks and
    ``Pattern:`` lines that contain the gold or pattern string are removed.
    """
    text = prompt or ""
    needles = [str(item).strip() for item in (gold, pattern) if item and str(item).strip()]
    if not needles:
        return text.strip()

    def _drop_examples(match: re.Match[str]) -> str:
        block = match.group(0)
        if any(needle in block for needle in needles):
            return "\n"
        return block

    text = _EXAMPLES_RE.sub(_drop_examples, text)
    for needle in needles:
        if len(needle) < 2:
            continue
        text = re.sub(
            rf"(?im)^Pattern:\s*{re.escape(needle)}\s*$",
            "Pattern: [hidden]",
            text,
        )
    return text.strip()
