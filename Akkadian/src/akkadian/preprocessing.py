from __future__ import annotations

import re
from typing import Iterable

# Unicode codepoints are represented with escapes to keep this file ASCII-only.
SUBSCRIPT_MAP = str.maketrans({
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    # Subscript plus (often misused as x in legacy data) and subscript x
    "\u208A": "x",
    "\u2093": "x",
})

ELLIPSIS = "\u2026"
HALF_BRACKET_LEFT = "\u02F9"
HALF_BRACKET_RIGHT = "\u02FA"
RIGHT_SINGLE_QUOTE = "\u2019"

LINE_NUMBER_TOKEN_RE = re.compile(r"(?<!\w)\d+(?:['" + RIGHT_SINGLE_QUOTE + r"]{1,2})?(?!\w)")
PAREN_CONTENT_RE = re.compile(r"\([^)]*\)")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_subscripts(text: str) -> str:
    return text.translate(SUBSCRIPT_MAP)


def replace_gap_markers(text: str, gap_token: str, big_gap_token: str) -> str:
    text = text.replace(gap_token, "__GAP__").replace(big_gap_token, "__BIG_GAP__")

    # Standard ellipsis markers
    text = text.replace(ELLIPSIS, big_gap_token)
    text = text.replace("...", big_gap_token)

    # Bracketed gaps
    text = re.sub(r"\[\s*x\s*\]", gap_token, text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*(?:x\s*){2,}\]", big_gap_token, text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*(?:\.{2,}|" + ELLIPSIS + r")\s*\]", big_gap_token, text)
    text = re.sub(r"\[\s*" + ELLIPSIS + r"\s*" + ELLIPSIS + r"\s*\]", big_gap_token, text)

    text = text.replace("__GAP__", gap_token).replace("__BIG_GAP__", big_gap_token)
    return text


def strip_line_numbers(text: str) -> str:
    # Remove standalone line numbers like 1, 1', 1''
    return LINE_NUMBER_TOKEN_RE.sub(" ", text)


def strip_scribal_markers(text: str, keep_gap_tokens: Iterable[str]) -> str:
    keep_gap_tokens = list(keep_gap_tokens)
    placeholder_map = {token: f"__KEEP_{idx}__" for idx, token in enumerate(keep_gap_tokens)}

    for token, placeholder in placeholder_map.items():
        text = text.replace(token, placeholder)

    # Remove parenthetical notes
    text = PAREN_CONTENT_RE.sub(" ", text)

    # Remove half brackets and square brackets (contents already handled)
    text = text.replace(HALF_BRACKET_LEFT, " ").replace(HALF_BRACKET_RIGHT, " ")
    text = text.replace("[", " ").replace("]", " ")

    # Remove angle brackets but keep contents (scribal insertions)
    text = re.sub(r"<<([^>]+)>>", r"\1", text)
    text = re.sub(r"<(?!gap>|big_gap>)([^>]+)>", r"\1", text)

    # Remove certainty markers and line dividers
    text = text.replace("!", " ").replace("?", " ")
    text = text.replace("/", " ")

    # Replace word dividers with spaces
    text = text.replace(":", " ").replace(".", " ")

    for token, placeholder in placeholder_map.items():
        text = text.replace(placeholder, token)

    return text


def normalize_transliteration(
    text: str,
    *,
    gap_token: str = "<gap>",
    big_gap_token: str = "<big_gap>",
) -> str:
    if text is None:
        return ""

    text = str(text)
    text = normalize_subscripts(text)
    text = replace_gap_markers(text, gap_token, big_gap_token)
    text = strip_line_numbers(text)
    text = strip_scribal_markers(text, keep_gap_tokens=[gap_token, big_gap_token])
    text = normalize_whitespace(text)
    return text


def normalize_translation(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = strip_line_numbers(text)
    # Remove bracket markers but keep content, avoid stripping normal punctuation.
    text = text.replace(HALF_BRACKET_LEFT, " ").replace(HALF_BRACKET_RIGHT, " ")
    text = text.replace("[", " ").replace("]", " ")
    text = PAREN_CONTENT_RE.sub(" ", text)
    text = normalize_whitespace(text)
    return text
