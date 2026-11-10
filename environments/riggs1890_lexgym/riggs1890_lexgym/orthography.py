"""Orthography rules for the Riggs (1890) Dakota dictionary.

The VLM extraction wrote the same printed glyph with different Unicode
characters (for example the nasal as U+014B, U+03B7, U+019E, U+1E45).
`canon` maps these variants to one character so that scoring does not punish
a transcription difference that is not in the book.

`is_suspect` flags forms that still contain a character outside the Riggs
alphabet after mapping. Suspect forms are kept in the graph but are never
used as gold answers.
"""
from __future__ import annotations

import re
import unicodedata

# variant -> canonical. Each line is a hypothesis about the extraction, not
# about the Dakota language. Review against the page image before changing.
VARIANT_MAP = {
    "š": "ś", "ṡ": "ś", "ș": "ś",
    "ž": "ź", "ż": "ź",
    "η": "ŋ", "ƞ": "ŋ", "ṅ": "ŋ", "ń": "ŋ", "ņ": "ŋ",
    "ǧ": "ġ", "ğ": "ġ", "ǵ": "ġ",
    "ȟ": "ḣ", "ħ": "ḣ",
    "č": "ć", "ċ": "ć",
    # Riggs prints no accented vowels inside Dakota words (stress is the mark
    # after the syllable). The VLM read the italic c-acute as e-acute.
    "é": "ć",
    "ţ": "ṭ", "ț": "ṭ", "ṫ": "ṭ",
    "ķ": "ḳ",
}
ALPHABET = set("abcdeghikmnopstuwyz" + "ćġḣŋśźṭḳṗ")
# Riggs marks the accented syllable with a mark after it: a-hi'-yo
ACCENT_MARKS = "'’ʹ´ˊ`′"
_ACCENT_RE = re.compile("[" + re.escape(ACCENT_MARKS) + "]")
_TRANS = str.maketrans(VARIANT_MAP)


def canon(text: str) -> str:
    """NFC, lower case, variant glyphs mapped, accent marks unified to '."""
    t = unicodedata.normalize("NFC", text or "").lower().translate(_TRANS)
    t = _ACCENT_RE.sub("'", t)
    return re.sub(r"\s+", " ", t).strip()


def plain(text: str) -> str:
    """Canonical form without syllable hyphens and accent marks."""
    return canon(text).replace("-", "").replace("'", "")


def loose(text: str) -> str:
    """Plain form with the n / ŋ distinction removed.

    The extraction is not consistent for this pair (daŋ / dan), so `loose`
    is used to report near matches. It is never the definition of correct.
    """
    return plain(text).replace("ŋ", "n").replace(" ", "")


def is_suspect(text: str) -> bool:
    """True if a character outside the Riggs alphabet remains."""
    return any(ch not in ALPHABET and ch != " " for ch in plain(text))


def parse_headword(headword: str) -> dict | None:
    """Split a printed headword such as a-hi'-yo-ka-sin.

    Returns None when the syllable division cannot be trusted.
    """
    c = canon(headword)
    if " " in c or not c or is_suspect(c):
        return None
    parts = c.split("-")
    syllables, stress = [], None
    for i, part in enumerate(parts):
        if part.count("'") > 1:
            return None
        if "'" in part:
            if not part.endswith("'") or stress is not None:
                return None
            stress = i
            part = part[:-1]
        if not part:
            return None
        syllables.append(part)
    if stress is None:
        if len(syllables) > 1:
            return None
        stress = 0
    return {
        "form": "-".join(s + ("'" if i == stress else "") for i, s in enumerate(syllables)),
        "plain": "".join(syllables),
        "syllables": syllables,
        "stress": stress,
    }
