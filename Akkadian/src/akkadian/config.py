from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetPaths:
    train_csv: Path
    test_csv: Path
    sample_submission_csv: Path
    published_texts_csv: Optional[Path] = None
    publications_csv: Optional[Path] = None
    bibliography_csv: Optional[Path] = None
    lexicon_csv: Optional[Path] = None

    @staticmethod
    def from_root(root: Path) -> "DatasetPaths":
        return DatasetPaths(
            train_csv=root / "train.csv",
            test_csv=root / "test.csv",
            sample_submission_csv=root / "sample_submission.csv",
            published_texts_csv=_optional(root / "published_texts.csv"),
            publications_csv=_optional(root / "publications.csv"),
            bibliography_csv=_optional(root / "bibliography.csv"),
            lexicon_csv=_optional(root / "OA_Lexicon_eBL.csv"),
        )


def _optional(path: Path) -> Optional[Path]:
    return path if path.exists() else None


PROMPTS = {
    "translate": (
        "Translate the following Old Assyrian transliteration into English. "
        "Return a single English sentence.\n\n{transliteration}"
    ),
    "reverse_translate": (
        "Translate the following English sentence into Old Assyrian transliteration. "
        "Preserve transliteration conventions exactly.\n\n{translation}"
    ),
    "normalize_transliteration": (
        "Normalize this Old Assyrian transliteration for ML use. "
        "Remove scribal notations and standardize gaps.\n\n{transliteration}"
    ),
    "normalize_lexeme": (
        "Normalize this Old Assyrian word by removing hyphens and sign length markers.\n\n{form}"
    ),
}
