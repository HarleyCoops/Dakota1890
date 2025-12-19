from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import DatasetPaths
from .preprocessing import normalize_transliteration, normalize_translation


PUBLISHED_TEXTS_COLUMNS = [
    "oare_id",
    "online transcript",
    "cdli_id",
    "aliases",
    "label",
    "publication_catalog",
    "description",
    "genre_label",
    "inventory_position",
    "online_catalog",
    "note",
    "interlinear_commentary",
    "online_information",
    "excavation_no",
    "oatp_key",
    "eBL_id",
    "AICC_translation",
    "transliteration_orig",
    "transliteration",
]

TRAIN_COLUMNS = ["oare_id", "transliteration", "translation"]

LEXICON_COLUMNS = ["type", "form", "norm", "lexeme", "eBL", "I_IV", "A_D", "Female(f)", "Alt_lex"]


class SourceDataError(RuntimeError):
    pass


def load_csv(path: Optional[Path], usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if path is None:
        raise SourceDataError("CSV path was not provided.")
    if not path.exists():
        raise SourceDataError(f"CSV path does not exist: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)


def load_source_documents(source_root: Path) -> DatasetPaths:
    paths = DatasetPaths.from_root(source_root)
    if not paths.train_csv.exists():
        raise SourceDataError(f"Missing train.csv at {paths.train_csv}")
    if not paths.test_csv.exists():
        raise SourceDataError(f"Missing test.csv at {paths.test_csv}")
    if not paths.published_texts_csv or not paths.published_texts_csv.exists():
        raise SourceDataError("Missing published_texts.csv")
    if not paths.lexicon_csv or not paths.lexicon_csv.exists():
        raise SourceDataError("Missing OA_Lexicon_eBL.csv")
    return paths


def build_document_index(
    source_root: Path,
    out_path: Path,
    normalize: bool = True,
    max_rows: int = -1,
) -> int:
    paths = load_source_documents(source_root)

    published_df = load_csv(paths.published_texts_csv, usecols=PUBLISHED_TEXTS_COLUMNS)
    if max_rows > 0:
        published_df = published_df.head(max_rows)

    train_df = load_csv(paths.train_csv, usecols=TRAIN_COLUMNS)
    train_map: Dict[str, Dict[str, str]] = {}
    for row in train_df.to_dict(orient="records"):
        key = row.get("oare_id", "")
        if not key:
            continue
        if key not in train_map:
            train_map[key] = {
                "train_transliteration": row.get("transliteration", ""),
                "train_translation": row.get("translation", ""),
            }

    records = []
    for row in published_df.to_dict(orient="records"):
        record = {
            "oare_id": row.get("oare_id", ""),
            "cdli_id": row.get("cdli_id", ""),
            "aliases": row.get("aliases", ""),
            "label": row.get("label", ""),
            "publication_catalog": row.get("publication_catalog", ""),
            "description": row.get("description", ""),
            "genre_label": row.get("genre_label", ""),
            "inventory_position": row.get("inventory_position", ""),
            "excavation_no": row.get("excavation_no", ""),
            "oatp_key": row.get("oatp_key", ""),
            "ebl_id": row.get("eBL_id", ""),
            "links": {
                "online_transcript": row.get("online transcript", ""),
                "online_catalog": row.get("online_catalog", ""),
                "online_information": row.get("online_information", ""),
                "aicc_translation": row.get("AICC_translation", ""),
            },
            "note": row.get("note", ""),
            "interlinear_commentary": row.get("interlinear_commentary", ""),
            "transliteration_orig": row.get("transliteration_orig", ""),
            "transliteration": row.get("transliteration", ""),
            "source": "published_texts.csv",
        }

        train_match = train_map.get(record["oare_id"], {})
        record.update(train_match)
        record["has_train_translation"] = bool(train_match.get("train_translation"))

        if normalize:
            if record["transliteration"]:
                record["transliteration_norm"] = normalize_transliteration(record["transliteration"])
            if record["transliteration_orig"]:
                record["transliteration_orig_norm"] = normalize_transliteration(
                    record["transliteration_orig"]
                )
            if record.get("train_translation"):
                record["train_translation_norm"] = normalize_translation(record["train_translation"])

        records.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)
