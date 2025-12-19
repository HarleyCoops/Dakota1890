from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import DatasetPaths, PROMPTS
from .ingest import load_csv, load_source_documents
from .preprocessing import normalize_transliteration, normalize_translation


@dataclass(frozen=True)
class SFTBuildOptions:
    include_normalization: bool = True
    include_lexicon: bool = True
    include_lexeme: bool = False
    normalize_translation: bool = True
    normalize_transliteration: bool = False
    max_train: int = -1
    max_published: int = -1
    max_lexicon: int = -1


def _build_translate_tasks(
    train_df: pd.DataFrame,
    options: SFTBuildOptions,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    rows = train_df.to_dict(orient="records")
    for idx, row in enumerate(rows):
        if options.max_train > 0 and idx >= options.max_train:
            break
        transliteration = row.get("transliteration", "")
        translation = row.get("translation", "")
        if not transliteration or not translation:
            continue
        if options.normalize_transliteration:
            transliteration = normalize_transliteration(transliteration)
        if options.normalize_translation:
            translation = normalize_translation(translation)

        prompt = PROMPTS["translate"].format(transliteration=transliteration)
        tasks.append(
            {
                "task_id": f"train_translate_{idx:06d}",
                "prompt": prompt,
                "answer": translation,
                "info": {
                    "task_type": "translate",
                    "oare_id": row.get("oare_id", ""),
                    "source": "train.csv",
                },
            }
        )
    return tasks


def _build_normalization_tasks(
    published_df: pd.DataFrame,
    options: SFTBuildOptions,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    rows = published_df.to_dict(orient="records")
    for idx, row in enumerate(rows):
        if options.max_published > 0 and idx >= options.max_published:
            break
        orig = row.get("transliteration_orig", "")
        clean = row.get("transliteration", "")
        if not orig or not clean:
            continue
        prompt = PROMPTS["normalize_transliteration"].format(transliteration=orig)
        tasks.append(
            {
                "task_id": f"normalize_transliteration_{idx:06d}",
                "prompt": prompt,
                "answer": clean,
                "info": {
                    "task_type": "normalize_transliteration",
                    "oare_id": row.get("oare_id", ""),
                    "source": "published_texts.csv",
                },
            }
        )
    return tasks


def _build_lexicon_tasks(
    lexicon_df: pd.DataFrame,
    options: SFTBuildOptions,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    rows = lexicon_df.to_dict(orient="records")
    for idx, row in enumerate(rows):
        if options.max_lexicon > 0 and idx >= options.max_lexicon:
            break
        form = row.get("form", "")
        norm = row.get("norm", "")
        lexeme = row.get("lexeme", "")
        if form and norm:
            prompt = PROMPTS["normalize_lexeme"].format(form=form)
            tasks.append(
                {
                    "task_id": f"lexicon_norm_{idx:06d}",
                    "prompt": prompt,
                    "answer": norm,
                    "info": {
                        "task_type": "normalize_lexeme",
                        "source": "OA_Lexicon_eBL.csv",
                        "lexeme": lexeme,
                        "word_type": row.get("type", ""),
                    },
                }
            )
        if options.include_lexeme and form and lexeme:
            prompt = f"Normalize this Old Assyrian word to its dictionary lemma.\n\n{form}"
            tasks.append(
                {
                    "task_id": f"lexicon_lemma_{idx:06d}",
                    "prompt": prompt,
                    "answer": lexeme,
                    "info": {
                        "task_type": "lexeme_lookup",
                        "source": "OA_Lexicon_eBL.csv",
                        "word_type": row.get("type", ""),
                    },
                }
            )
    return tasks


def build_sft_dataset(
    source_root: Path,
    out_path: Path,
    options: Optional[SFTBuildOptions] = None,
) -> int:
    options = options or SFTBuildOptions()
    paths = load_source_documents(source_root)

    tasks: List[Dict[str, object]] = []

    train_df = load_csv(paths.train_csv, usecols=["oare_id", "transliteration", "translation"])
    tasks.extend(_build_translate_tasks(train_df, options))

    if options.include_normalization:
        published_df = load_csv(
            paths.published_texts_csv,
            usecols=["oare_id", "transliteration_orig", "transliteration"],
        )
        tasks.extend(_build_normalization_tasks(published_df, options))

    if options.include_lexicon:
        lexicon_df = load_csv(
            paths.lexicon_csv,
            usecols=["type", "form", "norm", "lexeme"],
        )
        tasks.extend(_build_lexicon_tasks(lexicon_df, options))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    return len(tasks)
