"""Smoke tests for the Dakota OpenAI SFT baseline conversion step."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.conversion.convert_extracted_to_chat import prepare_fine_tuning_data


def test_prepare_fine_tuning_data_writes_chat_splits(tmp_path: Path) -> None:
    """The SFT baseline conversion should emit OpenAI chat JSONL files."""
    source_file = tmp_path / "synthetic.jsonl"
    output_dir = tmp_path / "openai"

    records = [
        {"question": "How do you say dog in Dakota?", "answer": "šúŋka"},
        {"question": "Translate šúŋka to English.", "answer": "dog"},
        {"question": "How do you say dream in Dakota?", "answer": "ȟáŋble"},
        {"question": "Translate ȟáŋble to English.", "answer": "dream"},
        {"question": "How do you say river in Dakota?", "answer": "wakpá"},
    ]
    source_file.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    stats = prepare_fine_tuning_data(str(source_file), str(output_dir), seed=7)

    train_path = output_dir / "dakota_train.jsonl"
    valid_path = output_dir / "dakota_valid.jsonl"
    assert train_path.exists()
    assert valid_path.exists()
    assert stats == {"total_examples": 5, "train_examples": 4, "valid_examples": 1}

    train_records = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    valid_records = [json.loads(line) for line in valid_path.read_text(encoding="utf-8").splitlines()]
    assert len(train_records) == 4
    assert len(valid_records) == 1
    assert train_records[0]["messages"][0]["role"] == "system"
    assert train_records[0]["messages"][1]["role"] == "user"
    assert train_records[0]["messages"][2]["role"] == "assistant"
