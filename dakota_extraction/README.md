# Dakota Extraction

This package contains the maintained Dakota extraction path used in step 0 of the repository cleanup:

`Dictionary/` or `grammardictionar00riggrich.pdf` -> `dakota_extraction.run_extraction` -> `data/extracted/` -> downstream SFT and RL artifacts.

## Canonical Entry Point

Use [`run_extraction.py`](/Users/chris/Dakota1890/dakota_extraction/run_extraction.py) for the end-to-end local extraction flow:

```bash
python -m dakota_extraction.run_extraction --start-page 95 --end-page 96
```

That entrypoint handles:

1. JP2 -> JPEG conversion via `tools.image_converter`
2. page-level dictionary extraction via `core.page_processor`
3. dataset assembly via `datasets.training_dataset_builder`

## Active Modules

- `core/page_processor.py`: maintained dictionary-page OCR and structure extraction
- `core/advanced_page_processor.py`: richer dictionary schema extraction used by `extract_dakota_dictionary_v2.py`
- `core/claude_page_processor.py`: Claude-backed page extraction helper
- `datasets/training_dataset_builder.py`: deterministic SFT/RL dataset assembly from saved page JSON
- `tools/image_converter.py`: JP2/JPEG conversion utilities for the Riggs source pages

## Related Docs

- [`PIPELINE.md`](/Users/chris/Dakota1890/PIPELINE.md): canonical DAG and stage descriptions
- [`SETUP.md`](/Users/chris/Dakota1890/SETUP.md): reproducible environment setup and smoke tests
- [`REPO_MAP.md`](/Users/chris/Dakota1890/REPO_MAP.md): repository inventory and file classifications

## Archived Legacy Material

Older guides that mixed Dakota work with Blackfeet experiments were moved to [`archive/step0_legacy/`](/Users/chris/Dakota1890/archive/step0_legacy/). They are preserved for historical context but are no longer the canonical instructions for running the Dakota pipeline.
