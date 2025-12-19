# Tinker Qwen3-VL Dakota Dictionary Extraction

This folder adds a **Thinking Machines / Tinker** vision backend for the Dakota dictionary extraction pipeline, using **Qwen3-VL** and emitting the **same JSON schema** as the existing Claude extraction (`DictionaryEntry`).

## Requirements

- Environment variable: `TINKER_API_KEY`
- Python deps already in this repo: `tinker`, `tinker-cookbook`, `Pillow`
- If you see an SDK version error from Tinker, upgrade: `python -m pip install -U tinker`

## Run (end-to-end: convert → extract → build datasets)

```powershell
$env:TINKER_API_KEY="tml-..."
python -m dakota_extraction.tinker_qwen3vl.run_extraction `
  --input Dictionary/grammardictionar00riggrich_jp2 `
  --start-page 1 --end-page 3 `
  --model "Qwen/Qwen3-VL-30B-A3B-Instruct" `
  --resize-long-edge 3000
```

Outputs:
- Extracted pages: `data/extracted_qwen3vl_tinker/page_###.json`
- Raw responses: `data/reasoning_traces_qwen3vl_tinker/page_###_qwen3vl_response.txt`
- Datasets: `data/training_datasets_qwen3vl_tinker/`

If you only want extraction (no dataset build / guardrails), add `--skip-datasets`.

## Confirm which pages are dictionary (two-column)

This writes a page-by-page manifest and prints contiguous page ranges:

```powershell
python -m dakota_extraction.tinker_qwen3vl.classify_layout `
  --images data/processed_images `
  --out data/page_layout_manifest.json
```

On the current `data/processed_images` set in this repo, the largest detected two-column span is `95-430` (and page `0437` is missing from the image sequence).

## Run full dictionary extraction (recommended settings)

```powershell
python -m dakota_extraction.tinker_qwen3vl.run_extraction `
  --input data/processed_images `
  --skip-conversion `
  --start-page 95 --end-page 430 `
  --split-columns `
  --skip-datasets `
  --skip-existing
```

## Validate schema compatibility

```powershell
python -m dakota_extraction.tinker_qwen3vl.validate_schema --extracted data/extracted_qwen3vl_tinker
```

## Notes on fidelity (special characters)

- JSON writing uses `ensure_ascii=False` to preserve exact Unicode.
- For highest OCR fidelity, avoid resizing; if you hit model/context limits, set `--resize-long-edge`.
- Image token accounting uses a patch-based estimate (`--patch-size`, default `28`).
