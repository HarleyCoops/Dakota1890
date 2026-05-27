# Dakota Dictionary Extraction Pipeline for SFT Training

## Overview

This pipeline extracts Dakota-English word pairs from the dictionary section (pages 95-430) and converts them into synthetic Q&A pairs for supervised training.

## Book Structure

- **Pages 1-92**: Grammar rules and linguistic notes. These are not missing from the Q&A pipeline; they belong to the separate grammar-rule extraction/RL task pipeline.
- **Pages 95-430**: Vocabulary dictionary entries (Dakota words with English definitions) for Q&A extraction.
- **Pages 431-441**: Blank, back-cover, circulation-card, or calibration scans; do not include these in paid extraction

## Pipeline Flow

```
Dictionary Images (pages 95-430)
    ↓
[1. Image Conversion] → JP2 → JPEG
    ↓
[2. Dictionary Extraction] → Dakota word + English definition pairs
    ↓
[3. Synthetic Q&A Generation] → Gemini generates Q&A pairs from word definitions
    ↓
[4. Format Conversion] → OpenAI chat format (train/val split)
    ↓
[5. Fine-Tuning] → OpenAI fine-tuning API
```

## Step 1: Dictionary Extraction

**Script**: `extract_dakota_dictionary_v2.py`

**What it extracts**:
- `headword`: Dakota word (e.g., "a-na'-hdo-ka")
- `definition_primary`: English definition (e.g., "to wear a hole in, as in a moccasin")
- `part_of_speech`: v., n., a., etc.
- `inflected_forms`: Conjugations/variants
- All special characters preserved (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.)

**Usage**:
```bash
# Test on first dictionary page
python extract_dakota_dictionary_v2.py --test

# Process first 20 pages
python extract_dakota_dictionary_v2.py --pages 95-114

# Process all dictionary pages
python extract_dakota_dictionary_v2.py --all-dictionary
```

**Output**: `data/extracted/page_095.json`, `page_096.json`, etc.

Each JSON file contains:
```json
{
  "page_metadata": {...},
  "entries": [
    {
      "headword": "a-na'-hdo-ka",
      "definition_primary": "to wear a hole in, as in a moccasin, on something",
      "part_of_speech": "v. a.",
      "inflected_forms": ["anawahdoka"],
      ...
    }
  ]
}
```

## Step 2: Synthetic Q&A Generation

**Script**: `generate_synthetic_dakota.py`

**What it does**:
- Reads all extracted dictionary entries from `data/extracted/*.json`
- Uses Gemini AI to generate diverse Q&A pairs from word-definition pairs
- Creates bidirectional questions (English → Dakota and Dakota → English)
- Ensures special character preservation

**Usage**:
```bash
python generate_synthetic_dakota.py
```

**Output**: `data/bilingual_training_set.jsonl`

Format:
```jsonl
{"question": "What is the Dakota word for 'to hide'?", "answer": "The Dakota word for 'to hide' is a-na'-hma.", "source_language": "english"}
{"question": "What does a-na'-hma mean?", "answer": "a-na'-hma means 'to hide, conceal' in English.", "source_language": "dakota"}
```

## Step 3: Format Conversion

**Script**: `convert_extracted_to_chat.py`

**What it does**:
- Converts Q&A pairs to OpenAI chat format
- Splits into 80/20 train/validation sets
- Adds system message for Dakota language assistant

**Usage**:
```bash
python convert_extracted_to_chat.py
```

**Output**:
- `OpenAIFineTune/dakota_train.jsonl` (80%)
- `OpenAIFineTune/dakota_valid.jsonl` (20%)

Format:
```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Step 4: Fine-Tuning

**Script**: `dakota_openai_finetune.py`

**What it does**:
- Uploads training files to OpenAI
- Launches fine-tuning job
- Tracks progress with W&B and HuggingFace integration

**Usage**:
```bash
python dakota_openai_finetune.py
```

## Key Differences: Dictionary vs Grammar Extraction

| Aspect | Dictionary Extraction (SFT) | Grammar Extraction (RL) |
|--------|----------------------------|-------------------------|
| **Pages** | 95-430 | 1-92 |
| **Purpose** | Vocabulary word-definition pairs for Q&A | Grammar rules for RL training |
| **Output** | `headword` + `definition_primary` | Grammar rules, patterns, examples |
| **Next Step** | Synthetic Q&A generation | RL task generation |
| **Training Type** | Supervised Fine-Tuning (SFT) | Reinforcement Learning (RL) |

## Special Character Preservation

Critical for Dakota language accuracy:

- **ć** (c-acute): Wićášta, mićú
- **š** (s-caron): wašte, Wićášta
- **ŋ** (eng): éiŋhiŋtku, toŋaŋa
- **ḣ** (h-dot): Various words
- **ṡ** (s-dot): Various words
- **á, é, í, ó, ú** (acute accents): Pitch markers
- **ʼ** (glottal stop): Distinct from apostrophe

All extraction scripts preserve these characters exactly as printed in the 1890 dictionary.

## Cost Estimates

- **Dictionary extraction**: ~$0.25 per page
  - 20 pages: ~$5
  - All content pages (336): ~$84.00
  - Current completion run after the May 2026 audit: about 98 pages to process/reprocess, or roughly $24.50 at $0.25/page
- **Synthetic Q&A generation**: Depends on Gemini API pricing
- **OpenAI fine-tuning**: Based on OpenAI pricing

## Next Steps After Extraction

1. **Review extracted entries**: Check `data/extracted/page_*.json` for quality
2. **Generate synthetic Q&A**: Run `generate_synthetic_dakota.py`
3. **Convert to chat format**: Run `convert_extracted_to_chat.py`
4. **Fine-tune**: Run `dakota_openai_finetune.py`

