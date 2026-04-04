# Dakota Dictionary Extraction - START HERE

## What We Built

A complete pipeline to extract structured linguistic data from your 440-page Dakota dictionary using **Claude Sonnet 4.5** (the AI you're using right now).

## Why This Works Better

**Problem with OpenRouter**: Rate limits, API errors, Unicode issues on Windows
**Solution**: Use Anthropic's Claude API directly - it's what you're already using!

## Quick Start (5 Minutes)

### Step 1: Install Package

```bash
pip install anthropic

```

### Step 2: Add API Key

Edit your `.env` file and add:

```

ANTHROPIC_API_KEY=your-anthropic-key

```

Get key from: <https://console.anthropic.com/>

### Step 3: Run Test

```bash
python test_claude_extraction.py

```

This extracts page 89 (first dictionary page) and shows you the results.

## What It Does

1. **Converts JP2 to JPEG** - Your dictionary is in JP2 format

2. **Sends to Claude Sonnet 4.5** - Same model as this chat

3. **Uses Sophisticated Prompt** - Understands two-column dictionary layout

4. **Extracts Structured Data**:

   - Dakota headwords with diacritics
   - Parts of speech
   - Etymology (root words)
   - English definitions
   - Inflected forms (conjugations)
   - Column position
   - Confidence scores

## Expected Output

**File: `data/extracted/page_089.json`**

```json
{
  "entries": [
    {
      "headword": "ki'-ci-ca-sta-ka",
      "part_of_speech": "v.",
      "derived_from": "kastaka",
      "definition_primary": "to smile for one",
      "inflected_forms": ["wecicastaka", "unkicicastakaπ"],
      "column": 1,
      "confidence": 0.95,
      "page_number": 89
    }
  ]
}

```

## Dictionary Structure

- **Pages 1-88**: Grammar rules (skip for now)

- **Pages 89-440**: Dictionary entries (352 pages - this is what we extract)

## Cost & Time

**Per page:**

- Cost: ~$0.05

- Time: ~30-60 seconds

**All dictionary (352 pages):**

- Cost: ~$19

- Time: ~3-5 hours

## Files Created

### Extraction Tools

- `test_claude_extraction.py` - Simple test script

- `blackfeet_extraction/core/claude_page_processor.py` - Claude processor

- `blackfeet_extraction/schemas/dictionary_schema.py` - Entry structure

- `blackfeet_extraction/core/extraction_prompt.py` - Specialized prompt

### Converters

- `blackfeet_extraction/tools/image_converter.py` - JP2 to JPEG

### Documentation

- `SETUP_CLAUDE_EXTRACTION.md` - Detailed setup guide

- `QUICK_START_DAKOTA.md` - Full project overview

- `CLAUDE.md` - Complete reference for future Claude instances

## After Test Works

### Option 1: Process More Pages Manually

Edit `test_claude_extraction.py` and add a loop to process pages 89-100.

### Option 2: Batch Script

I can create a batch processing script to handle all 352 pages automatically.

### Option 3: Build Datasets

Once you have extractions, run the dataset builder to create:

- Translation pairs (Dakota ↔ English)

- Instruction-following format

- Vocabulary with metadata

- Text corpus for language modeling

## Troubleshooting

**"anthropic not installed"**

```bash
pip install anthropic

```

**"API key not set"**

- Add `ANTHROPIC_API_KEY=sk-ant-...` to `.env` file

- Get key from <https://console.anthropic.com/>

**Unicode errors**

- Already fixed! All Unicode characters removed from Windows version

**"Page 89 not found"**

- Check `dictionary/grammardictionar00riggrich_jp2/` directory exists

- Verify files are named `grammardictionar00riggrich_0089.jp2`

## What's Different from OpenRouter Version

1. **No OpenRouter dependency** - Uses Anthropic directly

2. **No reasoning tokens** - Claude Sonnet 4.5 is smart enough without special mode

3. **Cheaper** - $0.05/page vs $0.25/page

4. **More reliable** - Direct API, no routing

5. **Same quality** - Identical extraction prompt and schema

## The Big Picture

You're following @harleycoops' Stoney Nakoda approach:

1.  Get historical dictionary (440 JP2 pages)

2.  Extract structured linguistic data (not just OCR)

3.  Build training datasets

4. → Train Dakota language model

5. → Build educational tools

---

## Ready to Start?

```bash

# 1. Install

pip install anthropic

# 2. Add key to .env

# ANTHROPIC_API_KEY=sk-ant-your-key

# 3. Test

python test_claude_extraction.py

```

Should complete in ~60 seconds and cost ~$0.05!
