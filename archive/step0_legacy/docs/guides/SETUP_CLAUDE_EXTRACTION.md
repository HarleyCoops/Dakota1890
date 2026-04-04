# Claude Sonnet 4.5 Extraction Setup

## Why Claude Direct?

**Anthropic's API is more reliable than OpenRouter** for this task:

-  No rate limiting issues

-  Consistent performance

-  Same sophisticated reasoning

-  You're already using Claude!

## Quick Setup (2 minutes)

### 1. Install Anthropic Package

```bash
pip install anthropic

```

### 2. Get Your API Key

1. Go to <https://console.anthropic.com/>

2. Sign in (or create account)

3. Go to "API Keys"

4. Create a new key

5. Copy it

### 3. Add to .env File

```bash

# Edit your .env file and add:

ANTHROPIC_API_KEY=sk-ant-your-key-here

```

### 4. Run Test

```bash
python test_claude_extraction.py

```

That's it! This will:

1. Convert page 89 from JP2 to JPEG

2. Send it to Claude Sonnet 4.5

3. Extract all dictionary entries

4. Show you the results

## What You'll Get

**Output:**

```

data/
├── extracted/
│   └── page_089.json                    # Structured entries

└── reasoning_traces/
    └── page_089_claude_response.txt    # Full Claude response

```

**Example entry from page_089.json:**

```json
{
  "headword": "a-ca-ka",
  "part_of_speech": "v. a.",
  "definition_primary": "to freeze on anything",
  "inflected_forms": ["wacaka", "unkcakaπ"],
  "column": 1,
  "confidence": 0.95,
  "page_number": 89
}

```

## Cost Estimate

**Per page with Claude Sonnet 4.5:**

- Input: ~2,000 tokens (image + prompt) = $0.006

- Output: ~8,000 tokens (structured JSON) = $0.048

- **Total: ~$0.054 per page**

**For all dictionary pages (352 pages):**

- Total cost: ~$19 (much cheaper than OpenRouter!)

## Batch Processing

After testing page 89, process more:

```python

# Edit test_claude_extraction.py and add loop:

for page_num in range(89, 100):  # Pages 89-99

    page_file = Path(f"dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_{page_num:04d}.jp2")
    if page_file.exists():
        image = converter.convert_jp2_to_jpeg(page_file)
        extraction = processor.extract_page(image, page_num)

```

Or create a batch script (I can help with this once test works).

## Advantages Over OpenRouter

1. **Reliability**: Direct API is more stable

2. **Cost**: ~$0.05/page vs $0.25/page

3. **Speed**: No routing overhead

4. **Quality**: Same Claude Sonnet 4.5 model

5. **Support**: Direct Anthropic support

## Troubleshooting

**"anthropic not installed"**

```bash
pip install anthropic

```

**"API key not set"**

- Check .env file has `ANTHROPIC_API_KEY=sk-ant-...`

- Make sure .env is in project root

- Restart terminal/IDE after editing .env

**"Image not found"**

- Check `dictionary/grammardictionar00riggrich_jp2/` exists

- Verify page 89 file: `grammardictionar00riggrich_0089.jp2`

## Next Steps

1. **Run test**: `python test_claude_extraction.py`

2. **Review output**: Check `data/extracted/page_089.json`

3. **If good**: Process more pages

4. **Build datasets**: Use extracted data for model training

---

**Ready?**

```bash
pip install anthropic

# Add ANTHROPIC_API_KEY to .env

python test_claude_extraction.py

```

Should take ~60 seconds and cost ~$0.05!
