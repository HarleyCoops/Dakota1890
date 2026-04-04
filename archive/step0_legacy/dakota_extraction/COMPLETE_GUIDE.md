

# Dakota Dictionary Extraction - Complete Guide

##  Project Goal

Extract structured linguistic data from the **1890 Dakota-English Dictionary** by Stephen Return Riggs to create training datasets for a Dakota language model, following the approach used by @harleycoops for the Stoney Nakoda language.

##  About the Dictionary

**Source**: U.S. Geological and Geological Survey of the Rocky Mountain Region, 1890
**Author**: Stephen Return Riggs
**Format**: 440 pages, JP2 (JPEG 2000) archival images
**Structure**: Two-column dictionary with rich linguistic metadata

##  Architecture

### What Makes This Different

This isn't simple OCR. We're extracting **structured linguistic data** including:

- Headwords with precise orthography (diacritics, syllable breaks)

- Part of speech classifications

- Etymology and word derivations

- Multiple English definitions

- Inflected forms (conjugations, plurals)

- Cross-references between entries

- Grammatical notes

### Why Qwen3-VL Thinking?

1. **Visual Understanding**: Recognizes bold headwords vs. regular text vs. italic definitions

2. **Reasoning Ability**: Can distinguish between headword and inflected forms

3. **Linguistic Awareness**: Understands dictionary structure and conventions

4. **High Accuracy**: With 6000 reasoning tokens, makes careful extraction decisions

5. **Transparency**: Reasoning traces let you verify extraction logic

##  Dictionary Entry Structure

Each entry follows this pattern:

```

**ki'-ći-ća-šta-ka**, v. of kaštaka; to smile for one,
  —wećićaštaka, uŋkićićaštakapį.

```

**Breakdown:**

- `ki'-ći-ća-šta-ka` = Headword (bold, with syllable breaks)

- `v.` = Part of speech (verb)

- `of kaštaka` = Etymology (derived from "kaštaka")

- `to smile for one` = English definition

- `—wećićaštaka, uŋkićićaštakapį` = Inflected forms

**Our extraction:**

```json
{
  "headword": "ki'-ći-ća-šta-ka",
  "part_of_speech": "v.",
  "derived_from": "kaštaka",
  "definition_primary": "to smile for one",
  "inflected_forms": ["wećićaštaka", "uŋkićićaštakapį"],
  "column": 1,
  "confidence": 0.98
}

```

##  Quick Start

### 1. Environment Setup

```bash

# Install dependencies

pip install -r requirements.txt

# Configure API key

cp .env.template .env

# Edit .env and add: OPENROUTER_API_KEY=your_key_here

```

### 2. Test on One Page

```bash
python extract_dakota_dictionary.py --test

```

This will:

- Convert first JP2 to JPEG

- Extract all entries with reasoning

- Show sample results

- Save to `data/extracted/page_001.json`

**Review the output!** Check if entries look correct before processing more.

### 3. Process More Pages

```bash

# Process pages 1-10

python extract_dakota_dictionary.py --pages 1-10

# Process pages 50-100

python extract_dakota_dictionary.py --pages 50-100

```

### 4. Generate Training Datasets

Datasets are automatically generated after extraction. Find them in `data/training_datasets/`:

- `translation_train.jsonl` - Translation pairs (Dakota ↔ English)

- `translation_val.jsonl` - Validation set

- `translation_test.jsonl` - Test set

- `instruction_dataset.jsonl` - Instruction-following format

- `vocabulary.json` - Complete vocabulary with metadata

- `blackfeet_corpus.txt` - Plain text for language modeling

##  Output Structure

```

data/
├── processed_images/           # Converted JPEG files

│   └── grammardictionar00riggrich_0001.jpg
│
├── extracted/                  # Raw extractions

│   └── page_001.json          # Full linguistic metadata

│
├── reasoning_traces/           # Model reasoning

│   └── page_001_reasoning.json # Why it made each decision

│
└── training_datasets/          # Ready for model training

    ├── translation_train.jsonl
    ├── instruction_dataset.jsonl
    └── vocabulary.json

```

##  Data Schema

See `blackfeet_extraction/schemas/dictionary_schema.py` for the complete schema.

**Key fields:**

- `headword`: Dakota word (exact orthography)

- `part_of_speech`: v., n., a., etc.

- `derived_from`: Root word if derived

- `definition_primary`: Main English meaning

- `definition_secondary`: Additional meanings

- `inflected_forms`: Array of conjugations/variants

- `see_also`: Cross-references

- `confidence`: 0.0-1.0 extraction confidence

##  Training a Language Model

### Translation Model (Dakota ↔ English)

```python

# Use translation_train.jsonl for training

# Each line is a translation pair:

{
  "source": "ki'-ći-ća-šta-ka",
  "target": "to smile for one",
  "direction": "dakota_to_english",
  "entry_id": "page_042_entry_015"
}

```

**Recommended models to fine-tune:**

- mT5 (multilingual)

- NLLB-200 (translation-focused)

- mBART (seq2seq)

### Instruction-Following Model

```python

# Use instruction_dataset.jsonl

{
  "instruction": "Translate this Dakota word to English: ki'-ći-ća-šta-ka",
  "input": "Part of speech: verb\nDerived from: kaštaka",
  "output": "to smile for one"
}

```

**Recommended models:**

- LLaMA 3 8B

- Qwen2.5 7B

- Mistral 7B

### Language Model (Dakota text generation)

```python

# Use blackfeet_corpus.txt for pre-training

# Plain text of all Dakota words and examples

```

##  Advanced Configuration

### Increase Thinking Budget

For difficult pages (water damage, unusual layout):

```python
processor.extract_page(
    image_path=page,
    page_number=42,
    thinking_budget=8000,  # Default is 6000

    page_context="Page has water damage in lower right",
)

```

### Custom Extraction Prompt

Edit `blackfeet_extraction/core/extraction_prompt.py` to adjust extraction instructions.

### Confidence Filtering

```python

# In training_dataset_builder.py

good_entries = [
    e for e in self.entries
    if e.get("confidence", 0) >= 0.8  # Increase threshold

]

```

##  Cost Estimation

**Per page:**

- ~6000 reasoning tokens

- ~2000 completion tokens

- ~$0.25 per page (OpenRouter Qwen3-VL pricing)

**For 440 pages:**

- Estimated cost: $110

- Estimated time: 12-15 hours

- Total tokens: ~3.5M

**Recommendation**: Start with 10-20 pages, review quality, then batch process.

##  Quality Assurance

### 1. Review Reasoning Traces

```bash

# Check how the model made decisions

cat data/reasoning_traces/page_001_reasoning.json

```

Look for:

- Clear logical steps

- Correct identification of headwords

- Proper handling of inflected forms

### 2. Validate Confidence Scores

```python

# Check distribution

python -c "
from blackfeet_extraction.datasets.training_dataset_builder import TrainingDatasetBuilder
builder = TrainingDatasetBuilder()
builder.generate_statistics()
"

```

### 3. Manual Spot Checks

Compare `data/extracted/page_001.json` against the original image. Verify:

- All entries extracted

- Headwords spelled correctly

- Definitions accurate

- Inflected forms properly separated

##  Troubleshooting

### JP2 Files Won't Open

**Error**: `PIL cannot read JP2`
**Solution**: Install OpenJPEG library

```bash

# Windows: Download from https://www.openjpeg.org/

# Linux: sudo apt-get install libopenjp2-7

# Mac: brew install openjpeg

```

### Low Confidence Scores

**Issue**: Many entries < 0.7 confidence
**Solutions**:

- Increase thinking budget to 8000

- Check for page damage/blur

- Add page-specific context

- Review extraction prompt

### Missing Entries

**Issue**: Fewer entries than expected
**Solutions**:

- Check if page is title/front matter

- Increase max_tokens in processor

- Review reasoning trace for clues

### API Rate Limits

**Issue**: Rate limit errors
**Solutions**:

- Add delays between pages

- Use lower tier OpenRouter plan

- Process in smaller batches

##  Next Steps After Extraction

1. **Clean Dataset**: Remove low-confidence entries, fix obvious errors

2. **Augment Data**: Generate back-translations, paraphrases

3. **Add Audio**: Pair with pronunciation recordings if available

4. **Train Models**: Start with small model, iterate

5. **Evaluate**: Test on held-out dictionary pages

6. **Deploy**: Build apps/tools using the trained model

##  Use Cases for Trained Model

- **Translation Tool**: Dakota ↔ English translator

- **Language Learning App**: Interactive vocabulary trainer

- **Spell Checker**: Dakota text correction

- **OCR Enhancement**: Train Dakota-specific OCR

- **Text Generation**: Generate Dakota language content

- **Cultural Preservation**: Digital archive of language

##  Following the Stoney Nakoda Approach

This project mirrors @harleycoops' approach for Stoney Nakoda:

1.  Source historical dictionary materials

2.  Extract structured linguistic data (not just OCR)

3.  Build comprehensive training datasets

4.  Include grammatical metadata

5.  Focus on quality over quantity

6.  Train language model (next step)

7.  Build educational tools (future)

##  References

- Original Dictionary: *Dakota-English Dictionary* (1890), Stephen Return Riggs

- Inspiration: Stoney Nakoda language project by @harleycoops

- Model: Qwen3-VL-235B-A22B-Thinking

- Framework: OpenRouter reasoning tokens API

## 🆘 Getting Help

1. Review reasoning traces in `data/reasoning_traces/`

2. Check example outputs in `data/extracted/`

3. Validate schema in `blackfeet_extraction/schemas/`

4. Adjust prompts in `blackfeet_extraction/core/extraction_prompt.py`

---

**Ready to start?**

```bash
python extract_dakota_dictionary.py --test

```

Then review `data/extracted/page_001.json` and decide next steps! 
