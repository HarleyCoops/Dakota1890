# Akkadian Translation Pipeline (Old Assyrian)

This folder implements a Dakota1890-style pipeline for Old Assyrian, but tuned
for the Kaggle task: translate Akkadian transliteration to English and score
with the geometric mean of corpus BLEU and chrF++ (micro-averaged across the
full test corpus). The design prioritizes linguistic structure (determinatives,
logograms, affixes) while keeping the data flow reproducible and auditable.

This README is intentionally verbose (PhD-level) so the full research intent is
self-contained in one file.

## 1. Ground-Truth Data (SourceDocuments)

All authoritative inputs live in `Akkadian/SourceDocuments`. These are not
auxiliary files; they are the primary ground truth.

- `train.csv`
  - Parallel data at the document level.
  - Fields: `oare_id`, `transliteration`, `translation`.
  - Primary supervision for transliteration -> English translation.

- `test.csv`
  - Sentence-level transliteration with line boundaries.
  - Fields: `id`, `text_id`, `line_start`, `line_end`, `transliteration`.
  - Target for Kaggle submissions.

- `published_texts.csv`
  - Large corpus of transliterations plus metadata and external links.
  - Fields include `oare_id`, `cdli_id`, `aliases`, `genre_label`,
    `transliteration_orig`, `transliteration`, and URLs such as
    `https://aicuneiform.com/search?q=P359543`.
  - Crucial for grammar induction and normalization tasks.

- `OA_Lexicon_eBL.csv`
  - Lexical forms and normalized/lemmatized forms.
  - Fields: `type`, `form`, `norm`, `lexeme`, `eBL`, `I_IV`, `A_D`, `Female(f)`.
  - Used to build morphology normalization tasks.

- `publications.csv` and `bibliography.csv`
  - OCR-derived texts and bibliographic metadata.
  - Not used in the first pass; reserved for later augmentation once
    SourceDocuments are fully exploited.

## 2. Identifier Model and Link Strategy

The dataset comes from multiple catalog systems. We consolidate around:

- `oare_id` (primary key across train and published_texts)
- `cdli_id` (secondary link to CDLI catalog)
- `aliases`, `label`, and `publication_catalog` (useful for matching OCR sources)

The document index (see below) stores all identifiers plus URLs so later
alignment and retrieval can be reproducible.

## 3. Transliteration Normalization (rule-based, no LLM)

Transliteration is highly structured and includes scribal markup. We apply a
conservative normalization to reduce noise while preserving linguistic signals:

- Remove line numbers such as `1`, `1'`, `1''`.
- Remove scribal certainty markers: `!` and `?`.
- Remove line dividers: `/`.
- Replace word dividers `:` and `.` with spaces.
- Remove parenthetical notes `( ... )`.
- Remove half-brackets and square brackets while keeping text inside.
- Normalize subscripts (Unicode subscript digits -> ASCII digits).
- Standardize breaks and gaps:
  - `[x]` -> `<gap>`
  - `[...]` or `...` or `...` -> `<big_gap>`
- Preserve determinatives in `{...}` and logograms in ALL CAPS.

Translation text is lightly normalized (line number stripping, bracket removal)
without aggressive punctuation changes.

Normalization logic lives in `Akkadian/src/akkadian/preprocessing.py` and is
purely deterministic.

## 4. Document Index (published_texts + train mapping)

We build a unified document index as JSONL:

- Source: `published_texts.csv` (one line per text)
- Enrichment: join `train.csv` by `oare_id` when available
- Output: `Akkadian/data/document_index.jsonl`

Each record includes:

- identifiers (`oare_id`, `cdli_id`, `aliases`, `label`, `publication_catalog`)
- metadata (`genre_label`, `description`, `inventory_position`)
- URLs (`online_transcript`, `online_catalog`, `online_information`, `AICC_translation`)
- transliteration variants (`transliteration_orig`, `transliteration`)
- optional `train_transliteration`, `train_translation`

Script:

```powershell
python Akkadian/scripts/build_document_index.py --source-root Akkadian/SourceDocuments --out Akkadian/data/document_index.jsonl
```

## 5. SFT Dataset Builder (first-pass supervision)

We build a multi-task SFT dataset from SourceDocuments only. This is the
foundation for later grammar induction and RL.

Task types:

1) Translation (train.csv)
   - Prompt: "Translate the following Old Assyrian transliteration into English."
   - Answer: English translation.

2) Transliteration normalization (published_texts.csv)
   - Prompt: "Normalize this Old Assyrian transliteration for ML use."
   - Answer: cleaned transliteration.

3) Lexicon normalization (OA_Lexicon_eBL.csv)
   - Prompt: "Normalize this Old Assyrian word..."
   - Answer: `norm` field.

Optional task:
- Lexeme lookup (`form` -> `lexeme`) if enabled.

Output format (JSONL):

```json
{
  "task_id": "train_translate_000123",
  "prompt": "Translate the following Old Assyrian transliteration into English.\n\n...",
  "answer": "English translation...",
  "info": {
    "task_type": "translate",
    "oare_id": "...",
    "source": "train.csv"
  }
}
```

Script:

```powershell
python Akkadian/scripts/build_sft_dataset.py --source-root Akkadian/SourceDocuments --out Akkadian/data/sft.jsonl
```

## 6. Grammar Rule Induction (GPT-5.2 only)

We induce explicit grammar rules to drive the RL gym. This step is implemented
in `Akkadian/scripts/induce_grammar_rules.py` and is restricted to GPT-5.2
models only.

The induction logic:

- Feed the combined corpus (train + published_texts + lexicon) to GPT-5.2.
- Extract rules with:
  - Affix patterns from hyphenated syllables
  - Determinative behavior in `{...}`
  - Logogram usage (ALL CAPS)
  - Frequent template structures (contract formulas, letters, invoices)
- Output rule JSON with fields:
  - `rule_id`, `pattern`, `constraints`, `positive_examples`, `difficulty`

These rules will seed the RL gym and synthetic dataset generation.

Run (requires `OPENAI_API_KEY`):

```powershell
python Akkadian/scripts/induce_grammar_rules.py --source-root Akkadian/SourceDocuments --out Akkadian/data/grammar_rules/rules.jsonl
```

## 7. RL Gym (planned)

Target metric is the Kaggle score:

- BLEU (corpus BLEU)
- chrF++ (corpus chrF)
- Final reward = sqrt(BLEU * chrF++)

For RL training we approximate the corpus-level metric with sentence-level
components, then micro-average across rollouts. The composite reward will
include structure checks from induced rules:

- Determinative preservation
- Logogram fidelity
- Affix pattern checks
- Length/copy penalties

This mirrors the Dakota composite reward design but replaces Dakota-specific
orthography checks with Akkadian-specific structure.

## 8. OCR Augmentation (later stage)

`publications.csv` contains OCR text blocks from ~900 PDFs. The Kaggle guidance
states that OCR extraction and alignment is essential, but we intentionally
start by exhausting SourceDocuments first.

Later steps will:

- Match OCR text to `published_texts.csv` using identifiers and aliases.
- Translate non-English OCR translations to English.
- Create sentence-level aligned pairs and append to training.

## 9. Outputs

Default outputs live in `Akkadian/data/`:

- `document_index.jsonl` (unified metadata + links)
- `sft.jsonl` (multi-task SFT dataset)

## 10. Dependencies

Install minimal requirements locally:

```powershell
python -m pip install -r Akkadian/requirements.txt
```

## 11. Summary

This pipeline treats SourceDocuments as the canonical ground truth, builds a
multi-task SFT dataset for translation and morphology normalization, and sets
up the foundation for GPT-5.2-driven grammar induction and RL. The goal is not
only translation accuracy but also explicit modeling of Akkadian structure
(scribal conventions, determinatives, logograms, and morphology).
