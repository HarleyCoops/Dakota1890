# Pipeline Audit

This document answers the step-0 audit questions against the current codebase after cleanup. When the code does not support a claim, the answer is marked `UNRESOLVABLE FROM CODE`.

## Executive View

The active Dakota chain is real and inspectable, but it is still a multi-command workflow rather than a one-command pipeline. The repository now has one canonical story:

1. historical source pages
2. VLM extraction
3. grammar-rule organization
4. RL task generation
5. packaged Dakota verifier environment
6. RL training and published inference surface

The synthetic-QA -> OpenAI fine-tune path remains in the repo as a comparison baseline, not the main narrative.

## Stage Audit

### Stage 1: Source Acquisition

- Source material lives in-repo as `grammardictionar00riggrich.pdf` plus `Dictionary/grammardictionar00riggrich_jp2/*.jp2`.
- The extraction scripts assume those JP2 assets exist locally.
- Status: `WORKING LOCALLY`

### Stage 2: VLM-Based OCR / Structured Extraction

- Primary grammar extractor: `scripts/extraction/extract_grammar_pages.py`
- Primary dictionary extractor: `dakota_extraction.core.claude_page_processor` and `advanced_page_processor`, driven by `dakota_extraction.run_extraction` or `scripts/extraction/extract_dakota_dictionary_v2.py`
- Canonical model in code: Claude Sonnet 4.5 via Anthropic API
- Alternative Qwen3-VL extraction code exists under `dakota_extraction/tinker_qwen3vl/`, but it is not the maintained path
- Measured extraction accuracy: `UNRESOLVABLE FROM CODE`
  - the repo contains claims like `92-95%`, but no reproducible benchmark or gold-set evaluation establishes that number
- Error taxonomy: `UNRESOLVABLE FROM CODE`

### Stage 3: Synthetic Dataset Generation

- Dictionary extraction artifacts live in `data/extracted/*.json`
- Synthetic QA generator: `scripts/conversion/generate_synthetic_dakota.py`
- OpenAI chat conversion: `scripts/conversion/convert_extracted_to_chat.py`
- Current preserved OpenAI baseline artifacts:
  - `OpenAIFineTune/dakota_train.jsonl` -> `980`
  - `OpenAIFineTune/dakota_valid.jsonl` -> `245`
- Quality filters:
  - source JSON must include question/answer fields
  - chat conversion now tolerates UTF-8 BOM on Windows
- Human validation step: `UNRESOLVABLE FROM CODE`

### Stage 4: Supervised Fine-Tuning (SFT)

- The active preserved SFT baseline is OpenAI chat-format fine-tuning data plus readiness checks in `scripts/rl/dakota_openai_finetune.py`
- A non-billing API smoke against `fine_tuning.jobs.list(limit=1)` succeeds in the current sandbox, so the launch surface is reachable with the configured key
- There is no active in-repo open-source SFT trainer wired as the canonical path
- Interpretation:
  - “OpenAI-style SFT” in this repo means OpenAI API fine-tune assets, not an OpenAI open-source base model
- Local reproducibility:
  - file-readiness is reproducible
  - live job launch still depends on a valid OpenAI key and paid API access

### Stage 5: Reinforcement Learning (GRPO)

- Organized rules: `data/rl_training_rules/all_rl_rules.json` -> `1,497`
- RL tasks: `dakota_rl_training/datasets/grammar_tasks_complete.jsonl` -> `10,576`
- Packaged environment: `environments/dakota_grammar_translation/dakota_grammar_translation/environment.py`
- Reward components in code:
  - exact match
  - character overlap
  - regex / hint pattern matching
  - affix accuracy
  - length penalty term (currently neutral at `1.0`)
- Dakota Grammar Gym:
  - this is the packaged verifier environment plus rubric, loaded through `load_environment()`
- Infrastructure:
  - local PrimeIntellect-style checks via `dakota_rl_training/train.py`
  - remote Thinking Machines path via `dakota_rl_training/tinker_train.py`

### Stage 6: Model Output and Deployment

- Published adapter/model-card surface: `HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890`
- Saved adapter metadata points to:
  - base model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
  - LoRA rank: `32`
- Local check model in configs: `Qwen/Qwen3-0.6B`
- Evaluation suite in code:
  - reward ledger metrics
  - orthography-sensitive rubric outputs
  - extraction evaluation harness under `eval/`
- Human speaker evaluation: `UNRESOLVABLE FROM CODE`

## Critical Questions

### 1. SFT lineage

Answer:

- The repo preserves an OpenAI API fine-tune baseline in `OpenAIFineTune/` and `scripts/rl/dakota_openai_finetune.py`.
- The RL path uses Qwen-family models.
- Earlier `Qwen2.5` references were stale documentation drift, not the current intended lineage.
- Recommendation: keep the OpenAI baseline as a comparison/education path.

### 2. Exact model card for the deployed model

Answer:

- Adapter id: `HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890`
- Base model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Adapter rank: `32`
- Context window in local configs: `8192`
- Quantization: `UNRESOLVABLE FROM CODE`

### 3. Dakota-specific tokenizer changes

Answer:

- No tokenizer augmentation or vocabulary-extension code was found.
- The model appears to rely on the base tokenizer while preserving Dakota orthography through data and reward design.

### 4. Single entrypoint for the full chain

Answer:

- `NO`
- The repo now has a documented canonical path, but still requires multiple commands across extraction, conversion, RL, and inference.

### 5. External dependencies

Answer:

- `ANTHROPIC_API_KEY`: grammar and dictionary extraction
- `GOOGLE_API_KEY`: synthetic QA generation
- `OPENAI_API_KEY`: optional baseline fine-tune submission or readiness
- `HF_TOKEN`: optional HF inference or publishing
- `TINKER_API_KEY`: remote Tinker RL path
- public-domain source files are already present in-repo

### 6. Copyright status of the 1890 source

Answer:

- The Riggs source appears to be public domain in the United States.
- Basis:
  - the work is dated `1890`
  - the U.S. Copyright Office / Library of Congress states that works published in the United States in `1928` entered the public domain in `2024`, and more generally that U.S. works published in `1929` and earlier are public domain as of `2025`
- Source:
  - [Library of Congress blog: Lifecycle of Copyright: 1928 Works in the Public Domain](https://blogs.loc.gov/copyright/2024/01/lifecycle-of-copyright-1928-works-in-the-public-domain/)
- Inference:
  - by straightforward application of that rule, an 1890 U.S. publication qualifies as public domain

### 7. Dakota-specific vs language-agnostic components

| Stage | Status | Notes |
| --- | --- | --- |
| JP2/PDF ingestion | Mostly generic | source format handling is reusable |
| Page extraction scaffolding | Generic shell, Dakota prompts | model call pattern is reusable; prompt/schema are Dakota-shaped |
| Dictionary schema | Partly generic | fields are general, but examples and orthography assumptions are Dakota-first |
| Synthetic QA generation | Mostly generic | depends on source schema and prompt tuning |
| Reward rubric | Dakota-specific | special characters, affix logic, and task semantics are hardcoded |
| Packaged environment | Reusable shell, Dakota-specific data | `load_environment()` shape is general, content is Dakota |

### 8. BC interior target language

Answer:

- `UNRESOLVABLE FROM CODE`
- No target language or community is specified in code or docs strongly enough to support a grant claim.

### 9. What a generalized language config should look like

Proposed shape:

```yaml
language:
  name: "<community-approved name>"
  iso_code: "<code or local identifier>"
  source_type: "dictionary|grammar|parallel_text|mixed"
source:
  pdf: "path-or-url"
  image_dir: "path-or-url"
  grammar_pages: "1-88"
  dictionary_pages: "95-440"
orthography:
  required_chars: ["...", "..."]
  normalize_rules: []
extraction:
  vlm_provider: "anthropic|gemini|other"
  page_prompt: "prompt-template-id"
dataset:
  rule_output_dir: "data/rl_training_rules"
  rl_task_output: "dakota_rl_training/datasets/grammar_tasks_complete.jsonl"
  sft_output: "OpenAIFineTune/"
reward:
  character_weight: 0.2
  affix_weight: 0.1
  pattern_weight: 0.15
  exact_match_weight: 0.4
community:
  review_required: true
  restricted_registers: []
  provenance_fields: ["source_page", "collector", "approval_status"]
```

### 10. Pragmatics gap

Answer:

- The codebase does not solve pragmatics.
- It learns from dictionary and grammar material, not living discourse.
- The grant narrative should explicitly add a community-in-the-loop layer for:
  - register validation
  - conversational data collection
  - ceremonial or restricted-use review
  - post-training corrections and preference signals

Operational details in code:

- `UNRESOLVABLE FROM CODE`

### 11. Temporal drift: 1890 Dakota vs 2025 Dakota

Answer:

- The code does not address temporal drift directly.
- This should be treated as an explicit limitation and a reason for community review rather than hidden in the grant narrative.

### 12. Existing evaluation metrics

Answer:

- Custom reward metrics exist:
  - exact match
  - character overlap
  - affix accuracy
  - pattern match
  - reward ledger decomposition
- Offline extraction evaluation tooling exists under `eval/`
- BLEU / chrF / perplexity are not part of the active maintained path

### 13. Held-out human-validated test set

Answer:

- `UNRESOLVABLE FROM CODE`
- No clear human-validated held-out Dakota evaluation set is present in the repo

### 14. Can an independent researcher reproduce from README alone?

Answer:

- Before step 0: effectively `NO`
- After step 0 cleanup: closer, but still `NOT FULLY`
- Reason:
  - the new docs make the path legible
  - live reproduction still depends on valid paid API access and environment-specific installation state
  - there is still no single-command pipeline runner
