# Dakota Language Pipeline - Complete Flow Documentation

## Executive Summary

This document demonstrates the complete Dakota language extraction and RL training pipeline, showing where each JSON output is located and how it feeds into the next step. The pipeline transforms historical 1890s grammar texts into modern RL training data.

**Current Status:**
-  Grammar extraction: Pages 1-92 completed (92 JSON files)
-  Dictionary extraction: Pages 109-128 completed (20 JSON files - test run)
-  RL training rules: 1,085 rules organized
-  Synthetic tasks: 5,657 training tasks generated
-  Ready for RL training

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: IMAGE CONVERSION                        │
│  Source: Dictionary/grammardictionar00riggrich_jp2/*.jp2          │
│  Script: convert_all_images.py                                     │
│  Output: data/processed_images/*.jpg                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│          STEP 2A: GRAMMAR EXTRACTION (Pages 1-92)                  │
│  Script: extract_grammar_pages.py --pages 1-92                     │
│  Input: data/processed_images/grammardictionar00riggrich_*.jpg    │
│  Output: data/grammar_extracted/grammar_page_*.json                │
│                                                                     │
│   Output Location: data/grammar_extracted/                       │
│     • grammar_page_001.json through grammar_page_092.json           │
│     • grammar_combined_31-92.json (combined output)                │
│                                                                     │
│   Statistics:                                                    │
│     • 92 individual page JSON files                                 │
│     • ~1,036 grammar rules extracted                                │
│     • 404 interlinear translation texts                             │
│     • 6 categories: morphology, syntax, phonology, conjugation,    │
│       particles, translation                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│          STEP 2B: DICTIONARY EXTRACTION (Pages 93-440)              │
│  Script: extract_dakota_dictionary_v2.py --pages 109-128          │
│  Input: data/processed_images/grammardictionar00riggrich_*.jpg     │
│  Output: data/extracted/page_*.json                                │
│                                                                     │
│   Output Location: data/extracted/                                │
│     • page_109.json through page_128.json (test run)               │
│     • ~350 dictionary entries per page                             │
│                                                                     │
│   Statistics:                                                    │
│     • 20 pages extracted (test run)                                │
│     • ~7,000 dictionary entries total                              │
│     • Format: {headword, part_of_speech, definition, inflected_forms}│
│                                                                     │
│    NOTE: This was a test run on pages 109-128. Full extraction   │
│     of pages 93-440 would yield ~100,000 entries.                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│     STEP 3: ORGANIZE GRAMMAR FOR RL TRAINING                        │
│  Script: organize_grammar_for_rl.py                                │
│  Input: data/grammar_extracted/grammar_page_*.json                 │
│  Output: data/rl_training_rules/*.json                             │
│                                                                     │
│   Output Location: data/rl_training_rules/                       │
│     • all_rl_rules.json (master file with all 1,085 rules)          │
│     • rules_morphology.json (346 rules)                             │
│     • rules_syntax.json (182 rules)                                 │
│     • rules_translation.json (403 rules)                            │
│     • rules_phonology.json (61 rules)                               │
│     • rules_conjugation.json (66 rules)                             │
│     • rules_particles.json (27 rules)                               │
│     • rl_rules_summary.txt (statistics)                            │
│                                                                     │
│   Transformation:                                                 │
│     • Extracts grammar rules from page JSON files                  │
│     • Converts to RL training format with positive/negative examples│
│     • Estimates difficulty (easy/medium/hard)                       │
│     • Creates verification patterns                                 │
│                                                                     │
│   Statistics:                                                    │
│     • Input: 92 grammar page JSON files                            │
│     • Output: 1,085 RL training rules                               │
│     • 1,868 positive examples                                       │
│     • Average confidence: 0.97                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│       STEP 4: GENERATE SYNTHETIC RL TRAINING TASKS                  │
│  Script: convert_rules_to_primeintellect.py                        │
│  Input: data/rl_training_rules/all_rl_rules.json                  │
│  Output: dakota_rl_training/datasets/*.jsonl                       │
│                                                                     │
│   Output Location: dakota_rl_training/datasets/                 │
│     • grammar_tasks_complete.jsonl (5,657 tasks - all)             │
│     • grammar_tasks_easy.jsonl (1,998 tasks)                        │
│     • grammar_tasks_medium.jsonl (2,155 tasks)                      │
│     • grammar_tasks_hard.jsonl (398 tasks)                          │
│     • sample_tasks.json (first 10 tasks for inspection)            │
│                                                                     │
│   Transformation:                                                 │
│     • 1 RL rule → ~5.5 training tasks                               │
│     • Multiple task types per rule:                                 │
│       - Morphology application tasks                                │
│       - Translation (Dakota → English)                              │
│       - Reverse translation (English → Dakota)                      │
│       - Syntax analysis                                             │
│       - Pattern identification                                      │
│                                                                     │
│   Statistics:                                                    │
│     • Input: 1,085 RL training rules                               │
│     • Output: 5,657 training tasks                                  │
│     • Task distribution:                                           │
│       - Easy: 1,998 tasks (35%)                                    │
│       - Medium: 2,155 tasks (38%)                                  │
│       - Hard: 398 tasks (7%)                                       │
│       - Advanced: 1,106 tasks (20%)                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│            STEP 5: RL TRAINING (Ready to Launch)                    │
│  Script: dakota_rl_training/train.py                               │
│  Input: dakota_rl_training/datasets/*.jsonl                       │
│  Output: dakota_rl_training/checkpoints/*.pt                       │
│                                                                     │
│   Configuration:                                                 │
│     • dakota_rl_training/configs/training_config.yaml              │
│     • dakota_rl_training/configs/train.toml                        │
│     • dakota_rl_training/configs/orch.toml                         │
│     • dakota_rl_training/configs/infer.toml                        │
│                                                                     │
│   Curriculum Learning:                                           │
│     Stage 1: Easy tasks (1,998) → target 80% accuracy              │
│     Stage 2: Medium tasks (2,155) → target 75% accuracy            │
│     Stage 3: Hard tasks (398) → target 70% accuracy                │
│                                                                     │
│   Expected Output:                                               │
│     • Model checkpoints in dakota_rl_training/checkpoints/         │
│     • Training logs and metrics                                    │
│     • Weights & Biases dashboard tracking                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Detailed File Locations and Data Flow

### Stage 1: Grammar Extraction Output

**Script:** `extract_grammar_pages.py`  
**Command:** `python extract_grammar_pages.py --pages 1-92 --yes`

**Input Files:**
- `data/processed_images/grammardictionar00riggrich_0001.jpg` through `0092.jpg`
- Source: Converted from JP2 files in `dictionary/grammardictionar00riggrich_jp2/`

**Output Files:**
```
data/grammar_extracted/
├── grammar_page_001.json      # Page 1 grammar rules
├── grammar_page_002.json      # Page 2 grammar rules
├── ...
├── grammar_page_092.json      # Page 92 grammar rules
└── grammar_combined_31-92.json  # Combined output (pages 31-92)
```

**JSON Structure (grammar_page_*.json):**
```json
{
  "page_number": 31,
  "grammar_rules": [
    {
      "rule_id": "grammar_p1_r1",
      "rule_type": "phonology",
      "rule_description": "Dakota has five vowels...",
      "dakota_pattern": "a, e, i, o, u",
      "english_explanation": "...",
      "examples": [
        {
          "dakota": "a",
          "english": "has the sound of English a in father",
          "confidence": 1.0
        }
      ],
      "confidence": 1.0
    }
  ],
  "interlinear_texts": [...],
  "linguistic_terms": [...]
}
```

**How Next Step Uses This:**
- `organize_grammar_for_rl.py` reads all `grammar_page_*.json` files
- Extracts `grammar_rules` and `interlinear_texts` arrays
- Converts each rule into RL training format

---

### Stage 2: Dictionary Extraction Output

**Script:** `extract_dakota_dictionary_v2.py`  
**Command:** `python extract_dakota_dictionary_v2.py --pages 109-128`

**Input Files:**
- `data/processed_images/grammardictionar00riggrich_0109.jpg` through `0128.jpg`

**Output Files:**
```
data/extracted/
├── page_109.json              # Dictionary entries from page 109
├── page_110.json              # Dictionary entries from page 110
├── ...
└── page_128.json              # Dictionary entries from page 128
```

**JSON Structure (page_*.json):**
```json
{
  "page_metadata": {
    "columns": 2,
    "page_number": 109
  },
  "entries": [
    {
      "entry_id": "page_109_entry_001",
      "headword": "a-na'-hdo-ka",
      "part_of_speech": "v. a.",
      "definition_primary": "to wear a hole in, as in a moccasin",
      "inflected_forms": ["anawahdoka"],
      "confidence": 0.95
    }
  ]
}
```

**Status:** This was a test extraction. Full dictionary would be pages 93-440 (~350 pages).

---

### Stage 3: RL Training Rules Organization

**Script:** `organize_grammar_for_rl.py`  
**Command:** `python organize_grammar_for_rl.py --input data/grammar_extracted/`

**Input Files:** All `data/grammar_extracted/grammar_page_*.json` files (92 files)

**Output Files:**
```
data/rl_training_rules/
├── all_rl_rules.json          # Master file: 1,085 rules
├── rules_morphology.json      # 346 morphology rules
├── rules_syntax.json          # 182 syntax rules
├── rules_translation.json     # 403 translation rules
├── rules_phonology.json       # 61 phonology rules
├── rules_conjugation.json     # 66 conjugation rules
├── rules_particles.json       # 27 particle rules
└── rl_rules_summary.txt       # Statistics summary
```

**JSON Structure (all_rl_rules.json):**
```json
{
  "total_rules": 1085,
  "categories": ["morphology", "syntax", "phonology", ...],
  "rules": [
    {
      "rule_id": "grammar_p?_r1",
      "rule_type": "morphology",
      "rule_description": "...",
      "dakota_pattern": "...",
      "positive_examples": [...],
      "negative_examples": [...],
      "verification_pattern": "verify_morphology: ...",
      "source_pages": [31],
      "confidence": 0.96,
      "difficulty": "easy"
    }
  ]
}
```

**How Next Step Uses This:**
- `convert_rules_to_primeintellect.py` reads `all_rl_rules.json`
- For each rule, generates multiple training tasks (morphology, translation, etc.)
- Creates prompt-answer pairs for RL training

---

### Stage 4: Synthetic Task Generation

**Script:** `convert_rules_to_primeintellect.py`  
**Command:** `python convert_rules_to_primeintellect.py`

**Input File:** `data/rl_training_rules/all_rl_rules.json` (1,085 rules)

**Output Files:**
```
dakota_rl_training/datasets/
├── grammar_tasks_complete.jsonl    # All 5,657 tasks
├── grammar_tasks_easy.jsonl         # 1,998 easy tasks
├── grammar_tasks_medium.jsonl       # 2,155 medium tasks
├── grammar_tasks_hard.jsonl         # 398 hard tasks
└── sample_tasks.json                # First 10 tasks (for inspection)
```

**JSONL Structure (grammar_tasks_*.jsonl):**
Each line is a JSON object:
```json
{
  "prompt": "Translate this Dakota sentence to English:\n\na",
  "answer": "has the sound of English a in father",
  "info": {
    "task_type": "word_translation",
    "rule_id": "grammar_p1_r1",
    "rule_type": "phonology",
    "dakota_text": "a",
    "special_chars": [],
    "difficulty": "easy",
    "source_pages": [31],
    "confidence": 1.0
  }
}
```

**How Next Step Uses This:**
- RL training script (`train.py`) reads JSONL files
- Each line becomes a training task
- Curriculum stages use different JSONL files (easy → medium → hard)
- Prompt-answer pairs train the RL agent

---

### Stage 5: RL Training Configuration

**Script:** `dakota_rl_training/train.py`  
**Command:** `prime-rl train --config dakota_rl_training/configs/training_config.yaml`

**Input Files:** 
- `dakota_rl_training/datasets/grammar_tasks_easy.jsonl` (curriculum stage 1)
- `dakota_rl_training/datasets/grammar_tasks_medium.jsonl` (curriculum stage 2)
- `dakota_rl_training/datasets/grammar_tasks_hard.jsonl` (curriculum stage 3)

**Output Files:**
```
dakota_rl_training/checkpoints/
├── checkpoint_stage1_step500.pt     # Stage 1 checkpoint
├── checkpoint_stage2_step1500.pt    # Stage 2 checkpoint
└── checkpoint_stage3_final.pt       # Final model
```

**Configuration Files:**
- `dakota_rl_training/configs/training_config.yaml` - Main config
- `dakota_rl_training/configs/train.toml` - Trainer config
- `dakota_rl_training/configs/orch.toml` - Orchestrator config
- `dakota_rl_training/configs/infer.toml` - Inference config

---

## Data Flow Summary

| Stage | Input | Processing | Output | Next Stage Input |
|-------|-------|------------|--------|------------------|
| **1. Grammar Extraction** | Images (JP2) | Claude Sonnet 4.5 extraction | `data/grammar_extracted/grammar_page_*.json` | → Step 2 |
| **2. Organize Rules** | `grammar_page_*.json` | Convert to RL format | `data/rl_training_rules/all_rl_rules.json` | → Step 3 |
| **3. Generate Tasks** | `all_rl_rules.json` | Create prompt-answer pairs | `dakota_rl_training/datasets/*.jsonl` | → Step 4 |
| **4. RL Training** | `*.jsonl` files | Train RL agent | `checkpoints/*.pt` | → Model ready |

---

## Key Statistics

### Grammar Extraction (Pages 1-92)
- **Pages Extracted:** 92 pages
- **Grammar Rules:** 1,036 rules extracted
- **RL Rules Created:** 1,085 rules (includes interlinear texts)
- **Interlinear Texts:** 404 examples
- **Positive Examples:** 1,868 examples
- **Average Confidence:** 0.97

### Dictionary Extraction (Test Run: Pages 109-128)
- **Pages Extracted:** 20 pages (test)
- **Dictionary Entries:** ~7,000 entries
- **Full Dictionary:** Pages 93-440 (347 pages remaining)
- **Estimated Full Entries:** ~100,000 entries

### Synthetic Task Generation
- **Total Tasks:** 5,657 tasks from 1,085 rules
- **Easy Tasks:** 1,998 (35%)
- **Medium Tasks:** 2,155 (38%)
- **Hard Tasks:** 398 (7%)
- **Advanced Tasks:** 1,106 (20%)
- **Tasks per Rule:** ~5.5 average

---

## Current Pipeline Status

###  Completed
1.  Image conversion (all 440 pages)
2.  Grammar extraction (pages 1-92)
3.  RL rules organization (1,085 rules)
4.  Synthetic task generation (5,657 tasks)

###  In Progress / Ready
5.  RL training (ready to launch)

###  Future Work
6.  Full dictionary extraction (pages 93-440)
7.  Synthetic vocabulary generation
8.  Combined grammar + vocabulary training

---

## Quick Reference: File Locations

```
data/
├── grammar_extracted/          # Step 1 output
│   ├── grammar_page_001.json
│   ├── ...
│   └── grammar_page_092.json
│
├── rl_training_rules/           # Step 2 output
│   ├── all_rl_rules.json
│   ├── rules_morphology.json
│   └── ...
│
└── extracted/                   # Dictionary extraction (test)
    ├── page_109.json
    └── ...

dakota_rl_training/
├── datasets/                    # Step 3 output
│   ├── grammar_tasks_complete.jsonl
│   ├── grammar_tasks_easy.jsonl
│   ├── grammar_tasks_medium.jsonl
│   └── grammar_tasks_hard.jsonl
│
└── configs/                     # Training configuration
    ├── training_config.yaml
    ├── train.toml
    ├── orch.toml
    └── infer.toml
```

---

## Next Steps

1. **Review Current Output:**
   ```powershell
   # Inspect grammar extraction
   Get-Content data\grammar_extracted\grammar_page_031.json | ConvertFrom-Json | Select-Object -First 1
   
   # Inspect RL rules
   Get-Content data\rl_training_rules\all_rl_rules.json | ConvertFrom-Json | Select-Object total_rules
   
   # Inspect sample tasks
   Get-Content dakota_rl_training\datasets\sample_tasks.json | ConvertFrom-Json | Select-Object -First 1
   ```

2. **Launch RL Training:**
   ```powershell
   cd dakota_rl_training
   prime-rl train --config configs/training_config.yaml --use-toploc
   ```

3. **Complete Dictionary Extraction (Optional):**
   ```powershell
   python extract_dakota_dictionary_v2.py --pages 93-440
   ```

---

**Last Updated:** 2025-01-XX  
**Pipeline Version:** 1.0  
**Status:** Grammar extraction complete, RL training ready

