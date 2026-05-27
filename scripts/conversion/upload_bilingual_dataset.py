"""
Upload Dakota bilingual training set to HuggingFace Datasets Hub.

This script takes the bilingual_training_set.jsonl file, splits it into
train/validation sets, and uploads to HuggingFace.
"""

import os
import json
import logging
import random
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict
from datetime import datetime, timezone

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_dataset(input_file: Path, train_ratio: float = 0.8, seed: int = 42) -> tuple[Path, Path]:
    """Split dataset into train and validation sets."""
    logger.info(f"Loading dataset from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_examples = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(all_examples)} examples")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(all_examples)
    
    # Split
    split_idx = int(len(all_examples) * train_ratio)
    train_examples = all_examples[:split_idx]
    valid_examples = all_examples[split_idx:]
    
    logger.info(f"Split: {len(train_examples)} train, {len(valid_examples)} validation")
    
    # Write to temporary files
    train_file = input_file.parent / "bilingual_train.jsonl"
    valid_file = input_file.parent / "bilingual_valid.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    with open(valid_file, 'w', encoding='utf-8') as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    logger.info(f"Created train file: {train_file}")
    logger.info(f"Created validation file: {valid_file}")
    
    return train_file, valid_file


def build_dataset_card(train_count: int, valid_count: int, timestamp: str, repo_id: str) -> str:
    """Build a dataset card for HuggingFace."""
    return f"""---
license: mit
task_categories:
- text-generation
- translation
language:
- en
language_bcp47:
- en-US
- dakota
tags:
- dakota-language
- indigenous-languages
- bilingual
- translation
- language-preservation
- qa-pairs
- reinforcement-learning
- grpo
- composite-rewards
- low-resource-language
size_categories:
- 1K<n<10K
---

# Dakota-English Bilingual Q&A Dataset

## Dataset Description

This dataset contains **{train_count + valid_count:,}** Dakota-English question-answer pairs generated from the 1890 Dakota-English Dictionary by Stephen Return Riggs. This dataset was created as part of a **world's first application** of Group Relative Policy Optimization (GRPO) with composite reward functions for training open-source language models on non-coding linguistic tasks, specifically low-resource Indigenous language preservation.

### Dataset Structure

- **Training examples**: {train_count:,}
- **Validation examples**: {valid_count:,}
- **Total examples**: {train_count + valid_count:,}
- **Format**: JSONL (one Q&A pair per line)
- **Language pairs**: Dakota ↔ English (bidirectional)

### Source

All data is extracted from:
- **Book**: Dakota-English Dictionary (1890) by Stephen Return Riggs
- **Pages**: 95-430 (dictionary entries)
- **Extraction method**: Vision-Language Model (Claude Sonnet 4.6 default for new extraction)
- **Q&A generation**: Google Gemini 3.5 Flash default for new generation

### Data Format

Each example is a JSON object with the following structure:

```json
{{
  "question": "What is the Dakota word for 'to hide'?",
  "answer": "The Dakota word for 'to hide' is a-na'-hma.",
  "source_language": "english",
  "generated_at": "2025-11-05T05:12:30.335812",
  "pair_id": 1
}}
```

### Fields

- **question**: The question text (in English or Dakota)
- **answer**: The answer text (in Dakota or English)
- **source_language**: Either "english" or "dakota" indicating the perspective
- **generated_at**: ISO timestamp of generation
- **pair_id**: Unique identifier for the pair

### Dakota Orthography

The dataset preserves Dakota's complex orthography including:
- **ć** (c-acute): Wićášta, mićú
- **š** (s-caron): wašte, Wićášta
- **ŋ** (eng): éiŋhiŋtku, toŋaŋa
- **ḣ** (h-dot): Various words
- **ṡ** (s-dot): Various words
- **á, é, í, ó, ú** (acute accents): Pitch markers
- **ʼ** (glottal stop): Distinct from apostrophe

### Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")

# Access training data
train_data = dataset["train"]
print(train_data[0])

# Access validation data
valid_data = dataset["validation"]
print(valid_data[0])
```

### Citation

```bibtex
@dataset{{dakota_bilingual_qa_2025,
  title={{Dakota-English Bilingual Q&A Dataset}},
  author={{Riggs, Stephen Return}},
  year={{2025}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```

### License

This dataset is derived from public domain historical materials (1890). The extracted and processed data is available under the MIT license.

### Dataset Creation

- **Extraction date**: {timestamp}
- **Processing pipeline**: Vision extraction → Synthetic Q&A generation via Gemini
- **Quality**: Special character preservation validated at extraction stage
- **Train/Val split**: 80/20 (random seed: 42)

---

## Research Innovation: Composite Rewards for Non-Coding Tasks

### World's First Application

This project represents the **first known application** of GRPO (Group Relative Policy Optimization) with **composite reward functions** for training open-source LLMs on **non-coding linguistic tasks**. While GRPO has been successfully applied to code generation and mathematical reasoning, this work pioneers its use for low-resource language learning—a fundamentally different domain requiring nuanced evaluation.

### Why Composite Rewards Matter for Linguistic Tasks

Traditional RLHF approaches typically use **binary or single-dimensional rewards** (correct/incorrect, BLEU score, human preference). For linguistic tasks, especially low-resource languages with complex orthography, this fails to capture critical nuances:

#### The Problem with Single-Dimensional Rewards

1. **Orthographic Preservation**: A translation can be semantically correct but fail to preserve special characters (ć, š, ŋ), making it unusable for language learning
2. **Morphological Accuracy**: An answer might be "mostly right" but miss critical affixes (e.g., -ku possessive, ta- prefix), leading to grammatically incorrect output
3. **Semantic Fidelity**: Perfect character preservation means nothing if the meaning is wrong

#### Our Composite Reward Solution

We decompose rewards into **task-specific weighted components**:

**For Morphology Tasks** (40% char + 40% affix + 20% semantic):
- **Character Preservation (40%)**: Ensures Dakota orthography is maintained
  - Verifies presence of special characters: ć, š, ŋ, ḣ, ṡ
  - Checks pitch accents: á, é, í, ó, ú
  - Validates glottal stops and syllable markers
- **Affix Accuracy (40%)**: Validates morphological transformations
  - Checks possessive suffixes: -ku, -ću, -tku
  - Verifies possessive prefixes: ta-, ti-, to-
  - Validates compound formation rules
- **Semantic Correctness (20%)**: Ensures meaning is preserved
  - Validates that transformations maintain semantic relationships

**For Translation Tasks** (30% char + 70% semantic):
- Higher weight on semantics since translation is meaning-focused
- Still penalizes character corruption to preserve orthography

**For Reverse Translation** (50% char + 50% semantic):
- Balanced weights since generating Dakota requires BOTH correctness
- Character preservation is critical for learner-facing output

### Why This Is Novel

1. **Task-Adaptive Weighting**: Unlike fixed reward structures, weights adapt to task type (morphology vs. translation vs. reverse translation), reflecting linguistic priorities

2. **Multi-Dimensional Learning Signal**: Provides granular feedback on **what** the model got wrong:
   - Low character reward → orthography issues
   - Low affix reward → morphological errors
   - Low semantic reward → meaning problems
   
   This enables more targeted learning compared to binary "correct/incorrect" signals.

3. **Difficulty-Adjusted Scoring**: Harder tasks receive 1.5x-2.0x multipliers, encouraging curriculum progression while preventing reward hacking on easy tasks

4. **Progressive Multi-Turn Rewards**: In multi-turn environments, rewards increase as the model improves across attempts, incentivizing learning from feedback

### Advantages Over Traditional RLHF

| Aspect | Traditional RLHF | Composite Rewards (This Work) |
|--------|-----------------|-------------------------------|
| **Signal Granularity** | Binary (correct/incorrect) | Multi-dimensional (char/affix/semantic) |
| **Orthography Handling** | Often ignored or binary | Continuous 0-1 preservation score |
| **Morphological Feedback** | None | Explicit affix-level verification |
| **Task Adaptation** | Fixed reward structure | Dynamic weights by task type |
| **Error Diagnosis** | "Wrong" (no specifics) | "Low char reward" or "Missing affix" |
| **Curriculum Learning** | Manual difficulty curation | Automatic via difficulty multipliers |

### Why GRPO Works for This Domain

GRPO (Group Relative Policy Optimization) is particularly well-suited for composite rewards because:

1. **Group Comparison**: Rather than absolute rewards, GRPO compares responses within groups, reducing variance from reward scale differences between task types
2. **Relative Ranking**: Helps models learn linguistic preferences even when absolute reward magnitudes vary significantly (e.g., morphology vs. translation)
3. **Stability**: The relative nature mitigates reward hacking that could occur with composite rewards (e.g., optimizing only for character preservation)

### Implications for Low-Resource Language Training

This methodology demonstrates that **sophisticated RL training** (previously reserved for coding/math) can be effectively applied to **linguistic tasks** when rewards are properly decomposed. Key insights:

- **Orthography matters**: Special characters must be explicitly rewarded, not treated as "nice to have"
- **Morphology requires separate signals**: Affix-level feedback is critical for agglutinative languages
- **Semantics alone is insufficient**: Perfect meaning with corrupted orthography produces unusable output for language learners

### Citation & Research Context

This dataset supports research on:
- Composite reward functions for linguistic RL training
- GRPO applications beyond code generation
- Low-resource language model fine-tuning with RL
- Orthography-preserving translation systems

For the complete training pipeline and environment code, see the associated repository.

---

**Note**: This dataset is part of a language preservation project focused on Dakota language revitalization through modern AI techniques. The composite reward methodology represents a novel contribution to RL-for-language-learning research.
"""


def upload_dataset_to_hf(
    input_file: str = "data/bilingual_training_set.jsonl",
    repo_id: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    train_ratio: float = 0.8,
) -> Dict[str, int]:
    """
    Upload Dakota bilingual dataset to HuggingFace Datasets Hub.
    
    Args:
        input_file: Path to input JSONL file
        repo_id: HuggingFace dataset repo ID (e.g., "username/dakota-bilingual")
        private: Whether to make the dataset private
        token: HuggingFace token (if not provided, will use login or env var)
        train_ratio: Ratio for train/val split (default 0.8)
    
    Returns:
        Dictionary with upload statistics
    """
    load_dotenv()
    
    # Get token
    if not token:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if not token:
        logger.info("No HF_TOKEN found. Attempting interactive login...")
        try:
            login()
            token = os.getenv("HF_TOKEN")
        except Exception as e:
            logger.error(f"Login failed: {e}")
            logger.error("Please set HF_TOKEN in your .env file or run: huggingface-cli login")
            raise
    
    # Get repo ID
    if not repo_id:
        # Try to get username from token if available
        try:
            api_temp = HfApi(token=token)
            username = api_temp.whoami()['name']
            default_repo = f"{username}/dakota-bilingual-qa"
        except:
            default_repo = os.getenv("HF_DATASET_REPO_ID", "HarleyCooper/dakota-bilingual-qa")
        repo_id = os.getenv("HF_DATASET_REPO_ID", default_repo)
        logger.info(f"Using repo_id from env or default: {repo_id}")
    
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Split dataset
    train_file, valid_file = split_dataset(input_path, train_ratio=train_ratio)
    
    # Count examples
    train_count = sum(1 for _ in open(train_file, 'r', encoding='utf-8'))
    valid_count = sum(1 for _ in open(valid_file, 'r', encoding='utf-8'))
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Training examples: {train_count:,}")
    logger.info(f"  Validation examples: {valid_count:,}")
    logger.info(f"  Total: {train_count + valid_count:,}")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repo
    logger.info(f"Creating/updating HuggingFace dataset repo: {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        logger.info(f"[OK] Repository ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logger.error(f"Failed to create repo: {e}")
        raise
    
    # Upload files
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    commit_message = f"Upload Dakota bilingual Q&A dataset ({timestamp} UTC)"
    
    uploads = [
        (train_file, "data/train.jsonl"),
        (valid_file, "data/validation.jsonl"),
    ]
    
    for local_path, remote_path in uploads:
        logger.info(f"Uploading {local_path.name} to {remote_path}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=commit_message,
            )
            logger.info(f"[OK] Uploaded {local_path.name}")
        except Exception as e:
            logger.error(f"Failed to upload {local_path.name}: {e}")
            raise
    
    # Upload dataset card
    logger.info("Uploading dataset card (README.md)...")
    dataset_card = build_dataset_card(train_count, valid_count, timestamp, repo_id)
    try:
        api.upload_file(
            path_or_fileobj=dataset_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=commit_message,
        )
        logger.info("[OK] Uploaded dataset card")
    except Exception as e:
        logger.warning(f"Failed to upload dataset card: {e}")
        # Not critical, continue
    
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    train_file.unlink()
    valid_file.unlink()
    logger.info("[OK] Cleaned up")
    
    logger.info("="*70)
    logger.info("[SUCCESS] Dataset successfully uploaded to HuggingFace!")
    logger.info(f"   Repository: https://huggingface.co/datasets/{repo_id}")
    logger.info("="*70)
    
    return {
        "train_examples": train_count,
        "valid_examples": valid_count,
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload Dakota bilingual Q&A dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with default repo name
  python upload_bilingual_dataset.py
  
  # Upload to specific repo
  python upload_bilingual_dataset.py --repo-id "username/dakota-bilingual-qa"
  
  # Upload as private dataset
  python upload_bilingual_dataset.py --private
  
  # Use custom input file
  python upload_bilingual_dataset.py --input-file "custom_data.jsonl"
        """
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/bilingual_training_set.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID (e.g., 'username/dakota-bilingual-qa')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio for train/val split (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    try:
        result = upload_dataset_to_hf(
            input_file=args.input_file,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
            train_ratio=args.train_ratio,
        )
        
        print("\n" + "="*70)
        print("UPLOAD SUMMARY")
        print("="*70)
        print(f"Repository: {result['repo_id']}")
        print(f"URL: {result['url']}")
        print(f"Training examples: {result['train_examples']:,}")
        print(f"Validation examples: {result['valid_examples']:,}")
        print(f"Total: {result['train_examples'] + result['valid_examples']:,}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

