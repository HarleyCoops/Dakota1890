"""
Upload Dakota Q&A dataset to HuggingFace Datasets Hub.

This script uploads the formatted training dataset (train/val split)
to HuggingFace for easy sharing and usage.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict
from datetime import datetime

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("ERROR: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_lines(file_path: Path) -> int:
    """Count lines in a JSONL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def build_dataset_card(train_count: int, valid_count: int, timestamp: str, repo_id: str) -> str:
    """Build a dataset card for HuggingFace."""
    return f"""---
license: mit
task_categories:
- text-generation
- translation
language:
- en
- dakota
tags:
- dakota-language
- indigenous-languages
- bilingual
- translation
- language-preservation
size_categories:
- 10K<n<100K
---

# Dakota-English Bilingual Training Dataset

## Dataset Description

This dataset contains **{train_count + valid_count:,}** Dakota-English question-answer pairs generated from the 1890 Dakota-English Dictionary by Stephen Return Riggs. The dataset is designed for supervised fine-tuning of language models on Dakota language tasks.

### Dataset Structure

- **Training examples**: {train_count:,}
- **Validation examples**: {valid_count:,}
- **Total examples**: {train_count + valid_count:,}
- **Format**: OpenAI chat format (JSONL)
- **Language pairs**: Dakota ↔ English (bidirectional)

### Source

All data is extracted from:
- **Book**: Dakota-English Dictionary (1890) by Stephen Return Riggs
- **Pages**: 95-430 (dictionary entries)
- **Extraction method**: Vision-Language Model (Claude Sonnet 4.6 default for new extraction)
- **Q&A generation**: Google Gemini 3.5 Flash default for new generation

### Data Format

Each example follows the OpenAI chat format:

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "You are a bilingual Dakota-English assistant..."
    }},
    {{
      "role": "user",
      "content": "What is the Dakota word for 'to hide'?"
    }},
    {{
      "role": "assistant",
      "content": "The Dakota word for 'to hide' is a-na'-hma."
    }}
  ]
}}
```

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
@dataset{{dakota_bilingual_2025,
  title={{Dakota-English Bilingual Training Dataset}},
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
- **Processing pipeline**: Vision extraction → Synthetic Q&A generation → Format conversion
- **Quality**: Special character preservation validated at extraction stage

---

**Note**: This dataset is part of a language preservation project focused on Dakota language revitalization through modern AI techniques.
"""


def upload_dataset_to_hf(
    train_file: str = "OpenAIFineTune/dakota_train.jsonl",
    valid_file: str = "OpenAIFineTune/dakota_valid.jsonl",
    repo_id: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> Dict[str, int]:
    """
    Upload Dakota dataset to HuggingFace Datasets Hub.
    
    Args:
        train_file: Path to training JSONL file
        valid_file: Path to validation JSONL file
        repo_id: HuggingFace dataset repo ID (e.g., "username/dakota-bilingual")
        private: Whether to make the dataset private
        token: HuggingFace token (if not provided, will use login or env var)
    
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
        repo_id = os.getenv("HF_DATASET_REPO_ID", "HarleyCoops/dakota-bilingual")
        logger.info(f"Using repo_id from env or default: {repo_id}")
    
    # Validate files exist
    train_path = Path(train_file)
    valid_path = Path(valid_file)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_file}")
    
    # Count examples
    train_count = count_lines(train_path)
    valid_count = count_lines(valid_path)
    
    if train_count == 0 or valid_count == 0:
        raise ValueError(f"Empty dataset: train={train_count}, valid={valid_count}")
    
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
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    commit_message = f"Upload Dakota bilingual dataset ({timestamp} UTC)"
    
    uploads = [
        (train_path, "data/train.jsonl"),
        (valid_path, "data/validation.jsonl"),
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
        description="Upload Dakota bilingual dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with default repo name
  python upload_to_huggingface.py
  
  # Upload to specific repo
  python upload_to_huggingface.py --repo-id "username/dakota-bilingual"
  
  # Upload as private dataset
  python upload_to_huggingface.py --private
  
  # Use custom file paths
  python upload_to_huggingface.py \\
    --train-file "custom_train.jsonl" \\
    --valid-file "custom_valid.jsonl"
        """
    )
    
    parser.add_argument(
        "--train-file",
        type=str,
        default="OpenAIFineTune/dakota_train.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--valid-file",
        type=str,
        default="OpenAIFineTune/dakota_valid.jsonl",
        help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID (e.g., 'username/dakota-bilingual')"
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
    
    args = parser.parse_args()
    
    try:
        result = upload_dataset_to_hf(
            train_file=args.train_file,
            valid_file=args.valid_file,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token,
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

