"""Convert Dakota synthetic QA JSONL into OpenAI fine-tuning chat files."""

import argparse
import json
import logging
import os
from pathlib import Path
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_fine_tuning_data(input_file: str, output_dir: str, seed: int = 42) -> dict[str, int]:
    """
    Converts a JSONL file of Q&A pairs to the OpenAI fine-tuning format,
    then splits it into training and validation sets.
    
    Args:
        input_file: Path to input JSONL file with Q&A pairs (e.g., data/bilingual_training_set.jsonl)
        output_dir: Directory to write train/val JSONL files
        seed: Random seed for reproducibility
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_train_file = os.path.join(output_dir, "dakota_train.jsonl")
    output_valid_file = os.path.join(output_dir, "dakota_valid.jsonl")

    data = []
    logger.info(f"Reading and converting data from {input_file}...")
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    question = entry.get("question")
                    answer = entry.get("answer")
                    
                    if not question or not answer:
                        logger.warning(f"Skipping entry on line {line_num} with missing 'question' or 'answer': {entry}")
                        continue

                    # Format as OpenAI chat messages
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a bilingual Dakota-English assistant. You have been fine-tuned on a comprehensive set of Dakota language data from the 1890 Dakota-English Dictionary. Your purpose is to provide accurate translations, explain grammatical concepts, and offer cultural context when appropriate. Always preserve Dakota orthography exactly, including special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.). Respond concisely and accurately."
                        },
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ]
                    data.append({"messages": messages})
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num} in {input_file}: {e}")
                except KeyError as e:
                    logger.warning(f"Skipping entry on line {line_num} with missing key {e} in {input_file}")

    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error reading input file: {e}") from e

    if not data:
        raise ValueError("No data was processed from the input JSONL.")

    # Shuffle and split the data
    logger.info(f"Successfully converted {len(data)} entries. Shuffling and splitting data...")
    random.seed(seed)
    random.shuffle(data)
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    valid_data = data[split_index:]

    # Write training data
    with open(output_train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(train_data)} lines to training file: {output_train_file}")

    # Write validation data
    with open(output_valid_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Wrote {len(valid_data)} lines to validation file: {output_valid_file}")

    logger.info("Data preparation complete.")
    logger.info(f"Train: {len(train_data)} examples ({len(train_data)/len(data)*100:.1f}%)")
    logger.info(f"Valid: {len(valid_data)} examples ({len(valid_data)/len(data)*100:.1f}%)")
    return {
        "total_examples": len(data),
        "train_examples": len(train_data),
        "valid_examples": len(valid_data),
    }


def main() -> None:
    """CLI for producing Dakota OpenAI chat fine-tune files."""
    parser = argparse.ArgumentParser(
        description="Convert Dakota synthetic QA JSONL into OpenAI chat fine-tuning files."
    )
    parser.add_argument(
        "--input-file",
        default="data/bilingual_training_set.jsonl",
        help="Input JSONL with question/answer pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default="OpenAIFineTune",
        help="Destination directory for dakota_train.jsonl and dakota_valid.jsonl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for the train/validation split.",
    )
    args = parser.parse_args()

    stats = prepare_fine_tuning_data(args.input_file, args.output_dir, seed=args.seed)
    logger.info(
        "Prepared %d total examples (%d train / %d valid)",
        stats["total_examples"],
        stats["train_examples"],
        stats["valid_examples"],
    )


if __name__ == "__main__":
    main()

