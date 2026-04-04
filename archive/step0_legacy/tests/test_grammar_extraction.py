"""
Test Grammar Extraction on Dakota Grammar Pages

This script tests the grammar-specific extraction pipeline
optimized for RL training task generation.
"""

import os
import sys
from pathlib import Path
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from blackfeet_extraction.core.grammar_page_processor import GrammarPageProcessor
from blackfeet_extraction.schemas.grammar_schema import validate_grammar_rule

# Load environment variables
load_dotenv()


def test_grammar_extraction():
    """Test grammar extraction on a sample page"""

    print("="*80)
    print("DAKOTA GRAMMAR EXTRACTION TEST")
    print("="*80)

    # Initialize processor
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        print("Please set it in .env file")
        return

    processor = GrammarPageProcessor(api_key=api_key)

    # Test on page 61 (known to have interlinear translations)
    test_page = 61
    image_path = Path(f"data/processed_images/grammardictionar00riggrich_{test_page:04d}.jpg")

    if not image_path.exists():
        print(f"ERROR: Test image not found: {image_path}")
        print("\nTrying alternate path...")
        image_path = Path("dictionary/grammardictionar00riggrich_jp2") / f"grammardictionar00riggrich_{test_page:04d}.jp2"

        if image_path.exists():
            print("Found JP2 file. Converting to JPEG first...")
            from blackfeet_extraction.tools.image_converter import ImageConverter

            output_dir = Path("data/processed_images")
            output_dir.mkdir(parents=True, exist_ok=True)

            converter = ImageConverter(output_dir=str(output_dir))
            jpeg_path = converter.convert_jp2_to_jpeg(
                image_path,
                output_dir / f"grammardictionar00riggrich_{test_page:04d}.jpg"
            )
            image_path = jpeg_path
        else:
            print("ERROR: Could not find source image")
            return

    print(f"\nProcessing page {test_page}...")
    print(f"Image: {image_path}")
    print("Context: Chapter IX - Interlinear Translations")
    print()

    # Process page
    grammar_page = processor.process_page(
        image_path=image_path,
        page_number=test_page,
        page_context="Chapter IX: Interlinear Translations - Parable of the Prodigal Son"
    )

    # Save results
    output_dir = Path("data/grammar_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"grammar_page_{test_page:03d}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(grammar_page.to_json())

    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Page Number: {grammar_page.page_number}")
    print(f"Chapter: {grammar_page.chapter}")
    print(f"Section: {grammar_page.section_title}")
    print(f"Grammar Rules: {len(grammar_page.grammar_rules)}")
    print(f"Interlinear Examples: {len(grammar_page.interlinear_examples)}")
    print(f"Quality Issues: {grammar_page.quality_issues or 'None'}")
    print(f"Extraction Confidence: {grammar_page.extraction_confidence:.2f}")
    print(f"\nOutput saved to: {output_path}")

    # Display grammar rules
    if grammar_page.grammar_rules:
        print("\n" + "="*80)
        print("EXTRACTED GRAMMAR RULES")
        print("="*80)

        for i, rule in enumerate(grammar_page.grammar_rules, 1):
            print(f"\n[Rule {i}] {rule.rule_name}")
            print(f"  Type: {rule.rule_type.value}")
            print(f"  Difficulty: {rule.difficulty.value}")
            print(f"  Pattern: {rule.pattern}")
            print(f"  Testable: {rule.testable}")
            print(f"  Transformations: {len(rule.transformations)}")

            # Validate rule
            is_valid, issues = validate_grammar_rule(rule)
            if not is_valid:
                print(f"   Validation Issues: {issues}")
            else:
                print("   Valid")

            # Show first transformation
            if rule.transformations:
                trans = rule.transformations[0]
                print(f"  Example: {trans.base_form} → {trans.transformed_form}")
                print(f"           {trans.gloss_base} → {trans.gloss_transformed}")

    # Display interlinear examples
    if grammar_page.interlinear_examples:
        print("\n" + "="*80)
        print("INTERLINEAR EXAMPLES")
        print("="*80)

        for i, example in enumerate(grammar_page.interlinear_examples[:5], 1):
            print(f"\n[Example {i}]")
            print(f"  Dakota:  {example.dakota_text}")
            print(f"  Glosses: {' | '.join(example.word_glosses)}")
            print(f"  English: {example.english_translation}")
            if example.special_characters_found:
                print(f"  Special chars: {', '.join(example.special_characters_found)}")

    # Generate RL tasks
    print("\n" + "="*80)
    print("GENERATED RL TRAINING TASKS")
    print("="*80)

    all_tasks = grammar_page.generate_all_rl_tasks()
    print(f"Total tasks generated: {len(all_tasks)}")

    # Save tasks to JSONL
    tasks_path = output_dir / f"rl_tasks_page_{test_page:03d}.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Tasks saved to: {tasks_path}")

    # Show sample tasks
    print("\nSample tasks (first 5):")
    for i, task in enumerate(all_tasks[:5], 1):
        print(f"\n--- Task {i} ---")
        print(f"Type: {task['info'].get('task_type', 'unknown')}")
        print(f"Prompt: {task['prompt']}")
        print(f"Answer: {task['answer']}")
        if task['info'].get('special_chars'):
            print(f"Special chars required: {task['info']['special_chars']}")

    # Statistics
    print("\n" + "="*80)
    print("TASK STATISTICS")
    print("="*80)

    task_types = {}
    for task in all_tasks:
        task_type = task['info'].get('task_type', 'unknown')
        task_types[task_type] = task_types.get(task_type, 0) + 1

    for task_type, count in sorted(task_types.items()):
        print(f"  {task_type}: {count}")

    # Count special characters
    special_char_counts = {}
    for task in all_tasks:
        for char in task['info'].get('special_chars', []):
            special_char_counts[char] = special_char_counts.get(char, 0) + 1

    if special_char_counts:
        print("\nSpecial characters in tasks:")
        for char, count in sorted(special_char_counts.items(), key=lambda x: -x[1]):
            print(f"  {char}: {count} tasks")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\n Grammar extraction successful!")
    print(f" Generated {len(all_tasks)} RL training tasks")
    print(f" Results saved to {output_dir}")


if __name__ == "__main__":
    test_grammar_extraction()
