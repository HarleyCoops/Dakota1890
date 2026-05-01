#!/usr/bin/env python3
"""
Complete Dakota Grammar RL Pipeline
Runs all three phases:
1. Extract grammar from images
2. Organize into RL training rules
3. Create RL environment
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(command, description):
    """Run a command and display output."""
    print("\n" + "="*70)
    print(f" {description}")
    print("="*70)

    result = subprocess.run(
        command,
        shell=False,
        capture_output=False,
        text=True,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with code {result.returncode}")
        sys.exit(1)

    print(f"\n[OK] {description} complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run complete Dakota grammar RL pipeline"
    )
    parser.add_argument(
        "--start-image",
        type=int,
        required=True,
        help="Starting image number (e.g., 31)"
    )
    parser.add_argument(
        "--end-image",
        type=int,
        required=True,
        help="Ending image number (e.g., 92)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for rules (default: 0.5)"
    )

    args = parser.parse_args()

    num_pages = args.end_image - args.start_image + 1

    print("\n" + "="*70)
    print(" DAKOTA GRAMMAR RL COMPLETE PIPELINE")
    print("="*70)
    print(f"\nImage range: {args.start_image:04d} - {args.end_image:04d}")
    print(f"Total pages: {num_pages}")
    print(f"Estimated cost: ${num_pages * 0.25:.2f}")
    print(f"Estimated time: {num_pages * 2} minutes (~{num_pages * 2 / 60:.1f} hours)")
    print(f"Min confidence: {args.min_confidence}")

    # Phase 1: Extract grammar
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "extraction" / "extract_grammar_pages.py"),
            "--pages",
            f"{args.start_image}-{args.end_image}",
            "--yes",
        ],
        f"PHASE 1: Extracting grammar from images {args.start_image}-{args.end_image}"
    )

    # Phase 2: Organize into RL rules
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "rl" / "organize_grammar_for_rl.py"),
            "--input",
            "data/grammar_extracted",
            "--min-confidence",
            str(args.min_confidence),
        ],
        "PHASE 2: Organizing grammar into RL training rules"
    )

    # Phase 3: Create RL environment
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "rl" / "create_grammar_rl_environment.py"),
            "--rules-dir",
            "data/rl_training_rules",
            "--demo-episodes",
            "3",
        ],
        "PHASE 3: Creating RL training environment"
    )

    # Final summary
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE!")
    print("="*70)

    print("\nOutput locations:")
    print("  Grammar extraction: data/grammar_extracted/")
    print("  RL training rules:  data/rl_training_rules/")
    print("  RL environment:     data/rl_environment/")

    print("\nReview outputs:")
    print("  cat data/rl_training_rules/rl_rules_summary.txt")
    print("  cat data/rl_environment/environment_config.json")

    print("\nNext steps:")
    print("  1. Review extracted rules and environment config")
    print("  2. Connect to PrimeIntellect verifier system")
    print("  3. Train RL agent on Dakota grammar")
    print()


if __name__ == "__main__":
    main()
