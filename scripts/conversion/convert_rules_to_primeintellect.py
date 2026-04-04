#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert organized Dakota grammar rules into PrimeIntellect/Tinker task JSONL."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import re


def extract_special_chars(text: str) -> List[str]:
    """Extract Dakota special characters from text"""
    special_chars = set("ćšŋḣṡáéíóúķśṅźėčžʼ")
    return sorted(list(set(c for c in text if c in special_chars)))


def extract_affixes(rule: Dict) -> List[str]:
    """Extract affixes from rule pattern"""
    pattern = rule.get('dakota_pattern', '')

    # Look for affix markers: -suffix, prefix-, or explicit mentions
    affixes = []

    # Find patterns like "-ku", "ta-", "-ću"
    affix_pattern = r'(?:^|\s)((?:\w+)?-\w+|-\w+(?:\w+)?|[\w]+-)(?:\s|$|,|\.)'
    matches = re.findall(affix_pattern, pattern)
    affixes.extend(matches)

    # Also check in constraints
    constraints = rule.get('constraints', '')
    matches = re.findall(affix_pattern, constraints)
    affixes.extend(matches)

    return list(set(affixes))


def create_morphology_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create basic morphology task from rule and example"""

    dakota_text = example.get('dakota', '')
    if not dakota_text:
        return None

    # Create prompt
    prompt = f"Apply this Dakota grammar rule: {rule['rule_description']}\n\n"

    if rule.get('dakota_pattern'):
        prompt += f"Pattern: {rule['dakota_pattern']}\n\n"

    if example.get('notes'):
        prompt += f"Context: {example['notes']}\n\n"

    prompt += f"Transform or analyze: {dakota_text}"

    # Answer is the English explanation or gloss
    answer = example.get('english', dakota_text)

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "morphology",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "base_form": dakota_text,
            "required_affixes": extract_affixes(rule),
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_multi_step_morphology_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """
    Create multi-step morphology task that requires progressive affix insertion.
    
    Example: Start with base form, then add possessive, then add plural, etc.
    """
    dakota_text = example.get('dakota', '')
    if not dakota_text:
        return None
    
    affixes = extract_affixes(rule)
    if len(affixes) < 2:  # Need multiple affixes for multi-step
        return None
    
    # Create multi-step prompt
    prompt = f"Apply this Dakota grammar rule step-by-step: {rule['rule_description']}\n\n"
    prompt += f"Pattern: {rule['dakota_pattern']}\n\n"
    prompt += f"Start with: {dakota_text}\n\n"
    prompt += "Task: Apply the affixes in order to create the complete form.\n"
    prompt += "Show each intermediate step:\n"
    prompt += "1. Base form\n"
    prompt += "2. After first affix\n"
    prompt += "3. Final form with all affixes\n"
    
    # Answer should show the progression
    answer = f"Step 1: {dakota_text}\n"
    # This is simplified - in practice, you'd derive intermediate forms
    if example.get('english'):
        answer += f"Step 2: [apply first affix]\n"
        answer += f"Final: {dakota_text} [with all affixes applied]"
    
    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "multi_step_morphology",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "base_form": dakota_text,
            "required_affixes": affixes,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": "hard",  # Multi-step is always harder
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0),
            "steps": len(affixes) + 1  # Number of transformation steps
        }
    }


def create_positive_negative_evidence_task(rule: Dict) -> Optional[Dict]:
    """
    Create task that shows positive examples (correct) and negative examples (incorrect)
    to help model learn the rule boundaries.
    """
    positive_examples = rule.get('positive_examples', [])
    negative_examples = rule.get('negative_examples', [])
    
    if not positive_examples or not negative_examples:
        return None
    
    # Pick one positive and one negative example
    pos_ex = positive_examples[0]
    neg_ex = negative_examples[0]
    
    prompt = f"Study this Dakota grammar rule: {rule['rule_description']}\n\n"
    prompt += "Correct example (follows the rule):\n"
    if pos_ex.get('dakota'):
        prompt += f"   {pos_ex['dakota']}"
        if pos_ex.get('english'):
            prompt += f" - {pos_ex['english']}"
        prompt += "\n\n"
    
    prompt += "Incorrect example (violates the rule):\n"
    if neg_ex.get('dakota'):
        prompt += f"   {neg_ex['dakota']}"
        if neg_ex.get('english'):
            prompt += f" - {neg_ex['english']}"
        prompt += "\n\n"
    
    prompt += "Explain why the incorrect example is wrong and how to fix it."
    
    answer = f"The incorrect example violates the rule because: [explanation]. "
    answer += f"The correct form should follow the pattern: {rule.get('dakota_pattern', 'N/A')}"
    
    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "positive_negative_evidence",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "positive_example": pos_ex.get('dakota', ''),
            "negative_example": neg_ex.get('dakota', ''),
            "special_chars": extract_special_chars(pos_ex.get('dakota', '') + neg_ex.get('dakota', '')),
            "difficulty": "medium",
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_affix_insertion_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """
    Create task that tests affix insertion at specific positions.
    
    Example: "Insert the possessive suffix -ku into 'iŋhiŋ' at the correct position"
    """
    dakota_text = example.get('dakota', '')
    if not dakota_text:
        return None
    
    affixes = extract_affixes(rule)
    if not affixes:
        return None
    
    # Find base form (strip affixes if present)
    base_form = dakota_text
    for affix in affixes:
        affix_clean = affix.strip("-")
        if affix.startswith("-"):
            # Suffix - remove it
            base_form = re.sub(rf'{re.escape(affix_clean)}$', '', base_form)
        elif affix.endswith("-"):
            # Prefix - remove it
            base_form = re.sub(rf'^{re.escape(affix_clean)}', '', base_form)
    
    if base_form == dakota_text:
        return None  # No affixes to insert
    
    prompt = f"Insert the Dakota affix(es) into the base form:\n\n"
    prompt += f"Base form: {base_form}\n"
    prompt += f"Affix(es) to insert: {', '.join(affixes)}\n"
    prompt += f"Rule: {rule['rule_description']}\n\n"
    prompt += "Show where each affix should be inserted and write the complete form."
    
    answer = dakota_text
    if example.get('english'):
        answer += f" ({example['english']})"
    
    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "affix_insertion",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "base_form": base_form,
            "target_form": dakota_text,
            "required_affixes": affixes,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": "medium",
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_exception_trigger_task(rule: Dict) -> Optional[Dict]:
    """
    Create task that tests knowledge of exceptions to the rule.
    
    Uses negative_examples or exceptions field to test when rule doesn't apply.
    """
    exceptions = rule.get('exceptions', [])
    negative_examples = rule.get('negative_examples', [])
    
    if not exceptions and not negative_examples:
        return None
    
    prompt = f"This Dakota grammar rule has exceptions: {rule['rule_description']}\n\n"
    prompt += f"Pattern: {rule['dakota_pattern']}\n\n"
    
    if exceptions:
        prompt += "Exceptions:\n"
        for exc in exceptions[:3]:
            prompt += f"  - {exc}\n"
        prompt += "\n"
    
    if negative_examples:
        prompt += "Words that DON'T follow this rule:\n"
        for neg_ex in negative_examples[:2]:
            if neg_ex.get('dakota'):
                prompt += f"  - {neg_ex['dakota']}"
                if neg_ex.get('english'):
                    prompt += f" ({neg_ex['english']})"
                prompt += "\n"
        prompt += "\n"
    
    prompt += "Explain why these are exceptions and what rule they follow instead."
    
    answer = f"These words are exceptions because: [explanation]. "
    if rule.get('constraints'):
        answer += f"They don't meet the constraint: {rule['constraints']}"
    
    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "exception_trigger",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "exceptions": exceptions[:3] if exceptions else [],
            "difficulty": "hard",  # Exceptions are always harder
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_translation_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create translation task from rule and example"""

    dakota_text = example.get('dakota', '')
    english_text = example.get('english', '')

    if not dakota_text or not english_text:
        return None

    # Create prompt for Dakota → English translation
    prompt = f"Translate this Dakota sentence to English:\n\n{dakota_text}"

    if example.get('gloss'):
        prompt += f"\n\nGloss: {example['gloss']}"

    answer = english_text

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "word_translation" if len(dakota_text.split()) <= 3 else "sentence_translation",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "dakota_text": dakota_text,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_reverse_translation_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create reverse translation task (English → Dakota)"""

    dakota_text = example.get('dakota', '')
    english_text = example.get('english', '')

    if not dakota_text or not english_text:
        return None

    # Create prompt for English → Dakota translation
    prompt = f"Translate this English sentence to Dakota:\n\n{english_text}"

    if example.get('notes'):
        prompt += f"\n\nNote: {example['notes']}"

    answer = dakota_text

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "reverse_translation",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "english_text": english_text,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": "advanced",  # Reverse translation is always hard
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_syntax_task(rule: Dict, example: Dict) -> Optional[Dict]:
    """Create syntax task from rule and example"""

    dakota_text = example.get('dakota', '')
    if not dakota_text:
        return None

    # Create prompt
    prompt = f"Analyze the syntax of this Dakota sentence:\n\n{dakota_text}\n\n"
    prompt += f"Rule: {rule['rule_description']}"

    if example.get('gloss'):
        prompt += f"\n\nGloss: {example['gloss']}"

    # Answer explains the syntax
    answer = example.get('english', '') or rule.get('english_explanation', '')

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "syntax",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "dakota_text": dakota_text,
            "special_chars": extract_special_chars(dakota_text),
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def create_pattern_identification_task(rule: Dict) -> Optional[Dict]:
    """Create pattern identification task"""

    if not rule.get('dakota_pattern'):
        return None

    prompt = "Identify the grammatical pattern in this Dakota rule:\n\n"
    prompt += f"{rule['rule_description']}\n\n"

    # Show examples if available
    if rule.get('positive_examples'):
        prompt += "Examples:\n"
        for ex in rule['positive_examples'][:2]:
            if ex.get('dakota'):
                prompt += f"  - {ex['dakota']}\n"

    answer = rule['dakota_pattern']

    return {
        "prompt": prompt,
        "answer": answer,
        "info": {
            "task_type": "identify_pattern",
            "rule_id": rule['rule_id'],
            "rule_type": rule['rule_type'],
            "pattern": rule['dakota_pattern'],
            "difficulty": rule['difficulty'],
            "source_pages": rule.get('source_pages', []),
            "confidence": rule.get('confidence', 0.0)
        }
    }


def convert_rule_to_tasks(rule: Dict) -> List[Dict]:
    """Convert a single rule to multiple training tasks"""

    tasks = []

    # Get examples
    positive_examples = rule.get('positive_examples', [])

    if not positive_examples:
        # If no examples, create a pattern identification task
        task = create_pattern_identification_task(rule)
        if task:
            tasks.append(task)
        return tasks

    # Create tasks based on rule type
    rule_type = rule.get('rule_type', '')

    for example in positive_examples:
        # Always create translation tasks if we have both Dakota and English
        if example.get('dakota') and example.get('english'):
            # Forward translation (Dakota → English)
            trans_task = create_translation_task(rule, example)
            if trans_task:
                tasks.append(trans_task)

            # Reverse translation (English → Dakota) - only for some difficulties
            if rule.get('difficulty') in ['medium', 'hard']:
                rev_task = create_reverse_translation_task(rule, example)
                if rev_task:
                    tasks.append(rev_task)

        # Type-specific tasks
        if rule_type == 'morphology':
            # Basic morphology task
            morph_task = create_morphology_task(rule, example)
            if morph_task:
                tasks.append(morph_task)
            
            # Multi-step morphology (if multiple affixes)
            if len(extract_affixes(rule)) >= 2:
                multi_step_task = create_multi_step_morphology_task(rule, example)
                if multi_step_task:
                    tasks.append(multi_step_task)
            
            # Affix insertion task
            affix_task = create_affix_insertion_task(rule, example)
            if affix_task:
                tasks.append(affix_task)

        elif rule_type == 'syntax':
            syntax_task = create_syntax_task(rule, example)
            if syntax_task:
                tasks.append(syntax_task)

    # Add pattern identification task
    if rule.get('dakota_pattern'):
        pattern_task = create_pattern_identification_task(rule)
        if pattern_task:
            tasks.append(pattern_task)
    
    # Add positive/negative evidence task (once per rule, not per example)
    if rule.get('positive_examples') and rule.get('negative_examples'):
        evidence_task = create_positive_negative_evidence_task(rule)
        if evidence_task:
            tasks.append(evidence_task)
    
    # Add exception trigger task (once per rule)
    if rule.get('exceptions') or rule.get('negative_examples'):
        exception_task = create_exception_trigger_task(rule)
        if exception_task:
            tasks.append(exception_task)

    return tasks


def main():
    print("\n" + "="*70)
    print(" CONVERTING RULES TO PRIMEINTELLECT FORMAT")
    print("="*70)

    # Load all rules
    rules_file = Path("data/rl_training_rules/all_rl_rules.json")

    if not rules_file.exists():
        print(f"\nERROR: Rules file not found: {rules_file}")
        print("Run: python organize_grammar_for_rl.py first")
        return

    with open(rules_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_rules = data['rules']
    print(f"\nLoaded {len(all_rules)} rules")

    # Convert all rules to tasks
    all_tasks = []
    stats = {
        'morphology': 0,
        'multi_step_morphology': 0,
        'affix_insertion': 0,
        'positive_negative_evidence': 0,
        'exception_trigger': 0,
        'syntax': 0,
        'translation': 0,
        'reverse_translation': 0,
        'identify_pattern': 0,
        'word_translation': 0,
        'sentence_translation': 0
    }

    difficulty_stats = {
        'easy': 0,
        'medium': 0,
        'hard': 0
    }

    for rule in all_rules:
        tasks = convert_rule_to_tasks(rule)
        all_tasks.extend(tasks)

        # Track stats
        for task in tasks:
            task_type = task['info']['task_type']
            stats[task_type] = stats.get(task_type, 0) + 1

            difficulty = task['info']['difficulty']
            difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1

    print(f"\nGenerated {len(all_tasks)} training tasks")
    print("\nBy task type:")
    for task_type, count in sorted(stats.items()):
        if count > 0:
            print(f"  {task_type:25s}: {count:4d} tasks")

    print("\nBy difficulty:")
    for difficulty, count in sorted(difficulty_stats.items()):
        print(f"  {difficulty:10s}: {count:4d} tasks")

    # Save to JSONL
    output_dir = Path("dakota_rl_training/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "grammar_tasks_complete.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {output_file}")

    # Also save by difficulty for curriculum learning
    for difficulty in ['easy', 'medium', 'hard']:
        difficulty_tasks = [t for t in all_tasks if t['info']['difficulty'] == difficulty]

        if difficulty_tasks:
            diff_file = output_dir / f"grammar_tasks_{difficulty}.jsonl"
            with open(diff_file, 'w', encoding='utf-8') as f:
                for task in difficulty_tasks:
                    f.write(json.dumps(task, ensure_ascii=False) + '\n')
            print(f"  Saved {len(difficulty_tasks)} {difficulty} tasks to: {diff_file.name}")

    # Generate sample tasks file for inspection
    sample_file = output_dir / "sample_tasks.json"
    sample_tasks = all_tasks[:10]

    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_tasks, f, indent=2, ensure_ascii=False)

    print(f"\nSample tasks (first 10): {sample_file}")

    # Print statistics
    print("\n" + "="*70)
    print(" CONVERSION COMPLETE")
    print("="*70)
    print(f"\nTotal tasks: {len(all_tasks)}")
    print(f"Source rules: {len(all_rules)}")
    print(f"Avg tasks per rule: {len(all_tasks) / len(all_rules):.1f}")

    print("\nNext steps:")
    print("  1. Review sample tasks: cat dakota_rl_training/datasets/sample_tasks.json")
    print("  2. Update training config: edit dakota_rl_training/configs/training_config.yaml")
    print("  3. Test verifier: python test_verifier_integration.py")
    print("  4. Run training: python dakota_rl_training/train.py")
    print()


if __name__ == "__main__":
    main()
