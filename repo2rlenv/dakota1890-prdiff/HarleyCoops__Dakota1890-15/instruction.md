# Issue

**Title:** Fix doc workflow to skip missing mkdocs and add hyperbolic dependency

## Description

## Summary
- guard doc build & deploy steps unless mkdocs config exists
- avoid opening duplicate workflow failure issues
- add hyperbolic dependency and handle missing HyperbolicClient in example

## Testing
- `python tools/update_progress.py`
- `python tools/validators/model_card_validator.py implementation/model_cards/*.md`
- `find . -name "*.py" -not -path "./venv/*" -exec python -m doctest {} +`
- `python -m pydoc -w .`
- `mkdocs build` *(fails: Config file 'mkdocs.yml' does not exist)*

------
https://chatgpt.com/codex/tasks/task_e_68a1e261af60832ebf212739d6492626

## Task

Modify the repository so that the issue described above is resolved. The repository is checked out at base commit `3ee7c470253f`. Edit files in place; the verifier captures your changes via `git diff` and scores them against an oracle patch using SWE-RL-style diff-similarity reward.