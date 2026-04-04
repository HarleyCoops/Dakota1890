# Contributing

Thanks for helping preserve the Dakota language. Please:

1. **Fork + branch** from `main`.
2. Keep PRs focused and small; include rationale, tests, and docs.
3. **Offline smoke tests** must pass:
   ```bash
   python -m pip install -r requirements.txt
   python -m pip install pytest ruff
   OFFLINE=1 pytest -q
   ruff check .
   ```
4. External API tests (Anthropic/Gemini/OpenAI/PrimeIntellect) are allowed locally, but are **skipped in CI**.
5. For evaluation, prefer the versioned subset in `eval/fixtures` and `eval/run_eval.py`.

## Commit style
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`).
- Reference issues when applicable (e.g., `Fixes #12`).
