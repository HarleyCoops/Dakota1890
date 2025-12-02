# Railroad Engineer 1959

This project aims to create a railroad extraction pipeline and an RL environment for training agents on railroad safety rules.

## Structure

- `railroad_extraction/`: Pipeline to convert PDF rules into tasks.
- `railroad_rl_training/`: RL training pipeline and verifiers.
    - See `railroad_rl_training/TINKER_LAUNCH.md` for the Thinking Machines (Tinker) path that wraps the PrimeIntellect `railroad_1959` gym (defaults to `Qwen/Qwen3-4B-Instruct-2507`, deterministic local rubric—no Anthropic).
