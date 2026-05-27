# Dakota1890 Qwen3.6-35B Full GRPO Run Findings

Generated: 2026-05-27T16:58:48

- W&B run: https://wandb.ai/christian-cooper-us/dakota-rl-grammar/runs/owf98569
- Base model: `Qwen/Qwen3.6-35B-A3B`
- Metric rows: 199
- Final step: 198
- Final state path: `tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/weights/final`
- Final sampler path: `tinker://1f23df9c-5d88-59d9-a7e8-dd4e169ea7d0:train:0/sampler_weights/final`

## Key Results

- Composite reward improved from 0.1664 to 0.2297; best observed 0.2297.
- Character-overlap reward improved from 0.1424 to 0.4027; best observed 0.4027.
- Affix reward stayed high, ending at 1.0000.
- Pattern reward is confirmed live: all-task pattern reward was nonzero in 186 of 199 logged training rows, with best all-task value 0.1797.
- `identify_pattern` pattern reward reached 0.9062 and was nonzero in 179 of 199 rows.
- Exact-match reward stayed at 0.0 throughout this mixed-task run, confirming it remains an experiment-design/prompting issue rather than the repaired pattern-channel plumbing bug.
- `composite_diff` stayed exactly 0.0 across the run, confirming that the emitted ledger reconstructs the scalar reward.

## Charts

![Dashboard](qwen36_dakota_full_run_dashboard.png)

![Reward progression](qwen36_reward_progression.png)

![Pattern channel](qwen36_pattern_channel.png)

![Reward components](qwen36_reward_components.png)
