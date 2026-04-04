# Dakota RL W&B Visualizations

This folder contains a ready-to-run Python script that will fetch public CSV/JSON artifacts
from the repository:

- `total_reward_time_series.csv`
- `reward_components_time_series.csv`
- `training_entropy_time_series.csv`
- `policy_probabilities_time_series.csv`
- `reward_components_summary.csv`
- `run_summary.json`
- `trainer_run_7nikv4vp.json`
- `orchestrator_run_29hn8w98.json`

All URLs resolve to the public `wandb_analysis/7nikv4vp` directory in
[HarleyCoops/Dakota1890](https://github.com/HarleyCoops/Dakota1890/tree/main/wandb_analysis/7nikv4vp).

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib requests
python scripts/analysis/dakota_rl_wandb_report.py
```

The script will create:
- `docs/reports/dakota_rl_outputs/dakota_rl_wandb_report.pdf` (multi-page PDF with all charts)
- `docs/reports/dakota_rl_outputs/figures/*.png` (individual charts)
- `docs/reports/dakota_rl_outputs/tables/*.csv` (derived metrics)

## What the charts show

1. **Total Reward** vs. step with a rolling mean: upward learning signal under GRPO.
2. **Reward Growth Rate** (Δreward/Δstep): where learning accelerates or stalls.
3. **Policy Entropy** vs. step: expected reduction as the policy sharpens.
4. **Policy Probabilities**: mean/median trends indicating confidence.
5. **Reward Components**: exact/pattern/affix/character/length terms overlayed over time.
6. **Reward vs. Entropy**: a Pareto-style view.
7. **Component Shares at Best Step**: contribution snapshot.
8. **Best Step Marker**: early-stopping cue on smoothed reward.
9. **Cumulative Reward**: intuition for overall learning progress.
10. **Component Stability**: rolling standard deviation to gauge convergence.

## Notes

- The script intentionally avoids seaborn and subplots, and does not set explicit colors.
- If column names differ slightly, the script uses heuristics to find the most likely fields.
- The run must expose the CSV/JSON artifacts at their public URLs; otherwise fetching will fail.
