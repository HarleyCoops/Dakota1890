"""
Reward Ledger Logging Utilities

Logs detailed reward component breakdowns to W&B and CSV for analysis.
"""

import csv
import statistics as stats
from pathlib import Path
from typing import Dict, List, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


LEDGER_PATH = Path("wandb_analysis") / "reward_ledger.csv"
LEDGER_FIELDS = [
    "step",
    # Weights & multiplier
    "w_exact", "w_char", "w_pattern", "w_affix", "w_length", "difficulty_multiplier",
    # Raw components
    "exact_match_raw", "char_overlap_raw", "pattern_raw", "affix_raw", "length_penalty_raw",
    # Normalized components
    "exact_match_norm", "char_overlap_norm", "pattern_norm", "affix_norm", "length_penalty_norm",
    # Composites
    "composite_pre", "composite_with_length", "composite_predicted",
    # Final reward + consistency check
    "reward_scalar", "composite_diff",
]


def _ensure_csv(path: Path):
    """Ensure CSV file exists with header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            writer.writeheader()


def _mean_from_infos(infos: List[Dict], key: str) -> float:
    """Extract mean value from list of info dicts."""
    vals = [i.get(key) for i in infos if key in i and i[key] is not None]
    if not vals:
        return float("nan")
    try:
        return float(stats.fmean(vals))
    except (TypeError, ValueError):
        # Handle non-numeric values
        return float("nan")


def log_step_ledger(step: int, infos: List[Dict], wandb_log: bool = True):
    """
    Aggregate and log reward ledger data for a training step.
    
    Args:
        step: Training step number
        infos: List of info dicts from environment steps (should contain ledger data)
        wandb_log: Whether to log to W&B (requires wandb to be initialized)
    """
    # Aggregate means across batch
    row = {"step": step}
    for k in LEDGER_FIELDS:
        if k == "step":
            continue
        row[k] = _mean_from_infos(infos, k)
    
    # W&B: flat scalar logs under ledger/* namespace
    if wandb_log and WANDB_AVAILABLE:
        try:
            wandb.log(
                {f"ledger/{k}": v for k, v in row.items() if k != "step"},
                step=step
            )
        except Exception as e:
            print(f"Warning: Failed to log to W&B: {e}")
    
    # CSV: append one row per step
    _ensure_csv(LEDGER_PATH)
    try:
        with open(LEDGER_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            writer.writerow(row)
    except Exception as e:
        print(f"Warning: Failed to write to CSV: {e}")


def extract_ledger_from_info(info: Dict) -> Optional[Dict[str, float]]:
    """
    Extract ledger data from an info dict.
    
    The info dict should contain ledger fields (either directly or nested).
    This handles different possible structures from the environment.
    """
    # Check if ledger is nested
    if "ledger" in info and isinstance(info["ledger"], dict):
        return info["ledger"]
    
    # Check if ledger fields are at top level
    ledger_keys = set(LEDGER_FIELDS) - {"step"}
    if any(k in info for k in ledger_keys):
        return {k: info[k] for k in ledger_keys if k in info}
    
    return None

