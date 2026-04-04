#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dakota RL W&B Report Generator
------------------------------
Fetches public CSV/JSON artifacts from:
  https://github.com/HarleyCoops/Dakota1890/tree/main/wandb_analysis/7nikv4vp

Builds an "epic" set of visualizations to illustrate GRPO training
dynamics and compositional rewards, then writes:
  - dakota_rl_wandb_report.pdf  (multi-page PDF)
  - figures/*.png               (individual charts)
  - tables/*.csv                (derived metrics tables)

Requirements:
  pip install pandas numpy matplotlib requests

Usage:
  python scripts/analysis/dakota_rl_wandb_report.py
"""

import io
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests


# -----------------------------
# Configuration
# -----------------------------

BASE = "https://raw.githubusercontent.com/HarleyCoops/Dakota1890/main/wandb_analysis/7nikv4vp/"
URLS = {
    "total_reward": BASE + "total_reward_time_series.csv",
    "components": BASE + "reward_components_time_series.csv",
    "entropy": BASE + "training_entropy_time_series.csv",
    "policy_probs": BASE + "policy_probabilities_time_series.csv",
    "summary": BASE + "reward_components_summary.csv",
    "run_summary": BASE + "run_summary.json",
    "trainer": BASE + "trainer_run_7nikv4vp.json",
    "orchestrator": BASE + "orchestrator_run_29hn8w98.json",
}

# OUT_DIR setup moved to after arg parsing


# -----------------------------
# Utilities
# -----------------------------

def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return json.loads(r.text)

def normalize_step_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a variety of plausible column names to a canonical 'step' and 'timestamp' when available.
    """
    df = df.copy()
    # Step
    step_cols = [c for c in df.columns if c.lower() in ["step", "global_step", "steps", "iteration", "iter"]]
    if step_cols:
        df.rename(columns={step_cols[0]: "step"}, inplace=True)
    else:
        # If no step, make an index
        df["step"] = np.arange(len(df))
    # Timestamp
    ts_cols = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower() or "created" in c.lower()]
    if ts_cols:
        # Try to parse first time-like column
        col = ts_cols[0]
        try:
            df["timestamp"] = pd.to_datetime(df[col])
        except Exception:
            df["timestamp"] = pd.NaT
    else:
        df["timestamp"] = pd.NaT
    # Sort by step
    df.sort_values("step", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def rolling_mean(series: pd.Series, frac: float = 0.02, min_win: int = 5) -> pd.Series:
    n = max(min_win, int(len(series) * frac))
    return series.rolling(n, min_periods=max(3, min_win)).mean()

def save_fig(path: str, title: Optional[str] = None):
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def safe_cols(df: pd.DataFrame, candidates):
    return [c for c in candidates if c in df.columns]

def pct(x: float) -> str:
    return f"{x*100:.2f}%"


# -----------------------------
# Data Loading
# -----------------------------

# -----------------------------
# Data Loading
# -----------------------------

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dakota RL W&B Report")
    parser.add_argument("--history-csv", help="Path to local W&B history CSV (from export_wandb_data.py)")
    parser.add_argument(
        "--output-dir",
        default="docs/reports/dakota_rl_outputs",
        help="Output directory",
    )
    return parser.parse_args()

args = parse_args()
OUT_DIR = os.path.abspath(args.output_dir)
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
REP_FILE = os.path.join(OUT_DIR, "dakota_rl_wandb_report.pdf")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

dfs: Dict[str, pd.DataFrame] = {}

if args.history_csv:
    print(f"Loading local history from {args.history_csv}...")
    df = pd.read_csv(args.history_csv)
    
    # Map columns to expected format
    # Total Reward
    dfs["total_reward"] = df.copy()
    if "reward/mean" in df.columns:
        dfs["total_reward"]["reward_total"] = df["reward/mean"]
    elif "reward" in df.columns:
        dfs["total_reward"]["reward_total"] = df["reward"]
        
    # Components
    dfs["components"] = df.copy()
    # Map ledger/x to x
    for col in df.columns:
        if "ledger/" in col:
            short_name = col.split("/")[-1]
            # Handle norm vs raw if needed, or just keep as is
            # Report expects "exact", "char", etc.
            if "exact_match_norm" in short_name: dfs["components"]["exact"] = df[col]
            if "char_overlap_norm" in short_name: dfs["components"]["char"] = df[col]
            if "pattern_norm" in short_name: dfs["components"]["pattern"] = df[col]
            if "affix_norm" in short_name: dfs["components"]["affix"] = df[col]
            if "length_penalty_norm" in short_name: dfs["components"]["length"] = df[col]
            
            # Support for new orchestrator logging format
            if "reward/exact_match/mean" in col: dfs["components"]["exact"] = df[col]
            if "reward/char_overlap/mean" in col: dfs["components"]["char"] = df[col]
            if "reward/pattern_match/mean" in col: dfs["components"]["pattern"] = df[col]
            if "reward/affix_match/mean" in col: dfs["components"]["affix"] = df[col]
            if "reward/length_penalty/mean" in col: dfs["components"]["length"] = df[col]

    # Also check for direct column names (from ledger export)
    for col in df.columns:
        if "exact_match_norm" in col: dfs["components"]["exact"] = df[col]
        if "char_overlap_norm" in col: dfs["components"]["char"] = df[col]
        if "pattern_norm" in col: dfs["components"]["pattern"] = df[col]
        if "affix_norm" in col: dfs["components"]["affix"] = df[col]
        if "length_penalty_norm" in col: dfs["components"]["length"] = df[col]
            
    # Entropy
    dfs["entropy"] = df.copy()
    if "entropy/mean" in df.columns:
        dfs["entropy"]["entropy"] = df["entropy/mean"]
        
    # Policy Probs
    dfs["policy_probs"] = df.copy()
    if "inference_probs/mean" in df.columns:
        dfs["policy_probs"]["prob_mean"] = df["inference_probs/mean"]
        
else:
    print("Fetching W&B-derived artifacts from GitHub...")
    for key, url in URLS.items():
        try:
            if url.endswith(".csv"):
                dfs[key] = fetch_csv(url)
                print(f"Loaded CSV: {key} -> {dfs[key].shape}")
            elif url.endswith(".json"):
                jsons[key] = fetch_json(url)
                print(f"Loaded JSON: {key} -> {len(jsons[key])} keys")
        except Exception as e:
            print(f"[WARN] Failed to load {key} from {url}: {e}")


# -----------------------------
# Canonicalize common frames
# -----------------------------

if "total_reward" in dfs:
    tr = normalize_step_index(dfs["total_reward"])
    # Try to detect total reward column
    tr_cols = [c for c in tr.columns if "reward" in c.lower() and "total" in c.lower()]
    if not tr_cols:
        tr_cols = [c for c in tr.columns if "reward" in c.lower()]
    if not tr_cols:
        raise RuntimeError("Could not find a reward column in total_reward_time_series.csv")
    tr.rename(columns={tr_cols[0]: "reward_total"}, inplace=True)
else:
    raise RuntimeError("total_reward_time_series.csv is required")

if "components" in dfs:
    rc = normalize_step_index(dfs["components"])
else:
    rc = None

if "entropy" in dfs:
    ent = normalize_step_index(dfs["entropy"])
    ent_cols = safe_cols(ent, ["entropy", "train_entropy", "policy_entropy"])
    if ent_cols:
        ent.rename(columns={ent_cols[0]: "entropy"}, inplace=True)
    else:
        ent["entropy"] = np.nan
else:
    ent = None

if "policy_probs" in dfs:
    pp = normalize_step_index(dfs["policy_probs"])
    # Try common names
    mcols = safe_cols(pp, ["mean_prob", "mean_probability", "inference_prob_mean", "policy_prob_mean"])
    xcols = safe_cols(pp, ["median_prob", "median_probability", "inference_prob_median", "policy_prob_median"])
    if mcols:
        pp.rename(columns={mcols[0]: "prob_mean"}, inplace=True)
    else:
        pp["prob_mean"] = np.nan
    if xcols:
        pp.rename(columns={xcols[0]: "prob_median"}, inplace=True)
    else:
        pp["prob_median"] = np.nan
else:
    pp = None


# -----------------------------
# Derived Metrics
# -----------------------------

# Reward smoothing & growth
tr["reward_ma"] = rolling_mean(tr["reward_total"], frac=0.02)

# Growth rate (first difference of smoothed reward)
tr["reward_growth"] = tr["reward_ma"].diff()

# Estimate doubling half-life on smoothed reward (if positive range)
def estimate_doubling_steps(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if len(s) < 10 or (s <= 0).all():
        return None
    # Normalize to start > 0
    s = s - s.min() + 1e-6
    # Fit log-linear model vs. step
    x = np.arange(len(s))
    y = np.log(s.values)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if m <= 0:
        return None
    # Doubling steps = ln(2)/m
    return float(np.log(2.0) / m)

doubling_steps = estimate_doubling_steps(tr["reward_ma"])

# Best step by smoothed reward
best_idx = int(tr["reward_ma"].idxmax()) if tr["reward_ma"].notna().any() else int(tr["reward_total"].idxmax())
best_step = int(tr.loc[best_idx, "step"])
best_reward = float(tr.loc[best_idx, "reward_ma"] if not math.isnan(tr.loc[best_idx, "reward_ma"]) else tr.loc[best_idx, "reward_total"])

# Start and end rewards
start_reward = float(tr["reward_total"].iloc[0])
end_reward = float(tr["reward_total"].iloc[-1])

# Early slope over first 10% of steps
n = len(tr)
k = max(5, n // 10)
early_slope = float(np.polyfit(tr["step"].iloc[:k], tr["reward_total"].iloc[:k], deg=1)[0]) if n >= 10 else float("nan")

# Component shares at (a) best_step and (b) final step
def component_snapshot(df: Optional[pd.DataFrame], step: int) -> Dict[str, float]:
    if df is None:
        return {}
    # Heuristics for common component names
    comp_cols = [c for c in df.columns if any(x in c.lower() for x in ["exact", "char", "pattern", "affix", "length"])]
    if not comp_cols:
        return {}
    row = df.iloc[(df["step"] - step).abs().argsort()[:1]].iloc[0]
    out = {}
    for c in comp_cols:
        val = row[c]
        if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float, np.floating)):
            out[c] = float(val)
    return out

comp_best = component_snapshot(rc, best_step)
comp_last = component_snapshot(rc, int(tr["step"].iloc[-1]))

# Save derived tables
pd.DataFrame([{
    "best_step": best_step,
    "best_reward": best_reward,
    "start_reward": start_reward,
    "end_reward": end_reward,
    "early_slope": early_slope,
    "doubling_steps_estimate": doubling_steps if doubling_steps is not None else np.nan,
}]).to_csv(os.path.join(TAB_DIR, "summary_metrics.csv"), index=False)

pd.DataFrame([comp_best]).to_csv(os.path.join(TAB_DIR, "component_snapshot_at_best.csv"), index=False)
pd.DataFrame([comp_last]).to_csv(os.path.join(TAB_DIR, "component_snapshot_at_end.csv"), index=False)


# -----------------------------
# Plots (one figure per chart; no seaborn; no explicit colors)
# -----------------------------

# 1) Total reward vs step with moving average
plt.figure()
plt.plot(tr["step"], tr["reward_total"], linewidth=1.0, label="reward_total")
if tr["reward_ma"].notna().any():
    plt.plot(tr["step"], tr["reward_ma"], linewidth=2.0, label="reward_ma (rolling)")
plt.xlabel("Step")
plt.ylabel("Total Reward")
plt.legend()
save_fig(os.path.join(FIG_DIR, "01_total_reward.png"), title="Total Reward over Training (GRPO)")

# 2) Reward growth rate (dReward/dStep)
plt.figure()
plt.plot(tr["step"], tr["reward_growth"])
plt.xlabel("Step")
plt.ylabel("ΔReward (smoothed)")
save_fig(os.path.join(FIG_DIR, "02_reward_growth.png"), title="Reward Growth Rate")

# 3) Entropy vs step
if ent is not None and "entropy" in ent.columns:
    plt.figure()
    plt.plot(ent["step"], ent["entropy"], linewidth=1.0)
    plt.xlabel("Step")
    plt.ylabel("Policy Entropy")
    save_fig(os.path.join(FIG_DIR, "03_entropy.png"), title="Policy Entropy over Training")

# 4) Policy probabilities (mean/median) vs step
if pp is not None and (("prob_mean" in pp.columns) or ("prob_median" in pp.columns)):
    plt.figure()
    if "prob_mean" in pp.columns:
        plt.plot(pp["step"], pp["prob_mean"], linewidth=1.2, label="mean")
    if "prob_median" in pp.columns:
        plt.plot(pp["step"], pp["prob_median"], linewidth=1.2, label="median")
    plt.xlabel("Step")
    plt.ylabel("Policy Probability")
    plt.legend()
    save_fig(os.path.join(FIG_DIR, "04_policy_probabilities.png"), title="Policy Probabilities over Training")

# 5) Reward components over time (overlayed)
if rc is not None:
    plt.figure()
    comp_cols = [c for c in rc.columns if any(x in c.lower() for x in ["exact", "char", "pattern", "affix", "length"])]
    if comp_cols:
        for c in comp_cols:
            plt.plot(rc["step"], rc[c], linewidth=1.0, label=c)
        plt.xlabel("Step")
        plt.ylabel("Reward Component Value")
        plt.legend()
        save_fig(os.path.join(FIG_DIR, "05_reward_components.png"), title="Reward Components over Time")

# 6) Reward vs Entropy (Pareto view)
if ent is not None and "entropy" in ent.columns:
    join = pd.merge_asof(tr.sort_values("step"), ent.sort_values("step"), on="step")
    plt.figure()
    plt.scatter(join["entropy"], join["reward_total"], s=8)
    plt.xlabel("Policy Entropy")
    plt.ylabel("Total Reward")
    save_fig(os.path.join(FIG_DIR, "06_reward_vs_entropy.png"), title="Reward vs Entropy")

# 7) Component share of total at best step
if comp_best:
    plt.figure()
    names = list(comp_best.keys())
    vals = [comp_best[k] for k in names]
    total = sum(v for v in vals if isinstance(v, (int, float, np.floating)))
    shares = [v / total if total and isinstance(v, (int, float, np.floating)) else 0 for v in vals]
    plt.bar(names, shares)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Share of Total (Best Step)")
    save_fig(os.path.join(FIG_DIR, "07_component_shares_best.png"), title="Component Shares at Best Step")

# 8) Early stopping marker (best smoothed reward)
plt.figure()
plt.plot(tr["step"], tr["reward_total"], linewidth=1.0, label="reward_total")
if tr["reward_ma"].notna().any():
    plt.plot(tr["step"], tr["reward_ma"], linewidth=2.0, label="reward_ma (rolling)")
plt.axvline(best_step, linestyle="--")
plt.xlabel("Step")
plt.ylabel("Total Reward")
plt.legend()
save_fig(os.path.join(FIG_DIR, "08_best_step_marker.png"), title=f"Best Step (≈{best_step}) on Smoothed Reward")

# 9) Cumulative reward (for intuition)
plt.figure()
cum = tr["reward_total"].fillna(0).cumsum()
plt.plot(tr["step"], cum, linewidth=1.2)
plt.xlabel("Step")
plt.ylabel("Cumulative Reward (arbitrary units)")
save_fig(os.path.join(FIG_DIR, "09_cumulative_reward.png"), title="Cumulative Reward (Training Course)")

# 10) Rolling window stability of components (std dev)
if rc is not None and comp_cols:
    plt.figure()
    window = max(10, len(rc)//50)
    comp_std = pd.DataFrame({
        c: rc[c].rolling(window, min_periods=5).std() for c in comp_cols
    })
    for c in comp_cols:
        plt.plot(rc["step"], comp_std[c], linewidth=1.0, label=c)
    plt.xlabel("Step")
    plt.ylabel(f"Std. Dev. (rolling {window})")
    plt.legend()
    save_fig(os.path.join(FIG_DIR, "10_component_stability.png"), title="Rolling Stability of Reward Components")


# -----------------------------
# PDF report assembly
# -----------------------------

from matplotlib.backends.backend_pdf import PdfPages

fig_paths = [
    "01_total_reward.png",
    "02_reward_growth.png",
    "03_entropy.png",
    "04_policy_probabilities.png",
    "05_reward_components.png",
    "06_reward_vs_entropy.png",
    "07_component_shares_best.png",
    "08_best_step_marker.png",
    "09_cumulative_reward.png",
    "10_component_stability.png",
]
fig_paths = [os.path.join(FIG_DIR, f) for f in fig_paths if os.path.exists(os.path.join(FIG_DIR, f))]

with PdfPages(REP_FILE) as pdf:
    # Title page
    plt.figure()
    plt.text(0.5, 0.8, "Dakota RL (GRPO) – W&B Training Report", ha="center", va="center", fontsize=16)
    # Pull a few summary stats if available
    lines = [
        f"Best step (smoothed reward): {best_step}",
        f"Best reward (smoothed): {best_reward:.4f}",
        f"Start → End reward: {start_reward:.4f} → {end_reward:.4f}",
        f"Early slope (first 10%): {early_slope:.6f} per step",
        f"Doubling steps (estimate): {doubling_steps:.1f}" if doubling_steps is not None else "Doubling steps: n/a",
    ]
    y = 0.7
    for ln in lines:
        plt.text(0.5, y, ln, ha="center", va="center", fontsize=11)
        y -= 0.06
    plt.axis("off")
    pdf.savefig(); plt.close()

    # Attach all figures
    for p in fig_paths:
        img = plt.imread(p)
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        pdf.savefig(); plt.close()

# Also write a small JSON manifest of derived metrics
manifest = {
    "best_step": best_step,
    "best_reward_smoothed": best_reward,
    "start_reward": start_reward,
    "end_reward": end_reward,
    "early_slope_first10pct": early_slope,
    "doubling_steps_estimate": doubling_steps,
    "component_names": list(comp_best.keys()) if comp_best else [],
}

with open(os.path.join(OUT_DIR, "derived_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nDone. Outputs written to: {OUT_DIR}")
print(f"- Multi-page PDF: {REP_FILE}")
print(f"- Figures:        {FIG_DIR}")
print(f"- Tables:         {TAB_DIR}")
