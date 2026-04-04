#!/usr/bin/env python3
"""
Canonical W&B and Hugging Face refresh pipeline for Dakota tracking assets.

This script:
1. Pulls full public W&B histories for the maintained Dakota runs.
2. Saves normalized CSV/JSON snapshots under wandb_analysis/.
3. Inventories Hugging Face model repos used by the project.
4. Rebuilds the core visualization set consumed by the README and model card.
5. Repairs the canonical reward-ledger CSV/PNG expected by the narrative docs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from huggingface_hub import hf_hub_download, list_repo_files
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "wandb_analysis"
VIS_DIR = ROOT / "wandb_visualizations"
HF_BASE_URL = "https://huggingface.co/{repo}/resolve/main/{path}"

BACKGROUND = "#07111f"
PANEL = "#0f1e33"
GRID = "#29415d"
TEXT = "#f5f7fb"
MUTED = "#9ab0c8"
ACCENT = "#53d1ff"
REWARD = "#59c3c3"
MORPH = "#9ef01a"
CHAR = "#ffb703"
EXACT = "#fb8500"
PATTERN = "#c77dff"
LOSS = "#ef476f"
ENTROPY = "#ffd166"
KL = "#ff6b6b"
THROUGHPUT = "#80ed99"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    label: str
    role: str
    entity: str = "christian-cooper-us"
    project: str = "dakota-rl-grammar"

    @property
    def run_path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"

    @property
    def url(self) -> str:
        return f"https://wandb.ai/{self.run_path}"


@dataclass(frozen=True)
class RunGroup:
    key: str
    label: str
    trainer: RunSpec
    orchestrator: RunSpec
    output_subdir: str


RUN_GROUPS: dict[str, RunGroup] = {
    "dakota_0_6b_final": RunGroup(
        key="dakota_0_6b_final",
        label="Dakota 0.6B Final (1000-step)",
        trainer=RunSpec("7nikv4vp", "Dakota 0.6B RL Trainer", "trainer"),
        orchestrator=RunSpec("29hn8w98", "Dakota 0.6B RL Orchestrator", "orchestrator"),
        output_subdir=".",
    ),
    "dakota_0_6b_ledger_400": RunGroup(
        key="dakota_0_6b_ledger_400",
        label="Dakota 0.6B Ledger Test (400-step)",
        trainer=RunSpec("yut26kcm", "Dakota 0.6B Ledger Trainer", "trainer"),
        orchestrator=RunSpec("1y33h9zr", "Dakota 0.6B Ledger Orchestrator", "orchestrator"),
        output_subdir="qwen4b",
    ),
}

HF_REPOS: dict[str, str] = {
    "qwen30b_model": "HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890",
    "qwen06b_model": "HarleyCooper/Qwen3-0.6B-Dakota-Grammar-RL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Dakota W&B and HF tracking assets.")
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted(RUN_GROUPS.keys()),
        help="Run group(s) to refresh. Defaults to all maintained groups.",
    )
    parser.add_argument(
        "--skip-hf-inventory",
        action="store_true",
        help="Skip Hugging Face repo inventory.",
    )
    parser.add_argument(
        "--sync-hf-visuals",
        action="store_true",
        help="Download remote visualizations from the 30B HF repo into wandb_visualizations/qwen30b.",
    )
    return parser.parse_args()


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.35, linestyle="--", linewidth=0.8)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)


def ensure_step(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_step" in df.columns:
        df["step"] = df["_step"]
    elif "step" not in df.columns:
        df["step"] = np.arange(len(df))
    df = df.sort_values("step").reset_index(drop=True)
    return df


def rolling(series: pd.Series, window: int = 25) -> pd.Series:
    return series.rolling(window=window, min_periods=max(3, window // 5)).mean()


def non_null_series(df: pd.DataFrame, candidates: Iterable[str]) -> tuple[pd.Series | None, str | None]:
    for key in candidates:
        if key in df.columns:
            series = pd.to_numeric(df[key], errors="coerce")
            if series.notna().any():
                return series, key
    return None, None


def last_value(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def start_value(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = series.dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def fetch_run_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    rows = list(run.scan_history(page_size=1000))
    df = pd.DataFrame(rows)
    return ensure_step(df)


def export_run_bundle(spec: RunSpec) -> dict[str, Any]:
    api = wandb.Api()
    run = api.run(spec.run_path)
    history = fetch_run_history(run)

    run_dir = ANALYSIS_DIR / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(run.summary)
    config = dict(run.config)
    metadata = {
        "run_id": spec.run_id,
        "label": spec.label,
        "role": spec.role,
        "run_path": spec.run_path,
        "url": spec.url,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "row_count": int(len(history)),
        "column_count": int(len(history.columns)),
        "history_columns": history.columns.tolist(),
    }

    history_path = run_dir / f"{spec.run_id}_history.csv"
    history.to_csv(history_path, index=False)
    (run_dir / f"{spec.run_id}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (run_dir / f"{spec.run_id}_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (run_dir / f"{spec.run_id}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "spec": asdict(spec),
        "metadata": metadata,
        "history_path": str(history_path),
        "summary_path": str(run_dir / f"{spec.run_id}_summary.json"),
        "config_path": str(run_dir / f"{spec.run_id}_config.json"),
        "history": history,
    }


def compute_milestones(steps: pd.Series, reward: pd.Series) -> list[dict[str, float]]:
    values = reward.dropna()
    if values.empty:
        return []
    baseline = float(values.iloc[0])
    final = float(values.iloc[-1])
    delta = final - baseline
    if abs(delta) < 1e-12:
        return []

    milestones = []
    for fraction in (0.25, 0.50, 0.75, 0.90):
        threshold = baseline + delta * fraction
        reached = reward[reward >= threshold]
        if reached.empty:
            continue
        index = reached.index[0]
        milestones.append(
            {
                "fraction": fraction,
                "step": float(steps.iloc[index]),
                "reward": float(reward.iloc[index]),
            }
        )
    return milestones


def plot_reward_progression(group: RunGroup, orchestrator: pd.DataFrame, output_dir: Path) -> None:
    steps = orchestrator["step"]
    reward, _ = non_null_series(orchestrator, ["reward/mean", "reward/total"])
    char, _ = non_null_series(orchestrator, ["metrics/char_overlap_reward"])
    affix, _ = non_null_series(orchestrator, ["metrics/affix_reward"])
    exact, _ = non_null_series(orchestrator, ["metrics/exact_match_reward"])
    pattern, _ = non_null_series(orchestrator, ["metrics/pattern_reward"])

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15, 11), facecolor=BACKGROUND)
    for ax in (ax_top, ax_bottom):
        style_axes(ax)

    if reward is not None:
        ax_top.plot(steps, reward, color=REWARD, linewidth=2.2, alpha=0.35)
        ax_top.plot(steps, rolling(reward), color=ACCENT, linewidth=3.0, label="Overall reward (smoothed)")
        milestones = compute_milestones(steps, reward)
        for item in milestones:
            label = f"{int(item['fraction'] * 100)}%"
            ax_top.scatter(item["step"], item["reward"], color=CHAR, s=90, zorder=4, edgecolors=TEXT, linewidth=0.8)
            ax_top.annotate(
                label,
                (item["step"], item["reward"]),
                xytext=(8, 8),
                textcoords="offset points",
                color=TEXT,
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": PANEL, "edgecolor": GRID},
            )
        ax_top.set_title(f"{group.label} Reward Progression", fontsize=16, fontweight="bold")
        ax_top.set_xlabel("Training step")
        ax_top.set_ylabel("Composite reward")
        ax_top.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    components = [
        ("Character", char, CHAR),
        ("Affix", affix, MORPH),
        ("Exact", exact, EXACT),
        ("Pattern", pattern, PATTERN),
    ]
    for label, series, color in components:
        if series is not None:
            ax_bottom.plot(steps, rolling(series), label=label, linewidth=2.3, color=color)

    ax_bottom.set_title("Reward Components", fontsize=15, fontweight="bold")
    ax_bottom.set_xlabel("Training step")
    ax_bottom.set_ylabel("Component score")
    ax_bottom.set_ylim(0.0, 1.05)
    ax_bottom.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    fig.suptitle("Dakota Grammar Gym Reward Storyboard", fontsize=20, color=TEXT, fontweight="bold")
    fig.text(0.02, 0.02, "Source: public W&B histories refreshed via scan_history()", color=MUTED, fontsize=10)
    save_figure(fig, output_dir / "reward_progression.png")


def plot_training_metrics(group: RunGroup, trainer: pd.DataFrame, output_dir: Path) -> None:
    steps = trainer["step"]
    loss, _ = non_null_series(trainer, ["loss/mean"])
    entropy, _ = non_null_series(trainer, ["entropy/mean"])
    kl_masked, _ = non_null_series(trainer, ["masked_mismatch_kl/mean"])
    kl_total, _ = non_null_series(trainer, ["mismatch_kl/mean"])
    probs_inf, _ = non_null_series(trainer, ["inference_probs/mean"])
    probs_train, _ = non_null_series(trainer, ["trainer_probs/mean"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BACKGROUND)
    fig.suptitle(f"{group.label} Trainer Metrics", fontsize=20, color=TEXT, fontweight="bold")
    axes = axes.flatten()
    for ax in axes:
        style_axes(ax)

    if loss is not None:
        axes[0].plot(steps, loss, color=LOSS, linewidth=2.0, alpha=0.35)
        axes[0].plot(steps, rolling(loss), color="#ff7aa2", linewidth=2.8)
        axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[0].set_title("Policy loss")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")

    if entropy is not None:
        axes[1].plot(steps, entropy, color=ENTROPY, linewidth=2.4)
        axes[1].plot(steps, rolling(entropy), color="#ffe29a", linewidth=3.0)
    axes[1].set_title("Entropy")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Entropy")

    if kl_masked is not None:
        axes[2].plot(steps, rolling(kl_masked), color=KL, linewidth=2.4, label="Masked KL")
    if kl_total is not None:
        axes[2].plot(steps, rolling(kl_total), color="#ff9aa2", linewidth=2.2, label="Overall KL")
    axes[2].set_title("Policy divergence")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("KL")
    axes[2].set_yscale("log")
    axes[2].legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    if probs_inf is not None:
        axes[3].plot(steps, rolling(probs_inf), color=ACCENT, linewidth=2.4, label="Inference probs")
    if probs_train is not None:
        axes[3].plot(steps, rolling(probs_train), color="#cdb4db", linewidth=2.2, label="Trainer probs")
    axes[3].set_title("Model confidence")
    axes[3].set_xlabel("Training step")
    axes[3].set_ylabel("Probability")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    save_figure(fig, output_dir / "training_metrics.png")


def plot_performance_metrics(group: RunGroup, trainer: pd.DataFrame, orchestrator: pd.DataFrame, output_dir: Path) -> None:
    trainer_steps = trainer["step"]
    orch_steps = orchestrator["step"]
    trainer_throughput, _ = non_null_series(trainer, ["perf/throughput"])
    orch_throughput, _ = non_null_series(orchestrator, ["perf/throughput"])
    memory, _ = non_null_series(trainer, ["perf/peak_memory"])
    mfu, _ = non_null_series(trainer, ["perf/mfu"])
    total_tokens, _ = non_null_series(orchestrator, ["progress/total_tokens"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BACKGROUND)
    fig.suptitle(f"{group.label} Performance Metrics", fontsize=20, color=TEXT, fontweight="bold")
    axes = axes.flatten()
    for ax in axes:
        style_axes(ax)

    if trainer_throughput is not None:
        axes[0].plot(trainer_steps, rolling(trainer_throughput), color=THROUGHPUT, linewidth=2.6, label="Trainer")
    if orch_throughput is not None:
        axes[0].plot(orch_steps, rolling(orch_throughput), color=ACCENT, linewidth=2.3, label="Orchestrator")
    axes[0].set_title("Throughput")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Tokens / sec")
    axes[0].legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    if memory is not None:
        axes[1].plot(trainer_steps, rolling(memory), color="#ffadad", linewidth=2.5)
    axes[1].set_title("Peak memory")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("GiB")

    if mfu is not None:
        axes[2].plot(trainer_steps, rolling(mfu), color="#90e0ef", linewidth=2.5)
    axes[2].set_title("MFU")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Percent")

    if total_tokens is not None:
        axes[3].plot(orch_steps, total_tokens / 1_000_000, color="#b7efc5", linewidth=2.8)
    axes[3].set_title("Cumulative tokens")
    axes[3].set_xlabel("Training step")
    axes[3].set_ylabel("Millions of tokens")

    save_figure(fig, output_dir / "performance_metrics.png")


def plot_comprehensive_dashboard(group: RunGroup, trainer: pd.DataFrame, orchestrator: pd.DataFrame, output_dir: Path) -> None:
    reward, _ = non_null_series(orchestrator, ["reward/mean"])
    char, _ = non_null_series(orchestrator, ["metrics/char_overlap_reward"])
    affix, _ = non_null_series(orchestrator, ["metrics/affix_reward"])
    throughput, _ = non_null_series(trainer, ["perf/throughput"])
    entropy, _ = non_null_series(trainer, ["entropy/mean"])
    loss, _ = non_null_series(trainer, ["loss/mean"])
    total_tokens, _ = non_null_series(orchestrator, ["progress/total_tokens"])
    reward_steps = orchestrator["step"]
    trainer_steps = trainer["step"]

    fig = plt.figure(figsize=(18, 12), facecolor=BACKGROUND)
    grid = GridSpec(3, 4, figure=fig, height_ratios=[0.55, 1.2, 1.0], hspace=0.32, wspace=0.22)

    fig.suptitle(f"{group.label} Tracking Dashboard", fontsize=24, color=TEXT, fontweight="bold", y=0.98)
    fig.text(0.03, 0.94, "Public W&B histories + HF inventory refresh", color=MUTED, fontsize=12)

    cards = [
        ("Reward", start_value(reward), last_value(reward), REWARD),
        ("Char", start_value(char), last_value(char), CHAR),
        ("Affix", start_value(affix), last_value(affix), MORPH),
        ("Tokens", 0.0, last_value(total_tokens), ACCENT),
    ]

    for idx, (label, start, end, color) in enumerate(cards):
        ax = fig.add_subplot(grid[0, idx])
        ax.set_facecolor(PANEL)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.text(0.05, 0.78, label, color=MUTED, fontsize=12, transform=ax.transAxes)
        if end is None:
            body = "n/a"
        elif label == "Tokens":
            body = f"{end / 1_000_000:.1f}M"
        elif start is None:
            body = f"{end:.3f}"
        else:
            body = f"{start:.3f} -> {end:.3f}"
        ax.text(0.05, 0.42, body, color=TEXT, fontsize=18, fontweight="bold", transform=ax.transAxes)
        if start is not None and end is not None and label != "Tokens":
            delta = end - start
            ax.text(0.05, 0.14, f"delta {delta:+.3f}", color=color, fontsize=12, transform=ax.transAxes)

    ax_reward = fig.add_subplot(grid[1, :2])
    style_axes(ax_reward)
    if reward is not None:
        ax_reward.plot(reward_steps, reward, color=REWARD, alpha=0.28, linewidth=1.6)
        ax_reward.plot(reward_steps, rolling(reward), color=ACCENT, linewidth=3.0, label="Overall reward")
    if char is not None:
        ax_reward.plot(reward_steps, rolling(char), color=CHAR, linewidth=2.2, label="Char overlap")
    if affix is not None:
        ax_reward.plot(reward_steps, rolling(affix), color=MORPH, linewidth=2.2, label="Affix match")
    ax_reward.set_title("Reward narrative", fontsize=16, fontweight="bold")
    ax_reward.set_xlabel("Training step")
    ax_reward.set_ylabel("Score")
    ax_reward.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    ax_loss = fig.add_subplot(grid[1, 2])
    style_axes(ax_loss)
    if loss is not None:
        ax_loss.plot(trainer_steps, rolling(loss), color=LOSS, linewidth=2.6)
        ax_loss.set_yscale("symlog", linthresh=1e-5)
    ax_loss.set_title("Loss", fontsize=15, fontweight="bold")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Loss")

    ax_entropy = fig.add_subplot(grid[1, 3])
    style_axes(ax_entropy)
    if entropy is not None:
        ax_entropy.plot(trainer_steps, rolling(entropy), color=ENTROPY, linewidth=2.6)
    ax_entropy.set_title("Entropy", fontsize=15, fontweight="bold")
    ax_entropy.set_xlabel("Training step")
    ax_entropy.set_ylabel("Entropy")

    ax_perf = fig.add_subplot(grid[2, :2])
    style_axes(ax_perf)
    if throughput is not None:
        ax_perf.plot(trainer_steps, rolling(throughput), color=THROUGHPUT, linewidth=2.8)
    ax_perf.set_title("Trainer throughput", fontsize=16, fontweight="bold")
    ax_perf.set_xlabel("Training step")
    ax_perf.set_ylabel("Tokens / sec")

    ax_bar = fig.add_subplot(grid[2, 2])
    style_axes(ax_bar)
    final_components = [
        ("Reward", last_value(reward), REWARD),
        ("Char", last_value(char), CHAR),
        ("Affix", last_value(affix), MORPH),
    ]
    labels = [item[0] for item in final_components if item[1] is not None]
    values = [item[1] for item in final_components if item[1] is not None]
    colors = [item[2] for item in final_components if item[1] is not None]
    if values:
        ax_bar.bar(labels, values, color=colors)
        ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_title("Final snapshot", fontsize=15, fontweight="bold")
    ax_bar.set_ylabel("Score")

    ax_text = fig.add_subplot(grid[2, 3])
    ax_text.set_facecolor(PANEL)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    for spine in ax_text.spines.values():
        spine.set_color(GRID)
    milestones = compute_milestones(reward_steps, reward) if reward is not None else []
    lines = [
        "Refresh notes",
        f"- trainer rows: {len(trainer)}",
        f"- orchestrator rows: {len(orchestrator)}",
        f"- public W&B read: yes",
        f"- HF inventory: yes",
    ]
    if milestones:
        first_90 = next((item for item in milestones if abs(item["fraction"] - 0.90) < 1e-9), None)
        if first_90 is not None:
            lines.append(f"- 90% improvement by step {int(first_90['step'])}")
    ax_text.text(0.05, 0.92, "\n".join(lines), va="top", color=TEXT, fontsize=12, transform=ax_text.transAxes)

    save_figure(fig, output_dir / "comprehensive_dashboard.png")


def build_reward_ledger_artifacts() -> dict[str, Any]:
    candidates = [
        ANALYSIS_DIR / "reward_ledger_tinker.csv",
        ANALYSIS_DIR / "reward_ledger_qwen4b.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {"status": "missing"}

    source = max(existing, key=lambda path: len(pd.read_csv(path)))
    canonical_csv = ANALYSIS_DIR / "reward_ledger.csv"
    shutil.copyfile(source, canonical_csv)

    df = pd.read_csv(canonical_csv).sort_values("step")
    fig, ax = plt.subplots(figsize=(15, 8), facecolor=BACKGROUND)
    style_axes(ax)
    ax.plot(df["step"], df["composite_pre"], label="Composite pre", color=ACCENT, linewidth=2.2)
    ax.plot(df["step"], df["composite_predicted"], label="Composite predicted", color=PATTERN, linewidth=2.2)
    ax.plot(df["step"], df["reward_scalar"], label="Reward scalar", color=LOSS, linewidth=2.4)
    ax.plot(df["step"], df["difficulty_multiplier"], label="Difficulty", color=MORPH, linewidth=1.7, alpha=0.8)
    ax.plot(df["step"], df["length_penalty_norm"], label="Length multiplier", color=ENTROPY, linewidth=1.7, alpha=0.8)
    ax.set_title("Reward Ledger Reconciliation", fontsize=18, fontweight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    save_figure(fig, ANALYSIS_DIR / "reward_ledger.png")

    return {
        "status": "ok",
        "source_csv": str(source),
        "canonical_csv": str(canonical_csv),
        "canonical_png": str(ANALYSIS_DIR / "reward_ledger.png"),
    }


def inventory_hf_repos(sync_visuals: bool) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for label, repo_id in HF_REPOS.items():
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="model")
            visual_files = [path for path in files if path.startswith("visualizations/")]
            inventory[label] = {
                "repo_id": repo_id,
                "file_count": len(files),
                "visual_files": visual_files,
                "json_files": [path for path in files if path.endswith(".json")],
                "csv_files": [path for path in files if path.endswith(".csv")],
            }

            if sync_visuals and label == "qwen30b_model" and visual_files:
                local_dir = VIS_DIR / "qwen30b"
                local_dir.mkdir(parents=True, exist_ok=True)
                downloaded: list[str] = []
                for remote_path in visual_files:
                    cached = hf_hub_download(repo_id=repo_id, filename=remote_path, repo_type="model")
                    target = local_dir / Path(remote_path).name
                    shutil.copyfile(cached, target)
                    downloaded.append(str(target))
                inventory[label]["synced_visuals"] = downloaded
        except Exception as exc:
            inventory[label] = {"repo_id": repo_id, "error": str(exc)}

    (ANALYSIS_DIR / "hf_repo_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


def write_run_inventory(run_inventory: dict[str, Any], hf_inventory: dict[str, Any], ledger_inventory: dict[str, Any]) -> None:
    payload = {
        "groups": run_inventory,
        "hf_inventory": hf_inventory,
        "reward_ledger": ledger_inventory,
    }
    (ANALYSIS_DIR / "tracking_inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_group_assets(group: RunGroup) -> dict[str, Any]:
    trainer_bundle = export_run_bundle(group.trainer)
    orchestrator_bundle = export_run_bundle(group.orchestrator)

    output_dir = VIS_DIR if group.output_subdir == "." else VIS_DIR / group.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_df = trainer_bundle["history"]
    orchestrator_df = orchestrator_bundle["history"]
    plot_reward_progression(group, orchestrator_df, output_dir)
    plot_training_metrics(group, trainer_df, output_dir)
    plot_performance_metrics(group, trainer_df, orchestrator_df, output_dir)
    plot_comprehensive_dashboard(group, trainer_df, orchestrator_df, output_dir)

    return {
        "group": asdict(group),
        "trainer": trainer_bundle["metadata"],
        "orchestrator": orchestrator_bundle["metadata"],
        "visual_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    selected_groups = args.group or list(RUN_GROUPS.keys())
    run_inventory: dict[str, Any] = {}
    for key in selected_groups:
        group = RUN_GROUPS[key]
        print(f"[refresh] {group.label}")
        run_inventory[key] = build_group_assets(group)

    ledger_inventory = build_reward_ledger_artifacts()
    hf_inventory = {} if args.skip_hf_inventory else inventory_hf_repos(sync_visuals=args.sync_hf_visuals)
    write_run_inventory(run_inventory, hf_inventory, ledger_inventory)

    print("[done] refreshed W&B/HF tracking assets")


if __name__ == "__main__":
    main()
