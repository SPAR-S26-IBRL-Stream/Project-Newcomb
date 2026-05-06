"""Plot helpers for trap-bandit summaries."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def line_style(kind: str) -> str:
    return "--" if kind == "ib" else "-"


def plot_percentile_band(summary: dict, metric: str, title: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, values in summary.items():
        p5, p50, p95 = values[metric]
        steps = np.arange(len(p50))
        ax.plot(steps, p50, label=kind, linestyle=line_style(kind))
        ax.fill_between(steps, p5, p95, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_log_regret(summary: dict, title: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, values in summary.items():
        p5, p50, p95 = values["regret_p5_p50_p95"]
        steps = np.arange(len(p50))
        ax.plot(steps, np.log1p(p50), label=kind, linestyle=line_style(kind))
        ax.fill_between(steps, np.log1p(p5), np.log1p(p95), alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("log(1 + cumulative expected regret)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_condition_grid(summary: dict, title: str, output_path: str) -> None:
    groups = [
        ("all", "Overall"),
        ("safe", "Safe worlds"),
        ("risky", "Risky worlds"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for row, (group_key, group_label) in enumerate(groups):
        ax_regret = axes[row, 0]
        ax_trapped = axes[row, 1]
        for kind, values in summary.items():
            group = values[group_key]
            p5, p50, p95 = group["regret_p5_p50_p95"]
            steps = np.arange(len(p50))
            ax_regret.plot(steps, np.log1p(p50), label=kind, linestyle=line_style(kind))
            ax_regret.fill_between(steps, np.log1p(p5), np.log1p(p95), alpha=0.12)

            p5, p50, p95 = group["trapped_p5_p50_p95"]
            ax_trapped.plot(steps, p50, label=kind, linestyle=line_style(kind))
            ax_trapped.fill_between(steps, p5, p95, alpha=0.12)

        ax_regret.set_ylabel(group_label)
        ax_regret.set_title("log(1 + cumulative regret)" if row == 0 else "")
        ax_trapped.set_title("argmax(p1,p2) pull rate" if row == 0 else "")
        ax_trapped.set_ylim(-0.02, 1.02)

    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
