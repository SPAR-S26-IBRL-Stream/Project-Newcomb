"""Cross-condition comparison plots for trap-bandit results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITION_LABELS = {
    "correct": "Correctly specified prior",
    "severely_misspecified": "Severely misspecified prior",
}

AGENT_LABELS = {
    "ib": "Infra-Bayesian",
    "bayes_greedy": "Greedy Bayesian",
    "bayes_thompson": "Thompson Sampling Bayesian",
}

AGENT_COLORS = {
    "ib": "tab:red",
    "bayes_greedy": "tab:blue",
    "bayes_thompson": "tab:orange",
}


def load_summary(results_dir: Path, condition: str) -> dict:
    return json.loads((results_dir / f"{condition}_summary.json").read_text())


def plot_risky_condition_grid(
    results_dir: Path,
    *,
    output_path: Path,
    conditions: list[str],
    agents: list[str],
) -> None:
    fig, axes = plt.subplots(len(conditions), 2, figsize=(9, 6), sharex=True)
    if len(conditions) == 1:
        axes = np.asarray([axes])

    for row, condition in enumerate(conditions):
        summary = load_summary(results_dir, condition)
        ax_regret = axes[row, 0]
        ax_trapped = axes[row, 1]

        for agent in agents:
            group = summary[agent]["risky"]
            p5, p50, p95 = np.asarray(group["regret_p5_p50_p95"])
            steps = np.arange(len(p50))
            label = AGENT_LABELS.get(agent, agent)
            linestyle = "--" if agent == "ib" else "-"
            color = AGENT_COLORS.get(agent)
            ax_regret.plot(
                steps,
                p50,
                label=label,
                linestyle=linestyle,
                color=color,
            )
            ax_regret.fill_between(steps, p5, p95, alpha=0.12, color=color)

            p5, p50, p95 = np.asarray(group["trapped_p5_p50_p95"])
            ax_trapped.plot(
                steps,
                p50,
                label=label,
                linestyle=linestyle,
                color=color,
            )
            ax_trapped.fill_between(steps, p5, p95, alpha=0.12, color=color)

        ax_regret.set_ylabel(CONDITION_LABELS.get(condition, condition))
        ax_regret.set_title("Cumulative expected regret" if row == 0 else "")
        ax_trapped.set_title("Risky arm pull rate" if row == 0 else "")
        ax_trapped.set_ylim(-0.02, 1.02)

    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["correct", "severely_misspecified"],
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["bayes_greedy", "bayes_thompson", "ib"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.results_dir / "risky_world_prior_comparison_grid.png"
    plot_risky_condition_grid(
        args.results_dir,
        output_path=output,
        conditions=args.conditions,
        agents=args.agents,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
