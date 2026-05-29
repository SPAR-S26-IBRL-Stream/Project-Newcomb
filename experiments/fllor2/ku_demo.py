"""Reproduce the KU worst-case robustness figure data.

The original exploration lives in ku_demo.ipynb. This script runs the same
two-arm Bernoulli-bandit setup and writes the exact worst-case regret series
used in "Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL
For Worst-Case Robustness".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ibrl.agents import DiscreteBayesianAgent, InfraBayesianAgent
from ibrl.environments import BernoulliBanditEnvironment
from ibrl.exploration import Greedy
from ibrl.infrabayesian import AMeasure, Infradistribution, MultiBernoulliWorldModel
from ibrl.simulators import simulate


RANGE_1 = (0.3, 0.7)
RANGE_2 = (0.4, 0.8)
DEFAULT_NUM_STEPS = 100


class FixedBeliefBayesianAgent(DiscreteBayesianAgent):
    """Non-learning classical agent with one fixed hypothesis per arm."""

    def __init__(self, p1: float, p2: float, *args, **kwargs):
        super().__init__(num_hypotheses=2, *args, **kwargs)
        self.hypotheses = np.array([
            [1.0 - p1, p1],
            [1.0 - p2, p2],
        ])

    def reset(self):
        super().reset()
        self.prior = np.eye(self.num_hypotheses)


def corners() -> list[tuple[float, float]]:
    return [
        (RANGE_1[0], RANGE_2[0]),
        (RANGE_1[1], RANGE_2[0]),
        (RANGE_1[0], RANGE_2[1]),
        (RANGE_1[1], RANGE_2[1]),
    ]


def classical_agent(agent_probs: tuple[float, float]) -> FixedBeliefBayesianAgent:
    return FixedBeliefBayesianAgent(
        p1=agent_probs[0],
        p2=agent_probs[1],
        num_actions=2,
        exploration_strategy=Greedy(),
    )


def ib_agent() -> InfraBayesianAgent:
    wm = MultiBernoulliWorldModel(2)
    measures = [
        AMeasure(wm.make_params(np.array([
            [1.0 - p1, p1],
            [1.0 - p2, p2],
        ])))
        for p1, p2 in corners()
    ]
    return InfraBayesianAgent(
        num_actions=2,
        hypotheses=[Infradistribution(measures, wm)],
    )


def cumulative_regret(result: dict) -> np.ndarray:
    avg_reward = result["average_reward"][0, :]
    return np.cumsum(result["optimal_reward"] - avg_reward)


def worst_case_regret_series(num_steps: int) -> tuple[np.ndarray, dict]:
    """Run the notebook setup and select worst final-regret curves."""
    sim_options = {
        "num_actions": 2,
        "num_steps": num_steps,
        "num_runs": 1,
    }
    env_options = {"num_actions": 2}

    classical_runs = []
    ib_runs = []
    for env_probs in corners():
        for agent_probs in corners():
            result = simulate(
                BernoulliBanditEnvironment(probs=env_probs, **env_options),
                classical_agent(agent_probs),
                sim_options,
            )
            classical_runs.append({
                "env_probs": env_probs,
                "agent_probs": agent_probs,
                "regret": cumulative_regret(result),
            })

        result = simulate(
            BernoulliBanditEnvironment(probs=env_probs, **env_options),
            ib_agent(),
            sim_options,
        )
        ib_runs.append({
            "env_probs": env_probs,
            "regret": cumulative_regret(result),
        })

    classical_worst = max(classical_runs, key=lambda run: run["regret"][-1])
    ib_worst = max(ib_runs, key=lambda run: run["regret"][-1])
    metadata = {
        "classical_worst_env_probs": classical_worst["env_probs"],
        "classical_worst_agent_probs": classical_worst["agent_probs"],
        "ib_worst_env_probs": ib_worst["env_probs"],
    }
    return np.column_stack([
        np.arange(1, num_steps + 1),
        classical_worst["regret"],
        ib_worst["regret"],
    ]), metadata


def write_pgfplots_table(data: np.ndarray, output_path: Path) -> None:
    lines = ["\\pgfplotstableread{", "  t classical  ib"]
    for t, classical, ib in data:
        lines.append(f"{int(t):3d} {classical:6.2f} {ib:6.2f}")
    lines.append("}\\data")
    output_path.write_text("\n".join(lines) + "\n")


def plot_ku_demo(data: np.ndarray, output_path: Path) -> None:
    fig, (env_ax, regret_ax) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    env_ax.fill([0.0, 1.0, 0.0], [0.0, 1.0, 1.0], color="green", alpha=0.35)
    env_ax.fill([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], color="tab:blue", alpha=0.35)
    env_ax.add_patch(
        plt.Rectangle(
            (RANGE_1[0], RANGE_2[0]),
            RANGE_1[1] - RANGE_1[0],
            RANGE_2[1] - RANGE_2[0],
            facecolor="white",
            edgecolor="black",
            alpha=0.85,
        )
    )
    env_ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1.5)
    env_ax.plot(RANGE_1[0], RANGE_2[0], "o", color="red")
    env_ax.text(0.28, 0.85, "$p_2 > p_1$", fontsize=9)
    env_ax.text(0.73, 0.10, "$p_1 > p_2$", fontsize=9)
    env_ax.set_xlim(0.0, 1.0)
    env_ax.set_ylim(0.0, 1.0)
    env_ax.set_xlabel("Probability $p_1$")
    env_ax.set_ylabel("Probability $p_2$")
    env_ax.grid(True, alpha=0.35)

    t = data[:, 0]
    regret_ax.fill_between(
        t,
        0.0,
        (RANGE_2[1] - RANGE_1[0]) * t,
        color="teal",
        alpha=0.35,
        label="Classical allowed",
    )
    regret_ax.fill_between(
        t,
        0.0,
        (RANGE_1[1] - RANGE_2[0]) * t,
        color="orange",
        alpha=0.35,
        label="IB allowed",
    )
    regret_ax.plot(t, data[:, 1], color="teal", linewidth=2.2, label="Classical worst-case")
    regret_ax.plot(t, data[:, 2], color="orange", linewidth=2.2, label="IB worst-case")
    regret_ax.set_xlim(0.0, float(t[-1]))
    regret_ax.set_ylim(0.0, max(55.0, float(data[:, 1].max()) + 3.0))
    regret_ax.set_xlabel("Episode")
    regret_ax.set_ylabel("Cumulative Regret")
    regret_ax.grid(True, alpha=0.35)
    regret_ax.legend(fontsize=7, ncols=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the KU robustness demo and write the figure data used in "
            "Infra-Bayesian Reinforcement Learning Agents Outperform Classical "
            "RL For Worst-Case Robustness."
        )
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="episodes to simulate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/fllor2/ku_demo_outputs"),
        help="directory for the PGFPlots data and preview figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = worst_case_regret_series(args.num_steps)

    table_path = args.output_dir / "ku_regret_data.tex"
    figure_path = args.output_dir / "ku_regret_preview.png"
    write_pgfplots_table(data, table_path)
    plot_ku_demo(data, figure_path)

    print("Worst-case selections:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print(f"Wrote {table_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
