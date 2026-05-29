"""Reproduce the Newcomb predictor-accuracy figure data.

The original exploration lives in newcomb.ipynb. This script runs the same
Newcomb setup and writes the reward and action-rate sweep used in
"Infra-Bayesian Reinforcement Learning Agents Outperform Classical RL For
Worst-Case Robustness".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ibrl.agents import InfraBayesianAgent
from ibrl.environments import NewcombEnvironment
from ibrl.infrabayesian import AMeasure, Infradistribution, NewcombWorldModel
from ibrl.simulators import simulate


DEFAULT_NUM_STEPS = 1000
DEFAULT_ACCURACIES = np.linspace(0.5, 1.0, 51)


def make_newcomb_environment(*, predictor_accuracy: float, seed: int) -> NewcombEnvironment:
    return NewcombEnvironment(
        boxA=1,
        boxB=10,
        predictor_accuracy=predictor_accuracy,
        seed=seed,
    )


def make_agent(env: NewcombEnvironment, *, predictor_accuracy: float, seed: int) -> InfraBayesianAgent:
    wm = NewcombWorldModel(env.reward_table)
    hypothesis = Infradistribution([AMeasure(wm.make_params(predictor_accuracy))], wm)
    return InfraBayesianAgent(
        num_actions=2,
        hypotheses=[hypothesis],
        reward_function=wm.agent_reward_matrix(),
        policy_discretisation=5,
        seed=seed,
    )


def optimal_action_rate(env: NewcombEnvironment) -> float:
    """Return the optimal probability of action 1 from BaseNewcombLikeEnvironment."""
    (a, b), (c, d) = env.reward_table.tolist()
    acc = env.predictor_accuracy
    A = acc * (a - c) + c
    C = (2 * acc - 1) * (a + d - b - c)
    D = acc * (d - b) + b
    B = D - A - C
    candidates = [(A, 0.0), (D, 1.0)]
    if C != 0:
        p = -B / (2 * C)
        if C < 0 and 0 <= p <= 1:
            candidates.append((A - B**2 / (4 * C), p))
    return max(candidates)[1]


def run_accuracy_sweep(
    accuracies: np.ndarray = DEFAULT_ACCURACIES,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
) -> np.ndarray:
    rows = []
    for i, accuracy in enumerate(accuracies):
        env = make_newcomb_environment(
            predictor_accuracy=float(accuracy),
            seed=0x89ABCDEF + i,
        )
        agent = make_agent(
            env,
            predictor_accuracy=float(accuracy),
            seed=0x01234567 + i,
        )
        result = simulate(env, agent, {"num_steps": num_steps})
        rewards = result["rewards"][0]
        rows.append((
            accuracy,
            result["optimal_reward"],
            result["average_reward"][0].mean(),
            result["probabilities"][0, :, 1].mean(),
            optimal_action_rate(env),
            rewards.std(ddof=0) / np.sqrt(len(rewards)),
        ))
    return np.array(rows)


def write_pgfplots_table(data: np.ndarray, output_path: Path) -> None:
    lines = [
        "\\pgfplotstableread{",
        "acc reward_opt reward_sim rate_sim rate_opt reward_sim_err",
    ]
    for acc, reward_opt, reward_sim, rate_sim, rate_opt, reward_sim_err in data:
        lines.append(
            f"{acc:.2f} {reward_opt:5.2f} {reward_sim:5.2f} "
            f"{rate_sim:.1f} {rate_opt:.1f} {reward_sim_err:.2f}"
        )
    lines.append("}\\data")
    output_path.write_text("\n".join(lines) + "\n")


def plot_accuracy_sweep(data: np.ndarray, output_path: Path) -> None:
    fig, (reward_ax, rate_ax) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    acc = data[:, 0]
    reward_opt = data[:, 1]
    reward_sim = data[:, 2]
    rate_sim = data[:, 3]
    rate_opt = data[:, 4]
    reward_sim_err = data[:, 5]

    reward_ax.plot(acc, reward_opt, color="teal", linewidth=2.2, label="Optimal")
    reward_ax.fill_between(
        acc,
        reward_sim - reward_sim_err,
        reward_sim + reward_sim_err,
        color="orange",
        alpha=0.35,
    )
    reward_ax.plot(acc, reward_sim, color="orange", linewidth=2.2, label="Simulated")
    reward_ax.set_xlim(0.5, 1.0)
    reward_ax.set_ylim(5.0, 10.0)
    reward_ax.set_xlabel("Predictor accuracy $\\alpha$")
    reward_ax.set_ylabel("Average reward")
    reward_ax.grid(True, alpha=0.35)

    rate_ax.plot(acc, 1 - rate_opt, color="teal", linewidth=2.2, label="Optimal")
    rate_ax.plot(acc, 1 - rate_sim, color="orange", linewidth=2.2, label="Simulated")
    rate_ax.set_xlim(0.5, 1.0)
    rate_ax.set_ylim(-0.02, 1.02)
    rate_ax.set_xlabel("Predictor accuracy $\\alpha$")
    rate_ax.set_ylabel("One-boxing rate")
    rate_ax.grid(True, alpha=0.35)
    rate_ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Newcomb predictor-accuracy sweep and write the figure data "
            "used in Infra-Bayesian Reinforcement Learning Agents Outperform "
            "Classical RL For Worst-Case Robustness."
        )
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="episodes to simulate for each predictor accuracy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/ib_architecture/newcomb_accuracy_outputs"),
        help="directory for the PGFPlots data and preview figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = run_accuracy_sweep(num_steps=args.num_steps)

    table_path = args.output_dir / "newcomb_accuracy_data.tex"
    figure_path = args.output_dir / "newcomb_accuracy_preview.png"
    write_pgfplots_table(data, table_path)
    plot_accuracy_sweep(data, figure_path)

    print(f"Wrote {table_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
