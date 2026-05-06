"""Validate that the IB agent reproduces classical Bayesian bandit behavior.

This script recreates the comparison from ib_validate_classical.ipynb in a
compact paper-friendly figure. A single-a-measure infra-Bayesian agent should
reduce to an ordinary Bayesian posterior-predictive bandit agent.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ibrl.agents import DiscreteBayesianAgent, InfraBayesianAgent
from ibrl.environments import BernoulliBanditEnvironment
from ibrl.infrabayesian import AMeasure, Infradistribution, MultiBernoulliWorldModel
from ibrl.simulators import simulate


OUTPUT_PATH = Path("experiments/fllor2/validate_classical.png")


def make_classical_ib_agent(
    *,
    num_actions: int,
    num_hypotheses: int = 5,
    **kwargs,
) -> InfraBayesianAgent:
    """Build an IB agent equivalent to a classical discrete Bayesian bandit."""
    wm = MultiBernoulliWorldModel(num_arms=num_actions)
    grid = [np.array([1.0 - p, p]) for p in np.linspace(0.0, 1.0, num_hypotheses)]
    params = wm.make_params([grid] * num_actions)
    hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
    return InfraBayesianAgent(
        num_actions=num_actions,
        hypotheses=[hypothesis],
        exploration_prefix=None,
        **kwargs,
    )


def run_validation() -> dict[str, dict[str, dict]]:
    options = {
        "num_actions": 2,
        "num_steps": 501,
        "num_runs": 20,
        "seed": 42,
        "verbose": 0,
    }
    shared = {
        "num_actions": 2,
        "seed": options["seed"] + 0x01234567,
        "verbose": options["verbose"],
        "epsilon": 0.1,
    }
    envs = {
        r"$p=(0.49,0.51)$": BernoulliBanditEnvironment(probs=[0.49, 0.51], **options),
        r"$p=(0.30,0.70)$": BernoulliBanditEnvironment(probs=[0.30, 0.70], **options),
        r"$p=(0.70,0.80)$": BernoulliBanditEnvironment(probs=[0.70, 0.80], **options),
        r"$p=(0.20,0.30)$": BernoulliBanditEnvironment(probs=[0.20, 0.30], **options),
    }

    results: dict[str, dict[str, dict]] = {}
    for env_name, env in envs.items():
        agents = {
            "Bayesian": DiscreteBayesianAgent(**shared, num_hypotheses=5),
            "Infra-Bayesian": make_classical_ib_agent(**shared, num_hypotheses=5),
        }
        results[env_name] = {
            agent_name: simulate(env, agent, options, 0x01234567, 0x89ABCDEF)
            for agent_name, agent in agents.items()
        }
    return results


def plot_validation(results: dict[str, dict[str, dict]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (env_name, agent_results) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        for agent_name, res in agent_results.items():
            linestyle = (0, (6, 5)) if agent_name == "Infra-Bayesian" else "-"
            alpha = 1.0 if agent_name == "Infra-Bayesian" else 0.4
            linewidth = 2.1 if agent_name == "Infra-Bayesian" else 1.6
            zorder = 3 if agent_name == "Infra-Bayesian" else 2

            avg_reward = res["average_reward"][0, :]
            regret = np.cumsum(res["optimal_reward"] - avg_reward)
            axes[0].plot(
                regret,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

            p_arm0 = res["probabilities"][:, :, 0].mean(axis=0)
            axes[1].plot(
                p_arm0,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

    axes[0].set_title("Cumulative Empirical Regret")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cumulative regret")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Action Probability")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(r"$P(\mathrm{arm}\ 0)$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    env_handles = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=env_name)
        for i, env_name in enumerate(results)
    ]
    agent_handles = [
        Line2D([0], [0], color="black", lw=1.8, alpha=0.4, linestyle="-", label="Bayesian"),
        Line2D([0], [0], color="black", lw=2.1, linestyle=(0, (6, 5)), label="Infra-Bayesian"),
    ]
    axes[0].legend(handles=env_handles, title="Environment", fontsize=8, title_fontsize=9)
    axes[1].legend(handles=agent_handles, title="Agent", fontsize=8, title_fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def assert_identical_results(results: dict[str, dict[str, dict]]) -> None:
    for env_name, agent_results in results.items():
        bayes = agent_results["Bayesian"]
        ib = agent_results["Infra-Bayesian"]
        prob_diff = np.abs(bayes["probabilities"] - ib["probabilities"]).max()
        reward_diff = np.abs(bayes["average_reward"] - ib["average_reward"]).max()
        if prob_diff != 0.0 or reward_diff != 0.0:
            raise AssertionError(
                f"{env_name}: expected identical curves, got "
                f"max probability diff={prob_diff}, max reward diff={reward_diff}"
            )


def main() -> None:
    results = run_validation()
    assert_identical_results(results)
    plot_validation(results, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
