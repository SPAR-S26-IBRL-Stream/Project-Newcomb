"""Run trap-bandit experiments."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ibrl.agents import InfraBayesianAgent
from ibrl.environments.trap_bandit import TrapBanditEnvironment
from ibrl.exploration import BayesianUCB, Greedy, HypothesisThompsonSampling
from ibrl.infrabayesian.builders.trap_bandit import (
    OUTCOME_CATASTROPHE,
    make_bayesian_hypothesis,
    make_ib_hypothesis,
    make_trap_bandit_hypotheses,
)



REWARD_FUNCTION = np.array([
    [0.0, 1.0, -1000.0],
    [0.0, 1.0, -1000.0],
])


AGENT_LABELS = {
    "ib": "Infra-Bayesian",
    "bayes_greedy": "Bayesian Greedy",
    "bayes_thompson": "Bayesian Thompson Sampling",
    "bayes_ucb": "Bayesian UCB",
}

AGENT_COLORS = {
    "ib": "tab:red",
    "bayes_greedy": "tab:blue",
    "bayes_thompson": "tab:orange",
    "bayes_ucb": "tab:green",
}

DEFAULT_KINDS = [
    "bayes_greedy",
    "bayes_thompson",
    "bayes_ucb",
    "ib",
]


@dataclass(frozen=True)
class ReportCondition:
    name: str
    label: str
    p_risky: float
    p_risky_prior: float


@dataclass(frozen=True)
class ReportFigure:
    name: str
    output_name: str
    conditions: tuple[ReportCondition, ReportCondition]
    column_titles: tuple[str, str]


REPORT_FIGURES = [
    ReportFigure(
        name="mostly_risky",
        output_name="mostly_risky_prior_comparison_grid.png",
        conditions=(
            ReportCondition(
                "mostly_risky_correct",
                "Mostly-risky correctly specified prior",
                p_risky=0.99,
                p_risky_prior=0.99,
            ),
            ReportCondition(
                "mostly_risky_severely_misspecified",
                "Mostly-risky severely misspecified prior",
                p_risky=0.99,
                p_risky_prior=0.01,
            ),
        ),
        column_titles=("Correctly specified prior", "Severely misspecified prior"),
    ),
    ReportFigure(
        name="mostly_safe",
        output_name="mostly_safe_prior_comparison_grid.png",
        conditions=(
            ReportCondition(
                "mostly_safe_correct",
                "Mostly-safe correctly specified prior",
                p_risky=0.01,
                p_risky_prior=0.01,
            ),
            ReportCondition(
                "mostly_safe_severely_pessimistic",
                "Mostly-safe severely pessimistic prior",
                p_risky=0.01,
                p_risky_prior=0.99,
            ),
        ),
        column_titles=("Correctly specified prior", "Severely pessimistic prior"),
    ),
    ReportFigure(
        name="balanced",
        output_name="balanced_prior_comparison_grid.png",
        conditions=(
            ReportCondition(
                "balanced_correct",
                "Balanced correctly specified prior",
                p_risky=0.5,
                p_risky_prior=0.5,
            ),
            ReportCondition(
                "balanced_severely_pessimistic",
                "Balanced severely pessimistic prior",
                p_risky=0.5,
                p_risky_prior=0.99,
            ),
        ),
        column_titles=("Correctly specified prior", "Severely pessimistic prior"),
    ),
]


@dataclass
class TrapBanditConfig:
    num_worlds: int = 200
    num_steps: int = 100
    p_cat: float = 0.01
    seed: int = 123
    p_risky: float = 0.99
    p_low: float = 0.3
    p_high: float = 0.7


def report_conditions() -> dict[str, ReportCondition]:
    return {
        condition.name: condition
        for figure in REPORT_FIGURES
        for condition in figure.conditions
    }


def p_pair_hypothesis_kwargs(config: TrapBanditConfig) -> dict:
    return {
        "p_pairs": [
            (config.p_low, config.p_high),
            (config.p_high, config.p_low),
        ],
        "p_pair_weights": np.array([0.5, 0.5]),
    }


def make_agent(
    kind: str,
    p_risky_prior: float,
    config: TrapBanditConfig,
    *,
    safe=None,
    risky=None,
):
    if safe is None or risky is None:
        _wm, safe, risky = make_trap_bandit_hypotheses(
            p_cat=config.p_cat,
            **p_pair_hypothesis_kwargs(config),
        )
    if kind == "ib":
        hypothesis = make_ib_hypothesis(safe, risky)
        strategy = None
    else:
        hypothesis = make_bayesian_hypothesis(safe, risky, p_risky=p_risky_prior)
        if kind == "bayes_greedy":
            strategy = Greedy()
        elif kind == "bayes_thompson":
            strategy = HypothesisThompsonSampling()
        elif kind == "bayes_ucb":
            strategy = BayesianUCB(quantile=0.95)
        else:
            raise ValueError(f"unknown agent kind {kind}")

    return InfraBayesianAgent(
        num_actions=2,
        hypotheses=[hypothesis],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        policy_discretisation=0,
        exploration_strategy=strategy,
    )


def sample_world(rng: np.random.Generator, config: TrapBanditConfig):
    risky = bool(rng.random() < config.p_risky)
    if rng.random() < 0.5:
        p1, p2 = config.p_low, config.p_high
    else:
        p1, p2 = config.p_high, config.p_low
    return {"p_risky": config.p_risky, "risky": risky, "p1": p1, "p2": p2}


def run_agent_on_world(
    agent: InfraBayesianAgent,
    world: dict,
    config: TrapBanditConfig,
    *,
    seed: int,
) -> dict:
    env = TrapBanditEnvironment(
        p1=world["p1"],
        p2=world["p2"],
        risky=world["risky"],
        p_cat=config.p_cat,
        seed=seed,
    )
    env.reset()
    # BaseAgent.reset() re-seeds from self.seed, so assign the per-world seed first.
    agent.seed = seed + 10_000
    agent.reset()

    rewards = np.zeros(config.num_steps)
    actions = np.zeros(config.num_steps, dtype=np.int64)
    catastrophes = np.zeros(config.num_steps, dtype=bool)
    trapped_pulls = np.zeros(config.num_steps, dtype=bool)
    expected_regret = np.zeros(config.num_steps)

    optimal = env.get_optimal_reward()
    for step in range(config.num_steps):
        probs = agent.get_probabilities()
        action = int(agent.random.choice(agent.num_actions, p=probs))
        outcome = env.step(probs, action)
        agent.update(probs, action, outcome)

        rewards[step] = outcome.reward
        actions[step] = action
        catastrophes[step] = outcome.observation == OUTCOME_CATASTROPHE
        trapped_pulls[step] = action == env.trapped_arm
        expected_regret[step] = optimal - env.expected_value(action)

    return {
        "rewards": rewards,
        "actions": actions,
        "catastrophes": catastrophes,
        "trapped_pulls": trapped_pulls,
        "expected_regret": expected_regret,
        "cumulative_expected_regret": np.cumsum(expected_regret),
        "risky": world["risky"],
    }


def _stack_runs(runs: list[dict]) -> dict:
    return {
        "rewards": np.stack([run["rewards"] for run in runs]),
        "actions": np.stack([run["actions"] for run in runs]),
        "catastrophes": np.stack([run["catastrophes"] for run in runs]),
        "trapped_pulls": np.stack([run["trapped_pulls"] for run in runs]),
        "expected_regret": np.stack([run["expected_regret"] for run in runs]),
        "cumulative_expected_regret": np.stack([
            run["cumulative_expected_regret"] for run in runs
        ]),
        "risky": np.array([run["risky"] for run in runs], dtype=bool),
    }


def run_condition(
    p_risky_prior: float,
    config: TrapBanditConfig,
    *,
    kinds: list[str] | None = None,
    p_risky: float | None = None,
) -> dict:
    if p_risky is not None:
        config = TrapBanditConfig(**{**config.__dict__, "p_risky": p_risky})
    rng = np.random.default_rng(config.seed)
    if kinds is None:
        kinds = DEFAULT_KINDS
    results = {kind: [] for kind in kinds}
    _wm, safe, risky = make_trap_bandit_hypotheses(
        p_cat=config.p_cat,
        **p_pair_hypothesis_kwargs(config),
    )

    for world_idx in range(config.num_worlds):
        world = sample_world(rng, config)
        for kind in kinds:
            agent = make_agent(kind, p_risky_prior, config, safe=safe, risky=risky)
            results[kind].append(
                run_agent_on_world(
                    agent,
                    world,
                    config,
                    seed=config.seed + world_idx,
                )
            )
    return results


def bootstrap_final_regret_percentile_cis(
    results: dict[str, dict],
    *,
    num_bootstrap: int = 2000,
    seed: int = 0,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
    ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Bootstrap CIs for final cumulative expected-regret percentiles.

    Resamples worlds with replacement. Each bootstrap replicate recomputes the
    requested percentile over final cumulative expected regret.
    """
    rng = np.random.default_rng(seed)
    output = {}
    for kind, stacked in results.items():
        final_regret = stacked["cumulative_expected_regret"][:, -1]
        num_worlds = len(final_regret)
        draws = np.empty((num_bootstrap, len(percentiles)))
        for draw_idx in range(num_bootstrap):
            indices = rng.integers(0, num_worlds, size=num_worlds)
            draws[draw_idx] = np.percentile(final_regret[indices], percentiles)
        point = np.percentile(final_regret, percentiles)
        bounds = np.percentile(draws, ci, axis=0).T
        output[kind] = {
            "percentiles": list(percentiles),
            "point": point,
            "ci": bounds,
        }
    return output


def summarize(results: dict) -> dict:
    summary = {}
    for kind, runs in results.items():
        risky = np.array([run["risky"] for run in runs], dtype=bool)
        summary[kind] = {
            "all": summarize_group(runs, np.ones(len(runs), dtype=bool)),
            "safe": summarize_group(runs, ~risky),
            "risky": summarize_group(runs, risky),
            "catastrophe_rate": float(np.mean([run["catastrophes"].any() for run in runs])),
        }
        summary[kind]["regret_p5_p50_p95"] = summary[kind]["all"]["regret_p5_p50_p95"]
        summary[kind]["trapped_p5_p50_p95"] = summary[kind]["all"]["trapped_p5_p50_p95"]
    return summary


def summarize_group(runs: list[dict], mask: np.ndarray) -> dict:
    selected = [run for run, include in zip(runs, mask) if include]
    if not selected:
        if not runs:
            raise ValueError("summarize_group requires at least one run")
        num_steps = len(runs[0]["cumulative_expected_regret"])
        nan = np.full((3, num_steps), np.nan)
        return {"regret_p5_p50_p95": nan, "trapped_p5_p50_p95": nan}
    regret = np.stack([run["cumulative_expected_regret"] for run in selected])
    trapped = np.stack([
        np.cumsum(run["trapped_pulls"]) / (np.arange(len(run["trapped_pulls"])) + 1)
        for run in selected
    ])
    return {
        "regret_p5_p50_p95": np.percentile(regret, [5, 50, 95], axis=0),
        "trapped_p5_p50_p95": np.percentile(trapped, [5, 50, 95], axis=0),
    }


def config_payload(config: TrapBanditConfig, kinds: list[str] | None) -> dict:
    return {
        "num_worlds": config.num_worlds,
        "num_steps": config.num_steps,
        "p_cat": config.p_cat,
        "seed": config.seed,
        "p_low": config.p_low,
        "p_high": config.p_high,
        "kinds": kinds,
        "figures": [
            {
                "name": figure.name,
                "output_name": figure.output_name,
                "conditions": [condition.__dict__ for condition in figure.conditions],
            }
            for figure in REPORT_FIGURES
        ],
    }


def save_summary(summary: dict, output_path: Path) -> None:
    serializable = {}
    for kind, values in summary.items():
        serializable[kind] = {
            "catastrophe_rate": values["catastrophe_rate"],
        }
        for group_key in ["all", "safe", "risky"]:
            serializable[kind][group_key] = {
                "regret_p5_p50_p95": values[group_key]["regret_p5_p50_p95"].tolist(),
                "trapped_p5_p50_p95": values[group_key]["trapped_p5_p50_p95"].tolist(),
            }
    output_path.write_text(json.dumps(serializable))


def save_bootstrap_summary(bootstrap: dict, output_path: Path) -> None:
    serializable = {}
    for kind, values in bootstrap.items():
        serializable[kind] = {
            "percentiles": values["percentiles"],
            "point": values["point"].tolist(),
            "ci": values["ci"].tolist(),
        }
    output_path.write_text(json.dumps(serializable))


def load_summary(results_dir: Path, condition: str) -> dict:
    return json.loads((results_dir / f"{condition}_summary.json").read_text())


def plot_prior_comparison_grid(
    results_dir: Path,
    *,
    output_path: Path,
    conditions: tuple[ReportCondition, ReportCondition],
    agents: list[str],
    column_titles: tuple[str, str],
) -> None:
    fig, axes = plt.subplots(2, len(conditions), figsize=(9, 6), sharex=True)

    for col, condition in enumerate(conditions):
        summary = load_summary(results_dir, condition.name)
        ax_regret = axes[0, col]
        ax_trapped = axes[1, col]

        for agent in agents:
            group = summary[agent]["all"]
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

        ax_regret.set_title(column_titles[col])
        ax_trapped.set_ylim(-0.02, 1.02)

    axes[0, 0].set_ylabel("Cumulative expected regret")
    axes[1, 0].set_ylabel("Risky arm pull rate")
    for ax in axes[-1, :]:
        ax.set_xlabel("step")
    axes[0, -1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def ci_cell(point: float, ci: list[float]) -> str:
    lo, hi = ci
    return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


def markdown_agent_label(agent_label: str) -> str:
    return agent_label.replace(r"\_", "_")


def build_markdown_table(results_dir: Path) -> str:
    lines = [
        "| figure | DGP p_risky | Bayesian p_risky_prior | agent | catastrophe rate | p5, 95% CI | p50, 95% CI | p95, 95% CI |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for figure in REPORT_FIGURES:
        for condition in figure.conditions:
            summary = json.loads((results_dir / f"{condition.name}_summary.json").read_text())
            bootstrap = json.loads((results_dir / f"{condition.name}_bootstrap_summary.json").read_text())
            for agent in DEFAULT_KINDS:
                bayes_prior = "n/a" if agent == "ib" else f"{condition.p_risky_prior:g}"
                cat = summary[agent]["catastrophe_rate"]
                points = bootstrap[agent]["point"]
                cis = bootstrap[agent]["ci"]
                p5 = ci_cell(points[0], cis[0])
                p50 = ci_cell(points[1], cis[1])
                p95 = ci_cell(points[2], cis[2])
                lines.append(
                    f"| {figure.name} | {condition.p_risky:g} | {bayes_prior} | "
                    f"{markdown_agent_label(agent)} | {cat:.3f} | {p5} | {p50} | {p95} |"
                )
    lines.append("")
    return "\n".join(lines)


def run_and_save(
    *,
    config: TrapBanditConfig,
    output_dir: Path,
    kinds: list[str] | None = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if kinds is None:
        kinds = DEFAULT_KINDS
    conditions = report_conditions()
    config_path = output_dir / "config.json"

    summaries = {}
    for condition in conditions.values():
        condition_results = run_condition(
            condition.p_risky_prior,
            config,
            kinds=kinds,
            p_risky=condition.p_risky,
        )
        summary = summarize(condition_results)
        stacked = {
            kind: _stack_runs(runs)
            for kind, runs in condition_results.items()
        }
        summaries[condition.name] = summary
        save_summary(summary, output_dir / f"{condition.name}_summary.json")
        if bootstrap_samples > 0:
            bootstrap = bootstrap_final_regret_percentile_cis(
                stacked,
                num_bootstrap=bootstrap_samples,
                seed=bootstrap_seed,
            )
            save_bootstrap_summary(
                bootstrap,
                output_dir / f"{condition.name}_bootstrap_summary.json",
            )

    config_path.write_text(json.dumps(config_payload(config, kinds), indent=2))
    for figure in REPORT_FIGURES:
        plot_prior_comparison_grid(
            output_dir,
            output_path=output_dir / figure.output_name,
            conditions=figure.conditions,
            agents=kinds,
            column_titles=figure.column_titles,
        )
    if bootstrap_samples > 0:
        (output_dir / "results_table.md").write_text(build_markdown_table(output_dir))
    return summaries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--p-cat", type=float, default=0.01)
    parser.add_argument("--p-low", type=float, default=0.3)
    parser.add_argument("--p-high", type=float, default=0.7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/alaro/trap_bandit/results_report_200_pcat001"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=DEFAULT_KINDS,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrapBanditConfig(
        num_worlds=args.num_worlds,
        num_steps=args.num_steps,
        p_cat=args.p_cat,
        seed=args.seed,
        p_low=args.p_low,
        p_high=args.p_high,
    )
    summaries = run_and_save(
        config=cfg,
        output_dir=args.output_dir,
        kinds=args.kinds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for condition, summary in summaries.items():
        print(condition)
        for kind, values in summary.items():
            print(kind, values["catastrophe_rate"], values["regret_p5_p50_p95"][:, -1])
