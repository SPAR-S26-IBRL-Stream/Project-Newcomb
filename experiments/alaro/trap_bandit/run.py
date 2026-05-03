"""Run trap-bandit experiments."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ibrl.agents import Greedy, InfraBayesianAgent, ThompsonSampling, UCB
from ibrl.infrabayesian.world_models.joint_bandit_world_model import OUTCOME_CATASTROPHE

from .environment import TrapBanditEnvironment
from .hypotheses import (
    make_bayesian_hypothesis,
    make_ib_hypothesis,
    make_trap_bandit_hypotheses,
)
from .plot import plot_condition_grid, plot_log_regret, plot_percentile_band


REWARD_FUNCTION = np.array([
    [0.0, 1.0, -1000.0],
    [0.0, 1.0, -1000.0],
])


@dataclass
class TrapBanditConfig:
    num_worlds: int = 100
    num_steps: int = 1000
    num_grid: int = 19
    p_cat: float = 0.01
    seed: int = 123
    alpha_dgp: tuple[float, float] = (2.0, 2.0)
    p_beta: tuple[float, float] = (2.0, 2.0)


def make_agent(
    kind: str,
    alpha_prior: tuple[float, float],
    config: TrapBanditConfig,
    *,
    safe=None,
    risky=None,
):
    if safe is None or risky is None:
        _wm, safe, risky = make_trap_bandit_hypotheses(
            num_grid=config.num_grid,
            p_cat=config.p_cat,
            p_beta=config.p_beta,
        )
    if kind == "ib":
        hypothesis = make_ib_hypothesis(safe, risky)
        strategy = Greedy()
    else:
        hypothesis = make_bayesian_hypothesis(safe, risky, alpha_beta=alpha_prior)
        if kind == "bayes_greedy":
            strategy = Greedy()
        elif kind == "bayes_thompson":
            strategy = ThompsonSampling()
        elif kind == "bayes_ucb":
            strategy = UCB(c=2.0)
        else:
            raise ValueError(f"unknown agent kind {kind}")

    return InfraBayesianAgent(
        num_actions=2,
        hypotheses=[hypothesis],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        policy_discretisation=0,
        exploration_prefix=0,
        exploration_strategy=strategy,
        epsilon=0.0,
    )


def sample_world(rng: np.random.Generator, config: TrapBanditConfig):
    alpha = rng.beta(*config.alpha_dgp)
    risky = bool(rng.random() < alpha)
    p1 = min(float(rng.beta(*config.p_beta)), 1.0 - config.p_cat)
    p2 = min(float(rng.beta(*config.p_beta)), 1.0 - config.p_cat)
    return {"alpha": alpha, "risky": risky, "p1": p1, "p2": p2}


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


def _unstack_runs(stacked: dict) -> list[dict]:
    return [
        {
            "rewards": stacked["rewards"][i],
            "actions": stacked["actions"][i],
            "catastrophes": stacked["catastrophes"][i],
            "trapped_pulls": stacked["trapped_pulls"][i],
            "expected_regret": stacked["expected_regret"][i],
            "cumulative_expected_regret": stacked["cumulative_expected_regret"][i],
            "risky": bool(stacked["risky"][i]),
        }
        for i in range(stacked["rewards"].shape[0])
    ]


def run_condition(
    alpha_prior: tuple[float, float],
    config: TrapBanditConfig,
    *,
    kinds: list[str] | None = None,
    alpha_dgp: tuple[float, float] | None = None,
) -> dict:
    if alpha_dgp is not None:
        config = replace(config, alpha_dgp=alpha_dgp)
    rng = np.random.default_rng(config.seed)
    if kinds is None:
        kinds = ["bayes_greedy", "bayes_thompson", "bayes_ucb", "ib"]
    results = {kind: [] for kind in kinds}
    _wm, safe, risky = make_trap_bandit_hypotheses(
        num_grid=config.num_grid,
        p_cat=config.p_cat,
        p_beta=config.p_beta,
    )

    for world_idx in range(config.num_worlds):
        world = sample_world(rng, config)
        for kind in kinds:
            agent = make_agent(kind, alpha_prior, config, safe=safe, risky=risky)
            results[kind].append(
                run_agent_on_world(agent, world, config, seed=config.seed + world_idx)
            )
    return results


def summarize_stacked(results: dict[str, dict]) -> dict:
    return summarize({kind: _unstack_runs(stacked) for kind, stacked in results.items()})


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
        "num_grid": config.num_grid,
        "p_cat": config.p_cat,
        "seed": config.seed,
        "alpha_dgp": list(config.alpha_dgp),
        "p_beta": list(config.p_beta),
        "kinds": kinds,
        "conditions": {
            "correct": {"prior": [2.0, 2.0], "dgp": [2.0, 2.0]},
            "misspecified": {"prior": [2.0, 5.0], "dgp": [2.0, 2.0]},
            "severely_misspecified": {"prior": [1.0, 99.0], "dgp": [2.0, 2.0]},
            "mostly_safe_correct": {"prior": [1.0, 99.0], "dgp": [1.0, 99.0]},
        },
    }


def config_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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


def raw_cache_path(output_dir: Path, condition_name: str) -> Path:
    return output_dir / f"{condition_name}_raw.npz"


def save_raw_results(output_path: Path, condition_results: dict[str, list[dict]]) -> None:
    payload = {}
    for kind, runs in condition_results.items():
        stacked = _stack_runs(runs)
        for key, value in stacked.items():
            payload[f"{kind}__{key}"] = value
    np.savez_compressed(output_path, **payload)


def load_raw_results(input_path: Path, kinds: list[str]) -> dict[str, dict]:
    data = np.load(input_path)
    results = {}
    keys = [
        "rewards",
        "actions",
        "catastrophes",
        "trapped_pulls",
        "expected_regret",
        "cumulative_expected_regret",
        "risky",
    ]
    for kind in kinds:
        results[kind] = {key: data[f"{kind}__{key}"] for key in keys}
    return results


def run_and_save(
    *,
    config: TrapBanditConfig,
    output_dir: Path,
    kinds: list[str] | None = None,
    force: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if kinds is None:
        kinds = ["bayes_greedy", "bayes_thompson", "bayes_ucb", "ib"]
    conditions = {
        "correct": {"prior": (2.0, 2.0), "dgp": config.alpha_dgp},
        "misspecified": {"prior": (2.0, 5.0), "dgp": config.alpha_dgp},
        "severely_misspecified": {"prior": (1.0, 99.0), "dgp": config.alpha_dgp},
        "mostly_safe_correct": {"prior": (1.0, 99.0), "dgp": (1.0, 99.0)},
    }
    payload = config_payload(config, kinds)
    payload["config_hash"] = config_hash(payload)
    config_path = output_dir / "config.json"
    cache_matches = False
    if config_path.exists():
        try:
            cache_matches = json.loads(config_path.read_text()) == payload
        except json.JSONDecodeError:
            cache_matches = False

    summaries = {}
    for name, condition in conditions.items():
        cache_path = raw_cache_path(output_dir, name)
        if cache_matches and cache_path.exists() and not force:
            stacked = load_raw_results(cache_path, kinds)
            summary = summarize_stacked(stacked)
        else:
            condition_results = run_condition(
                condition["prior"],
                config,
                kinds=kinds,
                alpha_dgp=condition["dgp"],
            )
            save_raw_results(cache_path, condition_results)
            summary = summarize(condition_results)
        summaries[name] = summary
        save_summary(summary, output_dir / f"{name}_summary.json")
        plot_log_regret(
            summary,
            f"Trap bandit log cumulative regret ({name})",
            str(output_dir / f"{name}_regret.png"),
        )
        plot_percentile_band(
            summary,
            "trapped_p5_p50_p95",
            f"Trap-bandit trapped-arm pull rate ({name})",
            str(output_dir / f"{name}_trapped_arm.png"),
        )
        plot_condition_grid(
            summary,
            f"Trap bandit ({name})",
            str(output_dir / f"{name}_grid.png"),
        )

    config_path.write_text(json.dumps(payload, indent=2))
    return summaries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--num-grid", type=int, default=9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/alaro/trap_bandit/results"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=["bayes_greedy", "bayes_thompson", "bayes_ucb", "ib"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrapBanditConfig(
        num_worlds=args.num_worlds,
        num_steps=args.num_steps,
        num_grid=args.num_grid,
        seed=args.seed,
    )
    summaries = run_and_save(
        config=cfg,
        output_dir=args.output_dir,
        kinds=args.kinds,
        force=args.force,
    )
    for condition, summary in summaries.items():
        print(condition)
        for kind, values in summary.items():
            print(kind, values["catastrophe_rate"], values["regret_p5_p50_p95"][:, -1])
