"""Iterated imperfect Newcomb with a non-stationary predictor accuracy.

Tests Option B from the experimenter's strategic analysis: does IB-KU
(multiple admissible reward-matrix beliefs) beat single-belief Bayesian
agents on a policy-dependent / non-stationary Newcomb-like environment?

Setup: 100 rounds, predictor accuracy α=0.95 for the first `flip_at`=50
rounds, then α=0.20 for the remaining 50. Optimal action flips at the
boundary (one-box optimal under α>0.5, two-box optimal under α<0.5).

Result (committed): NULL — same mechanism as the bandit and gridworld
experiments. KU IB collapses to single-belief behaviour once observations
resolve the likelihood; even the constant 'always two-box' heuristic
beats both IB variants because two-box happens to be reasonably-good
under both α regimes given this reward table.

Run with:  uv run python experiments/iterated_newcomb/main.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ibrl.environments.imperfect_newcomb import ImperfectNewcombEnvironment
from ibrl.simulators import simulate_multi_episode
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.infrabayesian import GaussianBelief
from ibrl.analysis import bootstrap_ci


def _make_pure_action_agent(action_idx: int):
    """Returns an InfraBayesianAgent subclass that ignores its belief and
    plays a pure strategy. Used as a hand-coded baseline."""
    class _PureAction(InfraBayesianAgent):
        def get_probabilities(self):
            p = np.zeros(self.num_actions)
            p[action_idx] = 1.0
            return p
    return _PureAction


def make_agents(seed: int):
    return {
        "one-box (pure)": _make_pure_action_agent(0)(
            num_actions=2, seed=seed, beliefs=[GaussianBelief(2)]),
        "two-box (pure)": _make_pure_action_agent(1)(
            num_actions=2, seed=seed, beliefs=[GaussianBelief(2)]),
        "ib-single (μ=5)": InfraBayesianAgent(
            num_actions=2, seed=seed,
            beliefs=[GaussianBelief(2, prior_mean=5.0)]),
        "ib-KU-2 (μ=0,10)": InfraBayesianAgent(
            num_actions=2, seed=seed, beliefs=[
                GaussianBelief(2, prior_mean=0.0),
                GaussianBelief(2, prior_mean=10.0),
            ]),
        "ib-KU-3 (μ=0,5,10)": InfraBayesianAgent(
            num_actions=2, seed=seed, beliefs=[
                GaussianBelief(2, prior_mean=0.0),
                GaussianBelief(2, prior_mean=5.0),
                GaussianBelief(2, prior_mean=10.0),
            ]),
    }


AGENT_NAMES = ["one-box (pure)", "two-box (pure)", "ib-single (μ=5)",
               "ib-KU-2 (μ=0,10)", "ib-KU-3 (μ=0,5,10)"]


def run(num_seeds: int, num_steps: int, flip_at: int,
        alpha_high: float, alpha_low: float,
        output_path: Path) -> None:
    t0 = time.time()
    rewards = {name: np.zeros((num_seeds, num_steps)) for name in AGENT_NAMES}
    cum_reward = {name: np.zeros(num_seeds) for name in AGENT_NAMES}

    for s in range(num_seeds):
        agents = make_agents(s)
        for name, agent in agents.items():
            env = ImperfectNewcombEnvironment(
                alpha_high=alpha_high, alpha_low=alpha_low,
                flip_at=flip_at, seed=s)
            r = simulate_multi_episode(env, agent,
                                       num_episodes=1, steps_per_episode=num_steps,
                                       seed=s)
            rewards[name][s] = r["rewards"][0]
            cum_reward[name][s] = r["rewards"][0].sum()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "meta__num_seeds":  np.int32(num_seeds),
        "meta__num_steps":  np.int32(num_steps),
        "meta__flip_at":    np.int32(flip_at),
        "meta__alpha_high": np.float64(alpha_high),
        "meta__alpha_low":  np.float64(alpha_low),
        "meta__agent_names": np.array(AGENT_NAMES),
    }
    for name in AGENT_NAMES:
        save[f"rewards__{name}"] = rewards[name]
    np.savez(output_path, **save)

    optimal_total = flip_at * (alpha_high * 10.0 + (1 - alpha_high) * 0.0) + \
                    (num_steps - flip_at) * (alpha_low * 5.0 + (1 - alpha_low) * 15.0)

    print(f"30 seeds × {num_steps} steps; predictor α flips "
          f"{alpha_high}→{alpha_low} at step {flip_at}.")
    print(f"Optimal total per seed = {optimal_total:.2f}")
    print(f"\n{'agent':22s}  {'total reward':>14s}  {'true regret':>12s}  "
          f"{'pre-flip avg':>13s}  {'post-flip avg':>14s}")
    for name in AGENT_NAMES:
        cr = cum_reward[name]
        regret = optimal_total - cr
        m_r, lo_r, hi_r = bootstrap_ci(regret.reshape(-1, 1), n_boot=2000)
        pre = rewards[name][:, :flip_at].mean()
        post = rewards[name][:, flip_at:].mean()
        print(f"  {name:20s}  {cr.mean():14.2f}  {m_r[0]:7.2f} ± {(hi_r[0]-lo_r[0])/2:5.2f}  "
              f"{pre:13.3f}  {post:14.3f}")
    print(f"\nSaved → {output_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--flip-at", type=int, default=50)
    p.add_argument("--alpha-high", type=float, default=0.95)
    p.add_argument("--alpha-low", type=float, default=0.20)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "outputs" / "results.npz")
    args = p.parse_args()
    run(num_seeds=args.num_seeds, num_steps=args.num_steps,
        flip_at=args.flip_at, alpha_high=args.alpha_high,
        alpha_low=args.alpha_low, output_path=args.output)


if __name__ == "__main__":
    main()
