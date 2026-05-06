"""Headline experiment: heavy-tailed bandit, multi-episode, IB vs baselines.

Each agent is run for `num_seeds` independent seeds. Each seed gives a
fresh heavy-tailed bandit (Cauchy-contaminated arms) that is held fixed
across `num_episodes`; the agent's belief carries forward across episodes.
This satisfies the three NeurIPS-bar criteria simultaneously:

  1. Multi-episode learning (belief persists; per-episode RNG fresh).
  2. Genuine non-realizability for any Normal-likelihood agent
     (Cauchy contamination is outside the Normal-Normal hypothesis class
      — no prior fixes the misspecification).
  3. Non-trivial baselines (Thompson sampling, UCB1) whose theoretical
     guarantees rely on sub-Gaussian tails — exactly what the heavy-tailed
     reward distribution violates.

Outputs: experiments/testing/outputs/results.npz with arrays of shape
(num_seeds, num_episodes, steps_per_episode) per agent. The headline
notebook then plots cumulative regret (flattened) and per-episode regret.

Run with:  uv run python experiments/testing/main.py
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from ibrl.agents.bayesian import BayesianAgent
from ibrl.agents.q_learning import QLearningAgent
from ibrl.agents.thompson import ThompsonSamplingAgent
from ibrl.agents.ucb import UCB1Agent
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.environments.heavy_tailed_bandit import HeavyTailedBanditEnvironment
from ibrl.infrabayesian import GaussianBelief
from ibrl.simulators import simulate_multi_episode


def build_agents(num_actions: int, seed: int):
    """Construct one fresh agent of each type, seeded for reproducibility."""
    return {
        "bayesian-greedy": BayesianAgent(num_actions=num_actions, seed=seed, epsilon=0.0),
        "q-learning": QLearningAgent(num_actions=num_actions, seed=seed, epsilon=0.1),
        "thompson": ThompsonSamplingAgent(num_actions=num_actions, seed=seed),
        "ucb1": UCB1Agent(num_actions=num_actions, seed=seed),
        "infrabayesian": InfraBayesianAgent(
            num_actions=num_actions, seed=seed,
            beliefs=[GaussianBelief(num_actions)]),
    }


def run(num_seeds: int, num_episodes: int, steps_per_episode: int,
        num_actions: int, output_path: Path) -> None:
    agent_names = list(build_agents(num_actions=num_actions, seed=0).keys())
    rewards_per_agent = {name: np.zeros((num_seeds, num_episodes, steps_per_episode))
                         for name in agent_names}
    optimal_per_agent = {name: np.zeros((num_seeds, num_episodes))
                         for name in agent_names}

    t0 = time.time()
    for s, seed in enumerate(range(num_seeds)):
        agents = build_agents(num_actions=num_actions, seed=seed)
        for name, agent in agents.items():
            env = HeavyTailedBanditEnvironment(num_actions=num_actions, seed=seed)
            r = simulate_multi_episode(
                env, agent,
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode,
                seed=seed)
            rewards_per_agent[name][s] = r["rewards"]
            optimal_per_agent[name][s] = r["optimal_per_episode"]
        if (s + 1) % 5 == 0 or s == 0:
            print(f"[seed {s+1}/{num_seeds}] elapsed {time.time()-t0:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    for name in agent_names:
        save_kwargs[f"rewards__{name}"] = rewards_per_agent[name]
        save_kwargs[f"optimal__{name}"] = optimal_per_agent[name]
    save_kwargs["meta__agent_names"] = np.array(agent_names)
    save_kwargs["meta__num_seeds"] = np.int32(num_seeds)
    save_kwargs["meta__num_episodes"] = np.int32(num_episodes)
    save_kwargs["meta__steps_per_episode"] = np.int32(steps_per_episode)
    save_kwargs["meta__num_actions"] = np.int32(num_actions)
    np.savez(output_path, **save_kwargs)
    print(f"Saved → {output_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--steps-per-episode", type=int, default=200)
    p.add_argument("--num-actions", type=int, default=10)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "outputs" / "results.npz")
    args = p.parse_args()
    run(num_seeds=args.num_seeds,
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        num_actions=args.num_actions,
        output_path=args.output)


if __name__ == "__main__":
    main()
