"""Robust gridworld headline experiment: IB vs Bayesian / Thompson / Robust DP.

Headline pitch: dynamic consistency, not robust planning. RobustDPAgent and
IBDPAgent share the same defensible initial polytope ([p_nominal − 2ε,
p_nominal + 2ε]); the only difference is that IB updates the polytope from
observations and re-plans every episode while Robust DP commits up front.
The shape that matters is "IB descending across episodes while Robust DP
stays flat" — that is the IB-specific contribution distinguishing this
result from a robust-DP recapitulation.

Run with:  uv run python experiments/gridworld/main.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ibrl.mdp import (
    GridworldEnvironment, IntervalKernelBelief, simulate_mdp_multi_episode,
    RobustDPAgent, BayesianDPAgent, ThompsonDPAgent, IBDPAgent,
)


def widen_polytope(P_lo: np.ndarray, P_hi: np.ndarray, factor: float = 2.0,
                   p_min: float = 0.0, p_max: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Widen the env's per-cell polytope by `factor` around its midpoint, to
    give the agent's RobustDP / IBDP starting point a defensible-but-wider
    interval (the agent doesn't know the env's exact polytope a priori).
    """
    mid = 0.5 * (P_lo + P_hi)
    half = 0.5 * (P_hi - P_lo)
    return (np.clip(mid - factor * half, p_min, p_max),
            np.clip(mid + factor * half, p_min, p_max))


def make_agent(name: str, env: GridworldEnvironment, *,
               seed: int, polytope: tuple[np.ndarray, np.ndarray]):
    base_kwargs = dict(num_states=env.num_states, num_actions=env.num_actions,
                       R=env.R, terminal_mask=env.terminal_mask,
                       gamma=env.gamma, seed=seed)
    if name == "robust-dp":
        return RobustDPAgent(**base_kwargs, initial_polytope=polytope)
    belief = IntervalKernelBelief(num_states=env.num_states,
                                  num_actions=env.num_actions,
                                  alpha_init=0.5, initial_polytope=polytope)
    if name == "bayesian-dp":
        return BayesianDPAgent(**base_kwargs, belief=belief)
    if name == "thompson-dp":
        return ThompsonDPAgent(**base_kwargs, belief=belief)
    if name == "ib-dp":
        return IBDPAgent(**base_kwargs, belief=belief)
    raise ValueError(name)


AGENT_NAMES = ["robust-dp", "bayesian-dp", "thompson-dp", "ib-dp"]


def run(num_seeds: int, num_episodes: int, max_steps: int,
        epsilons: list[float], output_path: Path,
        adversary_mode: str = "static") -> None:
    print(f"Adversary mode: {adversary_mode}")
    t0 = time.time()
    rewards = {(eps, name): np.zeros((num_seeds, num_episodes, max_steps))
               for eps in epsilons for name in AGENT_NAMES}
    ep_regret = {(eps, name): np.zeros((num_seeds, num_episodes))
                 for eps in epsilons for name in AGENT_NAMES}
    cum_regret = {(eps, name): np.zeros((num_seeds, num_episodes))
                  for eps in epsilons for name in AGENT_NAMES}
    ep_lengths = {(eps, name): np.zeros((num_seeds, num_episodes), dtype=np.int64)
                  for eps in epsilons for name in AGENT_NAMES}

    for s in range(num_seeds):
        for eps in epsilons:
            env_template = GridworldEnvironment(epsilon=eps, seed=s,
                                                adversary_mode=adversary_mode)
            env_template.reset()
            # Agent's initial polytope: 2× the env's polytope width, defensible-but-wider
            polytope = widen_polytope(*env_template.kernel_polytope(), factor=2.0)

            for name in AGENT_NAMES:
                env_i = GridworldEnvironment(epsilon=eps, seed=s,
                                             adversary_mode=adversary_mode)
                agent = make_agent(name, env_i, seed=s, polytope=polytope)
                r = simulate_mdp_multi_episode(env_i, agent,
                                               num_episodes=num_episodes,
                                               max_steps_per_episode=max_steps,
                                               seed=s)
                rewards[(eps, name)][s] = r["rewards"]
                ep_regret[(eps, name)][s] = r["episode_regret"]
                cum_regret[(eps, name)][s] = r["cumulative_regret"]
                ep_lengths[(eps, name)][s] = r["episode_lengths"]
        if (s + 1) % 5 == 0 or s == 0:
            print(f"[seed {s+1}/{num_seeds}] elapsed {time.time()-t0:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "meta__epsilons":     np.array(epsilons),
        "meta__agent_names":  np.array(AGENT_NAMES),
        "meta__num_seeds":    np.int32(num_seeds),
        "meta__num_episodes": np.int32(num_episodes),
        "meta__max_steps":    np.int32(max_steps),
        "meta__adversary_mode": np.array(adversary_mode),
    }
    for (eps, name), arr in rewards.items():
        save[f"rewards__eps{eps}__{name}"] = arr
    for (eps, name), arr in ep_regret.items():
        save[f"ep_regret__eps{eps}__{name}"] = arr
    for (eps, name), arr in cum_regret.items():
        save[f"cum_regret__eps{eps}__{name}"] = arr
    for (eps, name), arr in ep_lengths.items():
        save[f"ep_lengths__eps{eps}__{name}"] = arr
    np.savez(output_path, **save)
    print(f"Saved → {output_path}")
    print(f"Total runtime: {time.time()-t0:.1f}s")

    # Quick on-screen summary
    print("\nFinal episode-regret means (last 3 episodes) ± std across seeds:")
    for eps in epsilons:
        print(f"  eps={eps}:")
        for name in AGENT_NAMES:
            late = ep_regret[(eps, name)][:, -3:].mean(axis=1)
            cum_final = cum_regret[(eps, name)][:, -1]
            print(f"    {name:14s}  late-3-ep regret = {late.mean():+7.4f} ± {late.std():.4f}"
                  f"   cum-regret = {cum_final.mean():+7.3f} ± {cum_final.std():.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    p.add_argument("--adversary-mode", type=str, default="static",
                   choices=["static", "per_episode_visit"])
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "outputs" / "results.npz")
    args = p.parse_args()
    run(num_seeds=args.num_seeds, num_episodes=args.num_episodes,
        max_steps=args.max_steps, epsilons=args.epsilons,
        adversary_mode=args.adversary_mode, output_path=args.output)


if __name__ == "__main__":
    main()
