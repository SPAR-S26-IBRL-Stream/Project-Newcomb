"""Plot regret curves from the headline experiment results.

Loads experiments/testing/outputs/results.npz produced by main.py and
emits two PNG figures into the same outputs/ folder:

  regret_vs_total_steps.png — cumulative regret across the flattened
                              multi-episode trajectory, with bootstrap
                              95 percent CIs across seeds.
  per_episode_regret.png    — mean within-episode regret as a function
                              of episode index, with bootstrap CIs.

Run with:  uv run python experiments/testing/plot_results.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ibrl.analysis import (
    cumulative_regret, per_episode_regret,
    plot_regret_curves, plot_per_episode_regret,
)


def load(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    agents = list(d["meta__agent_names"])
    cum_curves = {}
    per_ep_curves = {}
    for name in agents:
        rew = d[f"rewards__{name}"]                  # (n_seeds, n_eps, T)
        opt = d[f"optimal__{name}"]                  # (n_seeds, n_eps)
        n_seeds = rew.shape[0]
        cum = np.zeros((n_seeds, rew.shape[1] * rew.shape[2]))
        per_ep = np.zeros((n_seeds, rew.shape[1]))
        for s in range(n_seeds):
            cum[s] = cumulative_regret(rew[s], opt[s])
            per_ep[s] = per_episode_regret(rew[s], opt[s])
        cum_curves[name] = cum
        per_ep_curves[name] = per_ep
    return cum_curves, per_ep_curves, dict(d.items())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path(__file__).parent / "outputs" / "results.npz")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "outputs")
    args = p.parse_args()

    cum_curves, per_ep_curves, meta = load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    plot_regret_curves(cum_curves, ax=ax1,
                       title="Cumulative regret on heavy-tailed bandit (95% bootstrap CI)")
    fig1.tight_layout()
    out1 = args.output_dir / "regret_vs_total_steps.png"
    fig1.savefig(out1, dpi=160)
    print(f"Saved → {out1}")

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    plot_per_episode_regret(per_ep_curves, ax=ax2,
                            title="Per-episode mean regret across episodes")
    fig2.tight_layout()
    out2 = args.output_dir / "per_episode_regret.png"
    fig2.savefig(out2, dpi=160)
    print(f"Saved → {out2}")

    print("\nFinal-step cumulative regret (mean ± std across seeds):")
    for name, cum in cum_curves.items():
        m, sd = float(cum[:, -1].mean()), float(cum[:, -1].std())
        print(f"  {name:18s}  {m:8.2f} ± {sd:7.2f}")


if __name__ == "__main__":
    main()
