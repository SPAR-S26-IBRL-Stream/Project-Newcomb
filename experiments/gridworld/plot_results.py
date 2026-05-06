"""Plot regret-vs-episode and final-regret-vs-epsilon from gridworld results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ibrl.analysis import bootstrap_ci


AGENT_ORDER = ["robust-dp", "bayesian-dp", "thompson-dp", "ib-dp"]
AGENT_COLORS = {
    "robust-dp": "tab:gray",
    "bayesian-dp": "tab:blue",
    "thompson-dp": "tab:green",
    "ib-dp": "tab:red",
}


def _load(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    epsilons = list(map(float, d["meta__epsilons"]))
    n_seeds = int(d["meta__num_seeds"])
    n_eps = int(d["meta__num_episodes"])
    adversary_mode = str(d["meta__adversary_mode"])
    return d, epsilons, n_seeds, n_eps, adversary_mode


def plot_regret_per_episode(d, epsilons, *, output_dir: Path, adversary_mode: str):
    """One figure per ε showing per-episode regret with bootstrap CI bands."""
    for eps in epsilons:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name in AGENT_ORDER:
            curves = d[f"ep_regret__eps{eps}__{name}"]  # (n_seeds, n_episodes)
            mean, lo, hi = bootstrap_ci(curves, n_boot=2000, ci=0.95, seed=0)
            x = np.arange(curves.shape[1])
            color = AGENT_COLORS[name]
            ax.plot(x, mean, label=name, color=color, lw=1.6,
                    marker="o", markersize=3)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.axhline(0, color="black", lw=0.6, alpha=0.6)
        ax.set_xlabel("episode")
        ax.set_ylabel("per-episode regret  (V*[start] − discounted return)")
        ax.set_title(f"Per-episode regret @ ε={eps}, adversary={adversary_mode} "
                     f"(95% bootstrap CI, 30 seeds)")
        ax.set_ylim(-0.3, 0.7)  # explicit fixed scale
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        out = output_dir / f"regret_per_episode__eps{eps}__{adversary_mode}.png"
        fig.savefig(out, dpi=160)
        print(f"Saved → {out}")
        plt.close(fig)


def plot_cum_regret_per_episode(d, epsilons, *, output_dir: Path, adversary_mode: str):
    """One figure per ε showing cumulative regret across episodes."""
    for eps in epsilons:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name in AGENT_ORDER:
            curves = d[f"cum_regret__eps{eps}__{name}"]
            mean, lo, hi = bootstrap_ci(curves, n_boot=2000, ci=0.95, seed=0)
            x = np.arange(curves.shape[1])
            color = AGENT_COLORS[name]
            ax.plot(x, mean, label=name, color=color, lw=1.6)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        ax.axhline(0, color="black", lw=0.6, alpha=0.6)
        ax.set_xlabel("episode")
        ax.set_ylabel("cumulative regret across episodes")
        ax.set_title(f"Cumulative regret @ ε={eps}, adversary={adversary_mode} "
                     f"(95% bootstrap CI, 30 seeds)")
        ax.set_ylim(-2, 12)  # explicit fixed scale
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        out = output_dir / f"cum_regret__eps{eps}__{adversary_mode}.png"
        fig.savefig(out, dpi=160)
        print(f"Saved → {out}")
        plt.close(fig)


def plot_final_regret_by_epsilon(d, epsilons, *, output_dir: Path, adversary_mode: str):
    """Final cumulative regret vs ε for each agent."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in AGENT_ORDER:
        means, lows, highs = [], [], []
        for eps in epsilons:
            cum = d[f"cum_regret__eps{eps}__{name}"][:, -1]  # (n_seeds,)
            m, lo, hi = bootstrap_ci(cum.reshape(-1, 1), n_boot=2000)
            means.append(float(m[0]))
            lows.append(float(lo[0]))
            highs.append(float(hi[0]))
        ax.errorbar(epsilons, means,
                    yerr=[np.array(means) - np.array(lows),
                          np.array(highs) - np.array(means)],
                    label=name, color=AGENT_COLORS[name],
                    lw=1.6, marker="o", capsize=4)
    ax.axhline(0, color="black", lw=0.6, alpha=0.6)
    ax.set_xlabel("polytope half-width ε")
    ax.set_ylabel("final cumulative regret (95% bootstrap CI)")
    ax.set_title(f"Final cumulative regret by ε, adversary={adversary_mode}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = output_dir / f"final_regret_by_epsilon__{adversary_mode}.png"
    fig.savefig(out, dpi=160)
    print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=Path(__file__).parent / "outputs" / "results.npz")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "outputs")
    args = p.parse_args()
    d, epsilons, n_seeds, n_eps, adversary_mode = _load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {args.input.name}: {n_seeds} seeds × {n_eps} episodes, "
          f"epsilons={epsilons}, adversary={adversary_mode}")
    plot_regret_per_episode(d, epsilons, output_dir=args.output_dir,
                            adversary_mode=adversary_mode)
    plot_cum_regret_per_episode(d, epsilons, output_dir=args.output_dir,
                                adversary_mode=adversary_mode)
    plot_final_regret_by_epsilon(d, epsilons, output_dir=args.output_dir,
                                 adversary_mode=adversary_mode)


if __name__ == "__main__":
    main()
