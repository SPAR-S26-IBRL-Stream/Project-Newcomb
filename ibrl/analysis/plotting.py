"""Regret-curve plotting with bootstrap CI bands."""
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .metrics import bootstrap_ci


def plot_regret_curves(
        results_by_agent: Mapping[str, NDArray],
        *,
        ax=None,
        n_boot: int = 1000,
        ci: float = 0.95,
        bootstrap_seed: int = 0,
        title: str | None = None,
        xlabel: str = "step",
        ylabel: str = "cumulative regret"):
    """Plot per-agent cumulative-regret curves with bootstrap CI bands.

    Arguments:
        results_by_agent:  dict mapping agent label to array of shape
                           (n_seeds, n_steps) — one cumulative-regret curve
                           per seed.
        ax:                optional matplotlib Axes; created if None.
        n_boot, ci:        forwarded to bootstrap_ci.
        bootstrap_seed:    RNG seed for bootstrap resampling.
        title, xlabel, ylabel: cosmetics.

    Returns the Axes used.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for label, curves in results_by_agent.items():
        curves = np.asarray(curves)
        mean, lo, hi = bootstrap_ci(curves, n_boot=n_boot, ci=ci, seed=bootstrap_seed)
        x = np.arange(mean.shape[0])
        line, = ax.plot(x, mean, label=label, lw=1.6)
        ax.fill_between(x, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_per_episode_regret(
        results_by_agent: Mapping[str, NDArray],
        *,
        ax=None,
        n_boot: int = 1000,
        ci: float = 0.95,
        bootstrap_seed: int = 0,
        title: str | None = None):
    """Plot per-episode mean regret with bootstrap CI bands.

    Arguments:
        results_by_agent:  dict mapping label to (n_seeds, n_episodes) array
                           of per-episode regret values.
        Other arguments as for plot_regret_curves.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for label, curves in results_by_agent.items():
        curves = np.asarray(curves)
        mean, lo, hi = bootstrap_ci(curves, n_boot=n_boot, ci=ci, seed=bootstrap_seed)
        x = np.arange(mean.shape[0])
        line, = ax.plot(x, mean, label=label, lw=1.6, marker="o", markersize=3)
        ax.fill_between(x, lo, hi, alpha=0.2, color=line.get_color())

    ax.set_xlabel("episode")
    ax.set_ylabel("mean regret per step")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
