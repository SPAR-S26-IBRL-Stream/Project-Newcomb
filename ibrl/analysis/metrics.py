"""Regret metrics and bootstrap CIs for multi-episode experiments."""
import numpy as np
from numpy.typing import NDArray


def cumulative_regret(rewards: NDArray, optimal_per_episode: NDArray) -> NDArray:
    """Per-step cumulative regret across the flattened multi-episode trajectory.

    Arguments:
        rewards:              shape (n_episodes, steps_per_episode)
        optimal_per_episode:  shape (n_episodes,)

    Returns:
        1-D array of length n_episodes * steps_per_episode — cumulative
        regret at each total step index.
    """
    rewards = np.asarray(rewards)
    optimal = np.asarray(optimal_per_episode)
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be 2-D, got shape {rewards.shape}")
    if optimal.shape != (rewards.shape[0],):
        raise ValueError(
            f"optimal_per_episode must have shape (n_episodes,), got {optimal.shape}")
    regret_per_step = optimal[:, None] - rewards
    return np.cumsum(regret_per_step.reshape(-1))


def per_episode_regret(rewards: NDArray, optimal_per_episode: NDArray) -> NDArray:
    """Mean within-episode regret per episode.

    Arguments same as cumulative_regret. Returns shape (n_episodes,).
    """
    rewards = np.asarray(rewards)
    optimal = np.asarray(optimal_per_episode)
    return (optimal[:, None] - rewards).mean(axis=1)


def bootstrap_ci(curves: NDArray, *, n_boot: int = 1000, ci: float = 0.95,
                 seed: int = 0) -> tuple[NDArray, NDArray, NDArray]:
    """Bootstrap mean curve and CI band over the seed (first) axis.

    Arguments:
        curves:  shape (n_seeds, T) — one curve per seed
        n_boot:  number of bootstrap resamples
        ci:      confidence level in (0, 1)

    Returns:
        (mean_curve, lo, hi) — each shape (T,). The mean is the empirical
        mean across seeds; (lo, hi) is the bootstrap CI on that mean.
    """
    curves = np.asarray(curves)
    if curves.ndim != 2:
        raise ValueError(f"curves must be 2-D (n_seeds, T), got {curves.shape}")
    n_seeds, T = curves.shape
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, T))
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[b] = curves[idx].mean(axis=0)

    alpha = (1.0 - ci) / 2.0
    lo = np.quantile(boot_means, alpha, axis=0)
    hi = np.quantile(boot_means, 1.0 - alpha, axis=0)
    mean_curve = curves.mean(axis=0)
    return mean_curve, lo, hi
