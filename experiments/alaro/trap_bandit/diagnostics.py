"""Diagnostics for cached trap-bandit trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_agent_runs(raw_path: Path, agent: str) -> dict[str, np.ndarray]:
    data = np.load(raw_path)
    keys = [
        "actions",
        "catastrophes",
        "trapped_pulls",
        "expected_regret",
        "cumulative_expected_regret",
        "risky",
    ]
    return {key: data[f"{agent}__{key}"] for key in keys}


def select_runs(
    runs: dict[str, np.ndarray],
    *,
    group: str,
    top_n: int,
) -> np.ndarray:
    if group == "risky":
        mask = runs["risky"]
    elif group == "safe":
        mask = ~runs["risky"]
    elif group == "all":
        mask = np.ones_like(runs["risky"], dtype=bool)
    else:
        raise ValueError(f"unknown group {group}")

    indices = np.flatnonzero(mask)
    final_regret = runs["cumulative_expected_regret"][:, -1]
    order = indices[np.argsort(final_regret[indices])[::-1]]
    return order[:top_n]


def plot_selected_runs(
    runs: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    condition: str,
    agent: str,
    group: str,
    output_path: Path,
) -> None:
    steps = np.arange(runs["actions"].shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    cmap = plt.get_cmap("tab10")

    for rank, world_idx in enumerate(indices):
        color = cmap(rank % 10)
        label = f"world {world_idx}"
        cumulative_regret = runs["cumulative_expected_regret"][world_idx]
        trapped_pulls = runs["trapped_pulls"][world_idx]
        cumulative_trapped_rate = np.cumsum(trapped_pulls) / (steps + 1)

        axes[0].plot(steps, cumulative_regret, color=color, label=label)
        axes[1].plot(steps, cumulative_trapped_rate, color=color, label=label)

        trapped_steps = steps[trapped_pulls]
        safe_steps = steps[~trapped_pulls]
        actions = runs["actions"][world_idx]
        axes[2].scatter(
            safe_steps,
            actions[safe_steps] + rank * 0.08,
            color=color,
            marker=".",
            s=10,
            alpha=0.35,
        )
        axes[2].scatter(
            trapped_steps,
            actions[trapped_steps] + rank * 0.08,
            color=color,
            marker="x",
            s=18,
        )
        catastrophe_steps = steps[runs["catastrophes"][world_idx]]
        if len(catastrophe_steps):
            axes[2].scatter(
                catastrophe_steps,
                actions[catastrophe_steps] + rank * 0.08,
                color="black",
                marker="*",
                s=55,
            )

    axes[0].set_ylabel("cum. expected regret")
    axes[1].set_ylabel("cum. trapped pull rate")
    axes[2].set_ylabel("action")
    axes[2].set_xlabel("step")
    axes[2].set_yticks([0, 1])
    axes[2].set_ylim(-0.25, 1.9)
    axes[0].legend(loc="upper left", fontsize=8, ncols=2)
    fig.suptitle(f"{condition}: top {len(indices)} {group} {agent} runs by final regret")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_run_table(runs: dict[str, np.ndarray], indices: np.ndarray) -> None:
    print("world,final_regret,total_trapped_pulls,last_half_trapped_rate,catastrophes")
    half = runs["trapped_pulls"].shape[1] // 2
    for world_idx in indices:
        final_regret = runs["cumulative_expected_regret"][world_idx, -1]
        trapped = runs["trapped_pulls"][world_idx]
        last_half_rate = float(np.mean(trapped[half:]))
        catastrophes = int(np.sum(runs["catastrophes"][world_idx]))
        print(
            f"{world_idx},{final_regret:.6f},{int(np.sum(trapped))},"
            f"{last_half_rate:.3f},{catastrophes}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--condition", default="correct")
    parser.add_argument("--agent", default="ib")
    parser.add_argument("--group", choices=["all", "risky", "safe"], default="risky")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.results_dir / f"{args.condition}_raw.npz"
    runs = load_agent_runs(raw_path, args.agent)
    indices = select_runs(runs, group=args.group, top_n=args.top_n)
    output_path = args.output
    if output_path is None:
        output_path = (
            args.results_dir
            / f"{args.condition}_{args.agent}_{args.group}_top_regret_runs.png"
        )
    plot_selected_runs(
        runs,
        indices,
        condition=args.condition,
        agent=args.agent,
        group=args.group,
        output_path=output_path,
    )
    print_run_table(runs, indices)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
