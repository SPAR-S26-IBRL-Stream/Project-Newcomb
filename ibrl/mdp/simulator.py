"""Episodic MDP simulation loop with multi-episode belief persistence."""
from __future__ import annotations

import numpy as np

from .base import MDPAgent, MDPEnvironment


def simulate_mdp_multi_episode(
        env: MDPEnvironment,
        agent: MDPAgent,
        *,
        num_episodes: int,
        max_steps_per_episode: int,
        seed: int | None = None,
        gamma: float = 0.95,
        verbose: int = 0) -> dict:
    """Run an MDP agent on an MDP env across `num_episodes`.

    Across-episode persistence: env transition kernel and agent belief
    both carry forward (only RNG / step counters reset).

    Returns dict with:
        rewards:               (num_episodes, max_steps_per_episode) — 0 after termination
        actions:               (num_episodes, max_steps_per_episode) int64 — −1 after termination
        states:                (num_episodes, max_steps_per_episode + 1) int64 — −1 after termination
        episode_lengths:       (num_episodes,) — number of real steps before termination
        episode_returns:       (num_episodes,) — sum of rewards per episode
        episode_regret:        (num_episodes,) — V*(s0) - episode_return (one-step-discounted)
        cumulative_regret:     (num_episodes,) — running sum of episode_regret
    """
    assert num_episodes >= 1
    assert max_steps_per_episode >= 1

    if seed is not None:
        agent.seed = int(seed)
        env.seed = int(seed) ^ 0xA5A5A5A5

    rewards = np.zeros((num_episodes, max_steps_per_episode))
    actions = np.full((num_episodes, max_steps_per_episode), -1, dtype=np.int64)
    states = np.full((num_episodes, max_steps_per_episode + 1), -1, dtype=np.int64)
    episode_lengths = np.zeros(num_episodes, dtype=np.int64)
    episode_returns = np.zeros(num_episodes)
    episode_regret = np.zeros(num_episodes)

    env.reset()
    agent.reset()
    oracle_V = env.get_oracle_value(gamma=gamma)

    for ep in range(num_episodes):
        if ep > 0:
            env.reset_episode()
            agent.reset_episode()

        state = env._initial_state()
        states[ep, 0] = state
        agent.plan()

        if verbose >= 1:
            print(f"Episode {ep}: start_state={state}, V*(s0)={oracle_V[state]:.3f}")

        ep_return = 0.0
        gamma_pow = 1.0
        for t in range(max_steps_per_episode):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.observe(state, action, next_state, reward, done)

            actions[ep, t] = action
            rewards[ep, t] = reward
            states[ep, t + 1] = next_state
            ep_return += gamma_pow * reward
            gamma_pow *= gamma

            if verbose >= 2:
                print(f"  t={t} s={state} a={action} -> s'={next_state} r={reward:.3f} done={done}")

            state = next_state
            if done:
                episode_lengths[ep] = t + 1
                break
        else:
            episode_lengths[ep] = max_steps_per_episode

        episode_returns[ep] = ep_return
        episode_regret[ep] = oracle_V[states[ep, 0]] - ep_return

    cumulative_regret = np.cumsum(episode_regret)

    return {
        "rewards": rewards,
        "actions": actions,
        "states": states,
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "episode_regret": episode_regret,
        "cumulative_regret": cumulative_regret,
        "oracle_V": oracle_V,
    }
