import numpy as np

from ..agents import BaseAgent
from ..environments import BaseEnvironment
from ..utils import sample_action,dump_array


def simulate(
        env : BaseEnvironment,
        agent : BaseAgent,
        options : dict = dict()) -> dict:
    """
    Simulate interactions between agent and environment

    Arguments:
        env:     the environment
        agent:   the agent
        options: optional dictionary of further options, namely
            num_steps:   Number of steps to simulate in each run
            num_runs:    Number of runs
            num_actions: Number of actions
            verbose:     Request debugging output

    Returns:
        A dictionary containing summary information, namely
            average_reward: a (2,num_steps) array containing the average reward and squared reward^2 at each step
            optimal_reward: the expected reward for an agent with optimal policy
            probabilities:  a (num_runs,num_steps,num_actions) array of all probabilities produced by the agent
            actions:        a (num_runs,num_steps) array of all actions sampled from the agent's probability distributions
            rewards:        a (num_runs,num_steps) array of all rewards received by the agent
    """
    num_steps = options.get("num_steps", 101)
    num_runs = options.get("num_runs", 1)
    verbose = options.get("verbose", 0)
    num_actions = options.get("num_actions", env.num_actions)

    average_reward = np.zeros((2,num_steps)) # average reward, average (reward^2)
    optimal_reward = 0
    all_probabilities = np.zeros((num_runs, num_steps, num_actions))
    actions = np.zeros((num_runs, num_steps), dtype=np.int64)
    rewards = np.zeros((num_runs, num_steps))

    for run in range(num_runs):
        if verbose > 0:
            print(f"Run: {run}")
        env.reset()
        agent.reset()
        optimal_reward += env.get_optimal_reward()
        for step in range(num_steps):
            probabilities = agent.get_probabilities()
            env.predict(probabilities)
            action = sample_action(agent.random, probabilities)
            reward = env.interact(action)
            agent.update(probabilities, action, reward)

            average_reward[0, step] += reward
            average_reward[1, step] += reward**2
            all_probabilities[run, step, :] = probabilities
            actions[run, step] = action
            rewards[run, step] = reward

            if verbose > 0:
                print(f"Step:{step:5d}; Action:{action:2d}; Reward:{reward:6.2f}; Probabilities: {dump_array(probabilities)}; Agent state: {agent.dump_state()}")

    average_reward /= num_runs
    optimal_reward /= num_runs

    results = {
        "average_reward": average_reward,
        "optimal_reward": optimal_reward,
        "probabilities": all_probabilities,
        "actions": actions,
        "rewards": rewards
    }

    return results


def simulate_multi_episode(
        env: BaseEnvironment,
        agent: BaseAgent,
        *,
        num_episodes: int,
        steps_per_episode: int,
        reset_agent_belief: bool = False,
        seed: int | None = None,
        verbose: int = 0,
        snapshot_belief: bool = False) -> dict:
    """
    Multi-episode simulation: agent's belief persists across episodes.

    Calls env.reset() once at the start to fully initialise environment state
    (e.g. sample bandit arm parameters), then env.reset_episode() at each
    subsequent episode boundary so persistent state (arm parameters) carries
    across episodes while per-episode RNG is fresh.

    Calls agent.reset() once at the start. At each subsequent episode
    boundary, calls agent.reset_episode() (RNG-only) to preserve learned
    belief across episodes. If reset_agent_belief=True, also calls
    agent.reset_belief() at every episode boundary (useful as a control
    condition: this should perform like single-episode learning).

    Arguments:
        env:                  the environment
        agent:                the agent
        num_episodes:         number of episodes
        steps_per_episode:    steps within each episode
        reset_agent_belief:   if True, wipe agent belief at each episode boundary
        seed:                 if given, override agent.seed and env.seed before resetting
        verbose:              0 (silent) | 1 (per-episode) | 2 (per-step)
        snapshot_belief:      if True, store agent.dump_state() at each episode end

    Returns dict with keys:
        rewards:              (num_episodes, steps_per_episode)
        actions:              (num_episodes, steps_per_episode) int64
        probabilities:        (num_episodes, steps_per_episode, num_actions)
        optimal_per_episode:  (num_episodes,) — env.get_optimal_reward() at each episode start
        regret_per_step:      (num_episodes, steps_per_episode) — optimal - reward
        cum_regret_flat:      (num_episodes * steps_per_episode,) — running sum across the full trajectory
        per_episode_regret:   (num_episodes,) — mean regret within each episode
        belief_snapshots:     list[str] of length num_episodes (if snapshot_belief), else []
    """
    assert num_episodes >= 1
    assert steps_per_episode >= 1

    if seed is not None:
        agent.seed = int(seed)
        env.seed = int(seed) ^ 0xA5A5A5A5

    num_actions = env.num_actions

    rewards = np.zeros((num_episodes, steps_per_episode))
    actions = np.zeros((num_episodes, steps_per_episode), dtype=np.int64)
    probabilities = np.zeros((num_episodes, steps_per_episode, num_actions))
    optimal_per_episode = np.zeros(num_episodes)
    belief_snapshots: list[str] = []

    env.reset()
    agent.reset()

    for ep in range(num_episodes):
        if ep > 0:
            env.reset_episode()
            if reset_agent_belief:
                agent.reset_belief()
            agent.reset_episode()

        optimal_per_episode[ep] = env.get_optimal_reward()

        if verbose >= 1:
            print(f"Episode {ep}: optimal={optimal_per_episode[ep]:.3f}")

        for step in range(steps_per_episode):
            probs = agent.get_probabilities()
            env.predict(probs)
            action = sample_action(agent.random, probs)
            reward = env.interact(action)
            agent.update(probs, action, reward)

            probabilities[ep, step, :] = probs
            actions[ep, step] = action
            rewards[ep, step] = reward

            if verbose >= 2:
                print(f"  Ep {ep} Step {step}: A={action}, R={reward:.3f}, P={dump_array(probs)}, S={agent.dump_state()}")

        if snapshot_belief:
            belief_snapshots.append(agent.dump_state())

    regret_per_step = optimal_per_episode[:, None] - rewards
    per_episode_regret = regret_per_step.mean(axis=1)
    cum_regret_flat = np.cumsum(regret_per_step.reshape(-1))

    return {
        "rewards": rewards,
        "actions": actions,
        "probabilities": probabilities,
        "optimal_per_episode": optimal_per_episode,
        "regret_per_step": regret_per_step,
        "cum_regret_flat": cum_regret_flat,
        "per_episode_regret": per_episode_regret,
        "belief_snapshots": belief_snapshots,
    }
