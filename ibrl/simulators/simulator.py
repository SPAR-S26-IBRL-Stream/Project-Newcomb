import numpy as np

from ..agents import BaseAgent
from ..environments import BaseEnvironment
from ..utils import sample_action,dump_array


def simulate(
        env : BaseEnvironment,
        agent : BaseAgent,
        options : dict = dict(),
        seed_env : int | None = None,
        seed_agent : int | None = None) -> dict:
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
        seed_env:    override environment seed
        seed_agent:  override agent seed

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

    if seed_env is not None:
        env.seed = seed_env
    if seed_agent is not None:
        agent.seed = seed_agent

    for run in range(num_runs):
        if verbose > 0:
            print(f"Run: {run}")
        env.reset()
        agent.reset()
        optimal_reward += env.get_optimal_reward()
        for step in range(num_steps):
            probabilities = agent.get_probabilities()
            action = sample_action(agent.random, probabilities)
            outcome = env.step(probabilities, action)
            agent.update(probabilities, action, outcome)

            reward = outcome.reward
            average_reward[0, step] += reward
            average_reward[1, step] += reward**2
            all_probabilities[run, step, :] = probabilities
            actions[run, step] = action
            rewards[run, step] = reward

            if verbose > 0:
                print(f"Step:{step:5d}; A={action:2d}; R={reward:6.2f}; P(A)={dump_array(probabilities)}; Agent state: {agent.dump_state()}")

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
