import argparse
import numpy as np

from ibrl.simulators import simulate
from ibrl.utils import construct_environment,construct_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL test with multi-armed bandit")
    parser.add_argument("environment",          help="Environment type",            type=str)
    parser.add_argument("agent",                help="Agent type",                  type=str)
    parser.add_argument("-k", "--arms",         help="Number of arms",              type=int,       default=2)
    parser.add_argument("-s", "--steps",        help="Number of steps per episode", type=int,       default=1001)
    parser.add_argument("-r", "--runs",         help="Number of episodes to run",   type=int,       default=1)
    parser.add_argument(      "--seed",         help="Random number seed",          type=int,       default=42)
    parser.add_argument("-v", "--verbose",      help="Debug output",                action="count", default=0)
    parsed_args = parser.parse_args()

    options = {
        "num_actions": parsed_args.arms,
        "num_steps":   parsed_args.steps,
        "num_runs":    parsed_args.runs,
        "seed":        parsed_args.seed,
        "verbose":     parsed_args.verbose
    }
    env = construct_environment(parsed_args.environment, options)
    agent = construct_agent(parsed_args.agent, options)
    results = simulate(env, agent, options)

    # Print average reward obtained (for plotting)
    average_reward = results["average_reward"]
    optimal_reward = results["optimal_reward"]
    for i in range(options["num_steps"]):
        average_reward_spread = np.sqrt(average_reward[1,i] - average_reward[0,i]**2)
        average_reward_unc = average_reward_spread / np.sqrt(options["num_runs"])

        print(
            i,
            optimal_reward,          # Expected reward when using optimal policy (as determined by environment)
            average_reward[0,i],     # Average reward received by agent
            average_reward_unc,      # Uncertainty of reward (converges to 0 as num_runs goes to infinity)
            average_reward_spread,   # Spread of reward (converges to constant)
        )
