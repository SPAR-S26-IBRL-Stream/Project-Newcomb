import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibrl.agents import InfraBayesianAgent
from ibrl.environments.newcomb import NewcombEnvironment as NewcombEnv
from ibrl.infrabayesian.beliefs import NewcombLikeBelief
from ibrl.simulators import simulate


def main() -> None:
    options = {
        "num_actions": 2,
        "num_steps": 100,
        "num_runs": 5,
        "seed": 42,
        "verbose": 0,
    }

    # In this environment:
    # action 0 = one-box
    # action 1 = two-box
    env = NewcombEnv(
        num_actions=options["num_actions"],
        seed=options["seed"],
        verbose=options["verbose"],
    )

    # The IB agent needs an explicit belief object. A Newcomb-like belief makes
    # sense here because the environment payoff depends on both prediction and action.
    agent = InfraBayesianAgent(
        num_actions=options["num_actions"],
        beliefs=[NewcombLikeBelief(num_actions=options["num_actions"])],
        epsilon=0.1,
        seed=options["seed"] + 0x01234567,
        verbose=options["verbose"],
    )

    # A few runs are enough without turning this into
    # a bigger benchmark script.
    results = simulate(env, agent, options)

    average_reward = results["average_reward"][0]

    # Look at the policy at the final step to see which action the agent prefers
    # after interacting with the environment for a while
    final_policy = results["probabilities"][:, -1, :].mean(axis=0)

    print("Newcomb InfraBayesian experiment")
    print(f"optimal_reward={results['optimal_reward']:.3f}")
    print(f"mean_reward={average_reward.mean():.3f}")
    print(f"final_step_reward={average_reward[-1]:.3f}")
    print(f"final_policy={np.array2string(final_policy, precision=3)}")


if __name__ == "__main__":
    main()