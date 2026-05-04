from abc import ABC,abstractmethod
import numpy as np

from ..outcome import Outcome


class BaseEnvironment(ABC):
    """
    Base class for all environments

    Assumes a finite number of discrete actions

    Arguments:
        num_actions: Number of discrete actions
        num_steps:   Number of steps per run (for planning)
        num_runs:    Number of runs (for planning)
        seed:        Seed for random number generator
        verbose:     Request debugging output
    """
    def __init__(self, *,
            num_actions : int,
            num_steps : int = None,
            num_runs : int = None,
            seed : int = 0x89abcdef,  # Default needs to be different from agent
            verbose : int = 0):
        """
        Initialise permanent state
        Must call reset() before initial interaction with agent
        """
        assert isinstance(num_actions,int) and num_actions >= 1
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.num_runs = num_runs
        self.seed = seed
        self.verbose = verbose

    @abstractmethod
    def step(self, probabilities : np.ndarray, action : int) -> Outcome:
        """Execute one round of interaction.

        For standard bandits, this just samples an observation/reward based on the action
        For Newcomb-like environments, there are two-steps:
        - Respond to the agent's policy (e.g. the predictor samples a prediction)
        - Resolve the payoff given the agent's action

        Arguments:
            probabilities: The agent's policy (probability distribution over actions)
            action:        The action sampled from the policy

        Returns:
            Outcome containing:
                the obtained reward (float)
                a discrete observation (int)
        """
        observation = self._respond(probabilities)
        reward = self._resolve(observation, action)
        return Outcome(reward=reward, observation=observation)

    @abstractmethod
    def get_optimal_reward(self) -> float:
        """
        Compute the average reward obtained by the optimal policy

        Returns:
            expected value of reward for optimal policy
        """
        pass

    def reset(self):
        """
        Reset internal state. Potentially initialise randomly
        """
        self.seed += 1
        self.random = np.random.default_rng(seed = self.seed)
