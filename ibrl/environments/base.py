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

    def step(self, probabilities : np.ndarray, action : int) -> Outcome:
        """Public environment interaction method.

        The template is:
        1. sample a discrete observation/event index, if the environment has one;
        2. resolve the scalar reward from that observation and the selected action;
        3. package both into Outcome.

        Arguments:
            probabilities: The agent's policy (probability distribution over actions)
            action:        The action sampled from the policy

        Returns:
            Outcome containing the scalar reward and a discrete observation index,
            or observation=None when the environment has no separate finite
            observation channel.
        """
        observation = self._respond(probabilities, action)
        reward = self._resolve(observation, action)
        return Outcome(reward=reward, observation=observation)

    def _respond(self, probabilities : np.ndarray, action : int) -> int | None:
        """Sample the environment's discrete observation/event index.

        For Newcomb-like environments, this may depend on the full agent policy.
        For observed bandits, this may depend on the selected action. For
        unobserved continuous-reward bandits, return None.
        """
        return None

    @abstractmethod
    def _resolve(self, observation : int | None, action : int) -> float:
        """Return the scalar reward for an observation/action pair.
        """
        pass

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
