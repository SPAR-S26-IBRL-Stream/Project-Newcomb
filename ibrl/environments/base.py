from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray

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
    def __init__(self,
            num_actions : int,
            num_steps : int = None,
            num_runs : int = None,
            *,
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

    def step(self, probabilities : NDArray[np.float64], action : int) -> Outcome:
        """Execute one round of interaction.

        The environment first responds to the agent's policy (e.g. the predictor
        samples a prediction), then resolves the payoff given the agent's action.

        Arguments:
            probabilities: The agent's policy (probability distribution over actions)
            action:        The action sampled from the policy

        Returns:
            Outcome containing the reward and any environment action
        """
        env_action = self._respond(probabilities)
        reward = self._resolve(env_action, action)
        if isinstance(reward, tuple):
            reward, outcome = reward
        else:
            outcome = None
        return Outcome(reward=reward, outcome=outcome, env_action=env_action)

    def _respond(self, probabilities : NDArray[np.float64]) -> int | None:
        """Environment's move given the agent's policy. Override in subclasses.

        For Newcomb-like environments, this is where the predictor samples its
        prediction. For standard bandits, returns None.
        """
        return None

    @abstractmethod
    def _resolve(self, env_action : int | None, action : int) -> float | tuple[float,int]:
        """Determine the reward given both moves. Override in subclasses.

        Returns either of:
            reward (float)
            tuple of reward (float) and outcome (int), if environment has discrete outcomes
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
