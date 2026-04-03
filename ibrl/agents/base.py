from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray

from ..outcome import Outcome


class BaseAgent(ABC):
    """
    Base class for all agents

    Arguments:
        num_actions: Number of discrete actions
        seed:        Seed for random number generator
        verbose:     Request debugging output
    """
    def __init__(self,
            num_actions : int,
            *,
            seed : int = 0x01234567,  # Default needs to be different from environment
            verbose : int = 0):
        """
        Initialise permanent state
        Must call reset() before initial interaction with environment
        """
        assert isinstance(num_actions,int) and num_actions >= 1
        self.num_actions = num_actions
        self.seed = seed
        self.verbose = verbose

    @abstractmethod
    def get_probabilities(self) -> NDArray[np.float64]:
        """
        Return the probability distribution to be used in the next episode. Action are sampled from this distribution.
        The distribution thus fixes the entire behaviour of the agent during the episode.
        """
        pass

    def update(self, probabilities : NDArray[np.float64], action : int, outcome : Outcome) -> None:
        """
        Update internal state based on outcome of the episode

        Arguments:
            probabilities: The probability distribution over actions given by the agent
            action:        The action that was selected from the probability distribution
            outcome:       The full outcome (reward + env_action)
        """
        self.step += 1

    def reset(self) -> None:
        """
        Reset internal state
        Called before interacting with a new environment
        """
        self.step = 1
        self.seed += 1
        self.random = np.random.default_rng(seed = self.seed)

    @abstractmethod
    def dump_state(self) -> str:
        """
        Return short (<1 line) representation of the agent's state (for debugging)
        Potentially output in more detail if self.verbose > 1
        """
        pass