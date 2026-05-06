from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray


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
        assert isinstance(num_actions,int) and num_actions >= 2
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

    def update(self, probabilities : NDArray[np.float64], action : int, reward : float) -> None:
        """
        Update internal state based on outcome of the episode

        Arguments:
            probabilities: The probability distribution over actions given by the agent
            action:        The action that was selected from the probability distribution
            reward:        The reward received
        """
        self.step += 1

    def reset(self) -> None:
        """
        Full reset: wipe learned belief AND re-seed RNG / zero step counter.
        Used by single-episode callers.
        """
        self.reset_belief()
        self.reset_episode()

    def reset_belief(self) -> None:
        """
        Wipe learned belief state ONLY (preserve RNG / step counter).
        Default: no-op. Subclasses with persistent belief should override.
        Used by simulate_multi_episode when reset_agent_belief=True.
        """
        pass

    def reset_episode(self) -> None:
        """
        Re-seed RNG and zero step counter ONLY (preserve learned belief).
        Used by simulate_multi_episode at every episode boundary so the
        agent's belief carries across episodes while per-episode RNG
        state is fresh.
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
