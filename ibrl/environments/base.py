from abc import ABC,abstractmethod
import numpy as np
from numpy.typing import NDArray


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
        assert isinstance(num_actions,int) and num_actions >= 2
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.num_runs = num_runs
        self.seed = seed
        self.verbose = verbose

    def predict(self, probabilities : NDArray[np.float64]) -> None:
        """
        Let the predictor set up the environment. The predictor has access to the probability distribution from which
        the agent samples its actions, but not the action itself. The predictor may adjust rewards or otherwise modify
        the environment based on this distribution.

        Arguments:
            probabilities: Probability distribution for actions by the agent
        """
        pass

    @abstractmethod
    def interact(self, action : int) -> float:
        """
        Perform the interaction of the agent with the environment, based on the action chosen by the agent.
        The interaction is purely classical, i.e. it does not depend on the agent's policy. Potential policy-dependence
        arises when the predictor sets up the environment prior to the interaction.

        Arguments:
            action: Action chosen by the agent

        Returns:
            reward of the interaction
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
        Full reset: re-seed RNG and re-initialise persistent state.
        Subclasses extend with state initialisation (e.g. resampling arms).
        Used by single-episode callers.
        """
        self.reset_episode()

    def reset_episode(self):
        """
        Re-seed RNG only; preserve persistent state across episodes.
        Used by simulate_multi_episode at every episode boundary so that
        e.g. bandit arm parameters persist while per-episode noise is fresh.
        Default implementation handles RNG only; subclasses generally
        do NOT need to override this.
        """
        self.seed += 1
        self.random = np.random.default_rng(seed = self.seed)
