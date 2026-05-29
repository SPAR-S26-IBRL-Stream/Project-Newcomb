import numpy as np

from . import BaseAgent
from ..exploration import EpsilonGreedy, ExplorationStrategy


class BaseGreedyAgent(BaseAgent):
    """
    Base class for agents that use an exploration strategy over action values.

    Arguments:
        exploration_strategy: Optional strategy object for converting values to a policy
    """
    def __init__(self, *,
            exploration_strategy : ExplorationStrategy | None = None,
            **kwargs):
        super().__init__(**kwargs)
        if exploration_strategy is None:
            exploration_strategy = EpsilonGreedy(0.1)
        self.exploration_strategy = exploration_strategy

    def build_greedy_policy(self, values : np.ndarray) -> np.ndarray:
        """
        Construct probabilities from reward estimates using the configured strategy.
        """
        return self.exploration_strategy.get_probabilities(self, values)
