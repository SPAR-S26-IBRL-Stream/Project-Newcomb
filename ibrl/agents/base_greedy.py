import numpy as np

from . import BaseAgent
from ..exploration import EpsilonGreedy, ExplorationStrategy, Softmax


class BaseGreedyAgent(BaseAgent):
    """
    Base class for agents that use an exploration strategy over action values.

    Arguments:
        exploration_strategy: Optional strategy object for converting values to a policy
        epsilon:        For epsilon-greedy policy
        temperature:    For softmax policy
        decay_type:     Select formula for decreasing rate (0: exponential, 1: linear)

    Both epsilon and temperature can be either a fixed float or a tuple (start,decay constant,min) for decreasing exploration.
    """
    def __init__(self, *,
            exploration_strategy : ExplorationStrategy | None = None,
            epsilon     : float | tuple[float] | None = None,
            temperature : float | tuple[float] | None = None,
            decay_type  : float = 0,
            **kwargs):
        super().__init__(**kwargs)

        if exploration_strategy is not None and (epsilon is not None or temperature is not None):
            raise RuntimeError("Cannot specify exploration_strategy with epsilon or temperature")
        if exploration_strategy is None:
            if epsilon is not None and temperature is not None:
                raise RuntimeError("Cannot specify both epsilon and temperature")
            if epsilon is None and temperature is None:
                epsilon = 0.1  # default value

        assert epsilon is None or isinstance(epsilon,float) or (isinstance(epsilon,tuple) and len(epsilon)==3)
        assert temperature is None or isinstance(temperature,float) or (isinstance(temperature,tuple) and len(temperature)==3)

        self.epsilon = epsilon
        self.temperature = temperature
        self.decay_type = int(decay_type)
        assert self.decay_type in [0,1]
        if exploration_strategy is not None:
            self.exploration_strategy = exploration_strategy
        elif epsilon is not None:
            self.exploration_strategy = EpsilonGreedy(epsilon, self.decay_type)
        elif temperature is not None:
            self.exploration_strategy = Softmax(temperature, self.decay_type)
        else:
            raise RuntimeError("Invalid state")

    def build_greedy_policy(self, values : np.ndarray) -> np.ndarray:
        """
        Construct probabilities from reward estimates using the configured strategy.
        """
        return self.exploration_strategy.get_probabilities(self, values)

    def parse_parameter(self, parameter : float | tuple[float]) -> float:
        """
        Parse exploration parameter (epsilon or temperature)
        Parameter may either be a fixed value or tuple specifying temporal decay
        """
        if isinstance(parameter,float):
            return parameter
        if self.decay_type == 0:  # exponential decay
            start,rate,end = parameter
            return max(start / (self.step ** rate), end)
        if self.decay_type == 1:  # linear decay
            start,last_step,end = parameter
            return end if self.step>=last_step else (start + (end-start) * (self.step / last_step))
        raise RuntimeError("Invalid state")
