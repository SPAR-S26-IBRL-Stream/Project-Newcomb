import numpy as np
from numpy.typing import NDArray

from . import BaseAgent


class BaseGreedyAgent(BaseAgent):
    """
    Base class for agents that use either an epsilon-greedy or softmax policy to encourage exploration.

    Arguments:
        epsilon:        For epsilon-greedy policy
        temperature:    For softmax policy
        decay_type:     Select formula for decreasing rate (0: exponential, 1: linear)

    Both epsilon and temperature can be either a fixed float or a tuple (start,decay constant,min) for decreasing exploration.
    """
    def __init__(self, *args,
            epsilon     : float | tuple[float] | None = None,
            temperature : float | tuple[float] | None = None,
            decay_type  : float = 0,
            **kwargs):
        super().__init__(*args, **kwargs)

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

    def build_greedy_policy(self, values : NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Construct probabilities based on given reward estimates and selected policy
        """
        if self.epsilon is not None:
            return self.build_epsilon_greedy_policy(values)
        if self.temperature is not None:
            return self.build_softmax_policy(values)
        raise RuntimeError("Invalid state")

    def build_epsilon_greedy_policy(self, values : NDArray[np.float64]) -> NDArray[np.float64]:
        # Exploitation: sample uniformly across actions with highest value
        best_actions = np.isclose(values, values.max())
        exploit = np.ones_like(values)*best_actions / best_actions.sum()

        # Exploration: sample uniformly across all actions
        explore = np.ones_like(values) / self.num_actions

        epsilon = self.parse_parameter(self.epsilon)
        return (1 - epsilon) * exploit + epsilon * explore

    def build_softmax_policy(self, values : NDArray[np.float64]) -> NDArray[np.float64]:
        temperature = self.parse_parameter(self.temperature)

        # Numerically stable softmax
        exp = np.exp((values - values.max()) / temperature)
        return exp / exp.sum()

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
