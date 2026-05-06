import numpy as np
from numpy.typing import NDArray

from . import BaseAgent
from ..utils import dump_array


class ThompsonSamplingAgent(BaseAgent):
    """
    Thompson sampling for Gaussian-reward bandits.

    Reuses the Normal-Normal precision-weighted posterior from BayesianAgent
    (same `update()` math), but selects actions by drawing one sample from
    the posterior of each arm and acting greedily on the samples — i.e. the
    classical posterior-sampling RL baseline.

    Probability output is one-hot on the sampled argmax (deterministic
    given the per-call posterior draw, which itself is RNG-driven via
    self.random for reproducibility).
    """

    def get_probabilities(self) -> NDArray[np.float64]:
        sigma = 1.0 / np.sqrt(self.precision)
        sample = self.random.normal(self.values, sigma)
        a = int(np.argmax(sample))
        probs = np.zeros(self.num_actions)
        probs[a] = 1.0
        return probs

    def update(self, probabilities: NDArray[np.float64], action: int, reward: float):
        super().update(probabilities, action, reward)
        # Same precision-weighted Normal-Normal posterior as BayesianAgent
        self.values[action] = (self.precision[action] * self.values[action] + reward) / (self.precision[action] + 1.0)
        self.precision[action] += 1

    def reset_belief(self):
        super().reset_belief()
        self.values = np.zeros(self.num_actions)
        self.precision = np.ones(self.num_actions) * 0.1

    def dump_state(self) -> str:
        return dump_array(self.values)
