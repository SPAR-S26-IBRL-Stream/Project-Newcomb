import numpy as np
from numpy.typing import NDArray

from . import BaseEnvironment


class BanditEnvironment(BaseEnvironment):
    """
    Multi-armed bandit environment

    Each action is associated with a fixed probability p. Upon taking an action, the reward is sampled
    from a Bernoulli distribution with that probability (i.e. reward is 1 with probability p, 0 otherwise).
    Upon initialisation, the probabilities are sampled uniformly from [0, 1].
    """
    def _resolve(self, env_action : int | None, action : int) -> float:
        return float(self.random.random() < self.rewards[action])

    def get_optimal_reward(self) -> float:
        return self.rewards.max()

    def reset(self):
        super().reset()
        self.rewards = self.random.random((self.num_actions,))
