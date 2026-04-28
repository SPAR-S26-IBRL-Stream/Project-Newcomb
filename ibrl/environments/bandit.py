import numpy as np

from . import BaseEnvironment


class BanditEnvironment(BaseEnvironment):
    """
    Multi-armed bandit environment

    Each action is associated with a fixed average reward. Upon taking an action, the reward is sampled randomly
    according to a normal distribution centred on the average value.
    Upon initialisation, the average rewards are sampled from a standard normal distribution.
    """
    def _resolve(self, env_action : int | None, action : int) -> float:
        return self.random.normal(self.rewards[action], 1)

    def get_optimal_reward(self) -> float:
        return self.rewards.max()

    def reset(self):
        super().reset()
        self.rewards = self.random.standard_normal((self.num_actions,))
