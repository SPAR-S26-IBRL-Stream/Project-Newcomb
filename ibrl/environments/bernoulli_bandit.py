import numpy as np

from . import BaseEnvironment


class BernoulliBanditEnvironment(BaseEnvironment):
    """
    Multi-armed bandit with Bernoulli (coin-flip) rewards.

    Each arm has a fixed probability p_i of paying reward 1; otherwise reward is 0.

    If probs are provided at construction, they are used as-is and persist across resets.
    Otherwise, probabilities are sampled uniformly from [0, 1] on each reset.
    """
    def __init__(self, probs=None, **kwargs):
        super().__init__(**kwargs)
        self._fixed_probs = np.array(probs, dtype=float) if probs is not None else None

    def _resolve(self, env_action: int | None, action: int) -> float:
        outcome = int(self.random.random() < self.probs[action])
        return float(outcome)

    def get_optimal_reward(self) -> float:
        return self.probs.max()

    def reset(self):
        super().reset()
        if self._fixed_probs is not None:
            self.probs = self._fixed_probs.copy()
        else:
            self.probs = self.random.random((self.num_actions,))
