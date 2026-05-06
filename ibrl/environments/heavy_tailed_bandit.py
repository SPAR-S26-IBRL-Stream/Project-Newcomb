import numpy as np

from . import BaseEnvironment


class HeavyTailedBanditEnvironment(BaseEnvironment):
    """
    Multi-armed bandit with Cauchy-contaminated rewards — non-realizable
    for any agent whose likelihood family is Normal (the prior cannot
    place mass on a Cauchy-contaminated distribution; the misspecification
    is in the likelihood family itself, not in parameter values).

    Per arm k, reward is sampled as a mixture:
        with probability (1 - eps_k):  Normal(median_k, 1)
        with probability eps_k:        median_k + gamma_k * StandardCauchy

    Arm parameters (median_k, eps_k, gamma_k, plus a centering shift c_k
    used by the Cauchy term) are drawn once at full reset() and held
    fixed across episodes. reset_episode() re-seeds the RNG only.

    `get_optimal_reward()` returns the highest median, since means are
    undefined under Cauchy contamination.

    Arguments:
        contam_low, contam_high:  bounds on per-arm contamination probability
        gamma_low, gamma_high:    bounds on per-arm Cauchy scale
    """

    def __init__(self,
            num_actions: int,
            *,
            contam_low: float = 0.05,
            contam_high: float = 0.20,
            gamma_low: float = 2.0,
            gamma_high: float = 5.0,
            **kwargs):
        super().__init__(num_actions, **kwargs)
        assert 0.0 <= contam_low <= contam_high <= 1.0
        assert 0.0 < gamma_low <= gamma_high
        self.contam_low = float(contam_low)
        self.contam_high = float(contam_high)
        self.gamma_low = float(gamma_low)
        self.gamma_high = float(gamma_high)

    def reset(self):
        super().reset()
        self.medians = self.random.standard_normal(self.num_actions)
        self.eps = self.random.uniform(self.contam_low, self.contam_high, self.num_actions)
        self.gamma = self.random.uniform(self.gamma_low, self.gamma_high, self.num_actions)

    def interact(self, action: int) -> float:
        if self.random.random() < self.eps[action]:
            return float(self.medians[action] + self.gamma[action] * self.random.standard_cauchy())
        return float(self.random.normal(self.medians[action], 1.0))

    def get_optimal_reward(self) -> float:
        return float(self.medians.max())
