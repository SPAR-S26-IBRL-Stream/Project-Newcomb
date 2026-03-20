import numpy as np
from numpy.typing import NDArray

from . import BaseEnvironment


class SwitchingAdversaryEnvironment(BaseEnvironment):
    """
    Classical bandit, where only one arm leads to a non-zero reward
    After a fixed number of interactions, the reward moves to a different arm
    """
    def __init__(self, *args,
            switch_at : int = None,
            **kwargs):
        super().__init__(*args, **kwargs)
        if switch_at is None:
            if self.num_steps is None:
                raise RuntimeError("SwitchingAdversaryEnvironment: require either switch_at or num_steps argument")
            switch_at = self.num_steps // 2
        self.switch_at = switch_at

    def _resolve(self, env_action : int | None, action : int) -> float:
        self._step_count += 1

        # At switch_at, the 'best' arm moves to the other side
        if self._step_count == self.switch_at:
            self.values = np.zeros((self.num_actions,))
            self.values[-1] = 1.0 # Move reward to the last arm

        return self.random.normal(self.values[action], 0.1)

    def get_optimal_reward(self) -> int:
        return 1.0 # The maximum reward is always 1.0

    def reset(self):
        super().reset()
        self._step_count = 0
        # Ensure Arm 0 is the best at the start
        self.values = np.zeros((self.num_actions,))
        self.values[0] = 1.0
