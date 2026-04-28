import numpy as np

from . import BaseNewcombLikeEnvironment


class CoordinationGameEnvironment(BaseNewcombLikeEnvironment):
    def __init__(self, *,
            rewardA : float = 2,  # reward in first equilibrium
            rewardB : float = 1,  # reward in second equilibrium
            **kwargs):
        super().__init__(reward_table=[
            [rewardA, 0      ],
            [0,       rewardB],
        ], **kwargs)
