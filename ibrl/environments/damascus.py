import numpy as np

from . import BaseNewcombLikeEnvironment


class DeathInDamascusEnvironment(BaseNewcombLikeEnvironment):
    def __init__(self, *,
        death : float =  0,  # reward upon death
        life  : float = 10,  # reward upon survival
        **kwargs):
        super().__init__(reward_table=[
            [death, life ],
            [life,  death],
        ], **kwargs)
