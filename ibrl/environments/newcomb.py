import numpy as np

from . import BaseNewcombLikeEnvironment


class NewcombEnvironment(BaseNewcombLikeEnvironment):
    def __init__(self, *,
        boxA : float =  5,  # guaranteed content of first box
        boxB : float = 10,  # conditional content of second box
        **kwargs):
        super().__init__(reward_table=[
            [boxB, boxB+boxA],
            [0,    boxA     ]
        ], **kwargs)
