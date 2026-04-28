from . import BaseNewcombLikeEnvironment

#reward was formatted into reward tables (unlike in the initial code), in order to match the repo's structure

class MatchEnvironment(BaseNewcombLikeEnvironment):
    # First environment: reward 1 if action matches observation, 0 otherwise
    # 0 = H, 1 = T
    def __init__(self, **kwargs):
        reward_table = [
            [1.0, 0.0],  # if observation is H
            [0.0, 1.0],  # if observation is T
        ]
        super().__init__(reward_table=reward_table, **kwargs)


class ReverseTailsEnvironment(BaseNewcombLikeEnvironment):
    # Second environment:
    # if H: reward 1 for mismatch, 0 for match
    # if T: reward always 0.5
    # 0 = H, 1 = T
    def __init__(self, **kwargs):
        reward_table = [
            [0.0, 1.0],  # if observation is H
            [0.5, 0.5],  # if observation is T
        ]
        super().__init__(reward_table=reward_table, **kwargs)
