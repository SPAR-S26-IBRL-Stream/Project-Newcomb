from .base import WorldModel
from .bernoulli_world_model import MultiBernoulliWorldModel
from .newcomb_world_model import NewcombWorldModel
from .supra_pomdp_world_model import SupraPOMDPWorldModel  # NEW

__all__ = [
    "WorldModel",
    "MultiBernoulliWorldModel",
    "NewcombWorldModel",
    "SupraPOMDPWorldModel",  # NEW
]
