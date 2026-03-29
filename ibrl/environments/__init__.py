from .base import BaseEnvironment
from .bandit import BanditEnvironment
from .switching import SwitchingAdversaryEnvironment
from .base_newcomb_like import BaseNewcombLikeEnvironment
from .newcomb import NewcombEnvironment
from .damascus import DeathInDamascusEnvironment
from .asymmetric_damascus import AsymmetricDeathInDamascusEnvironment
from .coordination import CoordinationGameEnvironment
from .policy_dependent_bandit import PolicyDependentBanditEnvironment
from .Coin_tossing_game_envs import MatchEnvironment, ReverseTailsEnvironment

__all__ = [
    "BaseEnvironment",
    "BanditEnvironment",
    "SwitchingAdversaryEnvironment",
    "BaseNewcombLikeEnvironment",
    "NewcombEnvironment",
    "DeathInDamascusEnvironment",
    "AsymmetricDeathInDamascusEnvironment",
    "CoordinationGameEnvironment",
    "PolicyDependentBanditEnvironment",
    "MatchEnvironment",
    "ReverseTailsEnvironment"
]
