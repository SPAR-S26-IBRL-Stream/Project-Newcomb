import numpy as np
import pytest
from ibrl.environments import (
    BanditEnvironment, NewcombEnvironment, DeathInDamascusEnvironment,
    CoordinationGameEnvironment, PolicyDependentBanditEnvironment
)


class TestNewcombEnvironment:
    def test_initialization(self, seed):
        env = NewcombEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2
        assert env.reward_table.shape == (2, 2)

    def test_predict_sets_rewards(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        action = 0
        outcome = newcomb_env.step(probs, action)
        assert isinstance(outcome.reward, (int, float, np.integer, np.floating))

    def test_interact(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        action = 0
        outcome = newcomb_env.step(probs, action)
        reward = outcome.reward
        assert isinstance(reward, (int, float, np.integer, np.floating))
        assert reward in newcomb_env.reward_table.flatten()

    def test_get_optimal_reward(self, newcomb_env):
        optimal = newcomb_env.get_optimal_reward()
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert optimal >= newcomb_env.reward_table.min()


class TestDeathInDamascusEnvironment:
    def test_initialization(self, seed):
        env = DeathInDamascusEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_get_optimal_reward(self, damascus_env):
        optimal = damascus_env.get_optimal_reward()
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert damascus_env.reward_table.min() <= optimal <= damascus_env.reward_table.max()
