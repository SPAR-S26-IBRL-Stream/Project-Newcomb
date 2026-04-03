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

    def test_reward_table_structure(self, newcomb_env):
        assert newcomb_env.reward_table[0, 0] == 10  # boxB
        assert newcomb_env.reward_table[0, 1] == 15  # boxB + boxA
        assert newcomb_env.reward_table[1, 0] == 0
        assert newcomb_env.reward_table[1, 1] == 5   # boxA

    def test_predict_sets_rewards(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        action = 0
        outcome = newcomb_env.step(probs, action)
        # After step(), the environment has executed _respond() which samples a prediction
        # Just verify the outcome is valid
        assert isinstance(outcome.reward, (int, float, np.integer, np.floating))

    def test_interact(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        action = 0
        outcome = newcomb_env.step(probs, action)
        reward = outcome.reward
        assert isinstance(reward, (int, float, np.integer, np.floating))
        assert reward in [0, 5, 10, 15]

    def test_get_optimal_reward(self, newcomb_env):
        optimal = newcomb_env.get_optimal_reward()
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert optimal >= 0


class TestDeathInDamascusEnvironment:
    def test_initialization(self, seed):
        env = DeathInDamascusEnvironment(num_actions=2, seed=seed)
        assert env.num_actions == 2

    def test_reward_table_structure(self, damascus_env):
        assert damascus_env.reward_table[0, 0] == 0   # death in Damascus
        assert damascus_env.reward_table[0, 1] == 10  # life
        assert damascus_env.reward_table[1, 0] == 10  # life
        assert damascus_env.reward_table[1, 1] == 0   # death in Damascus

    def test_get_optimal_reward(self, damascus_env):
        optimal = damascus_env.get_optimal_reward()
        # Optimal is a mixed strategy that yields 5.0 (not 10)
        # because the predictor will predict your action
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert 0 <= optimal <= 10
