import time
import pytest
from ibrl.simulators import simulate
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.environments import BanditEnvironment, NewcombEnvironment


class TestSmokeTests:
    """Run training for 10 seconds to verify nothing is obviously broken"""
    
    @pytest.mark.timeout(15)
    def test_smoke_q_learning_bandit(self):
        env = BanditEnvironment(num_actions=2, seed=42)
        agent = QLearningAgent(num_actions=2, seed=43)
        options = {"num_steps": 100, "num_runs": 1}
        results = simulate(env, agent, options)
        assert results is not None

    @pytest.mark.timeout(15)
    def test_smoke_bayesian_newcomb(self):
        env = NewcombEnvironment(num_actions=2, seed=42)
        agent = BayesianAgent(num_actions=2, seed=43)
        options = {"num_steps": 100, "num_runs": 1}
        results = simulate(env, agent, options)
        assert results is not None

    @pytest.mark.timeout(15)
    def test_smoke_exp3_bandit(self):
        env = BanditEnvironment(num_actions=2, seed=42)
        agent = EXP3Agent(num_actions=2, seed=43)
        options = {"num_steps": 100, "num_runs": 1}
        results = simulate(env, agent, options)
        assert results is not None
