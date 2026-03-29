import numpy as np
import pytest
from ibrl.simulators import simulate
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.environments import BanditEnvironment, NewcombEnvironment


class TestSimulator:
    """Parametrized simulator tests"""
    
    @pytest.mark.parametrize("agent_class,env_class", [
        pytest.param(QLearningAgent, BanditEnvironment, id="qlearn_bandit"),
        pytest.param(BayesianAgent, BanditEnvironment, id="bayesian_bandit"),
        pytest.param(EXP3Agent, BanditEnvironment, id="exp3_bandit"),
        pytest.param(QLearningAgent, NewcombEnvironment, id="qlearn_newcomb"),
    ])
    def test_simulate_runs_without_error(self, agent_class, env_class, num_actions, seed):
        env = env_class(num_actions=num_actions, seed=seed)
        agent = agent_class(num_actions=num_actions, seed=seed + 1)
        options = {"num_steps": 10, "num_runs": 2}
        results = simulate(env, agent, options)
        assert "average_reward" in results
        assert "optimal_reward" in results

    @pytest.mark.parametrize("num_steps,num_runs", [
        pytest.param(10, 2, id="small"),
        pytest.param(20, 3, id="medium"),
        pytest.param(5, 1, id="tiny"),
    ])
    def test_simulate_output_shapes(self, num_steps, num_runs, num_actions, seed):
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        agent = QLearningAgent(num_actions=num_actions, seed=seed + 1)
        options = {"num_steps": num_steps, "num_runs": num_runs}
        results = simulate(env, agent, options)
        
        assert results["average_reward"].shape == (2, num_steps)
        assert results["probabilities"].shape == (num_runs, num_steps, num_actions)
        assert results["actions"].shape == (num_runs, num_steps)
        assert results["rewards"].shape == (num_runs, num_steps)

    def test_simulate_rewards_valid(self, num_actions, seed):
        env = BanditEnvironment(num_actions=num_actions, seed=seed)
        agent = QLearningAgent(num_actions=num_actions, seed=seed + 1)
        options = {"num_steps": 10, "num_runs": 2}
        results = simulate(env, agent, options)
        
        assert np.all(np.isfinite(results["rewards"]))
        assert np.all(np.isfinite(results["average_reward"]))
