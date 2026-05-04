import pytest
import numpy as np
from ibrl.agents import QLearningAgent, BayesianAgent, EXP3Agent
from ibrl.environments import (
    BanditEnvironment, NewcombEnvironment, DeathInDamascusEnvironment,
    CoordinationGameEnvironment, PolicyDependentBanditEnvironment
)


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def num_actions():
    return 2


@pytest.fixture
def num_steps():
    return 10


@pytest.fixture
def q_learning_agent(num_actions, seed):
    agent = QLearningAgent(num_actions=num_actions, seed=seed)
    agent.reset()
    return agent


@pytest.fixture
def bayesian_agent(num_actions, seed):
    agent = BayesianAgent(num_actions=num_actions, seed=seed)
    agent.reset()
    return agent


@pytest.fixture
def exp3_agent(num_actions, seed):
    agent = EXP3Agent(num_actions=num_actions, seed=seed)
    agent.reset()
    return agent


@pytest.fixture
def bandit_env(num_actions, seed):
    env = BanditEnvironment(num_actions=num_actions, seed=seed)
    env.reset()
    return env


@pytest.fixture
def newcomb_env(seed):
    env = NewcombEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env


@pytest.fixture
def damascus_env(seed):
    env = DeathInDamascusEnvironment(num_actions=2, seed=seed)
    env.reset()
    return env
    
# ── SupraPOMDP Integration Test Fixtures ─────────────────────────────────

@pytest.fixture
def supra_pomdp_dist():
    """Fixture: SupraPOMDP Infradistribution for integration tests."""
    from ibrl.infrabayesian.a_measure import AMeasure
    from ibrl.infrabayesian.infradistribution import Infradistribution
    from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel
    
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    B = np.array([[0.85, 0.15], [0.15, 0.85]])
    R = np.zeros((2, 1, 2))
    
    thetas = np.linspace(0., 1., 3)
    hypotheses = [
        Infradistribution([AMeasure(wm.make_params(T, B, np.array([t, 1. - t]), R))],
                          world_model=wm)
        for t in thetas
    ]
    return Infradistribution.mix(hypotheses, np.ones(3) / 3)


@pytest.fixture
def supra_pomdp_world_model():
    """Fixture: Generic SupraPOMDPWorldModel for world model tests."""
    from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel
    return SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    
    
    
    
@pytest.fixture
def tiger_pomdp_params(supra_pomdp_world_model):
    """Fixture: Tiger POMDP parameters for testing."""
    wm = supra_pomdp_world_model
    
    # Extend to 3 actions for Tiger
    # TODO: This is a workaround; should create dedicated Tiger POMDP fixture
    wm_tiger = SupraPOMDPWorldModel(num_states=2, num_actions=3, num_obs=2, discount=0.95)
    
    T = np.zeros((2, 3, 2))
    T[:, 0, :] = 0.5  # Open left: reset
    T[:, 1, :] = 0.5  # Open right: reset
    T[0, 2, 0] = 1.0  # Listen: preserve tiger left
    T[1, 2, 1] = 1.0  # Listen: preserve tiger right
    
    B = np.array([[0.85, 0.15], [0.15, 0.85]])
    theta_0 = np.array([0.5, 0.5])
    
    R = np.zeros((2, 3, 2))
    R[:, 0, 0] = -100.; R[:, 0, 1] = 10.   # Open left
    R[:, 1, 1] = -100.; R[:, 1, 0] = 10.   # Open right
    R[:, 2, :] = -1.                        # Listen
    
    return wm_tiger.make_params(T, B, theta_0, R)


@pytest.fixture
def belief_discretisation_grid():
    """Fixture: Simplex grid for 2-state belief discretization."""
    from ibrl.utils.belief_discretization import simplex_grid
    return simplex_grid(num_states=2, num_points=3)


@pytest.fixture
def belief_discretisation_corners():
    """Fixture: Corner beliefs (fully observable states) for 2-state space."""
    from ibrl.utils.belief_discretization import corner_beliefs
    return corner_beliefs(num_states=2)
