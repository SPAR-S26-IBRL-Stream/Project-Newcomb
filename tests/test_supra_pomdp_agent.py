"""Behavioural tests for SupraPOMDPAgent.

These tests describe the intended repaired design:

- SupraPOMDP policies are represented as BeliefPolicy objects.
- One-state/flat policies are one-row BeliefPolicy objects.
- Belief-aware control queries the current belief, not an averaged policy.
- Policy-dependent Newcomb kernels see the committed BeliefPolicy table.
"""
import numpy as np

from ibrl.agents.supra_pomdp_agent import BeliefPolicy, SupraPOMDPAgent
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel


def _hypothesis(wm, params):
    return Infradistribution([AMeasure(params)], world_model=wm)


def test_supra_pomdp_single_state_uses_one_row_belief_policy():
    num_actions = 2
    wm = SupraPOMDPWorldModel(num_states=1, num_actions=num_actions, num_obs=1)
    T = np.ones((1, num_actions, 1))
    B = np.ones((1, 1))
    theta_0 = np.array([1.0])
    R = np.zeros((1, num_actions, 1))
    R[0, 0, 0] = 1.0
    R[0, 1, 0] = 0.0
    params = wm.make_params(T, B, theta_0, R)

    agent = SupraPOMDPAgent(
        num_actions=num_actions,
        hypotheses=[_hypothesis(wm, params)],
        prior=np.array([1.0]),
        policy_discretisation=0,
        exploration_prefix=0,
    )

    agent.reset()

    assert isinstance(agent.current_policy, BeliefPolicy)
    assert agent.current_policy.policy_table.shape == (1, num_actions)
    np.testing.assert_allclose(agent.get_probabilities(), agent.current_policy.policy_table[0])
    assert agent.get_probabilities()[0] > agent.get_probabilities()[1]


def test_supra_pomdp_tiger_policy_depends_on_belief():
    num_states = 2
    num_actions = 3
    wm = SupraPOMDPWorldModel(num_states=num_states, num_actions=num_actions,
                               num_obs=2, discount=0.95)
    # Actions: 0=listen, 1=open_left, 2=open_right.
    T = np.zeros((num_states, num_actions, num_states))
    T[:, 0, :] = np.eye(num_states)       # listen preserves state
    T[:, 1, :] = 0.5                      # opening resets
    T[:, 2, :] = 0.5
    B = np.array([[0.85, 0.15], [0.15, 0.85]])
    theta_0 = np.array([0.5, 0.5])
    R = np.zeros((num_states, num_actions, num_states))
    R[:, 0, :] = -1.0                     # listen cost
    R[0, 1, :] = -100.0                   # open left with tiger left
    R[1, 1, :] = 10.0
    R[0, 2, :] = 10.0
    R[1, 2, :] = -100.0                   # open right with tiger right
    params = wm.make_params(T, B, theta_0, R)

    agent = SupraPOMDPAgent(
        num_actions=num_actions,
        hypotheses=[_hypothesis(wm, params)],
        prior=np.array([1.0]),
        policy_discretisation=5,
        exploration_prefix=0,
    )

    agent.reset()

    uncertain = agent.get_belief_probabilities(np.array([0.5, 0.5]))
    certain_left = agent.get_belief_probabilities(np.array([0.99, 0.01]))
    certain_right = agent.get_belief_probabilities(np.array([0.01, 0.99]))

    assert uncertain[0] > uncertain[1]
    assert uncertain[0] > uncertain[2]
    assert certain_left[2] > certain_left[0]
    assert certain_right[1] > certain_right[0]


def test_supra_pomdp_transparent_newcomb_commits_with_belief_policy():
    eps = 0.05
    num_actions = 2
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=num_actions,
                               num_obs=2, discount=0.9)

    def theta_0_fn(policy: BeliefPolicy):
        p_one_box = (1 - eps) * policy.policy_table[0, 0] + eps * 0.5
        return np.array([p_one_box, 1.0 - p_one_box])

    T = np.zeros((2, num_actions, 2))
    T[0, :, 0] = 1.0
    T[1, :, 1] = 1.0
    B = np.eye(2)
    R = np.zeros((2, num_actions, 2))
    R[0, 0, 0] = 1.0    # one-box on full
    R[0, 1, 0] = 1.1    # two-box on full
    R[1, 1, 1] = 0.1    # two-box on empty
    params = wm.make_params(T, B, theta_0_fn, R)

    agent = SupraPOMDPAgent(
        num_actions=num_actions,
        hypotheses=[_hypothesis(wm, params)],
        prior=np.array([1.0]),
        policy_discretisation=10,
        exploration_prefix=0,
    )

    agent.reset()
    action_probs = agent.get_probabilities()

    assert isinstance(agent.current_policy, BeliefPolicy)
    assert action_probs[0] > action_probs[1]
    assert action_probs[0] > 0.7
