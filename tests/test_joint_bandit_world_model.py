import numpy as np

from ibrl.infrabayesian import AMeasure, Infradistribution
from ibrl.infrabayesian.world_models.joint_bandit_world_model import (
    OUTCOME_CATASTROPHE,
    OUTCOME_ONE,
    OUTCOME_ZERO,
    JointBanditComponent,
    JointBanditWorldModel,
)
from ibrl.outcome import Outcome


REWARD_FUNCTION = np.array([
    [0.0, 1.0, -1000.0],
    [0.0, 1.0, -1000.0],
])


def test_safe_component_arm_probabilities():
    wm = JointBanditWorldModel()
    component = JointBanditComponent("safe", p1=0.7, p2=0.3, p_cat=0.01)

    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 0),
        [0.3, 0.7, 0.0],
    )
    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 1),
        [0.7, 0.3, 0.0],
    )


def test_risky_component_traps_higher_bias_arm():
    wm = JointBanditWorldModel()
    component = JointBanditComponent("risky", p1=0.7, p2=0.3, p_cat=0.01)

    assert wm.trapped_arm(component) == 0
    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 0),
        [0.29, 0.7, 0.01],
    )
    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 1),
        [0.7, 0.3, 0.0],
    )


def test_event_index_uses_observation_not_reward():
    wm = JointBanditWorldModel()
    outcome = Outcome(reward=-1000.0, observation=OUTCOME_CATASTROPHE)
    assert wm.event_index(outcome, action=0) == OUTCOME_CATASTROPHE


def test_posterior_weights_shift_after_observation():
    wm = JointBanditWorldModel()
    low = JointBanditComponent("safe", p1=0.2, p2=0.2, p_cat=0.01)
    high = JointBanditComponent("safe", p1=0.8, p2=0.2, p_cat=0.01)
    params = wm.make_params([low, high], np.array([0.5, 0.5]))
    state = wm.initial_state()

    for _ in range(5):
        state = wm.update_state(
            state,
            Outcome(reward=1.0, observation=OUTCOME_ONE),
            action=0,
            policy=np.array([0.5, 0.5]),
        )

    weights = wm.posterior_component_weights(state, params)
    assert weights[1] > weights[0]


def test_mix_params_preserves_joint_components():
    wm = JointBanditWorldModel()
    p0 = wm.make_params([JointBanditComponent("safe", 0.2, 0.8, 0.01)])
    p1 = wm.make_params([JointBanditComponent("risky", 0.2, 0.8, 0.01)])

    mixed = wm.mix_params([p0, p1], np.array([0.75, 0.25]))

    assert len(mixed.components) == 2
    np.testing.assert_allclose(mixed.weights, [0.75, 0.25])


def test_bayesian_mix_averages_and_ku_takes_min():
    wm = JointBanditWorldModel()
    safe = Infradistribution([
        AMeasure(wm.make_params([JointBanditComponent("safe", 0.7, 0.3, 0.01)]))
    ], world_model=wm)
    risky = Infradistribution([
        AMeasure(wm.make_params([JointBanditComponent("risky", 0.7, 0.3, 0.01)]))
    ], world_model=wm)

    bayes = Infradistribution.mix([safe, risky], np.array([0.5, 0.5]))
    ku = Infradistribution.mixKU([safe, risky])

    bayes_value = bayes.evaluate_action(REWARD_FUNCTION[0], 0, np.array([0.5, 0.5]))
    ku_value = ku.evaluate_action(REWARD_FUNCTION[0], 0, np.array([0.5, 0.5]))

    assert np.isclose(bayes_value, (0.7 + (0.7 - 10.0)) / 2)
    assert np.isclose(ku_value, 0.7 - 10.0)
