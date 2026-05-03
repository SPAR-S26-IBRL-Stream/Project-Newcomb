import numpy as np

from ibrl.infrabayesian import AMeasure, Infradistribution
from ibrl.infrabayesian.builders.trap_bandit import (
    OUTCOME_CATASTROPHE,
    OUTCOME_ONE,
    trap_bandit_probs,
)
from ibrl.infrabayesian.world_models.joint_bandit_world_model import (
    JointBanditComponent,
    JointBanditWorldModel,
)
from ibrl.outcome import Outcome


REWARD_FUNCTION = np.array([
    [0.0, 1.0, -1000.0],
    [0.0, 1.0, -1000.0],
])


def test_safe_component_arm_probabilities():
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    component = JointBanditComponent(
        trap_bandit_probs(np.array([0.7, 0.3]), world_type="safe", p_cat=0.01)
    )

    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 0),
        [0.3, 0.7, 0.0],
    )
    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 1),
        [0.7, 0.3, 0.0],
    )


def test_risky_component_traps_higher_bias_arm():
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    component = JointBanditComponent(
        trap_bandit_probs(np.array([0.7, 0.3]), world_type="risky", p_cat=0.01),
        metadata={"trapped_arm": 0},
    )

    assert component.metadata["trapped_arm"] == 0
    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 0),
        [0.29, 0.7, 0.01],
    )
    np.testing.assert_allclose(
        wm.component_outcome_probs(component, 1),
        [0.7, 0.3, 0.0],
    )


def test_event_index_uses_observation_not_reward():
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    outcome = Outcome(reward=-1000.0, observation=OUTCOME_CATASTROPHE)
    assert wm.event_index(outcome, action=0) == OUTCOME_CATASTROPHE


def test_posterior_weights_shift_after_observation():
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    low = JointBanditComponent(
        trap_bandit_probs(np.array([0.2, 0.2]), world_type="safe", p_cat=0.01)
    )
    high = JointBanditComponent(
        trap_bandit_probs(np.array([0.8, 0.2]), world_type="safe", p_cat=0.01)
    )
    params = wm.make_params([low, high], np.array([0.5, 0.5]))
    state = wm.initial_state()

    for _ in range(5):
        state = wm.update_state(
            state,
            Outcome(reward=1.0, observation=OUTCOME_ONE),
            action=0,
            policy=np.array([0.5, 0.5]),
        )

    weights = wm.get_posterior_component_weights(state, params)
    assert weights[1] > weights[0]


def test_mix_params_preserves_joint_components():
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    p0 = wm.make_params([
        JointBanditComponent(
            trap_bandit_probs(np.array([0.2, 0.8]), world_type="safe", p_cat=0.01)
        )
    ])
    p1 = wm.make_params([
        JointBanditComponent(
            trap_bandit_probs(np.array([0.2, 0.8]), world_type="risky", p_cat=0.01)
        )
    ])

    mixed = wm.mix_params([p0, p1], np.array([0.75, 0.25]))

    assert len(mixed.components) == 2
    np.testing.assert_allclose(mixed.weights, [0.75, 0.25])


def test_bayesian_mix_averages_and_ku_takes_min():
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    safe = Infradistribution([
        AMeasure(wm.make_params([
            JointBanditComponent(
                trap_bandit_probs(np.array([0.7, 0.3]), world_type="safe", p_cat=0.01)
            )
        ]))
    ], world_model=wm)
    risky = Infradistribution([
        AMeasure(wm.make_params([
            JointBanditComponent(
                trap_bandit_probs(np.array([0.7, 0.3]), world_type="risky", p_cat=0.01)
            )
        ]))
    ], world_model=wm)

    bayes = Infradistribution.mix([safe, risky], np.array([0.5, 0.5]))
    ku = Infradistribution.mixKU([safe, risky])

    bayes_value = bayes.evaluate_action(REWARD_FUNCTION[0], 0, np.array([0.5, 0.5]))
    ku_value = ku.evaluate_action(REWARD_FUNCTION[0], 0, np.array([0.5, 0.5]))

    assert np.isclose(bayes_value, (0.7 + (0.7 - 10.0)) / 2)
    assert np.isclose(ku_value, 0.7 - 10.0)


def test_ku_update_removes_measure_that_makes_event_impossible():
    wm = JointBanditWorldModel(num_arms=2, num_outcomes=3)
    safe = Infradistribution([
        AMeasure(wm.make_params([
            JointBanditComponent(
                trap_bandit_probs(np.array([0.7, 0.3]), world_type="safe", p_cat=0.01)
            )
        ]))
    ], world_model=wm)
    risky = Infradistribution([
        AMeasure(wm.make_params([
            JointBanditComponent(
                trap_bandit_probs(np.array([0.7, 0.3]), world_type="risky", p_cat=0.01)
            )
        ]))
    ], world_model=wm)
    ku = Infradistribution.mixKU([safe, risky])

    ku.update(
        REWARD_FUNCTION,
        Outcome(reward=-1000.0, observation=OUTCOME_CATASTROPHE),
        action=0,
        policy=np.array([0.5, 0.5]),
    )

    assert len(ku.measures) == 1
    assert ku.world_model.compute_likelihood(
        ku.belief_state,
        Outcome(reward=-1000.0, observation=OUTCOME_CATASTROPHE),
        ku.measures[0].params,
        action=0,
        policy=np.array([0.5, 0.5]),
    ) > 0
