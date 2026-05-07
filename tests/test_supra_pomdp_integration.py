"""Tier 2 integration tests: SupraPOMDPWorldModel + Infradistribution.

These mirror the behavioural anchors in test_ib_world_model.py but built
around supra-POMDP hypotheses. They verify that the one-step
compute_expected_reward drives the IB update correctly.
"""
import numpy as np
import pytest

from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel
from ibrl.outcome import Outcome


# ── Shared fixtures ───────────────────────────────────────────────────────────

NUM_OBS = 2

def _make_wm(num_states=2, num_actions=2, num_obs=NUM_OBS, discount=0.9):
    return SupraPOMDPWorldModel(num_states=num_states, num_actions=num_actions,
                                 num_obs=num_obs, discount=discount)


def _identity_params(wm, theta_0_vec):
    """Identity T, identity B, specified θ₀, zero R. Use for single-step tests."""
    T = np.eye(wm.num_states)[np.newaxis, :, :].repeat(wm.num_actions, axis=0)
    T = T.transpose(1, 0, 2)
    B = np.eye(wm.num_states)[:, :wm.num_obs]
    theta_0 = np.array(theta_0_vec, dtype=float)
    R = np.zeros((wm.num_states, wm.num_actions, wm.num_states))
    return wm.make_params(T, B, theta_0, R)


def _noisy_params(wm, theta_0_vec, accuracy=0.85):
    """Identity T, noisy B (85% accurate), specified θ₀.

    Both observations are reachable from both states, so arbitrary observation
    sequences are physically consistent and the IB update assertion
    (probability > 0) is always satisfied across multiple steps.
    """
    T = np.eye(wm.num_states)[np.newaxis, :, :].repeat(wm.num_actions, axis=0)
    T = T.transpose(1, 0, 2)
    noise = (1. - accuracy) / (wm.num_obs - 1)
    B = np.full((wm.num_states, wm.num_obs), noise)
    for i in range(min(wm.num_states, wm.num_obs)):
        B[i, i] = accuracy
    theta_0 = np.array(theta_0_vec, dtype=float)
    R = np.zeros((wm.num_states, wm.num_actions, wm.num_states))
    return wm.make_params(T, B, theta_0, R)


def make_supra_pomdp_dist(num_hypotheses=3):
    """Uniform prior over hypotheses with differing θ₀ distributions.

    Uses noisy B so the alternating observation sequence in multi-step
    tests is always physically consistent (probability > 0 at each step).
    """
    wm = _make_wm()
    thetas = np.linspace(0., 1., num_hypotheses)
    hypotheses = [
        Infradistribution([AMeasure(_noisy_params(wm, [t, 1. - t]))],
                          world_model=wm)
        for t in thetas
    ]
    return Infradistribution.mix(hypotheses,
                                  np.ones(num_hypotheses) / num_hypotheses)


def obs(o: int, reward: float = 0.) -> Outcome:
    return Outcome(reward=reward, observation=o)


# Reward function: obs 0 → 0, obs 1 → 1 (used for IB update normalisation)
REWARD = np.array([0., 1.])
ACTION = 0


# ── §4.1  Normalisation conditions ───────────────────────────────────────────

def test_e_h_zero_is_zero_supra_pomdp():
    """E_H(rf=0) == 0 at initialisation."""
    dist = make_supra_pomdp_dist()
    rf_zero = np.zeros(NUM_OBS)
    assert abs(dist.evaluate_action(rf_zero, ACTION, None)) < 1e-9


def test_e_h_zero_stays_zero_after_updates():
    """E_H(rf=0) == 0 is preserved after multiple observations."""
    dist = make_supra_pomdp_dist()
    rf_2d = np.tile(REWARD, (2, 1))
    for o in [1, 0, 1, 1, 0]:
        dist.update(rf_2d, obs(o), ACTION, None)
    rf_zero = np.zeros(NUM_OBS)
    assert abs(dist.evaluate_action(rf_zero, ACTION, None)) < 1e-9


def test_evaluate_action_in_unit_interval():
    """evaluate_action ∈ [0, 1] for rf in [0, 1]."""
    dist = make_supra_pomdp_dist()
    v = dist.evaluate_action(REWARD, ACTION, None)
    assert 0. <= v <= 1.


def test_scale_and_offset_nonnegative_after_updates():
    """scale ≥ 0 and offset ≥ 0 for all a-measures after updates."""
    dist = make_supra_pomdp_dist()
    rf_2d = np.tile(REWARD, (2, 1))
    for o in [1, 0, 1, 1, 0, 0, 1]:
        dist.update(rf_2d, obs(o), ACTION, None)
    for m in dist.measures:
        assert m.scale >= 0.
        assert m.offset >= 0.


# ── §4.2  Mixing ──────────────────────────────────────────────────────────────

def test_mix_pessimistic_supra_pomdp():
    """Two POMDP hypotheses with different reward distributions.

    The IB pessimistic evaluation returns the minimum, not the average.
    """
    wm = _make_wm()
    # High-reward hypothesis: θ₀=[1,0] → always obs=0 → rf[0]=0 (no reward)
    p_low = _identity_params(wm, [1., 0.])
    # Low-reward hypothesis: θ₀=[0,1] → always obs=1 → rf[1]=1 (high reward)
    p_high = _identity_params(wm, [0., 1.])

    d_low  = Infradistribution([AMeasure(p_low)],  world_model=wm)
    d_high = Infradistribution([AMeasure(p_high)], world_model=wm)

    dist_bayesian = Infradistribution.mix([d_low, d_high], np.array([0.5, 0.5]))
    # Bayesian average: 0.5*0 + 0.5*1 = 0.5
    avg_val = dist_bayesian.evaluate_action(REWARD, ACTION, None)
    assert np.isclose(avg_val, 0.5, atol=1e-6)

    dist_ku = Infradistribution.mixKU([d_low, d_high])
    # KU pessimistic: min(0, 1) = 0
    ku_val = dist_ku.evaluate_action(REWARD, ACTION, None)
    assert np.isclose(ku_val, 0.0, atol=1e-6)


# ── §4.3  Update rule with credal reweighting ─────────────────────────────────

def test_update_concentrates_on_correct_hypothesis():
    """After repeated updates consistent with component 0, weight on it → 1."""
    wm = _make_wm()
    # Component 0: always in state 0 → obs=0 certain
    # Component 1: always in state 1 → obs=1 certain
    p0 = _identity_params(wm, [1., 0.])
    p1 = _identity_params(wm, [0., 1.])
    d0 = Infradistribution([AMeasure(p0)], world_model=wm)
    d1 = Infradistribution([AMeasure(p1)], world_model=wm)
    dist = Infradistribution.mix([d0, d1], np.array([0.5, 0.5]))
    rf_2d = np.tile(REWARD, (2, 1))

    # Feed observations consistent with obs=1 (component 1 is truth)
    for _ in range(20):
        dist.update(rf_2d, obs(1), ACTION, None)

    # evaluate_action should now reflect mostly component 1's view
    val = dist.evaluate_action(REWARD, ACTION, None)
    # Component 1 always gives obs=1, so rf[1]=1 → expected reward → 1
    assert val > 0.9, f"expected val → 1 after concentrating, got {val}"


def test_update_does_not_crash_at_zero_likelihood_step():
    """Observations with P=0 under one component: other components stay valid."""
    wm = _make_wm()
    p0 = _identity_params(wm, [1., 0.])  # obs=0 certain
    p1 = _identity_params(wm, [0., 1.])  # obs=1 certain
    d0 = Infradistribution([AMeasure(p0)], world_model=wm)
    d1 = Infradistribution([AMeasure(p1)], world_model=wm)
    dist = Infradistribution.mix([d0, d1], np.array([0.5, 0.5]))
    rf_2d = np.tile(REWARD, (2, 1))
    # After one obs=1, the IB update should not crash even though p0 gives P=0
    dist.update(rf_2d, obs(1), ACTION, None)
    val = dist.evaluate_action(REWARD, ACTION, None)
    assert 0. <= val <= 1.


# ── §4.4  Behavioural anchors ─────────────────────────────────────────────────

def test_supra_pomdp_evaluate_action_in_unit_interval_after_updates():
    """evaluate_action stays in [0, 1] after many varied updates."""
    dist = make_supra_pomdp_dist()
    rf_2d = np.tile(REWARD, (2, 1))
    rng = np.random.default_rng(7)
    for _ in range(15):
        o = int(rng.integers(2))
        dist.update(rf_2d, obs(o), ACTION, None)
    v = dist.evaluate_action(REWARD, ACTION, None)
    assert 0. <= v <= 1.


def test_supra_pomdp_higher_reward_after_favourable_observations():
    """evaluate_action increases after observations consistent with high reward."""
    dist = make_supra_pomdp_dist()
    rf_2d = np.tile(REWARD, (2, 1))
    ev_before = dist.evaluate_action(REWARD, ACTION, None)

    # obs=1 has reward 1; repeated obs=1 should push evaluation up
    for _ in range(15):
        dist.update(rf_2d, obs(1), ACTION, None)
    ev_after = dist.evaluate_action(REWARD, ACTION, None)
    assert ev_after > ev_before, (
        f"expected evaluate_action to increase, got {ev_before} → {ev_after}")
