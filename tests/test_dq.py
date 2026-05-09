"""Tests for the AvgRewardLSTDDQEstimator (LCD/Dq implementation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from functools import partial

from xp_gym.estimators.dq import (
    available_cars_per_zone,
    AvgRewardLSTDDQEstimator,
    AvgRewardLSTDDQEstimatorState,
)
from xp_gym.observation import Observation


# ---------------------------------------------------------------------------
# Mock helpers – lightweight stand-ins for or_gymnax types
# ---------------------------------------------------------------------------

@struct.dataclass
class MockEvent:
    t: jnp.ndarray


@struct.dataclass
class MockEnvState:
    locations: jnp.ndarray
    times: jnp.ndarray
    event: MockEvent


@struct.dataclass
class MockDesignInfo:
    env_state: MockEnvState
    src_2_zone: jnp.ndarray
    n_spatial_zones: int


def _make_obs(env_state, src_2_zone, n_spatial_zones, action, reward):
    """Build a minimal Observation with the fields the DQ estimator reads."""
    design_info = MockDesignInfo(
        env_state=env_state,
        src_2_zone=src_2_zone,
        n_spatial_zones=n_spatial_zones,
    )
    return Observation(
        obs=jnp.array([0.0]),
        action=jnp.array(action, dtype=jnp.float32),
        reward=jnp.array(reward, dtype=jnp.float32),
        info=None,
        design_info=design_info,
    )


def _make_obs_with_state(state_vec, action, reward, n_zones=3):
    """Build an Observation whose available_cars_per_zone equals `state_vec`.

    We achieve this by placing one car per zone with appropriate locations,
    setting all cars idle (times <= t).
    """
    state_vec = jnp.array(state_vec, dtype=jnp.int32)
    # Create locations: for zone i, place state_vec[i] cars at node i
    locations = []
    for zone_id in range(n_zones):
        locations.extend([zone_id] * int(state_vec[zone_id]))
    if len(locations) == 0:
        locations = [0]  # need at least one car
    locations = jnp.array(locations, dtype=jnp.int32)
    n_cars = len(locations)

    # src_2_zone: node i -> zone i (identity mapping for first n_zones nodes)
    src_2_zone = jnp.arange(n_zones, dtype=jnp.int32)

    env_state = MockEnvState(
        locations=locations,
        times=jnp.zeros(n_cars, dtype=jnp.int32),  # all idle at t=0
        event=MockEvent(t=jnp.array(0, dtype=jnp.int32)),
    )
    return _make_obs(env_state, src_2_zone, n_zones, action, reward)


# ===========================================================================
# 1. Tests for available_cars_per_zone
# ===========================================================================

class TestAvailableCarsPerZone:
    def test_basic_counting(self):
        """Cars at known nodes should be counted in the correct zones."""
        src_2_zone = jnp.array([0, 0, 1, 1, 2], dtype=jnp.int32)
        n_zones = 3
        env_state = MockEnvState(
            locations=jnp.array([0, 1, 2, 4], dtype=jnp.int32),
            times=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
            event=MockEvent(t=jnp.array(1, dtype=jnp.int32)),
        )
        counts = available_cars_per_zone(env_state, src_2_zone, n_zones)
        np.testing.assert_array_equal(counts, [2, 1, 1])

    def test_idle_filtering(self):
        """Cars with times > t should not be counted."""
        src_2_zone = jnp.array([0, 1], dtype=jnp.int32)
        n_zones = 2
        env_state = MockEnvState(
            locations=jnp.array([0, 0, 1], dtype=jnp.int32),
            times=jnp.array([0, 5, 0], dtype=jnp.int32),  # car 1 is busy
            event=MockEvent(t=jnp.array(3, dtype=jnp.int32)),
        )
        counts = available_cars_per_zone(env_state, src_2_zone, n_zones)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_out_of_bounds_locations(self):
        """Locations outside src_2_zone range should be ignored."""
        src_2_zone = jnp.array([0, 1], dtype=jnp.int32)
        n_zones = 2
        env_state = MockEnvState(
            locations=jnp.array([0, 99, 1], dtype=jnp.int32),
            times=jnp.array([0, 0, 0], dtype=jnp.int32),
            event=MockEvent(t=jnp.array(1, dtype=jnp.int32)),
        )
        counts = available_cars_per_zone(env_state, src_2_zone, n_zones)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_unmapped_zones(self):
        """Nodes mapped to zone -1 should be excluded."""
        src_2_zone = jnp.array([0, -1, 1], dtype=jnp.int32)
        n_zones = 2
        env_state = MockEnvState(
            locations=jnp.array([0, 1, 2], dtype=jnp.int32),
            times=jnp.array([0, 0, 0], dtype=jnp.int32),
            event=MockEvent(t=jnp.array(1, dtype=jnp.int32)),
        )
        counts = available_cars_per_zone(env_state, src_2_zone, n_zones)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_all_busy(self):
        """No idle cars → all-zero counts."""
        src_2_zone = jnp.array([0, 1], dtype=jnp.int32)
        n_zones = 2
        env_state = MockEnvState(
            locations=jnp.array([0, 1], dtype=jnp.int32),
            times=jnp.array([10, 10], dtype=jnp.int32),
            event=MockEvent(t=jnp.array(0, dtype=jnp.int32)),
        )
        counts = available_cars_per_zone(env_state, src_2_zone, n_zones)
        np.testing.assert_array_equal(counts, [0, 0])


# ===========================================================================
# 2. Tests for reset
# ===========================================================================

class TestReset:
    def test_initial_state_shapes_and_values(self):
        estimator = AvgRewardLSTDDQEstimator()
        state = estimator.reset(jax.random.PRNGKey(0), None, None, None)

        assert state.A_matrix.shape == (63, 63)
        assert state.sum_ss_rs.shape == (63,)
        assert state.sum_ss.shape == (63,)
        assert float(state.sum_rs) == 0.0
        assert int(state.transition_count) == 0
        assert int(state.count_treated) == 0
        assert int(state.count_control) == 0
        assert not state.has_prev_state

    def test_reg_param_passthrough(self):
        estimator = AvgRewardLSTDDQEstimator(reg_param=0.5)
        state = estimator.reset(jax.random.PRNGKey(0), None, None, None)
        assert float(state.reg_param) == pytest.approx(0.5)


# ===========================================================================
# 3. Tests for update accumulation
# ===========================================================================

class TestUpdate:
    """Test the update method's accumulation logic."""

    def _make_estimator_and_state(self, n_zones=3, reg_param=1e-3):
        estimator = AvgRewardLSTDDQEstimator(reg_param=reg_param)
        state = AvgRewardLSTDDQEstimatorState(
            A_matrix=jnp.zeros((n_zones, n_zones)),
            sum_ss_rs=jnp.zeros(n_zones),
            sum_ss=jnp.zeros(n_zones),
            sum_rs=0.0,
            transition_count=0,
            states_treated=jnp.zeros(n_zones),
            states_control=jnp.zeros(n_zones),
            count_treated=0,
            count_control=0,
            prev_state=jnp.zeros(n_zones),
            prev_reward=0.0,
            prev_action=0.0,
            has_prev_state=False,
            reg_param=reg_param,
        )
        return estimator, state

    def test_first_observation_no_accumulation(self):
        """First obs should NOT update A/b accumulators (no previous state)."""
        estimator, state = self._make_estimator_and_state()
        obs = _make_obs_with_state([1, 2, 0], action=1.0, reward=5.0)
        new_state = estimator.update(None, None, None, state, obs)

        # A matrix and accumulators unchanged
        np.testing.assert_array_equal(new_state.A_matrix, jnp.zeros((3, 3)))
        np.testing.assert_array_equal(new_state.sum_ss_rs, jnp.zeros(3))
        assert int(new_state.transition_count) == 0

        # But prev_state should be set
        assert new_state.has_prev_state
        np.testing.assert_array_equal(
            new_state.prev_state, jnp.array([1, 2, 0], dtype=jnp.float32)
        )
        assert float(new_state.prev_reward) == pytest.approx(5.0)
        assert float(new_state.prev_action) == pytest.approx(1.0)

        # Treatment/control counts unchanged (no prev state on first obs)
        assert int(new_state.count_treated) == 0
        assert int(new_state.count_control) == 0

    def test_second_observation_updates_A(self):
        """Second obs should update A with outer(prev_state, prev_state - curr_state)."""
        estimator, state = self._make_estimator_and_state()
        obs1 = _make_obs_with_state([1, 0, 0], action=1.0, reward=3.0)
        obs2 = _make_obs_with_state([0, 1, 0], action=0.0, reward=2.0)

        state = estimator.update(None, None, None, state, obs1)
        state = estimator.update(None, None, None, state, obs2)

        # A should be outer([1,0,0], [1,0,0] - [0,1,0]) = outer([1,0,0], [1,-1,0])
        s_prev = np.array([1, 0, 0], dtype=np.float32)
        s_curr = np.array([0, 1, 0], dtype=np.float32)
        expected_A = np.outer(s_prev, s_prev - s_curr)
        np.testing.assert_allclose(state.A_matrix, expected_A, atol=1e-6)

        assert int(state.transition_count) == 1
        # sum_ss_rs = prev_state * prev_reward = [1,0,0] * 3.0 = [3,0,0]
        np.testing.assert_allclose(
            state.sum_ss_rs, np.array([3, 0, 0], dtype=np.float32), atol=1e-6
        )
        # sum_ss = prev_state = [1,0,0]
        np.testing.assert_allclose(
            state.sum_ss, np.array([1, 0, 0], dtype=np.float32), atol=1e-6
        )
        assert float(state.sum_rs) == pytest.approx(3.0)

    def test_treatment_control_grouping(self):
        """Treated/control sums should accumulate based on prev_action."""
        estimator, state = self._make_estimator_and_state()

        # obs1: action=1 (treated), state=[1,0,0]
        obs1 = _make_obs_with_state([1, 0, 0], action=1.0, reward=1.0)
        # obs2: action=0 (control), state=[0,1,0]
        obs2 = _make_obs_with_state([0, 1, 0], action=0.0, reward=2.0)
        # obs3: action=1 (treated), state=[0,0,1]
        obs3 = _make_obs_with_state([0, 0, 1], action=1.0, reward=3.0)

        state = estimator.update(None, None, None, state, obs1)
        state = estimator.update(None, None, None, state, obs2)
        state = estimator.update(None, None, None, state, obs3)

        # After obs2: prev_action was 1.0 (from obs1) → treated += [1,0,0]
        # After obs3: prev_action was 0.0 (from obs2) → control += [0,1,0]
        np.testing.assert_allclose(
            state.states_treated, np.array([1, 0, 0], dtype=np.float32), atol=1e-6
        )
        np.testing.assert_allclose(
            state.states_control, np.array([0, 1, 0], dtype=np.float32), atol=1e-6
        )
        assert int(state.count_treated) == 1
        assert int(state.count_control) == 1

    def test_multi_step_accumulation(self):
        """Feed N observations and verify accumulated state matches manual computation."""
        n_zones = 3
        estimator, state = self._make_estimator_and_state(n_zones=n_zones)

        rng = np.random.RandomState(42)
        n_steps = 10
        states_list = []
        actions_list = []
        rewards_list = []

        for i in range(n_steps):
            s = rng.randint(0, 5, size=n_zones).tolist()
            a = float(rng.choice([0, 1]))
            r = float(rng.randn())
            states_list.append(s)
            actions_list.append(a)
            rewards_list.append(r)

            obs = _make_obs_with_state(s, action=a, reward=r, n_zones=n_zones)
            state = estimator.update(None, None, None, state, obs)

        # Manual computation: transitions use pairs (i, i+1) for i in 0..N-2
        states_arr = np.array(states_list, dtype=np.float32)
        actions_arr = np.array(actions_list, dtype=np.float32)
        rewards_arr = np.array(rewards_list, dtype=np.float32)

        expected_A = np.zeros((n_zones, n_zones), dtype=np.float32)
        expected_sum_ss_rs = np.zeros(n_zones, dtype=np.float32)
        expected_sum_ss = np.zeros(n_zones, dtype=np.float32)
        expected_sum_rs = 0.0

        for i in range(n_steps - 1):
            s_prev = states_arr[i]
            s_curr = states_arr[i + 1]
            r_prev = rewards_arr[i]
            expected_A += np.outer(s_prev, s_prev - s_curr)
            expected_sum_ss_rs += s_prev * r_prev
            expected_sum_ss += s_prev
            expected_sum_rs += r_prev

        np.testing.assert_allclose(state.A_matrix, expected_A, atol=1e-5)
        np.testing.assert_allclose(state.sum_ss_rs, expected_sum_ss_rs, atol=1e-5)
        np.testing.assert_allclose(state.sum_ss, expected_sum_ss, atol=1e-5)
        assert float(state.sum_rs) == pytest.approx(expected_sum_rs, abs=1e-5)
        assert int(state.transition_count) == n_steps - 1

        # Treatment/control grouping: uses prev_action, so actions[:-1]
        treated_mask = actions_arr[:-1] > 0.5
        expected_states_treated = states_arr[:-1][treated_mask].sum(axis=0)
        expected_states_control = states_arr[:-1][~treated_mask].sum(axis=0)

        np.testing.assert_allclose(
            state.states_treated, expected_states_treated, atol=1e-5
        )
        np.testing.assert_allclose(
            state.states_control, expected_states_control, atol=1e-5
        )
        assert int(state.count_treated) == int(treated_mask.sum())
        assert int(state.count_control) == int((~treated_mask).sum())


# ===========================================================================
# 4. Tests for estimate
# ===========================================================================

class TestEstimate:
    def _run_sequence(self, states, actions, rewards, reg_param=1e-3, n_zones=3):
        """Feed a sequence through the estimator and return final state."""
        estimator = AvgRewardLSTDDQEstimator(reg_param=reg_param)
        state = AvgRewardLSTDDQEstimatorState(
            A_matrix=jnp.zeros((n_zones, n_zones)),
            sum_ss_rs=jnp.zeros(n_zones),
            sum_ss=jnp.zeros(n_zones),
            sum_rs=0.0,
            transition_count=0,
            states_treated=jnp.zeros(n_zones),
            states_control=jnp.zeros(n_zones),
            count_treated=0,
            count_control=0,
            prev_state=jnp.zeros(n_zones),
            prev_reward=0.0,
            prev_action=0.0,
            has_prev_state=False,
            reg_param=reg_param,
        )
        for s, a, r in zip(states, actions, rewards):
            obs = _make_obs_with_state(s, action=a, reward=r, n_zones=n_zones)
            state = estimator.update(None, None, None, state, obs)
        return estimator, state

    def test_insufficient_data_returns_zero(self):
        """With no data or only one group, estimate should be 0."""
        estimator = AvgRewardLSTDDQEstimator()
        state = estimator.reset(jax.random.PRNGKey(0), None, None, None)
        result = estimator.estimate(None, None, None, state)
        assert float(result) == 0.0

    def test_insufficient_no_control(self):
        """All treated, no control → 0."""
        estimator, state = self._run_sequence(
            states=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            actions=[1.0, 1.0, 1.0],
            rewards=[1.0, 2.0, 3.0],
        )
        result = estimator.estimate(None, None, None, state)
        assert float(result) == 0.0

    def test_insufficient_no_treated(self):
        """All control, no treated → 0."""
        estimator, state = self._run_sequence(
            states=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            actions=[0.0, 0.0, 0.0],
            rewards=[1.0, 2.0, 3.0],
        )
        result = estimator.estimate(None, None, None, state)
        assert float(result) == 0.0

    def test_minimal_sufficient_data(self):
        """With 1 treated + 1 control + 1 transition, should return finite value."""
        estimator, state = self._run_sequence(
            states=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            actions=[1.0, 0.0, 1.0],
            rewards=[1.0, 2.0, 3.0],
        )
        result = estimator.estimate(None, None, None, state)
        assert np.isfinite(float(result))

    def test_batch_equivalence(self):
        """Online estimate should match a batch NumPy reference implementation."""
        n_zones = 3
        reg_param = 0.1
        rng = np.random.RandomState(123)
        n_steps = 20

        states = [rng.randint(0, 5, size=n_zones).tolist() for _ in range(n_steps)]
        actions = [float(rng.choice([0, 1])) for _ in range(n_steps)]
        rewards = [float(rng.randn()) for _ in range(n_steps)]

        # Online estimate
        estimator, state = self._run_sequence(
            states, actions, rewards, reg_param=reg_param, n_zones=n_zones
        )
        online_estimate = float(estimator.estimate(None, None, None, state))

        # Batch reference (NumPy)
        ss = np.array(states, dtype=np.float64)
        aa = np.array(actions, dtype=np.float64)
        rr = np.array(rewards, dtype=np.float64)

        # A = sum over i in 0..N-2 of outer(s_i, s_i - s_{i+1})
        A = np.zeros((n_zones, n_zones), dtype=np.float64)
        for i in range(n_steps - 1):
            A += np.outer(ss[i], ss[i] - ss[i + 1])

        # b = sum(s_i * r_i) - sum(s_i) * mean(r)   for i in 0..N-2
        ss_trans = ss[:-1]
        rr_trans = rr[:-1]
        rbar = rr_trans.mean()
        b = (ss_trans * rr_trans[:, None]).sum(axis=0) - ss_trans.sum(axis=0) * rbar

        # theta = (A + reg*I)^{-1} b
        theta = np.linalg.solve(A + reg_param * np.eye(n_zones), b)

        # delta_xbar = mean(s_treated) - mean(s_control), using actions[:-1]
        treated_mask = aa[:-1] > 0.5
        mean_treated = ss[:-1][treated_mask].mean(axis=0)
        mean_control = ss[:-1][~treated_mask].mean(axis=0)
        delta_xbar = mean_treated - mean_control

        batch_estimate = float(theta @ delta_xbar)

        np.testing.assert_allclose(online_estimate, batch_estimate, atol=1e-3)

    def test_determinism(self):
        """Running the same sequence twice should produce identical results."""
        states = [[1, 2, 0], [0, 1, 1], [2, 0, 1], [1, 1, 1]]
        actions = [1.0, 0.0, 1.0, 0.0]
        rewards = [1.0, -1.0, 2.0, 0.5]

        _, state1 = self._run_sequence(states, actions, rewards)
        _, state2 = self._run_sequence(states, actions, rewards)

        estimator = AvgRewardLSTDDQEstimator()
        r1 = float(estimator.estimate(None, None, None, state1))
        r2 = float(estimator.estimate(None, None, None, state2))
        assert r1 == r2


# ===========================================================================
# 5. Numerical stability tests
# ===========================================================================

class TestNumericalStability:
    def _run_sequence(self, states, actions, rewards, reg_param=1e-3, n_zones=3):
        estimator = AvgRewardLSTDDQEstimator(reg_param=reg_param)
        state = AvgRewardLSTDDQEstimatorState(
            A_matrix=jnp.zeros((n_zones, n_zones)),
            sum_ss_rs=jnp.zeros(n_zones),
            sum_ss=jnp.zeros(n_zones),
            sum_rs=0.0,
            transition_count=0,
            states_treated=jnp.zeros(n_zones),
            states_control=jnp.zeros(n_zones),
            count_treated=0,
            count_control=0,
            prev_state=jnp.zeros(n_zones),
            prev_reward=0.0,
            prev_action=0.0,
            has_prev_state=False,
            reg_param=reg_param,
        )
        for s, a, r in zip(states, actions, rewards):
            obs = _make_obs_with_state(s, action=a, reward=r, n_zones=n_zones)
            state = estimator.update(None, None, None, state, obs)
        return estimator, state

    def test_large_regularization_shrinks_estimate(self):
        """Large reg_param should push estimate toward 0."""
        states = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]
        actions = [1.0, 0.0, 1.0, 0.0, 1.0]
        rewards = [10.0, -10.0, 10.0, -10.0, 10.0]

        _, state_small = self._run_sequence(states, actions, rewards, reg_param=0.01)
        _, state_large = self._run_sequence(states, actions, rewards, reg_param=1000.0)

        est = AvgRewardLSTDDQEstimator()
        est_small = abs(float(est.estimate(None, None, None, state_small)))
        est_large = abs(float(est.estimate(None, None, None, state_large)))

        # Large regularization should give smaller absolute estimate
        assert est_large < est_small

    def test_small_reg_well_conditioned(self):
        """With very small reg_param, estimate should be close to reg_param=0 limit."""
        rng = np.random.RandomState(99)
        n_zones = 3
        n_steps = 30
        states = [rng.randint(0, 5, size=n_zones).tolist() for _ in range(n_steps)]
        actions = [float(rng.choice([0, 1])) for _ in range(n_steps)]
        rewards = [float(rng.randn()) for _ in range(n_steps)]

        _, state_small = self._run_sequence(
            states, actions, rewards, reg_param=1e-8, n_zones=n_zones
        )
        _, state_medium = self._run_sequence(
            states, actions, rewards, reg_param=1e-4, n_zones=n_zones
        )
        est = AvgRewardLSTDDQEstimator()
        r_small = float(est.estimate(None, None, None, state_small))
        r_medium = float(est.estimate(None, None, None, state_medium))
        assert np.isfinite(r_small)
        assert np.isfinite(r_medium)
        # They should be close since both regularizations are small
        np.testing.assert_allclose(r_small, r_medium, rtol=0.1)

    def test_identical_states_with_regularization(self):
        """All identical states → singular A, but regularization should prevent NaN."""
        states = [[1, 1, 1]] * 6
        actions = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        estimator, state = self._run_sequence(
            states, actions, rewards, reg_param=1.0
        )
        result = float(estimator.estimate(None, None, None, state))
        assert np.isfinite(result)
        # With identical states for treated and control, delta_xbar = 0 → estimate = 0
        assert result == pytest.approx(0.0, abs=1e-6)
