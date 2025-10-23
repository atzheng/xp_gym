from flax import struct
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from functools import partial
import numpy as np

from or_gymnax.rideshare import EnvState
from xp_gym.estimators.estimator import EstimatorState, Estimator
from xp_gym.observation import Observation


@partial(jax.jit, static_argnums=(2,))
def available_cars_per_zone(state, src_2_zone: jnp.ndarray, n_zones: int):
    """
    Return counts of idle cars per zone at time t = state.event.t.
    src_2_zone: shape (max_node_id+1,), values are zone_id or -1.
    """
    t = state.event.t
    locs = state.locations
    times = state.times

    is_idle = (times <= t).astype(jnp.int32)

    in_bounds = (locs >= 0) & (locs < src_2_zone.shape[0])
    zone_ids = jnp.where(in_bounds, src_2_zone[locs], -1)

    # keep only mapped zones
    contrib = jnp.where(zone_ids >= 0, is_idle, 0)
    safe_zone_ids = jnp.where(zone_ids >= 0, zone_ids, 0)

    counts = jnp.zeros((n_zones,), jnp.int32).at[safe_zone_ids].add(contrib)
    return counts

@struct.dataclass
class DynkinEstimatorState(EstimatorState):
    """
    State for Dynkin-based value difference estimator using LSTD.
    
    This estimator tracks state transitions and rewards to compute
    treatment effects using the Dynkin approach with LSTD equations.
    """
    # Accumulated matrices for LSTD computation
    A_matrix: jnp.ndarray  # sum of ss.T @ (ss - snews) 
    
    # Components for deferred b computation (to match batch version exactly)
    sum_ss_rs: jnp.ndarray  # sum of s_i * r_i (element-wise)
    sum_ss: jnp.ndarray  # sum of s_i states
    sum_rs: float  # sum of r_i rewards
    transition_count: int  # number of transitions processed
    
    # State and reward accumulators
    states_treated: jnp.ndarray  # sum of states when treated
    states_control: jnp.ndarray  # sum of states when control
    count_treated: int  # number of treated observations
    count_control: int  # number of control observations
    
    # Previous state for transitions (needed for LSTD)
    prev_state: jnp.ndarray  # previous state for transition computation
    prev_reward: float  # previous reward for proper LSTD temporal alignment
    prev_action: float  # previous action for proper treatment/control grouping
    has_prev_state: bool  # flag to track if we have a previous state
    
    # Regularization parameter
    reg_param: float


@struct.dataclass  
class DynkinEstimator(Estimator):
    """
    Dynkin-based value difference estimator using LSTD.
    
    This estimator implements the Dynkin method for treatment effect estimation
    by solving LSTD equations and computing the difference in expected values
    between treated and control groups based on state features.
    
    The estimation logic follows:
    1. Accumulate LSTD matrices A = Σ s(s-s')^T and b = Σ s(r-r̄)
    2. Solve θ = (A + λI)^(-1) b  
    3. Compute treatment effect as θ^T (x̄_treated - x̄_control)
    """
    discount_factor: float = 1.0
    reg_param: float = 1e-3
    
    def reset(self, rng, env_params):
        # Get state dimension from environment parameters
        # Now using available cars per zone instead of raw car locations
        # The state dimension is n_spatial_zones (number of zones, typically 63)
        # We need to get this from the design info, but for now use a reasonable default
        # This will be properly set when we see the first observation
        state_dim = 63  # This will be updated in the first update call
        
        return DynkinEstimatorState(
            A_matrix=jnp.zeros((state_dim, state_dim)),
            sum_ss_rs=jnp.zeros(state_dim),
            sum_ss=jnp.zeros(state_dim),
            sum_rs=0.0,
            transition_count=0,
            states_treated=jnp.zeros(state_dim),
            states_control=jnp.zeros(state_dim),
            count_treated=0,
            count_control=0,
            prev_state=jnp.zeros(state_dim),
            prev_reward=0.0,
            prev_action=0.0,
            has_prev_state=False,
            reg_param=self.reg_param,
        )

    def update(self, env, env_params, state: DynkinEstimatorState, obs: Observation):
        """
        Update Dynkin estimator with new observation.
        
        Accumulates LSTD matrices and state statistics needed for
        treatment effect estimation.
        """
        # Extract state representation: available cars per zone
        current_state = available_cars_per_zone(
            obs.design_info.env_state, 
            obs.design_info.src_2_zone, 
            obs.design_info.n_spatial_zones
        ).astype(jnp.float32)  # Convert to float for LSTD operations
        
        # jax.debug.print("available_cars_per_zone | shape={} | head10={}",
        #                 current_state.shape, current_state[:10])
        
        # Update with the current observation
        reward = obs.reward.astype(jnp.float32)
        action = obs.action.astype(jnp.float32)
        
        # LSTD-DQ equations: accumulate A and components for b separately
        # Only update if we have a previous state (for state transition)
        def update_with_transition():
            # A += s.T @ (s - s') = s @ (s - s').T  (outer product form)
            state_diff = state.prev_state - current_state  # (s - s')
            A_update = jnp.outer(state.prev_state, state_diff)  # s @ (s - s').T
            new_A = state.A_matrix + A_update
            
            # Accumulate components for b computation (to match batch version exactly)
            # b = ss.T @ (rs - rbar) = ss.T @ rs - (sum of ss) * rbar
            new_sum_ss_rs = state.sum_ss_rs + state.prev_state * state.prev_reward
            new_sum_ss = state.sum_ss + state.prev_state  
            new_sum_rs = state.sum_rs + state.prev_reward
            new_transition_count = state.transition_count + 1
            
            return new_A, new_sum_ss_rs, new_sum_ss, new_sum_rs, new_transition_count
        
        def no_update():
            return state.A_matrix, state.sum_ss_rs, state.sum_ss, state.sum_rs, state.transition_count
        
        # Only perform LSTD update if we have a previous state
        new_A_matrix, new_sum_ss_rs, new_sum_ss, new_sum_rs, new_transition_count = jax.lax.cond(
            state.has_prev_state,
            update_with_transition,
            no_update
        )
        
        # Update state statistics by treatment group
        # CRITICAL FIX: Use prev_state with prev_action (not current_state with current_action)
        # This matches the batch version: actions[:-1] paired with states[:-1]
        def update_treatment_groups():
            treated_update = jnp.where(state.prev_action > 0.5, state.prev_state, jnp.zeros_like(state.prev_state))
            control_update = jnp.where(state.prev_action <= 0.5, state.prev_state, jnp.zeros_like(state.prev_state))
            count_treated_update = jnp.where(state.prev_action > 0.5, 1, 0)
            count_control_update = jnp.where(state.prev_action <= 0.5, 1, 0)
            return treated_update, control_update, count_treated_update, count_control_update
        
        def no_treatment_update():
            return jnp.zeros_like(current_state), jnp.zeros_like(current_state), 0, 0
        
        # Only update treatment/control groups if we have previous state/action
        treated_update, control_update, count_treated_update, count_control_update = jax.lax.cond(
            state.has_prev_state,
            update_treatment_groups,
            no_treatment_update
        )
        
        new_states_treated = state.states_treated + treated_update
        new_states_control = state.states_control + control_update
        new_count_treated = state.count_treated + count_treated_update
        new_count_control = state.count_control + count_control_update
        
        return DynkinEstimatorState(
            A_matrix=new_A_matrix,
            sum_ss_rs=new_sum_ss_rs,
            sum_ss=new_sum_ss,
            sum_rs=new_sum_rs,
            transition_count=new_transition_count,
            states_treated=new_states_treated,
            states_control=new_states_control,
            count_treated=new_count_treated,
            count_control=new_count_control,
            prev_state=current_state,  # Store current state as next iteration's previous state
            prev_reward=reward,  # Store current reward as next iteration's previous reward
            prev_action=action,  # Store current action as next iteration's previous action
            has_prev_state=True,  # We now have a previous state for next iteration
            reg_param=state.reg_param,
        )

    def estimate(self, env, env_params, state: DynkinEstimatorState):
        """
        Compute Dynkin estimate using accumulated LSTD data.
        
        Returns the treatment effect estimate based on:
        1. Solving LSTD equations: θ = (A + λI)^(-1) b
        2. Computing state difference: Δx̄ = x̄_treated - x̄_control  
        3. Treatment effect: τ = θ^T Δx̄
        """
        # Return 0 if insufficient data
        def insufficient_data():
            return 0.0
        
        def compute_estimate():
            # Compute mean states for each treatment group
            mean_states_treated = state.states_treated / jnp.maximum(state.count_treated, 1)
            mean_states_control = state.states_control / jnp.maximum(state.count_control, 1)
            
            # State difference (delta_xbar in original code)
            delta_xbar = mean_states_treated - mean_states_control
            
            # Compute b vector exactly as in batch version: b = sum_ss_rs - sum_ss * rbar
            rbar = state.sum_rs / jnp.maximum(state.transition_count, 1)
            b_vector = state.sum_ss_rs - state.sum_ss * rbar
            
            # Regularization matrix
            reg_matrix = state.reg_param * jnp.eye(state.A_matrix.shape[0])
            
            # Solve LSTD equations: θ = (A + λI)^(-1) b
            theta_dq = solve(state.A_matrix + reg_matrix, b_vector)
            # Treatment effect estimate: τ = θ^T Δx̄
            estimate = jnp.dot(theta_dq, delta_xbar)
            return estimate
        
        # Check if we have sufficient data from both groups
        sufficient_data = (
            (state.count_treated > 0) & 
            (state.count_control > 0) & 
            (state.transition_count >= 1)
        )
        
        return jax.lax.cond(
            sufficient_data,
            compute_estimate,
            insufficient_data
        )