# WARNING -- not fully tested
from flax import struct
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from functools import partial

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
class LSTDLambdaEstimatorState(EstimatorState):
    """
    State for LSTD-λ based treatment effect estimator.
    
    This estimator maintains separate LSTD systems for treated and control groups
    to estimate off-policy evaluation (OPE) values, then computes the treatment
    effect as the difference between the two estimates.
    """
    # LSTD matrices for treatment group (action=1)
    A_matrix_treated: jnp.ndarray  # Σ s(s-s')^T for treated
    b_vector_treated: jnp.ndarray  # Σ s*r for treated
    count_treated: int
    
    # LSTD matrices for control group (action=0)  
    A_matrix_control: jnp.ndarray  # Σ s(s-s')^T for control
    b_vector_control: jnp.ndarray  # Σ s*r for control
    count_control: int
    
    # Previous state for transitions (needed for LSTD)
    prev_state: jnp.ndarray  # previous state for transition computation
    prev_action: float  # previous action for proper grouping
    prev_reward: float  # previous reward for proper LSTD updates
    has_prev_state: bool  # flag to track if we have a previous state
    
    # Regularization parameter
    reg_param: float
    
    # Lambda parameter for eligibility traces
    lambda_param: float


@struct.dataclass  
class LSTDLambdaEstimator(Estimator):
    """
    LSTD-λ based treatment effect estimator.
    
    This estimator implements off-policy evaluation using LSTD-λ by maintaining
    separate LSTD systems for treated and control groups. The treatment effect
    is estimated as the difference between the OPE values.
    
    The estimation logic follows:
    1. For each treatment group (treated/control):
       - Accumulate A = Σ s(s-s')^T and b = Σ s*r
       - Solve η = (A + λI)^(-1) b for OPE value
    2. Treatment effect = η_treated - η_control
    """
    discount_factor: float = 1.0
    alpha: float = 1.0  # Regularization parameter
    lambda_: float = 0.0  # Eligibility trace decay parameter
    
    def reset(self, rng, env, env_params, design):
        # Get state dimension from environment parameters
        # Now using available cars per zone instead of raw car locations
        # The state dimension is n_spatial_zones (fixed at 63 zones)
        state_dim = 63
        
        return LSTDLambdaEstimatorState(
            A_matrix_treated=jnp.zeros((state_dim, state_dim)),
            b_vector_treated=jnp.zeros(state_dim),
            count_treated=0,
            A_matrix_control=jnp.zeros((state_dim, state_dim)),
            b_vector_control=jnp.zeros(state_dim),
            count_control=0,
            prev_state=jnp.zeros(state_dim),
            prev_action=0.0,
            prev_reward=0.0,
            has_prev_state=False,
            reg_param=1e-3,
            lambda_param=self.lambda_,
        )

    def update(self, env, env_params, design, state: LSTDLambdaEstimatorState, obs: Observation):
        """
        Update LSTD-λ estimator with new observation.
        
        Maintains separate LSTD systems for treated and control groups.
        """
        # Extract state representation: available cars per zone
        current_state = available_cars_per_zone(
            obs.design_info.env_state, 
            obs.design_info.src_2_zone, 
            obs.design_info.n_spatial_zones
        ).astype(jnp.float32)  # Convert to float for LSTD operations
        
        state_dim = obs.design_info.n_spatial_zones  # Fixed at 63
        
        # Update with the current observation
        reward = obs.reward.astype(jnp.float32)
        action = obs.action.astype(jnp.float32)
        
        # LSTD-λ equations exactly like original: separate systems for treated/control
        # A = ss.T @ (ss - snews), b = ss.T @ rs for each group
        
        def update_with_transition():
            # Use previous state and action for proper LSTD updates
            prev_treated = state.prev_action > 0.5
            prev_control = state.prev_action <= 0.5
            
            # State transition: prev_state -> current_state
            state_diff = state.prev_state - current_state  # (s - s')
            
            # Update for treatment group (based on previous action)
            A_update_treated = jnp.where(
                prev_treated,
                jnp.outer(state.prev_state, state_diff),  # s @ (s - s').T
                jnp.zeros((state_dim, state_dim))
            )
            b_update_treated = jnp.where(
                prev_treated,
                state.prev_state * state.prev_reward,  # s * r
                jnp.zeros(state_dim)
            )
            count_update_treated = jnp.where(prev_treated, 1, 0)
            
            # Update for control group (based on previous action)
            A_update_control = jnp.where(
                prev_control,
                jnp.outer(state.prev_state, state_diff),  # s @ (s - s').T
                jnp.zeros((state_dim, state_dim))
            )
            b_update_control = jnp.where(
                prev_control,
                state.prev_state * state.prev_reward,  # s * r
                jnp.zeros(state_dim)
            )
            count_update_control = jnp.where(prev_control, 1, 0)
            
            return (A_update_treated, b_update_treated, count_update_treated,
                    A_update_control, b_update_control, count_update_control)
        
        def no_update():
            return (jnp.zeros((state_dim, state_dim)), jnp.zeros(state_dim), 0,
                    jnp.zeros((state_dim, state_dim)), jnp.zeros(state_dim), 0)
        
        # Only perform LSTD update if we have a previous state
        (A_update_treated, b_update_treated, count_update_treated,
         A_update_control, b_update_control, count_update_control) = jax.lax.cond(
            state.has_prev_state,
            update_with_transition,
            no_update
        )
        
        new_A_treated = state.A_matrix_treated + A_update_treated
        new_b_treated = state.b_vector_treated + b_update_treated
        new_count_treated = state.count_treated + count_update_treated
        
        new_A_control = state.A_matrix_control + A_update_control
        new_b_control = state.b_vector_control + b_update_control
        new_count_control = state.count_control + count_update_control
        
        return LSTDLambdaEstimatorState(
            A_matrix_treated=new_A_treated,
            b_vector_treated=new_b_treated,
            count_treated=new_count_treated,
            A_matrix_control=new_A_control,
            b_vector_control=new_b_control,
            count_control=new_count_control,
            prev_state=current_state,  # Store current state as next iteration's previous state
            prev_action=action,  # Store current action as next iteration's previous action
            prev_reward=reward,  # Store current reward as next iteration's previous reward
            has_prev_state=True,  # We now have a previous state for next iteration
            reg_param=state.reg_param,
            lambda_param=state.lambda_param,
        )

    def estimate(self, env, env_params, design, state: LSTDLambdaEstimatorState):
        """
        Compute LSTD-λ treatment effect estimate.
        
        Computes OPE values for both treatment groups using LSTD, then
        returns the difference as the treatment effect estimate.
        """
        # Return 0 if insufficient data
        def insufficient_data():
            return 0.0
        
        def compute_estimate():
            # Regularization matrix
            reg_matrix_treated = state.reg_param * jnp.eye(state.A_matrix_treated.shape[0])
            reg_matrix_control = state.reg_param * jnp.eye(state.A_matrix_control.shape[0])
            
            # OPE for treatment group (eta1)
            eta1 = solve(
                state.A_matrix_treated + reg_matrix_treated, 
                state.b_vector_treated
            )[0]  # Take first element like original implementation
            
            # OPE for control group (eta0)
            eta0 = solve(
                state.A_matrix_control + reg_matrix_control, 
                state.b_vector_control
            )[0]  # Take first element like original implementation
            
            # Treatment effect is difference
            return eta1 - eta0
        
        # Check if we have sufficient data from both groups
        sufficient_data = (
            (state.count_treated > 0) & 
            (state.count_control > 0)
        )
        
        return jax.lax.cond(
            sufficient_data,
            compute_estimate,
            insufficient_data
        )
