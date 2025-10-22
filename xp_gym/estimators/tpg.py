from flax import struct
import jax
import jax.numpy as jnp
from jax import Array

from xp_gym.estimators.estimator import EstimatorState, Estimator
from xp_gym.observation import Observation


@struct.dataclass
class TPGEstimatorState(EstimatorState):
    """
    State for Truncated DQ Estimator with proper sliding window.
    
    Maintains:
    - complete_estimate: Sum of IPW-weighted truncated returns for completed observations
    - complete_count: Number of completed observations  
    - Sliding window of last k observations for partial computation
    """
    complete_estimate: float  # Sum of completed IPW-weighted truncated returns
    complete_count: int  # Number of completed observations
    
    # Sliding window for incomplete observations
    reward_history: Array  # Shape: (k,) - recent rewards
    action_history: Array  # Shape: (k,) - recent actions
    p_history: Array  # Shape: (k,) - recent treatment probabilities
    
    total_observations: int  # Total observations processed


@struct.dataclass 
class TPGEstimator(Estimator):
    """
    Proper Truncated DQ Estimator implementation.
    
    Uses current reward + next k rewards with IPW weighting.
    Maintains sliding window to handle incomplete observations.
    """
    k: int  # Truncation horizon
    
    def reset(self, rng, env, env_params):
        # Handle k=0 case by using minimum size of 1 for arrays
        buffer_size = max(self.k, 1)
        return TPGEstimatorState(
            complete_estimate=0.0,
            complete_count=0,
            reward_history=jnp.zeros(buffer_size),
            action_history=jnp.zeros(buffer_size, dtype=jnp.int32),
            p_history=jnp.zeros(buffer_size),
            total_observations=0
        )
    
    def update(self, state: TPGEstimatorState, obs: Observation):
        """
        Update with proper truncated DQ logic using JAX-safe operations.
        Special case: when k=0, behave exactly like naive estimator.
        """
        
        def update_k_zero():
            """Handle k=0 case: immediate completion like naive estimator."""
            # Calculate IPW weight and add directly to complete estimate
            ipw_weight = obs.action / obs.design_info.p - (1 - obs.action) / (1 - obs.design_info.p)
            new_complete_estimate = state.complete_estimate + ipw_weight * obs.reward
            new_complete_count = state.complete_count + 1
            
            # Keep buffers unchanged (they're not used when k=0)
            return (new_complete_estimate, new_complete_count,
                   state.reward_history, state.action_history, state.p_history)
        
        def update_k_positive():
            """Handle k > 0 case: use sliding window logic."""
            
            def complete_oldest_and_shift():
                """Complete oldest observation and shift window."""
                # Get data for oldest observation (position 0)
                oldest_reward = state.reward_history[0]
                oldest_action = state.action_history[0]
                oldest_p = state.p_history[0]
                
                # Calculate truncated return: old reward + k future rewards  
                future_rewards = jnp.sum(state.reward_history[1:self.k]) + obs.reward
                truncated_return = oldest_reward + future_rewards
                
                # Calculate IPW weight and update complete estimate
                ipw_weight = oldest_action / oldest_p - (1 - oldest_action) / (1 - oldest_p)
                new_complete_estimate = state.complete_estimate + ipw_weight * truncated_return
                new_complete_count = state.complete_count + 1
                
                # Shift window: move everything left and add new observation at end
                new_reward_history = jnp.concatenate([
                    state.reward_history[1:self.k], 
                    obs.reward.reshape(-1)  # Ensure 1D array
                ])
                new_action_history = jnp.concatenate([
                    state.action_history[1:self.k], 
                    obs.action.reshape(-1)  # Ensure 1D array
                ])
                new_p_history = jnp.concatenate([
                    state.p_history[1:self.k], 
                    jnp.array([obs.design_info.p])
                ])
                
                return (new_complete_estimate, new_complete_count, 
                       new_reward_history, new_action_history, new_p_history)
            
            def add_to_growing_window():
                """Add observation to growing window (not yet full)."""
                window_pos = state.total_observations
                
                new_reward_history = state.reward_history.at[window_pos].set(obs.reward)
                new_action_history = state.action_history.at[window_pos].set(obs.action)
                new_p_history = state.p_history.at[window_pos].set(obs.design_info.p)
                
                return (state.complete_estimate, state.complete_count,
                       new_reward_history, new_action_history, new_p_history)
            
            # Use JAX conditional to choose update path for k > 0
            return jax.lax.cond(
                state.total_observations >= self.k,  # Window is full
                complete_oldest_and_shift,
                add_to_growing_window
            )
        
        # Choose between k=0 and k>0 logic
        complete_estimate, complete_count, reward_history, action_history, p_history = jax.lax.cond(
            self.k == 0,
            update_k_zero,
            update_k_positive
        )
        
        return TPGEstimatorState(
            complete_estimate=complete_estimate,
            complete_count=complete_count,
            reward_history=reward_history,
            action_history=action_history,
            p_history=p_history,
            total_observations=state.total_observations + 1
        )
    
    def estimate(self, state: TPGEstimatorState):
        """
        Compute final estimate combining complete and incomplete observations.
        Special case: when k=0, only use complete estimates (like naive estimator).
        """
        
        def compute_k_zero_estimate():
            """For k=0, all observations are immediately complete."""
            return state.complete_estimate / state.total_observations
        
        def compute_k_positive_estimate():
            """For k>0, combine complete and incomplete estimates."""
            # Start with completed estimates
            total_estimate = state.complete_estimate
            
            # Add estimates from incomplete observations in sliding window
            window_size = jnp.minimum(state.total_observations, self.k)
            
            # Process each position in window
            def add_partial_estimate(i, acc):
                # Calculate partial truncated return for observation at position i
                # This includes rewards from position i to end of current window
                buffer_size = max(self.k, 1)
                remaining_rewards = jnp.sum(
                    jnp.where(
                        jnp.arange(buffer_size) >= i,
                        jnp.where(jnp.arange(buffer_size) < window_size, state.reward_history, 0.0),
                        0.0
                    )
                )
                
                # Get IPW weight for this observation
                action = state.action_history[i]
                p = state.p_history[i]
                ipw_weight = action / p - (1 - action) / (1 - p)
                
                return acc + ipw_weight * remaining_rewards
            
            # Add incomplete estimates using JAX scan for efficiency
            incomplete_estimate = jax.lax.fori_loop(0, window_size, add_partial_estimate, 0.0)
            total_estimate += incomplete_estimate
            
            # Return average over all observations
            return total_estimate / state.total_observations
        
        def compute_estimate():
            """Choose estimation method based on k value."""
            return jax.lax.cond(
                self.k == 0,
                compute_k_zero_estimate,
                compute_k_positive_estimate
            )
        
        return jax.lax.cond(
            state.total_observations == 0,
            lambda: 0.0,
            compute_estimate
        )