from flax import struct
import jax
import jax.numpy as jnp
from jax import Array

from xp_gym.estimators.estimator import EstimatorState, Estimator
from xp_gym.observation import Observation


@struct.dataclass
class SwitchbackTPGEstimatorState(EstimatorState):
    """
    State for Switchback Truncated Policy Gradient Estimator.

    Operates at the cluster/interval level instead of event level.
    Aggregates rewards within each time interval and uses cluster averages.

    Maintains:
    - Sum of truncated returns for treatment and control clusters separately
    - Count of treatment and control clusters
    - Sliding window of last k clusters for partial computation
    - Current cluster accumulation buffers
    """
    # Complete clusters (exited the window)
    treatment_sum: float  # Sum of truncated returns for treatment clusters
    treatment_count: int  # Number of treatment clusters
    control_sum: float  # Sum of truncated returns for control clusters
    control_count: int  # Number of control clusters

    # Sliding window for incomplete clusters (cluster-level)
    avg_reward_history: Array  # Shape: (k,) - average reward per cluster
    action_history: Array  # Shape: (k,) - action for each cluster
    p_history: Array  # Shape: (k,) - treatment probability for each cluster (kept for compatibility)

    # Current cluster accumulation
    current_cluster_id: int  # ID of the current cluster being accumulated
    current_cluster_reward_sum: float  # Sum of rewards in current cluster
    current_cluster_count: int  # Number of events in current cluster
    current_cluster_action: int  # Action for current cluster (first event's action)
    current_cluster_p: float  # Treatment probability for current cluster (kept for compatibility)

    total_clusters: int  # Total clusters processed (including current)
    total_observations: int  # Total events processed


@struct.dataclass
class SwitchbackTPGEstimator(Estimator):
    """
    Switchback Truncated Policy Gradient Estimator.

    Similar to SwitchbackTPGEstimator but uses difference-in-means instead of IPW:
    1. Groups events into time intervals (clusters) based on cluster_id
    2. Computes average reward per cluster
    3. Uses current cluster avg + next k cluster avgs (truncated return)
    4. Averages treatment and control truncated returns separately
    5. Computes ATE as: mean(treatment_returns) - mean(control_returns)

    Parameters:
    - k: Truncation horizon (number of future clusters to look ahead)
    - switch_every: Time interval duration (used for validation/debugging)
    """
    k: int  # Truncation horizon in clusters
    switch_every: int = 5000  # Time period duration (optional, for reference)

    def reset(self, rng, env_params):
        # Handle k=0 case by using minimum size of 1 for arrays
        buffer_size = max(self.k, 1)
        return SwitchbackTPGEstimatorState(
            treatment_sum=0.0,
            treatment_count=0,
            control_sum=0.0,
            control_count=0,
            avg_reward_history=jnp.zeros(buffer_size),
            action_history=jnp.zeros(buffer_size, dtype=jnp.int32),
            p_history=jnp.zeros(buffer_size),
            current_cluster_id=-1,  # No cluster yet
            current_cluster_reward_sum=0.0,
            current_cluster_count=0,
            current_cluster_action=0,
            current_cluster_p=0.5,
            total_clusters=0,
            total_observations=0
        )

    def update(self, state: SwitchbackTPGEstimatorState, obs: Observation):
        """
        Update with cluster-level truncated policy gradient logic.

        When a new cluster is detected:
        1. Finalize the previous cluster (compute average reward)
        2. Add it to the sliding window (possibly completing an old cluster)
        3. Start accumulating the new cluster

        Within the same cluster:
        - Accumulate rewards and count events
        """
        # Get cluster_id from observation
        cluster_id = obs.design_info.cluster_id

        # Check if this is a new cluster
        is_new_cluster = cluster_id != state.current_cluster_id
        is_first_event = state.current_cluster_id == -1

        def handle_new_cluster():
            """Process when entering a new cluster."""

            def finalize_previous_cluster():
                """Finalize the previous cluster and add to window."""
                # Compute average reward for the completed cluster
                avg_reward = state.current_cluster_reward_sum / jnp.maximum(state.current_cluster_count, 1)

                # Now add this completed cluster to the sliding window
                # This is similar to the event-level logic but at cluster level

                def update_k_zero():
                    """Handle k=0: immediate completion with TPG."""
                    # Add to treatment or control sum based on action
                    new_treatment_sum = jnp.where(
                        state.current_cluster_action == 1,
                        state.treatment_sum + avg_reward,
                        state.treatment_sum
                    )
                    new_treatment_count = jnp.where(
                        state.current_cluster_action == 1,
                        state.treatment_count + 1,
                        state.treatment_count
                    )
                    new_control_sum = jnp.where(
                        state.current_cluster_action == 0,
                        state.control_sum + avg_reward,
                        state.control_sum
                    )
                    new_control_count = jnp.where(
                        state.current_cluster_action == 0,
                        state.control_count + 1,
                        state.control_count
                    )

                    return (new_treatment_sum, new_treatment_count, new_control_sum, new_control_count,
                           state.avg_reward_history, state.action_history, state.p_history)

                def update_k_positive():
                    """Handle k > 0: use sliding window logic with TPG."""

                    def complete_oldest_and_shift():
                        """Complete oldest cluster and shift window."""
                        # Get data for oldest cluster (position 0)
                        oldest_avg_reward = state.avg_reward_history[0]
                        oldest_action = state.action_history[0]

                        # Calculate truncated return: old cluster avg + k future cluster avgs
                        future_avg_rewards = jnp.sum(state.avg_reward_history[1:self.k]) + avg_reward
                        truncated_return = oldest_avg_reward + future_avg_rewards

                        # Add to treatment or control sum based on action
                        new_treatment_sum = jnp.where(
                            oldest_action == 1,
                            state.treatment_sum + truncated_return,
                            state.treatment_sum
                        )
                        new_treatment_count = jnp.where(
                            oldest_action == 1,
                            state.treatment_count + 1,
                            state.treatment_count
                        )
                        new_control_sum = jnp.where(
                            oldest_action == 0,
                            state.control_sum + truncated_return,
                            state.control_sum
                        )
                        new_control_count = jnp.where(
                            oldest_action == 0,
                            state.control_count + 1,
                            state.control_count
                        )

                        # Shift window: move everything left and add new cluster at end
                        new_avg_reward_history = jnp.concatenate([
                            state.avg_reward_history[1:self.k],
                            jnp.array([avg_reward])
                        ])
                        new_action_history = jnp.concatenate([
                            state.action_history[1:self.k],
                            jnp.array([state.current_cluster_action])
                        ])
                        new_p_history = jnp.concatenate([
                            state.p_history[1:self.k],
                            jnp.array([state.current_cluster_p])
                        ])

                        return (new_treatment_sum, new_treatment_count, new_control_sum, new_control_count,
                               new_avg_reward_history, new_action_history, new_p_history)

                    def add_to_growing_window():
                        """Add cluster to growing window (not yet full)."""
                        # total_clusters counts completed clusters, so use it as position
                        window_pos = state.total_clusters

                        new_avg_reward_history = state.avg_reward_history.at[window_pos].set(avg_reward)
                        new_action_history = state.action_history.at[window_pos].set(state.current_cluster_action)
                        new_p_history = state.p_history.at[window_pos].set(state.current_cluster_p)

                        return (state.treatment_sum, state.treatment_count, state.control_sum, state.control_count,
                               new_avg_reward_history, new_action_history, new_p_history)

                    # Use JAX conditional to choose update path for k > 0
                    return jax.lax.cond(
                        state.total_clusters >= self.k,  # Window is full
                        complete_oldest_and_shift,
                        add_to_growing_window
                    )

                # Choose between k=0 and k>0 logic
                (treatment_sum, treatment_count, control_sum, control_count,
                 avg_reward_history, action_history, p_history) = jax.lax.cond(
                    self.k == 0,
                    update_k_zero,
                    update_k_positive
                )

                # Start new cluster with current observation
                return SwitchbackTPGEstimatorState(
                    treatment_sum=treatment_sum,
                    treatment_count=treatment_count,
                    control_sum=control_sum,
                    control_count=control_count,
                    avg_reward_history=avg_reward_history,
                    action_history=action_history,
                    p_history=p_history,
                    current_cluster_id=cluster_id,
                    current_cluster_reward_sum=obs.reward,
                    current_cluster_count=1,
                    current_cluster_action=obs.action.astype(jnp.int32),
                    current_cluster_p=obs.design_info.p,
                    total_clusters=state.total_clusters + 1,  # Increment completed clusters
                    total_observations=state.total_observations + 1
                )

            def start_first_cluster():
                """Initialize the first cluster."""
                return state.replace(
                    current_cluster_id=cluster_id,
                    current_cluster_reward_sum=obs.reward,
                    current_cluster_count=1,
                    current_cluster_action=obs.action.astype(jnp.int32),
                    current_cluster_p=obs.design_info.p,
                    total_observations=state.total_observations + 1
                )

            # Choose between first cluster and subsequent clusters
            return jax.lax.cond(
                is_first_event,
                start_first_cluster,
                finalize_previous_cluster
            )

        def accumulate_in_current_cluster():
            """Accumulate rewards within the same cluster."""
            return state.replace(
                current_cluster_reward_sum=state.current_cluster_reward_sum + obs.reward,
                current_cluster_count=state.current_cluster_count + 1,
                total_observations=state.total_observations + 1
            )

        # Choose between new cluster and same cluster
        return jax.lax.cond(
            is_new_cluster,
            handle_new_cluster,
            accumulate_in_current_cluster
        )

    def estimate(self, state: SwitchbackTPGEstimatorState):
        """
        Compute final ATE estimate using policy gradient.

        ATE = mean(treatment_truncated_returns) - mean(control_truncated_returns)

        Includes:
        1. All completed clusters (those that have exited the window)
        2. Clusters in the sliding window (incomplete look-ahead)
        3. The current cluster being accumulated (if any)
        """

        def compute_with_clusters():
            """Compute estimate when we have clusters."""

            def compute_k_zero_estimate():
                """For k=0, only completed clusters contribute."""

                # Add contribution from current cluster if it exists
                def add_current_cluster():
                    avg_reward = state.current_cluster_reward_sum / jnp.maximum(state.current_cluster_count, 1)

                    # Add to treatment or control based on action
                    new_treatment_sum = jnp.where(
                        state.current_cluster_action == 1,
                        state.treatment_sum + avg_reward,
                        state.treatment_sum
                    )
                    new_treatment_count = jnp.where(
                        state.current_cluster_action == 1,
                        state.treatment_count + 1,
                        state.treatment_count
                    )
                    new_control_sum = jnp.where(
                        state.current_cluster_action == 0,
                        state.control_sum + avg_reward,
                        state.control_sum
                    )
                    new_control_count = jnp.where(
                        state.current_cluster_action == 0,
                        state.control_count + 1,
                        state.control_count
                    )

                    return (new_treatment_sum, new_treatment_count, new_control_sum, new_control_count)

                def just_complete():
                    return (state.treatment_sum, state.treatment_count, state.control_sum, state.control_count)

                treatment_sum, treatment_count, control_sum, control_count = jax.lax.cond(
                    state.current_cluster_count > 0,
                    add_current_cluster,
                    just_complete
                )

                # Compute difference in means
                treatment_mean = treatment_sum / jnp.maximum(treatment_count, 1)
                control_mean = control_sum / jnp.maximum(control_count, 1)

                return treatment_mean - control_mean

            def compute_k_positive_estimate():
                """For k>0, combine complete, window, and current cluster estimates."""
                # Start with completed sums
                treatment_sum_total = state.treatment_sum
                treatment_count_total = state.treatment_count
                control_sum_total = state.control_sum
                control_count_total = state.control_count

                # Add estimates from incomplete clusters in sliding window
                window_size = jnp.minimum(state.total_clusters, self.k)

                # Process each position in window
                def add_partial_estimate(i, acc):
                    treat_sum, treat_count, ctrl_sum, ctrl_count = acc

                    # Calculate partial truncated return for cluster at position i
                    # This includes average rewards from position i to end of current window
                    buffer_size = max(self.k, 1)
                    remaining_avg_rewards = jnp.sum(
                        jnp.where(
                            jnp.arange(buffer_size) >= i,
                            jnp.where(jnp.arange(buffer_size) < window_size, state.avg_reward_history, 0.0),
                            0.0
                        )
                    )

                    # Add current cluster's average if it exists
                    def add_current():
                        current_avg = state.current_cluster_reward_sum / jnp.maximum(state.current_cluster_count, 1)
                        return remaining_avg_rewards + current_avg

                    def just_remaining():
                        return remaining_avg_rewards

                    total_return = jax.lax.cond(
                        state.current_cluster_count > 0,
                        add_current,
                        just_remaining
                    )

                    # Add to treatment or control based on action
                    action = state.action_history[i]

                    new_treat_sum = jnp.where(
                        action == 1,
                        treat_sum + total_return,
                        treat_sum
                    )
                    new_treat_count = jnp.where(
                        action == 1,
                        treat_count + 1,
                        treat_count
                    )
                    new_ctrl_sum = jnp.where(
                        action == 0,
                        ctrl_sum + total_return,
                        ctrl_sum
                    )
                    new_ctrl_count = jnp.where(
                        action == 0,
                        ctrl_count + 1,
                        ctrl_count
                    )

                    return (new_treat_sum, new_treat_count, new_ctrl_sum, new_ctrl_count)

                # Add incomplete estimates using JAX fori_loop for efficiency
                (treatment_sum_total, treatment_count_total,
                 control_sum_total, control_count_total) = jax.lax.fori_loop(
                    0, window_size, add_partial_estimate,
                    (treatment_sum_total, treatment_count_total, control_sum_total, control_count_total)
                )

                # If current cluster exists and window is empty, add it separately
                def add_current_cluster_when_no_window():
                    avg_reward = state.current_cluster_reward_sum / jnp.maximum(state.current_cluster_count, 1)

                    new_treat_sum = jnp.where(
                        state.current_cluster_action == 1,
                        treatment_sum_total + avg_reward,
                        treatment_sum_total
                    )
                    new_treat_count = jnp.where(
                        state.current_cluster_action == 1,
                        treatment_count_total + 1,
                        treatment_count_total
                    )
                    new_ctrl_sum = jnp.where(
                        state.current_cluster_action == 0,
                        control_sum_total + avg_reward,
                        control_sum_total
                    )
                    new_ctrl_count = jnp.where(
                        state.current_cluster_action == 0,
                        control_count_total + 1,
                        control_count_total
                    )

                    return (new_treat_sum, new_treat_count, new_ctrl_sum, new_ctrl_count)

                (treatment_sum_final, treatment_count_final,
                 control_sum_final, control_count_final) = jax.lax.cond(
                    (window_size == 0) & (state.current_cluster_count > 0),
                    add_current_cluster_when_no_window,
                    lambda: (treatment_sum_total, treatment_count_total, control_sum_total, control_count_total)
                )

                # Compute difference in means
                treatment_mean = treatment_sum_final / jnp.maximum(treatment_count_final, 1)
                control_mean = control_sum_final / jnp.maximum(control_count_final, 1)

                return treatment_mean - control_mean

            # Choose estimation method based on k value
            return jax.lax.cond(
                self.k == 0,
                compute_k_zero_estimate,
                compute_k_positive_estimate
            )

        # Check if we have any data
        has_data = (state.total_clusters > 0) | (state.current_cluster_count > 0)

        return jax.lax.cond(
            has_data,
            compute_with_clusters,
            lambda: 0.0
        )
