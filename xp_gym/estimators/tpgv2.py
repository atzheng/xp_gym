#!/usr/bin/env python3

from network import LimitedMemoryNetworkEstimatorState, LimitedMemoryNetworkEstimator

class SWTPGEstimatorState(EstimatorState):
    lms: LimitedMemoryNetworkEstimatorState
    tr_estimate: float
    tr_count: int
    co_estimate: float
    co_count: int
    

class SWTPGEstimator(LimitedMemoryNetworkEstimator):
    """
    The SW-TPG estimator for experiments with clustered designs
    and a known interference graph.

    WARNING: This implementation implicitly assumes that an experimental
    unit A can only suffer interference effects from another unit B if
    B arrives in some fixed window BEFORE A.
    """
    k: int  # Switchback-cluster-level sliding window size

    def reset(self, rng, env, env_params, design):
        # TODO
        lms = super().reset(rng, env, env_params, design)
        

    def update(
        self,
        env,
        env_params,
        design,
        state: LimitedMemoryNetworkEstimatorState,
        obs,
    ):
        """Update SW-TPG estimator using network interference structure."""
    
        # Extract observation info
        cluster_id = obs.design_info.cluster_id
        p = obs.design_info.p
        z = obs.action.astype(jnp.float32)
        reward = obs.reward

        zc = state.design_cluster_treatments
        pc = state.design_cluster_treatment_probs
        interference_mask = self.interference_mask(env, env_params, state, obs)

        treated_mask = jnp.where(interference_mask, zc == 1, True)
        control_mask = jnp.where(interference_mask, zc == 0, True)

        all_tr_ipw = jnp.prod(jnp.where(treated_mask, 1 / pc, 1.0)) * z / p
        all_co_ipw = jnp.prod(jnp.where(control_mask, 1 / (1 - pc), 1.0)) * (1 - z) / (1 - p)
        ipw = all_tr_ipw - all_co_ipw

        estimate = state.estimate + ipw * reward
        return (
            super()
            .update(env, env_params, design, state, obs)
            .replace(estimate=estimate)
        )
    
    def estimate(self, env, env_params, design, state):
        return state.estimate / state.t
