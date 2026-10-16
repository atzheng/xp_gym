"""TSR estimators of Johari, Li, Liskovich & Weintraub (arXiv:2002.05670).

Requires the TwoSidedRandomizedDesign + XPRidesharePoolTSREnv.  With
outcome = reward (profit) of a request, the "rate of booking" (17) becomes
    Q_ij = (1/T) sum_t r_t 1[z_C(t) = i, z_L(car_t) = j]
(unfulfilled requests have r_t = 0 and no car).  Outputs (see `labels`):
  tsrn            naive TSR (21): Q11/(aC aL) - (Q01+Q10+Q00)/(1 - aC aL)
  cr, lr          TSR-normalised CR / LR estimators (the two terms of (27)):
                  cr = Q11/(aC aL) - Q01/((1-aC) aL),  lr = Q11/(aC aL) - Q10/(aC (1-aL))
  tsri{k}_b{beta} TSRI-k (28) for k in `ks`, beta in `betas`:
      beta     [cr - k (1-beta) (Q00/((1-aC)(1-aL)) - Q01/((1-aC) aL))]
    + (1-beta) [lr - k  beta    (Q00/((1-aC)(1-aL)) - Q10/(aC (1-aL)))]
The paper sets beta = exp(-lambda/tau) (market balance); here beta is a grid so
it can be chosen afterwards (include exp(-m) for your market-balance guess m).
a_C, a_L are the REALIZED treated shares: a_C = fraction of (post-burn-in) requests
with z_C = 1, a_L = fraction of cars with z_L = 1 (exactly round(a_L n_cars)/n_cars
under the design's complete randomization).  Normalizing by the nominal design
probabilities instead adds an error of order R (a_L_hat - a_L) / a_L.
`burn_in` drops steps t < burn_in (the fleet starts empty); T counts steps after it.
"""
from typing import Tuple

from flax import struct
import jax.numpy as jnp

from xp_gym.estimators.estimator import Estimator, EstimatorState


@struct.dataclass
class TSRIState(EstimatorState):
    Q: jnp.ndarray  # (2, 2) sum of rewards by (customer cond i, listing cond j)
    t: jnp.ndarray
    nC: jnp.ndarray  # number of post-burn-in requests with z_C = 1
    aL: jnp.ndarray  # realized share of treated cars


def tsr_estimates(Q, T, aC, aL, betas, ks):
    """Q: (2, 2) reward sums; returns [tsrn, cr, lr, tsri{k}_b{beta}...]."""
    q = Q / T
    q11 = q[1, 1] / (aC * aL)
    q01 = q[0, 1] / ((1 - aC) * aL)
    q10 = q[1, 0] / (aC * (1 - aL))
    q00 = q[0, 0] / ((1 - aC) * (1 - aL))
    tsrn = q11 - (q[0, 1] + q[1, 0] + q[0, 0]) / (1 - aC * aL)
    cr, lr = q11 - q01, q11 - q10
    out = [tsrn, cr, lr]
    for k in ks:
        for b in betas:
            out.append(b * (cr - k * (1 - b) * (q00 - q01))
                       + (1 - b) * (lr - k * b * (q00 - q10)))
    return jnp.stack(out)


@struct.dataclass
class TSRIEstimator(Estimator):
    betas: Tuple[float, ...] = struct.field(
        pytree_node=False, default=(0.0, 0.25, 0.3679, 0.5, 0.75, 1.0))
    ks: Tuple[float, ...] = struct.field(pytree_node=False, default=(1.0, 2.0))
    burn_in: int = struct.field(pytree_node=False, default=0)

    def __post_init__(self):
        for k in ("betas", "ks"):
            object.__setattr__(self, k, tuple(float(x) for x in getattr(self, k)))

    @property
    def labels(self):
        return ["tsrn", "cr", "lr"] + [f"tsri{k:g}_b{b:g}" for k in self.ks for b in self.betas]

    def reset(self, rng, env, env_params, design):
        return TSRIState(Q=jnp.zeros((2, 2), jnp.float32), t=jnp.array(0),
                         nC=jnp.array(0.0, jnp.float32), aL=jnp.array(0.5, jnp.float32))

    def update(self, env, env_params, design, state, obs):
        car = obs.info["car"].reshape(-1)[0]
        zC = obs.design_info.z_C.astype(jnp.int32)
        zL = obs.design_info.z_L[jnp.maximum(car, 0)].astype(jnp.int32)
        r = jnp.where(car >= 0, obs.reward, 0.0).astype(jnp.float32)
        live = state.t >= self.burn_in
        r = jnp.where(live, r, 0.0)
        aL = obs.design_info.z_L.astype(jnp.float32).mean()
        return TSRIState(Q=state.Q.at[zC, zL].add(r), t=state.t + 1,
                         nC=state.nC + (live & (zC == 1)).astype(jnp.float32), aL=aL)

    def estimate(self, env, env_params, design, state):
        T = jnp.maximum(state.t - self.burn_in, 1).astype(jnp.float32)
        aC = jnp.clip(state.nC / T, 1e-6, 1 - 1e-6)
        aL = jnp.clip(state.aL, 1e-6, 1 - 1e-6)
        return tsr_estimates(state.Q, T, aC, aL, self.betas, self.ks)
