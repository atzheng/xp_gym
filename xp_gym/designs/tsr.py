"""Two-sided randomization (TSR) design of Johari, Li, Liskovich & Weintraub,
"Experimental Design in Two-Sided Platforms: An Analysis of Bias"
(arXiv:2002.05670, Section 5.1).

Mapping onto the pooled dispatch env (xp_gym.environments.rideshare_pool_tsr):
  * customers <-> ride requests: each request is treated independently
    w.p. a_C (z_C ~ Bern(a_C), fresh at every step);
  * listings  <-> cars: each car is treated w.p. a_L, drawn ONCE per run at
    reset (z_L ~ Bern(a_L)^n_cars) and fixed thereafter, as in the paper;
  * the intervention applies only to treated-customer x treated-listing
    interactions (paper eqs. (13)-(14)): for request t, car j is screened with
    arm B's pooling savings threshold iff z_C(t) = 1 and z_L(j) = 1, and with
    arm A's threshold otherwise.  The greedy dispatcher then picks the cheapest
    eligible car among all cars (each judged by its own threshold).

The env action is therefore the per-car vector z_C(t) * z_L (bool, (n_cars,)).
Global control (a_C=0 or a_L=0) is the all-A policy and global treatment
(a_C=a_L=1) is the all-B policy, so the paper's GTE equals our ATE.

`market_balance` (lambda/tau, optional): if set, the effective assignment
probabilities follow the paper's heuristic (26),
    a_C(m) = (1 - e^-m) + a_C e^-m,   a_L(m) = a_L (1 - e^-m) + e^-m,
and `beta` = e^-m is the matching TSRI interpolation weight (Section 6.3).
"""
import math
from typing import Optional

from flax import struct
import jax
import jax.numpy as jnp
from jax import Array

from xp_gym.designs.design import Design, DesignState


def paper_tsr_params(a_C: float, a_L: float, market_balance: Optional[float]):
    """Effective (a_C, a_L, beta) under the paper's eq. (26) (beta = e^{-lambda/tau})."""
    if market_balance is None:
        return float(a_C), float(a_L), None
    e = math.exp(-float(market_balance))
    return (1 - e) + a_C * e, a_L * (1 - e) + e, e


@struct.dataclass
class TSRDesignState(DesignState):
    rng: Array
    z_L: Array  # (n_cars,) bool, fixed listing (car) assignment


@struct.dataclass
class TSRDesignInfo:
    p: float     # = a_C (customer-side probability), for compatibility
    z_C: Array   # request (customer) assignment at this step
    z_L: Array   # (n_cars,) car (listing) assignments
    a_C: float
    a_L: float


@struct.dataclass
class TwoSidedRandomizedDesign(Design):
    a_C: float = struct.field(pytree_node=False, default=0.5)
    a_L: float = struct.field(pytree_node=False, default=0.5)
    market_balance: Optional[float] = struct.field(pytree_node=False, default=None)

    @property
    def eff_a_C(self):
        return paper_tsr_params(self.a_C, self.a_L, self.market_balance)[0]

    @property
    def eff_a_L(self):
        return paper_tsr_params(self.a_C, self.a_L, self.market_balance)[1]

    @property
    def beta(self):
        return paper_tsr_params(self.a_C, self.a_L, self.market_balance)[2]

    def reset(self, rng, env_params) -> TSRDesignState:
        n_cars = env_params.env_params.n_cars
        k_L, k_C = jax.random.split(rng)
        z_L = jax.random.bernoulli(k_L, self.eff_a_L, (n_cars,))
        return TSRDesignState(rng=k_C, z_L=z_L)

    def assign_treatment(self, design_state: TSRDesignState, env_state):
        z_C = jax.random.bernoulli(
            jax.random.fold_in(design_state.rng, env_state.time), self.eff_a_C)
        info = TSRDesignInfo(p=self.eff_a_C, z_C=z_C, z_L=design_state.z_L,
                             a_C=self.eff_a_C, a_L=self.eff_a_L)
        return z_C & design_state.z_L, info

    def update(self, state, obs):
        return state
