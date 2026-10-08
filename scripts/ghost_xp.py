"""
Run xp_gym estimators on the exact ghost_sim.py demand scenario.

Reproduces the synthetic 3-node, 1-car rideshare environment with fixed
events from or-gymnax/scripts/ghost_sim.py, then runs xp_gym estimators
on top of it.

Usage:
    python scripts/ghost_xp.py
    python scripts/ghost_xp.py --n_steps=30 --seed=42
"""
import argparse
import jax
import jax.numpy as jnp
import pandas as pd

from or_gymnax import rideshare_pool as rp
from or_gymnax.rideshare import RideshareEvent
import chex
from xp_gym.environments.environment import XPEnvironment, XPEnvParams
from xp_gym.designs.design import UnitRandomizedDesign
from xp_gym.estimators.naive import NaiveEstimator
from xp_gym.estimators.dn import LimitedMemoryDNEstimator
from xp_gym.estimators.network import GhostInterferenceNetwork
from xp_gym.observation import Observation
from xp_gym.io import to_csv


# ── Demand scenario from ghost_sim.py ────────────────────────────────────────
N_CARS = 1
N_NODES = 3
N_EVENTS = 30
MAX_ACTIVE_TRIPS = 2
MAX_GHOSTS = 2 * N_NODES  # 6
GHOST_MAX_LIFESPAN = 50
THRESHOLD_A = 0.0
THRESHOLD_B = 0.6

_k1, _ = jax.random.split(jax.random.PRNGKey(1), 2)
_event_times = jnp.cumsum(jax.random.randint(_k1, (N_EVENTS,), 1, 5))
_event_srcs = jnp.zeros(N_EVENTS, dtype=jnp.int32)
_event_dests = (1 + jnp.arange(N_EVENTS).astype(jnp.int32)) % 2 + 1

EVENTS = RideshareEvent(t=_event_times, src=_event_srcs, dest=_event_dests)

DISTANCES = jnp.array(
    [[0, 1, 2],
     [1, 0, 1],
     [2, 1, 0]],
    dtype=jnp.int32,
) * 100
# ─────────────────────────────────────────────────────────────────────────────


class GhostSimXPEnvironment(XPEnvironment):
    """XPEnvironment that initialises car waypoints[:, 0] = 1 on reset,
    matching ghost_sim.py's manual fixup."""

    def reset_env(self, key: chex.PRNGKey, params: XPEnvParams):
        obs, state = super().reset_env(key, params)
        state = state.replace(waypoints=state.waypoints.at[:, 0].set(1))
        obs = self.env.get_obs(state, params.env_params)
        return obs, state


def build_env_and_params():
    inner_env = rp.RidesharePoolDispatch(
        n_cars=N_CARS, n_nodes=N_NODES, n_events=N_EVENTS
    )
    xp_env = GhostSimXPEnvironment(
        env=inner_env,
        policy_A=rp.GreedyPolicy(
            n_cars=N_CARS, temperature=0.1, savings_threshold=THRESHOLD_A
        ),
        policy_B=rp.GreedyPolicy(
            n_cars=N_CARS, temperature=0.1, savings_threshold=THRESHOLD_B
        ),
    )
    inner_params = rp.EnvParams(
        events=EVENTS,
        distances=DISTANCES,
        n_cars=N_CARS,
        max_active_trips=MAX_ACTIVE_TRIPS,
        max_ghosts=MAX_GHOSTS,
        ghost_max_lifespan=GHOST_MAX_LIFESPAN,
    )
    xp_params = XPEnvParams(
        max_steps_in_episode=N_EVENTS,
        env_params=inner_params,
    )
    return xp_env, xp_params


def main():
    parser = argparse.ArgumentParser(
        description="Run xp_gym estimators on the ghost_sim demand scenario"
    )
    parser.add_argument(
        "--n_steps", type=int, default=N_EVENTS,
        help="Total steps per environment (default: all events)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="output/ghost_xp.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    env, env_params = build_env_and_params()
    design = UnitRandomizedDesign(p=0.5)
    estimators = {
        "naive": NaiveEstimator(use_known_actions=True),
        "naive_ipw": NaiveEstimator(use_known_actions=False),
        "dn_ghost": LimitedMemoryDNEstimator(
            network=GhostInterferenceNetwork(),
            window_size=min(200, args.n_steps),
            use_known_actions=True,
            baseline=0.0,
        ),
    }

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)

    est_states = {
        name: est.reset(rng, env, env_params, design)
        for name, est in estimators.items()
    }
    design_state = design.reset(rng, env_params)

    records = []
    for step_idx in range(args.n_steps):
        rng, step_rng = jax.random.split(rng)
        action, design_info = design.assign_treatment(design_state, state)
        obs, next_state, reward, done, info = env.step(step_rng, state, action, env_params)

        xp_obs = Observation(
            obs=obs, action=action, reward=reward, info=info, design_info=design_info
        )
        est_states = {
            name: est.update(env, env_params, design, est_states[name], xp_obs)
            for name, est in estimators.items()
        }
        design_state = design.update(design_state, xp_obs)

        estimates = {
            name: float(est.estimate(env, env_params, design, est_states[name]))
            for name, est in estimators.items()
        }

        # info["action_A"] = canonical car (the one actually dispatched, or -1)
        # info["action_B"] = counterfactual car (not dispatched, ghost tracking only)
        # When treatment=0: canonical=A(0.0), cf=B(0.6)
        # When treatment=1: canonical=B(0.6), cf=A(0.0)
        canonical_car = int(info.get("action_A", -1))
        cf_car = int(info.get("action_B", -1))
        record = {
            "step": step_idx,
            "event_t": int(state.event.t),
            "event_src": int(state.event.src),
            "event_dest": int(state.event.dest),
            "treatment": int(action),
            "canonical_car": canonical_car,
            "cf_car": cf_car,
            "reward": float(reward),
            "n_active_ghosts": int(info.get("n_active_ghosts", 0)),
            "n_ghost_triggers": int(info.get("n_ghost_triggers", 0)),
            **{f"est_{name}": v for name, v in estimates.items()},
        }
        records.append(record)

        est_str = "  ".join(f"{name}={v:8.4f}" for name, v in estimates.items())
        print(
            f"  step={step_idx:3d}  t={record['event_t']:4d}"
            f"  src={record['event_src']} dest={record['event_dest']}"
            f"  z={record['treatment']}  canonical={canonical_car:2d}  cf={cf_car:2d}"
            f"  reward={record['reward']:6.1f}"
            f"  ghosts={record['n_active_ghosts']:2d}  triggers={record['n_ghost_triggers']}"
            f"  {est_str}"
        )

        state = next_state

    results_df = pd.DataFrame(records)
    to_csv(results_df, args.output)
    print(f"\nDone. Results written to {args.output}")


if __name__ == "__main__":
    main()
