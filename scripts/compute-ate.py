import jax
import jax.numpy as jnp
from functools import partial
import pandas as pd
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from xp_gym.io import to_csv

METRICS = [
    "reward",
    "is_unfulfill",
    "marginal_cost",
    "utilization",
    "pct_cars_on_trip",
]


def stepper(env, env_params, policy, carry, key):
    obs, state, accum = carry
    key, policy_key = jax.random.split(key)
    action, action_info = policy.apply(env_params, dict(), obs, policy_key)
    new_obs, new_state, reward, _, info = env.step(
        key, state, action, env_params
    )
    is_unfulfill = info["is_unfulfill"].astype(jnp.float32)
    marginal_cost = info["marginal_cost"].astype(jnp.float32)
    utilization = info["utilization"].astype(jnp.float32)
    pct_cars_on_trip = info["pct_cars_on_trip"].astype(jnp.float32)
    new_accum = {
        "reward_sum": accum["reward_sum"] + reward.astype(jnp.float32),
        "is_unfulfill_sum": accum["is_unfulfill_sum"] + is_unfulfill,
        "marginal_cost_sum": accum["marginal_cost_sum"]
        + marginal_cost * (1 - is_unfulfill),
        "fulfilled_count": accum["fulfilled_count"] + (1 - is_unfulfill),
        "utilization_sum": accum["utilization_sum"] + utilization,
        "pct_cars_on_trip_sum": accum["pct_cars_on_trip_sum"]
        + pct_cars_on_trip,
    }
    return (new_obs, new_state, new_accum), None


def run(env, env_params, policy, key, n_steps):
    keys = jax.random.split(key, n_steps)
    obs, state = env.reset(key, env_params)
    init_accum = {
        "reward_sum": jnp.array(0.0),
        "is_unfulfill_sum": jnp.array(0.0),
        "marginal_cost_sum": jnp.array(0.0),
        "fulfilled_count": jnp.array(0.0),
        "utilization_sum": jnp.array(0.0),
        "pct_cars_on_trip_sum": jnp.array(0.0),
    }
    final, _ = jax.lax.scan(
        partial(stepper, env, env_params, policy),
        (obs, state, init_accum),
        keys,
    )
    _, _, accum = final
    return {
        "reward": accum["reward_sum"] / n_steps,
        "is_unfulfill": accum["is_unfulfill_sum"] / n_steps,
        "marginal_cost": jnp.where(
            accum["fulfilled_count"] > 0,
            accum["marginal_cost_sum"] / accum["fulfilled_count"],
            jnp.nan,
        ),
        "utilization": accum["utilization_sum"] / n_steps,
        "pct_cars_on_trip": accum["pct_cars_on_trip_sum"] / n_steps,
    }


vmap_run = jax.vmap(run, in_axes=(None, None, None, 0, None))


def run_batch(env, env_params, A, B, key, n_steps, batch_size):
    keys = jax.random.split(key, batch_size)
    results_A = vmap_run(env, env_params, A, keys, n_steps)
    results_B = vmap_run(env, env_params, B, keys, n_steps)
    rows = []
    for metric in METRICS:
        for v in results_A[metric]:
            rows.append({"treatment": "A", "metric": metric, "value": float(v)})
        for v in results_B[metric]:
            rows.append({"treatment": "B", "metric": metric, "value": float(v)})
    return rows


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    seed = cfg.run.seed
    n_steps = cfg.ate.n_steps
    k = cfg.ate.k
    batch_size = cfg.ate.batch_size
    output = cfg.ate.output_path

    xp_env = instantiate(cfg.env)
    env = xp_env.env
    env_params_dict = dict(cfg.env_params)
    env_params = env.default_params.replace(**env_params_dict["env_params"])

    A = xp_env.policy_A
    B = xp_env.policy_B

    n_batches = k // batch_size
    jax.debug.print(f"n_batches: {n_batches}")
    keys = jax.random.split(jax.random.PRNGKey(seed), n_batches)
    all_rows = []
    for key in tqdm(keys):
        all_rows.extend(
            run_batch(env, env_params, A, B, key, n_steps, batch_size)
        )

    results_df = pd.DataFrame(all_rows)
    to_csv(results_df, output)

    reward_A = results_df[
        (results_df["treatment"] == "A") & (results_df["metric"] == "reward")
    ]["value"].mean()
    reward_B = results_df[
        (results_df["treatment"] == "B") & (results_df["metric"] == "reward")
    ]["value"].mean()
    ate = reward_B - reward_A
    print(f"Average ATE (B - A): {ate:.6f}")


if __name__ == "__main__":
    main()
