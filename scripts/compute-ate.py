import jax
from functools import partial
from or_gymnax.rideshare import (
    ManhattanRidesharePricing,
    SimplePricingPolicy,
)
import pandas as pd
from tqdm import tqdm
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def stepper(env, env_params, policy, carry, key):
    obs, state, total_reward = carry
    key, policy_key = jax.random.split(key)
    action, action_info = policy.apply(env_params, dict(), obs, policy_key)
    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)
    return (new_obs, new_state, total_reward + reward), None


def run(env, env_params, policy, key, n_steps):
    keys = jax.random.split(key, n_steps)
    obs, state = env.reset(key, env_params)
    final, _ = jax.lax.scan(
        partial(stepper, env, env_params, policy),
        (obs, state, 0),
        keys,
    )
    _, _, total_reward = final
    return total_reward / n_steps


vmap_run = jax.vmap(run, in_axes=(None, None, None, 0, None))


def run_batch(env, env_params, A, B, key, n_steps, batch_size):
    keys = jax.random.split(key, batch_size)
    results_A = vmap_run(env, env_params, A, keys, n_steps)
    results_B = vmap_run(env, env_params, B, keys, n_steps)
    return {
        "A": results_A,
        "B": results_B,
    }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # Extract ATE configuration
    seed = cfg.run.seed
    n_cars = cfg.env.n_cars
    n_events = cfg.ate.n_steps
    k = cfg.ate.k
    batch_size = cfg.ate.batch_size
    output = cfg.ate.output_path
    
    # Get environment parameters from config
    w_price = cfg.env_params.env_params.w_price
    w_eta = cfg.env_params.env_params.w_eta
    w_intercept = cfg.env_params.env_params.w_intercept
    price_per_distance_A = cfg.env.price_per_distance_A
    price_per_distance_B = cfg.env.price_per_distance_B
    
    env = ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events)
    env_params = env.default_params
    env_params = env_params.replace(
        w_price=w_price, w_eta=w_eta, w_intercept=w_intercept
    )
    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=price_per_distance_A)
    B = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=price_per_distance_B)

    n_batches = k // batch_size
    jax.debug.print(f"n_batches: {n_batches}")
    keys = jax.random.split(jax.random.PRNGKey(seed), n_batches)
    results = [
        run_batch(env, env_params, A, B, key, n_events, batch_size)
        for key in tqdm(keys)
    ]

    # Check if output directory exists and create if not
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Convert list of dicts into a DataFrame
    results_df = pd.concat(map(pd.DataFrame, results))
    results_df.to_csv(output, index=False)

    # --- Compute average ATE ---
    mean_A = results_df["A"].mean()
    mean_B = results_df["B"].mean()
    ate = mean_B - mean_A
    print(f"Average ATE (B - A): {ate:.6f}")


if __name__ == "__main__":
    main()
