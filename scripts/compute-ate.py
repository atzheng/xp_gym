import jax
from functools import partial
import pandas as pd
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from xp_gym.io import to_csv


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
    results = [
        run_batch(env, env_params, A, B, key, n_steps, batch_size)
        for key in tqdm(keys)
    ]

    results_df = pd.concat(map(pd.DataFrame, results))
    to_csv(results_df, output)

    mean_A = results_df["A"].mean()
    mean_B = results_df["B"].mean()
    ate = mean_B - mean_A
    print(f"Average ATE (B - A): {ate:.6f}")


if __name__ == "__main__":
    main()
