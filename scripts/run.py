import os
import jax
import jax.numpy as jnp
import pandas as pd

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from xp_gym.simulator import simulate


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed = cfg.run.seed

    env = instantiate(cfg.env)
    env_params = env.default_params.replace(**cfg.env_params)
    design = instantiate(cfg.design)
    estimators = {k: instantiate(v) for k, v in cfg.estimators.items()}

    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, cfg.run.n_envs)
    vmap_simulate = jax.vmap(
        simulate, in_axes=(None, None, None, None, 0, None, None)
    )

    _, results = vmap_simulate(
        estimators,
        design,
        env,
        env_params,
        rngs,
        cfg.run.n_steps,
        cfg.run.estimate_every_n_steps,
    )

    n_estimates = cfg.run.n_steps // cfg.run.estimate_every_n_steps

    results["env_id"] = jnp.tile(
        jnp.expand_dims(jnp.arange(cfg.run.n_envs), 1), (1, n_estimates)
    )

    results["steps"] = jnp.tile(
        jnp.expand_dims(
            jnp.arange(n_estimates) * cfg.run.estimate_every_n_steps, 0
        ),
        (cfg.run.n_envs, 1),
    )

    results_df = pd.DataFrame.from_dict(
        jax.tree_util.tree_map(lambda x: x.reshape(-1), results)
    )

    # Check if basedir exists and create if not
    os.makedirs(os.path.dirname(cfg.run.output_path), exist_ok=True)
    results_df.to_csv(cfg.run.output_path, index=False)


if __name__ == "__main__":
    run()
