# XP Gym

A JAX-based experimentation framework for running A/B tests and evaluating causal inference estimators in reinforcement learning environments.

## Overview

XP Gym provides a modular framework for conducting experiments that compare two policies (A and B) in various environments while tracking the performance of different causal inference estimators. The framework is built on JAX for high-performance computing and supports various experimental designs and estimation methods.

## Core Architecture

The framework is built around three main abstract classes that define the experimental structure:

### 1. Environment (`xp_gym.environments.environment`)

The `XPEnvironment` class wraps any Gymnax environment to facilitate A/B testing between two policies.

**Key Components:**
- `XPEnvParams`: Configuration for the experimental environment including policy parameters
- `XPEnvironment`: Main wrapper class that reduces action space to binary (policy A vs B)

**Core API:**
```python
class XPEnvironment(environment.Environment):
    def __init__(self, env: Environment, policy_A: Callable, policy_B: Callable)
    def step_env(self, key, state, action, params) -> (obs, state, reward, done, info)
    def reset_env(self, key, params) -> (obs, state)
    def action_space() -> Discrete(2)  # Binary choice: A or B
```

### 2. Experimental Design (`xp_gym.designs.design`)

Defines how treatment assignment is handled throughout the experiment.

**Key Components:**
- `Design`: Abstract base class for experimental designs
- `DesignState`: Stores the dynamic state of the design
- `DesignInfo`: Information about treatment assignment (e.g., probability) which is later passed to the estimator.

**Core API:**
```python
class Design:
    def reset(self, rng, env_params) -> DesignState
    def update(self, state, obs) -> DesignState  
    def assign_treatment(self, design_state, env_state) -> (treatment, DesignInfo)
```

**Built-in Designs:**
- `UnitRandomizedDesign`: Random assignment at each time step
- `SwitchbackDesign`: Time-based clustering with configurable frequency
- `ClusterRandomizedDesign`: Base class for cluster-based designs

### 3. Estimators (`xp_gym.estimators.estimator`)

Implements various causal inference methods to estimate treatment effects.

**Key Components:**
- `Estimator`: Abstract base class for all estimators
- `EstimatorState`: Stores the internal state of each estimator

**Core API:**
```python
class Estimator:
    def reset(self, rng, env_params) -> EstimatorState
    def update(self, state, obs) -> EstimatorState
    def estimate(self, state) -> float  # Treatment effect estimate
```

**Built-in Estimators:**
- `NaiveEstimator`: Simple difference-in-means with inverse probability weighting
- Additional estimators available in the `estimators/` directory

### 4. Observation Structure (`xp_gym.observation`)

The `Observation` class standardizes data flow between components:

```python
@struct.dataclass
class Observation:
    obs: Float[Array, "o_dim"]      # Environment observation
    action: Bool[Array, "1"]         # Treatment assignment (0=A, 1=B)  
    reward: Float[Array, "1"]        # Reward/outcome
    design_info: Any                 # Design-specific metadata
```

### 5. Simulation Engine (`xp_gym.simulator`)

The core simulation functions orchestrate the interaction between all components:

**Key Functions:**
- `step()`: Single environment step with estimator updates
- `step_n_and_estimate()`: Run N steps then compute estimates
- `simulate()`: Full simulation with periodic estimation

## Running Experiments

### Using scripts/run.py

The main entry point for running experiments is `scripts/run.py`, which uses Hydra for configuration management:

```bash
# Run with default configuration
python scripts/run.py

# Override specific parameters
python scripts/run.py run.n_steps=5000 run.seed=123

# Use different configuration file
python scripts/run.py --config-name=my_config
```

### Configuration Structure

The configuration is defined in `scripts/config/config.yaml`:

```yaml
run:
  output_path: "output/results.csv"
  n_envs: 4                    # Number of parallel environments
  n_steps: 10000              # Steps per environment
  estimate_every_n_steps: 1000 # Estimation frequency
  seed: 42

env:
  _target_: xp_gym.environments.abtest.ABTestEnv
  
env_params:
  noise_std: 10

design:
  _target_: xp_gym.designs.design.UnitRandomizedDesign
  p: 0.5                      # Treatment probability

estimators:
  naive:
    _target_: xp_gym.estimators.naive.NaiveEstimator
```

### Output

Results are saved as CSV files containing:
- `env_id`: Environment instance identifier
- `steps`: Time step of the estimate
- Estimator-specific columns with treatment effect estimates

## Available Environments

- **ABTestEnv**: Simple A/B testing environment
- **XPRidesharePricingEnv**: Rideshare pricing experiment using or-gymnax

## Dependencies

Key dependencies include:
- JAX/Flax: High-performance computing
- Gymnax: RL environment interface
- Hydra: Configuration management
- or-gymnax: Operations research environments
- Chex: JAX utilities

## Installation

```bash
# Install using Poetry
poetry install

# Or install dependencies manually
pip install jax flax gymnax hydra-core chex jaxtyping pyarrow
pip install git+https://github.com/atzheng/or-gymnax.git
```

## Extending the Framework

### Adding New Estimators

1. Inherit from `Estimator` and `EstimatorState`
2. Implement `reset()`, `update()`, and `estimate()` methods
3. Add to configuration file

### Adding New Designs  

1. Inherit from `Design` and optionally `DesignState`
2. Implement `assign_treatment()` method
3. Override `reset()` and `update()` if needed

### Adding New Environments

1. Inherit from `XPEnvironment`
2. Define policies A and B in the constructor
3. Configure in YAML file
