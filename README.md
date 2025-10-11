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
- `UnitRandomizedDesign`: Random assignment at each time step. Each observation is its own cluster.
- `SwitchbackDesign`: Time-based clustering where treatment assignment switches at fixed time intervals in seconds (configured via `switch_every` parameter). Uses real event time for efficient O(1) cluster assignment.
- `SpatioClusterDesign`: Spatiotemporal clustering that combines time periods with spatial zones. Creates clusters based on both when and where events occur, enabling spatial interference analysis. Uses ascending time order for efficient O(1) operations.
- `ClusterRandomizedDesign`: Base class for cluster-based designs that provides common cluster assignment infrastructure.

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

*Universal Estimators (work with all designs):*
- `NaiveEstimator` (`naive`): Simple inverse probability weighting (IPW) estimator. Computes treatment effect as the difference between IPW-weighted outcomes for treated vs control observations.
- `DMEstimator` (`dm`): Difference-in-means estimator. Computes simple average difference between treated and control groups without weighting.
- `TPGEstimator` (`tpg`): Truncated policy gradient estimator with configurable horizon `k`. Uses current reward plus next `k` rewards with IPW weighting to reduce variance. [[Johari R, Peng T, Xing W., 2025]](https://arxiv.org/abs/2506.05308)

*SpatioClusterDesign-specific Estimators:*
- `DNEstimator` (`dn`): Difference-in-Networks estimator. Accounts for spatial and temporal spillovers by adjusting estimates based on nearby clusters (configurable via `max_spatial_distance` and `lookahead_steps`). [[Peng T, Ye N, Zheng A., 2025]](https://arxiv.org/abs/2503.02271)
- `DynkinEstimator` (`dynkin`): Dynkin-based value difference estimator using LSTD. Learns state-value functions to estimate treatment effects by comparing expected values across treatment groups. [[Farias, Vivek, et al., 2022]](https://arxiv.org/abs/2206.02371)
- `LSTDLambdaEstimator` (`lstd_lambda`): Off-policy evaluation using LSTD-λ. Maintains separate LSTD systems for treated and control groups with eligibility traces (λ parameter). [[Farias, Vivek, et al., 2022]](https://arxiv.org/abs/2206.02371)

*SwitchbackDesign-specific Estimators:*
- `SWTPGEstimator` (`sw_tpg`): Switchback-specific truncated policy gradient estimator operated under interval averaged outcomes. [[Johari R, Peng T, Xing W., 2025]](https://arxiv.org/abs/2506.05308)

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

You can select multiple estimators to compute in a single run using the syntax [here](https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group/)

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

## Computing True ATE and Analyzing Results

### Computing True ATE (`compute-ate.py`)

The true Average Treatment Effect (ATE) can be computed by running both policies independently across many environments and averaging the difference:

```bash
python compute-ate.py
```

This script:
1. Runs policy A and policy B separately on the same environment seeds
2. Collects rewards for each policy across multiple trials (default: 100 trials with 500,000 events each)
3. Computes the true ATE as the mean difference: `ATE = mean(B) - mean(A)`
4. Saves results to a CSV file (default: `ate.csv`)

The true ATE serves as the ground truth for evaluating estimator performance.

### Analyzing Estimator Performance (`plot_estimator_results.py`)

After running experiments, analyze estimator convergence and bias using:

```bash
python plot_estimator_results.py
```

This generates two key plots:

1. **Convergence Plot** (`estimator_convergence.png`):
   - Shows mean estimates over time with 95% confidence intervals
   - Includes the true ATE as a reference line
   - Helps visualize how quickly each estimator converges to the true value

2. **Distribution Plot** (`estimator_comparison.png`):
   - Box plots showing the distribution of final estimates across environments
   - Displays mean estimates and bias for each estimator
   - Useful for comparing estimator variance and systematic bias

The script also prints summary statistics including:
- Final estimates (mean, std, range) for each estimator
- Convergence trends in the final 10% of time steps
- Bias analysis comparing estimates to the true ATE

## Available Environments

- **ABTestEnv**: Simple A/B testing environment
- **XPRidesharePricingEnv**: Rideshare pricing experiment using or-gymnax

## Data Files

The `data/` directory contains spatial data files required for spatiotemporal experimental designs:

- **`manhattan-nodes.parquet`**: Network nodes for Manhattan with geographic coordinates (latitude/longitude). Used to map event locations to spatial zones.
- **`taxi-zones.parquet`**: Taxi zone definitions for Manhattan. Provides zone identifiers and boundaries for spatial clustering.

These files are used by:
- `SpatioClusterDesign`: For creating spatiotemporal clusters based on geographic zones
- `DNEstimator`: For computing spatial adjacency matrices and accounting for spatial spillovers
- `DynkinEstimator` and `LSTDLambdaEstimator`: For extracting state representations (available cars per zone)

The design loads these files during initialization and computes:
- Zone assignments for each network node
- Distance matrices between zone centroids (using haversine distance)
- Spatial adjacency relationships for interference modeling

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
