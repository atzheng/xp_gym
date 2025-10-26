## Overview

XP Gym provides a modular, highly performant framework for conducting experiments that compare two policies (A and B) in various environments while tracking the performance of different treatment effect estimators. The framework is built on JAX, runs fast on GPUs, and supports various experimental designs and estimation methods.


## Installation

This repo uses Poetry for dependency management.

```bash
# Install using Poetry
poetry install
# On GPU:
poetry run pip install jax[cuda12]
# Activate environment
eval $(poetry env activate)
```

## Running Experiments
### Using scripts/run.py

The main entry point for running experiments is `scripts/run.py`, which uses [Hydra](https://hydra.cc) for configuration management:

```bash
# Run with default configuration
python scripts/run.py

# Override specific parameters
python scripts/run.py run.n_steps=5000 run.seed=123

# Use different configuration file
python scripts/run.py --config-name=limited_memory
```

See files in `scripts/config` for examples. You can select multiple estimators to compute in a single run using the syntax [here](https://hydra.cc/docs/patterns/select_multiple_configs_from_config_group/)

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

This script also accepts hydra configuration. The output is a csv file containing columns A, B, where each row is the average reward from a single random seed, and from which the ATE can be computed as the mean difference: `ATE = mean(B) - mean(A)`

### Analyzing Estimator Performance (`plot_estimator_results.py`)

After running experiments, analyze estimator convergence and bias using:

```bash
python plot_estimator_results.py <path-to-run.py-output> --ate <float-or-path-to-compute-ate.py-output> --output-dir plots
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
- A summary of Bias, Variance, and RMSE (saved to `<output-path>/estimator_summary_stats.csv`)

## Available Environments

- **ABTestEnv**: Simple A/B testing environment, with no interference
- **LimitedMemoryEnv**: Simple interference environment where the mean outcome at t depends arbitrarily on $z_{t - {\rm window size}} ... z_t$.
- **XPRidesharePricingEnv**: Rideshare pricing experiment using or-gymnax



## Extending the Framework

### Adding New Estimators

1. Inherit from `Estimator` and `EstimatorState`
2. Implement `reset()`, `update()`, and `estimate()` methods
3. Add to configuration file

We also provide some abstractions for estimators that require a known interference graph; see e.g., `network.py` and`LimitedMemoryHTEstimator`for examples.

### Adding New Designs

1. Inherit from `Design` and optionally `DesignState`
2. Implement `assign_treatment()` method
3. Override `reset()` and `update()` if needed

### Adding New Environments

Easiest method is to use the `XPEnvironment` wrapper:

1. Inherit from `XPEnvironment`
2. Define policies A and B in the constructor
3. Configure in YAML file
Running experiment

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
- `RideshareClusterDesign`: SpatioTemporal clustering for the Rideshare Pricing environment, replicating the DN paper.
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
- `NaiveEstimator` (`naive`): Simple inverse probability weighting (IPW) estimator. Computes treatment effect as the difference between IPW-weighted outcomes for treated vs control observations (Note that several papers refer to this as the Difference-in-Means (DM) estimator).
- `DMEstimator` (`dm`): Difference-in-means estimator. Similar to Naive, but computes simple average difference between treated and control groups without IPW.
- `TPGEstimator` (`tpg`): Truncated policy gradient estimator with configurable horizon `k`. Uses current reward plus next `k` rewards with IPW weighting to reduce variance. [[Johari R, Peng T, Xing W., 2025]](https://arxiv.org/abs/2506.05308)
<!-- - `DQ` (`dq`): Dynkin-based value difference estimator using LSTD. Learns state-value functions to estimate treatment effects by comparing expected values across treatment groups. [[Farias, Vivek, et al., 2022]](https://arxiv.org/abs/2206.02371) -->
<!-- - `LSTDLambdaEstimator` (`lstd_lambda`): Off-policy evaluation using LSTD-λ. Maintains separate LSTD systems for treated and control groups with eligibility traces (λ parameter). [[Farias, Vivek, et al., 2022]](https://arxiv.org/abs/2206.02371) -->

*Network Interference Estimators (work with all designs):*
These estimators require a known interference graph, which requires implementing an `InterferenceNetwork` object with `is_adjacent` method to identify whether two observations interfere. See `estimators/network.py` for several examples.

- `LimitedMemoryDNEstimator` (`dn`): Difference-in-Networks estimator. Accounts for spatial and temporal spillovers by adjusting estimates based on nearby clusters (configurable via `max_spatial_distance` and `lookahead_steps`). [[Peng T, Ye N, Zheng A., 2025]](https://arxiv.org/abs/2503.02271)
- `LimitedMemoryHTEstimator` (`ht`): H-T estimator accounting for all interference; unbiased (assuming graph is correct) but very high variance.

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

