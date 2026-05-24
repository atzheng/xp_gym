# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
```bash
# Install dependencies using Poetry
poetry install

# Or install manually with pip
pip install jax flax gymnax hydra-core chex jaxtyping pyarrow sacred tqdm haversine
pip install git+https://github.com/atzheng/or-gymnax.git
```

### Running Experiments
```bash
# Run experiments with default configuration
python scripts/run.py

# Override specific parameters
python scripts/run.py run.n_steps=5000 run.seed=123

# Use different configuration files
python scripts/run.py --config-name=my_config

# Select multiple estimators (see Hydra multirun docs)
python scripts/run.py estimators=naive,dm,tpg
```

### Analysis Scripts
```bash
# Compute true Average Treatment Effect (ATE)
python compute-ate.py

# Generate analysis plots after running experiments
python plot_estimator_results.py
```

### Configuration
- Main config: `scripts/config/config.yaml`
- Environment configs: `scripts/config/env/`
- Design configs: `scripts/config/design/` 
- Estimator configs: `scripts/config/estimators/`

## Architecture Overview

XP Gym is a modular JAX-based framework for A/B testing and causal inference evaluation in RL environments. The architecture consists of four core abstract classes that define the experimental structure:

### Core Components

**Environment (`xp_gym.environments.environment`)**
- `XPEnvironment`: Wraps any Gymnax environment for A/B testing between two policies
- Reduces action space to binary choice (policy A vs B)
- Key files: `xp_gym/environments/environment.py`, `abtest.py`, `rideshare.py`

**Experimental Design (`xp_gym.designs.design`)**
- `Design`: Abstract base for treatment assignment strategies
- Built-in designs: `UnitRandomizedDesign`, `SwitchbackDesign`, `SpatioClusterDesign`, `ClusterRandomizedDesign`
- Handles treatment assignment and provides metadata to estimators

**Estimators (`xp_gym.estimators/`)**
- `Estimator`: Abstract base for causal inference methods
- Universal estimators: `NaiveEstimator` (IPW), `DMEstimator`, `TPGEstimator`
- Design-specific: `DNEstimator`, `DynkinEstimator`, `LSTDLambdaEstimator`, `SWTPGEstimator`
- Each estimator maintains state and produces treatment effect estimates

**Simulation Engine (`xp_gym.simulator`)**
- Orchestrates interaction between environment, design, and estimators
- Key functions: `step()`, `step_n_and_estimate()`, `simulate()`
- Handles parallel environment execution

### Data Flow
1. `Observation` class standardizes data flow: `obs`, `action`, `reward`, `design_info`
2. Design assigns treatment and provides metadata 
3. Environment executes treatment choice and returns rewards
4. Estimators update internal state and compute treatment effect estimates
5. Results saved to CSV with estimator-specific columns

### Key Dependencies
- **JAX/Flax**: High-performance computing and neural networks
- **Gymnax**: RL environment interface  
- **Hydra**: Configuration management and experiment orchestration
- **or-gymnax**: Operations research environments (rideshare, etc.)
- **Sacred**: Experiment tracking for ATE computation
- **Chex**: JAX utilities and testing

### Spatial Data
- `data/manhattan-nodes.parquet`: Network nodes with GPS coordinates
- `data/taxi-zones.parquet`: Zone definitions for spatial clustering
- Used by spatial designs and estimators for geographic clustering

### Extension Points
- Add new estimators by inheriting from `Estimator` 
- Add new designs by inheriting from `Design`
- Add new environments by wrapping existing Gymnax environments in `XPEnvironment`
- Configure via YAML files using Hydra's `_target_` syntax for instantiation

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:7510c1e2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
