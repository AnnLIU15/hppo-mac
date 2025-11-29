# Notebook Training Flow

This note documents the step-by-step pipeline encoded in `main.ipynb`, highlighting how the notebook wires together configuration loading, environment construction, actor/critic assembly, and TorchRL’s PPO training loop.

## 1. Imports and Module Reload Hooks
- The first code cell imports utilities from the standard library (`dataclasses.replace`, `importlib.reload`, `Path`) and typing.
- Core dependencies include `torch`, `yaml`, and Loguru’s `logger`.
- For rapid iteration, commented-out `reload()` calls are provided for `algo.hppo`, `env.mac_simulator`, and `env.satellite_mac_env`—uncommenting them allows hot-swapping module edits without restarting the kernel.
- High-level facades are imported from `algo/__init__.py` (`HPPOConfig`, `build_hppo_modules`, `train_hppo`, `setup_logger`, `get_logger`) and from `env/__init__.py` (`SatelliteMACEnv`, `SatelliteMACEnvConfig`, `default_simulator_config`, `MACSimulatorConfig`). TorchRL wrappers (`ParallelEnv`, `SerialEnv`, `TransformedEnv`, `GymWrapper`, `Compose`, `DoubleToFloat`) complete the stack.

## 2. Configuration Loading and Logging Setup
- `CONFIG_PATH` points to `conf/default.yaml`. `_load_yaml_config()` reads the file with UTF-8 encoding and returns a dictionary; missing files trigger a warning and fall back to in-code defaults.
- The random seed is pulled from `raw_config['seed']` (defaulting to `20251106` in the notebook). `torch.manual_seed()` applies it globally.
- Logging configuration is fetched from the YAML `logging` section. `setup_logger()` installs Loguru sinks: stderr plus an optional file sink (`logs/notebook_training.log` by default).
- `get_logger()` yields a shared logger instance used throughout the notebook; both the module-local `logger` and the dedicated `my_logger` emit diagnostic messages to confirm configuration choices.

## 3. Environment Configuration Assembly
- `env_section` is read from YAML; `_override_sim_config()` starts from `default_simulator_config()` and selectively overrides:
  - `total_preambles`
  - `base_preamble_split` (validated to contain three entries)
- `SatelliteMACEnvConfig` is instantiated with merged parameters: slot count per step, decision horizon, history length, delta range, flattening flag, and the simulator config produced above.
- `num_envs` is pinned to 8 in the notebook to stress TorchRL’s vectorised collectors; `env_backend` defaults to the YAML value (or `parallel`). If `parallel_collection` is `True`, the backend is forced to `ParallelEnv`.
- `HPPOConfig` is created with bespoke hyperparameters intentionally differing from the YAML defaults: frames per batch 4096, mini-batch size 128, six PPO epochs, 100 iterations, larger clip epsilon (0.35), entropy coefficient 0.05, and tuned learning rates (`actor_lr=1e-4`, `critic_lr=3e-4`). All tensors run on the device declared in the YAML (CPU by default).
- A divisibility check warns when `frames_per_batch` is not a multiple of `num_envs`, mirroring `train_hppo.py`’s CLI behaviour.

## 4. TorchRL Environment Construction
- `_make_wrapped_env()` builds a `GymWrapper` around `SatelliteMACEnv`, seeding each replica with `seed + rank` to keep trajectories decorrelated.
- Depending on `num_envs` and `env_backend`:
  - Single-env runs use a lone `GymWrapper`.
  - Multi-env runs pick `ParallelEnv` (multi-process) or `SerialEnv` (single process with auto-reset).
- The transform stack is minimal: `Compose(DoubleToFloat())` promotes dtype consistency before the env is lifted into a `TransformedEnv`.
- Seeds are reapplied at the transformed level, and `rl_env.to(train_config.device)` moves specs and buffers to the target device.
- `build_hppo_modules(rl_env, feature_dim=128)` returns a `ProbabilisticActor` and `ValueOperator` tailored to the environment specs. The notebook logs observation shapes and backend choices, then displays the actor/critic modules for inspection.

### Actor/Critic internals (from `algo/hppo.py`)
- Observations are fed through a `_FeatureEncoder` MLP (Linear-256-ReLU-Linear-128-ReLU) to produce a shared latent `state_feature`.
- `_PolicyParamExtractor` produces logits for discrete heads and softplus-transformed concentration parameters for the beta-distributed ACB action.
- `ProbabilisticActor` wraps the tensor dictionary pipeline with a `CompositeDistribution` that maps `delta_cbra`/`delta_pbra` to `OneHotCategorical` branches and `q_ACB` to a custom `TanhNormal01` head. Log-prob aggregation is disabled so each branch contributes an independent PPO ratio.
- The critic uses a dedicated `_CriticNetwork` (encoder + value head) and returns `state_value` scalars.

## 5. Training Execution
- `_log_metrics()` formats PPO diagnostics into a single log line per iteration (iteration index, mean reward, and each loss component).
- `train_hppo(rl_env, actor, critic, train_config, logger_fn=_log_metrics)` drives the optimisation:
  1. `SyncDataCollector` streams experience batches of `frames_per_batch` frames, split into trajectories with reset-aware slicing.
  2. Generalised Advantage Estimation (`GAE`) annotates batches with `advantage` and `value_target` using the critic.
  3. `ClipPPOLoss` produces `loss_objective`, `loss_entropy`, and `loss_critic` terms; entropy and critic components are added into the total loss before backprop.
  4. Adam optimisers (one per network) update parameters across `rollout_epochs` × minibatches. Minibatch sampling is performed on-device using random permutations of the flattened batch.
  5. Collector weights are refreshed after each outer iteration to keep the policy and data stream in sync.
  6. Training halts once `max_iterations` batches have been processed, at which point the collector shuts down and scalar metrics from the final minibatch are returned.
- The notebook captures the returned `metrics` dict in a cell output, which contains the final recorded loss scalars.

## 6. Relationship to Scripted Entry Point
- The notebook mirrors `train_hppo.py`’s CLI workflow but keeps parameters inline for interactive tweaking. Both rely on the same helper functions (`_make_env`, `build_hppo_modules`, `train_hppo`) but differ in logging sinks and checkpoint handling (the notebook session does not persist weights by default).
- Users can migrate notebook-tuned hyperparameters back into `conf/default.yaml` or the CLI arguments once satisfied with convergence behaviour.

## 7. Practical Tips for Notebook Runs
- Expect significant CPU load when `num_envs=8` with `ParallelEnv`; adjust to `SerialEnv` or fewer environments when debugging.
- Keep the logging sink active—the PPO loop logs reward trends and loss magnitudes, making it easy to detect divergence or underfitting.
- The YAML loader gracefully handles missing keys; any extra keys are ignored, so additional customisations can be staged in `conf/default.yaml` without modifying notebook code.

This walkthrough should serve as a map for extending the notebook (e.g., adding evaluation rollouts, metric plots, or checkpoint export) while staying aligned with the shared TorchRL infrastructure in `algo/hppo.py`.
