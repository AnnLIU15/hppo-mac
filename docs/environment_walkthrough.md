# Satellite MAC Environment Walkthrough

This document captures the end-to-end control flow, configuration surface, and data products for the custom satellite medium-access-control (MAC) environment that powers the project. It summarises how the public gymnasium API hooks into the simulator core defined under `env/` and the utility routines under `utils/`.

## Module Topology
- `env/satellite_mac_env.py` exposes the `SatelliteMACEnv` Gym environment and its config dataclass `SatelliteMACEnvConfig`.
- `env/mac_simulator.py` provides the stochastic MAC simulator (`MACSimulator`) plus the configuration dataclasses that describe regional traffic, coverage drift, and backoff behaviour.
- `utils/traffic.py` contains the low-level stochastic models (Cox point process, batched Poisson arrivals, residence time sampling) that the simulator uses to synthesise demand and mobility signals.
- Tests in `tests/test_backoff_strategy.py` and `tests/test_region_mixture.py` exercise the simulator internals: they validate backoff queue scheduling and coverage-weight blending, ensuring the environment stays numerically stable.

## Configuration Layer
`SatelliteMACEnvConfig` controls how the gym environment wraps the simulator:
- `num_slots_per_step`: number of MAC slots simulated for each RL step.
- `decision_horizon`: per-episode cap; `step()` returns `truncated=True` once this many steps are executed.
- `history_len`: length of the rolling reward/collision backlog that the observation exposes.
- `simulator_config`: optional `MACSimulatorConfig` override; defaults to `default_simulator_config()` when omitted.
- `preamble_delta_range`: symmetric integer range used to discretise RA adjustments (e.g. range=3 yields 7 bins: -3…+3).
- `flatten_observation`: toggles between a single flattened observation vector (TorchRL friendly) and the structured Dict space.

`MACSimulatorConfig` (in `mac_simulator.py`) is a comprehensive dataclass that specifies:
- `regions`: tuple of `RegionTrafficProfile`, one per geographical class (e.g. urban/suburban/rural). Each profile carries baseline arrival densities, handover intensities, batch statistics, Cox process parameters, and residence time means.
- Coverage modelling parameters: optional `coverage_patches`, orbital drift radius/period, jitter, smoothing, and total footprint scaling. These drive `_update_region_mixture()` to produce smooth spatial weight transitions.
- Backoff behaviour through `BackoffStrategyConfig`, which bundles three `BackoffWindow`s (low/medium/high congestion), collision ratio thresholds for window selection, and an overall maximum backlog horizon.
- Reward shaping weights and the total RA preamble budget, which must match the `base_preamble_split` integrity constraints enforced in `__post_init__`.

## Environment Lifecycle
### Reset
1. `SatelliteMACEnv.reset()` reseeds both the gym RNG and the simulator RNG (`MACSimulator.reseed`).
2. The simulator prepares its internal state via `initialize_state()`: resets time slot counters, coverage phase, backlog queues, and draws an initial region mixture. It returns a full observation dict with typed numpy arrays.
3. The environment seeds its history buffer (`_init_history_buffer`) using the simulator-provided statistics and injects the initial `recent_stats` tensor before flattening (if requested).
4. `reset()` returns `(observation, info)` where `info['step']` starts at zero.

### Step
1. The raw TorchRL/TensorDict action is a `Dict` with keys `M_CBRA_delta`, `M_PBRA_delta`, and `q_ACB`.
2. `_parse_action()` translates the discrete heads back into integer deltas using `_decode_delta_component()` (supports index or one-hot encodings) and clamps the beta head to `[0, 1]`.
3. `MACSimulator.run_slots()` executes `num_slots_per_step` slots with the decoded adjustments:
   - `_apply_preamble_adjustment()` updates the running `(CBRA, PBRA, CFRA)` allocation while respecting the total preamble budget.
   - `_update_region_mixture()` evolves the geographic weighting via either a Markov transition matrix or the coverage patch model.
   - `_pop_backoff_queue()` releases retries that matured during the slot burst; `_schedule_backoff()` enqueues new retries based on collision ratios and the adaptive windows.
   - Per-region CBRA/PBRA/CFRA demands are generated with batched Poisson draws modulated by Cox-based density factors and residence time scaling. Collision ratios are sampled from beta distributions to produce successes vs. collisions and to update backlog counters.
   - Reward is computed from throughput and collision totals using the configured weights, and a dense `info` dictionary is assembled (region-level successes, collisions, backlog stats, coverage centre/phase, etc.).
4. Environment book-keeping: the history buffer is rolled forward with `_update_history_buffer()`, `recent_stats` is refreshed, and observations are flattened if necessary.
5. `step()` returns the observation, scalar reward, `terminated=False`, `truncated` flag based on the horizon, and the info dict (scalars are auto-cast from numpy arrays when possible).

## Observation Structure
When `flatten_observation=True` (the default used by TorchRL routes), the observation vector concatenates:
- Instantaneous counters: requests, collision ratios, success/collision totals, backlog sizes.
- Distributional snapshots: active terminal mix (`MAC_PROTOCOL_COUNT`), preamble usage (`PREAMBLE_SUBSET_COUNT`), preamble allocation ratios, region mixture, coverage centre (2-D), coverage phase.
- Simulator state dumps: raw `history_stats` ring buffer and the `recent_stats` history (flattened history_len × stats).

Setting `flatten_observation=False` exposes the same content as a `spaces.Dict`, preserving the typed sub-Boxes used in the simulator.

## Action Semantics
- `M_CBRA_delta` and `M_PBRA_delta`: categorical deltas mapping ↦ `[-range, …, range]`. One-hot vectors or scalar indices are accepted, facilitating TorchRL’s `OneHotCategorical` head.
- `q_ACB`: beta-distributed scalar where the actor outputs positive concentration parameters; the environment clips final values into `[0, 1]` before forwarding to the simulator.

## History and Diagnostics
- `_recent_stats` maintains a FIFO buffer of the last `history_len` rows for six key statistics (`requests_*`, `collision_ratio_*`, `pending_backoff_*`).
- The simulator stores a `history` ring buffer of past rewards inside its state, also surfaced through the observation.
- Extensive diagnostics are provided through the `info` dict in both the simulator and environment layers, enabling offline analysis (`main.py` and `scripts/plot_telemetry.py` rely on the same fields).

## Integration Points
- `train_hppo.py` and `main.ipynb` wrap `SatelliteMACEnv` in `GymWrapper` → `TransformedEnv` pipelines. They depend on the flattened observation and the Dict action space described above.
- `main.py` uses the raw environment class to run heuristic policies, generate telemetry archives, and sweep backoff parameters—demonstrating the environment’s deterministic seeding and the richness of the info payload.
- The TorchRL actor in `algo/hppo.py` introspects `env.action_spec` for the discrete bin counts and continuous bounds, so any modifications to the action space must remain compatible with TorchRL’s composite distributions.

## Supporting Utilities
- `utils/traffic.py` encapsulates high-variance stochastic primitives and guarantees reproducibility via explicit numpy `Generator` plumbing. Each routine accepts the RNG instance owned by `MACSimulator`, ensuring `reset(seed=…)` yields reproducible rollouts.
- Configuration defaults (`default_simulator_config`) provide three region profiles and conservative backoff thresholds so tests and notebooks can run without extra YAML overrides.

Use this document as the authoritative map when adjusting the simulator, expanding observation fields, or tightening the action encoding—it traces every dependency the training stack relies on.
