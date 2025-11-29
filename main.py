"""运行启发式策略与参数扫查以验证 MAC 环境功能与配置选项。"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from env.mac_simulator import (
    BackoffStrategyConfig,
    BackoffWindow,
    MACSimulatorConfig,
    default_simulator_config,
)
from env.satellite_mac_env import SatelliteMACEnv, SatelliteMACEnvConfig


@dataclass
class EpisodeStats:
    steps: int = 0
    reward: float = 0.0
    throughput: float = 0.0
    collisions: float = 0.0
    backlog_cbra: float = 0.0
    backlog_pbra: float = 0.0


@dataclass
class StepTrace:
    step: int
    reward: float
    throughput: float
    collisions: float
    backlog_cbra: float
    backlog_pbra: float
    cbra_delta: int
    pbra_delta: int
    acb: float
    collision_ratio_cbra: float
    collision_ratio_pbra: float
    action_valid: float


@dataclass(frozen=True)
class HeuristicConfig:
    collision_high: float = 0.18
    collision_medium: float = 0.1
    backlog_high: float = 200.0
    backlog_medium: float = 60.0
    load_high: float = 100.0
    load_medium: float = 50.0
    acb_levels: Tuple[float, float, float] = (0.35, 0.5, 0.65)
    low_collision_acb: float = 0.75
    tremble_prob: float = 0.05


@dataclass
class HeuristicDecision:
    action: Dict[str, np.ndarray]
    cbra_delta: int
    pbra_delta: int
    acb_value: float


def encode_delta(delta: int, delta_range: int) -> int:
    clipped = int(np.clip(delta, -delta_range, delta_range))
    return clipped + delta_range



def heuristic_policy(
    observation: Dict[str, np.ndarray],
    delta_range: int,
    rng: np.random.Generator,
    config: HeuristicConfig,
) -> HeuristicDecision:
    cbra_backlog = float(observation["pending_backoff_cbra"][0])
    pbra_backlog = float(observation["pending_backoff_pbra"][0])
    cbra_collision = float(observation["collision_ratio_cbra"][0])
    pbra_collision = float(observation["collision_ratio_pbra"][0])
    cbra_requests = float(observation["requests_cbra"][0])
    pbra_requests = float(observation["requests_pbra"][0])

    cbra_delta = 0
    pbra_delta = 0

    if cbra_collision >= config.collision_high or cbra_backlog >= config.backlog_high:
        cbra_delta = 2
    elif cbra_collision >= config.collision_medium or cbra_backlog >= config.backlog_medium:
        cbra_delta = 1
    elif cbra_collision < config.collision_medium * 0.4 and cbra_backlog < config.backlog_medium * 0.2:
        if observation["preamble_allocation"][0] > 0.4:
            cbra_delta = -1

    if pbra_collision >= config.collision_high or pbra_backlog >= config.backlog_high:
        pbra_delta = 2
    elif pbra_collision >= config.collision_medium or pbra_backlog >= config.backlog_medium:
        pbra_delta = 1
    elif pbra_collision < config.collision_medium * 0.4 and pbra_backlog < config.backlog_medium * 0.2:
        if observation["preamble_allocation"][1] > 0.4:
            pbra_delta = -1

    load_indicator = cbra_requests + pbra_requests + cbra_backlog + pbra_backlog
    if load_indicator >= config.load_high or max(cbra_collision, pbra_collision) >= config.collision_high:
        acb = config.acb_levels[0]
    elif load_indicator >= config.load_medium or max(cbra_collision, pbra_collision) >= config.collision_medium:
        acb = config.acb_levels[1]
    elif max(cbra_collision, pbra_collision) < config.collision_medium * 0.4:
        acb = config.low_collision_acb
    else:
        acb = config.acb_levels[2]

    if rng.random() < config.tremble_prob:
        cbra_delta += rng.integers(-1, 2)
        pbra_delta += rng.integers(-1, 2)

    action_dict = {
        "delta_cbra": np.array(encode_delta(cbra_delta, delta_range), dtype=np.int64),
        "delta_pbra": np.array(encode_delta(pbra_delta, delta_range), dtype=np.int64),
        "q_ACB": np.array([np.clip(acb, 0.0, 1.0)], dtype=np.float32),
    }
    return HeuristicDecision(action_dict, cbra_delta, pbra_delta, float(acb))


def validate_info(info: Dict[str, object]) -> None:
    mixture = info.get("region_mixture")
    if isinstance(mixture, np.ndarray):
        total = float(mixture.sum())
        if not np.isfinite(total) or abs(total - 1.0) > 1e-3:
            raise AssertionError("region_mixture 未正确归一化")
    for key in ("pending_backoff_cbra", "pending_backoff_pbra"):
        value = info.get(key)
        if isinstance(value, np.ndarray):
            scalar = float(value.item())
            if scalar < -1e-5:
                raise AssertionError(f"{key} 出现负数值")


def run_episode(
    env: SatelliteMACEnv,
    delta_range: int,
    rng: np.random.Generator,
    seed: int,
    fix: bool = False,
    fix_acb: float = 0.5,
) -> Tuple[EpisodeStats, Dict[str, object], List[StepTrace]]:
    observation, info = env.reset(seed=seed)
    validate_info(info)
    stats = EpisodeStats()
    last_info = info
    traces: List[StepTrace] = []
    heuristic_cfg = HeuristicConfig()
    done = False
    env.simulator.configure_access_state(cbra=24,cfra=16,pbra=24)
    while not done:
        if not fix:
            decision = heuristic_policy(observation, delta_range, rng, heuristic_cfg)
        else:
            action_dict = {
                "delta_cbra": np.array([0], dtype=np.int64),
                "delta_pbra": np.array([0], dtype=np.int64),
                "q_ACB": np.array([fix_acb], dtype=np.float32),
            }
            decision = HeuristicDecision(action_dict, 0, 0, fix_acb)

        observation, reward, terminated, truncated, info = env.step(decision.action, need_parse_action=False)
        validate_info(info)

        stats.steps += 1
        stats.reward += reward
        stats.throughput += float(info.get("throughput", 0.0))
        stats.collisions += float(info.get("collision_total", 0.0))
        stats.backlog_cbra += float(info.get("pending_backoff_cbra", 0.0))
        stats.backlog_pbra += float(info.get("pending_backoff_pbra", 0.0))

        traces.append(
            StepTrace(
                step=stats.steps,
                reward=float(reward),
                throughput=float(info.get("throughput", 0.0)),
                collisions=float(info.get("collision_total", 0.0)),
                backlog_cbra=float(info.get("pending_backoff_cbra", 0.0)),
                backlog_pbra=float(info.get("pending_backoff_pbra", 0.0)),
                cbra_delta=decision.cbra_delta,
                pbra_delta=decision.pbra_delta,
                acb=decision.acb_value,
                collision_ratio_cbra=float(info.get("collision_ratio_cbra", 0.0)),
                collision_ratio_pbra=float(info.get("collision_ratio_pbra", 0.0)),
                action_valid=float(info.get("action_valid", 1.0)),
            )
        )

        done = terminated or truncated
        last_info = info

    return stats, last_info, traces


def save_telemetry(traces: Sequence[StepTrace], out_path: Path) -> None:
    arr_map = {key: [] for key in StepTrace.__annotations__.keys()}
    for trace in traces:
        for field, value in trace.__dict__.items():
            arr_map[field].append(value)
    np_arrays = {}
    for key, values in arr_map.items():
        dtype = np.float32 if key != "step" else np.int32
        np_arrays[key] = np.asarray(values, dtype=dtype)
    np.savez_compressed(out_path, **np_arrays)


def analyze_telemetry(paths: Sequence[Path], out_dir: Path) -> None:
    if not paths:
        return
    summary_rows: List[Dict[str, float]] = []
    for path in paths:
        data = np.load(path)
        steps = data["step"].astype(int)
        backlog_cbra = data["backlog_cbra"]
        backlog_pbra = data["backlog_pbra"]
        collisions = data["collisions"]
        throughput = data["throughput"]
        reward = data["reward"]
        if steps.size == 0:
            continue
        idx_max_backlog = int(np.argmax(backlog_cbra))
        row = {
            "file": path.name,
            "steps": float(steps.size),
            "reward_sum": float(np.sum(reward)),
            "reward_mean": float(np.mean(reward)),
            "throughput_mean": float(np.mean(throughput)),
            "collisions_mean": float(np.mean(collisions)),
            "backlog_cbra_mean": float(np.mean(backlog_cbra)),
            "backlog_pbra_mean": float(np.mean(backlog_pbra)),
            "backlog_cbra_max": float(np.max(backlog_cbra)),
            "backlog_cbra_max_step": float(steps[idx_max_backlog]),
            "collision_ratio_cbra_mean": float(np.mean(data["collision_ratio_cbra"])),
            "collision_ratio_pbra_mean": float(np.mean(data["collision_ratio_pbra"])),
        }
        summary_rows.append(row)

    if not summary_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "telemetry_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    sorted_rows = sorted(summary_rows, key=lambda item: item["backlog_cbra_max"], reverse=True)
    worst = sorted_rows[0]
    best = sorted(summary_rows, key=lambda item: item["reward_mean"], reverse=True)[0]
    print("\nTelemetry analysis summary:")
    print(f"Worst backlog episode: {worst['file']} max={worst['backlog_cbra_max']:.1f} at step {worst['backlog_cbra_max_step']}")
    print(f"Best average reward episode: {best['file']} mean reward={best['reward_mean']:.2f}")
    print(f"Telemetry summary saved to: {csv_path}")


def build_simulator_config(
    medium_threshold: float,
    high_threshold: float,
    low_window: Tuple[int, int],
    medium_window: Tuple[int, int],
    high_window: Tuple[int, int],
    max_backoff: int,
) -> MACSimulatorConfig:
    base = default_simulator_config()
    strategy = BackoffStrategyConfig(
        low=BackoffWindow(*low_window),
        medium=BackoffWindow(*medium_window),
        high=BackoffWindow(*high_window),
        collision_threshold_medium=medium_threshold,
        collision_threshold_high=high_threshold,
        max_backlog_steps=max_backoff,
    )
    return replace(base, backoff_strategy=strategy)


def build_env_config(
    delta_range: int,
    decision_horizon: int,
    simulator_config: MACSimulatorConfig,
    flatten_observation: bool = False,
) -> SatelliteMACEnvConfig:
    return SatelliteMACEnvConfig(
        num_slots_per_step=800,
        decision_horizon=decision_horizon,
        history_len=8,
        preamble_delta_range=delta_range,
        flatten_observation=flatten_observation,
        simulator_config=simulator_config,
    )


def parameter_sweep(
    seeds: Iterable[int],
    delta_ranges: Sequence[int],
    medium_thresholds: Sequence[float],
    high_thresholds: Sequence[float],
    windows: Sequence[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int]],
    out_dir: Path,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for delta_range in delta_ranges:
        for medium_thr in medium_thresholds:
            for high_thr in high_thresholds:
                for low_w, med_w, high_w, max_backoff in windows:
                    sim_config = build_simulator_config(
                        medium_threshold=medium_thr,
                        high_threshold=high_thr,
                        low_window=low_w,
                        medium_window=med_w,
                        high_window=high_w,
                        max_backoff=max_backoff,
                    )
                    env_config = build_env_config(
                        delta_range=delta_range,
                        decision_horizon=128,
                        simulator_config=sim_config,
                    )
                    env = SatelliteMACEnv(env_config)
                    action_space = env.action_space
                    branch_bins = int(action_space["delta_cbra"].n)
                    effective_delta = (branch_bins - 1) // 2
                    aggregate = []
                    for seed in seeds:
                        stats, _, _ = run_episode(env, effective_delta, rng, seed)
                        aggregate.append(stats)
                    env.close()

                    total_reward = np.mean([s.reward for s in aggregate])
                    avg_throughput = np.mean([s.throughput / max(s.steps, 1) for s in aggregate])
                    avg_collisions = np.mean([s.collisions / max(s.steps, 1) for s in aggregate])
                    avg_backlog = np.mean(
                        [
                            (s.backlog_cbra + s.backlog_pbra) / max(s.steps, 1)
                            for s in aggregate
                        ]
                    )

                    record = {
                        "delta_range": float(delta_range),
                        "medium_thr": float(medium_thr),
                        "high_thr": float(high_thr),
                        "low_window": float(low_w[0]),
                        "low_window_max": float(low_w[1]),
                        "medium_window": float(med_w[0]),
                        "medium_window_max": float(med_w[1]),
                        "high_window": float(high_w[0]),
                        "high_window_max": float(high_w[1]),
                        "max_backoff": float(max_backoff),
                        "reward": float(total_reward),
                        "avg_throughput": float(avg_throughput),
                        "avg_collisions": float(avg_collisions),
                        "avg_backlog": float(avg_backlog),
                    }
                    results.append(record)

    with (out_dir / "sweep_summary.csv").open("w", newline="") as csvfile:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    return results


def main() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    sim_config = default_simulator_config()
    # build_simulator_config(
    #     medium_threshold=0.08,
    #     high_threshold=0.15,
    #     low_window=(0, 6),
    #     medium_window=(3, 14),
    #     high_window=(8, 28),
    #     max_backoff=48,
    # )
    env_config = build_env_config(
        delta_range=3,
        decision_horizon=256,
        simulator_config=sim_config,
    )
    env = SatelliteMACEnv(env_config)
    rng = np.random.default_rng(20251106)

    action_space = env.action_space
    branch_bins = int(action_space["delta_cbra"].n)
    delta_range = (branch_bins - 1) // 2

    episode_rewards = []
    episode_infos = []
    telemetry_paths: List[Path] = []

    for episode_idx in range(5):
        seed = int(rng.integers(0, 1_000_000))
        stats, last_info, traces = run_episode(env, delta_range, rng, seed)
        avg_throughput = stats.throughput / max(stats.steps, 1)
        avg_collisions = stats.collisions / max(stats.steps, 1)
        avg_backlog_cbra = stats.backlog_cbra / max(stats.steps, 1)
        avg_backlog_pbra = stats.backlog_pbra / max(stats.steps, 1)

        episode_rewards.append(stats.reward)
        episode_infos.append(last_info)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        telemetry_path = logs_dir / f"telemetry_ep{episode_idx + 1}_{timestamp}.npz"
        save_telemetry(traces, telemetry_path)
        telemetry_paths.append(telemetry_path)

        print(
            f"Episode {episode_idx + 1}: steps={stats.steps}, reward={stats.reward:.2f}, "
            f"avg_throughput={avg_throughput:.2f}, avg_collisions={avg_collisions:.2f}, "
            f"avg_backlog_cbra={avg_backlog_cbra:.2f}, avg_backlog_pbra={avg_backlog_pbra:.2f}"
        )

    print("\nSummary:")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Reward std: {np.std(episode_rewards):.2f}")

    sample_info = episode_infos[-1]
    if isinstance(sample_info.get("region_mixture"), np.ndarray):
        print("Final region mixture:", sample_info["region_mixture"].round(3))
    if isinstance(sample_info.get("preamble_allocation"), np.ndarray):
        print("Final preamble allocation:", sample_info["preamble_allocation"].round(3))

    env.close()

    analyze_telemetry(telemetry_paths, logs_dir)

    sweep_seed_list = [int(rng.integers(0, 1_000_000)) for _ in range(3)]
    sweep_windows = [
        ((0, 4), (2, 10), (6, 20), 32),
        ((0, 6), (3, 14), (8, 28), 40),
        ((0, 8), (4, 18), (10, 32), 48),
    ]
    sweep_dir = logs_dir / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    sweep_dir.mkdir(exist_ok=True)
    results = parameter_sweep(
        seeds=sweep_seed_list,
        delta_ranges=[2, 3, 4],
        medium_thresholds=[0.06, 0.08, 0.1],
        high_thresholds=[0.12, 0.15, 0.18],
        windows=sweep_windows,
        out_dir=sweep_dir,
        rng=rng,
    )

    if results:
        best = min(results, key=lambda item: item["avg_collisions"] + 0.01 * item["avg_backlog"])
        print("\nBest sweep configuration (by collision/backlog trade-off):")
        print(best)
        print(f"Sweep summary saved to: {sweep_dir / 'sweep_summary.csv'}")
        print("Episode telemetry files:")
        for path in telemetry_paths:
            print(f" - {path}")

        fine_delta = int(best["delta_range"])
        fine_medium = float(best["medium_thr"])
        fine_high = float(best["high_thr"])
        fine_max_backoff = int(best["max_backoff"])

        def clamp_probability(values: Iterable[float]) -> List[float]:
            return sorted(set([float(np.clip(v, 0.01, 0.95)) for v in values]))

        fine_delta_ranges = sorted({max(1, fine_delta - 1), fine_delta, fine_delta + 1})
        fine_medium_thresholds = clamp_probability([fine_medium - 0.01, fine_medium, fine_medium + 0.01])
        fine_high_thresholds = clamp_probability([fine_high - 0.015, fine_high, fine_high + 0.015])

        low_min = int(best["low_window"])
        low_max = int(best["low_window_max"])
        med_min = int(best["medium_window"])
        med_max = int(best["medium_window_max"])
        high_min = int(best["high_window"])
        high_max = int(best["high_window_max"])

        def window_variants(base_min: int, base_max: int, shifts: Sequence[int]) -> List[Tuple[int, int]]:
            variants = []
            for shift in shifts:
                new_min = max(0, base_min + shift)
                new_max = max(new_min, base_max + shift)
                variants.append((new_min, new_max))
            return variants

        window_shifts = [-2, 0, 2]
        fine_windows: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int]] = []
        for low_w in window_variants(low_min, low_max, window_shifts):
            for med_w in window_variants(med_min, med_max, window_shifts):
                for high_w in window_variants(high_min, high_max, window_shifts):
                    for max_backoff in {fine_max_backoff - 4, fine_max_backoff, fine_max_backoff + 4}:
                        if max_backoff < high_w[1]:
                            continue
                        fine_windows.append((low_w, med_w, high_w, int(max_backoff)))

        fine_dir = sweep_dir / "fine"
        fine_dir.mkdir(exist_ok=True)
        fine_results = parameter_sweep(
            seeds=sweep_seed_list,
            delta_ranges=fine_delta_ranges,
            medium_thresholds=fine_medium_thresholds,
            high_thresholds=fine_high_thresholds,
            windows=fine_windows,
            out_dir=fine_dir,
            rng=rng,
        )

        if fine_results:
            fine_best = min(fine_results, key=lambda item: item["avg_collisions"] + 0.01 * item["avg_backlog"])
            print("\nFine sweep best configuration:")
            print(fine_best)
            print(f"Fine sweep summary saved to: {fine_dir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
