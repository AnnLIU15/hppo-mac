"""运行启发式策略与参数扫查以验证 MAC 环境功能与配置选项。"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union, Dict, Iterable, List, Sequence, Tuple

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
    # Q-ALOHA 最优碰撞率和空闲率
    optimal_collision_rate: float = 0.264  # 约 26.4%
    optimal_idle_rate: float = 0.368       # 约 36.8%

    # ACB 调整步长 (ACB 等价于 Q-ALOHA 的 q 参数，表示发送概率)
    acb_step_up: float = 0.3      # 碰撞高时减小 ACB 的步长
    acb_step_down: float = 0.1    # 空闲高时增大 ACB 的步长

    # 碰撞率和空闲率的容忍范围
    collision_tolerance: float = 0.05  # ±5%
    idle_tolerance: float = 0.05       # ±5%

    # 前导码分配的最小保障
    min_preamble_per_type: int = 1

    # ACB 的边界
    acb_min: float = 0.01
    acb_max: float = 0.95

    # 当两者成功率都接近100%时的阈值
    success_rate_threshold: float = 0.95

    tremble_prob: float = 0.0


@dataclass
class HeuristicDecision:
    action: Dict[str, np.ndarray]
    cbra_delta: int
    pbra_delta: int
    acb_value: float


@dataclass
class HeuristicState:
    """维护启发式策略的内部状态"""
    last_acb: float = 0.5
    last_cbra_preambles: int = 0
    last_pbra_preambles: int = 0


def encode_delta(delta: int, delta_range: int) -> int:
    clipped = int(np.clip(delta, -delta_range, delta_range))
    return clipped + delta_range


def heuristic_policy(
    observation: Dict[str, np.ndarray],
    per_slot_cfra_arrival: int,
    preamble_allocated: Union[Tuple[int], List[int]],
    delta_range: int,
    rng: np.random.Generator,
    config: HeuristicConfig,
    state: HeuristicState,
    total_preambles: int = 64,
) -> HeuristicDecision:
    """
    新的启发式策略：
    1. CFRA 优先保障：确保 CFRA >= per_slot_cfra_arrival
    2. 剩余前导码按比例公平分配给 CBRA 和 PBRA（基于碰撞率）
    3. 当两者成功率都接近100%时，保持上次分配，只调整 ACB
    4. ACB 调整采用 Q-ALOHA 风格的反馈控制
    """
    # 提取观测值
    cbra_collision = float(observation["collision_ratio_cbra"][0])
    pbra_collision = float(observation["collision_ratio_pbra"][0])

    # 计算成功率（假设：成功率 = 1 - 碰撞率 - 空闲率，这里简化为 1 - 碰撞率作为近似）
    cbra_success_approx = 1.0 - cbra_collision
    pbra_success_approx = 1.0 - pbra_collision

    # 当前前导码分配
    current_cbra = int(preamble_allocated[0])
    current_pbra = int(preamble_allocated[1])
    # current_cfra 暂时未使用，但保留以备将来扩展
    # current_cfra = int(preamble_allocated[2]) if len(preamble_allocated) > 2 else 0

    # === 第一步：确保 CFRA 前导码数量 ===
    required_cfra = max(int(np.ceil(per_slot_cfra_arrival)), config.min_preamble_per_type)
    target_cfra = required_cfra

    # 剩余可分配的前导码
    remaining_preambles = total_preambles - target_cfra
    remaining_preambles = max(remaining_preambles, 2 * config.min_preamble_per_type)

    # === 第二步：判断是否需要重新分配 CBRA 和 PBRA ===
    both_high_success = (
        cbra_success_approx >= config.success_rate_threshold and
        pbra_success_approx >= config.success_rate_threshold
    )

    if both_high_success:
        # 两者成功率都接近100%，保持上次的前导码分配
        target_cbra = state.last_cbra_preambles
        target_pbra = state.last_pbra_preambles

        # 确保总和不超过剩余前导码
        if target_cbra + target_pbra > remaining_preambles:
            ratio = remaining_preambles / max(target_cbra + target_pbra, 1)
            target_cbra = int(target_cbra * ratio)
            target_pbra = remaining_preambles - target_cbra

            # 重新应用最小前导码约束
            target_cbra = max(target_cbra, config.min_preamble_per_type)
            target_pbra = max(target_pbra, config.min_preamble_per_type)

            # 如果仍超出，优先保证最小值，然后按比例分配剩余
            if target_cbra + target_pbra > remaining_preambles:
                min_total = 2 * config.min_preamble_per_type
                if remaining_preambles >= min_total:
                    target_cbra = config.min_preamble_per_type
                    target_pbra = remaining_preambles - target_cbra
                else:
                    # 极端情况：连最小值都无法满足
                    target_cbra = remaining_preambles // 2
                    target_pbra = remaining_preambles - target_cbra
    else:
        # 基于碰撞率进行比例公平分配
        # 碰撞率越高，需要的前导码越多
        # 使用碰撞率的倒数作为权重（碰撞率低的系统效率高，可以用更少的前导码）

        # 为了避免除零，添加一个小的 epsilon
        epsilon = 0.01
        cbra_weight = cbra_collision + epsilon
        pbra_weight = pbra_collision + epsilon

        total_weight = cbra_weight + pbra_weight

        # 按比例分配
        cbra_ratio = cbra_weight / total_weight

        target_cbra = int(cbra_ratio * remaining_preambles)
        target_pbra = remaining_preambles - target_cbra

        # 确保最小前导码数量
        target_cbra = max(target_cbra, config.min_preamble_per_type)
        target_pbra = max(target_pbra, config.min_preamble_per_type)

        # 如果超出剩余前导码，按比例缩减
        if target_cbra + target_pbra > remaining_preambles:
            total_target = target_cbra + target_pbra
            target_cbra = int((target_cbra / total_target) * remaining_preambles)
            target_pbra = remaining_preambles - target_cbra

            # 重新应用最小前导码约束
            target_cbra = max(target_cbra, config.min_preamble_per_type)
            target_pbra = max(target_pbra, config.min_preamble_per_type)

            # 如果仍超出，优先保证最小值，然后按比例分配剩余
            if target_cbra + target_pbra > remaining_preambles:
                min_total = 2 * config.min_preamble_per_type
                if remaining_preambles >= min_total:
                    target_cbra = config.min_preamble_per_type
                    target_pbra = remaining_preambles - target_cbra
                else:
                    # 极端情况：连最小值都无法满足
                    target_cbra = remaining_preambles // 2
                    target_pbra = remaining_preambles - target_cbra
    # print(f'target_cbra {target_cbra} target_pbra {target_pbra} target_cfra {target_cfra}')
    # 计算 delta
    cbra_delta = target_cbra - current_cbra
    pbra_delta = target_pbra - current_pbra
    # print(f'cbra_delta {cbra_delta} pbra_delta {pbra_delta}')
    # === 第三步：调整 ACB (Access Control Barring) ===
    # 注意：在本系统中，ACB 表示"允许接入的比例"，等价于 Q-ALOHA 的 q 参数
    #       ACB = q (发送概率)
    #       ACB = 1.0 → 所有终端都尝试接入（100% 发送概率）
    #       ACB = 0.0 → 没有终端尝试接入（0% 发送概率）
    #
    # Q-ALOHA 调整准则：
    # - 碰撞率高 → 竞争激烈 → 减小 ACB（降低发送概率，减少接入数量）
    # - 碰撞率低（空闲率高）→ 资源浪费 → 增大 ACB（提高发送概率，增加接入数量）
    # - 目标：收敛到最优碰撞率约 26.4%

    # 计算综合碰撞率（加权平均）
    total_active_preambles = max(current_cbra + current_pbra, 1)
    weighted_collision = (
        (current_cbra / total_active_preambles) * cbra_collision +
        (current_pbra / total_active_preambles) * pbra_collision
    )

    # Q-ALOHA 调整逻辑
    acb = state.last_acb

    if weighted_collision > config.optimal_collision_rate + config.collision_tolerance:
        # 碰撞率过高，降低 ACB（限制更多接入）
        acb = acb - config.acb_step_up
    elif weighted_collision < config.optimal_collision_rate - config.collision_tolerance:
        # 碰撞率过低（可能空闲率高），提高 ACB（允许更多接入）
        acb = acb + config.acb_step_down
    # else: 在最优范围内，保持不变

    # 边界限制
    acb = float(np.clip(acb, config.acb_min, config.acb_max))

    # === 第四步：添加探索噪声（可选）===
    if rng.random() < config.tremble_prob:
        cbra_delta += rng.integers(-1, 2)
        pbra_delta += rng.integers(-1, 2)

    # 更新状态
    state.last_acb = acb
    state.last_cbra_preambles = target_cbra
    state.last_pbra_preambles = target_pbra

    # 构造动作
    action_dict = {
        "delta_cbra": np.array(cbra_delta, dtype=np.int64),
        "delta_pbra": np.array(pbra_delta, dtype=np.int64),
        "q_ACB": np.array([acb], dtype=np.float32),
    }

    return HeuristicDecision(action_dict, cbra_delta, pbra_delta, acb)


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
    preamble_init: tuple = (40, 14, 10),
    total_preambles: int = 64,
) -> Tuple[EpisodeStats, Dict[str, object], List[StepTrace]]:
    observation, info = env.reset(seed=seed)
    validate_info(info)
    stats = EpisodeStats()
    last_info = info
    traces: List[StepTrace] = []
    heuristic_cfg = HeuristicConfig()
    heuristic_state = HeuristicState(
        last_acb=fix_acb,
        last_cbra_preambles=preamble_init[0],
        last_pbra_preambles=preamble_init[1],
    )
    done = False
    env.simulator.configure_access_state(cbra=preamble_init[0],
                                         pbra=preamble_init[1],
                                         cfra=preamble_init[2],)
    num_slots_per_step = env.unwrapped.config.num_slots_per_step

    last_cfra_total_arrivals_cfra = 0
    while not done:
        cur_cfra_total_arrivals_cfra = env.unwrapped.simulator._total_arrivals_cfra
        last_arrival_cfra = cur_cfra_total_arrivals_cfra - last_cfra_total_arrivals_cfra
        last_arrival_cfra_per_slot_avg = np.ceil(last_arrival_cfra / num_slots_per_step)
        last_cfra_total_arrivals_cfra = cur_cfra_total_arrivals_cfra
        cur_preamble_allocation = env.unwrapped.simulator._preamble_allocation
        # print('cur_preamble_allocation',cur_preamble_allocation)
        if not fix:
            decision = heuristic_policy(
                observation,
                last_arrival_cfra_per_slot_avg,
                cur_preamble_allocation,
                delta_range,
                rng,
                heuristic_cfg,
                heuristic_state,
                total_preambles,
            )

        else:
            action_dict = {
                "delta_cbra": np.array([0], dtype=np.int64),
                "delta_pbra": np.array([0], dtype=np.int64),
                "q_ACB": np.array([fix_acb], dtype=np.float32),
            }
            decision = HeuristicDecision(action_dict, 0, 0, fix_acb)
        # print('decision', decision)
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
