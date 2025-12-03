"""Simplified regional MAC simulator tailored to the linear coverage scenario."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Public constants exposed to the environment and policy stacks -----------------

MAC_PROTOCOL_COUNT = 3  # CBRA, PBRA, CFRA
PREAMBLE_SUBSET_COUNT = 8
DEFAULT_TOTAL_PREAMBLES = 64

_STAT_KEYS = (
    "requests_cbra",
    "requests_pbra",
    "collision_ratio_cbra",
    "collision_ratio_pbra",
    "pending_backoff_cbra",
    "pending_backoff_pbra",
)
DEFAULT_HISTORY_SIZE = 10
HISTORY_DIM = DEFAULT_HISTORY_SIZE * len(_STAT_KEYS)

# --- Dataclasses describing the simulator configuration ---------------------------


@dataclass(frozen=True)
class BackoffWindow:
    """Lower/upper bounds for the backoff delay (inclusive)."""

    min_steps: int
    max_steps: int

    def __post_init__(self) -> None:
        if self.min_steps < 0 or self.max_steps < self.min_steps:
            raise ValueError("Invalid backoff window")


@dataclass(frozen=True)
class BackoffStrategyConfig:
    """Grouping of adaptive windows driven by collision ratios."""

    low: BackoffWindow
    medium: BackoffWindow
    high: BackoffWindow
    collision_threshold_medium: float
    collision_threshold_high: float
    max_backlog_steps: int

    def __post_init__(self) -> None:
        if self.max_backlog_steps <= 0:
            raise ValueError("max_backlog_steps must be positive")


@dataclass(frozen=True)
class RegionTrafficProfile:
    """Per-category density template across the linear corridor."""

    name: str
    cbra_density: float
    pbra_density: float
    cfra_density: float

    def __post_init__(self) -> None:
        if min(self.cbra_density, self.pbra_density, self.cfra_density) < 0.0:
            raise ValueError("Region densities must be non-negative")


@dataclass(frozen=True)
class RegionSegment:
    """Contiguous linear segment mapped to a region category."""

    region_name: str
    length: float

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("Segment length must be positive")


def _default_backoff_strategy() -> BackoffStrategyConfig:
    return BackoffStrategyConfig(
        low=BackoffWindow(0, 4),
        medium=BackoffWindow(2, 10),
        high=BackoffWindow(6, 18),
        collision_threshold_medium=0.08,
        collision_threshold_high=0.16,
        max_backlog_steps=48,
    )


def _default_reward_weights() -> Dict[str, float]:
    return {"throughput": 1.0, "collision": -0.3}


@dataclass(frozen=True)
class MACSimulatorConfig:
    """Configuration object describing the linear MAC simulator."""

    regions: Tuple[RegionTrafficProfile, ...]
    segments: Tuple[RegionSegment, ...]
    total_preambles: int = DEFAULT_TOTAL_PREAMBLES
    base_preamble_split: Tuple[int, int, int] = (27, 27, 10)
    history_size: int = DEFAULT_HISTORY_SIZE
    coverage_window: float = 3.0
    motion_per_step: float = 0.1
    slots_per_motion: int = 10
    noise_scale: float = 0.05
    reward_weights: Dict[str, float] = field(default_factory=_default_reward_weights)
    backoff_strategy: BackoffStrategyConfig = field(default_factory=_default_backoff_strategy)

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("At least one region profile is required")
        if not self.segments:
            raise ValueError("At least one linear segment is required")
        valid_names = {region.name for region in self.regions}
        for segment in self.segments:
            if segment.region_name not in valid_names:
                raise ValueError(f"Unknown region reference '{segment.region_name}' in segments")
        if self.total_preambles <= 0:
            raise ValueError("total_preambles must be positive")
        if len(self.base_preamble_split) != 3:
            raise ValueError("base_preamble_split must contain CBRA/PBRA/CFRA counts")
        if sum(self.base_preamble_split) != self.total_preambles:
            raise ValueError("base_preamble_split must sum to total_preambles")
        if self.history_size != DEFAULT_HISTORY_SIZE:
            raise ValueError(
                "For the current environment binding history_size must equal DEFAULT_HISTORY_SIZE"
            )
        if self.history_size * len(_STAT_KEYS) != HISTORY_DIM:
            raise ValueError("history_size does not match HISTORY_DIM expectations")
        if self.slots_per_motion <= 0:
            raise ValueError("slots_per_motion must be positive")
        if self.coverage_window <= 0.0:
            raise ValueError("coverage_window must be positive")
        if self.motion_per_step <= 0.0:
            raise ValueError("motion_per_step must be positive")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale cannot be negative")
        if not {"throughput", "collision"}.issubset(self.reward_weights):
            raise ValueError("reward_weights must define throughput and collision keys")


# --- Core simulator implementation -------------------------------------------------


class MACSimulator:
    """Linear MAC simulator with moving coverage window and regional densities."""

    def __init__(self, config: MACSimulatorConfig, *, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self._rng = rng or np.random.default_rng()
        self._region_index = {region.name: idx for idx, region in enumerate(config.regions)}
        self._region_profiles = {region.name: region for region in config.regions}
        self._segments: List[Tuple[float, float, RegionSegment]] = []
        cursor = 0.0
        for segment in config.segments:
            start = cursor
            end = cursor + segment.length
            self._segments.append((start, end, segment))
            cursor = end
        self._track_length = cursor
        if self._track_length <= 0.0:
            raise ValueError("Total track length must be positive")
        self._motion_per_slot = config.motion_per_step / config.slots_per_motion
        self._history_buffer = np.zeros((config.history_size, len(_STAT_KEYS)), dtype=np.float32)
        self._history_index = 0
        self._preamble_allocation = np.array(config.base_preamble_split, dtype=np.int32)
        self._preamble_usage = np.zeros((PREAMBLE_SUBSET_COUNT,), dtype=np.float32)
        max_queue = config.backoff_strategy.max_backlog_steps + 1
        self._backoff_queue_cbra = np.zeros((max_queue,), dtype=np.float32)
        self._backoff_queue_pbra = np.zeros((max_queue,), dtype=np.float32)
        self._current_acb = 1.0
        self._request_scale = 10000
        self._backoff_scale = 100000
        self._active_terminals = np.zeros((MAC_PROTOCOL_COUNT,), dtype=np.float32)
        self._success_total = 0.0
        self._collision_total = 0.0
        self._success_breakdown = np.zeros((MAC_PROTOCOL_COUNT,), dtype=np.float32)
        self._collision_breakdown = np.zeros((MAC_PROTOCOL_COUNT,), dtype=np.float32)
        # Episode级别的累计统计 (总到达数和总成功数)
        self._total_arrivals_cbra = 0.0
        self._total_arrivals_pbra = 0.0
        self._total_arrivals_cfra = 0.0
        self._last_requests_cbra = 0.0
        self._last_requests_pbra = 0.0
        self._last_collision_ratio_cbra = 0.0
        self._last_collision_ratio_pbra = 0.0
        self._reset_coverage_position()

    def reseed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        self._rng = np.random.default_rng(seed)

    def reset_access_allocation(self) -> None:
        self._preamble_allocation = np.array(self.config.base_preamble_split, dtype=np.int32)
        self._update_preamble_usage(reset=True)

    def configure_access_state(
        self,
        *,
        cbra: Optional[int] = None,
        pbra: Optional[int] = None,
        cfra: Optional[int] = None,
    ) -> None:
        total = self.config.total_preambles
        current = self._preamble_allocation.astype(int)
        if cbra is not None:
            current[0] = int(cbra)
        if pbra is not None:
            current[1] = int(pbra)
        if cfra is not None:
            current[2] = int(cfra)
        if np.any(current < 0):
            raise ValueError("Preamble allocation cannot be negative")
        if int(np.sum(current)) != total:
            raise ValueError("Allocation must sum to total preambles")
        self._preamble_allocation = np.array(current).astype(int)
        self._update_preamble_usage(reset=True)

    def initialize_state(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.reseed(seed)
        self._history_buffer.fill(0.0)
        self._history_index = 0
        self._backoff_queue_cbra.fill(0.0)
        self._backoff_queue_pbra.fill(0.0)
        self._success_total = 0.0
        self._collision_total = 0.0
        self._success_breakdown.fill(0.0)
        self._collision_breakdown.fill(0.0)
        # 重置episode级别的累计统计
        self._total_arrivals_cbra = 0.0
        self._total_arrivals_pbra = 0.0
        self._total_arrivals_cfra = 0.0
        self._slot_counter = 0
        self._current_acb = 1.0
        self._reset_coverage_position()
        self._active_terminals.fill(0.0)
        self._last_requests_cbra = 0.0
        self._last_requests_pbra = 0.0
        self._last_collision_ratio_cbra = 0.0
        self._last_collision_ratio_pbra = 0.0
        self.reset_access_allocation()
        return self._build_observation()

    def run_slots(
        self,
        *,
        num_slots: int,
        params: Dict[str, float],
    ) -> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        arrival_cbra, arrival_pbra, arrival_cfra = self._forecast_arrivals(num_slots)

        # 累加本次step的arrivals到episode总计
        self._total_arrivals_cbra += float(np.sum(arrival_cbra))
        self._total_arrivals_pbra += float(np.sum(arrival_pbra))
        self._total_arrivals_cfra += float(np.sum(arrival_cfra))

        delta_cbra = int(params.get("M_CBRA", 0))
        delta_pbra = int(params.get("M_PBRA", 0))
        self._current_acb = float(np.clip(params.get("q_ACB", 1.0), 0.0, 1.0))
        action_valid = self._apply_preamble_delta(delta_cbra, delta_pbra)
        assert (
            int(np.sum(self._preamble_allocation)) == int(self.config.total_preambles)
        ), "Invalid preamble allocation after applying deltas"
        total_success_cbra = 0.0
        total_success_pbra = 0.0
        total_collision_cbra = 0.0
        total_collision_pbra = 0.0
        total_admitted_cbra = 0.0
        total_admitted_pbra = 0.0
        total_cfra_attempts = 0.0
        total_cbra_pool = 0.0
        total_pbra_pool = 0.0

        for slot_idx in range(num_slots):
            metrics = self._simulate_slot(
                cbra_new=int(arrival_cbra[slot_idx]),
                pbra_new=int(arrival_pbra[slot_idx]),
                cfra_demand=int(arrival_cfra[slot_idx]),
            )
            total_success_cbra += float(metrics["success_cbra"])
            total_success_pbra += float(metrics["success_pbra"])
            total_collision_cbra += float(metrics["collision_cbra"])
            total_collision_pbra += float(metrics["collision_pbra"])
            total_admitted_cbra += float(metrics["requests_cbra"])
            total_admitted_pbra += float(metrics["requests_pbra"])
            total_cfra_attempts += float(metrics["cfra_attempts"])
            total_cbra_pool += float(metrics["total_cbra_pool"])
            total_pbra_pool += float(metrics["total_pbra_pool"])

        available_cfra = int(num_slots * int(self._preamble_allocation[2]))
        success_cfra = min(total_cfra_attempts, available_cfra)
        collision_cfra = max(total_cfra_attempts - success_cfra, 0.0)

        # ============ 改进的Reward设计 v2.2 (优化权重与归一化) ============
        # 关键改进:
        # 1. 降低碰撞惩罚权重 (1.0 -> 0.5): 允许理论最优工作点的碰撞率
        # 2. CBRA和PBRA等权重 (均为1.0): 让Agent自然学习
        # 3. Log归一化Backlog: 适应数量级变化

        # 1. 分别计算CBRA和PBRA的资源容量
        preambles_cbra = int(self._preamble_allocation[0])
        preambles_pbra = int(self._preamble_allocation[1])
        capacity_cbra = float(num_slots * preambles_cbra)
        capacity_pbra = float(num_slots * preambles_pbra)

        # 2. 分别计算资源利用效率 (归一化到[0,1])
        # 理论上限约为 1/e ≈ 0.37 (Slotted ALOHA)
        # 乘以 e ≈ 2.718 使得最优表现接近 1.0
        if capacity_cbra > 0:
            efficiency_cbra = (total_success_cbra / capacity_cbra) * 2.718
        else:
            efficiency_cbra = 0.0

        if capacity_pbra > 0:
            efficiency_pbra = (total_success_pbra / capacity_pbra) * 2.718
        else:
            efficiency_pbra = 0.0

        # 3. 吞吐量奖励 (不加权重，让Agent自然学习)
        # CBRA和PBRA使用相同权重，让网络自主发现最优策略
        w_cbra, w_pbra = 1.0, 1.0
        throughput_reward = min(efficiency_cbra, efficiency_pbra)

        # 4. 分别计算碰撞率 (保持v2.1的优秀逻辑: 堵住ACB=0的漏洞)
        if total_admitted_cbra > 0:
            collision_rate_cbra = total_collision_cbra / total_admitted_cbra
        else:
            avg_queue_cbra = total_cbra_pool / float(num_slots)
            collision_rate_cbra = 1.0 if avg_queue_cbra > 1.0 else 0.0

        if total_admitted_pbra > 0:
            collision_rate_pbra = total_collision_pbra / total_admitted_pbra
        else:
            avg_queue_pbra = total_pbra_pool / float(num_slots)
            collision_rate_pbra = 1.0 if avg_queue_pbra > 1.0 else 0.0

        # 5. 碰撞惩罚 (修正: 降低权重!)
        # 理由: 在最优吞吐(Eff=1.0)时，碰撞率必然约为0.63
        # 如果系数是1.0，Net Gain = 1.0 - 0.63 = 0.37
        # 如果Agent保守(Eff=0.5, Coll=0.1)，Net Gain = 0.5 - 0.1 = 0.4
        # 结论: 系数过高会导致Agent保守。降至0.5后:
        # 最优: 1.0 - 0.5*0.63 = 0.685 > 保守: 0.5 - 0.5*0.1 = 0.45
        collision_penalty = 0.25 * max(collision_rate_cbra, collision_rate_pbra)

        # 6. CFRA资源不足惩罚 (保持不变，硬约束)
        # CFRA是已连接终端，资源不满足会导致服务中断
        cfra_penalty = 0.0
        if total_cfra_attempts > 0:
            cfra_shortage_rate = collision_cfra / total_cfra_attempts  # 资源不足率
            cfra_penalty = 5.0 * cfra_shortage_rate  # 5倍权重,强制保障CFRA资源充足

        # 7. 队列积压惩罚 (优化: 使用Log缩放，避免Magic Number 1000失效)
        # 使用log10可以处理从10到10000的大幅度波动
        # backlog=10 -> pen=0.1; backlog=100 -> pen=0.2; backlog=1000 -> pen=0.3
        # 相比线性除法，这能更平滑地处理爆发式流量（如卫星覆盖区切换）
        avg_backlog = (total_cbra_pool + total_pbra_pool) / float(num_slots)
        backlog_penalty = 0.1 * float(np.log10(avg_backlog + 1.0))

        # 综合reward
        # v2.2关键改进: 碰撞权重0.5 (鼓励探索容量边界) + 等权重策略 (让Agent自然学习)
        reward_total = throughput_reward - collision_penalty - cfra_penalty - backlog_penalty        # 用于统计的总吞吐量和总碰撞 (包含CFRA)
        throughput_total = total_success_cbra + total_success_pbra + success_cfra
        collision_total = total_collision_cbra + total_collision_pbra + collision_cfra

        avg_requests_cbra = total_admitted_cbra / float(num_slots)
        avg_requests_pbra = total_admitted_pbra / float(num_slots)
        ratio_cbra = (
            total_collision_cbra / total_admitted_cbra if total_admitted_cbra > 0.0 else 0.0
        )
        ratio_pbra = (
            total_collision_pbra / total_admitted_pbra if total_admitted_pbra > 0.0 else 0.0
        )

        self._success_total += throughput_total
        self._collision_total += collision_total
        self._success_breakdown += np.array(
            [total_success_cbra, total_success_pbra, success_cfra], dtype=np.float32
        )
        self._collision_breakdown += np.array(
            [total_collision_cbra, total_collision_pbra, collision_cfra], dtype=np.float32
        )

        self._last_requests_cbra = float(avg_requests_cbra)
        self._last_requests_pbra = float(avg_requests_pbra)
        self._last_collision_ratio_cbra = float(ratio_cbra)
        self._last_collision_ratio_pbra = float(ratio_pbra)

        preamble_cbra_capacity = max(float(num_slots * int(self._preamble_allocation[0])), 1.0)
        preamble_pbra_capacity = max(float(num_slots * int(self._preamble_allocation[1])), 1.0)
        preamble_cfra_capacity = max(float(num_slots * int(self._preamble_allocation[2])), 1.0)
        self._update_preamble_usage(
            cbra_util=total_success_cbra / preamble_cbra_capacity,
            pbra_util=total_success_pbra / preamble_pbra_capacity,
            cfra_util=success_cfra / preamble_cfra_capacity,
        )

        self._active_terminals = np.array(
            [total_cbra_pool, total_pbra_pool, total_cfra_attempts], dtype=np.float32
        )
        total_terminals = float(np.sum(self._active_terminals))
        if total_terminals > 0.0:
            self._active_terminals /= total_terminals
        else:
            self._active_terminals.fill(0.0)

        self._update_history(
            np.array(
                [
                    self._last_requests_cbra / self._request_scale,
                    self._last_requests_pbra / self._request_scale,
                    self._last_collision_ratio_cbra,
                    self._last_collision_ratio_pbra,
                    float(np.sum(self._backoff_queue_cbra)) / self._backoff_scale,
                    float(np.sum(self._backoff_queue_pbra)) / self._backoff_scale,
                ],
                dtype=np.float32,
            )
        )

        aggregates = {
            "throughput": throughput_total,
            "success_cbra": total_success_cbra,
            "success_pbra": total_success_pbra,
            "success_cfra": success_cfra,
            "collision_cbra": total_collision_cbra,
            "collision_pbra": total_collision_pbra,
            "collision_cfra": collision_cfra,
        }
        observation = self._build_observation()
        info = self._build_info(aggregates)
        info["action_valid"] = np.array([1.0 if action_valid else 0.0], dtype=np.float32)
        return reward_total, observation, info

    def _snapshot_motion_state(self) -> Tuple[int, float, float, List[Tuple[float, float]], np.ndarray]:
        prev_components_copy = [tuple(component) for component in self._prev_components]
        region_mixture_copy = getattr(self, "_region_mixture", np.zeros((len(self.config.regions),), dtype=np.float32)).copy()
        return (
            self._slot_counter,
            self._coverage_center,
            self._coverage_phase,
            prev_components_copy,
            region_mixture_copy,
        )

    def _restore_motion_state(
        self,
        snapshot: Tuple[int, float, float, List[Tuple[float, float]], np.ndarray],
    ) -> None:
        slot_counter, coverage_center, coverage_phase, prev_components, region_mixture = snapshot
        self._slot_counter = slot_counter
        self._coverage_center = coverage_center
        self._coverage_phase = coverage_phase
        self._prev_components = prev_components
        self._region_mixture = region_mixture

    def _forecast_arrivals(self, num_slots: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if num_slots <= 0:
            empty = np.zeros((0,), dtype=np.int32)
            return empty, empty, empty
        snapshot = self._snapshot_motion_state()
        cbra_arrivals = np.zeros((num_slots,), dtype=np.int32)
        pbra_arrivals = np.zeros((num_slots,), dtype=np.int32)
        cfra_arrivals = np.zeros((num_slots,), dtype=np.int32)

        for slot_idx in range(num_slots):
            components = self._current_interval_components()
            region_lengths = np.zeros((len(self.config.regions),), dtype=np.float32)
            cbra_demand = 0.0
            pbra_demand = 0.0
            cfra_density_weighted = 0.0
            for comp_start, comp_end in components:
                for seg_start, seg_end, segment in self._segments:
                    overlap = _interval_overlap(comp_start, comp_end, seg_start, seg_end)
                    if overlap <= 0.0:
                        continue
                    profile = self._region_profiles[segment.region_name]
                    noise = 1.0 + self._rng.uniform(-self.config.noise_scale, self.config.noise_scale)
                    cbra_demand += overlap * profile.cbra_density * noise
                    pbra_demand += overlap * profile.pbra_density * noise
                    cfra_density_weighted += overlap * profile.cfra_density
                    region_idx = self._region_index[profile.name]
                    region_lengths[region_idx] += overlap
            total_length = float(np.sum(region_lengths))
            if total_length > 0.0:
                average_cfra_density = cfra_density_weighted / total_length
            else:
                average_cfra_density = 0.0
            cbra_arrivals[slot_idx] = int(max(0, math.ceil(cbra_demand)))
            pbra_arrivals[slot_idx] = int(max(0, math.ceil(pbra_demand)))
            new_length = self._new_coverage_length(components)
            cfra_arrivals[slot_idx] = int(max(0, math.ceil(new_length * average_cfra_density)))
            self._prev_components = components
            self._advance_coverage()

        self._restore_motion_state(snapshot)
        return cbra_arrivals, pbra_arrivals, cfra_arrivals

    def _apply_preamble_delta(self, delta_cbra: int, delta_pbra: int) -> bool:
        if delta_cbra == 0 and delta_pbra == 0:
            return True

        base_allocation = self._preamble_allocation.astype(int)
        proposal = base_allocation.copy()
        proposal[0] += int(delta_cbra)
        proposal[1] += int(delta_pbra)
        proposal[2] = int(self.config.total_preambles) - int(proposal[0]) - int(proposal[1])

        if np.any(proposal < 1):
            return False
        if int(np.sum(proposal)) != int(self.config.total_preambles):
            raise RuntimeError("Preamble allocation proposal does not sum to total preambles")

        self._preamble_allocation = proposal.astype(np.int32)
        return True

    def count_success_and_fail(self, admitted: int, preamble: float) -> Tuple[int, int]:
        if admitted <= 0:
            return 0, 0
        elif admitted <= preamble:
            return admitted, 0
        elif preamble < admitted < preamble * 2:
            success = preamble * 2 - admitted
            collision = admitted - success
            return success, collision
        else:
            return 0, admitted

    def _simulate_slot(self, *, cbra_new: int, pbra_new: int, cfra_demand: int) -> Dict[str, float]:
        ready_cbra = self._pop_backoff_queue(self._backoff_queue_cbra, 1)
        ready_pbra = self._pop_backoff_queue(self._backoff_queue_pbra, 1)

        components = self._current_interval_components()
        self._update_region_mixture(components)

        ready_cbra_int = int(round(ready_cbra + cbra_new))
        ready_pbra_int = int(round(ready_pbra + pbra_new))

        allowed_cbra_new = min(int(round(self._current_acb * ready_cbra_int)), int(ready_cbra_int))
        allowed_pbra_new = min(int(round(self._current_acb * ready_pbra_int)), int(ready_pbra_int))
        blocked_cbra = max(int(ready_cbra_int) - allowed_cbra_new, 0)
        blocked_pbra = max(int(ready_pbra_int) - allowed_pbra_new, 0)

        success_cbra, collision_cbra = self.count_success_and_fail(
            allowed_cbra_new, int(self._preamble_allocation[0])
        )
        success_pbra, collision_pbra = self.count_success_and_fail(
            allowed_pbra_new, int(self._preamble_allocation[1])
        )
        need_to_backoff_cbra = int(blocked_cbra + collision_cbra)
        need_to_backoff_pbra = int(blocked_pbra + collision_pbra)
        if need_to_backoff_cbra > 0:
            self._schedule_backoff(
                self._backoff_queue_cbra,
                float(need_to_backoff_cbra),
                self.config.backoff_strategy.low,
            )
        if need_to_backoff_pbra > 0:
            self._schedule_backoff(
                self._backoff_queue_pbra,
                float(need_to_backoff_pbra),
                self.config.backoff_strategy.low,
            )
        cfra_attempts = int(max(cfra_demand, 0))

        self._prev_components = components
        self._advance_coverage()

        return {
            "success_cbra": float(success_cbra),
            "success_pbra": float(success_pbra),
            "collision_cbra": float(collision_cbra),
            "collision_pbra": float(collision_pbra),
            "requests_cbra": float(allowed_cbra_new),
            "requests_pbra": float(allowed_pbra_new),
            "cfra_attempts": float(cfra_attempts),
            "total_cbra_pool": float(ready_cbra_int),
            "total_pbra_pool": float(ready_pbra_int),
        }

    def _advance_coverage(self) -> None:
        self._slot_counter += 1
        self._coverage_center = (self._coverage_center + self._motion_per_slot) % self._track_length
        self._coverage_phase = (self._coverage_center % self._track_length) / self._track_length

    def _reset_coverage_position(self) -> None:
        self._slot_counter = 0
        self._coverage_center = self.config.coverage_window / 2.0
        self._coverage_phase = (self._coverage_center % self._track_length) / self._track_length
        components = self._current_interval_components()
        self._prev_components = components
        self._update_region_mixture(components)

    def _current_interval_components(self) -> List[Tuple[float, float]]:
        half = self.config.coverage_window / 2.0
        start = (self._coverage_center - half) % self._track_length
        end = start + self.config.coverage_window
        if end <= self._track_length:
            return [(start, end)]
        return [(start, self._track_length), (0.0, end - self._track_length)]

    def _new_coverage_length(self, current_components: List[Tuple[float, float]]) -> float:
        prev = self._prev_components
        current_length = sum(end - start for start, end in current_components)
        overlap = 0.0
        for c_start, c_end in current_components:
            for p_start, p_end in prev:
                overlap += _interval_overlap(c_start, c_end, p_start, p_end)
        return max(current_length - overlap, 0.0)

    def _select_backoff_window(self, collision_ratio: float) -> BackoffWindow:
        strategy = self.config.backoff_strategy
        if collision_ratio >= strategy.collision_threshold_high:
            return strategy.high
        if collision_ratio >= strategy.collision_threshold_medium:
            return strategy.medium
        return strategy.low

    def _schedule_backoff(
        self,
        queue: np.ndarray,
        amount: float,
        window: BackoffWindow,
    ) -> None:
        if amount <= 0.0:
            return
        queue_len = queue.shape[0]
        start_idx = max(0, min(window.min_steps, queue_len - 1))
        end_idx = max(0, min(window.max_steps, queue_len - 1))
        if end_idx < start_idx:
            end_idx = start_idx
        span = end_idx - start_idx + 1
        increment = amount / span
        queue[start_idx : end_idx + 1] += increment

    def _pop_backoff_queue(self, queue: np.ndarray, steps: int) -> float:
        if steps <= 0:
            return 0.0
        released = 0.0
        steps = min(steps, queue.shape[0])
        for _ in range(steps):
            released += float(queue[0])
            queue[:-1] = queue[1:]
            queue[-1] = 0.0
        return released * 0.8 # 20% retention factor

    def _update_history(self, stats: np.ndarray) -> None:
        self._history_buffer[self._history_index] = stats
        self._history_index = (self._history_index + 1) % self._history_buffer.shape[0]

    def _update_preamble_usage(
        self,
        *,
        cbra_util: float = 0.0,
        pbra_util: float = 0.0,
        cfra_util: float = 0.0,
        reset: bool = False,
    ) -> None:
        if reset:
            self._preamble_usage.fill(0.0)
            return
        util = np.array(
            [cbra_util, cbra_util, cbra_util, pbra_util, pbra_util, pbra_util, cfra_util, cfra_util],
            dtype=np.float32,
        )
        self._preamble_usage = 0.9 * self._preamble_usage + 0.1 * util

    def _update_region_mixture(self, components: List[Tuple[float, float]]) -> None:
        region_lengths = np.zeros((len(self.config.regions),), dtype=np.float32)
        for comp_start, comp_end in components:
            for seg_start, seg_end, segment in self._segments:
                overlap = _interval_overlap(comp_start, comp_end, seg_start, seg_end)
                if overlap <= 0.0:
                    continue
                region_idx = self._region_index[segment.region_name]
                region_lengths[region_idx] += overlap
        total_length = float(np.sum(region_lengths))
        if total_length > 0.0:
            self._region_mixture = (region_lengths / total_length).astype(np.float32)
        else:
            self._region_mixture = np.full(len(self.config.regions), 1.0 / len(self.config.regions), dtype=np.float32)

    def _build_observation(self) -> Dict[str, np.ndarray]:
        preamble_ratio = (self._preamble_allocation / self.config.total_preambles).astype(np.float32)
        history_flat = self._history_buffer.reshape(-1)
        return {
            "requests_cbra": np.array([
                self._last_requests_cbra / self._request_scale
            ], dtype=np.float32),
            "requests_pbra": np.array([
                self._last_requests_pbra / self._request_scale
            ], dtype=np.float32),
            "collision_ratio_cbra": np.array([self._last_collision_ratio_cbra], dtype=np.float32),
            "collision_ratio_pbra": np.array([self._last_collision_ratio_pbra], dtype=np.float32),
            "active_terminals_dist": self._active_terminals.astype(np.float32),
            "preamble_usage": self._preamble_usage.astype(np.float32),
            "preamble_allocation": preamble_ratio,
            "current_ACB_factor": np.array([self._current_acb], dtype=np.float32),
            "history_stats": history_flat.astype(np.float32),
            "pending_backoff_cbra": np.array(
                [float(np.sum(self._backoff_queue_cbra)) / self._backoff_scale], dtype=np.float32
            ),
            "pending_backoff_pbra": np.array(
                [float(np.sum(self._backoff_queue_pbra)) / self._backoff_scale], dtype=np.float32
            ),
            "region_mixture": self._region_mixture.astype(np.float32),
            "coverage_center": np.array([self._coverage_center, 0.0], dtype=np.float32),
            "coverage_phase": np.array([self._coverage_phase], dtype=np.float32),
        }

    def _build_info(self, aggregates: Dict[str, float]) -> Dict[str, np.ndarray]:
        info: Dict[str, np.ndarray] = {
            "throughput": np.array([aggregates["throughput"]], dtype=np.float32),
            "success_total": np.array([self._success_total], dtype=np.float32),
            "collision_total": np.array([self._collision_total], dtype=np.float32),
            "success_cbra": np.array([aggregates["success_cbra"]], dtype=np.float32),
            "success_pbra": np.array([aggregates["success_pbra"]], dtype=np.float32),
            "success_cfra": np.array([aggregates["success_cfra"]], dtype=np.float32),
            "collision_cbra": np.array([aggregates["collision_cbra"]], dtype=np.float32),
            "collision_pbra": np.array([aggregates["collision_pbra"]], dtype=np.float32),
            "collision_cfra": np.array([aggregates["collision_cfra"]], dtype=np.float32),
            # Episode级别的累计统计
            "total_arrivals_cbra": np.array([self._total_arrivals_cbra], dtype=np.float32),
            "total_arrivals_pbra": np.array([self._total_arrivals_pbra], dtype=np.float32),
            "total_arrivals_cfra": np.array([self._total_arrivals_cfra], dtype=np.float32),
            "total_success_cbra": np.array([self._success_breakdown[0]], dtype=np.float32),
            "total_success_pbra": np.array([self._success_breakdown[1]], dtype=np.float32),
            "total_success_cfra": np.array([self._success_breakdown[2]], dtype=np.float32),
            "pending_backoff_cbra": np.array([float(np.sum(self._backoff_queue_cbra))], dtype=np.float32),
            "pending_backoff_pbra": np.array([float(np.sum(self._backoff_queue_pbra))], dtype=np.float32),
            "collision_ratio_cbra": np.array([self._last_collision_ratio_cbra], dtype=np.float32),
            "collision_ratio_pbra": np.array([self._last_collision_ratio_pbra], dtype=np.float32),
            "region_mixture": self._region_mixture.astype(np.float32),
            "preamble_allocation": (
                self._preamble_allocation
            ).astype(np.int16),
            "preamble_allocation_counts": self._preamble_allocation.astype(np.float32),
            "preamble_usage": self._preamble_usage.astype(np.float32),
        }
        return info


# --- Utility functions ------------------------------------------------------------


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# --- Default configuration factory ------------------------------------------------
def default_simulator_config() -> MACSimulatorConfig:
    regions = (
        RegionTrafficProfile(name="suburban",
                             cbra_density=3.0 * 0.2,
                             pbra_density=3.0 * 0.7,
                             cfra_density=3.0 * 0.1),
        RegionTrafficProfile(name="periurbanType1",
                             cbra_density=4.0 * 0.2,
                             pbra_density=4.0 * 0.3,
                             cfra_density=4.0 * 0.5),
        RegionTrafficProfile(name="periurbanType2",
                             cbra_density=4.0 * 0.2,
                             pbra_density=4.0 * 0.7,
                             cfra_density=4.0 * 0.1),
        RegionTrafficProfile(name="urban",
                             cbra_density=100.0 * 0.75,
                             pbra_density=100.0 * 0.1,
                             cfra_density=100.0 * 0.15),
    )
    pattern: Sequence[Tuple[str, float]] = (
        ("suburban", 2.0),
        ("periurbanType1", 1.0),
        ("urban", 5.0),
        ("periurbanType2", 1.0),
        ("suburban", 2.0),
    )
    segments: List[RegionSegment] = []
    for _ in range(1):
        for name, length in pattern:
            segments.append(RegionSegment(region_name=name, length=length))
    return MACSimulatorConfig(regions=regions, segments=tuple(segments))


__all__ = [
    "MACSimulator",
    "MACSimulatorConfig",
    "BackoffWindow",
    "BackoffStrategyConfig",
    "RegionTrafficProfile",
    "RegionSegment",
    "default_simulator_config",
    "MAC_PROTOCOL_COUNT",
    "PREAMBLE_SUBSET_COUNT",
    "DEFAULT_TOTAL_PREAMBLES",
    "HISTORY_DIM",
]
