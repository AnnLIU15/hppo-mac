"""MAC 接入仿真占位实现。

该模块主要提供 ``MACSimulator`` 类，用于在强化学习环境中模拟
LEO 卫星的随机接入流量、切换事件与 MAC 反馈。目前实现为占位逻辑，
通过可配置的统计特征生成观测、奖励与调度信息，便于后续替换为
高保真模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.traffic import (
    BatchedPoissonConfig,
    CoxProcessConfig,
    batched_poisson_arrival,
    generate_cox_points,
    sample_residence_time,
)

# 默认观测维度常量，占位值可在后续阶段根据真实模型调整。
MAC_PROTOCOL_COUNT = 4
PREAMBLE_SUBSET_COUNT = 8
HISTORY_DIM = 10
DEFAULT_TOTAL_PREAMBLES = 64


@dataclass(frozen=True)
class RegionTrafficProfile:
    """区域流量画像。

    Attributes:
        name: 区域名称，用于日志记录。
        area_weight: 区域覆盖权重，归一化后用于采样区域。
        cbra_density: CBRA 小包接入的基线密度。
        pbra_density: PBRA 小包接入的基线密度。
        handover_intensity: 切换用户的基线强度（非竞争接入）。
        scheduling_period: 定期调度周期（以接入时隙为单位）。
        noise_scale: 叠加随机扰动的尺度，模拟人口进出的小尺度波动。
    cox_intensity_mean: Cox 过程的平均强度，决定空间热点水平。
    cox_intensity_variance: Cox 过程强度波动幅度。
    cox_grid: 生成 Cox 网格的形状，控制空间分辨率。
    region_bounds: 区域边界矩形，用于生成空间热点。
    batch_mean: 批量泊松过程的平均批量大小。
    batch_std: 批量泊松过程的批量标准差。
    residence_time_mean: 平均驻留时间，用于调节需求放大系数。
    """

    name: str
    area_weight: float
    cbra_density: float
    pbra_density: float
    handover_intensity: float
    scheduling_period: int
    noise_scale: float = 0.1
    cox_intensity_mean: float = 1.0
    cox_intensity_variance: float = 0.5
    cox_grid: Tuple[int, int] = (6, 6)
    region_bounds: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    batch_mean: float = 5.0
    batch_std: float = 0.5
    residence_time_mean: float = 5.0


@dataclass(frozen=True)
class CoveragePatch:
    """描述轨道覆盖范围中一个网格块的区域混合权重。"""

    name: str
    center: Tuple[float, float]
    radius: float
    strength: float
    region_weights: Tuple[float, ...]


@dataclass(frozen=True)
class BackoffWindow:
    """描述退避范围的上下界（以决策步为单位）。"""

    min_steps: int
    max_steps: int


@dataclass(frozen=True)
class BackoffStrategyConfig:
    """随机退避策略配置。"""

    low: BackoffWindow = field(default_factory=lambda: BackoffWindow(0, 4))
    medium: BackoffWindow = field(default_factory=lambda: BackoffWindow(1, 12))
    high: BackoffWindow = field(default_factory=lambda: BackoffWindow(6, 24))
    collision_threshold_medium: float = 0.05
    collision_threshold_high: float = 0.1
    max_backlog_steps: int = 48


@dataclass(frozen=True)
class MACSimulatorConfig:
    """MAC 仿真器的配置集合。"""

    regions: Tuple[RegionTrafficProfile, ...]
    history_size: int = HISTORY_DIM
    protocol_count: int = MAC_PROTOCOL_COUNT
    preamble_subset_count: int = PREAMBLE_SUBSET_COUNT
    total_preambles: int = DEFAULT_TOTAL_PREAMBLES
    base_preamble_split: Tuple[int, int, int] = (24, 24, 16)
    reward_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "throughput": 1.0,
            "collision": -0.5,
            # 当 CFRA 与理论可移交数量不匹配时的均方误差惩罚（通常为负值以降低奖励）
            "cfra_mse": -1,
        }
    )
    region_transition_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None
    coverage_patches: Optional[Tuple[CoveragePatch, ...]] = None
    coverage_drift_radius: float = 1.0
    coverage_cycle_slots: float = 1440.0
    coverage_jitter: float = 0.05
    coverage_smoothing: float = 0.2
    coverage_total_footprint: float = 1.0
    backoff_strategy: BackoffStrategyConfig = field(default_factory=BackoffStrategyConfig)

    def __post_init__(self) -> None:
        total = sum(self.base_preamble_split)
        if total != self.total_preambles:
            raise ValueError(
                "base_preamble_split 各部分之和必须等于 total_preambles"
            )
        if self.region_transition_matrix is not None:
            matrix = self.region_transition_matrix
            row_count = len(matrix)
            if row_count != len(self.regions):
                raise ValueError("region_transition_matrix 行数应与 regions 数量一致")
            for row in matrix:
                if len(row) != len(self.regions):
                    raise ValueError("region_transition_matrix 应为方阵")
                if any(value < 0 for value in row):
                    raise ValueError("region_transition_matrix 中的元素需为非负数")
                if sum(row) <= 0:
                    raise ValueError("region_transition_matrix 的行和需为正值")
        if self.coverage_cycle_slots <= 0:
            raise ValueError("coverage_cycle_slots 应为正数")
        if self.coverage_drift_radius < 0:
            raise ValueError("coverage_drift_radius 应为非负数")
        if self.coverage_jitter < 0:
            raise ValueError("coverage_jitter 应为非负数")
        if not 0.0 <= self.coverage_smoothing < 1.0:
            raise ValueError("coverage_smoothing 应位于 [0, 1) 区间")
        if self.coverage_total_footprint <= 0:
            raise ValueError("coverage_total_footprint 应为正数")
        if self.coverage_patches is not None:
            for patch in self.coverage_patches:
                if len(patch.region_weights) != len(self.regions):
                    raise ValueError("coverage_patch.region_weights 长度需与 regions 数量一致")
                if patch.radius <= 0:
                    raise ValueError("coverage_patch.radius 应为正数")
                if patch.strength < 0:
                    raise ValueError("coverage_patch.strength 应为非负数")
        strategy = self.backoff_strategy
        if strategy.max_backlog_steps < 1:
            raise ValueError("max_backlog_steps 应为正整数")
        for window in (strategy.low, strategy.medium, strategy.high):
            if window.min_steps < 0 or window.max_steps < window.min_steps:
                raise ValueError("BackoffWindow 的范围定义不合法")
            if window.max_steps > strategy.max_backlog_steps:
                raise ValueError("BackoffWindow 上界需不超过 max_backlog_steps")
        if not 0.0 <= strategy.collision_threshold_medium <= 1.0:
            raise ValueError("collision_threshold_medium 应位于 [0, 1]")
        if not 0.0 <= strategy.collision_threshold_high <= 1.0:
            raise ValueError("collision_threshold_high 应位于 [0, 1]")
        if strategy.collision_threshold_high < strategy.collision_threshold_medium:
            raise ValueError("高拥塞阈值应不小于中等拥塞阈值")


class MACSimulator:
    """提供初始化状态与接入仿真的占位实现。

    该类暴露 ``initialize_state`` 与 ``run_slots`` 两个核心方法，供
    gymnasium 环境直接调用。未来可在此类内部替换为高保真的物理与流量模型。
    """

    def __init__(self, config: MACSimulatorConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self._rng = rng or np.random.default_rng()
        self._time_slot = 0
        self._history = np.zeros((self.config.history_size,), dtype=np.float32)
        self._base_preambles = np.asarray(self.config.base_preamble_split, dtype=np.float32)
        self._initial_preambles = self._base_preambles.copy()
        self._current_preambles = self._base_preambles.copy()

        self._regions = self.config.regions
        self._region_count = len(self._regions)
        if self._region_count == 0:
            raise ValueError("至少应配置一个区域画像")

        self._cox_configs = [
            CoxProcessConfig(
                intensity_mean=profile.cox_intensity_mean,
                intensity_variance=profile.cox_intensity_variance,
                grid_shape=profile.cox_grid,
                region_bounds=profile.region_bounds,
            )
            for profile in self._regions
        ]
        self._cox_cell_counts = np.array(
            [cfg.grid_shape[0] * cfg.grid_shape[1] for cfg in self._cox_configs],
            dtype=np.float32,
        )
        self._batch_mean = np.array(
            [profile.batch_mean for profile in self._regions],
            dtype=np.float32,
        )
        self._batch_std = np.array(
            [profile.batch_std for profile in self._regions],
            dtype=np.float32,
        )
        self._residence_means = np.array(
            [profile.residence_time_mean for profile in self._regions],
            dtype=np.float32,
        )

        region_weights = np.array([profile.area_weight for profile in self._regions], dtype=np.float64)
        weight_sum = region_weights.sum()
        if weight_sum <= 0:
            raise ValueError("region area_weight 应为正值")
        self._region_prob = region_weights / weight_sum

        self._coverage_patches = tuple(self.config.coverage_patches or ())
        if self._coverage_patches:
            self._coverage_reference = float(
                max(sum(patch.strength for patch in self._coverage_patches), 1.0)
            )
        else:
            self._coverage_reference = float(self._region_count)

        if self.config.region_transition_matrix is not None:
            matrix = np.asarray(self.config.region_transition_matrix, dtype=np.float64)
            row_sums = matrix.sum(axis=1, keepdims=True)
            self._transition_matrix = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix),
                where=row_sums != 0,
            )
            row_totals = self._transition_matrix.sum(axis=1)
            for idx, total in enumerate(row_totals):
                if total <= 0.0:
                    self._transition_matrix[idx] = self._region_prob
        else:
            if self._region_count == 1:
                self._transition_matrix = np.ones((1, 1), dtype=np.float64)
            else:
                stay_prob = 0.6
                off_diag = (1.0 - stay_prob) / (self._region_count - 1)
                matrix = np.full((self._region_count, self._region_count), off_diag, dtype=np.float64)
                np.fill_diagonal(matrix, stay_prob)
                self._transition_matrix = matrix

        self._current_region_idx: Optional[int] = None
        self._coverage_phase = float(self._rng.random())
        self._footprint_center = np.zeros(2, dtype=np.float32)
        self._region_mixture_state = self._region_prob.astype(np.float32)
        self._backoff_strategy = self.config.backoff_strategy
        backlog_len = self._backoff_strategy.max_backlog_steps + 1
        self._backoff_queue_cbra = np.zeros(backlog_len, dtype=np.float32)
        self._backoff_queue_pbra = np.zeros(backlog_len, dtype=np.float32)
        self._last_collision_ratio_cbra = 0.0
        self._last_collision_ratio_pbra = 0.0

    def reseed(self, seed: Optional[int]) -> None:
        """重置随机数生成器。"""

        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def compute_combo_mask(self, delta_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
        """基于当前前导分配计算合法的 (ΔCBRA, ΔPBRA) 组合 mask。"""

        total = float(self.config.total_preambles)
        current_cbra = float(self._current_preambles[0])
        current_pbra = float(self._current_preambles[1])
        mask = np.zeros(len(delta_pairs), dtype=np.float32)

        for idx, (delta_cbra, delta_pbra) in enumerate(delta_pairs):
            next_cbra = current_cbra + float(delta_cbra)
            next_pbra = current_pbra + float(delta_pbra)
            if next_cbra < 0.0 or next_pbra < 0.0:
                continue
            if next_cbra + next_pbra > total:
                continue
            mask[idx] = 1.0

        if not np.any(mask):
            try:
                zero_idx = delta_pairs.index((0, 0))  # type: ignore[arg-type]
            except ValueError:
                zero_idx = 0
            mask[zero_idx] = 1.0
        return mask

    def reset_access_allocation(self) -> None:
        """重置接入资源分配到初始状态。"""

        self._initial_preambles = self._base_preambles.copy()
        self._current_preambles = self._base_preambles.copy()

    def initialize_state(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """生成初始观测状态。

        Args:
            seed: 可选的随机种子，用于复现。

        Returns:
            与环境观测空间匹配的张量字典。
        """

        self.reseed(seed)
        self._time_slot = 0
        self._history.fill(0.0)
        self._current_preambles = self._initial_preambles.copy()
        self._current_region_idx = int(self._rng.choice(self._region_count, p=self._region_prob))
        self._coverage_phase = float(self._rng.random())
        self._footprint_center.fill(0.0)
        self._region_mixture_state = self._region_prob.astype(np.float32)
        self._backoff_queue_cbra.fill(0.0)
        self._backoff_queue_pbra.fill(0.0)
        self._last_collision_ratio_cbra = 0.0
        self._last_collision_ratio_pbra = 0.0
        return self._sample_observation(region_mixture=self._region_mixture_state)

    def configure_access_state(
        self,
        *,
        cbra: Optional[int] = None,
        pbra: Optional[int] = None,
        cfra: Optional[int] = None,
    ) -> None:
        """强制设置初始 RA 资源。"""

        total = float(self.config.total_preambles)
        current = self._initial_preambles.copy().astype(np.float64)

        if cbra is not None:
            current[0] = float(cbra)
        if pbra is not None:
            current[1] = float(pbra)
        if cfra is not None:
            current[2] = float(cfra)
        else:
            current[2] = total - current[0] - current[1]

        current = np.clip(current, 0.0, None)
        total_sum = current.sum()
        if total_sum <= 0:
            raise ValueError("RA 资源总量必须为正数")

        if not np.isclose(total_sum, total):
            current = current / total_sum * total

        self._initial_preambles = current.astype(np.float32)
        self._current_preambles = self._initial_preambles.copy()
        self._base_preambles = self._initial_preambles.copy()

    def run_slots(
        self,
        num_slots: int,
        params: Mapping[str, float],
    ) -> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """执行给定数量的接入时隙仿真。

        Args:
            num_slots: 本次仿真包含的接入时隙数量。
            params: 来自强化学习动作的 MAC 参数调整。

        Returns:
            reward: 标量奖励。
            next_state: 下一时刻的观测字典。
            info: 额外诊断信息的字典。
        """

        if num_slots <= 0:
            raise ValueError("num_slots 应为正整数")

        # 解码动作与调整资源
        cbra_adj = float(params.get("M_CBRA", 0.0))
        pbra_adj = float(params.get("M_PBRA", 0.0))
        acb_factor = float(np.clip(params.get("q_ACB", 0.5), 0.0, 1.0))

        cbra_count, pbra_count, cfra_count = self._apply_preamble_adjustment(
            cbra_adj=cbra_adj,
            pbra_adj=pbra_adj,
        )
        # 覆盖混合更新与重试释放
        mixture = self._update_region_mixture(num_slots)
        release_steps = 1 if num_slots > 0 else 0
        cbra_retries = (
            self._pop_backoff_queue(self._backoff_queue_cbra, release_steps)
            if np.any(self._backoff_queue_cbra)
            else 0.0
        )
        pbra_retries = (
            self._pop_backoff_queue(self._backoff_queue_pbra, release_steps)
            if np.any(self._backoff_queue_pbra)
            else 0.0
        )

        footprint_scale = float(self.config.coverage_total_footprint)
        slot_count = max(int(num_slots), 1)
        per_slot_region_requests_cbra = np.zeros((slot_count, self._region_count), dtype=np.float64)
        per_slot_region_requests_pbra = np.zeros((slot_count, self._region_count), dtype=np.float64)
        per_region_requests_cfra = np.zeros(self._region_count, dtype=np.float64)

        # 使用确定性期望值替代原始随机批次泊松抽样：
        # demand = base_density * region_scale * density_factor * num_slots * batch_mean * residence_scale
        for idx, weight in enumerate(mixture):
            if weight <= 0.0:
                continue
            region = self._regions[idx]
            density_factor, dwell_time = self._sample_region_factors(idx)
            region_scale = max(weight * footprint_scale, 0.0)

            # 基于期望到达率而非随机抽样
            base_cbra = max(region.cbra_density * region_scale, 0.0)
            base_pbra = max(region.pbra_density * region_scale, 0.0)

            # 利用区域设定的 batch_mean 作为平均批次规模，得到“单个 slot”的期望到达数
            residence_scale = max(dwell_time / max(self._residence_means[idx], 1e-10), 0.1)
            expected_cbra_per_slot = float(
                base_cbra * density_factor * float(self._batch_mean[idx]) * residence_scale
            )
            print(expected_cbra_per_slot, mixture)
            expected_pbra_per_slot = float(
                base_pbra * density_factor * float(self._batch_mean[idx]) * residence_scale
            )

            # handover 期望值（使用平均批次大小近似）
            expected_handover = float(
                region.handover_intensity
                * region_scale
                * density_factor
                * max(num_slots, 1)
                * float(self._batch_mean[idx])
            )

            per_slot_region_requests_cbra[:, idx] = expected_cbra_per_slot
            per_slot_region_requests_pbra[:, idx] = expected_pbra_per_slot
            per_region_requests_cfra[idx] = expected_handover
        # 将等待释放的重试按混合权重分配到区域（期望值基础）
        if cbra_retries > 0.0:
            retry_per_slot = float(cbra_retries) / slot_count
            per_slot_region_requests_cbra += retry_per_slot * mixture
        if pbra_retries > 0.0:
            retry_per_slot = float(pbra_retries) / slot_count
            per_slot_region_requests_pbra += retry_per_slot * mixture

        per_region_requests_cbra = per_slot_region_requests_cbra.sum(axis=0)
        per_region_requests_pbra = per_slot_region_requests_pbra.sum(axis=0)

        slot_requests_cbra = per_slot_region_requests_cbra.sum(axis=1)
        slot_requests_pbra = per_slot_region_requests_pbra.sum(axis=1)

        cbra_requests = float(np.sum(slot_requests_cbra))
        pbra_requests = float(np.sum(slot_requests_pbra))
        handover_requests = float(np.sum(per_region_requests_cfra))
        # ACB（接入控制）在此以确定性接受率应用：被允许进入的尝试数 = arrivals * acb_factor
        accepted_slot_cbra = slot_requests_cbra * acb_factor
        accepted_slot_pbra = slot_requests_pbra * acb_factor

        accepted_cbra = float(np.sum(accepted_slot_cbra))
        accepted_pbra = float(np.sum(accepted_slot_pbra))

        # 包括因退避释放的重试（已经加回到 per_region_requests_* 中），上面已计入

        # 资源竞争：引入“缓冲”逻辑，每个 slot 单独计算碰撞
        eps = 1e-6
        success_cbra, collision_cbra = self._apply_slot_buffer(
            slot_requests=accepted_slot_cbra,
            capacity=float(cbra_count),
            queue=self._backoff_queue_cbra,
        )
        success_pbra, collision_pbra = self._apply_slot_buffer(
            slot_requests=accepted_slot_pbra,
            capacity=float(pbra_count),
            queue=self._backoff_queue_pbra,
        )

        actual_transfer = min(handover_requests, float(cfra_count) * num_slots)
        handover_success = float(actual_transfer)
        handover_blocked = max(handover_requests - handover_success, 0.0)

        success_total = success_cbra + success_pbra + handover_success * 5
        collision_total = collision_cbra + collision_pbra

        eps = 1e-6
        share_cbra = per_region_requests_cbra / max(cbra_requests, eps)
        share_pbra = per_region_requests_pbra / max(pbra_requests, eps)
        share_cfra = per_region_requests_cfra / max(handover_requests, eps)

        per_region_success_cbra = success_cbra * share_cbra
        per_region_success_pbra = success_pbra * share_pbra
        per_region_success_cfra = handover_success * share_cfra
        per_region_collision_cbra = collision_cbra * share_cbra
        per_region_collision_pbra = collision_pbra * share_pbra

        reward = (
            self.config.reward_weights.get("throughput", 0.0) * success_total
            + self.config.reward_weights.get("collision", 0.0) * collision_total
        )

        # CFRA 匹配惩罚（相对误差的 MSE）——当 handover_requests 为 0 时不计入惩罚
        if handover_requests > eps:
            ratio = (float(cfra_count) * num_slots) / (handover_requests + eps)
            cfra_mse = float((ratio - 1.0) ** 2)
        else:
            cfra_mse = 0.0
        reward += float(self.config.reward_weights.get("cfra_mse", 0.0)) * cfra_mse

        # 碰撞率按已接受的尝试比例计算（接受但失败的比例）
        collision_ratio_cbra = float(collision_cbra / max(accepted_cbra, eps))
        collision_ratio_pbra = float(collision_pbra / max(accepted_pbra, eps))

        window_cbra = self._select_backoff_window(collision_ratio_cbra)
        window_pbra = self._select_backoff_window(collision_ratio_pbra)

        self._last_collision_ratio_cbra = float(collision_ratio_cbra)
        self._last_collision_ratio_pbra = float(collision_ratio_pbra)

        next_state = self._sample_observation(
            cbra_requests=cbra_requests,
            pbra_requests=pbra_requests,
            collision_ratio_cbra=collision_ratio_cbra,
            collision_ratio_pbra=collision_ratio_pbra,
            acb_factor=acb_factor,
            preamble_allocation=self._current_preamble_ratio(),
            success_total=success_total,
            collision_total=collision_total,
            region_mixture=mixture,
            coverage_center=self._footprint_center,
            coverage_phase=self._coverage_phase,
            pending_backoff_cbra=self._backoff_queue_cbra.sum(),
            pending_backoff_pbra=self._backoff_queue_pbra.sum(),
        )

        dominant_idx = int(np.argmax(mixture)) if mixture.size else 0
        self._current_region_idx = dominant_idx
        dominant_region_name = self._regions[dominant_idx].name

        info = {
            "region": dominant_region_name,
            "dominant_region": dominant_region_name,
            "region_index": np.array([dominant_idx], dtype=np.float32),
            "region_mixture": mixture.astype(np.float32),
            "coverage_center": self._footprint_center.astype(np.float32),
            "coverage_phase": np.array([self._coverage_phase], dtype=np.float32),
            "requests_cbra": np.array([cbra_requests], dtype=np.float32),
            "requests_pbra": np.array([pbra_requests], dtype=np.float32),
            "requests_cfra": np.array([handover_requests], dtype=np.float32),
            "region_requests_cbra": per_region_requests_cbra.astype(np.float32),
            "region_requests_pbra": per_region_requests_pbra.astype(np.float32),
            "region_requests_cfra": per_region_requests_cfra.astype(np.float32),
            "success_cbra": np.array([success_cbra], dtype=np.float32),
            "success_pbra": np.array([success_pbra], dtype=np.float32),
            "success_cfra": np.array([handover_success], dtype=np.float32),
            "collision_cbra": np.array([collision_cbra], dtype=np.float32),
            "collision_pbra": np.array([collision_pbra], dtype=np.float32),
            "collision_cfra": np.array([handover_blocked], dtype=np.float32),
            "region_success_cbra": per_region_success_cbra.astype(np.float32),
            "region_success_pbra": per_region_success_pbra.astype(np.float32),
            "region_success_cfra": per_region_success_cfra.astype(np.float32),
            "region_collision_cbra": per_region_collision_cbra.astype(np.float32),
            "region_collision_pbra": per_region_collision_pbra.astype(np.float32),
            "preamble_allocation": self._current_preamble_ratio().astype(np.float32),
            "success_total": np.array([success_total], dtype=np.float32),
            "collision_total": np.array([collision_total], dtype=np.float32),
            "throughput": np.array([success_total], dtype=np.float32),
            "slots_simulated": np.array([num_slots], dtype=np.float32),
            "pending_backoff_cbra": np.array([self._backoff_queue_cbra.sum()], dtype=np.float32),
            "pending_backoff_pbra": np.array([self._backoff_queue_pbra.sum()], dtype=np.float32),
            "backoff_window_cbra": np.array([window_cbra.min_steps, window_cbra.max_steps], dtype=np.float32),
            "backoff_window_pbra": np.array([window_pbra.min_steps, window_pbra.max_steps], dtype=np.float32),
            "retries_released_cbra": np.array([cbra_retries], dtype=np.float32),
            "retries_released_pbra": np.array([pbra_retries], dtype=np.float32),
            "acb_factor": acb_factor,
        }

        self._time_slot += num_slots
        self._update_history(reward)

        return float(reward), next_state, info

    def _sample_region_factors(self, region_idx: int) -> Tuple[float, float]:
        """结合 Cox 过程与驻留时间采样区域流量因子。"""

        cfg = self._cox_configs[region_idx]
        points = generate_cox_points(cfg, self._rng)
        cell_count = float(max(self._cox_cell_counts[region_idx], 1.0))
        density_factor = max(len(points), 1.0) / cell_count
        dwell_time = sample_residence_time(float(self._residence_means[region_idx]), self._rng)
        return density_factor, dwell_time

    def _draw_demand(
        self,
        base_density: float,
        delta: float,
        num_slots: int,
        *,
        region_idx: int,
        density_factor: float,
        dwell_time: float,
    ) -> float:
        """使用批量泊松与 Cox 过程生成随机到达需求。"""

        seasonal = np.sin(2 * np.pi * (self._time_slot % 1440) / 1440.0)
        baseline = max(base_density * (1.0 + delta), 0.0)
        effective_rate = max(baseline * density_factor * (1.0 + 0.2 * seasonal), 1e-6)
        poisson_cfg = BatchedPoissonConfig(
            rate=effective_rate,
            batch_mean=float(self._batch_mean[region_idx]),
            batch_std=float(self._batch_std[region_idx]),
        )
        arrivals = batched_poisson_arrival(poisson_cfg, steps=max(num_slots, 1), rng=self._rng)
        demand = float(np.sum(arrivals))
        residence_scale = max(dwell_time / max(self._residence_means[region_idx], 1e-3), 0.1)
        return max(demand * residence_scale, 0.0)

    def _draw_handover(
        self,
        intensity: float,
        num_slots: int,
        *,
        region_idx: int,
        density_factor: float,
    ) -> float:
        """结合批量泊松过程模拟切换用户到达。"""

        base_rate = max(intensity * density_factor, 1e-6)
        poisson_cfg = BatchedPoissonConfig(
            rate=base_rate,
            batch_mean=float(self._batch_mean[region_idx]),
            batch_std=float(self._batch_std[region_idx]),
        )
        arrivals = batched_poisson_arrival(poisson_cfg, steps=max(num_slots, 1), rng=self._rng)
        return float(max(np.sum(arrivals), 0.0))

    def _sample_observation(
        self,
        *,
        cbra_requests: Optional[float] = None,
        pbra_requests: Optional[float] = None,
        collision_ratio_cbra: Optional[float] = None,
        collision_ratio_pbra: Optional[float] = None,
        acb_factor: Optional[float] = None,
        preamble_allocation: Optional[np.ndarray] = None,
        success_total: Optional[float] = None,
        collision_total: Optional[float] = None,
        region_mixture: Optional[np.ndarray] = None,
        coverage_center: Optional[np.ndarray] = None,
        coverage_phase: Optional[float] = None,
        pending_backoff_cbra: Optional[float] = None,
        pending_backoff_pbra: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """生成观测张量字典。"""

        rng = self._rng
        observation = {
            "requests_cbra": np.array([cbra_requests if cbra_requests is not None else rng.gamma(2.0, 5.0)], dtype=np.float32),
            "requests_pbra": np.array([pbra_requests if pbra_requests is not None else rng.gamma(2.0, 4.0)], dtype=np.float32),
            "collision_ratio_cbra": np.array([
                collision_ratio_cbra if collision_ratio_cbra is not None else rng.beta(2.0, 5.0)
            ], dtype=np.float32),
            "collision_ratio_pbra": np.array([
                collision_ratio_pbra if collision_ratio_pbra is not None else rng.beta(2.0, 5.0)
            ], dtype=np.float32),
            "active_terminals_dist": rng.dirichlet(
                alpha=np.ones(self.config.protocol_count, dtype=np.float64) + 0.1
            ).astype(np.float32),
            "preamble_usage": rng.dirichlet(
                alpha=np.ones(self.config.preamble_subset_count, dtype=np.float64) + 0.1
            ).astype(np.float32),
            "current_ACB_factor": np.array([
                acb_factor if acb_factor is not None else rng.uniform(0.2, 0.8)
            ], dtype=np.float32),
            "history_stats": self._history.astype(np.float32).copy(),
            "preamble_allocation": (
                preamble_allocation.astype(np.float32)
                if preamble_allocation is not None
                else self._current_preamble_ratio().astype(np.float32)
            ),
            "success_total": np.array([
                success_total if success_total is not None else 0.0
            ], dtype=np.float32),
            "collision_total": np.array([
                collision_total if collision_total is not None else 0.0
            ], dtype=np.float32),
            "region_mixture": (
                np.asarray(region_mixture, dtype=np.float32)
                if region_mixture is not None
                else self._region_mixture_state.astype(np.float32)
            ),
            "coverage_center": (
                np.asarray(coverage_center, dtype=np.float32)
                if coverage_center is not None
                else self._footprint_center.astype(np.float32)
            ),
            "coverage_phase": np.array([
                coverage_phase if coverage_phase is not None else self._coverage_phase
            ], dtype=np.float32),
            "pending_backoff_cbra": np.array([
                pending_backoff_cbra
                if pending_backoff_cbra is not None
                else float(self._backoff_queue_cbra.sum())
            ], dtype=np.float32),
            "pending_backoff_pbra": np.array([
                pending_backoff_pbra
                if pending_backoff_pbra is not None
                else float(self._backoff_queue_pbra.sum())
            ], dtype=np.float32),
        }
        return observation

    def _update_history(self, value: float) -> None:
        """滚动更新历史统计。"""

        self._history = np.roll(self._history, shift=-1)
        self._history[-1] = value

    def _apply_preamble_adjustment(self, *, cbra_adj: float, pbra_adj: float) -> Tuple[float, float, float]:
        """根据动作调整 CBRA/PBRA/CFRA 资源分配。"""

        deltas = np.array([round(cbra_adj), round(pbra_adj), 0], dtype=np.float32)
        updated = self._current_preambles.copy()
        updated[0] = np.clip(updated[0] + deltas[0], 0, self.config.total_preambles)
        updated[1] = np.clip(updated[1] + deltas[1], 0, self.config.total_preambles)

        overflow = updated[0] + updated[1] - self.config.total_preambles
        if overflow > 0:
            reduce_pbra = min(overflow, updated[1])
            updated[1] -= reduce_pbra
            overflow -= reduce_pbra
            if overflow > 0:
                updated[0] = max(updated[0] - overflow, 0)

        updated[2] = self.config.total_preambles - updated[0] - updated[1]

        if updated[2] < 0:
            deficit = -updated[2]
            updated[2] = 0
            if updated[1] > updated[0]:
                updated[1] = max(updated[1] - deficit, 0)
            else:
                updated[0] = max(updated[0] - deficit, 0)
        updated[2] = self.config.total_preambles - updated[0] - updated[1]

        self._current_preambles = updated
        return float(updated[0]), float(updated[1]), float(updated[2])

    def _update_region_mixture(self, num_slots: int) -> np.ndarray:
        """更新覆盖权重向量，实现区域的平滑过渡。"""

        if not self._coverage_patches:
            if self._current_region_idx is None:
                idx = int(self._rng.choice(self._region_count, p=self._region_prob))
            else:
                transition = self._transition_matrix[self._current_region_idx]
                idx = int(self._rng.choice(self._region_count, p=transition))
            mixture = np.zeros(self._region_count, dtype=np.float32)
            mixture[idx] = 1.0
            self._region_mixture_state = mixture
            self._current_region_idx = idx
            return self._region_mixture_state

        cycle = max(float(self.config.coverage_cycle_slots), 1e-6)
        self._coverage_phase = (self._coverage_phase + num_slots / cycle) % 1.0
        angle = 2.0 * np.pi * self._coverage_phase

        target_center = np.array(
            [
                self.config.coverage_drift_radius * np.cos(angle),
                self.config.coverage_drift_radius * np.sin(angle),
            ],
            dtype=np.float32,
        )

        if self.config.coverage_jitter > 0.0:
            jitter = self._rng.normal(0.0, self.config.coverage_jitter, size=2).astype(np.float32)
            target_center += jitter

        smoothing = float(self.config.coverage_smoothing)
        if smoothing > 0.0:
            target_center = (1.0 - smoothing) * target_center + smoothing * self._footprint_center
        self._footprint_center = target_center.astype(np.float32)

        weights = self._coverage_reference * self._region_prob.astype(np.float64)
        center_vec = self._footprint_center.astype(np.float64)
        for patch in self._coverage_patches:
            patch_center = np.asarray(patch.center, dtype=np.float64)
            diff = center_vec - patch_center
            dist = float(np.linalg.norm(diff))
            influence = patch.strength * np.exp(-0.5 * (dist / patch.radius) ** 2)
            if influence <= 0.0:
                continue
            weights += influence * np.asarray(patch.region_weights, dtype=np.float64)

        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            weights = self._region_prob.astype(np.float64)
            total = float(weights.sum())
        if total <= 0.0:
            weights = np.full(self._region_count, 1.0 / max(self._region_count, 1), dtype=np.float64)
        else:
            weights /= total

        if smoothing > 0.0:
            weights = (1.0 - smoothing) * weights + smoothing * self._region_mixture_state.astype(np.float64)
            weights = np.clip(weights, 1e-8, None)
            weights /= float(weights.sum())

        self._region_mixture_state = weights.astype(np.float32)
        return self._region_mixture_state

    def _current_preamble_ratio(self) -> np.ndarray:
        return self._current_preambles / max(self.config.total_preambles, 1)

    def _pop_backoff_queue(self, queue: np.ndarray, steps: int) -> float:
        """释放在给定步数内到期的退避重试。"""

        if steps <= 0:
            return 0.0
        max_index = len(queue) - 1
        steps = int(min(max(steps, 0), max_index))
        slice_end = steps + 1
        due = float(np.sum(queue[:slice_end]))
        if slice_end >= len(queue):
            queue.fill(0.0)
        else:
            remaining = queue[slice_end:]
            queue[: len(remaining)] = remaining
            queue[len(remaining) :] = 0.0
        return due

    def _schedule_backoff(self, queue: np.ndarray, amount: float, window: BackoffWindow) -> None:
        """按照退避窗口为碰撞的终端重新排队。"""

        if amount <= 0.0:
            return
        min_step = max(int(window.min_steps), 0)
        max_step = max(int(window.max_steps), min_step)
        max_index = len(queue) - 1
        min_step = min(min_step, max_index)
        max_step = min(max_step, max_index)
        if max_step <= min_step:
            queue[min_step] += float(amount)
            return
        bins = max_step - min_step + 1
        weights = self._rng.dirichlet(np.ones(bins, dtype=np.float64))
        for offset, weight in enumerate(weights):
            step = min_step + offset
            queue[step] += float(amount * weight)

    def _select_backoff_window(self, collision_ratio: float) -> BackoffWindow:
        """根据碰撞率选择退避窗口。"""

        strategy = self._backoff_strategy
        if collision_ratio >= strategy.collision_threshold_high:
            return strategy.high
        if collision_ratio >= strategy.collision_threshold_medium:
            return strategy.medium
        return strategy.low

    def _apply_slot_buffer(
        self,
        *,
        slot_requests: np.ndarray,
        capacity: float,
        queue: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """按照 slot 粒度计算成功与碰撞数量，并在 slot 级别安排退避。"""

        if capacity <= 0.0:
            total_requests = float(np.sum(slot_requests))
            if queue is not None and total_requests > 0.0:
                window = self._select_backoff_window(1.0)
                self._schedule_backoff(queue, total_requests, window)
            return 0.0, total_requests

        slot_requests = np.asarray(slot_requests, dtype=np.float64)
        success_total = 0.0
        collision_total = 0.0
        eps = 1e-6
        for slot_value in slot_requests:
            slot_value = max(float(slot_value), 0.0)
            slot_success, slot_collision = self._single_slot_competition(
                slot_requests=slot_value,
                capacity=capacity,
            )
            success_total += slot_success
            collision_total += slot_collision
            if queue is not None and slot_collision > 0.0:
                slot_ratio = slot_collision / max(slot_value, eps)
                window = self._select_backoff_window(slot_ratio)
                self._schedule_backoff(queue, float(slot_collision), window)
        return success_total, collision_total

    @staticmethod
    def _single_slot_competition(slot_requests: float, capacity: float) -> Tuple[float, float]:
        """根据单个 slot 的接入请求与每个 preamble 的平均负载计算成功/碰撞。"""

        if slot_requests <= 0.0 or capacity <= 0.0:
            return 0.0, max(slot_requests, 0.0)

        load_per_preamble = slot_requests / capacity
        if load_per_preamble <= 1.0:
            return slot_requests, 0.0
        if load_per_preamble >= 2.0:
            return 0.0, slot_requests

        success = capacity * (2.0 - load_per_preamble)
        success = max(min(success, slot_requests), 0.0)
        collision = slot_requests - success
        return success, collision


def default_simulator_config() -> MACSimulatorConfig:
    """提供一个默认的区域配置，用于占位测试。"""

    return MACSimulatorConfig(
        regions=(
            RegionTrafficProfile(
                name="urban",
                area_weight=0.4,
                cbra_density=1.2,
                pbra_density=1.5,
                handover_intensity=0.8,
                scheduling_period=160,
                noise_scale=0.2,
                cox_intensity_mean=1.6,
                cox_intensity_variance=0.7,
                cox_grid=(10, 10),
                region_bounds=(-2.0, 2.0, -2.0, 2.0),
                batch_mean=60.0,
                batch_std=0.6,
                residence_time_mean=20,
            ),
            RegionTrafficProfile(
                name="suburban",
                area_weight=0.35,
                cbra_density=0.8,
                pbra_density=0.9,
                handover_intensity=0.5,
                scheduling_period=200,
                noise_scale=0.15,
                cox_intensity_mean=1.2,
                cox_intensity_variance=0.5,
                cox_grid=(8, 8),
                region_bounds=(-1.5, 1.5, -1.5, 1.5),
                batch_mean=8.0,
                batch_std=0.5,
                residence_time_mean=6.0,
            ),
            RegionTrafficProfile(
                name="rural",
                area_weight=0.25,
                cbra_density=0.3,
                pbra_density=0.4,
                handover_intensity=0.2,
                scheduling_period=320,
                noise_scale=0.1,
                cox_intensity_mean=0.9,
                cox_intensity_variance=0.3,
                cox_grid=(6, 6),
                region_bounds=(-1.0, 1.0, -1.0, 1.0),
                batch_mean=4.0,
                batch_std=0.4,
                residence_time_mean=4.5,
            ),
        ),
    )
