"""自定义卫星 MAC 强化学习环境实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from loguru import logger
from .mac_simulator import (
    MACSimulator,
    MACSimulatorConfig,
    MAC_PROTOCOL_COUNT,
    PREAMBLE_SUBSET_COUNT,
    HISTORY_DIM,
    default_simulator_config,
)


@dataclass(frozen=True)
class SatelliteMACEnvConfig:
    """环境配置对象。"""

    num_slots_per_step: int = 160
    decision_horizon: int = 1
    history_len: int = 4
    simulator_config: Optional[MACSimulatorConfig] = None
    preamble_delta_range: int = 1
    flatten_observation: bool = True


class SatelliteMACEnv(gym.Env):
    """面向 PPO 训练的卫星 MAC 环境。"""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[SatelliteMACEnvConfig] = None, **kwargs) -> None:
        # TorchRL 的 GymEnv 会通过 make_kwargs 传递配置，这里兼容该行为。
        make_kwargs = kwargs.pop("make_kwargs", None)
        config = kwargs.pop("config", config)
        if make_kwargs:
            config = make_kwargs.get("config", config)
        self.config = config or SatelliteMACEnvConfig()
        sim_config = self.config.simulator_config or default_simulator_config()
        self.simulator = MACSimulator(sim_config)
        self._region_count = len(self.simulator.config.regions)
        self._stat_keys = (
            "requests_cbra",
            "requests_pbra",
            "collision_ratio_cbra",
            "collision_ratio_pbra",
            "pending_backoff_cbra",
            "pending_backoff_pbra",
        )
        self.history_len = max(1, self.config.history_len)
        self._recent_stats: List[np.ndarray] = []
        self._delta_range = max(1, int(self.config.preamble_delta_range))
        self._delta_values = np.arange(-self._delta_range, self._delta_range + 1, dtype=np.int64)
        self._delta_bins = int(self._delta_values.size)
        self._delta_pairs = [
            (int(delta_cbra), int(delta_pbra))
            for delta_cbra in self._delta_values
            for delta_pbra in self._delta_values
        ]
        self._combo_count = len(self._delta_pairs)
        self._flat_observation = bool(self.config.flatten_observation)

        combo_space = spaces.Discrete(self._combo_count)
        q_acb_box = spaces.Box(low=np.zeros((1,), dtype=np.float32), high=np.ones((1,), dtype=np.float32), dtype=np.float32)

        self.action_space = spaces.Dict(
            {
                "delta_combo": combo_space,
                "q_ACB": q_acb_box,
            }
        )

        if self._flat_observation:
            self._obs_low, self._obs_high = self._build_observation_bounds()
            self.observation_space = spaces.Box(
                low=self._obs_low,
                high=self._obs_high,
                dtype=np.float32,
            )
        else:
            inf = np.finfo(np.float32).max
            self.observation_space = spaces.Dict(
                {
                    "requests_cbra": spaces.Box(low=0.0, high=inf, shape=(1,), dtype=np.float32),
                    "requests_pbra": spaces.Box(low=0.0, high=inf, shape=(1,), dtype=np.float32),
                    "collision_ratio_cbra": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "collision_ratio_pbra": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "active_terminals_dist": spaces.Box(
                        low=0.0, high=1.0, shape=(MAC_PROTOCOL_COUNT,), dtype=np.float32
                    ),
                    "preamble_usage": spaces.Box(
                        low=0.0, high=1.0, shape=(PREAMBLE_SUBSET_COUNT,), dtype=np.float32
                    ),
                    "preamble_allocation": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                    "current_ACB_factor": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "history_stats": spaces.Box(low=-np.inf, high=np.inf, shape=(HISTORY_DIM,), dtype=np.float32),
                    "pending_backoff_cbra": spaces.Box(low=0.0, high=inf, shape=(1,), dtype=np.float32),
                    "pending_backoff_pbra": spaces.Box(low=0.0, high=inf, shape=(1,), dtype=np.float32),
                    "region_mixture": spaces.Box(low=0.0, high=1.0, shape=(self._region_count,), dtype=np.float32),
                    "coverage_center": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                    "coverage_phase": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "recent_stats": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.history_len, len(self._stat_keys)),
                        dtype=np.float32,
                    ),
                }
            )

        self._np_random, _ = seeding.np_random()
        self._step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, float]] = None,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Dict[str, float]]:
        """重置环境。"""

        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = seeding.np_random(seed)
        self.simulator.reseed(seed)
        self.simulator.reset_access_allocation()
        self.simulator._reset_coverage_position()
        self._step_count = 0
        observation_dict = self.simulator.initialize_state(seed)
        self._init_history_buffer(observation_dict)
        observation_dict["recent_stats"] = self._recent_stats_array()
        observation = self._format_observation(observation_dict)
        combo_mask = self._delta_combo_mask()
        info = {
            "step": self._step_count,
            "delta_combo_mask": combo_mask,
        }
        return observation, info

    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, bool, Dict[str, float]]:
        """执行一步决策。"""

        parsed_action = self._parse_action(action)
        # logger.info(f'parsed_action {parsed_action}')
        reward, next_state, sim_info = self.simulator.run_slots(
            num_slots=self.config.num_slots_per_step,
            params=parsed_action,
        )

        self._step_count += 1
        truncated = self._step_count >= self.config.decision_horizon
        terminated = False

        self._update_history_buffer(next_state)
        next_state["recent_stats"] = self._recent_stats_array()

        observation = self._format_observation(next_state)

        combo_mask = self._delta_combo_mask()
        info = {
            "step": self._step_count,
            "delta_combo_mask": combo_mask,
        }
        for key, value in sim_info.items():
            if isinstance(value, (float, int, str)):
                info[key] = value
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    info[key] = float(value.item())
                else:
                    info[key] = value.astype(np.float32)

        return observation, float(reward), terminated, truncated, info

    def render(self):
        raise NotImplementedError("当前环境未实现渲染。")

    def _parse_action(self, action: Dict[str, np.ndarray]) -> Dict[str, float]:
        """将策略输出的动作转换为仿真器可识别的格式。"""

        try:
            delta_cbra, delta_pbra = self._decode_combo_component(action["delta_combo"])
            q_acb_raw = np.asarray(action["q_ACB"], dtype=np.float32).reshape(-1)
        except KeyError as err:
            raise ValueError("缺少必要的动作分量。") from err

        q_acb = float(np.clip(q_acb_raw[0] if q_acb_raw.size else 0.0, 0.0, 1.0))

        return {
            "M_CBRA": float(delta_cbra),
            "M_PBRA": float(delta_pbra),
            "q_ACB": float(q_acb),
        }

    def _decode_combo_component(self, raw_component: Union[int, float, np.ndarray]) -> Tuple[float, float]:
        """将离散组合动作索引或独热向量映射为 (delta_cbra, delta_pbra)。"""

        values = np.asarray(raw_component).astype(np.float32).reshape(-1)
        if values.size == self._combo_count:
            idx = int(np.argmax(values))
        elif values.size == 1:
            idx = int(values.item())
        else:
            raise ValueError("无法解析给定的组合动作分量。")
        idx = int(np.clip(idx, 0, self._combo_count - 1))
        delta_cbra, delta_pbra = self._delta_pairs[idx]
        return float(delta_cbra), float(delta_pbra)

    def encode_delta_combo(self, delta_cbra: int, delta_pbra: int) -> int:
        """将给定的增量对编码为组合动作索引（便于脚本与基线策略使用）。"""

        clipped_cbra = int(np.clip(delta_cbra, -self._delta_range, self._delta_range))
        clipped_pbra = int(np.clip(delta_pbra, -self._delta_range, self._delta_range))
        cbra_idx = clipped_cbra + self._delta_range
        pbra_idx = clipped_pbra + self._delta_range
        return cbra_idx * self._delta_bins + pbra_idx

    def _delta_combo_mask(self) -> np.ndarray:
        """获取当前时刻可行的 delta 组合掩码。"""

        mask = self.simulator.compute_combo_mask(self._delta_pairs)
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        return mask

    def configure_access_state(
        self,
        *,
        cbra: Optional[int] = None,
        pbra: Optional[int] = None,
        cfra: Optional[int] = None,
    ) -> None:
        """外部接口：强制设置初始 RA 资源。"""

        self.simulator.configure_access_state(
            cbra=cbra,
            pbra=pbra,
            cfra=cfra,
        )

    def _init_history_buffer(self, observation: Dict[str, np.ndarray]) -> None:
        stats_vector = self._extract_stats(observation)
        self._recent_stats = [stats_vector.copy() for _ in range(self.history_len)]

    def _update_history_buffer(self, observation: Dict[str, np.ndarray]) -> None:
        stats_vector = self._extract_stats(observation)
        self._recent_stats.append(stats_vector)
        if len(self._recent_stats) > self.history_len:
            self._recent_stats.pop(0)

    def _recent_stats_array(self) -> np.ndarray:
        if not self._recent_stats:
            return np.zeros((self.history_len, len(self._stat_keys)), dtype=np.float32)
        return np.stack(self._recent_stats, axis=0).astype(np.float32)

    def _extract_stats(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        values = []
        for key in self._stat_keys:
            raw = np.asarray(observation[key], dtype=np.float32).reshape(-1)
            values.append(float(raw[0]))
        return np.asarray(values, dtype=np.float32)

    def _build_observation_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        inf = np.finfo(np.float32).max
        neg_inf = -np.finfo(np.float32).max
        components: List[Tuple[str, int, float, float]] = []

        def add_component(name: str, size: int, low: float, high: float) -> None:
            components.append((name, size, low, high))

        add_component("requests_cbra", 1, 0.0, inf)
        add_component("requests_pbra", 1, 0.0, inf)
        add_component("collision_ratio_cbra", 1, 0.0, 1.0)
        add_component("collision_ratio_pbra", 1, 0.0, 1.0)
        add_component("active_terminals_dist", MAC_PROTOCOL_COUNT, 0.0, 1.0)
        add_component("preamble_usage", PREAMBLE_SUBSET_COUNT, 0.0, 1.0)
        add_component("preamble_allocation", 3, 0.0, 1.0)
        add_component("current_ACB_factor", 1, 0.0, 1.0)
        add_component("history_stats", HISTORY_DIM, -np.inf, np.inf)
        add_component("pending_backoff_cbra", 1, 0.0, inf)
        add_component("pending_backoff_pbra", 1, 0.0, inf)
        add_component("region_mixture", self._region_count, 0.0, 1.0)
        add_component("coverage_center", 2, neg_inf, inf)
        add_component("coverage_phase", 1, 0.0, 1.0)

        stat_bounds = {
            "requests_cbra": (0.0, inf),
            "requests_pbra": (0.0, inf),
            "collision_ratio_cbra": (0.0, 1.0),
            "collision_ratio_pbra": (0.0, 1.0),
            "pending_backoff_cbra": (0.0, inf),
            "pending_backoff_pbra": (0.0, inf),
        }
        for _ in range(self.history_len):
            for key in self._stat_keys:
                low, high = stat_bounds[key]
                add_component(f"recent_stats::{key}", 1, low, high)

        self._obs_components = components

        lows = []
        highs = []
        for _, size, low, high in components:
            lows.append(np.full((size,), low, dtype=np.float32))
            highs.append(np.full((size,), high, dtype=np.float32))

        low_arr = np.concatenate(lows).astype(np.float32)
        high_arr = np.concatenate(highs).astype(np.float32)
        return low_arr, high_arr

    def _format_observation(self, observation: Dict[str, np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if not self._flat_observation:
            return observation
        vectors: List[np.ndarray] = []
        vectors.append(np.asarray(observation["requests_cbra"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["requests_pbra"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["collision_ratio_cbra"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["collision_ratio_pbra"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["active_terminals_dist"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["preamble_usage"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["preamble_allocation"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["current_ACB_factor"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["history_stats"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["pending_backoff_cbra"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["pending_backoff_pbra"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["region_mixture"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["coverage_center"], dtype=np.float32).reshape(-1))
        vectors.append(np.asarray(observation["coverage_phase"], dtype=np.float32).reshape(-1))

        stats = np.asarray(observation["recent_stats"], dtype=np.float32)
        vectors.append(stats.reshape(-1))

        flat = np.concatenate(vectors).astype(np.float32)
        return flat
