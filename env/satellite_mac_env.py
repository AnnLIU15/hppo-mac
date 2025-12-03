"""自定义卫星 MAC 强化学习环境实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
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
    simulator_config: Optional[MACSimulatorConfig] = None
    preamble_delta_range: int = 1
    flatten_observation: bool = True


class SatelliteMACEnv(gym.Env):
    """面向 RL 训练的卫星 MAC 环境。"""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[SatelliteMACEnvConfig] = None, **kwargs) -> None:
        # TorchRL 的 GymEnv 会通过 make_kwargs 传递配置，这里兼容该行为。
        make_kwargs = kwargs.pop("make_kwargs", None)
        print(config)
        config = kwargs.pop("config", config)
        if make_kwargs:
            config = make_kwargs.get("config", config)

        self.config = config or SatelliteMACEnvConfig()
        print("config",self.config.preamble_delta_range)
        sim_config = self.config.simulator_config or default_simulator_config()
        self.simulator = MACSimulator(sim_config)
        self._region_count = len(self.simulator.config.regions)
        self._delta_range = max(1, int(self.config.preamble_delta_range))
        self._delta_values = np.arange(-self._delta_range, self._delta_range + 1, dtype=np.int64)
        self._delta_bins = int(self._delta_values.size)
        self._flat_observation = bool(self.config.flatten_observation)

        delta_space = spaces.Discrete(self._delta_bins)
        q_acb_box = spaces.Box(low=np.zeros((1,), dtype=np.float32), high=np.ones((1,), dtype=np.float32), dtype=np.float32)

        self.action_space = spaces.Dict(
            {
                "delta_cbra": delta_space,
                "delta_pbra": delta_space,
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
        observation = self.simulator.initialize_state(seed)
        observation = self._format_observation(observation)
        info = {
            "step": np.array(self._step_count, dtype=np.float32),
            "action_valid": np.array(1.0, dtype=np.float32),
        }
        return observation, info

    def step(
        self,
        action: Dict[str, np.ndarray],
        need_parse_action: bool = True,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, bool, Dict[str, float]]:
        """执行一步决策。

        Args:
            action: 动作字典，包含 delta_cbra, delta_pbra, q_ACB
            need_parse_action: 是否需要解析动作
                - True: 用于 PPO 等 RL 算法，需要从离散索引映射到增量值
                - False: 用于启发式算法，直接传入数值
        """
        if need_parse_action:
            parsed_action = self._parse_action(action)
        else:
            # 启发式算法直接传入数值，不需要索引映射
            parsed_action = {
                "M_CBRA": float(action["delta_cbra"].item()),
                "M_PBRA": float(action["delta_pbra"].item()),
                "q_ACB": float(action["q_ACB"].item()),
            }

        reward, next_state, sim_info = self.simulator.run_slots(
            num_slots=self.config.num_slots_per_step,
            params=parsed_action,
        )

        self._step_count += 1
        truncated = self._step_count >= self.config.decision_horizon
        terminated = False

        observation = self._format_observation(next_state)

        info = {
            "step": np.array(self._step_count, dtype=np.float32),
            "action_valid": np.array(1.0, dtype=np.float32),
        }
        for key, value in sim_info.items():
            if isinstance(value, (float, int)):
                info[key] = np.array(value, dtype=np.float32)
            elif isinstance(value, str):
                # 字符串类型跳过，TorchRL不支持
                continue
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    info[key] = np.array(float(value.item()), dtype=np.float32)
                else:
                    info[key] = value.astype(np.float32)

        return observation, float(reward), terminated, truncated, info

    def render(self):
        raise NotImplementedError("当前环境未实现渲染。")

    def _parse_action(self, action: Dict[str, np.ndarray]) -> Dict[str, float]:
        """将策略输出的动作转换为仿真器可识别的格式。"""

        try:
            delta_cbra = self._decode_delta_component(action["delta_cbra"])
            delta_pbra = self._decode_delta_component(action["delta_pbra"])
            q_acb_raw = np.asarray(action["q_ACB"], dtype=np.float32).reshape(-1)
        except KeyError as err:
            raise ValueError("缺少必要的动作分量。") from err

        q_acb = float(np.clip(q_acb_raw[0] if q_acb_raw.size else 0.0, 0.0, 1.0))

        return {
            "M_CBRA": float(delta_cbra),
            "M_PBRA": float(delta_pbra),
            "q_ACB": float(q_acb),
        }

    def _decode_delta_component(self, raw_component: Union[int, float, np.ndarray]) -> float:
        """解析离散动作分量为整数增量。"""

        values = np.asarray(raw_component).astype(np.float32).reshape(-1)
        if values.size == self._delta_bins:
            idx = int(np.argmax(values))
        elif values.size == 1:
            idx = int(values.item())
        else:
            raise ValueError("无法解析给定的离散动作分量。")
        idx = int(np.clip(idx, 0, self._delta_bins - 1))
        return float(self._delta_values[idx])

    def encode_delta_index(self, delta: int) -> int:
        """将给定增量编码为离散动作索引。"""

        clipped = int(np.clip(delta, -self._delta_range, self._delta_range))
        return clipped + self._delta_range

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

        flat = np.concatenate(vectors).astype(np.float32)
        return flat
