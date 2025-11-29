H-PPO MAC Environment
=====================

该项目提供了一个面向强化学习的卫星随机接入 MAC 仿真环境、启发式控制脚本以及指标分析工具，重点演示以下能力：

- 区域混合覆盖模型：在 `env/mac_simulator.py` 中实现多个 `CoveragePatch`，结合轨迹相位漂移、随机抖动与平滑插值，生成平滑的区域权重。
- 可配置的 ACB 随机退避：通过 `BackoffStrategyConfig` 实现基于碰撞率的退避窗口选择，并维护 CBRA/PBRA 重试队列。
- Gymnasium 环境封装：`env/satellite_mac_env.py` 提供 `SatelliteMACEnv`，支持扁平/字典观测、历史统计堆叠与预设动作离散化。
- 启发式仿真与参数扫查：`main.py` 内置启发式策略、批量仿真、退避参数扫查以及自动化微调。
- 结果可视化：`scripts/plot_telemetry.py` 使用 matplotlib + SciencePlots，绘制退避队列、吞吐与控制信号曲线。

项目结构
--------

```
env/
    mac_simulator.py      # 核心仿真器，含退避、覆盖、流量生成实现
    satellite_mac_env.py  # Gymnasium 环境封装
algo/
    hppo.py               # 预留强化学习算法入口（占位）
utils/
    traffic.py            # 流量采样辅助函数
main.py                  # 启发式控制、扫参、日志与分析主脚本
scripts/plot_telemetry.py# 使用 SciencePlots 绘制指标
tests/                   # 单元测试（区域权重、退避逻辑）
logs/                    # 自动生成的仿真输出、扫参结果、telemetry
```

环境要求与安装
------------

项目使用 `Python >= 3.13`，依赖列表见 `pyproject.toml`。推荐步骤：

1. 准备虚拟环境（例如 `uv`, `venv`, `conda`）。
2. 安装依赖：
   ```bash
   pip install -e .
   ```
3. 如需绘图，请确认 `matplotlib`、`scienceplots`、`numpy` 已安装。

核心组件概述
------------

### MAC 仿真器 `MACSimulator`

- 支持 `RegionTrafficProfile` 与 `CoveragePatch` 配置，模拟覆盖混合与流量聚集。
- `BackoffStrategyConfig` 允许调节碰撞阈值、退避窗口与最大排队长度。
- `run_slots` 返回奖励、观测字典和详细 `info`（各区域请求/成功/退避队列等）。

### Gymnasium 环境 `SatelliteMACEnv`

- 自动构建动作空间：`delta_cbra` / `delta_pbra`（独立离散分支）与 `q_ACB` （连续）。
- 观测包含当前退避累积、区域混合、历史奖励序列等；可生成扁平向量或保持字典形式。
- `configure_access_state` 提供外部调节入口，便于单元测试或策略初始化。

### 启发式仿真 `main.py`

功能概览：

- 运行多回合启发式策略，依据碰撞率/退避队列动态调整前导序列与 ACB。
- 保存每一步状态至 `StepTrace`，压缩存档为 `.npz`，输出于 `logs/telemetry_ep*.npz`。
- 自动生成 `logs/telemetry_summary.csv`，统计每个回合的奖励、碰撞、退避峰值以及时间位置。
- 执行两轮参数扫查：
  1. **Coarse Sweep**：遍历退避阈值、窗口、动作离散范围，记录碰撞与积压指标。
  2. **Fine Sweep**：围绕最佳配置局部搜索，并将结果写入 `logs/sweep_*/fine/sweep_summary.csv`。

运行指南
--------

1. **执行启发式仿真、生成遥测与扫参**
   ```bash
   python main.py
   ```
   输出包括：
   - 控制台的回合指标、最佳参数组合。
   - `logs/telemetry_ep*.npz`：逐步遥测（奖励、吞吐、退避队列、ACB 等）。
   - `logs/telemetry_summary.csv`：每个回合的统计摘要。
   - `logs/sweep_*/sweep_summary.csv`：粗扫结果；`logs/sweep_*/fine/sweep_summary.csv`：细扫结果。

2. **运行 H-PPO 训练（TorchRL）**
    ```bash
    python train_hppo.py --frames-per-batch 512 --rollout-epochs 2 --max-iterations 4 --num-envs 4 --parallel-collection
    ```
    默认配置使用混合离散/连续动作空间，可通过 `--num-envs` 配置并行环境数量、`--env-backend {serial,parallel}` 选择矢量环境后端，或直接加上 `--parallel-collection` 快捷键启用 `ParallelEnv` 多进程采样。按需调节批量大小、决策地平线、特征维度或设备（如 `--device cuda:0`）。脚本会使用 Loguru 将每轮指标写入 `logs/hppo_training.log`，并将训练完成后的 actor/critic 权重保存至 `logs/checkpoints/`（参数 `--log-file`、`--checkpoint-dir` 支持自定义路径或关闭）。

3. **绘制遥测曲线**
   ```bash
   python scripts/plot_telemetry.py logs/telemetry_ep*.npz --output logs/telemetry_plot.png
   ```
   可选参数：
   - `--show`：绘制后展示窗口。
   - `--dpi`：指定保存分辨率。

4. **单元测试**
   ```bash
   python -m unittest tests.test_region_mixture tests.test_backoff_strategy
   ```

关键日志产物
------------

- `telemetry_ep*.npz`：包含字段 `step`、`reward`、`throughput`、`collisions`、`backlog_cbra`、`backlog_pbra`、`acb`、`collision_ratio_*` 等。
- `telemetry_summary.csv`：整合每个回合的平均/最大指标，以及退避峰值发生步。
- `sweep_summary.csv`：记录参数组合与对应的奖励、吞吐、碰撞、积压。

进一步扩展
----------

- 强化学习：`SatelliteMACEnv` 与日志体系已准备好与 PPO / TorchRL 集成，可在 `algo/` 内添加训练脚本。
- 数据分析：遥测 `.npz` 与 CSV 文件可用于外部数据工具（pandas、Jupyter）深入分析。
- 配置扩展：
  - 在 `default.yaml` 或自定义配置中加入更多区域/覆盖参数；
  - 扩充 `BackoffStrategyConfig` 以支持指数退避、差异化终端分类等。

问题排查
--------

- **遥测为空**：确认 `main.py` 成功运行且生成 `.npz` 文件；脚本会自动跳过缺失文件。
- **SciencePlots 未安装**：确保 `pip install scienceplots` 或使用 `pip install -e .`。
- **数值异常（退避为负）**：环境在仿真内部做了断言，如触发 `AssertionError`，请检查自定义配置是否违反最大步数约束。

版权与许可
----------

项目用于科研实验，默认遵循内部使用条款。如需对外发布，请联系维护者确认许可与引用方式。H-PPO 训练管线与环境实现计划
本计划分为五个主要阶段：环境与接口定义、数据规范化、网络架构构建、PPO 核心模块配置和主训练循环。

阶段 1：仿真环境构建与接口定义 (S22)
您需要构建一个继承自 gymnasium.Env 的自定义类 SatelliteMACEnv，以适配强化学习框架。

1.1. 环境骨架 (SatelliteMACEnv)
环境必须包含以下核心方法：__init__、reset 和 step。

LEO卫星通常运行在 2,000 公里以下的轨道区域 ，其轨道速度极高，大约为 7.5 km/s 。这种高速运动导致卫星每 90 到 110 分钟即可完成一次绕地球轨道运行 。对于地面用户终端（UT）而言，单个卫星提供的通信窗口（即服务持续时间）非常短暂，通常持续 5 到 15 分钟，且一天内可能仅出现 6 到 8 次 。

空间分布用户位置密度 ($\rho$)Cox点过程捕获地理聚类和非均匀性2业务需求会话到达率 ($\lambda$)批次泊松过程建模容量需求（爱尔朗）和状态转换

准确模拟用户进入和容量需求需要对流量进行现实的、非均匀的空间和时间建模。2.1 地面终端的空间分布模型在系统级分析中，均匀卫星分布的简化空间模型有助于进行覆盖和容量的初步分析 9。然而，这种均匀模型无法捕捉用户或设备位置的真实空间异构性，尤其在人口密度差异巨大的地理区域 9。为了进行高保真仿真，捕捉用户的聚类和非均匀性是至关重要的。推荐采用以下高级随机几何模型：泊松点过程（PPP）： 假定用户随机散布，但在某一区域内的密度 $\lambda$ 是均匀的。Cox点过程（双随机过程）： 这是模拟空间异构性的首选模型 2。在Cox过程中，底层密度 ($\rho$) 本身被视为一个随机变量，从而能够现实地模拟高密度城市集群（高 $\rho$）和稀疏农村地区（低 $\rho$）2。这种模型需要输入反映实际人口分布的集群参数和强度测度 2。选择Cox点过程模型并非仅是提升精度，更具有架构层面的意义：异构的密度分布直接要求系统采用灵活的资源管理策略 2。高度利用的波束（服务于密集集群）需要动态容量分配机制（如基于速率的动态容量分配 RBDC 或基于容量的 VBDC 11），这直接将空间建模（第 2 节）与波束管理及容量约束（第 4 节）联系起来。在实际仿真中，这种非均匀分布通过将地理区域定义为“区域目标”并为其分配特定的流量强度（爱尔朗）来实现，作为卫星流量模拟器（STS）的输入 12。2.2 业务和会话到达建模流量需求必须以爱尔朗（Erlangs）为单位进行刻画，并根据地理定义的“区域目标”按小时累计 12。对于会话的到达过程，新呼叫或数据会话通常被建模为参数为 $\lambda_u$（到达率）的泊松过程 8。这意味着在调度周期（时隙）内，用户 $u$ 的数据到达量 $A_u(t)$ 服从参数为 $\lambda_u \tau$（其中 $\tau$ 为时隙长度）的泊松分布 8。在进行高密度区域的容量分析时，尤其是涉及大型容量单元的马尔可夫链计算时，经常使用批次泊松过程（Batched Poisson Process）13。在该模型中，新呼叫以批次 $s_k$ 的大小到达，其中 $s_k$ 通常服从几何分布 13。
3. 用户驻留时间与接入/离开的解析建模LEO卫星的高移动性决定了卫星可见性的几何约束主导了连接持续时间，这是模拟用户“进入”和“退出”的基础时间约束。3.1 驻留时间特征刻画 ($T_D$)定义与解析： 驻留时间 ($T_D$) 或卫星通过持续时间 ($T_{Pass}$)，是指用户终端在最小仰角要求以上保持对特定卫星视距连接的总时间 5。由于其统计分布反映了实际星座部署，因此需要采用随机几何工具对其进行解析推导，将卫星位置视为球形包裹均匀点过程 5。$T_{Pass}$ 的统计分布对于预测切换间隔至关重要 5。分析结果表明，平均通过持续时间与轨道高度存在简单的依赖关系，并且能够很好地近似实际的Walker-Delta星座部署 5。与切换的关联： 驻留时间的解析推导是后续切换优化框架的关键输入。在马尔可夫链引导的模拟退火（MCSA）模型中，$T_{Pass}$ 被正式化为偏好评分 $Score_{ij}^t$ 的一个组成部分——可见性时间 ($RT_{ij}^t$) 3。通过最大化 $RT_{ij}^t$，系统有效地优化了服务的寿命，直接缓解了LEO短暂通过时间的限制。呼叫驻留时间： 当考虑地面用户移动性时，呼叫在波束覆盖区域内的驻留时间 $t_c$ 的建模有所不同：对于发起于该区域的“新呼叫”， $t_c$ 通常在 $[0, L/V_{tr}]$ 之间均匀分布；而对于跨越相邻小区边界的“切换呼叫”，其 $t_c$ 可以近似为一个确定性的值 $T_c = L/V_{tr}$ 14。3.2 用户移动性模型与随机链路动态性对于地面用户终端（TTGs），必须采用移动性模型来反映其位置随时间的变化。**随机游走移动模型（Random Walk Mobility Model）**是无线网络中常用的方法 15。该模型通过简化二维马尔可夫链，降低了复杂性，有助于推导位置更新率和驻留时间等关键指标 15。地面用户的移动（例如随机游走）是导致信道链路随机动态性的重要因素 3。用户位置或路径的突然变化（如进入城市障碍物区域）会导致信道质量快速波动，要求预测性切换策略能够实时响应 3。用户进入与退出概念的界定：LEO环境下的用户动态接入和离开必须被视为一个复杂的过程而非简单的连接建立与断开：用户“进入”（Entry）： 当用户的服务请求成功找到一颗卫星 $j$，并获得了足够高的偏好评分 $Score_{ij}^t$ 以完成分配（即 $x_{ij}=1$）时，即为成功“进入”。在此过程中，必须确保 $RT_{ij}^t$ 足够长以确认连接的可行性。用户“退出”（Exit）（计划性）： 当用户服务在预测的 $T_{Pass}$ 结束时成功完成，或者服务通过成功的切换继续进行，这属于计划性的服务结束或转移。用户“退出”（Exit）（非计划性/失败）： 非计划性退出等同于切换失败或呼叫阻塞事件 16。这发生在切换窗口内，系统未能为下一颗卫星分配必要的资源（如信道或容量 $c_j$），导致链路中断。


Python

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatelliteMACEnv(gym.Env):
    def __init__(self, num_slots=160, mac_config={...}):
        super().__init__()
        self.num_slots = num_slots
        self.current_slot = 0

        # 动作空间：组合离散动作 + 1个连续动作
        # a_t = (a_{t, combo}, a_{t, c})
        self.action_space = spaces.Dict({
            # delta 分支：CBRA/PBRA 各自的增量 (例如 delta_range=1 -> 3 个离散取值)
            "delta_cbra": spaces.Discrete(3),
            "delta_pbra": spaces.Discrete(3),
            # c: ACB 因子 q_ACB (0到1之间的连续值)
            "q_ACB": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        # 观测空间：包含多项统计信息的组合 (状态 s_t)
        # 实际维度应根据您的仿真模型确定
        self.observation_space = spaces.Dict({
            "requests_cbra": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "requests_pbra": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "collision_ratio_cbra": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "collision_ratio_pbra": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "active_terminals_dist": spaces.Box(low=0, high=np.inf, shape=(MAC_PROTOCOL_COUNT,), dtype=np.float32),
            "preamble_usage": spaces.Box(low=0, high=1, shape=(PREAMBLE_SUBSET_COUNT,), dtype=np.float32),
            "current_ACB_factor": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # 历史信息/系统状态
            "history_stats": spaces.Box(low=-np.inf, high=np.inf, shape=(HISTORY_DIM,), dtype=np.float32)
        })
        # 初始化 MAC/信道/终端模型 (省略细节)
        self.mac_model = MACSimulator(mac_config)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_slot = 0
        initial_state = self.mac_model.initialize_state(seed) # 初始化仿真环境的状态
        return initial_state, {} # 返回初始状态

    def step(self, action: dict):
        # S23：执行动作，状态转移，反馈观测与奖励

    # 1. 解析动作 a_t
    delta_cbra_idx = action["delta_cbra"]  # 离散动作索引 (0 .. bins-1)
    delta_pbra_idx = action["delta_pbra"]
    q_ACB = action["q_ACB"]              # 连续动作 (0.0 to 1.0)

    # 将离散索引映射为实际增量（例如：-1, 0, 1）
    delta_range = 1
    bins = 2 * delta_range + 1
    d1_change = delta_cbra_idx - delta_range
    d2_change = delta_pbra_idx - delta_range

        # 2. 仿真环境执行动作：更新 MAC 参数
        new_MAC_params = {
            "M_CBRA": d1_change,
            "M_PBRA": d2_change,
            "q_ACB": q_ACB
        }

        # 3. 仿真模型运行 160 个接入时隙
        # 仿真环境进行状态转移并反馈新的观测与奖励
        reward, next_state, info = self.mac_model.run_slots(
            num_slots=self.num_slots,
            params=new_MAC_params
        )

        self.current_slot += 1 # 理论上此环境只有一个时间步（一次决策对应一个回合/决策周期）

        # 确定结束条件：如果您的仿真环境是一个定长回合 (T=1)，则 done=True
        terminated = self.current_slot >= 1
        truncated = False # 没有提前截断

        return next_state, reward, terminated, truncated, info
阶段 2：TorchRL 环境管线配置
此阶段使用 TorchRL 的 TransformedEnv 来桥接您的自定义环境与 PPO 模块，并处理状态和动作的格式转换。

2.1. 状态扁平化 (CatTensors)
您的观测空间是 Dict 类型，标准 MLP 策略网络需要一个扁平的张量输入。我们必须使用 CatTensors 转换将其转换为单一的 'observation' 键 。

Python

from torchrl.envs import GymEnv, Compose, CatTensors, DoubleToFloat

# 实例化您的自定义环境
base_env = GymEnv(SatelliteMACEnv, device="cpu")

# 定义环境转换链
# 1. DoubleToFloat: 确保所有张量为 float32
# 2. CatTensors: 将 Dict 观测空间中的所有键（如 requests_cbra, active_terminals_dist 等）拼接为单一的 'observation' 键
# 3. ObservationNorm (可选): 对状态进行归一化，以稳定训练
env = Compose(
    base_env,
    DoubleToFloat(),
    CatTensors(
        in_keys=list(base_env.observation_space.keys()),
        out_key="observation",
        del_keys=True
    )
    # ObservationNorm() # 如果需要状态归一化，在此处添加
)
阶段 3：H-PPO 网络架构构建 (Actor θ, Value φ)
此阶段严格按照您的要求构建 Actor 网络（多头）和 Value 网络（单头）。Actor 和 Value 网络都将使用一个共享的状态编码网络来提取状态特征。

3.1. 状态编码网络 (State Encoder)
Python

from tensordict.nn import TensorDictModule
from torch import nn

# 状态编码网络 (共享)
# 输入: 'observation' (扁平化的状态张量)
# 输出: 'state_feature' (状态特征向量)
feature_dim = 128 # 状态特征维度
state_encoder = nn.Sequential(
    nn.Linear(env.observation_spec["observation"].shape[-1], 256),
    nn.ReLU(),
    nn.Linear(256, feature_dim),
    nn.ReLU(),
)
# 封装为 TensorDictModule
StateEncoder = TensorDictModule(
    state_encoder,
    in_keys=["observation"],
    out_keys=["state_feature"],
)
3.2. Actor 网络 (θ) — 多头策略
Actor 网络由状态编码网络和三个并行的策略头部组成。

Python

from torchrl.modules import ProbabilisticActor
from tensordict.nn.distributions import CompositeDistribution
from torch.distributions import Categorical, TanhNormal

# 1. 策略头部：从 'state_feature' 输出三个动作所需的参数
actor_heads = nn.ModuleDict({
    # 离散动作 D1: M_CBRA_delta (3个类别: -1, 0, 1) -> 输出 Logits
    "M_CBRA_delta": nn.Linear(feature_dim, 3),
    # 离散动作 D2: M_PBRA_delta (3个类别: -1, 0, 1) -> 输出 Logits
    "M_PBRA_delta": nn.Linear(feature_dim, 3),
    # 连续动作 C1: q_ACB (0到1) -> 输出均值(loc)和对数标准差(scale)
    # TanhNormal 用于将连续动作限制在  范围内 (需适配您的 Box 范围)
    "q_ACB_params": nn.Sequential(
        nn.Linear(feature_dim, 2) # 输出 loc 和 scale (2个参数)
        # 注意: 如果使用 TanhNormal，action_spec 的界限会被自动应用
    )
})

# 2. 策略封装与分布：使用 CompositeDistribution 聚合三个头部的概率
# 定义每个动作组件的分布类型
action_dist_map = {
    "M_CBRA_delta": Categorical,
    "M_PBRA_delta": Categorical,
    # TanhNormal 分布的参数通常是 "loc" 和 "scale"
    "q_ACB": TanhNormal # TanhNormal 确保输出在 Box 空间内 [1, 2]
}

# ProbabilisticActor 结合特征编码器、策略头和 CompositeDistribution
Actor = ProbabilisticActor(
    module=nn.Sequential(StateEncoder, actor_heads),
    in_keys={"M_CBRA_delta": ["state_feature"], "M_PBRA_delta": ["state_feature"], "q_ACB_params": ["state_feature"]},
    # 映射输出键到分布参数
    out_keys={
        "M_CBRA_delta": ["logits"],
        "M_PBRA_delta": ["logits"],
        "q_ACB": ["loc", "scale"] # 连续动作需要 loc, scale
    },
    spec=env.action_spec,
    distribution_class=CompositeDistribution,
    distribution_kwargs={
        "M_CBRA_delta": {"base_dist": Categorical, "temperature": 1.0},
        "M_PBRA_delta": {"base_dist": Categorical, "temperature": 1.0},
        "q_ACB": {"base_dist": TanhNormal, "min": 0.0, "max": 1.0}
    },
    return_log_prob=True # 必须返回对数概率 'sample_log_prob' 用于 PPO 损失计算
)
3.3. Value 网络 (φ) — 单头评论家
Value 网络用于估计状态价值 V
φ
​
 (s
t
​
 )。

Python

from torchrl.modules import ValueOperator

# 状态特征提取网络 (使用与 Actor 不同的编码器或仅是单独的线性头)
value_head = nn.Sequential(
    nn.Linear(feature_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 1) # 输出状态价值 V(s)
)

# ValueOperator 封装特征编码器和价值头
# 假设我们与 Actor 共享 StateEncoder
Value = ValueOperator(
    module=nn.Sequential(StateEncoder, value_head),
    in_keys=["observation"], # 输入状态
    out_keys=["state_value"], # 输出状态价值 V_phi(s_t)
)
阶段 4：PPO 核心模块配置 (S21, S23, S24)
4.1. 初始化训练参数 (S21)
Python

import torch.optim as optim
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

# S21: 初始化训练参数
K = 1000       # 迭代次数 K (epochs)
T = 10 * 160   # 轨迹长度 T (frames_per_batch), 设定为10个决策周期的数据
B = 64         # 小批量大小 B (sub_batch_size)
clip_epsilon = 0.2 # 裁剪参数 epsilon [3]
gamma = 0.99   # 折扣因子 gamma
lmbda = 0.95   # GAE lambda
lr_pi = 3e-4   # Actor 学习率 lambda_pi
lr_V = 1e-3    # Value 学习率 lambda_V
num_epochs = 10 # 优化步骤 k (内循环迭代次数)
4.2. 数据采集器 (S23)
用于采集经验数据 (s
t
​
 ,a
t
​
 ,r
t
​
 ,s
t+1
​
 )。

Python

# S23: 卫星接入点与仿真环境交互采集经验数据
collector = SyncDataCollector(
    env,
    Actor,
    frames_per_batch=T, # 每次采集 T 步经验
    max_frames_per_traj=env.current_env_num_slots, # 决策周期长度 (160 slots)
    device="cpu", # 采集通常在 CPU 上进行
)
4.3. 优势估计 (GAE) (S24)
TorchRL 的 GAE 模块会自动计算 A
t
​
  ('advantage') 和
V
^

φ
​
 (s
t
​
 ) ('value_target') 。  

Python

# S24: 计算经验数据的优势估计
advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=Value, # 使用 Value 网络进行价值估计
)

# 确保 GAE 知道 V(s_t) 存储在 'state_value' 键中
# GAE 将读取 state_value, reward, done/terminated 来计算 advantage 和 value_target
阶段 5：优化与损失函数 (S25, S26)
5.1. 损失模块与优化器定义 (S25)
使用 ClipPPOLoss 模块封装您的 J
π
​
 (θ) 和 J
V
​
 (φ) 目标函数。

Python

# 损失模块
loss_module = ClipPPOLoss(
    actor=Actor,
    critic=Value,
    clip_epsilon=clip_epsilon,
    loss_critic_type="smooth_l1", # 价值损失 L2/L1/smooth_l1, 默认 L2 (MSE)
    entropy_coeff=0.01, # 熵正则化系数
    normalize_advantage=True, # 建议对优势进行归一化以提高稳定性
)

# 优化器
# 建议使用两个独立的优化器，以针对策略和价值函数使用不同的学习率
optim_pi = optim.Adam(loss_module.actor_network.parameters(), lr=lr_pi) # 优化 theta
optim_V = optim.Adam(loss_module.critic_network.parameters(), lr=lr_V) # 优化 phi
5.2. PPO 主训练循环 (S26)
Python

# S26: 重复 K 次 S23 至 S25，输出训练好的网络
for iteration, data in enumerate(collector):
    if iteration >= K:
        break

    # 1. (S24) 计算优势估计和价值目标
    with torch.no_grad():
        data = advantage_module(data) # data 中现在包含 'advantage' 和 'value_target'

    # 2. (S251) 数据缓存区与小批量划分
    # 使用 ReplayBuffer 存储当前批次数据，以便重复采样
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(data.shape),
        sampler=SamplerWithoutReplacement(), # PPO 需要不重复采样
        batch_size=B,
    ).extend(data)

    # 3. (S252) 重复优化 num_epochs 次 (K次)
    for epoch in range(num_epochs):
        for batch in replay_buffer:
            # S252: 用小批量数据对模型进行更新

            # **策略更新 J_pi(theta):**
            # ClipPPOLoss 计算联合策略比率
            loss_td = loss_module(batch)

            loss_pi = loss_td["loss_objective"] # J_pi(theta)
            loss_V = loss_td["loss_value"]     # J_V(phi)
            loss_ent = loss_td["loss_entropy"] # 熵损失

            # 策略优化 (J_pi)
            optim_pi.zero_grad()
            loss_pi.backward()
            optim_pi.step()

            # 价值优化 (J_V)
            optim_V.zero_grad()
            loss_V.backward()
            optim_V.step()

    # 记录日志，衰减学习率等...

print("训练完成，Actor 网络 (theta) 和 Value 网络 (phi) 已训练完毕。")
# 保存最终模型
# torch.save(Actor.state_dict(), "actor_final.pth")
# torch.save(Value.state_dict(), "value_final.pth")