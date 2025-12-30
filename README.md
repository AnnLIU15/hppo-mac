# H-PPO MAC Environment

基于强化学习的卫星随机接入 MAC 协议优化环境与训练框架

---

## 📖 项目简介

本项目提供了一个面向强化学习的**卫星随机接入 MAC 仿真环境**，用于研究和优化卫星通信系统中的媒体访问控制协议。项目实现了完整的训练、评估和可视化工具链，支持混合动作空间的策略优化（H-PPO）。

### 核心特性

- **🛰️ 真实卫星场景建模**
  - 多区域动态覆盖模型，模拟卫星轨道运动
  - 基于相位漂移和平滑插值的区域权重生成
  - 可配置的流量模式和终端分布

- **🎯 精细的 MAC 协议仿真**
  - 支持 CBRA（竞争退避随机接入）和 PBRA（前导退避随机接入）
  - ACB（接入类限制）机制动态调控
  - 碰撞检测、退避队列管理、前导序列分配

- **🤖 强化学习环境**
  - 符合 Gymnasium 标准的环境接口
  - 混合动作空间（离散 + 连续）
  - 灵活的观测空间配置（扁平/字典）
  - 支持历史状态堆叠

- **⚡ 高效训练框架**
  - 基于 TorchRL 的 H-PPO 实现
  - 支持多进程并行环境采集
  - GPU 加速训练
  - 完善的日志和检查点管理

- **📊 丰富的分析工具**
  - 启发式策略基线
  - 参数扫描和自动调优
  - 可视化遥测数据
  - 静态/动态场景对比分析

---

## 📁 项目结构

```
hppo-mac/
├── env/                          # 环境模块
│   ├── mac_simulator.py          # MAC 协议核心仿真器
│   ├── satellite_mac_env.py      # Gymnasium 环境封装
│   └── gym_helpers.py            # 环境辅助工具
├── algo/                         # 算法模块
│   ├── hppo.py                   # H-PPO 训练实现
│   └── logger.py                 # 训练日志工具
├── utils/                        # 工具模块
│   └── evaluate.py               # 评估和场景生成工具
├── conf/                         # 配置文件
│   └── default.yaml              # 默认训练配置
├── output/                       # 输出目录
│   ├── dynamic/                  # 动态场景结果
│   ├── static/                   # 静态场景结果
│   └── model/                    # 训练模型检查点
├── baseline.py                   # 启发式策略基线
├── *.ipynb                       # Jupyter 分析笔记本
│   ├── dynamics_cmp.ipynb        # 动态场景对比
│   ├── static_density_sweep_*.ipynb  # 密度扫描实验
│   └── static_ratio_sweep.ipynb  # 比例扫描实验
├── pyproject.toml                # 项目依赖配置
└── README.md                     # 项目文档
```

---

## 🚀 快速开始

### 环境要求

- **Python**: >= 3.13
- **操作系统**: Windows / Linux / macOS
- **硬件**: 建议使用 GPU（CUDA）进行训练加速

### 安装步骤

1. **安装UV**(https://docs.astral.sh/uv/)
   ```bash
   # windows powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   # 或者在 Linux/macOS 终端执行
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

2. **同步/创建虚拟环境**
   ```bash
   # 使用 uv
   uv synv # 自动安装依赖
   source .venv/bin/activate  # Linux/macOS
   # 或
   .venv\Scripts\activate     # Windows
   ```

### 依赖包说明

- **核心依赖**：`gymnasium`, `torch`, `torchrl`, `numpy`
- **日志与可视化**：`loguru`, `matplotlib`, `scienceplots`, `pandas`
- **交互式开发**：`jupyter`, `tqdm`

---

## 💡 使用指南

### 1. 运行启发式基线

启发式策略提供了一个不依赖学习的基线方案，用于验证环境功能和对比强化学习性能。

```bash
python baseline.py
```

**输出内容**：
- 控制台实时显示每回合的性能指标
- `logs/telemetry_ep*.npz`：详细的步级遥测数据
- `logs/telemetry_summary.csv`：回合级统计摘要
- `logs/sweep_*/`：参数扫描结果

### 2. 训练与评估

项目提供了多个 Jupyter Notebook 用于 H-PPO 智能体训练、性能评估和深入分析。启动 Jupyter 后运行相应笔记本即可：

```bash
jupyter notebook
```

**可用笔记本**：

- **`dynamics_cmp.ipynb`**：动态场景下的训练与对比评估
  - H-PPO 智能体完整训练流程
  - 启发式策略基线对比
  - 性能指标可视化分析
  - 训练模型保存与加载

- **`static_density_sweep_Balanced.ipynb`**：平衡场景下的密度扫描实验
- **`static_density_sweep_CBRA.ipynb`**：CBRA 密集场景性能评估
- **`static_density_sweep_PBRA.ipynb`**：PBRA 密集场景性能评估
- **`static_density_sweep_HO.ipynb`**：高密度场景性能分析
- **`static_ratio_sweep.ipynb`**：不同协议比例下的性能分析

**训练输出**：
- `output/model/`：训练模型检查点（actor/critic）
- `output/dynamic/` 和 `output/static/`：场景评估结果
- Notebook 内嵌可视化图表


---

## 🔧 核心组件详解

### MAC 仿真器（`MACSimulator`）

位于 `env/mac_simulator.py`，实现了完整的 MAC 协议仿真逻辑。

**主要功能**：
- **区域覆盖模型**：`CoveragePatch` 配置，支持多区域动态权重
- **流量生成**：`RegionTrafficProfile` 定义各区域的流量特征
- **退避策略**：`BackoffStrategyConfig` 配置 CBRA/PBRA 退避参数
- **前导序列管理**：动态分配和碰撞检测
- **ACB 控制**：接入概率调控

**关键方法**：
```python
reward, obs, info = simulator.run_slots(
    slots=160,
    cbra_preambles=27,
    pbra_preambles=27,
    q_ACB=0.8
)
```

### 强化学习环境（`SatelliteMACEnv`）

位于 `env/satellite_mac_env.py`，提供符合 Gymnasium 标准的环境接口。

**动作空间**（混合）：
- `delta_cbra`: CBRA 前导序列调整（离散）
- `delta_pbra`: PBRA 前导序列调整（离散）
- `q_ACB`: ACB 因子（连续，0-1）

**观测空间**：
- 请求队列状态（CBRA/PBRA）
- 碰撞率统计
- 活跃终端分布
- 前导序列使用率
- 区域混合权重
- 历史统计（可配置）

**配置示例**：
```python
from env.satellite_mac_env import SatelliteMACEnv, SatelliteMACEnvConfig

config = SatelliteMACEnvConfig(
    num_slots_per_step=160,
    decision_horizon=1,
    preamble_delta_range=2,
    flatten_observation=True
)
env = SatelliteMACEnv(config=config)
```

### H-PPO 算法（`algo/hppo.py`）

基于 TorchRL 实现的混合动作空间 PPO 算法。

**核心特性**：
- **混合策略网络**：独立的离散和连续策略头
- **GAE 优势估计**：广义优势估计
- **PPO-Clip**：裁剪策略梯度
- **价值函数**：共享特征提取器的 Critic
- **并行采集**：支持多进程环境并行

**训练流程**：
```python
from algo.hppo import train_hppo

actor, critic = train_hppo(
    env=env,
    frames_per_batch=1600,
    rollout_epochs=10,
    max_iterations=1000,
    device="cuda"
)
```

---

## 📊 实验与结果

### 场景类型

1. **静态场景**：固定流量分布，用于基础性能评估
   - [平衡场景（Balanced）](static_density_sweep_Balanced.ipynb)
   - [CBRA 密集场景](static_density_sweep_CBRA.ipynb)
   - [PBRA 密集场景](static_density_sweep_PBRA.ipynb)
   - [高密度场景（HO）](static_density_sweep_HO.ipynb)

2. **动态场景**：时变流量和覆盖，模拟真实卫星运行 [dynamics_cmp.ipynb](dynamics_cmp.ipynb)
   - 轨道相位变化
   - 区域权重动态调整
   - 流量波动

### 评估指标

- **吞吐量（Throughput）**：成功接入的平均速率
- **碰撞率（Collision Ratio）**：碰撞请求占比
- **退避队列（Backlog）**：CBRA/PBRA 队列长度
- **奖励（Reward）**：综合性能指标

### 基线对比

- **启发式策略**：基于碰撞率阈值的规则控制
- **H-PPO**：基于强化学习的自适应控制
- **静态配置**：固定参数不调整

结果存储在 `output/` 目录，可通过 Jupyter Notebook 进行可视化分析。

---

## 🛠️ 配置说明

### 环境配置（`conf/default.yaml`）

```yaml
seed: 42

environment:
  num_slots: 160          # 每步时隙数
  history_size: 10        # 历史缓冲区长度
  total_preambles: 64     # 总前导序列数
  base_preamble_split: [27, 27, 10]  # CBRA/PBRA/保留

algorithm:
  frames_per_batch: 1600  # 批次帧数
  mini_batch_size: 64     # 小批次大小
  max_epochs: 1000        # 最大训练轮数
  clip_epsilon: 0.2       # PPO 裁剪参数
  gamma: 0.99             # 折扣因子
  gae_lambda: 0.95        # GAE λ 参数
  actor_lr: 0.0003        # Actor 学习率
  critic_lr: 0.001        # Critic 学习率
  entropy_coeff: 0.01     # 熵正则化系数
```

### 仿真器配置（`MACSimulatorConfig`）

关键参数可在代码中直接配置：

```python
from env.mac_simulator import (
    MACSimulatorConfig,
    BackoffStrategyConfig,
    BackoffWindow
)

sim_config = MACSimulatorConfig(
    regions=[...],  # 区域流量配置
    backoff_strategy=BackoffStrategyConfig(
        cbra_collision_threshold=0.3,
        pbra_collision_threshold=0.25,
        cbra_window=BackoffWindow.MEDIUM,
        pbra_window=BackoffWindow.LARGE,
        max_backlog_cbra=500,
        max_backlog_pbra=500
    )
)
```