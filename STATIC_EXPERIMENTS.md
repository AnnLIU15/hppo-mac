# 静态场景实验说明

本目录包含两个独立的静态场景实验notebook，用于评估H-PPO算法在不同静态流量配置下的性能。

## 📁 文件结构

```
hppo-mac/
├── static_density_sweep.ipynb      # 密度扫描实验
├── static_ratio_sweep.ipynb        # 比例扫描实验
├── static_cmp.ipynb               # 原始合并版本（已废弃）
└── utils/
    ├── evaluate.py                 # 公共评估函数
    └── __init__.py                # 工具模块导出
```

## 🔬 实验类型

### 1. 密度扫描实验 (`static_density_sweep.ipynb`)

**目标**: 固定业务比例，扫描不同总密度下的性能表现

**配置**:
- **4种业务混合模式**:
  - `CBRA_Heavy`: CBRA竞争为主 (70% CBRA, 20% PBRA, 10% CFRA)
  - `PBRA_Heavy`: PBRA调度为主 (20% CBRA, 70% PBRA, 10% CFRA)
  - `HO_Interference_Heavy`: 干扰严重场景 (20% CBRA, 10% PBRA, 70% CFRA)
  - `Balanced`: 相对均衡 (40% CBRA, 35% PBRA, 25% CFRA)

- **7个密度点**: [10, 30, 50, 80, 100, 150, 200]

- **总场景数**: 4 × 7 = 28个场景

**输出**:
- 训练日志: `logs/{seed}_density_sweep.log`
- 可视化结果: `output/static/density_sweep_comparison.svg`
- 2×2子图展示四种业务模式下的密度-性能曲线

### 2. 比例扫描实验 (`static_ratio_sweep.ipynb`)

**目标**: 固定总密度，扫描不同业务比例下的性能表现

**配置**:
- **固定总密度**: 100.0
- **固定CFRA比例**: 15%
- **CBRA比例范围**: 在剩余85%容量中从10%扫描到90%
- **扫描点数**: 9个点 (alpha = 0.1 到 0.9)

- **总场景数**: 9个场景

**输出**:
- 训练日志: `logs/{seed}_ratio_sweep.log`
- 可视化结果: `output/static/ratio_sweep_comparison.svg`
- 单图展示CBRA比例对性能的影响

## 🛠️ 公共函数 (`utils/evaluate.py`)

### 核心函数

1. **`static_sensitivity_config()`**
   - 创建静态场景的模拟器配置
   - 支持自定义密度和业务比例
   - 自动校验比例之和为1.0

2. **`evaluate_trained_agent()`**
   - 评估训练好的agent
   - 收集详细的协议统计信息
   - 计算成功率和奖励指标

3. **`run_baseline_comparison()`**
   - 运行固定分配和启发式算法基线
   - **关键**：使用与RL相同的环境配置（`simulator_config`）
   - 仅修改 `flatten_observation=False`，确保公平对比
   - 返回两种策略的性能指标

4. **`train_and_evaluate_scenario()`**
   - 完整的单场景训练和评估流程
   - 包括环境构建、模型训练、评估、基线对比
   - 自动深拷贝保护数据独立性

5. **`print_results_summary()`**
   - 打印所有场景的结果摘要
   - 显示相对提升百分比

### 辅助函数

- **`_scalar()`**: 从info字典中提取标量值

### 🔑 环境配置一致性保证

**重要**：启发式算法的环境配置与RL完全相同，确保公平对比：

```python
# 在 train_and_evaluate_scenario() 中：
# 1. RL训练使用的配置
scenario_env_config = replace(env_config, simulator_config=sim_config)

# 2. Baseline使用相同的simulator_config
# 在 run_baseline_comparison() 中：
baseline_env_config = replace(scenario_env_config, flatten_observation=False)
# 这确保了：
# - 相同的流量密度和比例（sim_config）
# - 相同的preamble配置
# - 相同的随机种子偏移
# - 唯一区别：flatten_observation（因为baseline不需要扁平化观察）
```

## 🚀 使用方法

### 运行密度扫描实验

```bash
# 在Jupyter中打开并运行所有单元格
jupyter notebook static_density_sweep.ipynb
```

### 运行比例扫描实验

```bash
# 在Jupyter中打开并运行所有单元格
jupyter notebook static_ratio_sweep.ipynb
```

### 自定义实验

两个notebook都使用相同的训练配置（100轮迭代），您可以根据需要调整：

```python
# 从utils导入配置函数
from utils.evaluate import static_sensitivity_config

# 创建自定义场景
custom_config = static_sensitivity_config(
    total_density=120.0,   # 自定义密度
    cbra_ratio=0.5,        # 50% CBRA
    pbra_ratio=0.3,        # 30% PBRA
    cfra_ratio=0.2,        # 20% CFRA
    duration=11.0          # 持续时间
)

# 调整训练配置
train_config = HPPOConfig(
    frames_per_batch=2560,
    mini_batch_size=256,
    rollout_epochs=4,
    max_iterations=100,  # 可以调整训练轮数
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coeff=0.1,
    actor_lr=3e-4,
    critic_lr=3e-4,
    device=torch.device("cpu"),
)
```

## 📊 结果分析

每个实验完成后会：

1. **打印训练进度**: 每10轮显示一次当前奖励
2. **保存训练日志**: 完整的迭代记录保存在logs目录
3. **生成对比图表**:
   - PPO (Ours) vs 固定分配 vs 启发式算法
   - SVG格式，高分辨率（300 DPI）
4. **输出统计摘要**:
   - 每个场景的平均奖励
   - 相对于基线的性能提升百分比

## 🔧 技术要点

### 数据隔离
- 所有场景结果使用 `deepcopy()` 存储
- 避免多场景间的数据干扰

### 并行训练
- 使用8个并行环境 (`num_envs=8`)
- `ParallelEnv` 后端加速采样

### 模型保护
- 加载模型权重前进行深拷贝
- 保护最佳模型不被后续评估修改

### 错误处理
- 单个场景失败不影响其他场景
- 打印详细的错误堆栈信息

## 📝 注意事项

1. **计算资源**: 每个实验需要较长时间（28场景约数小时，9场景约1小时）
2. **内存占用**: 建议至少16GB RAM
3. **GPU支持**: 可在配置中修改 `device` 为 `cuda` 以使用GPU加速
4. **随机种子**: 默认使用配置文件中的种子，确保结果可重复

## 🔄 从旧版本迁移

如果您之前使用 `static_cmp.ipynb`，现在应该：

1. **密度扫描实验**: 使用 `static_density_sweep.ipynb`
2. **比例扫描实验**: 使用 `static_ratio_sweep.ipynb`
3. **公共函数**: 从 `utils.evaluate` 导入，无需重复定义

旧的 `static_cmp.ipynb` 保留用于参考，但推荐使用新的拆分版本。
