# 评估函数使用指南

## 概述

评估函数已经重构，将训练和评估分离，支持以下场景：
1. **完整训练+评估**：一次性完成训练和所有评估
2. **仅训练**：只训练模型，不进行评估（用于大规模训练）
3. **仅评估PPO**：加载已训练的模型进行评估
4. **仅评估Baseline**：独立运行baseline策略（方便调试）
5. **混合评估**：加载模型后与baseline对比

## 函数说明

### 1. `train_scenario()` - 仅训练
训练单个场景，不包含评估。

```python
from utils.evaluate import train_scenario

train_result = train_scenario(
    scenario_name="test_scenario",
    sim_config=sim_config,
    env_config=env_config,
    train_config=train_config,
    build_env_func=build_env,
    build_modules_func=build_modules,
    train_func=train_ppo,
    logger=logger,
    num_envs=8,
    seed=42
)

# 返回值包含：
# - rl_env: TorchRL环境
# - actor, critic: 模型
# - best_model, last_model: 最佳和最后一轮的state dict
# - env_config, sim_config: 配置
# - metrics_history: 训练历史
```

### 2. `evaluate_scenario_ppo()` - 仅评估PPO
评估已训练的PPO模型。

```python
from utils.evaluate import evaluate_scenario_ppo

ppo_results = evaluate_scenario_ppo(
    scenario_name="test_scenario",
    scenario_env_config=train_result['env_config'],
    actor=train_result['actor'],
    critic=train_result['critic'],
    rl_env=train_result['rl_env'],
    best_model=train_result['best_model'],
    last_model=train_result['last_model'],
    num_episodes=20,
    seed=42
)

# 返回值包含：
# - eval_results_best: 最佳模型的评估结果
# - eval_results_last: 最后一轮模型的评估结果
```

### 3. `evaluate_scenario_baseline()` - 仅评估Baseline
独立评估baseline策略，方便调试。

```python
from utils.evaluate import evaluate_scenario_baseline
from baseline import run_episode

baseline_results = evaluate_scenario_baseline(
    scenario_name="test_scenario",
    scenario_env_config=env_config,  # 可以自定义配置
    run_episode_func=run_episode,
    num_episodes=20,
    seed=42,
    fix_acb=0.2
)

# 返回值包含：
# - baseline_results: 包含"固定分配"和"启发式算法"的结果
```

### 4. `train_and_evaluate_scenario()` - 完整流程
一次性完成训练和所有评估（原有功能）。

```python
from utils.evaluate import train_and_evaluate_scenario

full_result = train_and_evaluate_scenario(
    scenario_name="test_scenario",
    sim_config=sim_config,
    env_config=env_config,
    train_config=train_config,
    build_env_func=build_env,
    build_modules_func=build_modules,
    train_func=train_ppo,
    run_episode_func=run_episode,
    logger=logger,
    num_envs=8,
    num_episodes=20,
    seed=42
)

# 返回值包含所有结果：
# - env_config, sim_config
# - best_model, last_model
# - eval_results_best, eval_results_last
# - baseline_results
```

## 使用场景

### 场景 1：调试Baseline策略

当你修改了 `baseline.py` 的启发式策略，想快速测试：

```python
from utils.evaluate import evaluate_scenario_baseline, static_sensitivity_config
from baseline import run_episode

# 创建测试配置
sim_config = static_sensitivity_config(
    total_density=50.0,
    cbra_ratio=0.5,
    pbra_ratio=0.3,
    cfra_ratio=0.2
)

# 使用与训练相同的env_config
env_config = build_env_config(...)

# 只运行baseline评估
baseline_results = evaluate_scenario_baseline(
    scenario_name="debug_heuristic",
    scenario_env_config=replace(env_config, simulator_config=sim_config),
    run_episode_func=run_episode,
    num_episodes=10,  # 少量episode快速测试
    seed=42
)

print(f"固定分配: {baseline_results['baseline_results']['固定分配']['reward']:.4f}")
print(f"启发式: {baseline_results['baseline_results']['启发式算法']['reward']:.4f}")
```

### 场景 2：加载模型进行测试

加载已保存的模型，在新场景下测试：

```python
import torch
from utils.evaluate import evaluate_scenario_ppo

# 加载保存的模型
checkpoint = torch.load("output/model/best_model.pth")

# 重建环境和模型
# ... (创建 rl_env, actor, critic)

# 评估
ppo_results = evaluate_scenario_ppo(
    scenario_name="test_new_scenario",
    scenario_env_config=new_env_config,
    actor=actor,
    critic=critic,
    rl_env=rl_env,
    best_model=checkpoint['best_model'],
    last_model=checkpoint['last_model'],
    num_episodes=50,
    seed=42
)
```

### 场景 3：大规模训练后批量评估

先训练多个模型，再统一评估：

```python
# 第一阶段：批量训练
trained_models = []
for scenario_config in scenario_configs:
    train_result = train_scenario(
        scenario_name=scenario_config['name'],
        sim_config=scenario_config['sim_config'],
        # ... 其他参数
    )
    trained_models.append(train_result)

    # 保存模型
    torch.save({
        'best_model': train_result['best_model'],
        'last_model': train_result['last_model'],
    }, f"output/model/{scenario_config['name']}.pth")

# 第二阶段：批量评估（可以在不同时间进行）
for train_result in trained_models:
    # 评估PPO
    ppo_results = evaluate_scenario_ppo(...)

    # 评估Baseline
    baseline_results = evaluate_scenario_baseline(...)

    # 对比和保存结果
    # ...
```

### 场景 4：单独调试某个Baseline配置

修改 `HeuristicConfig` 后快速测试：

```python
from baseline import HeuristicConfig, run_episode

# 在 baseline.py 中临时修改配置
# class HeuristicConfig:
#     acb_step_up: float = 0.4  # 修改这里
#     acb_step_down: float = 0.05  # 修改这里

# 运行评估
baseline_results = evaluate_scenario_baseline(
    scenario_name="test_acb_params",
    scenario_env_config=env_config,
    run_episode_func=run_episode,
    num_episodes=20,
    seed=42
)

# 查看结果
print("CBRA成功率:", baseline_results['baseline_results']['启发式算法']['rate_cbra'])
print("PBRA成功率:", baseline_results['baseline_results']['启发式算法']['rate_pbra'])
```

## 主要改进

1. **解耦训练和评估**：可以独立运行训练或评估
2. **支持加载模型**：方便测试已训练的模型
3. **独立Baseline评估**：不需要训练就能测试baseline
4. **灵活组合**：根据需要选择运行哪些部分
5. **调试友好**：修改baseline代码后可以快速测试

## 注意事项

1. **环境配置一致性**：评估时使用的 `env_config` 应该与训练时一致（除了 `flatten_observation`）
2. **随机种子**：为了可复现性，建议设置固定的种子
3. **Baseline的flatten_observation**：baseline评估时会自动将 `flatten_observation` 设为 `False`
4. **模型state dict**：保存模型时使用深拷贝，避免修改原始模型

## 示例：完整的独立评估脚本

```python
# test_baseline_only.py
from pathlib import Path
from baseline import run_episode
from utils.evaluate import evaluate_scenario_baseline, static_sensitivity_config
from env import SatelliteMACEnvConfig, default_simulator_config
from dataclasses import replace

# 1. 创建场景配置
sim_config = static_sensitivity_config(
    total_density=50.0,
    cbra_ratio=0.5,
    pbra_ratio=0.3,
    cfra_ratio=0.2,
    duration=13.0
)

# 2. 创建环境配置
env_config = SatelliteMACEnvConfig(
    num_slots_per_step=800,
    decision_horizon=256,
    history_len=8,
    preamble_delta_range=3,
    flatten_observation=False,
    simulator_config=sim_config,
)

# 3. 运行baseline评估
results = evaluate_scenario_baseline(
    scenario_name="standalone_test",
    scenario_env_config=env_config,
    run_episode_func=run_episode,
    num_episodes=20,
    seed=42
)

# 4. 打印结果
print("\n=== Baseline评估结果 ===")
for strategy, metrics in results['baseline_results'].items():
    print(f"\n{strategy}:")
    print(f"  平均奖励: {metrics['reward']:.4f}")
    print(f"  CBRA成功率: {metrics['rate_cbra']:.2f}%")
    print(f"  PBRA成功率: {metrics['rate_pbra']:.2f}%")
    print(f"  CFRA成功率: {metrics['rate_cfra']:.2f}%")
```

运行：
```bash
python test_baseline_only.py
```

---

**更新日期**：2025年12月4日
**版本**：2.0 - 训练评估分离版
