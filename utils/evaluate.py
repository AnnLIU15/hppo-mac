"""
评估和基线对比的公共函数
用于静态和动态场景的训练评估
"""

from typing import Dict
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import yaml

from env import SatelliteMACEnv, SatelliteMACEnvConfig, MACSimulatorConfig
from env.mac_simulator import RegionSegment, RegionTrafficProfile
from torchrl.envs.libs.gym import GymWrapper
from dataclasses import replace


def load_yaml_config(path: Path) -> Dict:
    """从YAML文件加载配置

    Args:
        path: YAML配置文件路径

    Returns:
        配置字典，如果文件不存在返回空字典
    """
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            print(f"Loaded configuration from {path}")
            return data
    print(f"Configuration file {path} not found, using defaults.")
    return {}


def static_sensitivity_config(
    total_density: float,
    cbra_ratio: float,
    pbra_ratio: float,
    cfra_ratio: float,
    duration: float = 13.0
) -> MACSimulatorConfig:
    """创建静态场景的模拟器配置

    Args:
        total_density: 总业务密度
        cbra_ratio: CBRA业务比例
        pbra_ratio: PBRA业务比例
        cfra_ratio: CFRA业务比例
        duration: 场景持续时间

    Returns:
        配置好的MACSimulatorConfig
    """
    # 1. 校验比例之和是否为 1
    total_ratio = cbra_ratio + pbra_ratio + cfra_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Error: Ratios must sum to 1.0. Current sum: {total_ratio:.4f}")

    # 2. 根据比例计算绝对密度
    profile_name = f"static_D{total_density}_C{cbra_ratio:.1f}_P{pbra_ratio:.1f}_F{cfra_ratio:.1f}"

    region = RegionTrafficProfile(
        name=profile_name,
        cbra_density=total_density * cbra_ratio,
        pbra_density=total_density * pbra_ratio,
        cfra_density=total_density * cfra_ratio
    )

    # 3. 创建单一的时间段
    segments = [RegionSegment(region_name=profile_name, length=duration)]

    return MACSimulatorConfig(regions=(region,), segments=tuple(segments))


def _scalar(info: dict, key: str, default: float = 0.0) -> float:
    """从info字典中提取标量值

    Args:
        info: info字典
        key: 要提取的键
        default: 默认值

    Returns:
        提取的标量值
    """
    raw = info.get(key, default)
    arr = np.asarray(raw)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def evaluate_trained_agent(actor, rl_env, num_episodes=20, seed_offset=0, seed=None):
    """评估训练好的agent并收集详细统计信息

    Args:
        actor: 训练好的actor网络
        rl_env: 训练时使用的TorchRL环境
        num_episodes: 评估的episode数量
        seed_offset: 随机种子偏移量
        seed: 基础随机种子，如果为None则不设置种子

    Returns:
        包含评估指标的字典
    """
    actor.eval()

    all_rewards = []
    all_arrivals_cbra = []
    all_arrivals_pbra = []
    all_arrivals_cfra = []
    all_success_cbra = []
    all_success_pbra = []
    all_success_cfra = []
    all_episode_lengths = []
    all_theoretical_optimal = []

    for episode_idx in range(num_episodes):
        if seed is not None:
            episode_seed = seed + seed_offset + episode_idx
            rl_env.set_seed(episode_seed)

        # 重置环境 - 使用TorchRL环境
        td = rl_env.reset()

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # 使用actor选择动作
            with torch.no_grad():
                td_action = actor(td)
            # 执行动作并获取下一个状态
            td_next = rl_env.step(td_action)
            # 提取reward和done信息
            reward = td_next.get(("next", "reward")).mean()
            done = td_next.get(("next", "done"))[0]

            episode_reward += reward
            episode_length += 1

            # 准备下一步
            td = rl_env.reset() if done else td_next["next"]

        # 收集episode统计信息
        all_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)

        # 尝试从info中提取统计信息（如果TorchRL环境有提供）
        def extract_scalar(td, key, default=0.0):
            try:
                if "next" in td.keys(True):
                    info_td = td["next"]
                    if key in info_td.keys():
                        val = info_td[key].mean()
                        if torch.is_tensor(val):
                            return float(val.item())
                        return float(val)
            except (KeyError, AttributeError, RuntimeError):
                pass
            return float(default)

        all_arrivals_cbra.append(extract_scalar(td_next, "total_arrivals_cbra", 0))
        all_arrivals_pbra.append(extract_scalar(td_next, "total_arrivals_pbra", 0))
        all_arrivals_cfra.append(extract_scalar(td_next, "total_arrivals_cfra", 0))
        all_success_cbra.append(extract_scalar(td_next, "total_success_cbra", 0))
        all_success_pbra.append(extract_scalar(td_next, "total_success_pbra", 0))
        all_success_cfra.append(extract_scalar(td_next, "total_success_cfra", 0))
        all_theoretical_optimal.append(extract_scalar(td_next, "total_theoretical_optimal", 0))

    # 计算统计结果
    results = {
        "eval_reward_mean": float(np.mean(all_rewards)),
        "eval_reward_std": float(np.std(all_rewards)),
        "eval_episode_length_mean": float(np.mean(all_episode_lengths)),
        "eval_arrivals_cbra": float(np.mean(all_arrivals_cbra)),
        "eval_arrivals_pbra": float(np.mean(all_arrivals_pbra)),
        "eval_arrivals_cfra": float(np.mean(all_arrivals_cfra)),
        "eval_success_cbra": float(np.mean(all_success_cbra)),
        "eval_success_pbra": float(np.mean(all_success_pbra)),
        "eval_success_cfra": float(np.mean(all_success_cfra)),
        "eval_theoretical_optimal": float(np.mean(all_theoretical_optimal)),
    }

    # 计算成功率
    if results["eval_arrivals_cbra"] > 0:
        results["eval_rate_cbra"] = results["eval_success_cbra"] / results["eval_arrivals_cbra"]
    else:
        results["eval_rate_cbra"] = 0.0

    if results["eval_arrivals_pbra"] > 0:
        results["eval_rate_pbra"] = results["eval_success_pbra"] / results["eval_arrivals_pbra"]
    else:
        results["eval_rate_pbra"] = 0.0

    if results["eval_arrivals_cfra"] > 0:
        results["eval_rate_cfra"] = results["eval_success_cfra"] / results["eval_arrivals_cfra"]
    else:
        results["eval_rate_cfra"] = 0.0

    actor.train()
    return results


def run_baseline_comparison(
    scenario_env_config: SatelliteMACEnvConfig,
    run_episode_func,
    num_episodes: int = 20,
    seed: int = None,
    fix_acb: float = 0.2
) -> Dict[str, Dict]:
    """运行基线策略对比（固定分配和启发式算法）

    注意：baseline使用与RL相同的环境配置，只是将flatten_observation设为False
    这样确保baseline和RL在相同的流量、参数配置下进行公平对比

    Args:
        scenario_env_config: 环境配置（与RL训练时使用的配置相同）
        run_episode_func: 运行episode的函数（从baseline模块导入）
        num_episodes: 每个策略运行的episode数量
        seed: 随机种子
        fix_acb: 固定分配时的ACB比例

    Returns:
        包含两种基线策略结果的字典
    """
    # 创建baseline环境配置 - 使用与RL相同的simulator_config
    # 只修改flatten_observation，因为baseline策略不需要扁平化的观察
    baseline_env_config = replace(scenario_env_config, flatten_observation=False)
    baseline_env = SatelliteMACEnv(config=baseline_env_config)
    baseline_rng = np.random.default_rng(seed + 4242 if seed is not None else None)
    delta_range = int(baseline_env_config.preamble_delta_range)

    baseline_results = {}

    for fix in [True, False]:
        strategy_name = "固定分配" if fix else "启发式算法"
        print(f"  运行{strategy_name}基线...")

        strategy_records = []
        for episode_idx in range(num_episodes):
            episode_seed = int(baseline_rng.integers(0, 1_000_000))
            stats, last_info, _ = run_episode_func(
                baseline_env, delta_range, baseline_rng, episode_seed,
                fix=fix, fix_acb=fix_acb
            )

            arrivals_cbra = float(_scalar(last_info, "total_arrivals_cbra", 0.0))
            arrivals_pbra = float(_scalar(last_info, "total_arrivals_pbra", 0.0))
            arrivals_cfra = float(_scalar(last_info, "total_arrivals_cfra", 0.0))
            success_cbra = float(_scalar(last_info, "total_success_cbra", 0.0))
            success_pbra = float(_scalar(last_info, "total_success_pbra", 0.0))
            success_cfra = float(_scalar(last_info, "total_success_cfra", 0.0))
            theoretical_optimal = float(_scalar(last_info, "total_theoretical_optimal", 0.0))

            strategy_records.append({
                "reward": stats.reward / max(stats.steps, 1),
                "arrivals_cbra": arrivals_cbra,
                "arrivals_pbra": arrivals_pbra,
                "arrivals_cfra": arrivals_cfra,
                "success_cbra": success_cbra,
                "success_pbra": success_pbra,
                "success_cfra": success_cfra,
                "theoretical_optimal": theoretical_optimal,
            })

        # 计算平均值
        baseline_results[strategy_name] = {
            "reward": np.mean([r["reward"] for r in strategy_records]),
            "arrivals_cbra": np.mean([r["arrivals_cbra"] for r in strategy_records]),
            "arrivals_pbra": np.mean([r["arrivals_pbra"] for r in strategy_records]),
            "arrivals_cfra": np.mean([r["arrivals_cfra"] for r in strategy_records]),
            "success_cbra": np.mean([r["success_cbra"] for r in strategy_records]),
            "success_pbra": np.mean([r["success_pbra"] for r in strategy_records]),
            "success_cfra": np.mean([r["success_cfra"] for r in strategy_records]),
            "theoretical_optimal": np.mean([r["theoretical_optimal"] for r in strategy_records]),
        }

        # 计算成功率
        for protocol in ['cbra', 'pbra', 'cfra']:
            arr_key = f'arrivals_{protocol}'
            succ_key = f'success_{protocol}'
            rate_key = f'rate_{protocol}'
            arr_val = baseline_results[strategy_name][arr_key]
            succ_val = baseline_results[strategy_name][succ_key]
            baseline_results[strategy_name][rate_key] = (succ_val / arr_val * 100) if arr_val > 0 else 0

    return baseline_results


def train_scenario(
    scenario_name: str,
    sim_config: MACSimulatorConfig,
    env_config: SatelliteMACEnvConfig,
    train_config,
    build_env_func,
    build_modules_func,
    train_func,
    logger,
    num_envs: int = 8,
    env_backend: str = "parallel",
    seed: int = None
) -> Dict:
    """训练单个场景（不包含评估）

    Args:
        scenario_name: 场景名称
        sim_config: 模拟器配置
        env_config: 环境配置
        train_config: 训练配置
        build_env_func: 构建环境的函数
        build_modules_func: 构建actor/critic的函数
        train_func: 训练函数
        logger: 日志记录器
        num_envs: 并行环境数量
        env_backend: 环境后端类型
        seed: 随机种子

    Returns:
        包含训练结果的字典（环境、模型等）
    """
    from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
    from torchrl.envs.transforms import Compose, DoubleToFloat

    print(f"\n{'='*80}")
    print(f"开始训练场景: {scenario_name}")
    print(f"{'='*80}")

    # 1. 创建该场景专用的环境配置
    scenario_env_config = replace(env_config, simulator_config=sim_config)

    # 2. 构建环境
    def _make_scenario_env(rank: int = 0, **_: object) -> GymWrapper:
        env_seed = seed + rank if seed is not None else None
        wrapped = build_env_func(config=deepcopy(scenario_env_config))
        wrapped.set_seed(env_seed)
        wrapped.auto_register_info_dict()
        return wrapped

    if num_envs <= 1:
        scenario_base_env = _make_scenario_env(0)
    else:
        if env_backend == "parallel":
            scenario_base_env = ParallelEnv(num_envs, _make_scenario_env)
        else:
            scenario_base_env = SerialEnv(num_envs, _make_scenario_env, auto_reset=True)

    scenario_transforms = Compose(DoubleToFloat())
    scenario_rl_env = TransformedEnv(scenario_base_env, scenario_transforms)
    if seed is not None:
        scenario_rl_env.set_seed(seed)
    scenario_rl_env.to(train_config.device)

    # 3. 构建模型
    scenario_actor, scenario_critic = build_modules_func(scenario_rl_env, feature_dim=128)
    scenario_actor.to(train_config.device)
    scenario_critic.to(train_config.device)

    # 4. 训练
    scenario_metrics_history = []

    def _scenario_log_metrics(*, iteration: int, reward: float, **metrics: float) -> None:
        record = {
            "iteration": iteration,
            "reward": reward,
        }
        for key, value in sorted(metrics.items()):
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if key in {
                "requests_cbra", "requests_pbra", "preamble_cbra", "preamble_pbra", "preamble_cfra",
                "arr_cbra", "arr_pbra", "arr_cfra",
                "succ_cbra", "succ_pbra", "succ_cfra"
            }:
                clean_value = int(round(scalar))
            elif key.startswith("collision_ratio"):
                clean_value = round(max(0.0, min(scalar, 1.0)), 4)
            else:
                clean_value = scalar
            record[key] = clean_value
        scenario_metrics_history.append(record)

        if iteration % 10 == 0:  # 每10轮打印一次
            print(f"  iter={iteration:04d}, reward={reward:.4f}")

    logger.info(f"Starting PPO training for scenario: {scenario_name}")
    scenario_metrics, scenario_best_model, scenario_last_model = train_func(
        scenario_rl_env, scenario_actor, scenario_critic, train_config,
        logger_fn=_scenario_log_metrics
    )

    print(f"训练完成: 最佳奖励={scenario_best_model['reward']:.4f} (迭代{scenario_best_model['iteration']})")

    # 返回训练结果，包含环境、模型等
    return {
        'scenario_name': scenario_name,
        'env_config': deepcopy(scenario_env_config),
        'sim_config': deepcopy(sim_config),
        'rl_env': scenario_rl_env,
        'actor': scenario_actor,
        'critic': scenario_critic,
        'best_model': deepcopy(scenario_best_model),
        'last_model': deepcopy(scenario_last_model),
        'metrics_history': scenario_metrics_history,
    }


def evaluate_scenario_ppo(
    scenario_name: str,
    scenario_env_config: SatelliteMACEnvConfig,
    actor,
    critic,
    rl_env,
    best_model: Dict,
    last_model: Dict,
    num_episodes: int = 20,
    seed: int = None
) -> Dict:
    """评估PPO模型（最佳和最后一轮）

    Args:
        scenario_name: 场景名称
        scenario_env_config: 环境配置
        actor: actor网络
        critic: critic网络
        rl_env: TorchRL环境
        best_model: 最佳模型的state dict
        last_model: 最后一轮模型的state dict
        num_episodes: 评估的episode数量
        seed: 随机种子

    Returns:
        包含评估结果的字典
    """
    print(f"\n评估场景 {scenario_name} 的PPO模型...")

    # 评估最佳模型
    print("  评估最佳模型...")
    actor.load_state_dict(deepcopy(best_model['actor']))
    critic.load_state_dict(deepcopy(best_model['critic']))

    eval_results_best = evaluate_trained_agent(
        actor=actor,
        rl_env=rl_env,
        num_episodes=num_episodes,
        seed=seed
    )

    # 评估最后一轮模型
    print("  评估最后一轮模型...")
    actor.load_state_dict(deepcopy(last_model['actor']))
    critic.load_state_dict(deepcopy(last_model['critic']))

    eval_results_last = evaluate_trained_agent(
        actor=actor,
        rl_env=rl_env,
        num_episodes=num_episodes,
        seed=seed
    )

    print(f"  PPO最佳奖励: {eval_results_best['eval_reward_mean']:.4f}")
    print(f"  PPO最后奖励: {eval_results_last['eval_reward_mean']:.4f}")

    return {
        'eval_results_best': deepcopy(eval_results_best),
        'eval_results_last': deepcopy(eval_results_last),
    }


def evaluate_scenario_baseline(
    scenario_name: str,
    scenario_env_config: SatelliteMACEnvConfig,
    run_episode_func,
    num_episodes: int = 20,
    seed: int = None,
    fix_acb: float = 0.2
) -> Dict:
    """评估baseline策略（固定分配和启发式）

    Args:
        scenario_name: 场景名称
        scenario_env_config: 环境配置
        run_episode_func: 运行episode的函数（从baseline模块导入）
        num_episodes: 评估的episode数量
        seed: 随机种子
        fix_acb: 固定分配时的ACB比例

    Returns:
        包含baseline评估结果的字典
    """
    print(f"\n评估场景 {scenario_name} 的Baseline策略...")

    baseline_results = run_baseline_comparison(
        scenario_env_config=scenario_env_config,
        run_episode_func=run_episode_func,
        num_episodes=num_episodes,
        seed=seed,
        fix_acb=fix_acb
    )

    print(f"  固定分配奖励: {baseline_results['固定分配']['reward']:.4f}")
    print(f"  启发式算法奖励: {baseline_results['启发式算法']['reward']:.4f}")

    return {
        'baseline_results': deepcopy(baseline_results),
    }


def train_and_evaluate_scenario(
    scenario_name: str,
    sim_config: MACSimulatorConfig,
    env_config: SatelliteMACEnvConfig,
    train_config,
    build_env_func,
    build_modules_func,
    train_func,
    run_episode_func,
    logger,
    num_envs: int = 8,
    env_backend: str = "parallel",
    num_episodes: int = 20,
    seed: int = None
) -> Dict:
    """训练和评估单个场景（完整流程）

    Args:
        scenario_name: 场景名称
        sim_config: 模拟器配置
        env_config: 环境配置
        train_config: 训练配置
        build_env_func: 构建环境的函数
        build_modules_func: 构建actor/critic的函数
        train_func: 训练函数
        run_episode_func: 运行episode的函数（用于基线对比）
        logger: 日志记录器
        num_envs: 并行环境数量
        env_backend: 环境后端类型
        num_episodes: 评估的episode数量
        seed: 随机种子

    Returns:
        包含所有结果的字典
    """
    # 1. 训练
    train_result = train_scenario(
        scenario_name=scenario_name,
        sim_config=sim_config,
        env_config=env_config,
        train_config=train_config,
        build_env_func=build_env_func,
        build_modules_func=build_modules_func,
        train_func=train_func,
        logger=logger,
        num_envs=num_envs,
        env_backend=env_backend,
        seed=seed
    )

    # 2. 评估PPO
    ppo_eval_result = evaluate_scenario_ppo(
        scenario_name=scenario_name,
        scenario_env_config=train_result['env_config'],
        actor=train_result['actor'],
        critic=train_result['critic'],
        rl_env=train_result['rl_env'],
        best_model=train_result['best_model'],
        last_model=train_result['last_model'],
        num_episodes=num_episodes,
        seed=seed
    )

    # 3. 评估Baseline
    baseline_eval_result = evaluate_scenario_baseline(
        scenario_name=scenario_name,
        scenario_env_config=train_result['env_config'],
        run_episode_func=run_episode_func,
        num_episodes=num_episodes,
        seed=seed
    )

    # 4. 合并所有结果
    scenario_result = {
        'scenario_name': scenario_name,
        'env_config': train_result['env_config'],
        'sim_config': train_result['sim_config'],
        'best_model': train_result['best_model'],
        'last_model': train_result['last_model'],
        'eval_results_best': ppo_eval_result['eval_results_best'],
        'eval_results_last': ppo_eval_result['eval_results_last'],
        'baseline_results': baseline_eval_result['baseline_results'],
    }

    print(f"\n场景 {scenario_name} 完成!")
    print(f"  PPO最佳奖励: {ppo_eval_result['eval_results_best']['eval_reward_mean']:.4f}")
    print(f"  PPO最后奖励: {ppo_eval_result['eval_results_last']['eval_reward_mean']:.4f}")
    print(f"  固定分配奖励: {baseline_eval_result['baseline_results']['固定分配']['reward']:.4f}")
    print(f"  启发式算法奖励: {baseline_eval_result['baseline_results']['启发式算法']['reward']:.4f}")

    return scenario_result


def print_results_summary(all_scenarios_results: OrderedDict) -> None:
    """打印所有场景的结果摘要

    Args:
        all_scenarios_results: 包含所有场景结果的OrderedDict
    """
    print("\n结果摘要:")
    for scenario_name, result in all_scenarios_results.items():
        ppo_reward_best = result['eval_results_best']['eval_reward_mean']
        ppo_reward_last = result['eval_results_last']['eval_reward_mean']
        fixed_reward = result['baseline_results']['固定分配']['reward']
        heuristic_reward = result['baseline_results']['启发式算法']['reward']

        print(f"\n{scenario_name}:")
        print(f"  PPO(最佳): {ppo_reward_best:.4f}")
        print(f"  PPO(最后): {ppo_reward_last:.4f}")
        print(f"  固定分配: {fixed_reward:.4f} (vs最佳: {(ppo_reward_best/fixed_reward-1)*100:+.2f}%)")
        print(f"  启发式: {heuristic_reward:.4f} (vs最佳: {(ppo_reward_best/heuristic_reward-1)*100:+.2f}%)")
