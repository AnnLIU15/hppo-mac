"""Evaluation utilities for trained H-PPO agents."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import numpy as np
from torchrl.modules import ProbabilisticActor
from env import SatelliteMACEnv, SatelliteMACEnvConfig


def evaluate_agent(
    actor: ProbabilisticActor,
    env_config: SatelliteMACEnvConfig,
    num_episodes: int = 10,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    评估训练好的agent，收集详细的统计信息。

    Args:
        actor: 训练好的actor网络
        env_config: 环境配置
        num_episodes: 运行的episode数量
        seed: 随机种子
        device: 运行设备

    Returns:
        包含平均统计信息的字典
    """
    device = device or torch.device("cpu")
    actor.eval()
    actor.to(device)

    # 创建评估环境（不使用TorchRL wrapper，直接使用gym环境）
    env = SatelliteMACEnv(config=env_config)

    all_rewards = []
    all_arrivals_cbra = []
    all_arrivals_pbra = []
    all_arrivals_cfra = []
    all_success_cbra = []
    all_success_pbra = []
    all_success_cfra = []
    all_episode_lengths = []

    for episode_idx in range(num_episodes):
        episode_seed = seed + episode_idx if seed is not None else None
        obs, info = env.reset(seed=episode_seed)

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # 将observation转换为tensor
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            # 使用actor选择动作（deterministic mode）
            with torch.no_grad():
                # 构建TensorDict输入
                from tensordict import TensorDict
                td_input = TensorDict(
                    {"observation": obs_tensor},
                    batch_size=[1],
                    device=device
                )

                # 获取动作
                td_output = actor(td_input)

                # 提取动作并转换为numpy
                action = {
                    "delta_cbra": td_output["delta_cbra"].cpu().numpy()[0],
                    "delta_pbra": td_output["delta_pbra"].cpu().numpy()[0],
                    "q_ACB": td_output["q_ACB"].cpu().numpy()[0],
                }

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        # 收集episode统计信息
        all_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)

        # 从最终的info中提取累计统计
        all_arrivals_cbra.append(float(info.get("total_arrivals_cbra", 0)))
        all_arrivals_pbra.append(float(info.get("total_arrivals_pbra", 0)))
        all_arrivals_cfra.append(float(info.get("total_arrivals_cfra", 0)))
        all_success_cbra.append(float(info.get("total_success_cbra", 0)))
        all_success_pbra.append(float(info.get("total_success_pbra", 0)))
        all_success_cfra.append(float(info.get("total_success_cfra", 0)))

    # 计算平均值
    results = {
        "eval_reward_mean": float(np.mean(all_rewards)),
        "eval_reward_std": float(np.std(all_rewards)),
        "eval_episode_length_mean": float(np.mean(all_episode_lengths)),
        "eval_arrivals_cbra_mean": float(np.mean(all_arrivals_cbra)),
        "eval_arrivals_pbra_mean": float(np.mean(all_arrivals_pbra)),
        "eval_arrivals_cfra_mean": float(np.mean(all_arrivals_cfra)),
        "eval_success_cbra_mean": float(np.mean(all_success_cbra)),
        "eval_success_pbra_mean": float(np.mean(all_success_pbra)),
        "eval_success_cfra_mean": float(np.mean(all_success_cfra)),
    }

    # 计算成功率
    if results["eval_arrivals_cbra_mean"] > 0:
        results["eval_rate_cbra"] = results["eval_success_cbra_mean"] / results["eval_arrivals_cbra_mean"]
    if results["eval_arrivals_pbra_mean"] > 0:
        results["eval_rate_pbra"] = results["eval_success_pbra_mean"] / results["eval_arrivals_pbra_mean"]
    if results["eval_arrivals_cfra_mean"] > 0:
        results["eval_rate_cfra"] = results["eval_success_cfra_mean"] / results["eval_arrivals_cfra_mean"]

    actor.train()
    return results


__all__ = ["evaluate_agent"]
