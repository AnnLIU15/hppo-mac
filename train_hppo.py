"""Entry-point script to launch H-PPO training with TorchRL."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from loguru import logger

from torchrl.envs import ParallelEnv, SerialEnv, TransformedEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import Compose, DoubleToFloat

from algo.hppo import HPPOConfig, build_hppo_modules, train_hppo
from env.satellite_mac_env import SatelliteMACEnvConfig
from env.gym_helpers import build_gym_env


def _wrap_env(config: SatelliteMACEnvConfig, seed: Optional[int]) -> GymWrapper:
    env = build_gym_env(config)
    env.set_seed(seed)
    return env


def _make_env(
    config: SatelliteMACEnvConfig,
    device: torch.device,
    seed: Optional[int],
    num_envs: int,
    backend: str,
) -> TransformedEnv:
    if num_envs <= 1:
        base_env = _wrap_env(config, seed)
    else:
        def _factory(rank: Optional[int] = None) -> GymWrapper:
            seed_offset = rank if rank is not None else 0
            env_seed = seed + seed_offset if seed is not None else None
            return _wrap_env(config, env_seed)

        if backend == "parallel":
            base_env = ParallelEnv(num_envs, _factory)
        else:
            base_env = SerialEnv(num_envs, _factory, auto_reset=True)

    transforms = Compose(DoubleToFloat())
    env = TransformedEnv(base_env, transforms)
    env.set_seed(seed)
    env.to(device)
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the H-PPO agent on the Satellite MAC environment.")
    parser.add_argument("--frames-per-batch", type=int, default=512, help="Number of frames collected per update batch.")
    parser.add_argument("--mini-batch-size", type=int, default=64, help="Mini-batch size used during PPO optimization.")
    parser.add_argument("--rollout-epochs", type=int, default=2, help="Number of PPO epochs over each collected batch.")
    parser.add_argument("--max-iterations", type=int, default=4, help="Maximum number of PPO iterations to run.")
    parser.add_argument("--feature-dim", type=int, default=128, help="Hidden feature size for the actor/critic encoders.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to run the training on (e.g. cpu or cuda:0).")
    parser.add_argument(
        "--delta-range",
        type=int,
        default=3,
        help="Range of CBRA/PBRA integer deltas for each discrete branch (2*range+1 bins per head).",
    )
    parser.add_argument("--decision-horizon", type=int, default=128, help="Maximum number of steps per episode.")
    parser.add_argument(
        "--num-slots-per-step",
        type=int,
        default=1,
        help="Number of MAC slots simulated for each environment step.",
    )
    parser.add_argument("--history-length", type=int, default=8, help="Rolling window length for aggregated statistics.")
    parser.add_argument("--seed", type=int, default=20251106, help="Random seed applied to Torch and the environment.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environment replicas used for parallel data collection.")
    parser.add_argument(
        "--env-backend",
        choices=["serial", "parallel"],
        default="serial",
        help="Vectorized environment backend to employ for data collection.",
    )
    parser.add_argument(
        "--parallel-collection",
        action="store_true",
        help="Shortcut to enable ParallelEnv-based multiprocess data collection.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/hppo_training.log",
        help="Path to the log file recorded via Loguru. Pass an empty string to disable file logging.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="logs/checkpoints",
        help="Directory used to persist trained actor/critic checkpoints. Pass an empty string to disable checkpointing.",
    )
    return parser.parse_args()


def _build_configs(args: argparse.Namespace) -> Tuple[SatelliteMACEnvConfig, HPPOConfig]:
    env_config = SatelliteMACEnvConfig(
        num_slots_per_step=args.num_slots_per_step,
        decision_horizon=args.decision_horizon,
        history_len=args.history_length,
        preamble_delta_range=args.delta_range,
        flatten_observation=True,
    )

    device = torch.device(args.device)

    train_config = HPPOConfig(
        frames_per_batch=args.frames_per_batch,
        mini_batch_size=args.mini_batch_size,
        rollout_epochs=args.rollout_epochs,
        max_iterations=args.max_iterations,
        device=device,
    )

    return env_config, train_config


def _configure_logging(log_path: Optional[Path]) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level="INFO", enqueue=True, rotation="10 MB")


def _build_logger_callback() -> Callable[..., None]:
    def _log_metrics(*, iteration: int, reward: float, **metrics: float) -> None:
        components = [f"iter={iteration:04d}", f"reward={reward:.4f}"]
        for key, value in sorted(metrics.items()):
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            components.append(f"{key}={scalar:.4f}")
        logger.info(" ".join(components))

    return _log_metrics


def main() -> None:
    args = _parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    log_path = Path(args.log_file) if args.log_file else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    _configure_logging(log_path)

    env_config, train_config = _build_configs(args)
    device = train_config.device or torch.device("cpu")

    num_envs = max(1, int(args.num_envs))
    env_backend = args.env_backend
    if args.parallel_collection:
        env_backend = "parallel"

    if train_config.frames_per_batch % num_envs != 0:
        logger.warning(
            "frames_per_batch (%d) is not divisible by num_envs (%d); collector will pad the last mini-batch.",
            train_config.frames_per_batch,
            num_envs,
        )

    logger.info(f"Using {env_backend} env backend with {num_envs} replicas")
    env = _make_env(env_config, device, args.seed, num_envs, env_backend)
    actor, critic = build_hppo_modules(env, feature_dim=args.feature_dim)

    logger.info("Starting H-PPO training")
    metrics = train_hppo(env, actor, critic, train_config, logger_fn=_build_logger_callback())
    if metrics:
        pretty = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        logger.info("Final PPO metrics: {}", pretty)
    else:
        logger.warning("Training finished without returned metrics.")

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        actor_path = checkpoint_dir / "hppo_actor.pt"
        critic_path = checkpoint_dir / "hppo_critic.pt"

        shared_metadata = {
            "train_config": asdict(train_config),
            "env_config": asdict(env_config),
            "num_envs": num_envs,
            "env_backend": env_backend,
            "parallel_collection": bool(args.parallel_collection),
        }
        torch.save({"state_dict": actor.state_dict(), **shared_metadata}, actor_path)
        torch.save({"state_dict": critic.state_dict(), **shared_metadata}, critic_path)
        logger.info("Saved actor checkpoint to {}", actor_path)
        logger.info("Saved critic checkpoint to {}", critic_path)


if __name__ == "__main__":
    main()
