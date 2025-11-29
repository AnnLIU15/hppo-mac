import torch

from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, DoubleToFloat

from algo.hppo import HPPOConfig, build_hppo_modules, train_hppo
from env.satellite_mac_env import SatelliteMACEnvConfig
from env.gym_helpers import build_gym_env


def _make_training_env(delta_range: int) -> TransformedEnv:
    config = SatelliteMACEnvConfig(preamble_delta_range=delta_range, flatten_observation=True)
    base_env = build_gym_env(config)
    transforms = Compose(DoubleToFloat())
    env = TransformedEnv(base_env, transforms)
    env.set_seed(1234)
    return env


def test_train_hppo_executes_single_iteration():
    torch.manual_seed(1234)

    env = _make_training_env(delta_range=2)
    actor, critic = build_hppo_modules(env, feature_dim=64)

    config = HPPOConfig(
        frames_per_batch=64,
        mini_batch_size=16,
        rollout_epochs=1,
        max_iterations=1,
        device=torch.device("cpu"),
    )

    metrics = train_hppo(env, actor, critic, config)

    assert "loss_objective" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss_objective"]))
