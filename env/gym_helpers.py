"""TorchRL / Gym wrapper helpers for the Satellite MAC environment."""

from __future__ import annotations

from typing import Optional

from torchrl.envs.libs.gym import GymWrapper

from .satellite_mac_env import SatelliteMACEnv, SatelliteMACEnvConfig


def build_gym_env(config: Optional[SatelliteMACEnvConfig] = None) -> GymWrapper:
    """Factory returning a bare GymWrapper around the Satellite MAC environment."""

    base_env = SatelliteMACEnv(config=config)
    return GymWrapper(base_env)


__all__ = ["build_gym_env"]
