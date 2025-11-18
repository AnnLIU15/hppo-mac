"""TorchRL / Gym wrapper helpers for Satellite MAC envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torchrl.data import Composite, Unbounded
from torchrl.envs.gym_like import BaseInfoDictReader
from torchrl.envs.libs.gym import GymWrapper

from .satellite_mac_env import SatelliteMACEnv, SatelliteMACEnvConfig


@dataclass
class _DeltaComboMaskInfoReader(BaseInfoDictReader):
    """Info-dict reader that exposes the delta combo mask to TorchRL."""

    mask_dim: int

    def __post_init__(self) -> None:
        mask_spec = Unbounded(shape=torch.Size([self.mask_dim]), dtype=torch.float32)
        info_composite = Composite({"delta_combo_mask": mask_spec}, shape=[])
        self._info_spec = Composite({"info": info_composite}, shape=[])

    def __call__(self, info_dict, tensordict):
        mask = info_dict.get("delta_combo_mask") if isinstance(info_dict, dict) else None
        if mask is not None:
            mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
            tensordict.set(("info", "delta_combo_mask"), mask_tensor)
        return tensordict

    @property
    def info_spec(self):
        return self._info_spec


def with_delta_mask_info(wrapper: GymWrapper, *, mask_dim: Optional[int] = None) -> GymWrapper:
    """Registers a reader so TorchRL policies can access the delta combo mask."""

    resolved_dim = mask_dim
    if resolved_dim is None:
        base_env = getattr(wrapper, "_env", None)
        if isinstance(base_env, SatelliteMACEnv):
            resolved_dim = int(base_env.action_space["delta_combo"].n)
        else:
            combo_spec = wrapper.action_spec.get("delta_combo")
            if combo_spec is None or not hasattr(combo_spec, "space"):
                raise ValueError("Cannot infer combo mask dimensionality from the wrapped env.")
            resolved_dim = int(combo_spec.space.n)
    reader = _DeltaComboMaskInfoReader(mask_dim=int(resolved_dim))
    wrapper.set_info_dict_reader(info_dict_reader=reader)
    return wrapper


def build_masked_gym_env(config: Optional[SatelliteMACEnvConfig] = None) -> GymWrapper:
    """Convenience helper returning a GymWrapper with mask info registered."""

    base_env = SatelliteMACEnv(config=config)
    wrapper = GymWrapper(base_env)
    return with_delta_mask_info(wrapper)
