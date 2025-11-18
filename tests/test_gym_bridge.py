import numpy as np
import torch

from torchrl.envs.libs.gym import GymWrapper

from algo.hppo import build_hppo_modules
from env.satellite_mac_env import SatelliteMACEnv
from env.gym_helpers import with_delta_mask_info


def _make_env() -> SatelliteMACEnv:
    return SatelliteMACEnv()


def test_parse_action_accepts_index_and_one_hot():
    env = _make_env()
    combo_idx = env.encode_delta_combo(1, -1)
    index_action = {
        "delta_combo": np.array(combo_idx, dtype=np.int64),
        "q_ACB": np.array([0.75], dtype=np.float32),
    }
    parsed_index = env._parse_action(index_action)

    combo_bins = env._combo_count
    one_hot_action = {
        "delta_combo": np.eye(combo_bins, dtype=np.float32)[combo_idx],
        "q_ACB": np.array([0.75], dtype=np.float32),
    }
    parsed_one_hot = env._parse_action(one_hot_action)

    assert parsed_index == parsed_one_hot


def test_gym_wrapper_random_step_runs():
    base_env = _make_env()
    env = with_delta_mask_info(GymWrapper(base_env))

    td = env.reset()
    rollout = env.rand_step(td)

    assert "next" in rollout
    assert rollout["next", "observation"].shape[-1] == base_env.observation_space.shape[0]


def test_action_mask_available_in_info():
    env = _make_env()
    obs, info = env.reset()
    mask = info.get("delta_combo_mask")
    assert mask is not None
    assert mask.shape == (env._combo_count,)  # type: ignore[attr-defined]
    assert np.any(mask > 0.5)
    assert np.all((mask == 0.0) | (mask == 1.0))


def test_hppo_policy_param_shapes_follow_action_spec():
    base_env = _make_env()
    env = with_delta_mask_info(GymWrapper(base_env))

    actor, _ = build_hppo_modules(env)
    td = env.reset()
    params_td = actor(td.clone())

    combo_logits = params_td["params", "delta_combo", "logits"]

    assert combo_logits.shape[-1] == base_env.action_space["delta_combo"].n
    assert torch.all(combo_logits.isfinite())
