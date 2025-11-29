import numpy as np
import torch

from algo.hppo import build_hppo_modules
from env.satellite_mac_env import SatelliteMACEnv
from env.gym_helpers import build_gym_env


def _make_env() -> SatelliteMACEnv:
    return SatelliteMACEnv()


def test_parse_action_accepts_index_and_one_hot():
    env = _make_env()
    index_action = {
        "delta_cbra": np.array(env.encode_delta_index(1), dtype=np.int64),
        "delta_pbra": np.array(env.encode_delta_index(-1), dtype=np.int64),
        "q_ACB": np.array([0.75], dtype=np.float32),
    }
    parsed_index = env._parse_action(index_action)

    branch_bins = 2 * env._delta_range + 1
    one_hot_action = {
        "delta_cbra": np.eye(branch_bins, dtype=np.float32)[env.encode_delta_index(1)],
        "delta_pbra": np.eye(branch_bins, dtype=np.float32)[env.encode_delta_index(-1)],
        "q_ACB": np.array([0.75], dtype=np.float32),
    }
    parsed_one_hot = env._parse_action(one_hot_action)

    assert parsed_index == parsed_one_hot


def test_gym_wrapper_random_step_runs():
    base_env = _make_env()
    env = build_gym_env(base_env.config)

    td = env.reset()
    rollout = env.rand_step(td)

    assert "next" in rollout
    assert rollout["next", "observation"].shape[-1] == base_env.observation_space.shape[0]


def test_action_valid_flag_available_in_info():
    env = _make_env()
    _, info = env.reset()
    assert "action_valid" in info
    assert info["action_valid"] == 1.0


def test_hppo_policy_param_shapes_follow_action_spec():
    base_env = _make_env()
    env = build_gym_env(base_env.config)

    actor, _ = build_hppo_modules(env)
    td = env.reset()
    params_td = actor(td.clone())

    logits_cbra = params_td["params", "delta_cbra", "logits"]
    logits_pbra = params_td["params", "delta_pbra", "logits"]

    assert logits_cbra.shape[-1] == base_env.action_space["delta_cbra"].n
    assert logits_pbra.shape[-1] == base_env.action_space["delta_pbra"].n
    assert torch.all(logits_cbra.isfinite())
    assert torch.all(logits_pbra.isfinite())
