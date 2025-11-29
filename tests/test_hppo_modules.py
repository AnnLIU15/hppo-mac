import numpy as np
import torch

from algo.hppo import build_hppo_modules
from env.satellite_mac_env import SatelliteMACEnv, SatelliteMACEnvConfig
from env.gym_helpers import build_gym_env


def _make_env(delta_range: int = 1):
    cfg = SatelliteMACEnvConfig(preamble_delta_range=delta_range, flatten_observation=True)
    return build_gym_env(cfg)


def test_policy_head_matches_action_bins():
    delta_range = 2
    env = _make_env(delta_range)
    td = env.reset()

    actor, critic = build_hppo_modules(env)

    out_td = actor(td.clone())
    logits_cbra = out_td["params", "delta_cbra", "logits"]
    logits_pbra = out_td["params", "delta_pbra", "logits"]

    expected_bins = 2 * delta_range + 1
    assert logits_cbra.shape[-1] == expected_bins
    assert logits_pbra.shape[-1] == expected_bins

    sample_cbra = out_td["delta_cbra"]
    sample_pbra = out_td["delta_pbra"]
    assert torch.isclose(sample_cbra.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(sample_pbra.sum(), torch.tensor(1.0), atol=1e-5)

    critic_td = critic(td.clone())
    assert critic_td["state_value"].shape == torch.Size([1])


def test_action_decoding_accepts_one_hot():
    delta_range = 3
    env = SatelliteMACEnv(SatelliteMACEnvConfig(preamble_delta_range=delta_range))

    branch_bins = 2 * delta_range + 1
    idx = branch_bins - 2
    one_hot = np.eye(branch_bins, dtype=np.float32)[idx]

    decoded_from_one_hot = env._decode_delta_component(one_hot)
    decoded_from_index = env._decode_delta_component(np.array(env.encode_delta_index(-2), dtype=np.int64))

    expected_delta = float(env._delta_values[idx])
    assert decoded_from_one_hot == expected_delta
    assert decoded_from_index == -2.0


def test_step_with_one_hot_actions():
    delta_range = 2
    env = _make_env(delta_range)
    td = env.reset()

    branch_bins = 2 * delta_range + 1
    action_td = td.clone()
    action_td.set("delta_cbra", torch.from_numpy(np.eye(branch_bins, dtype=np.float32)[3]))
    action_td.set("delta_pbra", torch.from_numpy(np.eye(branch_bins, dtype=np.float32)[1]))
    action_td.set("q_ACB", torch.tensor([0.5], dtype=torch.float32))

    next_td = env.step(action_td)

    assert ("next", "observation") in next_td.keys(include_nested=True)
    assert ("next", "reward") in next_td.keys(include_nested=True)
    assert ("next", "done") in next_td.keys(include_nested=True)


def test_invalid_branch_action_is_flagged():
    config = SatelliteMACEnvConfig(preamble_delta_range=10)
    env = SatelliteMACEnv(config)

    before = env.simulator._preamble_allocation.copy()

    invalid_action = {
        "delta_cbra": np.array(env.encode_delta_index(10), dtype=np.int64),
        "delta_pbra": np.array(env.encode_delta_index(10), dtype=np.int64),
        "q_ACB": np.array([1.0], dtype=np.float32),
    }

    _, _, _, _, info = env.step(invalid_action)
    assert info.get("action_valid") == 0.0
    assert np.array_equal(env.simulator._preamble_allocation, before)
