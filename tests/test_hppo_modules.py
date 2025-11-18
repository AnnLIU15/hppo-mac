import numpy as np
import torch

from torchrl.envs.libs.gym import GymWrapper

from algo.hppo import build_hppo_modules
from env.satellite_mac_env import SatelliteMACEnv, SatelliteMACEnvConfig
from env.gym_helpers import with_delta_mask_info


def _make_env(delta_range: int = 1) -> GymWrapper:
    cfg = SatelliteMACEnvConfig(preamble_delta_range=delta_range, flatten_observation=True)
    base_env = SatelliteMACEnv(config=cfg)
    return with_delta_mask_info(GymWrapper(base_env))


def test_policy_head_matches_action_bins():
    delta_range = 2  # expect 5 discrete values -> 25 combo bins
    env = _make_env(delta_range)
    td = env.reset()

    actor, critic = build_hppo_modules(env)

    # Ensure the encoder produces logits with the expected dimensionality.
    out_td = actor(td.clone())
    logits_combo = out_td["params", "delta_combo", "logits"]

    expected_bins = (2 * delta_range + 1) ** 2
    assert logits_combo.shape[-1] == expected_bins

    # Actor sample should be a one-hot style vector.
    sample_combo = out_td["delta_combo"]
    assert torch.isclose(sample_combo.sum(), torch.tensor(1.0), atol=1e-5)

    # Critic should output a scalar value estimate.
    critic_td = critic(td.clone())
    assert critic_td["state_value"].shape == torch.Size([1])


def test_masked_logits_penalize_invalid_actions():
    env = _make_env(delta_range=1)
    td = env.reset()
    mask = td["info", "delta_combo_mask"]
    actor, _ = build_hppo_modules(env)
    out_td = actor(td.clone())
    logits_combo = out_td["params", "delta_combo", "logits"]

    invalid = mask < 0.5
    if torch.any(invalid):
        assert torch.all(logits_combo[invalid] < -1e8)


def test_action_decoding_accepts_one_hot():
    delta_range = 3
    env = _make_env(delta_range)
    env.reset()

    combo_bins = env._env._combo_count  # underlying Gym env
    idx = 5
    one_hot = np.eye(combo_bins, dtype=np.float32)[idx]

    delta_cbra, delta_pbra = env._env._decode_combo_component(one_hot)
    expected_pair = env._env._delta_pairs[idx]
    assert (delta_cbra, delta_pbra) == tuple(float(x) for x in expected_pair)


def test_step_with_one_hot_actions():
    delta_range = 2
    env = _make_env(delta_range)
    td = env.reset()

    combo_bins = env._env._combo_count
    action_td = td.clone()
    action_td.set("delta_combo", torch.from_numpy(np.eye(combo_bins, dtype=np.float32)[3]))
    action_td.set("q_ACB", torch.tensor([0.5], dtype=torch.float32))

    next_td = env.step(action_td)

    # Ensure the step returns the standard TorchRL keys
    assert ("next", "observation") in next_td.keys(include_nested=True)
    assert ("next", "reward") in next_td.keys(include_nested=True)
    assert ("next", "done") in next_td.keys(include_nested=True)
