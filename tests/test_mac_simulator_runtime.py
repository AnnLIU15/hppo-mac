import numpy as np

from env.mac_simulator import MACSimulator, default_simulator_config


def test_run_slots_returns_expected_payload():
    config = default_simulator_config()
    simulator = MACSimulator(config, rng=np.random.default_rng(20251122))

    state = simulator.initialize_state(seed=7)
    assert "requests_cbra" in state
    assert "region_mixture" in state
    assert state["region_mixture"].shape[0] == len(config.regions)

    params = {"M_CBRA": 0.0, "M_PBRA": 0.0, "q_ACB": 0.6}
    reward, next_state, info = simulator.run_slots(num_slots=800, params=params)

    assert np.isfinite(reward)
    assert "requests_cbra" in next_state
    assert next_state["requests_cbra"].shape == (1,)
    assert "region_mixture" in info
    assert np.isclose(float(np.sum(info["region_mixture"])), 1.0, atol=1e-3)
    assert info["success_total"].shape == (1,)
    assert info["collision_total"].shape == (1,)

    before_allocation = simulator._preamble_allocation.copy()  # type: ignore[attr-defined]
    reward_invalid, _, invalid_info = simulator.run_slots(
        num_slots=40,
        params={"M_CBRA": float(config.total_preambles), "M_PBRA": 0.0, "q_ACB": 0.4},
    )

    assert np.isfinite(reward_invalid)
    action_valid = invalid_info.get("action_valid")
    assert isinstance(action_valid, np.ndarray)
    assert action_valid.shape == (1,)
    assert np.isclose(action_valid.item(), 0.0)
    assert np.array_equal(simulator._preamble_allocation, before_allocation)  # type: ignore[attr-defined]


# def test_run_slots_updates_backoff_queues():
#     config = default_simulator_config()
#     simulator = MACSimulator(config, rng=np.random.default_rng(20251123))
#     simulator.initialize_state(seed=13)

#     params = {"M_CBRA": 0.0, "M_PBRA": 0.0, "q_ACB": 0.3}
#     simulator.run_slots(num_slots=20, params=params)

#     backoff_cbra = float(np.sum(simulator._backoff_queue_cbra))  # type: ignore[attr-defined]
#     backoff_pbra = float(np.sum(simulator._backoff_queue_pbra))  # type: ignore[attr-defined]
#     assert backoff_cbra >= 0.0
#     assert backoff_pbra >= 0.0

if __name__ == "__main__":
    test_run_slots_returns_expected_payload()
    # test_run_slots_updates_backoff_queues()
    print("All tests passed.")