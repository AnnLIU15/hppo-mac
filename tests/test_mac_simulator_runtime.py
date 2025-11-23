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

    combo_mask = simulator.compute_combo_mask([(0, 0), (1, -1), (-1, 1)])
    assert combo_mask.shape == (3,)
    assert np.all((combo_mask == 0.0) | (combo_mask == 1.0))


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