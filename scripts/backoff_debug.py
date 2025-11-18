import json
import sys
from pathlib import Path


def _import_mac_simulator():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from env.mac_simulator import MACSimulator, default_simulator_config

    return MACSimulator, default_simulator_config

if __name__ == "__main__":
    MACSimulator, default_simulator_config = _import_mac_simulator()
    sim = MACSimulator(default_simulator_config())
    sim.initialize_state(seed=0)
    for _ in range(100):
        params = {"M_CBRA": 0.0, "M_PBRA": 0.0, "q_ACB": 1}
        reward, next_state, info = sim.run_slots(num_slots=160, params=params)
    print(json.dumps({
        "reward": reward,
        "success_total": float(info["success_total"][0]),
        "collision_total": float(info["collision_total"][0]),
        "pending_backoff_cbra": float(info["pending_backoff_cbra"][0]),
        "pending_backoff_pbra": float(info["pending_backoff_pbra"][0]),
        "success_cbra": float(info["success_cbra"][0]),
        "success_pbra": float(info["success_pbra"][0]),
        "success_cfra": float(info["success_cfra"][0]),
        "collision_cbra": float(info["collision_cbra"][0]),
        "collision_pbra": float(info["collision_pbra"][0]),
        "collision_cfra": float(info["collision_cfra"][0]),
    }, indent=2))
