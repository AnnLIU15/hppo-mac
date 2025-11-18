import unittest

import numpy as np

from env.mac_simulator import (
    BackoffWindow,
    MACSimulator,
    default_simulator_config,
)


class BackoffStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        config = default_simulator_config()
        self.simulator = MACSimulator(config, rng=np.random.default_rng(42))
        self.simulator.initialize_state(seed=123)

    def test_schedule_backoff_within_window(self) -> None:
        window = self.simulator.config.backoff_strategy.high
        amount = 20.0
        self.simulator._schedule_backoff(  # type: ignore[attr-defined]
            self.simulator._backoff_queue_cbra,  # type: ignore[attr-defined]
            amount,
            window,
        )
        indices = np.nonzero(self.simulator._backoff_queue_cbra)[0]  # type: ignore[attr-defined]
        self.assertTrue(np.all(indices >= window.min_steps))
        self.assertTrue(np.all(indices <= window.max_steps))
        self.assertAlmostEqual(
            float(np.sum(self.simulator._backoff_queue_cbra)),  # type: ignore[attr-defined]
            amount,
            places=5,
        )

    def test_pop_backoff_queue_releases_due_amount(self) -> None:
        queue = self.simulator._backoff_queue_pbra  # type: ignore[attr-defined]
        queue.fill(0.0)
        fixed_window = BackoffWindow(3, 3)
        self.simulator._schedule_backoff(queue, 5.0, fixed_window)  # type: ignore[attr-defined]
        released_first = self.simulator._pop_backoff_queue(queue, 2)  # type: ignore[attr-defined]
        self.assertAlmostEqual(released_first, 0.0, places=5)
        released_second = self.simulator._pop_backoff_queue(queue, 1)  # type: ignore[attr-defined]
        self.assertAlmostEqual(released_second, 5.0, places=5)
        self.assertAlmostEqual(float(queue.sum()), 0.0, places=5)

    def test_select_backoff_window_thresholds(self) -> None:
        strategy = self.simulator.config.backoff_strategy
        low_window = self.simulator._select_backoff_window(0.0)  # type: ignore[attr-defined]
        self.assertEqual(low_window, strategy.low)
        medium_window = self.simulator._select_backoff_window(  # type: ignore[attr-defined]
            strategy.collision_threshold_medium
        )
        self.assertEqual(medium_window, strategy.medium)
        high_window = self.simulator._select_backoff_window(  # type: ignore[attr-defined]
            strategy.collision_threshold_high + 1e-3
        )
        self.assertEqual(high_window, strategy.high)


if __name__ == "__main__":
    unittest.main()
