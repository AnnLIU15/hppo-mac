import unittest

import numpy as np

from env.mac_simulator import (
    CoveragePatch,
    MACSimulator,
    MACSimulatorConfig,
    RegionTrafficProfile,
)


class RegionMixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        raw_patch_weights = np.random.random_sample(3)
        normalized_weights = raw_patch_weights / raw_patch_weights.sum()

        self._config = MACSimulatorConfig(
            regions=(
                RegionTrafficProfile(
                    name="urban",
                    area_weight=0.4,
                    cbra_density=1.2,
                    pbra_density=1.5,
                    handover_intensity=0.8,
                    scheduling_period=160,
                ),
                RegionTrafficProfile(
                    name="suburban",
                    area_weight=0.35,
                    cbra_density=0.8,
                    pbra_density=0.9,
                    handover_intensity=0.5,
                    scheduling_period=200,
                ),
                RegionTrafficProfile(
                    name="rural",
                    area_weight=0.25,
                    cbra_density=0.3,
                    pbra_density=0.4,
                    handover_intensity=0.2,
                    scheduling_period=320,
                ),
            ),
            coverage_patches=(
                CoveragePatch(
                    name="hotspot",
                    center=(0.0, 0.0),
                    radius=0.3,
                    strength=5.0,
                    region_weights=tuple(normalized_weights),
                ),
            ),
            coverage_drift_radius=0.0,
            coverage_cycle_slots=100.0,
            coverage_jitter=0.0,
            coverage_smoothing=0.0,
            coverage_total_footprint=1.0,
        )
        self._simulator = MACSimulator(self._config, rng=np.random.default_rng(1234))
        self._raw_patch_weights = normalized_weights

    def test_region_mixture_is_normalized(self) -> None:
        self._simulator.initialize_state(seed=42)
        mixture = self._simulator._update_region_mixture(num_slots=160)

        self.assertTrue(np.all(mixture >= 0.0))
        self.assertAlmostEqual(float(mixture.sum()), 1.0, places=6)

    def test_region_mixture_matches_expected_weights(self) -> None:
        self._simulator.initialize_state(seed=7)
        mixture = self._simulator._update_region_mixture(num_slots=160)

        reference = self._simulator._coverage_reference
        region_prob = self._simulator._region_prob.astype(np.float64)
        patch_strength = self._config.coverage_patches[0].strength
        expected_raw = reference * region_prob + patch_strength * self._raw_patch_weights
        expected = expected_raw / expected_raw.sum()

        np.testing.assert_allclose(mixture, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
