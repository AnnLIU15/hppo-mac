"""随机流量与驻留时间工具模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True)
class CoxProcessConfig:
    intensity_mean: float
    intensity_variance: float
    grid_shape: Tuple[int, int]
    region_bounds: Tuple[float, float, float, float]


def generate_cox_points(cfg: CoxProcessConfig, rng: Generator) -> np.ndarray:
    """生成 Cox 点过程样本。"""

    min_x, max_x, min_y, max_y = cfg.region_bounds
    gx, gy = cfg.grid_shape

    base_intensity = cfg.intensity_mean
    variance = max(cfg.intensity_variance, 1e-6)
    cell_width = (max_x - min_x) / gx
    cell_height = (max_y - min_y) / gy

    intensities = rng.lognormal(mean=base_intensity, sigma=np.sqrt(variance), size=(gx, gy))
    intensities = intensities / intensities.sum()

    points = []
    for ix in range(gx):
        for iy in range(gy):
            lam = intensities[ix, iy] * base_intensity * gx * gy
            count = rng.poisson(lam)
            if count == 0:
                continue

            x0 = min_x + ix * cell_width
            y0 = min_y + iy * cell_height
            xs = rng.uniform(x0, x0 + cell_width, size=count)
            ys = rng.uniform(y0, y0 + cell_height, size=count)
            cell_points = np.stack([xs, ys], axis=-1)
            points.append(cell_points)

    if not points:
        return np.empty((0, 2), dtype=np.float32)

    return np.concatenate(points, axis=0).astype(np.float32)


@dataclass(frozen=True)
class BatchedPoissonConfig:
    rate: float
    batch_mean: float
    batch_std: float = 1.0


def batched_poisson_arrival(cfg: BatchedPoissonConfig, steps: int, rng: Generator) -> np.ndarray:
    """批次泊松到达过程。"""

    lam = max(cfg.rate, 1e-6)
    batches = rng.poisson(lam=lam, size=steps)
    sizes = rng.lognormal(mean=np.log(max(cfg.batch_mean, 1e-3)), sigma=cfg.batch_std, size=steps)
    arrivals = (batches * sizes).astype(np.float32)
    return arrivals


def sample_residence_time(mean_time: float, rng: Generator) -> float:
    """驻留时间采样，指数分布占位。"""

    return float(rng.exponential(scale=max(mean_time, 1e-3)))
