"""实用工具模块。"""

from .traffic import (
    CoxProcessConfig,
    BatchedPoissonConfig,
    batched_poisson_arrival,
    generate_cox_points,
    sample_residence_time,
)

__all__ = [
    "CoxProcessConfig",
    "BatchedPoissonConfig",
    "batched_poisson_arrival",
    "generate_cox_points",
    "sample_residence_time",
]
