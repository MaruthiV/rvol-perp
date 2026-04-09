"""Clamped-linear funding rate for the variance perp.

Sign convention: positive rate = longs pay shorts (mark > index → penalize longs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FundingParams:
    """Hourly funding parameters.

    Attributes:
        dampening: dimensionless divisor k in f = clip(basis / k, -cap, +cap).
            Larger k → gentler per-hour funding response to a given basis.
        cap: per-interval hard cap on |f|. Expressed as a rate per interval
            (e.g. 0.005 = 0.5% per hour).
        interval_hours: funding payment interval. Defaults to 1h to match HL.
    """

    dampening: float = 333.0
    cap: float = 0.005
    interval_hours: float = 1.0

    def __post_init__(self) -> None:
        if self.dampening <= 0:
            raise ValueError(f"dampening must be positive, got {self.dampening}")
        if self.cap <= 0:
            raise ValueError(f"cap must be positive, got {self.cap}")
        if self.interval_hours <= 0:
            raise ValueError(f"interval_hours must be positive, got {self.interval_hours}")


def funding_rate(mark: float, index: float, params: FundingParams) -> float:
    """Per-interval funding rate given current mark and index.

    f = clip((mark - index) / index / k, -cap, +cap)

    Returns 0.0 if index <= 0 (degenerate / uninitialized index).
    """
    if index <= 0 or not np.isfinite(index) or not np.isfinite(mark):
        return 0.0
    basis = (mark - index) / index
    raw = basis / params.dampening
    return float(np.clip(raw, -params.cap, params.cap))


def funding_rate_series(
    mark: pd.Series, index: pd.Series, params: FundingParams
) -> pd.Series:
    """Vectorized funding rate over aligned mark and index series.

    Inputs must share an index (or will be inner-joined). Returns a series on
    the common index with dtype float64. Non-finite or non-positive index
    values produce 0.0 funding at that timestamp.
    """
    common = mark.index.intersection(index.index)
    m = mark.loc[common].to_numpy(dtype=np.float64)
    i = index.loc[common].to_numpy(dtype=np.float64)
    valid = np.isfinite(m) & np.isfinite(i) & (i > 0)
    basis = np.zeros_like(m)
    np.divide(m - i, i, out=basis, where=valid)
    raw = basis / params.dampening
    clipped = np.clip(raw, -params.cap, params.cap)
    clipped[~valid] = 0.0
    return pd.Series(clipped, index=common, name="funding_rate")
