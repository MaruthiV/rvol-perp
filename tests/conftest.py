"""Shared fixtures for rvol test suite."""

import numpy as np
import pandas as pd
import pytest


def _utc_minute_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="1min", tz="UTC")


@pytest.fixture
def flat_price_series() -> pd.Series:
    """Constant price = 50000 for 1441 minutes. RV must equal exactly 0."""
    idx = _utc_minute_index("2024-01-01", 1441)
    return pd.Series(50000.0, index=idx, name="price")


@pytest.fixture
def gbm_price_series() -> pd.Series:
    """Geometric Brownian Motion with σ = 0.001 per minute (~45% annualized vol).

    Seeded for reproducibility. 30 days + 1 for differencing = 43,201 prices.
    True per-window variance (30d): E[RV_30] = 43200 × σ² = 0.0432.
    """
    rng = np.random.default_rng(42)
    n = 43201
    log_returns = rng.normal(0.0, 0.001, n - 1)
    log_prices = np.concatenate([[np.log(50000.0)], np.log(50000.0) + np.cumsum(log_returns)])
    prices = np.exp(log_prices)
    idx = _utc_minute_index("2024-01-01", n)
    return pd.Series(prices, index=idx, name="price")


@pytest.fixture
def spike_price_series() -> pd.Series:
    """Normal GBM returns with 3 artificial 10σ spikes injected.

    Used to verify that capping reduces spike contributions.
    """
    rng = np.random.default_rng(99)
    n = 5000
    log_returns = rng.normal(0.0, 0.001, n - 1)
    # Inject 3 spikes: return ≈ ±10σ = ±0.01
    spike_idx = [1000, 2500, 4000]
    log_returns[spike_idx[0]] = 0.01    # +10σ
    log_returns[spike_idx[1]] = -0.01   # -10σ
    log_returns[spike_idx[2]] = 0.015   # +15σ
    log_prices = np.concatenate([[np.log(50000.0)], np.log(50000.0) + np.cumsum(log_returns)])
    idx = _utc_minute_index("2024-01-01", n)
    return pd.Series(np.exp(log_prices), index=idx, name="price")


@pytest.fixture
def gapped_price_series() -> pd.Series:
    """Normal price series with a 2-hour gap (120 minutes) in the middle.

    Used to test gap detection and is_valid=False propagation.
    First 1000 minutes normal, then skip 120 minutes, then 1000 more.
    """
    rng = np.random.default_rng(7)
    n1, n2 = 1001, 1001
    log_returns_1 = rng.normal(0.0, 0.001, n1 - 1)
    log_returns_2 = rng.normal(0.0, 0.001, n2 - 1)

    # First segment: normal 1-minute spacing
    idx1 = _utc_minute_index("2024-01-01 00:00", n1)
    prices_1 = 50000.0 * np.exp(
        np.concatenate([[0.0], np.cumsum(log_returns_1)])
    )

    # Second segment: starts 120 minutes after the last price of segment 1
    gap_start = idx1[-1] + pd.Timedelta(minutes=121)
    idx2 = pd.date_range(gap_start, periods=n2, freq="1min", tz="UTC")
    prices_2 = prices_1[-1] * np.exp(
        np.concatenate([[0.0], np.cumsum(log_returns_2)])
    )

    full_idx = idx1.append(idx2)
    full_prices = np.concatenate([prices_1, prices_2])
    return pd.Series(full_prices, index=full_idx, name="price")


@pytest.fixture
def synthetic_trade_df() -> "pd.DataFrame":
    """Small synthetic trade DataFrame matching TRADE_SCHEMA."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 500
    base_ts = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1_000_000)
    timestamps = base_ts + np.cumsum(rng.integers(1000, 5000, n))
    prices = 50000.0 + rng.normal(0, 50, n).cumsum()
    sizes = rng.uniform(0.01, 2.0, n)
    sides = rng.choice(["buy", "sell"], n)
    return pd.DataFrame(
        {
            "timestamp_us": timestamps.astype(np.int64),
            "price": prices,
            "size": sizes,
            "side": sides,
        }
    )
