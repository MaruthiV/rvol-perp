"""Tests for rolling_realized_variance and point_in_time_rv."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rvol.index.spec import IndexSpec, SPEC_30D
from rvol.index.returns import log_returns_from_prices
from rvol.index.variance import rolling_realized_variance, point_in_time_rv


# --- point_in_time_rv ---

def test_rv_zero_for_flat_prices(flat_price_series):
    rets = log_returns_from_prices(flat_price_series, handle_gaps=False)
    spec = IndexSpec(window_days=1, min_obs_fraction=0.5)
    result = point_in_time_rv(rets.values, spec)
    assert result["rv"] == 0.0
    assert result["rv_annualized"] == 0.0


def test_rv_nonnegative_basic():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.001, 1000)
    spec = IndexSpec(window_days=1, min_obs_fraction=0.5)
    result = point_in_time_rv(returns, spec)
    assert result["rv"] >= 0.0


def test_rv_annualized_formula():
    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.001, 1440)
    spec = IndexSpec(window_days=1, annualization_days=365)
    result = point_in_time_rv(returns, spec)
    expected_annualized = result["rv"] * (365 / 1)
    assert np.isclose(result["rv_annualized"], expected_annualized)


def test_rv_converges_to_true_variance_gbm(gbm_price_series):
    """RV should converge to the true variance under GBM within 5%.

    True per-30d variance = 43200 × σ² = 43200 × 1e-6 = 0.0432.
    """
    rets = log_returns_from_prices(gbm_price_series, handle_gaps=False)
    spec = IndexSpec(window_days=30, min_obs_fraction=0.5)
    result = point_in_time_rv(rets.values, spec)
    true_variance = 43200 * (0.001**2)  # 0.0432
    rel_error = abs(result["rv"] - true_variance) / true_variance
    assert rel_error < 0.05, (
        f"RV {result['rv']:.6f} too far from true variance {true_variance:.6f} "
        f"(relative error: {rel_error:.3f})"
    )


def test_n_obs_counts_non_nan():
    rng = np.random.default_rng(8)
    returns = rng.normal(0, 0.001, 500)
    returns[100] = np.nan
    returns[200] = np.nan
    spec = IndexSpec(window_days=1, min_obs_fraction=0.1)
    result = point_in_time_rv(returns, spec)
    assert result["n_obs"] == 498  # 500 - 2 NaNs


def test_is_valid_below_threshold():
    spec = IndexSpec(window_days=1, min_obs_fraction=0.9)
    # Only 100 observations, expected ~1440 — should be invalid
    returns = np.random.default_rng(9).normal(0, 0.001, 100)
    result = point_in_time_rv(returns, spec)
    assert not result["is_valid"]


def test_capped_count_reported():
    rng = np.random.default_rng(10)
    returns = rng.normal(0, 0.001, 2000)
    returns[1000] = 0.05  # large spike — will be capped
    spec = IndexSpec(window_days=1, min_obs_fraction=0.1)
    result = point_in_time_rv(returns, spec)
    assert result["n_capped"] >= 1


# --- rolling_realized_variance ---

def test_rolling_rv_returns_dataframe(flat_price_series):
    rets = log_returns_from_prices(flat_price_series, handle_gaps=False)
    spec = IndexSpec(window_days=1)
    df = rolling_realized_variance(rets, spec)
    assert isinstance(df, pd.DataFrame)
    for col in ["rv", "rv_annualized", "n_obs", "n_capped", "is_valid"]:
        assert col in df.columns


def test_rolling_rv_same_length_as_input(flat_price_series):
    rets = log_returns_from_prices(flat_price_series, handle_gaps=False)
    spec = IndexSpec(window_days=1)
    df = rolling_realized_variance(rets, spec)
    assert len(df) == len(rets)


def test_rolling_rv_zero_for_flat(flat_price_series):
    rets = log_returns_from_prices(flat_price_series, handle_gaps=False)
    spec = IndexSpec(window_days=1, min_obs_fraction=0.1)
    df = rolling_realized_variance(rets, spec)
    assert np.allclose(df["rv"].values, 0.0)


def test_rolling_rv_nonnegative(gbm_price_series):
    rets = log_returns_from_prices(gbm_price_series, handle_gaps=False)
    df = rolling_realized_variance(rets, SPEC_30D)
    assert (df["rv"] >= 0).all()


def test_rolling_rv_is_valid_flag(gapped_price_series):
    """After the 2-hour gap, windows that cross it should have is_valid=False."""
    rets = log_returns_from_prices(gapped_price_series, freq="1min", handle_gaps=True)
    spec = IndexSpec(window_days=1, min_obs_fraction=0.9)
    df = rolling_realized_variance(rets, spec)
    # At least some rows must be invalid (the ones near the gap)
    assert not df["is_valid"].all(), "expected some invalid rows around the gap"


def test_rolling_rv_spike_capping(spike_price_series):
    """With capping enabled, spikes should be dampened."""
    rets_raw = log_returns_from_prices(spike_price_series, handle_gaps=False)

    spec_normal = IndexSpec(window_days=1, sigma_cap=4.0, min_obs_fraction=0.1)
    spec_nocap = IndexSpec(window_days=1, sigma_cap=1000.0, min_obs_fraction=0.1)

    df_capped = rolling_realized_variance(rets_raw, spec_normal)
    df_nocap = rolling_realized_variance(rets_raw, spec_nocap)

    # Total RV should be lower with capping
    assert df_capped["rv"].sum() <= df_nocap["rv"].sum()


def test_rolling_rv_empty_input():
    empty = pd.Series(dtype=float, name="log_return")
    spec = IndexSpec(window_days=1)
    df = rolling_realized_variance(empty, spec)
    assert len(df) == 0


def test_rolling_rv_annualized_scaling(gbm_price_series):
    rets = log_returns_from_prices(gbm_price_series, handle_gaps=False)
    spec = IndexSpec(window_days=30, annualization_days=365)
    df = rolling_realized_variance(rets, spec)
    expected_scale = 365 / 30
    valid = df["rv"] > 0
    ratios = df.loc[valid, "rv_annualized"] / df.loc[valid, "rv"]
    assert np.allclose(ratios, expected_scale)


# --- Hypothesis property tests ---

@given(
    returns=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=2000,
    )
)
@settings(max_examples=300)
def test_rv_nonnegative_arbitrary_inputs(returns):
    arr = np.array(returns)
    spec = IndexSpec(window_days=1, min_obs_fraction=0.01)
    result = point_in_time_rv(arr, spec)
    assert result["rv"] >= 0.0, f"rv was negative: {result['rv']}"
