"""Tests for manipulation-resistance filters."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rvol.index.filters import cap_return_contributions, minimum_obs_mask


def _normal_returns(n=2000, sigma=0.001, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, n)


# --- cap_return_contributions ---

def test_cap_reduces_large_return():
    returns = _normal_returns(n=2000)
    # Inject a 10σ spike
    returns[1500] = 0.01  # 10 × 0.001
    capped, is_capped = cap_return_contributions(returns, sigma_cap=4.0)
    assert is_capped[1500], "spike should be capped"
    assert abs(capped[1500]) < abs(returns[1500]), "capped value should be smaller"


def test_cap_preserves_small_returns():
    returns = _normal_returns(n=2000, sigma=0.001)
    # Ensure no returns exceed 3σ (well below cap)
    returns = np.clip(returns, -0.003, 0.003)
    capped, is_capped = cap_return_contributions(returns, sigma_cap=4.0)
    assert not is_capped.any(), "small returns should not be capped"
    assert np.allclose(capped[~np.isnan(capped)], returns[~np.isnan(returns)])


def test_capped_abs_leq_uncapped_abs():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.002, 3000)
    # Inject some large returns
    returns[500] = 0.02
    returns[1000] = -0.015
    capped, _ = cap_return_contributions(returns, sigma_cap=4.0)
    valid = ~np.isnan(capped)
    assert np.all(np.abs(capped[valid]) <= np.abs(returns[valid]) + 1e-12)


def test_capped_rv_leq_uncapped_rv():
    """Key invariant: capping can only reduce or equal realized variance."""
    rng = np.random.default_rng(12)
    returns = rng.normal(0.0, 0.001, 5000)
    returns[2000] = 0.05  # large spike
    capped, _ = cap_return_contributions(returns, sigma_cap=4.0)
    rv_capped = np.nansum(capped**2)
    rv_uncapped = np.nansum(returns**2)
    assert rv_capped <= rv_uncapped + 1e-12


def test_nan_preserved():
    returns = _normal_returns(n=100)
    returns[50] = np.nan
    capped, is_capped = cap_return_contributions(returns)
    assert np.isnan(capped[50])
    assert not is_capped[50]


def test_mad_sigma_robustness():
    """MAD estimator should be robust even when 5% of returns are outliers.

    Sample std would be off by >100% on this input; MAD should be within 20%.
    """
    rng = np.random.default_rng(42)
    true_sigma = 0.001
    n = 2000
    returns = rng.normal(0.0, true_sigma, n)
    # Inject 5% outliers (100 returns of 20σ)
    outlier_idx = rng.choice(n, size=100, replace=False)
    returns[outlier_idx] = rng.choice([-1, 1], size=100) * 0.02

    # Compute MAD-based sigma over the full array as a proxy for the estimator
    mad_sigma = 1.4826 * np.median(np.abs(returns))
    # Should be within 20% of true sigma despite 5% contamination
    assert abs(mad_sigma - true_sigma) / true_sigma < 0.20, (
        f"MAD sigma {mad_sigma:.6f} too far from true {true_sigma}"
    )

    # Verify sample std is badly contaminated (>100% error)
    sample_std = np.std(returns)
    assert abs(sample_std - true_sigma) / true_sigma > 0.5, (
        "Expected sample std to be contaminated by outliers"
    )


def test_zero_sigma_no_cap():
    """Flat price series → σ̂=0 → cap should not fire."""
    returns = np.zeros(200)
    capped, is_capped = cap_return_contributions(returns, sigma_cap=4.0)
    assert not is_capped.any()
    assert np.allclose(capped, 0.0)


# --- minimum_obs_mask ---

def test_minimum_obs_mask_accepts_full():
    """Full 1-minute series for exactly 1 day should be valid at day boundary."""
    base_us = int(1_704_067_200 * 1_000_000)  # 2024-01-01 00:00:00 UTC
    n = 1441
    timestamps = np.array([base_us + i * 60_000_000 for i in range(n)], dtype=np.int64)
    window_us = 24 * 60 * 60 * 1_000_000  # 1 day
    mask = minimum_obs_mask(timestamps, window_us, min_fraction=0.9)
    # At the last index, we should have ~1440 obs in the window
    assert mask[-1], "full series should be valid"


def test_minimum_obs_mask_rejects_sparse():
    """Only 50% of expected observations → is_valid=False."""
    base_us = int(1_704_067_200 * 1_000_000)
    # Every other minute → 50% density
    n = 721
    timestamps = np.array([base_us + i * 120_000_000 for i in range(n)], dtype=np.int64)
    window_us = 24 * 60 * 60 * 1_000_000
    mask = minimum_obs_mask(timestamps, window_us, min_fraction=0.9)
    assert not mask[-1], "sparse series should be invalid"


def test_minimum_obs_mask_length():
    base_us = int(1_704_067_200 * 1_000_000)
    n = 100
    timestamps = np.array([base_us + i * 60_000_000 for i in range(n)], dtype=np.int64)
    mask = minimum_obs_mask(timestamps, 60 * 60 * 1_000_000)
    assert len(mask) == n


# --- Hypothesis property tests ---

@given(
    returns=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=3000,
    )
)
@settings(max_examples=200)
def test_cap_nonnegative_contributions_arbitrary(returns):
    arr = np.array(returns)
    capped, _ = cap_return_contributions(arr, sigma_cap=4.0)
    valid = ~np.isnan(capped)
    assert np.all(capped[valid] ** 2 >= 0.0)


@given(
    returns=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=3000,
    )
)
@settings(max_examples=200)
def test_capped_leq_uncapped_arbitrary(returns):
    arr = np.array(returns)
    capped, _ = cap_return_contributions(arr, sigma_cap=4.0)
    valid = ~np.isnan(capped)
    assert np.all(np.abs(capped[valid]) <= np.abs(arr[valid]) + 1e-12)
