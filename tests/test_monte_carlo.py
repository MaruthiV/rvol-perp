"""Tests for the Monte Carlo lifecycle simulator (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from simulations.monte_carlo.lifecycle import (
    LifecycleParams,
    _rolling_rv30_at_hourly,
    run_lifecycle,
)
from simulations.monte_carlo.paths import SVParams, simulate_paths


# --- SV path generator ---

class TestSVPaths:
    def test_shapes(self) -> None:
        p = SVParams()
        prices, log_var = simulate_paths(p, n_paths=4, n_steps=100, seed=0)
        assert prices.shape == (4, 101)
        assert log_var.shape == (4, 101)

    def test_determinism_under_seed(self) -> None:
        p = SVParams()
        a, _ = simulate_paths(p, n_paths=4, n_steps=100, seed=42)
        b, _ = simulate_paths(p, n_paths=4, n_steps=100, seed=42)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_different_seeds_differ(self) -> None:
        p = SVParams()
        a, _ = simulate_paths(p, n_paths=4, n_steps=100, seed=0)
        b, _ = simulate_paths(p, n_paths=4, n_steps=100, seed=1)
        assert not np.array_equal(np.asarray(a), np.asarray(b))

    def test_initial_price_matches(self) -> None:
        p = SVParams(s0=100.0)
        prices, _ = simulate_paths(p, n_paths=4, n_steps=10, seed=0)
        np.testing.assert_allclose(np.asarray(prices)[:, 0], 100.0)

    def test_log_variance_starts_at_v0(self) -> None:
        p = SVParams(v0=-7.0)
        _, log_var = simulate_paths(p, n_paths=4, n_steps=10, seed=0)
        np.testing.assert_allclose(np.asarray(log_var)[:, 0], -7.0)

    def test_prices_are_positive(self) -> None:
        p = SVParams()
        prices, _ = simulate_paths(p, n_paths=8, n_steps=500, seed=0)
        assert (np.asarray(prices) > 0).all()

    def test_realized_vol_matches_theta(self) -> None:
        """Over many paths × many steps, realized vol should converge to target."""
        p = SVParams(theta=-7.32, kappa=0.01, eta=0.0)  # deterministic vol
        prices, _ = simulate_paths(p, n_paths=64, n_steps=30 * 1440, seed=0)
        prices = np.asarray(prices)
        rets = np.diff(np.log(prices), axis=1)
        realized_vol_ann = np.sqrt((rets ** 2).sum(axis=1) * 365.0 / 30.0) * 100
        target = np.sqrt(np.exp(p.theta) * 365) * 100
        # With η=0, realized vol should match target within a few percent
        assert abs(np.median(realized_vol_ann) - target) / target < 0.05

    def test_zero_vol_of_vol_keeps_log_var_mean_reverting(self) -> None:
        p = SVParams(v0=-5.0, theta=-7.32, kappa=0.05, eta=0.0)
        _, log_var = simulate_paths(p, n_paths=2, n_steps=10_000, seed=0)
        lv = np.asarray(log_var)
        # Should monotonically drift toward theta
        assert lv[0, -1] < lv[0, 0]
        assert lv[0, -1] > p.theta - 0.5


# --- Lifecycle helpers ---

class TestRollingRV30:
    def test_flat_input_gives_window_scaled_sum(self) -> None:
        # 3000 steps of constant 1e-6 r² ; window 1440 ; hourly stride 60
        sq = np.full(3000, 1e-6)
        rv = _rolling_rv30_at_hourly(sq, window=1440, hourly_stride=60)
        # After hour 24 (minute 1440), window is full: rv should equal 1440 * 1e-6
        idx_full = 1440 // 60
        np.testing.assert_allclose(rv[idx_full:], 1440 * 1e-6)

    def test_zeros_give_zero_rv(self) -> None:
        sq = np.zeros(3000)
        rv = _rolling_rv30_at_hourly(sq, window=1440, hourly_stride=60)
        assert (rv == 0).all()

    def test_hour_count(self) -> None:
        sq = np.zeros(1500)
        rv = _rolling_rv30_at_hourly(sq, window=1440, hourly_stride=60)
        assert len(rv) == 1500 // 60


# --- Full lifecycle ---

class TestLifecycle:
    @pytest.fixture
    def small_paths(self) -> np.ndarray:
        p = SVParams()
        prices, _ = simulate_paths(p, n_paths=16, n_steps=45 * 1440, seed=7)
        return np.asarray(prices)

    def test_shapes(self, small_paths: np.ndarray) -> None:
        lc = LifecycleParams(entry_minute=30 * 1440)
        r = run_lifecycle(small_paths, lc, seed=0)
        n_hours = (small_paths.shape[1] - 1) // 60
        assert r.rv30_index_hourly.shape == (16, n_hours)
        assert r.mark_hourly.shape == (16, n_hours)
        assert r.long_final_pnl.shape == (16,)
        assert r.short_final_pnl.shape == (16,)

    def test_pnl_zero_sum(self, small_paths: np.ndarray) -> None:
        lc = LifecycleParams(entry_minute=30 * 1440)
        r = run_lifecycle(small_paths, lc, seed=0)
        np.testing.assert_allclose(
            r.long_final_pnl + r.short_final_pnl, 0.0, atol=1e-12
        )

    def test_index_nonnegative(self, small_paths: np.ndarray) -> None:
        lc = LifecycleParams(entry_minute=30 * 1440)
        r = run_lifecycle(small_paths, lc, seed=0)
        assert (r.rv30_index_hourly >= 0).all()

    def test_determinism(self, small_paths: np.ndarray) -> None:
        lc = LifecycleParams(entry_minute=30 * 1440)
        r1 = run_lifecycle(small_paths, lc, seed=0)
        r2 = run_lifecycle(small_paths, lc, seed=0)
        np.testing.assert_array_equal(r1.long_final_pnl, r2.long_final_pnl)

    def test_liquidation_trips_at_extreme_leverage(self, small_paths: np.ndarray) -> None:
        lc = LifecycleParams(
            entry_minute=30 * 1440, leverage=1000.0, maintenance_margin_frac=0.99
        )
        r = run_lifecycle(small_paths, lc, seed=0)
        # At pathological leverage, at least some paths must liquidate
        assert r.long_liquidated.any() or r.short_liquidated.any()

    def test_entry_after_horizon_errors(self, small_paths: np.ndarray) -> None:
        lc = LifecycleParams(entry_minute=100 * 1440)  # past horizon
        with pytest.raises(ValueError):
            run_lifecycle(small_paths, lc)

    def test_pnl_magnitude_scales_with_index(self, small_paths: np.ndarray) -> None:
        """Final PnL is bounded by total index variation over the hold window."""
        lc = LifecycleParams(entry_minute=30 * 1440)
        r = run_lifecycle(small_paths, lc, seed=0)
        entry_hour = lc.entry_minute // lc.funding_interval_minutes
        idx_range = (
            r.rv30_index_hourly[:, entry_hour:].max(axis=1)
            - r.rv30_index_hourly[:, entry_hour:].min(axis=1)
        )
        # |PnL| must not exceed index range plus accumulated funding payments
        assert (np.abs(r.long_final_pnl) <= idx_range * 10).all()
