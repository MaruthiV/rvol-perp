"""Tests for the funding mechanism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rvol.funding import (
    BasisProcess,
    FundingParams,
    funding_rate,
    funding_rate_series,
    simulate_mark_convergence,
)


class TestFundingParams:
    def test_defaults(self) -> None:
        p = FundingParams()
        assert p.dampening > 0 and p.cap > 0 and p.interval_hours > 0

    @pytest.mark.parametrize("bad", [{"dampening": 0}, {"cap": -0.01}, {"interval_hours": 0}])
    def test_validation(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            FundingParams(**bad)


class TestFundingRateScalar:
    def test_zero_basis_zero_funding(self) -> None:
        p = FundingParams()
        assert funding_rate(mark=100.0, index=100.0, params=p) == 0.0

    def test_sign_convention_mark_above(self) -> None:
        """mark > index → longs pay shorts → funding > 0."""
        p = FundingParams(dampening=1.0, cap=1.0)
        assert funding_rate(mark=1.10, index=1.00, params=p) > 0

    def test_sign_convention_mark_below(self) -> None:
        p = FundingParams(dampening=1.0, cap=1.0)
        assert funding_rate(mark=0.90, index=1.00, params=p) < 0

    def test_cap_upper(self) -> None:
        p = FundingParams(dampening=1.0, cap=0.01)
        # basis = 100% → raw = 1.0 → clipped to +0.01
        assert funding_rate(mark=2.0, index=1.0, params=p) == pytest.approx(0.01)

    def test_cap_lower(self) -> None:
        p = FundingParams(dampening=1.0, cap=0.01)
        assert funding_rate(mark=0.0, index=1.0, params=p) == pytest.approx(-0.01)

    def test_dampening_reduces_rate(self) -> None:
        p1 = FundingParams(dampening=1.0, cap=1.0)
        p2 = FundingParams(dampening=10.0, cap=1.0)
        r1 = funding_rate(mark=1.05, index=1.00, params=p1)
        r2 = funding_rate(mark=1.05, index=1.00, params=p2)
        assert r2 == pytest.approx(r1 / 10.0)

    def test_degenerate_index(self) -> None:
        p = FundingParams()
        assert funding_rate(mark=1.0, index=0.0, params=p) == 0.0
        assert funding_rate(mark=1.0, index=-1.0, params=p) == 0.0
        assert funding_rate(mark=np.nan, index=1.0, params=p) == 0.0
        assert funding_rate(mark=1.0, index=np.nan, params=p) == 0.0


class TestFundingRateSeries:
    def test_matches_scalar(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        mark = pd.Series([1.00, 1.01, 1.05, 0.95, 2.00], index=idx)
        index = pd.Series([1.00, 1.00, 1.00, 1.00, 1.00], index=idx)
        p = FundingParams(dampening=1.0, cap=0.5)
        s = funding_rate_series(mark, index, p)
        expected = [funding_rate(m, i, p) for m, i in zip(mark, index)]
        np.testing.assert_allclose(s.to_numpy(), expected)

    def test_zero_index_handled(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        mark = pd.Series([1.0, 1.0, 1.0], index=idx)
        index = pd.Series([1.0, 0.0, -1.0], index=idx)
        s = funding_rate_series(mark, index, FundingParams())
        assert s.iloc[1] == 0.0 and s.iloc[2] == 0.0


class TestSimulation:
    @pytest.fixture
    def index_series(self) -> pd.Series:
        # Flat index at 0.002 daily variance (~55% ann vol) for 720 hours (30d)
        idx = pd.date_range("2024-01-01", periods=720, freq="1h", tz="UTC")
        return pd.Series(0.002, index=idx, name="index")

    def test_zero_basis_zero_shocks_zero_funding(self, index_series: pd.Series) -> None:
        """With no initial basis and no noise, mark stays pinned and funding is zero."""
        bp = BasisProcess(rho=0.9, sigma=0.0, gamma=0.0, initial_basis=0.0)
        r = simulate_mark_convergence(index_series, FundingParams(), bp, seed=0)
        assert r.mean_abs_basis == pytest.approx(0.0, abs=1e-12)
        assert r.funding.abs().max() == pytest.approx(0.0)
        # Long and short PnL must be exactly zero (flat index, no funding)
        assert r.long_pnl.iloc[-1] == pytest.approx(0.0)
        assert r.short_pnl.iloc[-1] == pytest.approx(0.0)

    def test_pnl_is_zero_sum(self, index_series: pd.Series) -> None:
        bp = BasisProcess(rho=0.9, sigma=0.01, gamma=5.0, initial_basis=0.1)
        r = simulate_mark_convergence(index_series, FundingParams(), bp, seed=42)
        total = r.long_pnl + r.short_pnl
        np.testing.assert_allclose(total.to_numpy(), 0.0, atol=1e-12)

    def test_initial_basis_decays(self, index_series: pd.Series) -> None:
        """Positive initial basis should decay toward zero under mean reversion + arb."""
        bp = BasisProcess(rho=0.9, sigma=0.0, gamma=10.0, initial_basis=0.2)
        r = simulate_mark_convergence(index_series, FundingParams(), bp, seed=0)
        assert abs(r.basis.iloc[-1]) < abs(r.basis.iloc[0])
        assert abs(r.basis.iloc[-1]) < 0.01  # converges close to zero

    def test_funding_opposes_basis(self, index_series: pd.Series) -> None:
        bp = BasisProcess(rho=0.99, sigma=0.0, gamma=0.0, initial_basis=0.1)
        r = simulate_mark_convergence(index_series, FundingParams(), bp, seed=0)
        # With positive basis throughout, funding should be uniformly positive
        assert (r.funding >= 0).all()
        assert r.funding.iloc[0] > 0

    def test_cap_binds_under_huge_basis(self, index_series: pd.Series) -> None:
        bp = BasisProcess(rho=1.0, sigma=0.0, gamma=0.0, initial_basis=5.0)
        p = FundingParams(dampening=100.0, cap=0.005)
        r = simulate_mark_convergence(index_series, p, bp, seed=0)
        assert r.cap_hit_fraction > 0.9

    def test_requires_two_observations(self) -> None:
        idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
        with pytest.raises(ValueError):
            simulate_mark_convergence(
                pd.Series([0.002], index=idx), FundingParams(), BasisProcess()
            )
