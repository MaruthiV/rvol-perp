"""Tests for the attack injection module (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from rvol.attacks.inject import (
    AttackParams,
    estimate_cost_usd,
    inject_single_spike,
    inject_sustained,
    _local_mad_sigma,
    _rv_window,
)


@pytest.fixture
def benign_returns() -> np.ndarray:
    """60 days of synthetic benign 1-minute returns, ~50% annualized vol."""
    rng = np.random.default_rng(0)
    n = 60 * 1440
    sigma_1m = 0.5 / np.sqrt(365 * 1440)  # 50% ann vol
    return rng.normal(0.0, sigma_1m, size=n)


class TestLocalMadSigma:
    def test_zero_for_flat(self) -> None:
        r = np.zeros(2000)
        assert _local_mad_sigma(r, offset=1500) == 0.0

    def test_nonzero_for_noisy(self, benign_returns: np.ndarray) -> None:
        s = _local_mad_sigma(benign_returns, offset=30 * 1440)
        assert s > 0


class TestRvWindow:
    def test_sum_of_squares(self) -> None:
        r = np.array([0.01, 0.02, -0.01, 0.0])
        assert _rv_window(r, end=4, window=4) == pytest.approx(0.0006)


class TestSingleSpike:
    def test_basic_flow(self, benign_returns: np.ndarray) -> None:
        p = AttackParams(k_sigma=10.0, n_minutes=1, attack_offset=40 * 1440)
        result = inject_single_spike(benign_returns, p)
        assert result.rv30_uncapped > result.rv30_baseline
        assert result.rv30_capped > result.rv30_baseline
        # Uncapped impact must be at least as large as capped
        assert result.delta_uncapped >= result.delta_capped

    def test_cap_reduces_large_spike(self, benign_returns: np.ndarray) -> None:
        """At k=20σ, the cap should reduce attack impact by >>50%."""
        p = AttackParams(k_sigma=20.0, n_minutes=1, attack_offset=40 * 1440)
        r = inject_single_spike(benign_returns, p)
        assert r.cap_reduction > 0.50

    def test_small_spike_below_cap_not_reduced(self, benign_returns: np.ndarray) -> None:
        """At k=2σ (below the 4σ cap), the cap is a no-op."""
        p = AttackParams(k_sigma=2.0, n_minutes=1, attack_offset=40 * 1440)
        r = inject_single_spike(benign_returns, p)
        # Reduction should be effectively zero (small floating-point slop ok)
        assert r.cap_reduction < 0.05

    def test_n_minutes_must_be_1(self, benign_returns: np.ndarray) -> None:
        p = AttackParams(k_sigma=10.0, n_minutes=5, attack_offset=40 * 1440)
        with pytest.raises(ValueError):
            inject_single_spike(benign_returns, p)

    def test_attack_past_end_errors(self, benign_returns: np.ndarray) -> None:
        p = AttackParams(k_sigma=10.0, n_minutes=1, attack_offset=len(benign_returns))
        with pytest.raises(ValueError):
            inject_single_spike(benign_returns, p)


class TestSustained:
    def test_basic_flow(self, benign_returns: np.ndarray) -> None:
        p = AttackParams(k_sigma=10.0, n_minutes=60, attack_offset=40 * 1440)
        r = inject_sustained(benign_returns, p)
        assert r.rv30_capped > r.rv30_baseline
        assert r.delta_uncapped >= r.delta_capped

    def test_longer_attack_costs_more(self, benign_returns: np.ndarray) -> None:
        p_short = AttackParams(k_sigma=10.0, n_minutes=10, attack_offset=40 * 1440)
        p_long = AttackParams(k_sigma=10.0, n_minutes=60, attack_offset=40 * 1440)
        r_short = inject_sustained(benign_returns, p_short)
        r_long = inject_sustained(benign_returns, p_long)
        assert r_long.cost_usd > r_short.cost_usd

    def test_cap_reduction_strong_for_large_k(self, benign_returns: np.ndarray) -> None:
        p = AttackParams(k_sigma=15.0, n_minutes=30, attack_offset=40 * 1440)
        r = inject_sustained(benign_returns, p)
        # At 15σ with 4σ cap, capped magnitude is ~4/15 = 27%, so variance is
        # ~(4/15)² ≈ 7% of uncapped. Cap reduction should exceed 85%.
        assert r.cap_reduction > 0.85

    def test_requires_positive_n(self, benign_returns: np.ndarray) -> None:
        p = AttackParams(k_sigma=10.0, n_minutes=0, attack_offset=40 * 1440)
        with pytest.raises(ValueError):
            inject_sustained(benign_returns, p)


class TestCostModel:
    def test_linear_in_magnitude(self) -> None:
        p = AttackParams(k_sigma=1.0, n_minutes=1, attack_offset=0)
        c1 = estimate_cost_usd(p, np.array([0.01]))
        c2 = estimate_cost_usd(p, np.array([0.02]))
        assert c2 == pytest.approx(2 * c1)

    def test_sustained_has_round_trip(self) -> None:
        p = AttackParams(
            k_sigma=1.0, n_minutes=10, attack_offset=0, round_trip_factor=2.0
        )
        rets = np.full(10, 0.01)
        c = estimate_cost_usd(p, rets)
        # 10 minutes × 1% × $5M × 2 = $1M
        assert c == pytest.approx(1_000_000.0)

    def test_one_shot_no_round_trip(self) -> None:
        p = AttackParams(k_sigma=1.0, n_minutes=1, attack_offset=0, round_trip_factor=2.0)
        c = estimate_cost_usd(p, np.array([0.01]))
        # 1% × $5M = $50k (no round trip for single spike)
        assert c == pytest.approx(50_000.0)


class TestInvariants:
    @pytest.mark.parametrize("k", [1.0, 4.0, 8.0, 15.0])
    def test_capped_delta_never_exceeds_uncapped(
        self, benign_returns: np.ndarray, k: float
    ) -> None:
        p = AttackParams(k_sigma=k, n_minutes=30, attack_offset=40 * 1440)
        r = inject_sustained(benign_returns, p)
        assert r.delta_capped <= r.delta_uncapped + 1e-15

    @pytest.mark.parametrize("k", [1.0, 4.0, 8.0, 15.0])
    def test_cost_is_nonnegative(
        self, benign_returns: np.ndarray, k: float
    ) -> None:
        p = AttackParams(k_sigma=k, n_minutes=30, attack_offset=40 * 1440)
        r = inject_sustained(benign_returns, p)
        assert r.cost_usd >= 0
