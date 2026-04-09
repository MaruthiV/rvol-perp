"""Tests for margin tiers and liquidation math (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest

from rvol.margin import (
    PAYOFF_CAP,
    TIERS,
    Tier,
    capped_pnl_frac,
    initial_margin,
    liquidation_index_level,
    maintenance_margin,
    tier_for,
    vega_notional_to_variance,
    variance_to_vega_notional,
)


class TestCappedPayoff:
    def test_cap_level(self) -> None:
        assert PAYOFF_CAP == 2.5

    def test_long_gains_clipped_above(self) -> None:
        # Index triples → raw = +2, clipped at cap-1 = 1.5
        assert capped_pnl_frac(1.0, 3.0, "long") == pytest.approx(1.5)

    def test_long_loss_clipped_at_minus_one(self) -> None:
        assert capped_pnl_frac(1.0, 0.0, "long") == pytest.approx(-1.0)

    def test_short_is_negation(self) -> None:
        assert capped_pnl_frac(1.0, 2.0, "short") == pytest.approx(
            -capped_pnl_frac(1.0, 2.0, "long")
        )

    def test_short_max_loss_equals_cap_minus_one(self) -> None:
        # Extreme: index moves to 100× entry. Short loses min(move, cap-1) = 1.5
        assert capped_pnl_frac(1.0, 100.0, "short") == pytest.approx(-1.5)

    def test_within_cap_is_linear(self) -> None:
        assert capped_pnl_frac(1.0, 1.25, "long") == pytest.approx(0.25)
        assert capped_pnl_frac(1.0, 0.75, "long") == pytest.approx(-0.25)

    def test_invalid_side(self) -> None:
        with pytest.raises(ValueError):
            capped_pnl_frac(1.0, 2.0, "flat")

    def test_zero_entry_is_safe_zero(self) -> None:
        assert capped_pnl_frac(0.0, 1.0, "long") == 0.0


class TestTierTable:
    def test_tiers_are_monotonic(self) -> None:
        prev_ceiling = 0.0
        for t in TIERS:
            assert t.max_vega_notional_usd > prev_ceiling
            prev_ceiling = t.max_vega_notional_usd
        # IM and MM should be non-decreasing across tiers
        ims = [t.initial_margin_rate for t in TIERS]
        mms = [t.maintenance_margin_rate for t in TIERS]
        assert ims == sorted(ims)
        assert mms == sorted(mms)

    def test_mm_below_im(self) -> None:
        for t in TIERS:
            assert t.maintenance_margin_rate < t.initial_margin_rate

    def test_top_tier_is_infinite(self) -> None:
        assert np.isinf(TIERS[-1].max_vega_notional_usd)

    def test_max_leverage_derived(self) -> None:
        # Post-Phase-8.6 capped design: leverage ≤ 1.5×, overcollat at top
        assert TIERS[0].max_leverage == pytest.approx(1.0 / TIERS[0].initial_margin_rate)
        assert TIERS[-1].max_leverage < 1.0  # top tier is overcollateralized


class TestTierFor:
    def test_tier_boundaries(self) -> None:
        # At exactly the ceiling, we stay in the tier
        assert tier_for(50_000).index == 1
        assert tier_for(50_001).index == 2
        assert tier_for(250_000).index == 2
        assert tier_for(1_000_001).index == 4
        assert tier_for(100_000_000).index == 5

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            tier_for(-1)

    def test_zero_returns_first_tier(self) -> None:
        assert tier_for(0).index == 1


class TestMarginFunctions:
    def test_initial_margin_scales(self) -> None:
        # IM is the tier-1 rate × notional, whatever that rate is.
        assert initial_margin(10_000) == pytest.approx(
            TIERS[0].initial_margin_rate * 10_000
        )
        assert initial_margin(500_000) == pytest.approx(
            TIERS[2].initial_margin_rate * 500_000
        )

    def test_maintenance_below_initial(self) -> None:
        for v in [1e3, 1e5, 1e6, 1e7]:
            assert maintenance_margin(v) < initial_margin(v)


class TestVegaConversion:
    def test_round_trip(self) -> None:
        I = 0.00044  # ≈ 40% ann vol in daily variance units
        vega = 100_000.0
        variance = vega_notional_to_variance(vega, I)
        back = variance_to_vega_notional(variance, I)
        assert back == pytest.approx(vega, rel=1e-10)

    def test_reference_index_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            vega_notional_to_variance(1.0, 0.0)
        with pytest.raises(ValueError):
            variance_to_vega_notional(1.0, -0.001)


class TestLiquidation:
    def test_long_below_entry_short_above(self) -> None:
        I = 0.00044
        lq_long = liquidation_index_level(I, 100_000, 20_000, "long")
        lq_short = liquidation_index_level(I, 100_000, 20_000, "short")
        assert lq_long < I
        assert lq_short > I

    def test_more_collateral_pushes_liquidation_further(self) -> None:
        I = 0.00044
        lq_tight = liquidation_index_level(I, 100_000, 15_000, "long")
        lq_safe = liquidation_index_level(I, 100_000, 30_000, "long")
        assert lq_safe < lq_tight  # further from entry = safer

    def test_larger_position_same_collat_frac_has_less_buffer(self) -> None:
        """At fixed collateral fraction, larger tier = tighter buffer → liq closer to entry."""
        I = 0.00044
        # Same collateral fraction (80%), different tiers.
        small_vega = 40_000   # tier 1
        large_vega = 400_000  # tier 2 (different MM rate)
        lq_small = liquidation_index_level(I, small_vega, small_vega * 0.80, "long")
        lq_large = liquidation_index_level(I, large_vega, large_vega * 0.80, "long")
        # Post-8.6 tiers have flat MM = 5% across tiers, so both have the
        # same buffer → liq levels equal. Assert non-strict.
        assert lq_large >= lq_small

    def test_invalid_side_rejected(self) -> None:
        with pytest.raises(ValueError):
            liquidation_index_level(0.0004, 100_000, 20_000, "sideways")

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            liquidation_index_level(0.0004, 0, 10_000, "long")
        with pytest.raises(ValueError):
            liquidation_index_level(0.0, 100_000, 10_000, "long")

    def test_liquidation_at_zero_collateral_above_mm(self) -> None:
        """If collateral < MM at entry, liquidation is at entry."""
        I = 0.00044
        vega = 100_000
        mm = maintenance_margin(vega)  # 20_000
        # Give exactly mm as collateral → buffer = 0 → liq at entry
        lq = liquidation_index_level(I, vega, mm, "long")
        assert lq == pytest.approx(I, rel=1e-10)
