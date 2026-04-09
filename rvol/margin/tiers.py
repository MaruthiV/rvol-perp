"""Margin tiers, liquidation math, and vega-notional conversion.

The variance perp's on-chain state is in daily variance units (e.g., 0.0004
≈ 40% annualized vol). User-facing sizing is in vega-notional (USD PnL per
+1 annualized vol point of RV30) because variance units are unintuitive.

Conversion: for small moves around vol level `σ_ann`,
    Δvariance_daily ≈ 2 σ_ann × Δσ_ann / 365
so one annualized vol point at reference vol `σ_ref` equals
    dv_per_vol_pt = 2 × (σ_ref / 100) × (1 / 100) / 365
                  = σ_ref × 2e-4 / 365
variance-notional (variance units of exposure) = vega_notional_usd / (usd_value_of_one_vol_pt)
where `usd_value_of_one_vol_pt = dv_per_vol_pt × variance_notional × price_ref`
is, by construction, $1 per vol pt × vega_notional_usd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tier:
    """A single margin tier on vega-notional size."""

    index: int
    max_vega_notional_usd: float  # upper bound; np.inf for top tier
    initial_margin_rate: float
    maintenance_margin_rate: float

    @property
    def max_leverage(self) -> float:
        return 1.0 / self.initial_margin_rate


# ---------------------------------------------------------------------------
# Capped-payoff variance perp (Phase 8.6)
# ---------------------------------------------------------------------------
# Every OTC variance swap since 1998 has been a *capped* variance swap — the
# uncapped payoff is unbounded above (variance can move by multiples) and
# impossible to margin safely. Phase 8.5 historical replay (`docs/design/
# 07-phase-8.5-findings.md`) confirmed this empirically for the BTC RV30
# index. We therefore cap position PnL at `2.5 × entry_index`, which is the
# standard cap level used in TradFi variance swap contracts.

PAYOFF_CAP: float = 2.5


def capped_pnl_frac(entry_index: float, current_index: float, side: str) -> float:
    """Capped PnL per unit notional, as a fraction of entry index.

    Long  PnL = clip( (I - I0)/I0, -1, cap-1 )
    Short PnL = -clip( (I - I0)/I0, -1, cap-1 )
    """
    if entry_index <= 0:
        return 0.0
    raw = (current_index - entry_index) / entry_index
    clipped = max(-1.0, min(raw, PAYOFF_CAP - 1.0))
    if side == "long":
        return clipped
    if side == "short":
        return -clipped
    raise ValueError("side must be 'long' or 'short'")


# Tiers are asymmetric: longs are bounded at -100%, shorts at +150% (from
# the cap). The short side therefore requires more margin. Rates below
# come from the calibration sweep in `scripts/analyze_margin_capped.py`.

TIERS: tuple[Tier, ...] = (
    Tier(1,    50_000.0,  0.67, 0.05),   # 1.5× lev
    Tier(2,   250_000.0,  0.75, 0.05),   # 1.33× lev
    Tier(3, 1_000_000.0,  0.80, 0.05),   # 1.25× lev
    Tier(4, 5_000_000.0,  1.00, 0.05),   # 1× lev
    Tier(5,       np.inf, 1.50, 0.05),   # 0.67× lev (overcollat)
)


def tier_for(vega_notional_usd: float) -> Tier:
    """Return the lowest tier whose ceiling contains `vega_notional_usd`."""
    if vega_notional_usd < 0:
        raise ValueError("vega_notional must be non-negative")
    for t in TIERS:
        if vega_notional_usd <= t.max_vega_notional_usd:
            return t
    return TIERS[-1]  # unreachable given inf ceiling


def initial_margin(vega_notional_usd: float) -> float:
    """Initial margin in USD for a given vega-notional."""
    return tier_for(vega_notional_usd).initial_margin_rate * vega_notional_usd


def maintenance_margin(vega_notional_usd: float) -> float:
    """Maintenance margin in USD for a given vega-notional."""
    return tier_for(vega_notional_usd).maintenance_margin_rate * vega_notional_usd


def liquidation_index_level(
    entry_index: float,
    vega_notional_usd: float,
    collateral_usd: float,
    side: str,
) -> float:
    """Index level at which equity crosses the maintenance margin.

    Position PnL is linear in the index (via vega-notional), so:
        long:  equity = collateral + vega × (I_t − I_entry) × 100 × sqrt(365/30) / (2·σ_ref)
    We simplify by working in "vol points": vega-notional gives $ per vol pt
    directly, so PnL = vega × Δvol_pt. Liquidation when:
        collateral + vega × Δvol_pt = mm_usd          (long)
        collateral − vega × Δvol_pt = mm_usd          (short)
    → Δvol_pt_liq = ±(collateral − mm) / vega
    Convert Δvol_pt back to Δindex via dv/dvol_pt = 2·σ_ref_ann·0.01 / 365
    at `σ_ref_ann = sqrt(entry_index · 365) · 100`.

    Funding payments are ignored here — caller must track equity including
    funding separately if tight liquidation estimates are needed.
    """
    if vega_notional_usd <= 0:
        raise ValueError("vega_notional must be positive")
    if entry_index <= 0:
        raise ValueError("entry_index must be positive")
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    mm_usd = maintenance_margin(vega_notional_usd)
    buffer_usd = collateral_usd - mm_usd
    d_vol_pt_liq = buffer_usd / vega_notional_usd  # magnitude of adverse move

    sigma_ref_ann_pct = float(np.sqrt(max(entry_index, 0.0) * 365.0) * 100.0)
    # One annualized vol point corresponds to this Δ in daily-variance:
    # d(σ²/365)/dσ = 2σ/365, so Δv = 2·σ·Δσ / 365, with σ in decimal (1 vol pt = 0.01).
    dv_per_vol_pt = 2.0 * (sigma_ref_ann_pct / 100.0) * 0.01 / 365.0
    if dv_per_vol_pt <= 0:
        return entry_index  # degenerate

    delta_index = d_vol_pt_liq * dv_per_vol_pt
    if side == "long":
        return max(0.0, entry_index - delta_index)
    return entry_index + delta_index


def vega_notional_to_variance(
    vega_notional_usd: float, reference_index: float
) -> float:
    """Convert $/vol-pt to variance-notional (units of variance exposure).

    Inverse of `variance_to_vega_notional` at the same reference index.
    """
    if reference_index <= 0:
        raise ValueError("reference_index must be positive")
    sigma_ref_ann_pct = float(np.sqrt(reference_index * 365.0) * 100.0)
    dv_per_vol_pt = 2.0 * (sigma_ref_ann_pct / 100.0) * 0.01 / 365.0
    if dv_per_vol_pt <= 0:
        return 0.0
    return vega_notional_usd / dv_per_vol_pt


def variance_to_vega_notional(
    variance_notional: float, reference_index: float
) -> float:
    """Convert a variance-unit position size to $/vol-pt at a reference index."""
    if reference_index <= 0:
        raise ValueError("reference_index must be positive")
    sigma_ref_ann_pct = float(np.sqrt(reference_index * 365.0) * 100.0)
    dv_per_vol_pt = 2.0 * (sigma_ref_ann_pct / 100.0) * 0.01 / 365.0
    return variance_notional * dv_per_vol_pt
