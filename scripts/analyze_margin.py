"""Margin tier analysis — Phase 8.

Sweeps leverage across Monte Carlo lifecycle paths and reports liquidation
rates per tier. Checks that tier-max leverage produces <0.5% liq rate.

Usage:
    python scripts/analyze_margin.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rvol.margin import TIERS
from simulations.monte_carlo.lifecycle import LifecycleParams, run_lifecycle
from simulations.monte_carlo.paths import SVParams, simulate_paths

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def main() -> None:
    print("Simulating 512 × 60-day paths...")
    sv = SVParams()
    prices, _ = simulate_paths(sv, n_paths=512, n_steps=60 * 1440, seed=0)
    prices = np.asarray(prices)

    rows = []
    for tier in TIERS:
        lev = tier.max_leverage
        mm = tier.maintenance_margin_rate
        params = LifecycleParams(
            entry_minute=30 * 1440,
            leverage=lev,
            maintenance_margin_frac=mm,
        )
        res = run_lifecycle(prices, params, seed=0)
        long_liq = float(res.long_liquidated.mean())
        short_liq = float(res.short_liquidated.mean())
        rows.append({
            "tier": tier.index,
            "max_vega_usd": tier.max_vega_notional_usd,
            "leverage": lev,
            "im_rate": tier.initial_margin_rate,
            "mm_rate": mm,
            "long_liq_rate": long_liq,
            "short_liq_rate": short_liq,
        })
        print(
            f"  Tier {tier.index}  lev={lev:4.2f}×  "
            f"long_liq={long_liq*100:5.2f}%  short_liq={short_liq*100:5.2f}%"
        )

    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "margin_liquidation_table.csv"
    df.to_csv(out, index=False)
    print(f"\nTable → {out}")

    # --- Stress: 2× tier-max leverage, check residual losses ---
    print("\n" + "=" * 60)
    print("STRESS: 2× tier-max leverage (insurance fund check)")
    print("=" * 60)
    for tier in TIERS:
        lev = tier.max_leverage * 2.0
        params = LifecycleParams(
            entry_minute=30 * 1440,
            leverage=lev,
            maintenance_margin_frac=tier.maintenance_margin_rate,
        )
        res = run_lifecycle(prices, params, seed=0)
        long_liq = float(res.long_liquidated.mean())
        short_liq = float(res.short_liquidated.mean())
        print(
            f"  Tier {tier.index} @ 2×lev ({lev:4.2f}×): "
            f"long_liq={long_liq*100:5.2f}%  short_liq={short_liq*100:5.2f}%"
        )

    # --- Gate ---
    print("\n" + "=" * 60)
    print("GATE CHECK")
    print("=" * 60)
    max_liq = df[["long_liq_rate", "short_liq_rate"]].max().max()
    ok = max_liq < 0.005
    sym = "✓" if ok else "✗"
    print(f"  {sym} Tier-max liquidation rate < 0.5%  (max {max_liq*100:.3f}%)")


if __name__ == "__main__":
    main()
