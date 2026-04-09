"""Variance Risk Premium (VRP) analysis — the hard gate for Phase 3.

Computes: VRP_t = implied_variance_t - realized_variance_{t, t+30d}
  where implied_variance = (DVOL_t / 100)^2 / 365  (daily rate)
  and   realized_variance = RV30_t (already in daily-rate units)

A positive average VRP means implied vol > realized vol on average —
the economic case for short-vol sellers and the natural carry trade
that makes the product viable.

Output: prints key stats and saves two plots to data/processed/
  - vrp_timeseries.png  : RV30 vs DVOL annualized vol over time
  - vrp_distribution.png: distribution of daily VRP

Usage:
    python scripts/analyze_vrp.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "index" / "btcusdt_rv_index.parquet"
DVOL_PATH = Path(__file__).parent.parent / "data" / "raw" / "deribit" / "btc" / "dvol" / "dvol.parquet"


def load_rv30_daily() -> pd.Series:
    """Load RV30 index, resample to daily close (end-of-day value)."""
    tbl = pq.read_table(INDEX_PATH, columns=["timestamp_us", "rv30", "is_valid"])
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df = df[df["is_valid"]].set_index("ts")
    # Daily close: last valid RV30 value of each UTC day
    daily = df["rv30"].resample("1D").last().dropna()
    return daily


def load_dvol_daily() -> pd.Series:
    """Load DVOL hourly, resample to daily close."""
    tbl = pq.read_table(DVOL_PATH)
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df = df.set_index("ts")
    daily = df["dvol_annualized"].resample("1D").last().dropna()
    return daily


def main() -> None:
    print("Loading data...")
    rv30 = load_rv30_daily()
    dvol = load_dvol_daily()

    # Align on common dates
    common = rv30.index.intersection(dvol.index)
    rv30 = rv30.loc[common]
    dvol = dvol.loc[common]

    print(f"Overlapping period: {common[0].date()} → {common[-1].date()} ({len(common)} days)")

    # Convert DVOL (annualized %) to daily implied variance
    # DVOL is annualized vol in %, e.g. 50.0 means 50% annualized
    # implied_var_daily = (dvol/100)^2 / 365
    iv_daily = (dvol / 100) ** 2 / 365

    # RV30 is already sum of squared log returns over 30-day window
    # To compare on same scale, convert to daily-rate variance
    # rv30_daily_rate = rv30 / 30
    rv30_daily = rv30 / 30

    # VRP = implied daily variance - realized daily variance (forward-matched)
    # Forward-match: DVOL at t predicts realized var over [t, t+30d]
    # Shift RV30 back 30 days to align with the DVOL that "predicted" it
    rv30_forward = rv30_daily.shift(-30)  # realized var that DVOL_t was predicting

    vrp = iv_daily - rv30_forward
    vrp = vrp.dropna()

    # --- Key statistics ---
    vrp_annualized = vrp * 365 * 100  # convert to annualized percentage points

    print("\n" + "="*55)
    print("VARIANCE RISK PREMIUM ANALYSIS")
    print("="*55)
    print(f"\nVRP = Implied variance - Forward realized variance")
    print(f"(Positive VRP = implied > realized = short-vol earns carry)\n")

    print(f"Mean VRP (annualized):   {vrp_annualized.mean():+.2f} pp")
    print(f"Median VRP (annualized): {vrp_annualized.median():+.2f} pp")
    print(f"Std VRP (annualized):    {vrp_annualized.std():.2f} pp")
    print(f"Sharpe of short-vol:     {vrp_annualized.mean() / vrp_annualized.std():.3f}")
    print(f"% days VRP > 0:          {(vrp > 0).mean()*100:.1f}%")

    print(f"\nRV30 annualized vol (median):  {np.sqrt(rv30_daily.median() * 365) * 100:.1f}%")
    print(f"DVOL (median):                 {dvol.median():.1f}%")
    print(f"Implied - Realized spread:     {dvol.median() - np.sqrt(rv30_daily.median() * 365)*100:.1f} vol points")

    # Regime breakdown
    print("\nVRP by year:")
    for year in sorted(vrp_annualized.index.year.unique()):
        yr = vrp_annualized[vrp_annualized.index.year == year]
        print(f"  {year}: mean={yr.mean():+.1f} pp  median={yr.median():+.1f} pp  n={len(yr)}")

    # Gate check
    print("\n" + "="*55)
    mean_vrp = vrp_annualized.mean()
    if mean_vrp > 0:
        print(f"✓ GATE PASSED: Mean VRP = {mean_vrp:+.2f} pp > 0")
        print("  Economic case confirmed. Proceed to Phase 3.")
    else:
        print(f"✗ GATE FAILED: Mean VRP = {mean_vrp:+.2f} pp ≤ 0")
        print("  Reassess contract design before simulation work.")
    print("="*55)

    # --- Plots ---
    out_dir = PROCESSED_DIR
    _plot_timeseries(rv30_daily, dvol, vrp_annualized, out_dir)
    _plot_distribution(vrp_annualized, out_dir)


def _plot_timeseries(rv30_daily: pd.Series, dvol: pd.Series, vrp_ann: pd.Series, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("BTC Realized Variance Index vs Deribit DVOL", fontsize=13)

    # Top panel: annualized vol comparison
    ax = axes[0]
    rv_ann_vol = np.sqrt(rv30_daily * 365) * 100
    ax.plot(rv_ann_vol.index, rv_ann_vol, label="RV30 (annualized vol %)", color="#2196F3", lw=1.2)
    ax.plot(dvol.index, dvol, label="Deribit DVOL (implied vol %)", color="#FF5722", lw=1.2, alpha=0.8)
    ax.set_ylabel("Annualized Vol (%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom panel: VRP
    ax2 = axes[1]
    pos = vrp_ann.clip(lower=0)
    neg = vrp_ann.clip(upper=0)
    ax2.fill_between(vrp_ann.index, pos, 0, alpha=0.6, color="#4CAF50", label="VRP > 0 (short earns)")
    ax2.fill_between(vrp_ann.index, neg, 0, alpha=0.6, color="#F44336", label="VRP < 0 (short loses)")
    ax2.axhline(vrp_ann.mean(), color="black", lw=1.5, ls="--", label=f"Mean = {vrp_ann.mean():+.1f} pp")
    ax2.set_ylabel("VRP (annualized pp)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = out_dir / "vrp_timeseries.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {path}")


def _plot_distribution(vrp_ann: pd.Series, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vrp_ann, bins=80, color="#2196F3", edgecolor="white", linewidth=0.3, alpha=0.8)
    ax.axvline(0, color="black", lw=1.5, ls="-", label="Zero")
    ax.axvline(vrp_ann.mean(), color="#F44336", lw=2, ls="--",
               label=f"Mean = {vrp_ann.mean():+.1f} pp")
    ax.axvline(vrp_ann.median(), color="#FF9800", lw=2, ls="--",
               label=f"Median = {vrp_ann.median():+.1f} pp")
    ax.set_xlabel("VRP (annualized percentage points)")
    ax.set_ylabel("Days")
    ax.set_title("Distribution of Daily Variance Risk Premium (BTC)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "vrp_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {path}")


if __name__ == "__main__":
    main()
