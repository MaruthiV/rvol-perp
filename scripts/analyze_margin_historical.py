"""Historical margin replay — Phase 8.5.

Closes the empirical gap left by Phase 8: the Monte Carlo margin sweep
produced 0% liquidations, but SV paths don't contain regime breaks. This
script replays the real BTC RV30 index (2020–present) through the margin
and funding mechanics and reports:

  1. Event study — positions opened 30 days before known stress events,
     held 60 days, at each tier's max leverage. Did they liquidate?

  2. Rolling sweep — entries every 7 days across the full history at
     tier-1 max leverage (10×). What fraction liquidate?

The real index path contains COVID (Mar 2020), LUNA (May 2022), FTX
(Nov 2022), SVB (Mar 2023), and the ETF approval (Jan 2024).

Usage:
    python scripts/analyze_margin_historical.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rvol.funding.rate import FundingParams
from rvol.funding.simulate import BasisProcess, simulate_mark_convergence
from rvol.margin import TIERS

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "index" / "btcusdt_rv_index.parquet"

# The production index value is the 30-day sum-of-squares divided by the
# number of days in the window → daily variance units.
WINDOW_DAYS = 30


def load_hourly_index() -> pd.Series:
    """Load the production RV30 index, downsampled to hourly, valid-only."""
    tbl = pq.read_table(INDEX_PATH, columns=["timestamp_us", "rv30", "is_valid"])
    df = tbl.to_pandas()
    df = df[df["is_valid"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp_us"], unit="us")
    df = df.set_index("timestamp")
    # Convert sum-of-squares to daily-variance units
    df["index_daily_var"] = df["rv30"] / WINDOW_DAYS
    # Downsample to hourly (take last minute of each hour)
    hourly = df["index_daily_var"].resample("1h").last().dropna()
    return hourly


def simulate_position(
    index_slice: pd.Series,
    leverage: float,
    mm_rate: float,
    basis_seed: int = 0,
) -> tuple[bool, bool, float, float]:
    """Run a long and a short through `index_slice` at `leverage`.

    Returns (long_liquidated, short_liquidated, long_final_pnl_frac, short_final_pnl_frac).
    PnL is expressed as fraction of entry index (dimensionless), so an IM of
    1/leverage and an MM of `mm_rate` are directly comparable.
    """
    if len(index_slice) < 2:
        return False, False, 0.0, 0.0

    fp = FundingParams(dampening=100.0, cap=0.001, interval_hours=1)
    bp = BasisProcess(rho=0.90, sigma=0.005, gamma=10.0, initial_basis=0.0)
    res = simulate_mark_convergence(index_slice, fp, bp, seed=basis_seed)

    idx = index_slice.to_numpy(dtype=np.float64)
    I0 = idx[0]
    if I0 <= 0:
        return False, False, 0.0, 0.0

    # Long / short equity, normalized by I0 so units match margin rates.
    # Start with initial margin = 1/leverage. PnL per step:
    #   long:  dI/I0 - f_t  (funding paid out of equity when positive)
    #   short: -dI/I0 + f_t
    dI = np.diff(idx, prepend=idx[0]) / I0
    funding = res.funding.to_numpy()

    long_inc = dI - funding
    short_inc = -dI + funding

    long_eq = 1.0 / leverage + np.cumsum(long_inc)
    short_eq = 1.0 / leverage + np.cumsum(short_inc)

    long_liq = bool(np.any(long_eq < mm_rate))
    short_liq = bool(np.any(short_eq < mm_rate))
    return long_liq, short_liq, float(long_eq[-1] - 1.0 / leverage), float(short_eq[-1] - 1.0 / leverage)


STRESS_EVENTS = {
    "COVID":        "2020-03-12",
    "LUNA":         "2022-05-09",
    "FTX":          "2022-11-08",
    "SVB":          "2023-03-10",
    "ETF_approval": "2024-01-10",
}


def event_study(index: pd.Series) -> pd.DataFrame:
    """For each event × tier, open 30 days before and hold 60 days."""
    rows = []
    for name, date_str in STRESS_EVENTS.items():
        event = pd.Timestamp(date_str)
        entry = event - pd.Timedelta(days=30)
        exit = entry + pd.Timedelta(days=60)
        if entry < index.index[0] or exit > index.index[-1]:
            print(f"  [skip] {name}: out of range")
            continue
        window = index.loc[entry:exit]
        I_entry = float(window.iloc[0])
        I_peak = float(window.max())
        peak_vol = float(np.sqrt(I_peak * 365) * 100)
        entry_vol = float(np.sqrt(I_entry * 365) * 100)
        for tier in TIERS:
            lliq, sliq, lpnl, spnl = simulate_position(
                window, tier.max_leverage, tier.maintenance_margin_rate
            )
            rows.append({
                "event": name,
                "tier": tier.index,
                "leverage": tier.max_leverage,
                "entry_vol_ann_pct": entry_vol,
                "peak_vol_ann_pct": peak_vol,
                "long_liq": lliq,
                "short_liq": sliq,
                "long_pnl_frac": lpnl,
                "short_pnl_frac": spnl,
            })
    return pd.DataFrame(rows)


def worst_excursions(index: pd.Series, step_days: int = 7, hold_days: int = 60) -> pd.DataFrame:
    """For each rolling 60-day window, compute the worst (most adverse)
    equity excursion for a long and a short over the window, expressed as
    a fraction of entry index. This is the empirical draw required of any
    maintenance margin rate: MM ≥ worst_excursion percentile → survives.

    With funding turned off (it's tiny vs. spot index moves) this reduces
    to the min/max of normalized cumulative index change.
    """
    entries = pd.date_range(
        start=index.index[0],
        end=index.index[-1] - pd.Timedelta(days=hold_days),
        freq=f"{step_days}D",
    )
    rows = []
    for i, e in enumerate(entries):
        exit = e + pd.Timedelta(days=hold_days)
        window = index.loc[e:exit]
        if len(window) < 24 * hold_days // 2:
            continue
        I0 = float(window.iloc[0])
        if I0 <= 0:
            continue
        norm = window.to_numpy() / I0
        cum_d = norm - 1.0
        # Worst long excursion = min of cum_d (largest drop below entry)
        # Worst short excursion = max of cum_d (largest rise above entry)
        rows.append({
            "entry": e,
            "I0_ann_vol_pct": float(np.sqrt(I0 * 365) * 100),
            "worst_long_draw": float(cum_d.min()),   # negative
            "worst_short_draw": float(cum_d.max()),  # positive
        })
    return pd.DataFrame(rows)


def calibrate_tiers(exc: pd.DataFrame, survival: float = 0.99) -> pd.DataFrame:
    """Report the MM (and implied IM = 2·MM, leverage = 1/IM) required to
    cover `survival` fraction of historical 60-day windows."""
    pcts = [0.90, 0.95, 0.99, 1.00]
    rows = []
    for p in pcts:
        # For longs: MM must cover the worst draw (most negative)
        # so IM + worst_long_draw > MM  →  IM - MM > -worst_long_draw
        # take the p-th percentile of -worst_long_draw (magnitude).
        long_req = float(np.quantile(-exc["worst_long_draw"], p))
        short_req = float(np.quantile(exc["worst_short_draw"], p))
        mm_required = max(long_req, short_req)
        # A reasonable convention: IM = MM + buffer; leverage ≤ 1/(IM)
        # Use IM = 1.5 × MM so equity starts 0.5·MM above MM
        im = 1.5 * mm_required
        rows.append({
            "survival_pct": p * 100,
            "long_draw_req": long_req,
            "short_draw_req": short_req,
            "mm_required": mm_required,
            "im_required": im,
            "max_leverage": 1.0 / im if im > 0 else float("inf"),
        })
    return pd.DataFrame(rows)


def rolling_sweep(index: pd.Series, step_days: int = 7, hold_days: int = 60) -> pd.DataFrame:
    """Open positions every `step_days` at each tier's max leverage."""
    entries = pd.date_range(
        start=index.index[0],
        end=index.index[-1] - pd.Timedelta(days=hold_days),
        freq=f"{step_days}D",
    )
    rows = []
    for tier in TIERS:
        n_total = 0
        n_long_liq = 0
        n_short_liq = 0
        for i, e in enumerate(entries):
            exit = e + pd.Timedelta(days=hold_days)
            window = index.loc[e:exit]
            if len(window) < 24 * hold_days // 2:
                continue
            lliq, sliq, _, _ = simulate_position(
                window, tier.max_leverage, tier.maintenance_margin_rate, basis_seed=i,
            )
            n_total += 1
            n_long_liq += int(lliq)
            n_short_liq += int(sliq)
        rows.append({
            "tier": tier.index,
            "leverage": tier.max_leverage,
            "n_positions": n_total,
            "long_liq_rate": n_long_liq / max(n_total, 1),
            "short_liq_rate": n_short_liq / max(n_total, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading real RV30 index...")
    index = load_hourly_index()
    print(f"  {len(index):,} hourly observations  "
          f"| {index.index[0].date()} → {index.index[-1].date()}")
    print(f"  Index range (ann vol %): "
          f"{np.sqrt(index.min() * 365) * 100:.1f}% – "
          f"{np.sqrt(index.max() * 365) * 100:.1f}%")

    # --- Event study ---
    print("\n" + "=" * 70)
    print("EVENT STUDY — open 30d before event, hold 60d")
    print("=" * 70)
    ev_df = event_study(index)
    out_csv = PROCESSED_DIR / "margin_historical_events.csv"
    ev_df.to_csv(out_csv, index=False)
    print(ev_df.to_string(index=False,
                          float_format=lambda v: f"{v:8.3f}"))
    print(f"\nEvent table → {out_csv}")

    # --- Rolling sweep ---
    print("\n" + "=" * 70)
    print("ROLLING SWEEP — entry every 7d, hold 60d, across full history")
    print("=" * 70)
    sw_df = rolling_sweep(index)
    out_csv2 = PROCESSED_DIR / "margin_historical_rolling.csv"
    sw_df.to_csv(out_csv2, index=False)
    print(sw_df.to_string(index=False,
                          float_format=lambda v: f"{v:8.4f}"))
    print(f"\nRolling table → {out_csv2}")

    # --- Empirical margin calibration ---
    print("\n" + "=" * 70)
    print("EMPIRICAL MARGIN CALIBRATION — what IM/MM does history require?")
    print("=" * 70)
    exc = worst_excursions(index)
    print(f"  {len(exc)} rolling 60-day windows analyzed")
    print(f"  Worst long draw (index fell):  "
          f"min={exc['worst_long_draw'].min():.3f}  "
          f"median={exc['worst_long_draw'].median():.3f}")
    print(f"  Worst short draw (index rose): "
          f"max={exc['worst_short_draw'].max():.3f}  "
          f"median={exc['worst_short_draw'].median():.3f}")
    cal = calibrate_tiers(exc)
    print("\n  Required margin to survive X% of windows:")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    cal.to_csv(PROCESSED_DIR / "margin_historical_calibration.csv", index=False)

    # --- Gates ---
    print("\n" + "=" * 70)
    print("GATE CHECKS")
    print("=" * 70)

    # Gate 1: Tier 1 positions must survive COVID and FTX
    covid_ftx = ev_df[(ev_df["event"].isin(["COVID", "FTX"])) & (ev_df["tier"] == 1)]
    tier1_survived = (~covid_ftx["long_liq"]).all() and (~covid_ftx["short_liq"]).all()
    print(f"  {'✓' if tier1_survived else '✗'} Tier-1 (10×) survives COVID + FTX  "
          f"(long_liq={covid_ftx['long_liq'].any()}, short_liq={covid_ftx['short_liq'].any()})")

    # Gate 2: Rolling long liq rate at tier max < 5% across all tiers
    max_long = sw_df["long_liq_rate"].max()
    max_short = sw_df["short_liq_rate"].max()
    gate2 = max_long < 0.05 and max_short < 0.05
    print(f"  {'✓' if gate2 else '✗'} Rolling liq rate at tier-max < 5%  "
          f"(max long {max_long*100:.2f}%, max short {max_short*100:.2f}%)")

    # Gate 3: Top tier (2×) never liquidates in any event
    top = ev_df[ev_df["tier"] == 5]
    gate3 = (~top["long_liq"]).all() and (~top["short_liq"]).all()
    print(f"  {'✓' if gate3 else '✗'} Tier-5 (2×) survives all stress events")


if __name__ == "__main__":
    main()
