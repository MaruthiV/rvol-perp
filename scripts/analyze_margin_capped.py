"""Phase 8.6: margin calibration under the 2.5× payoff cap.

Same structure as `analyze_margin_historical.py`, but with position PnL
clipped at ±(cap−1, −1) per the capped variance swap design. Runs the
rolling 60-day sweep with the cap active, reports liquidation rates per
tier, and sweeps IM rates to find the minimum collateral at which each
side survives 99% of historical windows.

Outputs:
  - data/processed/margin_capped_rolling.csv
  - data/processed/margin_capped_calibration.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rvol.funding.rate import FundingParams
from rvol.funding.simulate import BasisProcess, simulate_mark_convergence
from rvol.margin import PAYOFF_CAP, TIERS

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "index" / "btcusdt_rv_index.parquet"
WINDOW_DAYS = 30


def load_hourly_index() -> pd.Series:
    tbl = pq.read_table(INDEX_PATH, columns=["timestamp_us", "rv30", "is_valid"])
    df = tbl.to_pandas()
    df = df[df["is_valid"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp_us"], unit="us")
    df = df.set_index("timestamp")
    df["index_daily_var"] = df["rv30"] / WINDOW_DAYS
    return df["index_daily_var"].resample("1h").last().dropna()


def capped_path_pnl(index_slice: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (long_equity_frac_over_time, short_equity_frac_over_time, funding).

    Equity starts at 0 (before collateral is added); liquidation test is
    made by the caller against `IM + equity vs MM`. All quantities are
    fractions of entry index (same units as IM/MM rates).
    """
    fp = FundingParams(dampening=100.0, cap=0.001, interval_hours=1)
    bp = BasisProcess(rho=0.90, sigma=0.005, gamma=10.0, initial_basis=0.0)
    res = simulate_mark_convergence(index_slice, fp, bp, seed=0)

    idx = index_slice.to_numpy(dtype=np.float64)
    I0 = float(idx[0])
    if I0 <= 0:
        z = np.zeros_like(idx)
        return z, z, z

    raw = (idx - I0) / I0
    clipped = np.clip(raw, -1.0, PAYOFF_CAP - 1.0)
    # Cumulative funding paid (by longs) as a fraction of I0
    funding = res.funding.to_numpy()
    cum_funding = np.cumsum(funding)

    long_eq = clipped - cum_funding
    short_eq = -clipped + cum_funding
    return long_eq, short_eq, funding


def tier_liq(index: pd.Series, im: float, mm: float,
             step_days: int = 2, hold_days: int = 7) -> tuple[float, float, int]:
    entries = pd.date_range(
        start=index.index[0],
        end=index.index[-1] - pd.Timedelta(days=hold_days),
        freq=f"{step_days}D",
    )
    n_total = 0
    n_long_liq = 0
    n_short_liq = 0
    for e in entries:
        exit = e + pd.Timedelta(days=hold_days)
        window = index.loc[e:exit]
        if len(window) < 24 * hold_days // 2:
            continue
        long_eq, short_eq, _ = capped_path_pnl(window)
        n_total += 1
        if np.any(im + long_eq < mm):
            n_long_liq += 1
        if np.any(im + short_eq < mm):
            n_short_liq += 1
    return n_long_liq / max(n_total, 1), n_short_liq / max(n_total, 1), n_total


def main() -> None:
    print(f"Loading index...  (payoff cap = {PAYOFF_CAP}× entry strike)")
    index = load_hourly_index()
    print(f"  {len(index):,} hourly obs  "
          f"| {index.index[0].date()} → {index.index[-1].date()}")

    # --- Rolling sweep with the current (Phase 8.6) tier table ---
    print("\n" + "=" * 70)
    print("ROLLING SWEEP — capped payoff, current tier table")
    print("=" * 70)
    rows = []
    for t in TIERS:
        ll, sl, n = tier_liq(index, t.initial_margin_rate, t.maintenance_margin_rate)
        rows.append({
            "tier": t.index,
            "max_vega_usd": t.max_vega_notional_usd,
            "im_rate": t.initial_margin_rate,
            "mm_rate": t.maintenance_margin_rate,
            "max_leverage": t.max_leverage,
            "n_positions": n,
            "long_liq_rate": ll,
            "short_liq_rate": sl,
        })
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_DIR / "margin_capped_rolling.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # --- Calibration sweep: small MM, sweep IM ---
    print("\n" + "=" * 70)
    print("IM CALIBRATION — sweep IM with small fixed MM = 0.05")
    print("=" * 70)
    im_grid = [0.20, 0.25, 0.33, 0.40, 0.50, 0.67, 0.80, 1.00, 1.25, 1.50]
    cal_rows = []
    for im in im_grid:
        mm = 0.05
        ll, sl, n = tier_liq(index, im, mm)
        cal_rows.append({
            "im": im, "mm": mm, "leverage": 1.0 / im,
            "long_liq_rate": ll, "short_liq_rate": sl, "n": n,
        })
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(PROCESSED_DIR / "margin_capped_calibration.csv", index=False)
    print(cal.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # --- Gate ---
    print("\n" + "=" * 70)
    print("GATE CHECK — tier-max liq rate < 5% (both sides)")
    print("=" * 70)
    max_ll = df["long_liq_rate"].max()
    max_sl = df["short_liq_rate"].max()
    ok = max_ll < 0.05 and max_sl < 0.05
    print(f"  {'✓' if ok else '✗'} max long {max_ll*100:.2f}%  "
          f"max short {max_sl*100:.2f}%")


if __name__ == "__main__":
    main()
