"""Build rolling realized variance index from the processed returns series.

Computes 7-day, 14-day, and 30-day rolling realized variance for each
minute in the return series and writes to data/processed/index/.

Sanity checks are printed at the end:
  - COVID crash (March 2020): RV30 should spike significantly
  - FTX collapse (November 2022): another RV spike
  - ETF approval rally (January 2024): moderate vol increase

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --symbol BTCUSDT
    python scripts/build_index.py --hourly    # compute at 1h resolution (faster)

Output:
    data/processed/index/{symbol_lower}_rv_index.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rvol.index.spec import SPEC_7D, SPEC_14D, SPEC_30D
from rvol.index.variance import rolling_realized_variance
from rvol.pipeline.storage import INDEX_SCHEMA, RETURN_SCHEMA, write_parquet

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RETURNS_DIR = PROCESSED_DIR / "returns"
INDEX_DIR = PROCESSED_DIR / "index"


def build_index(symbol: str = "BTCUSDT", hourly: bool = False) -> None:
    """Compute rolling RV index and write to parquet."""
    returns_path = RETURNS_DIR / f"{symbol.lower()}_1m_returns.parquet"
    if not returns_path.exists():
        print(f"Returns file not found: {returns_path}")
        print("Run scripts/build_returns.py first.")
        return

    print(f"Loading returns from {returns_path}...")
    table = pq.read_table(returns_path)
    df = table.to_pandas()

    ts = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    rets = pd.Series(df["log_return"].values, index=ts, name="log_return")

    if hourly:
        # Downsample to hourly for faster iteration during development
        # RV7 and RV14 are computed at hourly resolution (less precise but fast)
        print("Downsampling to hourly resolution...")
        rets = rets.resample("1h").apply(lambda x: np.sqrt(np.nansum(x**2)))
        print(f"  {len(rets):,} hourly returns")
    else:
        print(f"  {len(rets):,} 1-minute returns")

    print("Computing RV7...")
    df7 = rolling_realized_variance(rets, SPEC_7D)
    print("Computing RV14...")
    df14 = rolling_realized_variance(rets, SPEC_14D)
    print("Computing RV30...")
    df30 = rolling_realized_variance(rets, SPEC_30D)

    # Merge into INDEX_SCHEMA
    ts_us = np.array(rets.index, dtype="datetime64[us]").astype(np.int64)
    result_df = pd.DataFrame({
        "timestamp_us": ts_us,
        "rv7": df7["rv"].values,
        "rv14": df14["rv"].values,
        "rv30": df30["rv"].values,
        "n_obs": df30["n_obs"].values,
        "n_capped": df30["n_capped"].values,
        "is_valid": df30["is_valid"].values,
    })

    table = pa.Table.from_pandas(result_df, schema=INDEX_SCHEMA, preserve_index=False)
    out_path = INDEX_DIR / f"{symbol.lower()}_rv_index.parquet"
    write_parquet(table, out_path, schema=INDEX_SCHEMA)
    print(f"Index written → {out_path}")

    # Sanity checks
    _print_sanity_checks(result_df, symbol)


def _print_sanity_checks(df: pd.DataFrame, symbol: str) -> None:
    """Print spot-check stats for known high-volatility periods."""
    ts = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df = df.copy()
    df.index = ts

    annualization_30 = 365 / 30
    df["rv30_annvol"] = np.sqrt(df["rv30"] * annualization_30) * 100  # as % annualized vol

    print("\nSanity checks (RV30 annualized vol equivalent):")
    print(f"  Overall median: {df['rv30_annvol'].median():.1f}%  (expected ~40-60%)")

    # Key events
    events = {
        "COVID crash (Mar 2020)": ("2020-03-01", "2020-03-31"),
        "FTX collapse (Nov 2022)": ("2022-11-01", "2022-11-30"),
        "ETF approval rally (Jan 2024)": ("2024-01-01", "2024-01-31"),
    }
    for event_name, (start, end) in events.items():
        mask = (df.index >= start) & (df.index <= end)
        if mask.any():
            peak = df.loc[mask, "rv30_annvol"].max()
            print(f"  {event_name}: peak {peak:.1f}%")
        else:
            print(f"  {event_name}: no data in range")

    valid_pct = df["is_valid"].mean() * 100
    print(f"\n  Valid index fraction: {valid_pct:.2f}%  (expected >95%)")

    rv30_range = df["rv30"].quantile([0.01, 0.99])
    print(f"  RV30 1st-99th pct: [{rv30_range.iloc[0]:.4f}, {rv30_range.iloc[1]:.4f}]")
    print(f"  (Expected range: [0.001, 0.15] for BTC)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rolling RV index")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--hourly",
        action="store_true",
        help="Compute at 1h resolution (faster, for development)",
    )
    args = parser.parse_args()

    build_index(args.symbol, args.hourly)


if __name__ == "__main__":
    main()
