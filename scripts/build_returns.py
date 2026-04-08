"""Build clean 1-minute log return series from raw Binance klines.

Steps:
  1. Load all monthly kline parquet files from data/raw/binance/{symbol}/klines_1m/
  2. Sort by timestamp, deduplicate
  3. Compute 1-minute log returns
  4. Apply gap detection (intervals > 65s → NaN)
  5. Apply 4σ MAD-based return contribution cap
  6. Write RETURN_SCHEMA parquet to data/processed/returns/

Usage:
    python scripts/build_returns.py
    python scripts/build_returns.py --symbol ETHUSDT
    python scripts/build_returns.py --start 2022-01-01 --end 2024-12-31
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rvol.index.filters import cap_return_contributions
from rvol.index.returns import log_returns_from_prices
from rvol.index.spec import SPEC_30D
from rvol.pipeline.storage import RETURN_SCHEMA, write_parquet

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "binance"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed" / "returns"


def build_returns(symbol: str = "BTCUSDT", start: str | None = None, end: str | None = None) -> None:
    """Build and save the clean 1-minute log return series."""
    symbol_raw_dir = RAW_DIR / symbol.lower() / "klines_1m"
    if not symbol_raw_dir.exists():
        print(f"No klines data found at {symbol_raw_dir}")
        print("Run scripts/fetch_binance.py first.")
        return

    # Load all monthly partitions
    parquet_files = sorted(symbol_raw_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {symbol_raw_dir}")
        return

    print(f"Loading {len(parquet_files)} monthly partitions...")
    tables = []
    for pf in parquet_files:
        ym = pf.parent.name  # YYYY-MM
        if start and ym < start[:7]:
            continue
        if end and ym > end[:7]:
            continue
        tables.append(pq.read_table(pf, columns=["timestamp_us", "close"]))

    if not tables:
        print("No data in the specified date range.")
        return

    full_table = pa.concat_tables(tables)
    df = full_table.to_pandas().sort_values("timestamp_us").drop_duplicates(subset=["timestamp_us"])
    print(f"Loaded {len(df):,} rows ({df['timestamp_us'].min()} → {df['timestamp_us'].max()} us)")

    # Convert timestamps to DatetimeIndex for log_returns_from_prices
    ts = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    prices = pd.Series(df["close"].values, index=ts, name="price")

    # Compute log returns with gap detection
    print("Computing log returns...")
    rets = log_returns_from_prices(prices, freq="1min", handle_gaps=True)
    is_gap = rets.attrs.get("is_gap", np.zeros(len(rets), dtype=bool))

    # Apply MAD-based cap
    print("Applying manipulation-resistance cap...")
    raw_values = rets.values.copy()
    capped_values, is_capped = cap_return_contributions(
        raw_values,
        sigma_cap=SPEC_30D.sigma_cap,
        mad_lookback=SPEC_30D.mad_lookback_minutes,
    )

    n_gaps = int(np.sum(is_gap))
    n_capped = int(np.sum(is_capped))
    n_total = len(rets)
    cap_rate = n_capped / n_total * 100
    gap_rate = n_gaps / n_total * 100

    print(f"  Total returns:  {n_total:,}")
    print(f"  Gaps (NaN'd):   {n_gaps:,} ({gap_rate:.4f}%)")
    print(f"  Capped returns: {n_capped:,} ({cap_rate:.4f}%)")

    if cap_rate > 0.1:
        print(f"  WARNING: cap rate {cap_rate:.4f}% > 0.1% — check data quality")

    # Build output table
    ts_us = np.array(rets.index, dtype="datetime64[us]").astype(np.int64)
    result_df = pd.DataFrame({
        "timestamp_us": ts_us,
        "log_return": capped_values,
        "is_capped": is_capped,
        "is_gap": is_gap,
    })

    table = pa.Table.from_pandas(result_df, schema=RETURN_SCHEMA, preserve_index=False)

    out_name = f"{symbol.lower()}_1m_returns.parquet"
    if start or end:
        suffix = f"_{(start or 'start')[:7]}_{(end or 'end')[:7]}"
        out_name = f"{symbol.lower()}_1m_returns{suffix}.parquet"

    out_path = PROCESSED_DIR / out_name
    write_parquet(table, out_path, schema=RETURN_SCHEMA)
    print(f"Returns written → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 1m log return series from Binance klines")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    build_returns(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
