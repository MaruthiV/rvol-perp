"""Download Binance BTCUSDT perpetual 1m klines and 8h funding rates.

Usage:
    python scripts/fetch_binance.py                          # klines + funding
    python scripts/fetch_binance.py --klines-only
    python scripts/fetch_binance.py --funding-only
    python scripts/fetch_binance.py --start 2022-01-01
    python scripts/fetch_binance.py --symbol ETHUSDT

Output:
    data/raw/binance/{symbol_lower}/klines_1m/YYYY-MM/klines.parquet
    data/raw/binance/{symbol_lower}/funding_rates/funding.parquet
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from rvol.pipeline.storage import FUNDING_SCHEMA, OHLCV_SCHEMA, write_parquet

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "binance"
DEFAULT_START = "2020-01-01"
DEFAULT_SYMBOL = "BTCUSDT"
KLINE_INTERVAL = "1m"
KLINE_LIMIT = 1500  # Binance max per request


def fetch_klines(symbol: str, start: str, end: str | None = None) -> None:
    """Download 1-minute OHLCV klines and write to monthly partitioned parquet."""
    from binance.client import Client

    client = Client()
    symbol_dir = DATA_DIR / symbol.lower() / "klines_1m"

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc) if end is None else datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"Fetching {symbol} 1m klines from {start_dt.date()} to {end_dt.date()}")

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows: list[dict] = []
    current_ms = start_ms
    current_month: str | None = None

    with tqdm(total=int((end_ms - start_ms) / (60 * 1000)), unit="min", desc="klines") as pbar:
        while current_ms < end_ms:
            try:
                klines = client.get_historical_klines(
                    symbol,
                    KLINE_INTERVAL,
                    start_str=current_ms,
                    end_str=end_ms,
                    limit=KLINE_LIMIT,
                )
            except Exception as e:
                print(f"\nError fetching klines at {current_ms}: {e}. Retrying in 5s...")
                time.sleep(5)
                continue

            if not klines:
                break

            for k in klines:
                row = {
                    "timestamp_us": int(k[0]) * 1000,  # ms → us
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[7]),
                    "n_trades": int(k[8]),
                }
                all_rows.append(row)

            last_ts_ms = klines[-1][0]
            pbar.update(len(klines))
            current_ms = last_ts_ms + 60_000  # advance by 1 minute

            # Flush monthly partition when month changes
            month = datetime.fromtimestamp(last_ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m")
            if current_month is None:
                current_month = month
            if month != current_month:
                _flush_klines(all_rows, symbol_dir, current_month)
                all_rows = []
                current_month = month

            # Rate limit: 1200 weight per minute; each klines request = 2 weight
            time.sleep(0.15)

    # Flush remaining
    if all_rows and current_month:
        _flush_klines(all_rows, symbol_dir, current_month)

    _validate_klines_gaps(symbol_dir)
    print(f"Klines done → {symbol_dir}")


def _flush_klines(rows: list[dict], base_dir: Path, month: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, schema=OHLCV_SCHEMA, preserve_index=False)
    path = base_dir / month / "klines.parquet"
    write_parquet(table, path, schema=OHLCV_SCHEMA)


def _validate_klines_gaps(base_dir: Path) -> None:
    """Print gap summary: any interval > 65s is flagged."""
    all_ts = []
    for parquet_file in sorted(base_dir.rglob("*.parquet")):
        tbl = pq.read_table(parquet_file, columns=["timestamp_us"])
        all_ts.extend(tbl.column("timestamp_us").to_pylist())

    if not all_ts:
        return

    all_ts = sorted(set(all_ts))
    ts_arr = np.array(all_ts, dtype=np.int64)
    diffs_us = np.diff(ts_arr)
    gap_mask = diffs_us > 65_000_000  # 65s in us
    n_gaps = int(gap_mask.sum())
    total = len(diffs_us)
    print(f"Gap summary: {n_gaps}/{total} intervals > 65s ({n_gaps / total * 100:.3f}%)")
    if n_gaps > 0 and n_gaps / total > 0.001:
        print("  WARNING: gap rate > 0.1% — check data quality")


def fetch_funding_rates(symbol: str) -> None:
    """Download 8h perpetual funding rates."""
    from binance.client import Client

    client = Client()
    symbol_dir = DATA_DIR / symbol.lower() / "funding_rates"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {symbol} funding rates...")
    all_rates = []
    end_time = None

    while True:
        try:
            kwargs: dict = {"symbol": symbol, "limit": 1000}
            if end_time is not None:
                kwargs["endTime"] = end_time
            rates = client.get_funding_rate(**kwargs)
        except Exception as e:
            print(f"Error fetching funding rates: {e}")
            break

        if not rates:
            break

        for r in rates:
            all_rates.append({
                "timestamp_us": int(r["fundingTime"]) * 1000,
                "funding_rate": float(r["fundingRate"]),
            })

        oldest_ts = rates[0]["fundingTime"]
        end_time = oldest_ts - 1
        time.sleep(0.2)

        if len(rates) < 1000:
            break

    if not all_rates:
        print("No funding rate data retrieved.")
        return

    df = pd.DataFrame(sorted(all_rates, key=lambda x: x["timestamp_us"]))
    table = pa.Table.from_pandas(df, schema=FUNDING_SCHEMA, preserve_index=False)
    path = symbol_dir / "funding.parquet"
    write_parquet(table, path, schema=FUNDING_SCHEMA)
    print(f"Funding rates done → {path} ({len(all_rates)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance perpetual data")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--klines-only", action="store_true")
    parser.add_argument("--funding-only", action="store_true")
    args = parser.parse_args()

    if not args.funding_only:
        fetch_klines(args.symbol, args.start, args.end)
    if not args.klines_only:
        fetch_funding_rates(args.symbol)


if __name__ == "__main__":
    main()
