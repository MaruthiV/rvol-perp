"""Download Binance BTCUSDT perpetual 1m klines and 8h funding rates.

Uses data.binance.vision — Binance's public bulk data portal.
No API key required. No geo-restrictions. Bulk zip downloads, much faster
than the REST API.

URL patterns:
  Klines:       https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{YYYY-MM}.zip
  Funding:      https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{YYYY-MM}.zip

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
import io
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from rvol.pipeline.storage import FUNDING_SCHEMA, OHLCV_SCHEMA, write_parquet

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "binance"
DEFAULT_START = "2020-01-01"
DEFAULT_SYMBOL = "BTCUSDT"

BASE_URL = "https://data.binance.vision/data/futures/um/monthly"

KLINE_COLS = [
    "timestamp_us", "open", "high", "low", "close",
    "volume", "close_time_ms", "quote_volume", "n_trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

FUNDING_COLS = ["calc_time_ms", "funding_interval_hours", "last_funding_rate"]


def _month_range(start: str, end: str | None = None) -> list[str]:
    """Return list of YYYY-MM strings from start to end (inclusive)."""
    start_dt = datetime.strptime(start[:7], "%Y-%m").replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc) if end is None else \
        datetime.strptime(end[:7], "%Y-%m").replace(tzinfo=timezone.utc)

    months = []
    cur = start_dt
    while cur <= end_dt:
        months.append(cur.strftime("%Y-%m"))
        cur = cur.replace(month=cur.month % 12 + 1, year=cur.year + (1 if cur.month == 12 else 0))
    return months


def _download_zip(url: str) -> bytes | None:
    """Download a zip file, return raw bytes or None on 404."""
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        print(f"\n  Download failed ({url}): {e}")
        return None


def fetch_klines(symbol: str, start: str, end: str | None = None) -> None:
    """Download 1-minute OHLCV klines from data.binance.vision."""
    symbol_dir = DATA_DIR / symbol.lower() / "klines_1m"
    months = _month_range(start, end)

    print(f"Fetching {symbol} 1m klines: {months[0]} → {months[-1]} ({len(months)} months)")

    for ym in tqdm(months, desc="months"):
        out_path = symbol_dir / ym / "klines.parquet"
        if out_path.exists():
            continue

        url = f"{BASE_URL}/klines/{symbol}/1m/{symbol}-1m-{ym}.zip"
        raw = _download_zip(url)
        if raw is None:
            # Month not yet available (current month)
            continue

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                raw_df = pd.read_csv(f, header=None, names=KLINE_COLS)
            # data.binance.vision CSVs sometimes include a header row — drop it
            df = raw_df[raw_df.iloc[:, 0].astype(str).str.match(r"^\d+$")].copy()

        # timestamp in ms → us
        df["timestamp_us"] = df["timestamp_us"].astype(np.int64) * 1000
        df["n_trades"] = df["n_trades"].astype(np.int32)
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(np.float64)

        table = pa.Table.from_pandas(
            df[["timestamp_us", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]],
            schema=OHLCV_SCHEMA,
            preserve_index=False,
        )
        write_parquet(table, out_path, schema=OHLCV_SCHEMA)
        time.sleep(0.1)

    _validate_klines_gaps(symbol_dir)
    print(f"Klines done → {symbol_dir}")


def _validate_klines_gaps(base_dir: Path) -> None:
    """Print gap summary: any interval > 65s in the full timestamp sequence."""
    all_ts = []
    for pf in sorted(base_dir.rglob("*.parquet")):
        tbl = pq.read_table(pf, columns=["timestamp_us"])
        all_ts.extend(tbl.column("timestamp_us").to_pylist())

    if not all_ts:
        return

    ts_arr = np.array(sorted(set(all_ts)), dtype=np.int64)
    diffs_us = np.diff(ts_arr)
    gap_mask = diffs_us > 65_000_000  # 65s in microseconds
    n_gaps = int(gap_mask.sum())
    total = len(diffs_us)
    pct = n_gaps / total * 100
    print(f"Gap summary: {n_gaps}/{total} intervals > 65s ({pct:.4f}%)")
    if pct > 0.1:
        print("  WARNING: gap rate > 0.1% — check data quality")


def fetch_funding_rates(symbol: str, start: str = DEFAULT_START, end: str | None = None) -> None:
    """Download 8h funding rates from data.binance.vision."""
    symbol_dir = DATA_DIR / symbol.lower() / "funding_rates"
    months = _month_range(start, end)

    print(f"Fetching {symbol} funding rates: {months[0]} → {months[-1]}")

    all_rows: list[dict] = []
    for ym in tqdm(months, desc="funding"):
        url = f"{BASE_URL}/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        raw = _download_zip(url)
        if raw is None:
            continue

        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                raw_df = pd.read_csv(f, header=None, names=FUNDING_COLS)
            df = raw_df[raw_df.iloc[:, 0].astype(str).str.match(r"^\d+$")].copy()

        for _, row in df.iterrows():
            all_rows.append({
                "timestamp_us": int(row["calc_time_ms"]) * 1000,
                "funding_rate": float(row["last_funding_rate"]),
            })
        time.sleep(0.1)

    if not all_rows:
        print("No funding rate data retrieved.")
        return

    result_df = pd.DataFrame(sorted(all_rows, key=lambda x: x["timestamp_us"]))
    result_df = result_df.drop_duplicates(subset=["timestamp_us"])
    table = pa.Table.from_pandas(result_df, schema=FUNDING_SCHEMA, preserve_index=False)
    out_path = symbol_dir / "funding.parquet"
    write_parquet(table, out_path, schema=FUNDING_SCHEMA)
    print(f"Funding rates done → {out_path} ({len(result_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance perpetual data (via data.binance.vision)")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--klines-only", action="store_true")
    parser.add_argument("--funding-only", action="store_true")
    args = parser.parse_args()

    if not args.funding_only:
        fetch_klines(args.symbol, args.start, args.end)
    if not args.klines_only:
        fetch_funding_rates(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
