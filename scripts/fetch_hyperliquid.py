"""Download Hyperliquid BTC trade data.

Strategy:
  - Historical (> 7 days ago): download from public S3 archive bucket.
    Bucket path: s3://hyperliquid-archive/market_data/{coin}/trades/{YYYY-MM}.parquet.gz
    No AWS credentials required (public bucket, anonymous access).
  - Recent (last 7 days): use official hyperliquid-python-sdk Info.trades_by_coin().

Usage:
    python scripts/fetch_hyperliquid.py
    python scripts/fetch_hyperliquid.py --coin ETH
    python scripts/fetch_hyperliquid.py --start 2024-01-01
    python scripts/fetch_hyperliquid.py --recent-only   # only last 7 days via SDK

Output:
    data/raw/hyperliquid/{coin_lower}/trades/YYYY-MM/trades.parquet
"""

import argparse
import gzip
import io
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from tqdm import tqdm

from rvol.pipeline.storage import TRADE_SCHEMA, write_parquet

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "hyperliquid"

# HL launched March 2023; before this there is no data
HL_LAUNCH_DATE = "2023-03"
HL_API_URL = "https://api.hyperliquid.xyz"

# S3 bucket — public, no credentials needed
S3_BASE = "https://hyperliquid-archive.s3.amazonaws.com/market_data/{coin}/trades/{year_month}.parquet.gz"


def fetch_historical_s3(coin: str, start_ym: str, end_ym: str | None = None) -> None:
    """Download monthly parquet.gz files from the public S3 archive."""
    coin_dir = DATA_DIR / coin.lower() / "trades"

    # Enumerate months from start_ym to now
    start_dt = datetime.strptime(start_ym + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc).replace(day=1) if end_ym is None else \
        datetime.strptime(end_ym + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)

    months = []
    current = start_dt
    while current <= end_dt:
        months.append(current.strftime("%Y-%m"))
        # Advance by 1 month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    print(f"Fetching {coin} historical trades from S3: {len(months)} months")

    for ym in tqdm(months, desc="months"):
        out_path = coin_dir / ym / "trades.parquet"
        if out_path.exists():
            continue  # already downloaded

        url = S3_BASE.format(coin=coin, year_month=ym)
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 404:
                print(f"\n  {ym}: not found in S3 (may not exist yet)")
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"\n  {ym}: download failed — {e}")
            continue

        # Decompress and parse
        try:
            with gzip.open(io.BytesIO(resp.content)) as f:
                raw_bytes = f.read()
            import pyarrow.parquet as pq
            table = pq.read_table(io.BytesIO(raw_bytes))
            table = _normalize_hl_schema(table, ym)
            write_parquet(table, out_path, schema=TRADE_SCHEMA)
        except Exception as e:
            print(f"\n  {ym}: parse failed — {e}")
            continue

        time.sleep(0.5)  # polite delay

    print(f"Historical download done → {coin_dir}")


def _normalize_hl_schema(table: pa.Table, ym: str) -> pa.Table:
    """Normalize Hyperliquid S3 data to TRADE_SCHEMA.

    The S3 parquet schema may differ across months as Hyperliquid evolves
    their data format. This function handles the most common variants.
    """
    df = table.to_pandas()

    # Timestamp: could be 'time', 'timestamp', 'ts', 'time_ms' etc.
    ts_col = next(
        (c for c in df.columns if "time" in c.lower() or c == "ts"),
        None,
    )
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {ym} data. Columns: {list(df.columns)}")

    # Convert to microseconds
    ts = df[ts_col].values.astype(np.int64)
    # Detect unit: if max value suggests milliseconds (13-digit), convert
    if ts.max() < 2_000_000_000_000:  # ms (13 digits)
        ts = ts * 1000
    elif ts.max() < 2_000_000_000_000_000:  # us (16 digits), already correct
        pass
    else:  # ns (19 digits)
        ts = ts // 1000

    # Price column
    price_col = next((c for c in df.columns if "price" in c.lower() or c == "px"), None)
    if price_col is None:
        raise ValueError(f"No price column in {ym}. Columns: {list(df.columns)}")

    # Size column
    size_col = next(
        (c for c in df.columns if c in ("size", "sz", "qty", "amount", "quantity")),
        None,
    )
    if size_col is None:
        raise ValueError(f"No size column in {ym}. Columns: {list(df.columns)}")

    # Side column (optional)
    side_col = next((c for c in df.columns if c in ("side", "dir", "direction")), None)
    side_values = df[side_col].astype(str).tolist() if side_col else [""] * len(df)

    result_df = pd.DataFrame({
        "timestamp_us": ts,
        "price": df[price_col].astype(float),
        "size": df[size_col].astype(float),
        "side": side_values,
    })

    # Drop invalid rows
    result_df = result_df[(result_df["price"] > 0) & (result_df["size"] > 0)]
    result_df = result_df.sort_values("timestamp_us").reset_index(drop=True)

    return pa.Table.from_pandas(result_df, schema=TRADE_SCHEMA, preserve_index=False)


def fetch_recent_sdk(coin: str, days: int = 7) -> None:
    """Fetch recent trades via the official Hyperliquid Python SDK."""
    try:
        from hyperliquid.info import Info  # type: ignore[import]
    except ImportError:
        print("hyperliquid-python-sdk not installed. Run: uv pip install hyperliquid-python-sdk")
        return

    info = Info(base_url=HL_API_URL)
    coin_dir = DATA_DIR / coin.lower() / "trades"

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    print(f"Fetching {coin} recent trades via SDK ({days} days)...")

    all_rows = []
    chunk_ms = 3600 * 1000  # 1-hour chunks to avoid hitting limits
    current = start_ms

    with tqdm(total=(end_ms - start_ms) // 1000, unit="s") as pbar:
        while current < end_ms:
            chunk_end = min(current + chunk_ms, end_ms)
            try:
                trades = info.trades_by_coin(coin, startTime=current, endTime=chunk_end)
            except Exception as e:
                print(f"\nSDK error: {e}. Skipping chunk.")
                current = chunk_end
                continue

            for t in trades:
                all_rows.append({
                    "timestamp_us": int(t.get("time", t.get("timestamp", 0))) * 1000,
                    "price": float(t.get("px", t.get("price", 0))),
                    "size": float(t.get("sz", t.get("size", 0))),
                    "side": str(t.get("side", t.get("dir", ""))),
                })

            pbar.update((chunk_end - current) // 1000)
            current = chunk_end
            time.sleep(0.1)

    if not all_rows:
        print("No recent trade data retrieved.")
        return

    df = pd.DataFrame(all_rows)
    df = df[(df["price"] > 0) & (df["size"] > 0)].sort_values("timestamp_us")

    # Write to current month partition
    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    out_path = coin_dir / current_ym / "recent_trades.parquet"
    table = pa.Table.from_pandas(df, schema=TRADE_SCHEMA, preserve_index=False)
    write_parquet(table, out_path, schema=TRADE_SCHEMA)
    print(f"Recent trades done → {out_path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid trade data")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", default=HL_LAUNCH_DATE, help="Start YYYY-MM")
    parser.add_argument("--end", default=None, help="End YYYY-MM (inclusive)")
    parser.add_argument("--recent-only", action="store_true", help="Only fetch last 7 days via SDK")
    parser.add_argument("--recent-days", type=int, default=7)
    args = parser.parse_args()

    if not args.recent_only:
        fetch_historical_s3(args.coin, args.start, args.end)
    fetch_recent_sdk(args.coin, days=args.recent_days)


if __name__ == "__main__":
    main()
