"""Download Hyperliquid BTC funding rate history via the official SDK.

NOTE: HL data is supplementary. Binance 1m klines are the primary data
source for realized variance computation. HL funding rates are used for:
  - Phase 3 correlation analysis (HL funding vs realized vol)
  - Phase 9 HIP-3 deployment proposal context

candles_snapshot() appears restricted for historical data — skip it.
Funding history works fine and is the more important dataset for the paper.

Usage:
    python scripts/fetch_hyperliquid.py
    python scripts/fetch_hyperliquid.py --coin ETH
    python scripts/fetch_hyperliquid.py --start 2024-01-01

Output:
    data/raw/hyperliquid/{coin_lower}/funding_rates/funding.parquet
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa

from rvol.pipeline.storage import FUNDING_SCHEMA, write_parquet

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "hyperliquid"
HL_API_URL = "https://api.hyperliquid.xyz"
HL_LAUNCH_DATE = "2023-03-01"


def fetch_funding_rates(coin: str, start: str) -> None:
    """Download HL perpetual funding rate history."""
    from hyperliquid.info import Info

    info = Info(base_url=HL_API_URL)
    coin_dir = DATA_DIR / coin.lower() / "funding_rates"

    start_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    print(f"Fetching {coin} funding rate history from HL (since {start})...")
    try:
        rates = info.funding_history(coin, startTime=start_ms)
    except Exception as e:
        print(f"  Error: {e}")
        return

    if not rates:
        print("  No funding rate data returned.")
        return

    rows = [
        {
            "timestamp_us": int(r["time"]) * 1000,
            "funding_rate": float(r["fundingRate"]),
        }
        for r in rates
    ]

    df = pd.DataFrame(rows).sort_values("timestamp_us").drop_duplicates(subset=["timestamp_us"])
    table = pa.Table.from_pandas(df, schema=FUNDING_SCHEMA, preserve_index=False)
    out_path = coin_dir / "funding.parquet"
    write_parquet(table, out_path, schema=FUNDING_SCHEMA)
    print(f"Done → {out_path} ({len(df)} rows, {df['timestamp_us'].min()} → {df['timestamp_us'].max()} us)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid funding rate history")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", default=HL_LAUNCH_DATE)
    args = parser.parse_args()

    fetch_funding_rates(args.coin, args.start)


if __name__ == "__main__":
    main()
