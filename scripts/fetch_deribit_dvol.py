"""Download Deribit DVOL (30-day implied volatility index) via direct REST API.

Uses /public/get_volatility_index_data — not CCXT (CCXT doesn't expose the index).
Data format: [timestamp_ms, open, high, low, close] — we use close as the DVOL value.
Resolution: 3600 (1-hour bars). DVOL available from ~Jan 2020 for BTC.

Usage:
    python scripts/fetch_deribit_dvol.py
    python scripts/fetch_deribit_dvol.py --coin ETH
    python scripts/fetch_deribit_dvol.py --start 2021-01-01

Output:
    data/raw/deribit/{coin_lower}/dvol/dvol.parquet
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import requests
from tqdm import tqdm

from rvol.pipeline.storage import DVOL_SCHEMA, write_parquet

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "deribit"
BASE_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"

# DVOL available from Jan 2020 (before that returns empty)
DEFAULT_START = "2020-01-01"
# Fetch 30-day chunks at hourly resolution (720 bars per request)
CHUNK_DAYS = 30
RESOLUTION = 3600  # 1-hour bars


def fetch_dvol(coin: str, start: str, end: str | None = None) -> None:
    start_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int((datetime.now(timezone.utc) if end is None else
                  datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)).timestamp() * 1000)

    chunk_ms = CHUNK_DAYS * 24 * 3600 * 1000
    n_chunks = max(1, (end_ms - start_ms) // chunk_ms + 1)

    print(f"Fetching {coin} DVOL from Deribit ({start} → {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()})...")

    all_rows: list[dict] = []
    current_ms = start_ms

    with tqdm(total=n_chunks, unit="chunk") as pbar:
        while current_ms < end_ms:
            chunk_end = min(current_ms + chunk_ms, end_ms)
            params = {
                "currency": coin.upper(),
                "start_timestamp": current_ms,
                "end_timestamp": chunk_end,
                "resolution": RESOLUTION,
            }
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                result = resp.json().get("result", {})
                bars = result.get("data", [])
            except Exception as e:
                print(f"\n  Error at {current_ms}: {e}")
                current_ms = chunk_end
                pbar.update(1)
                time.sleep(1)
                continue

            for bar in bars:
                ts_ms, _, _, _, close = bar
                all_rows.append({
                    "timestamp_us": int(ts_ms) * 1000,
                    "dvol_annualized": float(close),
                })

            current_ms = chunk_end
            pbar.update(1)
            time.sleep(0.2)

    if not all_rows:
        print("No data retrieved. DVOL may only be available from 2020-01-01 onward.")
        return

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["timestamp_us"]).sort_values("timestamp_us")

    coin_dir = DATA_DIR / coin.lower() / "dvol"
    table = pa.Table.from_pandas(df, schema=DVOL_SCHEMA, preserve_index=False)
    out_path = coin_dir / "dvol.parquet"
    write_parquet(table, out_path, schema=DVOL_SCHEMA)

    start_dt = datetime.fromtimestamp(df["timestamp_us"].min() / 1_000_000, tz=timezone.utc).date()
    end_dt = datetime.fromtimestamp(df["timestamp_us"].max() / 1_000_000, tz=timezone.utc).date()
    print(f"Done → {out_path}")
    print(f"  {len(df)} hourly bars  |  {start_dt} → {end_dt}")
    print(f"  DVOL range: {df['dvol_annualized'].min():.1f}% – {df['dvol_annualized'].max():.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Deribit DVOL index")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    fetch_dvol(args.coin, args.start, args.end)


if __name__ == "__main__":
    main()
