"""Download Deribit DVOL (30-day implied volatility index) via CCXT.

DVOL is available for BTC and ETH since January 2019.
It represents 30-day annualized implied volatility (same convention as VIX).

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

from rvol.pipeline.storage import DVOL_SCHEMA, write_parquet

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "deribit"
DEFAULT_START = "2019-01-01"  # DVOL history starts ~Jan 2019


def fetch_dvol(coin: str, start: str, end: str | None = None) -> None:
    """Download DVOL daily OHLCV from Deribit via CCXT."""
    try:
        import ccxt  # type: ignore[import]
    except ImportError:
        print("ccxt not installed. Run: uv pip install ccxt")
        return

    deribit = ccxt.deribit()
    symbol = f"{coin.upper()}_USDC-DVOL"  # CCXT symbol format for Deribit DVOL

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc) if end is None else \
        datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    since_ms = int(start_dt.timestamp() * 1000)
    until_ms = int(end_dt.timestamp() * 1000)

    print(f"Fetching {coin} DVOL from Deribit ({start_dt.date()} → {end_dt.date()})...")

    all_rows = []
    current_ms = since_ms
    limit = 500  # Deribit returns up to 500 bars per request

    while current_ms < until_ms:
        try:
            ohlcv = deribit.fetch_ohlcv(
                symbol,
                timeframe="1d",
                since=current_ms,
                limit=limit,
            )
        except Exception as e:
            print(f"Error fetching DVOL at {current_ms}: {e}")
            time.sleep(2)
            break

        if not ohlcv:
            break

        for bar in ohlcv:
            ts_ms = bar[0]
            close_dvol = bar[4]  # close price = DVOL value
            if ts_ms > until_ms:
                break
            all_rows.append({
                "timestamp_us": ts_ms * 1000,  # ms → us
                "dvol_annualized": float(close_dvol),
            })

        last_ts_ms = ohlcv[-1][0]
        current_ms = last_ts_ms + 86_400_000  # advance 1 day
        time.sleep(0.3)

        if len(ohlcv) < limit:
            break

    if not all_rows:
        print("No DVOL data retrieved. Trying alternative symbol format...")
        _try_alternative_fetch(deribit, coin, since_ms, until_ms, all_rows)

    if not all_rows:
        print("Could not retrieve DVOL data. Check ccxt Deribit symbol format.")
        return

    df = pd.DataFrame(sorted(all_rows, key=lambda x: x["timestamp_us"]))
    df = df.drop_duplicates(subset=["timestamp_us"])

    coin_dir = DATA_DIR / coin.lower() / "dvol"
    table = pa.Table.from_pandas(df, schema=DVOL_SCHEMA, preserve_index=False)
    out_path = coin_dir / "dvol.parquet"
    write_parquet(table, out_path, schema=DVOL_SCHEMA)
    print(f"DVOL done → {out_path} ({len(df)} rows)")
    print(f"  Date range: {df['timestamp_us'].min()} to {df['timestamp_us'].max()}")
    print(f"  DVOL range: {df['dvol_annualized'].min():.1f}% - {df['dvol_annualized'].max():.1f}%")


def _try_alternative_fetch(
    deribit: "ccxt.deribit",
    coin: str,
    since_ms: int,
    until_ms: int,
    all_rows: list,
) -> None:
    """Try alternative CCXT symbol formats for DVOL if the primary one fails."""
    alternatives = [
        f"{coin.upper()}-DVOL",
        f"BTC_USDC-DVOL" if coin.upper() == "BTC" else f"ETH_USDC-DVOL",
    ]
    for sym in alternatives:
        try:
            ohlcv = deribit.fetch_ohlcv(sym, timeframe="1d", since=since_ms, limit=10)
            if ohlcv:
                print(f"  Symbol {sym} works — re-run with this symbol")
                return
        except Exception:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Deribit DVOL data")
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    fetch_dvol(args.coin, args.start, args.end)


if __name__ == "__main__":
    main()
