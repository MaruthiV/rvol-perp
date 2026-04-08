"""Log return computation from price series or trade data."""

import numpy as np
import pandas as pd


def log_returns_from_prices(
    prices: pd.Series,
    freq: str = "1min",
    handle_gaps: bool = True,
    gap_threshold_factor: float = 1.5,
) -> pd.Series:
    """Compute 1-period log returns from a price series.

    Parameters
    ----------
    prices:
        Price series. Index must be DatetimeTZAware (UTC) or int64 microseconds.
        If int64, the series is treated as microsecond timestamps.
    freq:
        Expected sampling frequency as a pandas offset string (e.g. "1min").
        Used only for gap detection when handle_gaps=True.
    handle_gaps:
        If True, returns where the prior price observation is more than
        gap_threshold_factor × freq seconds ago are set to NaN.
    gap_threshold_factor:
        Multiplier for gap detection. Default 1.5 → gaps > 90s are NaN'd
        for 1-minute freq.

    Returns
    -------
    pd.Series of log returns with the same index as prices[1:].
    The `is_gap` attribute is set on the returned series via `attrs`.
    """
    if prices.empty:
        return pd.Series(dtype=float, name="log_return")

    log_prices = np.log(prices.values.astype(float))
    raw_returns = np.diff(log_prices)

    # Build return index (same as prices[1:])
    return_index = prices.index[1:]
    result = pd.Series(raw_returns, index=return_index, name="log_return")

    if handle_gaps:
        gap_mask = _detect_gaps(prices.index, freq, gap_threshold_factor)
        result[gap_mask] = np.nan
        result.attrs["is_gap"] = gap_mask
    else:
        result.attrs["is_gap"] = np.zeros(len(result), dtype=bool)

    return result


def _detect_gaps(
    index: pd.Index,
    freq: str,
    gap_threshold_factor: float,
) -> np.ndarray:
    """Return boolean mask of length len(index)-1.

    True where the interval between consecutive timestamps exceeds
    gap_threshold_factor × freq.
    """
    if isinstance(index, pd.DatetimeIndex):
        # numpy array conversion handles tz-aware DatetimeIndex and normalizes
        # to microseconds regardless of stored resolution (ns or us in pandas 2.x).
        timestamps_us = np.array(index, dtype="datetime64[us]").astype(np.int64)
    else:
        timestamps_us = np.asarray(index, dtype=np.int64)

    diffs_us = np.diff(timestamps_us)

    # Convert freq string to microseconds via Timedelta
    freq_us = int(pd.Timedelta(freq).total_seconds() * 1_000_000)
    threshold_us = int(gap_threshold_factor * freq_us)

    return diffs_us > threshold_us


def resample_trades_to_ohlcv(
    trades: pd.DataFrame,
    freq: str = "1min",
) -> pd.DataFrame:
    """Aggregate tick-level trade data to OHLCV bars.

    Parameters
    ----------
    trades:
        DataFrame with columns: timestamp_us (int64 microseconds UTC),
        price (float64), size (float64).
    freq:
        Target bar frequency as a pandas offset string.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC) and columns:
        open, high, low, close, volume, vwap, n_trades.
    The `close` column uses VWAP within each bar as the representative price.
    """
    required = {"timestamp_us", "price", "size"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades DataFrame missing columns: {missing}")

    ts = pd.to_datetime(trades["timestamp_us"], unit="us", utc=True)
    df = trades[["price", "size"]].copy()
    df.index = ts
    df = df.sort_index()

    ohlcv = df["price"].resample(freq).ohlc()
    volume = df["size"].resample(freq).sum()
    n_trades = df["price"].resample(freq).count()
    # VWAP: sum(price × size) / sum(size)
    pv = (df["price"] * df["size"]).resample(freq).sum()
    sv = df["size"].resample(freq).sum()
    vwap = pv / sv

    result = ohlcv.copy()
    result["volume"] = volume
    result["vwap"] = vwap
    result["n_trades"] = n_trades

    # Drop bars with no trades
    result = result[result["n_trades"] > 0]

    return result
