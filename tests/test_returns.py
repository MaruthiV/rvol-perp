"""Tests for log return computation."""

import numpy as np
import pandas as pd
import pytest

from rvol.index.returns import log_returns_from_prices, resample_trades_to_ohlcv


def _make_price_series(prices, start="2024-01-01", freq="1min"):
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.Series(prices, index=idx, name="price")


def test_flat_prices_give_zero_returns(flat_price_series):
    rets = log_returns_from_prices(flat_price_series)
    assert np.allclose(rets.dropna(), 0.0)


def test_log_return_formula_correctness():
    prices = _make_price_series([100.0, 110.0, 99.0])
    rets = log_returns_from_prices(prices, handle_gaps=False)
    expected = [np.log(110.0 / 100.0), np.log(99.0 / 110.0)]
    assert np.allclose(rets.values, expected)


def test_return_length_is_n_minus_1():
    prices = _make_price_series(np.random.default_rng(1).uniform(100, 200, 50))
    rets = log_returns_from_prices(prices, handle_gaps=False)
    assert len(rets) == len(prices) - 1


def test_gap_detection_creates_nan(gapped_price_series):
    rets = log_returns_from_prices(gapped_price_series, freq="1min", handle_gaps=True)
    # The return immediately after the 2-hour gap should be NaN
    nan_positions = rets.index[rets.isna()]
    assert len(nan_positions) >= 1


def test_no_gap_in_regular_series(flat_price_series):
    rets = log_returns_from_prices(flat_price_series, freq="1min", handle_gaps=True)
    # Regular 1-minute spacing → no gaps
    assert not rets.isna().any()


def test_is_gap_attr_populated(gapped_price_series):
    rets = log_returns_from_prices(gapped_price_series, freq="1min", handle_gaps=True)
    assert "is_gap" in rets.attrs
    assert rets.attrs["is_gap"].sum() >= 1


def test_empty_series():
    empty = pd.Series(dtype=float, name="price")
    rets = log_returns_from_prices(empty)
    assert len(rets) == 0


def test_two_prices_returns_one_return():
    prices = _make_price_series([50000.0, 50100.0])
    rets = log_returns_from_prices(prices, handle_gaps=False)
    assert len(rets) == 1
    assert np.isclose(rets.iloc[0], np.log(50100.0 / 50000.0))


# --- resample_trades_to_ohlcv ---

def test_ohlcv_columns_present(synthetic_trade_df):
    ohlcv = resample_trades_to_ohlcv(synthetic_trade_df, freq="1min")
    for col in ["open", "high", "low", "close", "volume", "vwap", "n_trades"]:
        assert col in ohlcv.columns


def test_ohlcv_high_gte_low(synthetic_trade_df):
    ohlcv = resample_trades_to_ohlcv(synthetic_trade_df, freq="1min")
    assert (ohlcv["high"] >= ohlcv["low"]).all()


def test_ohlcv_vwap_in_range(synthetic_trade_df):
    ohlcv = resample_trades_to_ohlcv(synthetic_trade_df, freq="1min")
    assert (ohlcv["vwap"] >= ohlcv["low"]).all()
    assert (ohlcv["vwap"] <= ohlcv["high"]).all()


def test_ohlcv_missing_columns_raises():
    bad_df = pd.DataFrame({"timestamp_us": [1], "price": [100.0]})  # missing 'size'
    with pytest.raises(ValueError, match="missing columns"):
        resample_trades_to_ohlcv(bad_df)
