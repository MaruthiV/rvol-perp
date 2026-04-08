"""Schema round-trip tests for storage.py."""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from rvol.pipeline.storage import (
    DVOL_SCHEMA,
    FUNDING_SCHEMA,
    INDEX_SCHEMA,
    OHLCV_SCHEMA,
    RETURN_SCHEMA,
    TRADE_SCHEMA,
    list_monthly_partitions,
    list_parquet_files,
    read_parquet,
    write_parquet,
)


def _make_trade_table(n: int = 10) -> pa.Table:
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "timestamp_us": np.arange(n, dtype=np.int64) * 60_000_000 + 1_700_000_000_000_000,
            "price": rng.uniform(40_000, 60_000, n),
            "size": rng.uniform(0.01, 2.0, n),
            "side": ["buy" if i % 2 == 0 else "sell" for i in range(n)],
        },
        schema=TRADE_SCHEMA,
    )


def _make_return_table(n: int = 10) -> pa.Table:
    rng = np.random.default_rng(1)
    return pa.table(
        {
            "timestamp_us": np.arange(n, dtype=np.int64) * 60_000_000 + 1_700_000_000_000_000,
            "log_return": rng.normal(0, 0.001, n),
            "is_capped": np.zeros(n, dtype=bool),
            "is_gap": np.zeros(n, dtype=bool),
        },
        schema=RETURN_SCHEMA,
    )


def _make_index_table(n: int = 10) -> pa.Table:
    rng = np.random.default_rng(2)
    return pa.table(
        {
            "timestamp_us": np.arange(n, dtype=np.int64) * 60_000_000 + 1_700_000_000_000_000,
            "rv7": rng.uniform(0.001, 0.01, n),
            "rv14": rng.uniform(0.001, 0.01, n),
            "rv30": rng.uniform(0.001, 0.01, n),
            "n_obs": np.full(n, 43200, dtype=np.int32),
            "n_capped": np.zeros(n, dtype=np.int32),
            "is_valid": np.ones(n, dtype=bool),
        },
        schema=INDEX_SCHEMA,
    )


# --- Schema round-trip tests ---

@pytest.mark.parametrize(
    "make_table, schema, name",
    [
        (_make_trade_table, TRADE_SCHEMA, "TRADE"),
        (_make_return_table, RETURN_SCHEMA, "RETURN"),
        (_make_index_table, INDEX_SCHEMA, "INDEX"),
    ],
)
def test_schema_roundtrip(make_table, schema, name, tmp_path):
    table = make_table()
    path = tmp_path / f"test_{name.lower()}.parquet"
    write_parquet(table, path, schema=schema)
    loaded = read_parquet(path, schema=schema)
    # Schema must match exactly
    assert loaded.schema.equals(schema), (
        f"{name}: loaded schema {loaded.schema} != expected {schema}"
    )
    # Row count preserved
    assert loaded.num_rows == table.num_rows


def test_write_creates_parent_dirs(tmp_path):
    table = _make_trade_table()
    nested_path = tmp_path / "a" / "b" / "c" / "trades.parquet"
    write_parquet(table, nested_path, schema=TRADE_SCHEMA)
    assert nested_path.exists()


def test_type_mismatch_raises_on_write(tmp_path):
    """Writing a string column where float64 is expected must fail."""
    bad_table = pa.table(
        {
            "timestamp_us": np.array([1], dtype=np.int64),
            "price": pa.array(["not_a_number"]),  # string where float64 expected
            "size": np.array([1.0], dtype=np.float64),
            "side": ["buy"],
        }
    )
    path = tmp_path / "bad.parquet"
    with pytest.raises(Exception):
        write_parquet(bad_table, path, schema=TRADE_SCHEMA)


def test_column_projection(tmp_path):
    table = _make_trade_table()
    path = tmp_path / "trades.parquet"
    write_parquet(table, path)
    loaded = read_parquet(path, columns=["timestamp_us", "price"])
    assert loaded.column_names == ["timestamp_us", "price"]


def test_trade_values_preserved(tmp_path):
    table = _make_trade_table(n=5)
    path = tmp_path / "trades.parquet"
    write_parquet(table, path, schema=TRADE_SCHEMA)
    loaded = read_parquet(path, schema=TRADE_SCHEMA)
    assert loaded.column("price").to_pylist() == pytest.approx(
        table.column("price").to_pylist(), rel=1e-9
    )


# --- list_monthly_partitions ---

def test_list_monthly_partitions_empty(tmp_path):
    result = list_monthly_partitions(tmp_path)
    assert result == []


def test_list_monthly_partitions_nonexistent():
    result = list_monthly_partitions(Path("/nonexistent/path"))
    assert result == []


def test_list_monthly_partitions_finds_dirs(tmp_path):
    for name in ["2024-01", "2024-02", "2024-03", "not-a-month", "2024"]:
        (tmp_path / name).mkdir()
    result = list_monthly_partitions(tmp_path)
    names = [p.name for p in result]
    assert names == ["2024-01", "2024-02", "2024-03"]


def test_list_monthly_partitions_year_filter(tmp_path):
    for name in ["2022-06", "2023-01", "2024-03"]:
        (tmp_path / name).mkdir()
    result = list_monthly_partitions(tmp_path, year_range=(2023, 2023))
    assert [p.name for p in result] == ["2023-01"]


def test_list_parquet_files(tmp_path):
    for name in ["a.parquet", "b.parquet", "c.csv"]:
        (tmp_path / name).touch()
    result = list_parquet_files(tmp_path)
    names = [p.name for p in result]
    assert names == ["a.parquet", "b.parquet"]
