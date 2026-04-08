"""Typed Parquet I/O for the rvol data pipeline.

All schemas are explicit PyArrow schemas enforced at write time.
All timestamps are int64 microseconds UTC — no timezone-aware types.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Canonical schemas
# ---------------------------------------------------------------------------

TRADE_SCHEMA = pa.schema(
    [
        pa.field("timestamp_us", pa.int64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.float64(), nullable=False),
        pa.field("side", pa.string(), nullable=True),
    ]
)

OHLCV_SCHEMA = pa.schema(
    [
        pa.field("timestamp_us", pa.int64(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=True),
        pa.field("quote_volume", pa.float64(), nullable=True),
        pa.field("n_trades", pa.int32(), nullable=True),
    ]
)

FUNDING_SCHEMA = pa.schema(
    [
        pa.field("timestamp_us", pa.int64(), nullable=False),
        pa.field("funding_rate", pa.float64(), nullable=False),
    ]
)

DVOL_SCHEMA = pa.schema(
    [
        pa.field("timestamp_us", pa.int64(), nullable=False),
        pa.field("dvol_annualized", pa.float64(), nullable=False),
    ]
)

RETURN_SCHEMA = pa.schema(
    [
        pa.field("timestamp_us", pa.int64(), nullable=False),
        pa.field("log_return", pa.float64(), nullable=True),
        pa.field("is_capped", pa.bool_(), nullable=False),
        pa.field("is_gap", pa.bool_(), nullable=False),
    ]
)

INDEX_SCHEMA = pa.schema(
    [
        pa.field("timestamp_us", pa.int64(), nullable=False),
        pa.field("rv7", pa.float64(), nullable=True),
        pa.field("rv14", pa.float64(), nullable=True),
        pa.field("rv30", pa.float64(), nullable=True),
        pa.field("n_obs", pa.int32(), nullable=False),
        pa.field("n_capped", pa.int32(), nullable=False),
        pa.field("is_valid", pa.bool_(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------

_COMPRESSION = "snappy"


def write_parquet(
    table: pa.Table,
    path: Path,
    schema: pa.Schema | None = None,
) -> None:
    """Write a PyArrow Table to Parquet with optional schema enforcement.

    If schema is provided, the table is cast to that schema before writing.
    This catches type mismatches at write time rather than silently producing
    incorrect data (e.g., float64 silently downcast to float32).

    Parameters
    ----------
    table:
        PyArrow Table to write.
    path:
        Destination file path. Parent directory is created if needed.
    schema:
        If provided, cast table to this schema before writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if schema is not None:
        table = table.cast(schema)

    pq.write_table(table, path, compression=_COMPRESSION)


def read_parquet(
    path: Path,
    schema: pa.Schema | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Read a Parquet file, optionally validating the schema.

    Parameters
    ----------
    path:
        Source file path.
    schema:
        If provided, the table is cast to this schema after reading.
        Raises if columns don't match.
    columns:
        If provided, only read these columns (projection pushdown).

    Returns
    -------
    pa.Table
    """
    table = pq.read_table(path, columns=columns)
    if schema is not None:
        table = table.cast(schema)
    return table


def list_monthly_partitions(base_dir: Path, year_range: tuple[int, int] | None = None) -> list[Path]:
    """List all YYYY-MM partition directories under base_dir.

    Parameters
    ----------
    base_dir:
        Parent directory containing YYYY-MM subdirectories.
    year_range:
        Optional (start_year, end_year) inclusive filter.

    Returns
    -------
    Sorted list of Path objects for each monthly partition directory.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    partitions = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and len(p.name) == 7 and p.name[4] == "-"
    )

    if year_range is not None:
        start_y, end_y = year_range
        partitions = [
            p for p in partitions
            if start_y <= int(p.name[:4]) <= end_y
        ]

    return partitions


def list_parquet_files(partition_dir: Path) -> list[Path]:
    """Return sorted list of .parquet files in a partition directory."""
    return sorted(partition_dir.glob("*.parquet"))
