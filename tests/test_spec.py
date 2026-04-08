"""Tests for IndexSpec validation and derived fields."""

import pytest

from rvol.index.spec import IndexSpec, SPEC_7D, SPEC_14D, SPEC_30D


def test_default_spec_valid():
    spec = IndexSpec()
    assert spec.window_days == 30
    assert spec.sigma_cap == 4.0
    assert spec.min_obs_fraction == 0.9


def test_expected_obs_per_window():
    spec = IndexSpec(window_days=30, interval_minutes=1)
    assert spec.expected_obs_per_window == 30 * 24 * 60  # 43200


def test_min_obs_count():
    spec = IndexSpec(window_days=30, min_obs_fraction=0.9)
    assert spec.min_obs_count == int(43200 * 0.9)  # 38880


def test_window_us():
    spec = IndexSpec(window_days=1)
    assert spec.window_us == 24 * 60 * 60 * 1_000_000


def test_prebuilt_specs():
    assert SPEC_7D.window_days == 7
    assert SPEC_14D.window_days == 14
    assert SPEC_30D.window_days == 30


def test_invalid_window_days_too_small():
    with pytest.raises(ValueError, match="window_days"):
        IndexSpec(window_days=0)


def test_invalid_window_days_too_large():
    with pytest.raises(ValueError, match="window_days"):
        IndexSpec(window_days=400)


def test_invalid_sigma_cap():
    with pytest.raises(ValueError, match="sigma_cap"):
        IndexSpec(sigma_cap=0.0)


def test_invalid_min_obs_fraction_low():
    with pytest.raises(ValueError, match="min_obs_fraction"):
        IndexSpec(min_obs_fraction=0.005)


def test_invalid_min_obs_fraction_high():
    with pytest.raises(ValueError, match="min_obs_fraction"):
        IndexSpec(min_obs_fraction=1.1)


def test_spec_is_frozen():
    spec = IndexSpec()
    with pytest.raises((AttributeError, TypeError)):
        spec.window_days = 10  # type: ignore[misc]


def test_spec_hashable():
    """Frozen dataclasses must be hashable — used as dict keys in simulations."""
    spec = IndexSpec(window_days=7)
    d = {spec: "test"}
    assert d[spec] == "test"
