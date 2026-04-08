"""Rolling realized variance computation — the mathematical core of the index."""

import numpy as np
import pandas as pd

from .filters import cap_return_contributions, minimum_obs_mask
from .spec import IndexSpec


def rolling_realized_variance(
    returns: pd.Series,
    spec: IndexSpec,
) -> pd.DataFrame:
    """Compute rolling realized variance over a calendar-day window.

    The window is calendar-day-based (not fixed-count). For each timestamp T,
    the window covers all returns in [T - window_days, T].

    Parameters
    ----------
    returns:
        Series of 1-minute log returns with DatetimeIndex (UTC).
        NaN values represent gaps and are excluded from the RV sum.
    spec:
        IndexSpec instance defining all index parameters.

    Returns
    -------
    pd.DataFrame with the same index as `returns`, columns:
        rv          — realized variance (sum of capped squared returns)
        rv_annualized — rv × (annualization_days / window_days)
        n_obs       — number of non-NaN returns in the window
        n_capped    — number of returns that were capped
        is_valid    — True if n_obs >= spec.min_obs_count
    """
    if returns.empty:
        return _empty_result(returns.index)

    raw = returns.values.astype(float)
    n = len(raw)

    # Apply MAD-based cap to the entire series
    capped, is_capped_mask = cap_return_contributions(
        raw,
        sigma_cap=spec.sigma_cap,
        mad_lookback=spec.mad_lookback_minutes,
    )

    # Convert DatetimeIndex to int64 microseconds for window arithmetic
    if isinstance(returns.index, pd.DatetimeIndex):
        # numpy conversion handles tz-aware indexes and normalizes to microseconds
        timestamps_us = np.array(returns.index, dtype="datetime64[us]").astype(np.int64)
    else:
        timestamps_us = np.asarray(returns.index, dtype=np.int64)

    # Pre-compute contributions: (capped_return)^2, NaN where gap
    contributions = np.where(np.isnan(capped), np.nan, capped**2)

    # Rolling window computation using two-pointer approach
    rv = np.full(n, np.nan)
    n_obs = np.zeros(n, dtype=np.int32)
    n_capped_arr = np.zeros(n, dtype=np.int32)

    left = 0
    # Running sums for efficiency
    running_rv = 0.0
    running_n_obs = 0
    running_n_capped = 0

    for right in range(n):
        window_start_us = timestamps_us[right] - spec.window_us

        # Add right observation
        if not np.isnan(contributions[right]):
            running_rv += contributions[right]
            running_n_obs += 1
            if is_capped_mask[right]:
                running_n_capped += 1

        # Remove observations that fell out of the window from the left
        while left <= right and timestamps_us[left] <= window_start_us:
            if not np.isnan(contributions[left]):
                running_rv -= contributions[left]
                running_n_obs -= 1
                if is_capped_mask[left]:
                    running_n_capped -= 1
            left += 1

        # Guard against floating point drift below zero
        rv[right] = max(running_rv, 0.0)
        n_obs[right] = running_n_obs
        n_capped_arr[right] = running_n_capped

    annualization_factor = spec.annualization_days / spec.window_days
    rv_annualized = rv * annualization_factor

    is_valid = n_obs >= spec.min_obs_count

    return pd.DataFrame(
        {
            "rv": rv,
            "rv_annualized": rv_annualized,
            "n_obs": n_obs,
            "n_capped": n_capped_arr,
            "is_valid": is_valid,
        },
        index=returns.index,
    )


def point_in_time_rv(
    returns: np.ndarray,
    spec: IndexSpec,
) -> dict:
    """Compute realized variance for a single fixed-length window.

    Used in unit tests (where the exact window is known) and in the
    live oracle updater (where we pass exactly the last window_days
    of returns at each block).

    Parameters
    ----------
    returns:
        1-D array of log returns for the window. May contain NaN for gaps.
        The entire array is treated as the window (no rolling).
    spec:
        IndexSpec instance.

    Returns
    -------
    dict with keys: rv, rv_annualized, n_obs, n_capped, is_valid.
    """
    raw = np.asarray(returns, dtype=float)

    capped, is_capped_mask = cap_return_contributions(
        raw,
        sigma_cap=spec.sigma_cap,
        mad_lookback=spec.mad_lookback_minutes,
    )

    valid_mask = ~np.isnan(capped)
    contributions = np.where(valid_mask, capped**2, 0.0)

    rv = float(max(np.sum(contributions), 0.0))
    n_obs = int(np.sum(valid_mask))
    n_capped = int(np.sum(is_capped_mask & valid_mask))
    annualization_factor = spec.annualization_days / spec.window_days
    rv_annualized = rv * annualization_factor
    is_valid = n_obs >= spec.min_obs_count

    return {
        "rv": rv,
        "rv_annualized": rv_annualized,
        "n_obs": n_obs,
        "n_capped": n_capped,
        "is_valid": is_valid,
    }


def _empty_result(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rv": pd.Series(dtype=float),
            "rv_annualized": pd.Series(dtype=float),
            "n_obs": pd.Series(dtype=np.int32),
            "n_capped": pd.Series(dtype=np.int32),
            "is_valid": pd.Series(dtype=bool),
        },
        index=index,
    )
