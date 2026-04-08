"""Manipulation-resistance filters for return series."""

import numpy as np


def cap_return_contributions(
    returns: np.ndarray,
    sigma_cap: float = 4.0,
    mad_lookback: int = 1440,
    min_lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a MAD-based sigma cap to return contributions.

    For each return r_t, the contribution to realized variance is r_t².
    This function caps |r_t| at sigma_cap × σ̂_t where σ̂_t is estimated
    from the prior mad_lookback observations using the median absolute
    deviation (MAD):

        σ̂_t = 1.4826 × median(|r_{t-k}|, ..., |r_{t-1}|)

    The MAD estimator is preferred over sample std because sample std is
    inflated by the same outliers we are trying to cap.

    Parameters
    ----------
    returns:
        1-D array of log returns (may contain NaN for gaps).
    sigma_cap:
        Cap threshold in units of σ̂. Default 4.0.
    mad_lookback:
        Number of prior observations used to estimate σ̂. Default 1440 (1 day).
    min_lookback:
        Minimum number of non-NaN prior obs required before applying cap.
        If fewer are available, the cap is applied using whatever is available.

    Returns
    -------
    (capped_returns, is_capped_mask)
        capped_returns: same shape as returns, NaNs preserved.
        is_capped_mask: bool array, True where |r_t| was reduced.
    """
    n = len(returns)
    capped = returns.copy().astype(float)
    is_capped = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(returns[i]):
            continue

        # Lookback window: [max(0, i-mad_lookback), i)
        start = max(0, i - mad_lookback)
        lookback = returns[start:i]
        valid_lookback = lookback[~np.isnan(lookback)]

        if len(valid_lookback) < min_lookback:
            # Not enough history — skip cap for this observation
            continue

        sigma_hat = 1.4826 * np.median(np.abs(valid_lookback))

        if sigma_hat == 0.0:
            # Flat price period — no cap needed
            continue

        threshold = sigma_cap * sigma_hat
        abs_r = abs(returns[i])
        if abs_r > threshold:
            capped[i] = np.sign(returns[i]) * threshold
            is_capped[i] = True

    return capped, is_capped


def minimum_obs_mask(
    timestamps_us: np.ndarray,
    window_us: int,
    min_fraction: float = 0.90,
    expected_per_minute: int = 1,
    interval_minutes: int = 1,
) -> np.ndarray:
    """Compute is_valid mask based on minimum observation count.

    For each index i, checks whether the rolling backward window
    [timestamps_us[i] - window_us, timestamps_us[i]] contains at least
    min_fraction × expected_obs non-NaN observations.

    Parameters
    ----------
    timestamps_us:
        Array of int64 microsecond timestamps (sorted ascending).
    window_us:
        Window size in microseconds.
    min_fraction:
        Minimum fraction of expected observations required.
    expected_per_minute:
        Expected observations per minute in the window (usually 1).
    interval_minutes:
        Sampling interval in minutes. Used to compute expected_per_window.

    Returns
    -------
    bool array of same length as timestamps_us.
    True where the window has sufficient observations.
    """
    n = len(timestamps_us)
    window_days = window_us / (24 * 60 * 60 * 1_000_000)
    expected_obs = int(window_days * 24 * 60 * expected_per_minute / interval_minutes)
    min_obs = int(expected_obs * min_fraction)

    is_valid = np.zeros(n, dtype=bool)

    # Two-pointer / binary search approach for efficiency
    left = 0
    for right in range(n):
        window_start = timestamps_us[right] - window_us
        # Advance left pointer so timestamps_us[left] >= window_start
        while left <= right and timestamps_us[left] < window_start:
            left += 1
        obs_in_window = right - left + 1
        is_valid[right] = obs_in_window >= min_obs

    return is_valid
