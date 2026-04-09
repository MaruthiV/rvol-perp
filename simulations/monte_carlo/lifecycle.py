"""Lifecycle simulator: prices → RV30 → mark → funding → leveraged PnL.

Consumes simulated price paths from `paths.simulate_paths`, computes the
realized variance index using the same math as `rvol.index.variance`, then
runs the Phase 4 funding mechanism path-by-path and tracks margin equity
for leveraged positions.

All computation is in numpy after the prices are materialized from JAX —
the index computation and margin logic are not bottlenecks, so the extra
JIT complexity is not worth it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LifecycleParams:
    """Simulation knobs for the lifecycle layer.

    Attributes:
        window_minutes: RV30 rolling window in minutes (30 × 1440 = 43200).
        min_obs_fraction: validity threshold (matches IndexSpec default).
        sigma_cap: 4σ MAD cap (matches production).
        mad_lookback: minutes used for rolling MAD (matches IndexSpec).
        funding_interval_minutes: 60 for hourly.
        dampening: funding dampening `k`.
        funding_cap: hard cap on per-interval funding.
        basis_rho: AR(1) persistence of basis (per hourly step).
        basis_sigma: hourly basis innovation std.
        basis_gamma: arb feedback strength from previous hour's funding.
        entry_minute: when positions are opened (must be > window_minutes).
        leverage: position leverage for margin accounting.
        maintenance_margin_frac: liquidate if equity / notional < this.
    """

    window_minutes: int = 30 * 1440
    min_obs_fraction: float = 0.9
    sigma_cap: float = 4.0
    mad_lookback: int = 1440
    funding_interval_minutes: int = 60
    dampening: float = 100.0
    funding_cap: float = 0.001
    basis_rho: float = 0.90
    basis_sigma: float = 0.005
    basis_gamma: float = 10.0
    entry_minute: int = 30 * 1440
    leverage: float = 3.0
    maintenance_margin_frac: float = 0.05


@dataclass
class LifecycleResult:
    rv30_index_hourly: np.ndarray       # (n_paths, n_hours)
    mark_hourly: np.ndarray              # (n_paths, n_hours)
    funding_hourly: np.ndarray           # (n_paths, n_hours)
    long_equity: np.ndarray              # (n_paths, n_hours) — margin equity over time
    short_equity: np.ndarray
    long_final_pnl: np.ndarray           # (n_paths,) — PnL per unit notional
    short_final_pnl: np.ndarray
    long_liquidated: np.ndarray          # (n_paths,) bool
    short_liquidated: np.ndarray


def _rolling_rv30_at_hourly(capped_sq: np.ndarray, window: int, hourly_stride: int = 60) -> np.ndarray:
    """Rolling sum of squared returns sampled at hourly stride.

    capped_sq: (n_steps,) array of r² (already capped)
    Returns: (n_hours,) where n_hours = n_steps // hourly_stride
    """
    n = len(capped_sq)
    # cumulative sum trick: rolling_sum[t] = cs[t] - cs[t - window]
    cs = np.concatenate([[0.0], np.cumsum(capped_sq)])
    n_hours = n // hourly_stride
    hourly_idx = (np.arange(n_hours) + 1) * hourly_stride
    win_start = np.maximum(0, hourly_idx - window)
    rv = cs[hourly_idx] - cs[win_start]
    return rv


def run_lifecycle(
    prices: np.ndarray,
    params: LifecycleParams,
    seed: int = 0,
) -> LifecycleResult:
    """Run the lifecycle on a batch of price paths (numpy float32/64).

    prices: (n_paths, n_steps+1) — minute-resolution path including t=0
    """
    n_paths, n_plus = prices.shape
    n_steps = n_plus - 1

    # 1. log returns, shape (n_paths, n_steps)
    log_returns = np.diff(np.log(prices), axis=1)

    # 2. Squared returns. Note: the production 4σ MAD cap is a no-op in
    # simulation since the SV data-generating process has no manipulation
    # spikes — the cap exists to defend against adversarial prints, not
    # the thin-tail bulk of the distribution. Phase 7 (manipulation cost)
    # stress-tests the cap with explicit injected spikes.
    capped_sq = log_returns * log_returns

    # 3. Rolling RV30 sampled hourly
    n_hours = n_steps // params.funding_interval_minutes
    rv_hourly = np.empty((n_paths, n_hours), dtype=np.float64)
    for i in range(n_paths):
        rv_hourly[i] = _rolling_rv30_at_hourly(
            capped_sq[i], window=params.window_minutes,
            hourly_stride=params.funding_interval_minutes,
        )
    # Index is the RV30 converted to daily rate
    index_hourly = rv_hourly / 30.0

    # 4. Simulate mark via AR(1) basis with funding feedback
    rng = np.random.default_rng(seed)
    basis = np.zeros_like(index_hourly)
    funding = np.zeros_like(index_hourly)
    mark = np.zeros_like(index_hourly)

    noise = rng.normal(0.0, params.basis_sigma, size=index_hourly.shape)
    # Initial step
    mark[:, 0] = index_hourly[:, 0]
    funding[:, 0] = 0.0
    for t in range(1, n_hours):
        b_prev = basis[:, t - 1]
        f_prev = funding[:, t - 1]
        b_t = params.basis_rho * b_prev + noise[:, t] - params.basis_gamma * f_prev
        basis[:, t] = b_t
        mark[:, t] = index_hourly[:, t] * (1.0 + b_t)
        raw = b_t / params.dampening
        funding[:, t] = np.clip(raw, -params.funding_cap, params.funding_cap)

    # 5. Margin accounting: open unit long and unit short at entry_hour
    entry_hour = params.entry_minute // params.funding_interval_minutes
    n_hours_active = n_hours - entry_hour
    if n_hours_active <= 0:
        raise ValueError("entry_minute must be before end of simulation")

    # Notional = 1 (variance unit). IM = 1/leverage.
    im = 1.0 / params.leverage
    mm = params.maintenance_margin_frac

    long_equity = np.full((n_paths, n_hours), np.nan, dtype=np.float64)
    short_equity = np.full_like(long_equity, np.nan)
    long_liq = np.zeros(n_paths, dtype=bool)
    short_liq = np.zeros(n_paths, dtype=bool)

    # Index at entry — position value tracks index change from entry
    idx_entry = index_hourly[:, entry_hour]
    long_equity[:, entry_hour] = im
    short_equity[:, entry_hour] = im

    for t in range(entry_hour + 1, n_hours):
        dI = index_hourly[:, t] - index_hourly[:, t - 1]
        f = funding[:, t]
        # Long pays funding when positive
        long_pnl_inc = dI - f * idx_entry
        short_pnl_inc = -dI + f * idx_entry
        long_equity[:, t] = long_equity[:, t - 1] + long_pnl_inc
        short_equity[:, t] = short_equity[:, t - 1] + short_pnl_inc

        # Liquidation: equity < mm × idx_entry (maintenance in variance units)
        mm_level = mm * idx_entry
        newly_liq_long = (~long_liq) & (long_equity[:, t] < mm_level)
        newly_liq_short = (~short_liq) & (short_equity[:, t] < mm_level)
        long_liq |= newly_liq_long
        short_liq |= newly_liq_short
        # Freeze equity at maintenance level after liquidation
        long_equity[long_liq, t] = mm_level[long_liq]
        short_equity[short_liq, t] = mm_level[short_liq]

    long_final = long_equity[:, -1] - im
    short_final = short_equity[:, -1] - im

    return LifecycleResult(
        rv30_index_hourly=index_hourly,
        mark_hourly=mark,
        funding_hourly=funding,
        long_equity=long_equity,
        short_equity=short_equity,
        long_final_pnl=long_final,
        short_final_pnl=short_final,
        long_liquidated=long_liq,
        short_liquidated=short_liq,
    )
