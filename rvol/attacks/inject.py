"""Adversarial return injection for the manipulation cost analysis.

Consumes a clean 1-minute return series, injects an attacker's prints at a
chosen offset, and reports the resulting RV30 with and without the 4σ MAD
cap active. The cost model is a linear Kyle-style price impact calibrated
to Binance BTC-USDT perp depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rvol.index.filters import cap_return_contributions


@dataclass(frozen=True)
class AttackParams:
    """Attack configuration.

    Attributes:
        k_sigma: attack size in units of the local MAD-sigma. k=4 is at the
            cap threshold; k=10 is deeply into the tail.
        n_minutes: how many consecutive minutes the attack lasts. 1 = spike.
        attack_offset: index into the return series where the attack begins.
            Should be close to the end of the RV30 window to maximize
            persistence in the index.
        depth_usd_per_1pct: USD capital required to move BTC price 1%.
        round_trip_factor: multiplier for sustained attacks (entry + exit).
    """

    k_sigma: float
    n_minutes: int
    attack_offset: int
    depth_usd_per_1pct: float = 5_000_000.0
    round_trip_factor: float = 2.0


@dataclass
class AttackResult:
    rv30_baseline: float          # RV30 with no attack (capped, from real data)
    rv30_uncapped: float          # RV30 with attack, no cap applied
    rv30_capped: float            # RV30 with attack, 4σ MAD cap applied
    delta_uncapped: float         # rv30_uncapped − rv30_baseline
    delta_capped: float           # rv30_capped   − rv30_baseline
    cap_reduction: float          # 1 − |delta_capped| / |delta_uncapped|
    attack_return_magnitude: float  # per-minute log return the attacker printed
    cost_usd: float
    delta_ann_vol_pt_uncapped: float  # Δ in annualized vol points
    delta_ann_vol_pt_capped: float
    usd_per_vol_pt_capped: float  # cost / |delta_ann_vol_pt_capped|


def _local_mad_sigma(
    returns: np.ndarray, offset: int, lookback: int = 1440
) -> float:
    """MAD-based sigma at the attack location (matches IndexSpec defaults)."""
    start = max(0, offset - lookback)
    window = returns[start:offset]
    if len(window) < 2:
        return 0.0
    med = np.median(np.abs(window))
    return 1.4826 * float(med)


def _rv_window(returns: np.ndarray, end: int, window: int = 30 * 1440) -> float:
    """Sum of squared returns over the window ending at `end` (exclusive)."""
    start = max(0, end - window)
    slice_ = returns[start:end]
    return float(np.sum(slice_ ** 2))


def _apply_cap_local(
    returns: np.ndarray, start: int, end: int, sigma_cap: float = 4.0
) -> np.ndarray:
    """Return a copy of returns[start:end] with the 4σ MAD cap applied.

    Only the local slice needs capping because the caller computes RV over
    that same window. The MAD lookback requires ≤1440 obs of history before
    `start`, which we pass through cap_return_contributions by including
    them in the input and then dropping them from the output.
    """
    lookback = 1440
    ctx_start = max(0, start - lookback)
    ctx = returns[ctx_start:end]
    capped_ctx, _ = cap_return_contributions(ctx, sigma_cap=sigma_cap)
    full = returns.copy()
    full[ctx_start:end] = capped_ctx
    return full


def _compute_attack(
    returns: np.ndarray,
    params: AttackParams,
    attack_returns: np.ndarray,
) -> AttackResult:
    """Core attack harness shared by spike and sustained injectors."""
    end_idx = params.attack_offset + params.n_minutes
    if end_idx > len(returns):
        raise ValueError("attack window extends past end of return series")

    sigma_mad = _local_mad_sigma(returns, params.attack_offset)
    if sigma_mad <= 0:
        raise ValueError("degenerate MAD sigma at attack offset")

    # Two baselines: one with cap active (what the index reports), one without
    # (pure arithmetic). Deltas must compare like-for-like — capped attack vs
    # capped baseline, uncapped attack vs uncapped baseline — otherwise the
    # delta picks up the background capping of benign returns rather than the
    # marginal attack impact.
    rv_window = 30 * 1440
    rv_start = max(0, end_idx - rv_window)
    baseline_capped_arr = _apply_cap_local(returns, rv_start, end_idx)
    rv_baseline_capped = _rv_window(baseline_capped_arr, end_idx)
    rv_baseline_uncapped = _rv_window(returns, end_idx)

    # Attack path
    attacked = returns.copy()
    attacked[params.attack_offset : end_idx] = attack_returns

    rv_uncapped = _rv_window(attacked, end_idx)
    attacked_capped = _apply_cap_local(attacked, rv_start, end_idx)
    rv_capped = _rv_window(attacked_capped, end_idx)

    delta_uncapped = rv_uncapped - rv_baseline_uncapped
    delta_capped = rv_capped - rv_baseline_capped
    rv_baseline = rv_baseline_capped  # report the live-index value as baseline
    reduction = (
        1.0 - abs(delta_capped) / abs(delta_uncapped)
        if abs(delta_uncapped) > 1e-18
        else 0.0
    )

    # Convert RV30 sum to annualized vol: ann_vol = sqrt(rv * 365 / 30)
    def rv_to_ann_vol_pct(rv: float) -> float:
        return float(np.sqrt(max(rv, 0.0) * 365.0 / 30.0) * 100.0)

    vol_base = rv_to_ann_vol_pct(rv_baseline)
    vol_uncap = rv_to_ann_vol_pct(rv_uncapped)
    vol_cap = rv_to_ann_vol_pct(rv_capped)
    d_vol_uncap = vol_uncap - vol_base
    d_vol_cap = vol_cap - vol_base

    cost = estimate_cost_usd(params, attack_returns)
    usd_per_vol_pt = cost / abs(d_vol_cap) if abs(d_vol_cap) > 1e-6 else float("inf")

    return AttackResult(
        rv30_baseline=rv_baseline,
        rv30_uncapped=rv_uncapped,
        rv30_capped=rv_capped,
        delta_uncapped=delta_uncapped,
        delta_capped=delta_capped,
        cap_reduction=reduction,
        attack_return_magnitude=float(np.abs(attack_returns).mean()),
        cost_usd=cost,
        delta_ann_vol_pt_uncapped=d_vol_uncap,
        delta_ann_vol_pt_capped=d_vol_cap,
        usd_per_vol_pt_capped=usd_per_vol_pt,
    )


def inject_single_spike(
    returns: np.ndarray, params: AttackParams
) -> AttackResult:
    """Attack A: replace one minute's return with `k × local_sigma_MAD`."""
    if params.n_minutes != 1:
        raise ValueError("inject_single_spike requires n_minutes=1")
    sigma = _local_mad_sigma(returns, params.attack_offset)
    attack_return = params.k_sigma * sigma
    arr = np.array([attack_return], dtype=np.float64)
    return _compute_attack(returns, params, arr)


def inject_sustained(
    returns: np.ndarray, params: AttackParams
) -> AttackResult:
    """Attack B: sustain `k × sigma_MAD` for `n_minutes` consecutive minutes.

    Each minute's printed return has magnitude `k × sigma`. Sign alternates
    so the attacker isn't cumulatively trending the price far from fair
    (otherwise a 1000-minute attack would move the price to absurd levels).
    """
    if params.n_minutes < 1:
        raise ValueError("n_minutes must be >= 1")
    sigma = _local_mad_sigma(returns, params.attack_offset)
    # Alternating sign → attacker oscillates price ±k·σ around fair
    signs = np.where(np.arange(params.n_minutes) % 2 == 0, 1.0, -1.0)
    arr = signs * params.k_sigma * sigma
    return _compute_attack(returns, params, arr)


def estimate_cost_usd(params: AttackParams, attack_returns: np.ndarray) -> float:
    """Linear Kyle cost model.

    Each per-minute print of |return|=r (as a fraction, so 0.01 = 1%) costs
    `r × 100 × depth_usd_per_1pct` to execute one-way. Sustained attacks
    pay `round_trip_factor` per minute (entry + exit of that tick's position).
    """
    abs_r_pct = np.abs(attack_returns) * 100.0
    per_minute_cost = abs_r_pct * params.depth_usd_per_1pct / 100.0 * params.round_trip_factor
    # For a single spike, no round-trip factor needed (it's a one-shot print)
    if params.n_minutes == 1:
        per_minute_cost = abs_r_pct * params.depth_usd_per_1pct / 100.0
    return float(np.sum(per_minute_cost))
