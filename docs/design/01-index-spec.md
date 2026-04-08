# Index Specification: BTC Realized Variance Index

**Version**: 0.1.0
**Status**: DRAFT — must be frozen before data pipeline work begins
**Becomes**: Appendix A of the arXiv paper

---

## 1. Overview

The BTC Realized Variance Index (BRVX) measures the realized variance of BTC/USD log returns over a rolling 30-calendar-day window, computed from 1-minute last-trade prices. The index is designed to be:

1. **Deterministic** — two independent implementations produce identical values
2. **Manipulation-resistant** — return contributions are capped using a robust estimator
3. **Oracle-free** — computable entirely from on-chain price history
4. **Continuous** — the index updates every minute with each new price observation

---

## 2. Formal Definition

### 2.1 Price and Return

Let `P_t` denote the last-trade price of BTC/USD at minute `t` (UTC).

The 1-minute log return at time `t` is:

```
r_t = ln(P_t / P_{t-1})
```

**Gap handling**: If the time elapsed between consecutive price observations exceeds 65 seconds, return `r_t` is set to `NaN` and excluded from all subsequent computations. This handles exchange outages, network interruptions, and missing data without introducing stale-price artifacts.

### 2.2 Return Contribution Cap (Manipulation Resistance)

Raw returns are transformed before contributing to the variance sum:

```
r_t_capped = sign(r_t) × min(|r_t|, σ_cap × σ̂_t)
```

where:

```
σ̂_t = 1.4826 × median(|r_{t-1440}|, |r_{t-1439}|, ..., |r_{t-1}|)
```

is the median absolute deviation (MAD) estimate of the return standard deviation, using the prior 1440 observations (1 calendar day). The factor 1.4826 makes the MAD estimate consistent with the normal distribution standard deviation.

**`σ_cap = 4.0`** (the cap multiplier).

**Why MAD, not sample std**: If sample standard deviation were used for `σ̂_t`, large outlier returns would inflate the threshold and defeat the cap. MAD is robust to outliers by construction — a single spike cannot move the median by more than `O(1/n)`.

**Minimum lookback**: The cap requires at least 1440 prior observations. During the first 1440 minutes of index history, the cap is applied using whatever prior observations are available (minimum 60).

### 2.3 Rolling Realized Variance

Let `W_T` denote the window of valid (non-NaN) return indices within 30 calendar days of time `T`:

```
W_T = { t : T - 30d ≤ t ≤ T,  r_t ≠ NaN }
```

The 30-day realized variance at time `T` is:

```
RV_T = Σ_{t ∈ W_T} (r_t_capped)²
```

This is in units of **variance per 30-day window** (dimensionless, log²).

### 2.4 Annualized Realized Variance

```
RV_T_annualized = RV_T × (365 / 30)
```

The corresponding annualized realized volatility (for display only — not used in settlement):

```
RVOL_T = sqrt(RV_T_annualized)
```

**Annualization convention**: 365 calendar days (standard in crypto, not 252 trading days).

### 2.5 Validity Flag

The index value at time `T` is marked **invalid** (`is_valid = False`) if:

```
|W_T| < 0.90 × (30 × 1440)  =  |W_T| < 38,880
```

i.e., fewer than 90% of the 43,200 expected 1-minute observations are present in the window. Invalid index values must not be used for settlement.

---

## 3. Parameters Summary

| Parameter | Value | Rationale |
|---|---|---|
| Underlying | BTC/USD last-trade price | Most liquid BTC pair |
| Return interval | 1 minute | Avoids microstructure noise of sub-minute |
| Window length | 30 calendar days | Industry convention; ~43,200 expected obs |
| Cap multiplier σ_cap | 4.0 | Clips ≈0.006% of observations under GBM |
| σ estimator | 1-day rolling MAD | Robust to regime shifts and outliers |
| Minimum obs fraction | 90% | Tolerates ~4.3h of outage per 30-day window |
| Gap threshold | 65 seconds | 1.5 × 1-minute interval |
| Annualization | × (365/30) | Crypto calendar convention |

---

## 4. Reference Implementation

The canonical Python implementation is in `rvol/index/`:

```
spec.py      → IndexSpec dataclass encoding all parameters above
returns.py   → log_returns_from_prices(), resample_trades_to_ohlcv()
filters.py   → cap_return_contributions(), minimum_obs_mask()
variance.py  → rolling_realized_variance(), point_in_time_rv()
```

Two implementations are cross-checked in `tests/test_variance.py::test_rv_converges_to_true_variance_gbm`: the custom implementation is tested against the `arch` library's realized variance estimator on the same synthetic price path.

---

## 5. Numerical Example

Suppose BTC 1-minute log returns over the past 30 days are drawn from N(0, σ²) with σ = 0.001 (= 0.1% per minute).

Expected 30-day realized variance:

```
E[RV_30] = 43,200 × σ² = 43,200 × 0.000001 = 0.0432
```

Annualized:

```
RV_annualized = 0.0432 × (365/30) = 0.526   → annualized vol = √0.526 ≈ 72.5%
```

Plausible range for BTC (20%–100% annualized vol):

```
RV_30 ∈ [0.0049, 0.123]
```

Index values outside this range during calm markets indicate a data problem.

---

## 6. Manipulation Analysis (Preview)

Full analysis in Phase 7. Preview of the key bound:

An adversary wishing to move `RV_T` by amount `δ` must inject returns `r_attack` such that `Σ r_attack² = δ`. Each injected return `r_attack` moves the spot price by `e^{r_attack} - 1 ≈ r_attack`. The cost is the market impact of that price move.

The MAD cap limits any single return contribution to `(4σ̂)²`. Therefore, to move `RV_T` by `δ`, an attacker needs at least `δ / (4σ̂)²` separate attack transactions, each of which moves the price by `4σ̂`. Under typical BTC market depth, moving price by `4σ̂ ≈ 0.4%` costs approximately $X million in market impact (computed in Phase 7 using Binance order book data).

---

## 7. Versioning and Freeze Protocol

This spec is versioned. Once frozen (marked `Status: FROZEN`), changes require:
1. New version number
2. New backtested index series
3. Updated paper appendix

**Current status: DRAFT** — may change during Phase 2 data validation.
