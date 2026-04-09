# 4. The Index

This section defines the reference index used for settlement, funding, and margining. We call it `BRVX` (BTC Realized Variance indeX). The index is a rolling 30-calendar-day realized variance of 1-minute log returns on BTC-USD, with a robust cap on per-minute contributions and an explicit validity flag. Every design choice is made with two constraints in mind: (i) the index must be computable on-chain with O(1) state updates per observation, and (ii) the index must resist adversarial manipulation by a trader printing off-market prices on the underlying venue.

## 4.1 Price, return, and gap handling

Let `P_t` denote the last-trade price of the BTC-USD perpetual on the host venue at minute boundary `t` (UTC, aligned to whole-minute timestamps). The 1-minute log return is

$$ r_t \;=\; \ln\!\left(\frac{P_t}{P_{t-1}}\right). $$

Two operational rules accompany this definition.

**Gap rule.** If the elapsed wall-clock time between the observations used to form $P_{t-1}$ and $P_t$ exceeds **65 seconds** (1.5 × the 60-second interval), we set `r_t = NaN` and exclude the minute from all subsequent computations. This handles exchange outages, sequencer hiccups, and sparse trading intervals without introducing stale-price artifacts. The 65-second threshold is chosen as the smallest value strictly greater than the return interval that still admits normal jitter in trade timestamps.

**Minute alignment.** Returns are formed from last-trade prices at the closed end of each 1-minute bucket, not from VWAP. Last-trade alignment is deterministic, cheap to verify on-chain, and resistant to weighting-scheme manipulation.

We use 1-minute sampling — not tick — for three reasons. First, microstructure noise (bid-ask bounce, discrete pricing) is materially smaller at 1-minute than at sub-minute, so the naive realized-variance estimator is close to unbiased without requiring noise-robust kernels [Barndorff-Nielsen, Hansen, Lunde, Shephard 2008]. Second, 1-minute sampling yields 43,200 observations per 30-day window — sufficient for convergence to quadratic variation [Andersen, Bollerslev, Diebold, Labys 2003] while keeping on-chain storage modest. Third, the O(1) update requirement rules out kernel estimators whose bandwidth changes the accounting each minute.

## 4.2 Return-contribution cap (manipulation resistance)

The raw return series is transformed before contributing to the variance sum:

$$ r_t^{\text{capped}} \;=\; \operatorname{sign}(r_t)\cdot \min\!\bigl(|r_t|,\; \sigma_{\text{cap}} \cdot \hat\sigma_t\bigr), $$

with $\sigma_{\text{cap}} = 4.0$ and $\hat\sigma_t$ the **median-absolute-deviation (MAD) estimator** of the per-minute return standard deviation computed over the trailing 1-day window of 1,440 observations:

$$ \hat\sigma_t \;=\; 1.4826 \,\cdot\, \operatorname{median}\!\Bigl(\bigl|r_{t-1440}\bigr|,\,\bigl|r_{t-1439}\bigr|,\,\ldots,\,\bigl|r_{t-1}\bigr|\Bigr). $$

The constant $1.4826 = 1/\Phi^{-1}(0.75)$ is the standard normal-consistency factor that makes the MAD estimator equal to $\sigma$ in expectation under Gaussian returns.

**Why MAD and not sample standard deviation.** If $\hat\sigma_t$ were a rolling sample standard deviation, a single 10σ return would inflate $\hat\sigma_t$ and raise the cap on its own contribution — the estimator would actively defeat the cap. The median, by contrast, is robust: a single outlier cannot move it by more than $O(1/n)$. Under typical BTC returns the MAD and sample-std estimates agree to within a few percent, but in precisely the regime where the cap matters — the presence of a large outlier — they diverge, and only the MAD cap is enforceable.

**Clip fraction under Gaussian returns.** Under i.i.d. $\mathcal{N}(0,\sigma^2)$ returns, $\mathbb{P}(|r|>4\sigma) \approx 6.3\times 10^{-5}$, so the cap clips roughly **0.006% of observations** in clean conditions. The cap is designed to be inactive almost always and active only on the tail events that would otherwise dominate the index.

**Minimum lookback.** The MAD estimator requires at least 1,440 prior observations to fully converge. In the first 24 hours of index operation (or after a validity outage that forces the estimator to reinitialize), we apply the cap with whatever prior observations are available, subject to a minimum of 60 observations; below 60, the cap is disabled and the window is flagged invalid (§4.4).

The cap is the single most important manipulation-resistance primitive in the index. Section 8 quantifies its effectiveness: attacks of size $k\cdot\hat\sigma$ for $k\ge15$ have their contribution reduced by more than 93% regardless of attack geometry, and the minimum cost to move the annualized index by 1 volatility point exceeds \$3.7M under a standard Kyle (1985) price-impact model calibrated to HL order-book depth.

## 4.3 Rolling realized variance

Let $W_T$ denote the set of valid (non-NaN) minute indices in the 30 calendar days preceding time $T$:

$$ W_T \;=\; \bigl\{\, t \,:\, T - 30\text{d} \le t \le T,\quad r_t \ne \text{NaN} \,\bigr\}. $$

The 30-day realized variance index at time $T$ is

$$ \boxed{\;\mathrm{RV}_T \;=\; \sum_{t\in W_T} \bigl(r_t^{\text{capped}}\bigr)^{2}.\;} $$

This is the primitive over which the contract settles. It is in dimensionless units of squared log returns, summed over the window; following standard practice we also publish a display-only annualized vol equivalent,

$$ \operatorname{RVOL}_T \;=\; \sqrt{\mathrm{RV}_T \cdot \tfrac{365}{30}} \;\times\; 100\% , $$

using the crypto-calendar convention of 365 annualization days (not 252). This is the number traders see in the UI; settlement, funding, and margin logic all reference $\mathrm{RV}_T$ directly, never its square root.

Under Andersen–Bollerslev–Diebold–Labys, $\mathrm{RV}_T$ is a consistent estimator of the integrated variance over the 30-day window as the sampling frequency $\Delta t \to 0$ and in the absence of the cap. Our choice of $\Delta t = 1$ minute trades a small asymptotic bias (noise) against the O(1) per-update cost required for on-chain computation.

### 4.3.1 Algorithmic state (O(1) update)

The on-chain contract maintains exactly four pieces of state:

1. A ring buffer `B` of length 43,200 storing the last 43,200 capped squared returns.
2. A running sum `S` equal to $\sum_{t\in W_T} (r_t^{\text{capped}})^2$.
3. A shorter MAD-buffer `M` of length 1,440 storing the last 1,440 absolute raw returns (for the rolling median).
4. An observation counter `n` equal to $|W_T|$.

Each minute the contract:
1. Reads the new last-trade price $P_t$ from venue state.
2. Computes $r_t = \ln(P_t/P_{t-1})$ (or marks NaN under the gap rule).
3. Updates the median of `M` incrementally and derives $\hat\sigma_t$.
4. Caps $r_t$ and squares it to get $b_t = (r_t^{\text{capped}})^2$.
5. Rolls `B`: subtracts the evicted entry from `S`, overwrites it with $b_t$, adds $b_t$ to `S`.
6. Updates $n$ and the validity flag (§4.4).

All updates are $O(\log n)$ at worst (the incremental median) and $O(1)$ amortized. Total storage is $43{,}200 \times 8$ bytes $+\ 1{,}440 \times 8$ bytes $\approx$ **350 kB**, manageable as a single contract state buffer on Hyperliquid.

## 4.4 Validity flag

The index value at time `T` is marked **invalid** ($\mathrm{is\_valid}_T = \text{False}$) whenever

$$ |W_T| \;<\; 0.90 \,\cdot\, (30 \times 1440) \;=\; 38{,}880. $$

That is, fewer than 90% of the 43,200 expected 1-minute observations are present in the trailing 30-day window. The 10% budget tolerates approximately **4.3 hours of aggregate outage** per 30-day window, which comfortably exceeds any single venue-downtime event we observe in the 2020–2026 history (§5).

**Downstream behavior.** When `is_valid = False`:
- The index value is not used for settlement or mark–index funding computation.
- Funding is **paused** (set to zero) until validity is restored.
- **New positions are rejected.** Existing positions remain open and mark to the last valid index until validity resumes.
- A circuit breaker triggers after **5 consecutive missed updates**, freezing funding and emitting a governance event.

The validity flag is the single most important "halt" primitive. It ensures that an attacker who succeeds in silencing the underlying venue cannot drive the index arbitrarily via a shrunken observation set — the index simply stops updating.

## 4.5 Annualization and display conventions

The contract's economic primitive is $\mathrm{RV}_T$ in its raw (30-day, unit-less) units. Two derived quantities appear in the UI only:

$$ \mathrm{RV}_T^{\text{daily}} \;=\; \frac{\mathrm{RV}_T}{30}, \qquad \mathrm{RV}_T^{\text{ann}} \;=\; \mathrm{RV}_T \cdot \tfrac{365}{30}. $$

We use daily-variance units (i.e. $\mathrm{RV}_T/30$) as the **internal accounting unit** for margin computation, because it makes the numerical scale comparable to vol-squared at common BTC regimes: 40% annualized vol corresponds to $\mathrm{RV}^{\text{daily}} \approx 4.4\times10^{-4}$.

**Numerical example.** Suppose 1-minute returns over the past 30 days are drawn i.i.d. from $\mathcal{N}(0,\sigma^2)$ with $\sigma = 0.001$ (0.1% per minute). Then

$$ \mathbb{E}[\mathrm{RV}_{30}] \;=\; 43{,}200\,\sigma^2 \;=\; 0.0432, $$

$$ \mathrm{RV}_{30}^{\text{ann}} \;=\; 0.0432 \cdot \tfrac{365}{30} \;=\; 0.526, \qquad \mathrm{RVOL}_{30} \;\approx\; 72.5\%. $$

For plausible BTC regimes (20%–100% annualized vol), the 30-day $\mathrm{RV}$ lies in approximately $[0.0049,\ 0.123]$. Values outside this range during calm periods indicate a data pipeline error. In our 2020–2026 empirical sample (§5, §6), the observed range of $\mathrm{RVOL}_{30}$ is **18.7% to 147.7%**, with the upper bound occurring during the March 2020 COVID crash.

## 4.6 Parameter summary

| Parameter | Value | Rationale |
|---|---|---|
| Underlying | BTC-USD last-trade | Most liquid BTC pair on venue |
| Return interval | 1 minute | Avoids sub-minute microstructure noise |
| Gap threshold | 65 seconds | 1.5 × interval |
| Window length | 30 calendar days | Industry convention; 43,200 expected obs |
| Cap multiplier $\sigma_{\text{cap}}$ | 4.0 | Clips ≈ 0.006% under Gaussian |
| $\sigma$ estimator | 1-day rolling MAD | Robust to outliers, $O(1)$ updatable |
| MAD normal-consistency factor | 1.4826 | $1/\Phi^{-1}(0.75)$ |
| Minimum obs fraction | 90% | ≈4.3 h outage tolerance per 30-day window |
| Annualization factor | 365/30 | Crypto calendar convention |
| Storage footprint | ≈350 kB | $43{,}200 + 1{,}440$ buffers @ 8 bytes |

All parameters are fixed in the index specification and are not subject to governance modification after launch. Section 13 discusses the tradeoffs of making any of them governable.

## 4.7 Formal invariants

The index enforces the following invariants, each of which is tested in the reference implementation (`tests/test_variance.py`) and each of which should hold against any implementation claiming compliance with this spec:

1. **Non-negativity.** $\mathrm{RV}_T \ge 0$ for all $T$ (sum of squares).
2. **Cap monotonicity.** $\mathrm{RV}_T^{\text{capped}} \le \mathrm{RV}_T^{\text{uncapped}}$ for all $T$; capping can only reduce, never increase, the index.
3. **Flat-price identity.** If $P_t$ is constant over the window, $\mathrm{RV}_T = 0$ exactly.
4. **GBM consistency.** If $r_t \sim \mathcal{N}(0,\sigma^2)$ i.i.d., then $\mathrm{RV}_T / (|W_T|\,\sigma^2) \to 1$ in probability as $|W_T|\to\infty$ (convergence to integrated variance).
5. **Gap exclusion.** Minutes with gap-flagged returns contribute zero to $\mathrm{RV}_T$ and are excluded from $|W_T|$.
6. **Cap determinism.** For identical input price histories, two independent implementations produce bit-identical $\mathrm{RV}_T$ values.

Invariant 6 is the strongest requirement: it means that the off-chain reference Python implementation (`rvol/index/variance.py`) and the on-chain Solidity oracle must produce the same bytes on the same inputs, which is the prerequisite for any trust-minimized settlement. We enforce this via a cross-implementation test harness that replays historical return streams through both implementations and asserts equality.
