# Appendix A. Formal index specification and invariants

This appendix gives the complete, self-contained specification of the realized-variance index used as the settlement reference of the contract. It restates the main-text definitions in a form suitable for a reference implementer, and it enumerates the invariants that any conforming implementation (Python, Solidity, Rust) must satisfy. Every parameter listed here is frozen at deployment; changing any of them constitutes a new contract.

## A.1 Parameters

| Symbol | Value | Units | Description |
|---|---|---|---|
| $\Delta t$ | 60 | seconds | Sampling interval (1-minute bars) |
| $W$ | 43,200 | minutes | Rolling RV window (30 calendar days) |
| $L_{\text{MAD}}$ | 1,440 | minutes | MAD lookback (1 calendar day) |
| $\sigma_{\text{cap}}$ | 4 | — | MAD cap multiplier |
| $\kappa$ | 1.4826 | — | Normal-consistency constant for MAD |
| $G_{\max}$ | 65 | seconds | Maximum gap before a bar is marked invalid |
| $\nu_{\min}$ | 38,880 | bars | Minimum valid observations (0.9 × $W$) |
| $A$ | 525,600 | minutes/year | Annualization factor |

## A.2 Inputs

The implementation requires a single input stream: a sequence of `(timestamp, last_trade_price)` pairs from the host venue's BTC-USD perpetual market, sampled at 1-minute boundaries. Timestamps are integer microseconds UTC. Prices are positive reals represented in 18-decimal fixed-point.

## A.3 Definitions

**Log return.** For bars $t$ and $t-1$ with prices $P_t, P_{t-1}$:
$$ r_t = \ln(P_t / P_{t-1}). $$

**Gap flag.** Let $\tau_t$ be the wall-clock timestamp of bar $t$. Then
$$ g_t = \mathbb{1}\!\left[\tau_t - \tau_{t-1} > G_{\max}\right]. $$
If $g_t = 1$, the bar is marked NaN; $r_t$ is undefined and is excluded from every downstream sum.

**Rolling MAD estimator.** Over the trailing $L_{\text{MAD}}$ non-NaN bars:
$$ \hat\sigma_t = \kappa \cdot \operatorname{median}\bigl(\,|r_{t-1}|,\, |r_{t-2}|,\, \ldots,\, |r_{t-L_{\text{MAD}}}|\,\bigr). $$
$\hat\sigma_t$ is computed **from past data only**; the current bar $r_t$ is not included, eliminating any feedback loop between the cap and the current observation.

**Capped return.**
$$ r_t^{\text{capped}} = \operatorname{sign}(r_t) \cdot \min\!\bigl(|r_t|,\ \sigma_{\text{cap}} \cdot \hat\sigma_t\bigr). $$

**Cap-active indicator.** $a_t = \mathbb{1}\!\left[|r_t| > \sigma_{\text{cap}} \cdot \hat\sigma_t\right]$.

**Rolling realized variance (per-minute units).**
$$ \mathrm{RV}_T = \sum_{t = T - W + 1}^{T} \bigl(r_t^{\text{capped}}\bigr)^2 \cdot \mathbb{1}[g_t = 0]. $$

**Valid-observation count.**
$$ \nu_T = \sum_{t = T - W + 1}^{T} \mathbb{1}[g_t = 0]. $$

**Validity flag.**
$$ V_T = \mathbb{1}[\nu_T \ge \nu_{\min}]. $$

**Annualized volatility (display form).**
$$ \sigma^{\text{ann}}_T = \sqrt{\,\mathrm{RV}_T \cdot (A / W)\,} \cdot 100 \quad \text{(percent).} $$
Note: all contract math — funding, margin, payoff — uses $\mathrm{RV}_T$ directly in per-minute units, not $\sigma^{\text{ann}}_T$. The annualized form is a human-readable display only.

## A.4 State machine

A conforming implementation maintains the following state:

- A ring buffer $B^{\text{RV}}$ of size $W$ holding $(r_t^{\text{capped}})^2$ for non-NaN bars, zero for NaN bars.
- A ring buffer $B^{\text{MAD}}$ of size $L_{\text{MAD}}$ holding $|r_t|$ for non-NaN bars, zero for NaN bars.
- An auxiliary sorted structure over $B^{\text{MAD}}$ supporting O(log $L_{\text{MAD}}$) insertion, deletion, and median query.
- A scalar $S_T = \sum B^{\text{RV}}$, maintained incrementally.
- A scalar $\nu_T$, maintained incrementally.
- A scalar $\tau_{T}$, the timestamp of the most recent update.
- A scalar $P_T$, the price of the most recent non-NaN bar.
- A ring write pointer $h$.

Per-bar update (pseudocode):

```
on_new_bar(tau, P):
    gap = (tau - tau_prev) > G_max
    if gap:
        r = NaN
    else:
        r = ln(P / P_prev)

    # MAD update (must precede cap application)
    evict_mad = B_MAD[h mod L_MAD]
    sorted_mad.remove(evict_mad)
    insert_mad = (|r| if not gap else 0)
    sorted_mad.insert(insert_mad)
    B_MAD[h mod L_MAD] = insert_mad
    sigma_hat = kappa * sorted_mad.median()

    # Cap and square
    if gap:
        new_sq = 0
        delta_nu = -1 if evicted_was_valid else 0
    else:
        r_cap = sign(r) * min(|r|, sigma_cap * sigma_hat)
        new_sq = r_cap * r_cap
        delta_nu = (+1 if evicted_was_nan else 0) +
                   (-1 if evicted_was_valid and this_is_nan else 0)
        # simplified: maintain nu = count of non-NaN in B_RV

    # RV buffer update
    evict_sq = B_RV[h mod W]
    S = S - evict_sq + new_sq
    B_RV[h mod W] = new_sq

    # Scalars
    nu = nu + delta_nu
    V = (nu >= nu_min)
    tau_prev = tau
    if not gap: P_prev = P
    h = h + 1
```

The update is O(log $L_{\text{MAD}}$) amortized per bar, dominated by the sorted-structure operations; the RV buffer update is O(1). The total state is $W + L_{\text{MAD}} + O(1) \approx$ 44,640 scalar slots.

## A.5 Invariants

Any conforming implementation must satisfy the following six invariants at every state transition. Violating any of them is a spec bug and blocks launch.

**I1 (non-negativity).** $\mathrm{RV}_T \ge 0$ for all $T$. Sum of squares; follows trivially if the squaring is not buggy.

**I2 (cap monotonicity).** $\mathrm{RV}_T^{\text{capped}} \le \mathrm{RV}_T^{\text{uncapped}}$ pointwise. Capping can only reduce the contribution of any bar; it can never increase the index.

**I3 (gap exclusion).** If $g_t = 1$, then bar $t$ contributes zero to $\mathrm{RV}_T$ for all $T \ge t$ until bar $t$ is evicted from the window. A gap bar is not a zero-return bar — it is an undefined bar, and any downstream consumer must treat it as missing, not as calm.

**I4 (O(1) incremental consistency).** At any $T$, the scalar $S_T$ maintained incrementally must equal the explicit sum $\sum_{t = T - W + 1}^T B^{\text{RV}}[t \bmod W]$. Deviation indicates drift in the incremental update; it is checked by a background audit that recomputes the explicit sum periodically and compares.

**I5 (MAD backward-looking).** $\hat\sigma_t$ depends only on bars $t-1, t-2, \ldots, t-L_{\text{MAD}}$. In particular $\hat\sigma_t$ does not depend on $r_t$, which eliminates the feedback loop that defeats sample-standard-deviation-based caps.

**I6 (validity monotonicity).** $V_T$ transitions from false to true only when $\nu_T$ first reaches $\nu_{\min}$, and from true to false only when $\nu_T$ falls below $\nu_{\min}$. There is no hysteresis, no smoothing, and no discretionary override.

## A.6 Determinism requirement

A conforming implementation must produce byte-identical $S_T$, $\nu_T$, and $V_T$ at every $T$ given the same input price history, across languages and across hardware. This is enforced by (a) standardizing on 18-decimal fixed-point arithmetic throughout, (b) using a single reference implementation of the natural log routine (a lookup-table-assisted Taylor expansion with 1-ulp guarantee in the fixed-point format), and (c) specifying the MAD ordering as "lexicographic by (value, insertion-timestamp) for tie-breaking," so that equal absolute returns are broken deterministically. Any implementation that diverges from the reference at any minute in the launch-gate testnet parallel-run (§11.5, Phase L0) is blocked from promotion.

## A.7 What this spec does not cover

This appendix specifies the index. It does not specify: the mark-price determination (left to the host venue's matching engine), the funding mechanism (§7), the margin system (§9), the liquidation logic (§9.4), the insurance fund (§9.5), or the governance parameters (§13.4). Those are separate specs that take $\mathrm{RV}_T$ as their single input. The separation is deliberate: any component above can be redesigned without touching the index spec, and the index spec is the only component whose correctness is load-bearing for every other claim in the paper.
