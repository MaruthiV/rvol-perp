# 9. Margin and liquidation

This section specifies the margin system: how position size is measured, how much collateral is required at entry and maintenance, how the payoff is capped, how liquidation prices are computed, and how the insurance fund is sized. The calibration is empirical — every margin rate in the tier table below is derived from the distribution of worst-case equity excursions observed over 1,124 rolling 7-day windows of the real BTC RV30 index, not from a theoretical stochastic-volatility model. Section 10 recounts the path by which this became the calibration method, and why an earlier Monte-Carlo-based calibration failed catastrophically in historical replay.

## 9.1 Position sizing: vega-notional

The on-chain settlement index $\mathrm{RV}_T$ lives in daily-variance units — a quantity near $4.4\times 10^{-4}$ at 40% annualized vol. Asking a trader to size a position in "units of daily variance" is operationally hopeless. We therefore expose position size to users in **vega-notional** $V$, defined as the USD profit-and-loss a long position earns per +1 annualized vol point of $\mathrm{RVOL}_{30}$.

Conversion between the two is a small-move linearization around a reference vol level $\sigma_{\text{ref}}$ (in percent):

$$ \frac{\mathrm{d}\mathrm{RV}^{\text{daily}}}{\mathrm{d}\sigma_{\text{ann}}} \;=\; \frac{2\sigma_{\text{ref}}}{365 \cdot 10{,}000}, $$

so that one annualized vol point of move corresponds to $\mathrm{d}v_{\text{per vol pt}} = 2 (\sigma_{\text{ref}}/100)(0.01)/365$ daily-variance units. The corresponding variance-notional (the quantity the on-chain accounting carries in variance units) is

$$ N_{\text{var}} \;=\; \frac{V}{\mathrm{d}v_{\text{per vol pt}}}, $$

with $V$ in USD per vol point. Both conversions are implemented in `rvol/margin/tiers.py::vega_notional_to_variance` / `variance_to_vega_notional`; round-trip agreement to machine precision is verified in `tests/test_margin.py::TestVegaConversion`. The vega-notional is the number quoted in the UI and the number used for the tier lookup; the variance-notional is the internal state carried by the contract.

## 9.2 Payoff cap: the single most consequential design decision

The naive variance-swap payoff of §2 is unbounded above because the index is unbounded above. Section 10 documents empirically that this is not a theoretical nitpick but the reason a naive variance perp cannot be leveraged at any reasonable rate. We follow the OTC variance-swap convention established by Demeterfi, Derman, Kamal, and Zou (1999) and **cap the per-position payoff** at $c = 2.5$ times the entry index:

$$ \mathrm{pnl}^{\text{long}}_t \;=\; \mathrm{clip}\!\left(\frac{I_t - I_{\text{entry}}}{I_{\text{entry}}},\; -1,\; c-1\right)\,\cdot\, \text{notional} \;-\; \text{funding}, $$

$$ \mathrm{pnl}^{\text{short}}_t \;=\; -\,\mathrm{clip}\!\left(\frac{I_t - I_{\text{entry}}}{I_{\text{entry}}},\; -1,\; c-1\right)\,\cdot\, \text{notional} \;+\; \text{funding}. $$

Equivalently, the position settles as if the index were $\mathrm{clip}(I_t,\, 0,\, c\cdot I_{\text{entry}})$. The cap constant $c = 2.5$ is taken directly from the TradFi convention, which has been the standard for every dealer variance swap since 1999. The cap is applied **per position, at the position's own entry strike** — not globally to the index. Two positions entered at different times therefore have different absolute cap levels. The on-chain accounting stores the entry index alongside each position as the strike reference.

**Bounded max loss per side.** Under the cap:

- **Long** maximum loss is approximately $-1\times$ notional (the index can fall only to zero).
- **Short** maximum loss is exactly $(c-1)\times$ notional $= 1.5\times$ notional.

The short side still has a larger downside than the long side by construction, because variance is bounded below at zero but can rise to $c\cdot I_{\text{entry}}$ above. This asymmetry is baked into the tier table: short positions require more collateral than long positions at equal vega-notional, which in practice we implement by setting a single initial margin rate sized to cover the more-adverse (short) side at each tier.

The cap is implemented in `rvol/margin/tiers.py::capped_pnl_frac` and verified across a suite of invariants in `tests/test_margin.py::TestCappedPayoff` including: cap level equals 2.5, long gains above entry clip at $c-1 = 1.5$, long losses clip at $-1$, short PnL is the exact negation of long PnL, the short's worst case equals $-(c-1)$, and within-cap payoffs are linear in the index.

## 9.3 Tier table

Tiers are on vega-notional $V$ and are monotone in both ceiling and initial-margin rate.

| Tier | Max $V$ (USD/vol pt) | IM | MM | Max leverage |
|---:|---:|---:|---:|---:|
| 1 |    50,000 |  67% | 5% | **1.50×** |
| 2 |   250,000 |  75% | 5% | 1.33× |
| 3 | 1,000,000 |  80% | 5% | 1.25× |
| 4 | 5,000,000 | 100% | 5% | 1.00× |
| 5 |         ∞ | 150% | 5% | 0.67× |

All tiers share a **flat 5% maintenance margin**. The initial margin rate escalates across tiers from 67% to 150%; the top tier is overcollateralized (IM > 100%), and leverage ceiling is the reciprocal of the IM rate, ranging from 1.5× at tier 1 to 0.67× at tier 5.

Two things about this table will strike any reader coming from a standard perp venue. Both are deliberate.

**The leverage ceiling is 1.5×, not 10×.** Crypto perps on spot assets routinely offer 20× or 100× leverage; even our own original Phase 8 design doc proposed 10× leverage at tier 1. The Phase 8.5 historical replay (§10) demonstrated empirically that any such number is empirically false: at 10× leverage on an uncapped variance perp, **80% of long positions and 75% of short positions get liquidated** in rolling 60-day windows over the real 2020–2026 history. The leverage ceiling was recalibrated to the number at which both sides clear a <5% liquidation gate under the historical distribution **with** the 2.5× cap in force. That number is 1.5×.

**The top tier is overcollateralized.** The tier 5 rate of 150% IM means a trader wanting to hold more than \$5M of vega-notional must post more collateral than the notional itself. This reflects the fact that at very large size, the short-side max loss of $1.5\times$ notional cannot be covered by a margin rate below 150% without exposing the insurance fund to residual loss, and the insurance fund is not intended to underwrite the largest positions on the book.

### 9.3.1 Empirical calibration

The rates in the table are chosen as the smallest initial-margin values at which the 7-day rolling liquidation rate, measured over 1,124 windows of real BTC RV30 data (2020-01-28 → 2026-03-31), stays below a 5% gate on **both** sides of the book. The calibration procedure is implemented in `scripts/analyze_margin_capped.py` and runs as follows:

1. For each rolling 7-day window $W$, record the entry index $I_0$ and the per-position worst-case adverse excursion for both sides under the 2.5× cap: $\Delta^{\text{long}}_W = -\min_{t\in W}(I_t - I_0)/I_0$ clipped at $-1$, and $\Delta^{\text{short}}_W = +\max_{t\in W}(I_t - I_0)/I_0$ clipped at $c-1$.
2. For a candidate (IM, MM) pair, compute the fraction of windows in which the worst-case excursion $\times$ notional exhausts the collateral buffer $(\text{IM} - \text{MM})\times V$.
3. Scan candidates in a 2D grid of $(\text{IM}, \text{MM})$; pick the smallest IM at fixed MM $= 5\%$ that drives the liquidation rate below the gate.

The resulting empirical liquidation rates for the final table are:

| Tier | Max $V$ | IM | Long liq rate | Short liq rate |
|---:|---:|---:|---:|---:|
| 1 |    50,000 |  67% | **0.36%** | **3.65%** |
| 2 |   250,000 |  75% | 0.00% | 2.67% |
| 3 | 1,000,000 |  80% | 0.00% | 1.42% |
| 4 | 5,000,000 | 100% | 0.00% | 0.98% |
| 5 |         ∞ | 150% | 0.00% | 0.80% |

The worst-case observed rate on the worst side of the worst tier is **3.65%** — inside the 5% gate by a comfortable margin. At tiers 2 through 5 the long side is perfectly covered (no liquidations in the sample) and the short side is progressively better protected as collateral scales. The hold period of 7 days is the maximum the calibration procedure can tolerate: with a 60-day hold, even the capped contract produces uncomfortable tail events during the COVID 2020 window, which would force the IM rates even higher. Section 10 discusses the hold-period choice in detail.

## 9.4 Liquidation rule

A position is marked for liquidation when its equity falls below the maintenance margin:

$$ \text{equity}_t \;<\; \text{mm\_rate} \cdot V, $$

with

$$ \text{equity}_t \;=\; \text{collateral} \,+\, V\cdot (\mathrm{RVOL}_t - \mathrm{RVOL}_{\text{entry}}) \cdot (\pm 1) \;-\; \sum_{u\le t}\text{funding}_u, $$

where the $\pm 1$ is $+1$ for longs and $-1$ for shorts, and the vega-notional $V$ is expressed in USD per annualized vol point. On liquidation:

1. The position is flattened at the current mark (best effort through the orderbook).
2. Any shortfall (equity gone negative between the maintenance-margin threshold and the actual fill) is absorbed by the insurance fund.
3. Auto-deleverage (ADL) is triggered only if the insurance fund cannot cover the shortfall.

The liquidation index level — the value of $I_t$ at which the equity of an initially-well-collateralized position crosses the MM threshold — is computed in closed form by `rvol/margin/tiers.py::liquidation_index_level`. For a long,

$$ I_t^{\text{liq,long}} \;=\; I_{\text{entry}} \;-\; \frac{(\text{collateral} - \text{mm\_usd})}{V}\cdot \mathrm{d}v_{\text{per vol pt}}, $$

and symmetrically above entry for a short. This closed form holds within the linear band of the payoff (i.e., before the cap is reached). Beyond the cap the position cannot lose any further regardless of index moves, so no liquidation is possible in that region — the capped payoff makes the post-cap regime safe by construction, which is the single most important consequence of §9.2. The test `tests/test_margin.py::TestLiquidation` verifies the expected behavior: longs liquidate below entry and shorts above; more collateral pushes the liquidation level further from entry; entering with exactly the MM as collateral produces a liquidation level exactly at entry; invalid inputs (zero notional, zero entry index, unknown side) raise.

## 9.5 Insurance fund

The insurance fund is the catch-basin for residual losses when a liquidation fill does not fully cover the position's negative equity. It is seeded at \$500,000 and funded by a liquidation penalty paid on every closed-out position:

$$ \text{penalty} \;=\; 0.5\% \cdot V \cdot I_{\text{entry}} \cdot \text{(ann vol at entry)}. $$

The target steady-state balance is **\$5M within 90 days** of launch, assuming the first-month liquidation flow is consistent with the historical-replay gate rate of $\sim 1\%$ per week per open position. If the fund exhausts, auto-deleverage triggers in reverse-profit order (the most profitable counterparty is ADL'd first), matching the standard convention on HL and other perp venues.

The seed of \$500k is sized to cover the expected residual loss of approximately **10 simultaneous tier-1 liquidations at the 99th-percentile capped excursion**. At tier 1, $V = 50{,}000$; the 99th-percentile short excursion under the cap is 1.5× notional; the collateral buffer is $(\text{IM} - \text{MM}) \cdot V = 0.62 \cdot 50{,}000 = \$31{,}000$, leaving a worst-case residual of $(1.5 - 0.62) \cdot 50{,}000 = \$44{,}000$ per position. Ten simultaneous worst-case residuals is \$440k, within the \$500k seed. Larger tiers have proportionally more protection because their IM rates are higher, so the seed is sized by the smallest (most vulnerable) tier.

## 9.6 What the tier table assumes, and what happens when it doesn't hold

Three assumptions are load-bearing for the calibration, and each would require recalibration if it changes.

**(a) The 2.5× cap holds per position.** If the cap were removed or raised significantly (say to 5×), the short-side worst-case loss would scale linearly with the cap increment, and the IM rates would need to rise in step. At cap $= \infty$ the system reverts to the Phase 8.5 regime where 80% of positions liquidate at 10× and no finite leverage is safe.

**(b) The historical 2020–2026 sample is a reasonable prior for the future.** The calibration sample includes exactly one 3× regime break (COVID 2020). If a future regime produces a 5× break — BTC realized vol reaching 250%+ sustained for weeks — the 7-day empirical distribution will shift and the tier-1 long liquidation rate may exceed the 5% gate. The launch plan (§11) includes an empirical recalibration review every quarter using the expanding history, and a discretionary recalibration trigger if observed liquidation rates exceed 2× the calibrated gate in any rolling 90-day window.

**(c) The hold period is 7 days.** Longer holds produce fatter tail excursions and would require higher IM rates. We discuss the hold-period choice in §10 as part of the honest-negative-result discussion: 60-day holds against a capped payoff still produce uncomfortable liquidation rates, and the 7-day calibration is the honest maximum at which both sides of the book clear the gate.

These three assumptions are individually conservative but jointly tight. The tier table is the smallest IM schedule consistent with the historical data and the 2.5× cap, which means it has no fat built in — any tightening of assumptions (larger cap, longer hold, out-of-sample regime) will move some rate in the table upward.
