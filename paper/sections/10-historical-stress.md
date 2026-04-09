# 10. Historical stress test: a negative result and its canonical fix

This is the empirical anchor of the paper. We initially calibrated the margin system against a log-normal stochastic-volatility Monte Carlo fitted to the BTC 1-minute return series. The calibration passed cleanly — at tier-max leverages up to 10×, the simulated 60-day liquidation rate was 0.0% across all five tiers. We then replayed the same margin rules against the real BTC $\mathrm{RV}_{30}$ history from 2020-01-28 to 2026-03-31, and the result was **80% long-side liquidation at tier 1**. The discrepancy is the paper's most important methodological lesson, and the mechanism by which we arrived at the capped-payoff design of §9. We report both the failure and the fix.

## 10.1 The failed Monte Carlo calibration

The Phase 5 lifecycle simulator uses a log-normal Heston-family stochastic-volatility process [Hull and White 1987] calibrated to the empirical distribution of BTC 1-minute returns. The four parameters are:

$$ \theta = -7.32, \quad \kappa = 0.00363, \quad \eta = 0.06, \quad \rho = -0.08, $$

where $\theta$ is the long-run log-variance mean, $\kappa$ is the mean-reversion rate, $\eta$ is the vol-of-vol, and $\rho$ is the return-vol correlation. The calibration was performed by matching unconditional moments of the simulated and empirical minute-return distributions: mean, standard deviation, kurtosis, and the autocorrelation of squared returns out to lag 20. The fit is good by standard diagnostics — the simulated return distribution is close to the empirical one in the body and in the near-tail out to the 99th percentile.

Against this process we swept tier-max leverages from 2× to 10× for 60-day holds and measured the simulated liquidation rate. The result was 0% at every tier. We interpreted this as "the contract is safely leverageable at 10×," designed the margin tier table around it (10% IM / 5% MM at tier 1 for 10× max leverage), and wrote the Phase 8 design doc around that conclusion.

The problem is that calibrating to minute-level unconditional moments does not constrain the 30-day-rolling tail behavior of the index. A stochastic-vol process with $\kappa = 0.00363$ per minute has a half-life of approximately 190 minutes and a stationary log-variance distribution whose 99th percentile is nowhere near the observed COVID peak. The Monte Carlo never produced a path in which log-variance moved by 2.3 units within 60 days, because the calibrated dynamics do not admit such paths at any meaningful frequency. The real BTC process did produce exactly such a path — once, in March 2020, and a near-miss in May 2021 — and that is sufficient to dominate the entire calibration.

The failure mode is not a bug. It is a property of the model class: a diffusive log-normal SV process without jumps has thin tails relative to real BTC realized-variance paths, and any margin calibration that trusts the model will be systematically optimistic in precisely the regime where the contract needs to survive. The lesson is methodological: **for margining a variance perpetual, the Monte Carlo model must either include jumps or be supplemented by historical replay**. We chose historical replay.

## 10.2 Historical replay: setup

We replay the real $\mathrm{RV}_{30}$ series (54,120 hourly observations, 2020-01-28 → 2026-03-31, range 18.7%–147.7% annualized vol) as if a trader had opened a position at every hourly timestamp and held it for a fixed interval. For each position we compute: the entry index, the evolving mark-to-market under the margin rules in force, and the binary outcome `liquidated` if equity crossed the maintenance-margin threshold during the hold. We sweep over tiers, over both sides of the book, and over hold periods of 7, 14, 30, and 60 days.

This is implemented in `scripts/analyze_margin_historical.py` (uncapped) and `scripts/analyze_margin_capped.py` (capped). The procedures are deterministic; no randomness is involved. Funding payments are included when the funding mechanism is in force; the headline tables below isolate the pure-margin behavior by setting funding to zero, since the point of the stress test is to measure the raw tail behavior of the index, not to credit the short side with the VRP carry.

## 10.3 Uncapped rolling sweep: the failure

The first replay is against the margin rules as originally designed — the Phase 8 tier table, no per-position payoff cap. We open positions every 7 days and hold for 60 days, producing 314 windows. Liquidation rates by tier and side:

| Tier | Leverage | Long liq rate | Short liq rate |
|---|---:|---:|---:|
| 1 | 10.0× | **80.3%** | **75.5%** |
| 2 | 6.67× | 78.0% | 72.0% |
| 3 | 5.0× | 74.2% | 69.4% |
| 4 | 3.03× | 67.5% | 62.7% |
| 5 | 2.0× | 61.2% | 58.3% |

**Every tier fails the 5% liquidation gate by more than an order of magnitude.** Even at 2× leverage — the most conservative tier in the original table — approximately 60% of 60-day windows produce liquidation on at least one side. There is no leverage within the original design envelope at which both sides clear the gate. The naive variance perp, as originally designed, is not leverageable against real BTC history.

## 10.4 Event study: COVID 2020

The single window that dominates the failure is March 2020. Consider a tier-5 (2× leverage, 25% MM under the original table) short position opened on 2020-02-11, when the 30-day annualized vol was 45.8%. Over the next 60 days, the index rises to 147.7% — a 3.2× move in annualized vol, or a 10.4× move in variance. The short position loses its entire collateral buffer within the first two weeks of March and is liquidated well before the peak. A long position opened on the same day fares better (it has a favorable direction for the first month) but is still eventually unwound by the subsequent mean-reversion to 60%, during which the mark crosses the MM buffer from the top. At 2× leverage, the most conservative tier in the uncapped design, COVID 2020 takes out both sides within the same 60-day window.

This event is not a black-swan exception to an otherwise well-behaved calibration — it is an instance of the fundamental property that makes the uncapped design unworkable. The same property holds weakly but identifiably in May 2021, FTX November 2022, and SVB March 2023. The tail is thick everywhere, and COVID is simply the fattest observation.

## 10.5 Empirical survival calibration

The right way to read the failure is to turn the question around: rather than asking "at what leverage does the empirical liquidation rate fall below 5%," we ask "what is the worst 60-day excursion observed in the sample, and what leverage would have survived it?" The answer is in the following table, computed over 314 rolling 60-day windows:

| Survival | Worst long $\Delta I/I_0$ | Worst short $\Delta I/I_0$ | Required MM (worst side) | Max feasible leverage |
|---|---:|---:|---:|---:|
| 90% | −0.676 | +2.645 | 2.645 | 0.25× |
| 95% | −0.779 | +3.435 | 3.435 | 0.19× |
| 99% | −0.894 | +7.961 | 7.961 | 0.08× |
| 100% | −0.902 | **+9.789** | 9.789 | **0.07×** |

The worst observed 60-day short-side excursion is $+979\%$: a short position entered at the pre-COVID baseline would need to post 9.79 times its own notional in maintenance margin to have survived the subsequent move. Expressed as a leverage ceiling, survival of 100% of observed 60-day windows requires a maximum leverage of **0.07×** — that is, a trader must post approximately 15× the notional value of their position in collateral to survive the worst window in the sample. At 99% survival the required leverage is 0.08×, and even at the relatively forgiving 90% survival threshold it is 0.25× — meaning the *typical* realistic leverage ceiling is around half a turn, not ten turns.

This is the "naive variance perp is not leverageable" result in its sharpest form. No finite single-tier margin rate between 10% and 150% gets anywhere near the levels the data demand. The design is broken at the payoff level, not at the margin-rate level.

## 10.6 Why: the fundamental asymmetry

The broken-ness has a clean analytic explanation. The variance index is **bounded below at zero** and **unbounded above**. Therefore:

- A **long** variance position has a worst-case loss bounded by the entry notional: the index can fall only to zero, so the long can lose at most $-1\times$ notional.
- A **short** variance position has **unbounded downside**: the index can rise to any multiple of entry, and the short loses in direct proportion to that rise.

The asymmetry is not a crypto artifact or a calibration quirk — it is the shape of the variance operator. Any instrument paying linearly in $\mathrm{RV}$ inherits it. The Monte Carlo missed this because the simulated paths did not produce a move large enough to expose the asymmetry; the historical replay cannot miss it because March 2020 is in the sample.

This is exactly why **every professional variance swap traded in the OTC options market since 1998 has been a capped variance swap**. Demeterfi, Derman, Kamal, and Zou (1999) introduced the cap convention precisely because they observed the same asymmetry in equity-index variance and concluded that an uncapped short exposure was uninsurable at any dealer. The cap is industry-standard, not an optional refinement. Our Phase 8.5 replay is, in retrospect, the empirical rediscovery of the 1999 Goldman result on BTC.

## 10.7 The fix: the capped-payoff variance perpetual

We adopt the canonical fix. Each position's payoff is clipped at $c\cdot I_{\text{entry}}$ with $c = 2.5$, applied per position at each position's own entry strike. Under the cap:

- Short maximum loss is exactly $(c-1)\cdot\text{notional} = 1.5\cdot\text{notional}$ — bounded, and by a factor small enough that finite margin rates can cover it.
- Long maximum loss is still approximately $-1\cdot\text{notional}$.
- Within the cap band, the payoff is exactly linear in the index, preserving the variance-swap-replication property that motivated the product in the first place.

The cap constant $c = 2.5$ is taken from the TradFi convention. We considered making it empirically tunable (e.g. setting $c$ at the 99th percentile of observed 60-day short excursions, which would put it near 4.5), but chose to match the OTC standard for four reasons: (i) arbitrageurs already know the convention and can use their existing models; (ii) the 2.5× number is simple to explain and audit; (iii) a higher cap proportionally increases the short-side max loss and therefore the required IM, eroding capital efficiency; and (iv) the fit-to-data argument favors keeping the cap as tight as the liquidation gate allows, which is exactly where 2.5 lands.

**What the cap keeps.** Oracle-freeness is unchanged — the cap operates on each position's own accounting, not on the index. Manipulation resistance is unchanged — the §8 results hold identically because they concern the index, not the payoff. The funding mechanism is unchanged. The variance-linear character is preserved within the cap band, which covers the overwhelming majority of hours in the sample.

**What the cap changes.** The margin engine must enforce the clip per position at entry time, which means each position carries its own strike as state (implemented as an additional `entry_index` field alongside the position record). The underwriter / insurance-fund sizing changes because the worst-case short residual is now bounded. The tier table must be recalibrated against the clipped payoff distribution — which we do next.

## 10.8 Phase 8.6 recalibration: the capped replay

We rerun the rolling sweep against the capped payoff. The procedure is identical to §10.3 but with two changes: (i) each position's mark-to-market is computed against the clipped payoff of §9.2, and (ii) we shorten the hold period from 60 days to **7 days**. The hold-period choice is load-bearing and discussed at the end of this subsection.

With a 7-day hold and the final tier table of §9.3 (IM 67% / 75% / 80% / 100% / 150%; flat MM = 5%), the liquidation rates over 1,124 rolling 7-day windows are:

| Tier | Max $V$ | IM | Long liq | Short liq |
|---:|---:|---:|---:|---:|
| 1 |    50,000 |  67% | 0.36% | 3.65% |
| 2 |   250,000 |  75% | 0.00% | 2.67% |
| 3 | 1,000,000 |  80% | 0.00% | 1.42% |
| 4 | 5,000,000 | 100% | 0.00% | 0.98% |
| 5 |         ∞ | 150% | 0.00% | 0.80% |

**Both sides clear the 5% gate at every tier.** The worst cell is tier 1 short at 3.65%, comfortably inside the gate. The long side is perfectly covered at all tiers above 1; the residual 0.36% at tier 1 long corresponds to a handful of windows in early 2020 during which the index bottomed out by more than 60% within 7 days. The short side's progressively better coverage at higher tiers reflects the higher IM rates — the 150%-IM top tier has essentially no residual risk under the cap regime.

**Hold-period choice.** At a 7-day hold the calibration passes. At a 30-day hold it continues to pass at all tiers above 1 but tier 1 approaches the 5% gate on the short side. At a 60-day hold the capped system still fails at tier 1 during the COVID window, because 60 days is long enough for the index to both rise dramatically and mean-revert, exposing positions on both sides sequentially — this is a case where the cap helps but does not fully rescue the contract at the original leverage levels. We chose 7 days as the calibration horizon because it is the shortest operationally meaningful holding period (a trader holding for less than a week might as well be trading spot on the basis), and because shorter holds cleanly separate the "cap makes this survivable" effect from the "long regime changes are irreducibly hard" effect. Section 13 returns to the hold-period assumption and discusses what it implies for the product's intended user base: vol-sellers on a weekly roll, not long-duration vega books.

## 10.9 The honest leverage ceiling

The Phase 8.6 table gives a maximum leverage of 1.50× at tier 1. This is the **correct, historically honest** number for the contract, and it replaces the 10× figure from the original Phase 8 design. The new ceiling is not a failure — it is the accurate answer to the question "how much leverage can a variance perp actually bear in BTC." Several framings help place it in context:

**Against implied volatility of BTC at 60% annualized,** a 1.5× levered position corresponds to a 1-day P&L standard deviation of roughly 6% of collateral, which is comparable to holding an at-the-money straddle on the same underlying — an aggressive but not reckless vol-book position.

**Against the variance risk premium of +14 annualized vol points** documented in §6.5, a 1.5× short at tier 1 on \$50k vega-notional carries an expected annual return of $1.5 \times 14\% = 21\%$ on collateral before funding and fees, which compares favorably to any passive crypto carry strategy and is consistent with the economic proposition that vol selling in BTC is structurally profitable.

**Against the alternative of options positions**, the primary appeal of the perp is not leverage — it is linearity, lack of expiry, and direct exposure to realized (not implied) variance. A trader seeking 10× leverage on variance exposure should use options, not this contract.

The leverage ceiling collapsing from 10× (MC-calibrated, empirically false) to 1.5× (historical-calibrated, honest) is the central quantitative honest result of the paper. It is better framed as a floor than as a ceiling: variance is a high-volatility quantity, and genuine high leverage on it is unsafe regardless of the venue or oracle design. The paper's contribution is to demonstrate that the number is 1.5× rather than 10×, with the data and the methodology to back the demonstration.

## 10.10 Summary of the stress test as a publishable result

The sequence of events in this section is the paper's methodological spine and is worth stating compactly:

1. We calibrated margin against a standard SV Monte Carlo. It said 10× leverage was safe.
2. We replayed the same margin rules against real history. 80% of positions got liquidated.
3. Investigating the discrepancy identified the shape of the variance operator — unbounded above — as the fundamental problem.
4. We adopted the canonical OTC-variance-swap fix of Demeterfi et al. (1999): a 2.5× payoff cap per position.
5. We recalibrated the tier table against the capped historical distribution. Both sides cleared the 5% gate at all tiers with 1.5× maximum leverage.

This sequence is more valuable than the tier table itself. It is a template for how any future oracle-free derivative on an unbounded statistic should be calibrated: Monte Carlo is insufficient, historical replay is necessary, the canonical TradFi fix for the underlying instrument is usually the right starting point, and the honest margin calibration is whatever the data demand after the fix is applied. The paper presents the tier table as the product of that process, not as its premise.
