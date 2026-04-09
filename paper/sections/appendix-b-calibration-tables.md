# Appendix B. Calibration tables

This appendix collects the full numerical calibration output referenced throughout the paper. Every table is reproducible from a single script in the accompanying repository under `scripts/`, and each row is traceable to a specific row of the output artifact under `analysis/` or `data/processed/`.

## B.1 Index parameters (frozen at deployment)

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Sampling interval | $\Delta t$ | 60 s | §4.1 |
| RV window length | $W$ | 43,200 bars (30 days) | §4.1 |
| MAD lookback | $L_{\text{MAD}}$ | 1,440 bars (1 day) | §4.2 |
| MAD cap multiplier | $\sigma_{\text{cap}}$ | 4 | §4.2 |
| Normal-consistency constant | $\kappa$ | 1.4826 | §4.2 |
| Max inter-bar gap | $G_{\max}$ | 65 s | §4.3 |
| Min valid observations | $\nu_{\min}$ | 38,880 (90% of $W$) | §4.4 |
| Annualization factor | $A$ | 525,600 min/year | §4.5 |

## B.2 Funding parameters

| Parameter | Symbol | Value | Method |
|---|---|---|---|
| Funding cadence | — | hourly | §7.2 |
| Basis normalization | $k$ | 100 | grid-search, §7.4 |
| Clip level | $c$ | 0.001 (10 bps/hr) | grid-search, §7.4 |
| Cap binding frequency | — | < 1% of hours | §7.4 |
| Basis convergence half-life | — | ≈ 31 hours | §7.4 |

## B.3 Empirical RV30 summary statistics (2020-01-28 → 2026-03-31)

| Statistic | Value |
|---|---|
| Sample length | 54,120 hourly observations |
| Mean annualized vol | ≈ 62% |
| Median annualized vol | ≈ 58% |
| Minimum | 18.7% |
| Maximum (COVID 2020-03) | 147.7% |
| AR(1) coefficient | 0.996 |
| Std. of daily changes | 6.1 ann. vol points |
| DVOL correlation | 0.82 |
| Mean DVOL − RV30 gap | +16.4 ann. vol points |
| Mean VRP | +14 ann. vol points |
| VRP positive-day fraction | 71% |
| VRP Newey-West t-stat | ≈ 8.9 |

## B.4 Regime event table

| Event | Date | Peak RV30 (ann. %) |
|---|---|---|
| COVID crash | 2020-03 | 147.7 |
| May 2021 liquidation cascade | 2021-05 | ≈ 115 |
| LUNA/UST collapse | 2022-05 | ≈ 95 |
| FTX bankruptcy | 2022-11 | ≈ 85 |
| SVB banking stress | 2023-03 | ≈ 75 |
| Spot-ETF approval week | 2024-01 | ≈ 70 |
| Low-vol trough | 2023-10 | 18.7 |

## B.5 Manipulation cost (Kyle linear impact, $\lambda = \$5\text{M}/1\%$)

| Attack geometry | Duration | Index move (ann. vol pts) | Adversary cost (USD) | Cost per vol-point |
|---|---|---|---|---|
| Single 10σ spike | 1 min | ≈ 0.3 | ≈ \$5.0M | ≈ \$16.7M |
| Single 20σ spike | 1 min | ≈ 0.6 | ≈ \$10.0M | ≈ \$16.7M |
| Sustained 15σ | 1 hour | ≈ 1.0 | ≈ \$3.7M | ≈ \$3.7M |
| Sustained 15σ | 12 hours | ≈ 5.0 | ≈ \$20.0M+ | ≈ \$4.0M |

Gate: minimum cost-per-vol-point across all tested geometries ≥ \$3.7M. Cap attenuation of single-minute shock at $k \ge 15\sigma$: ≥ 93%.

## B.6 Historical replay — uncapped design (Phase 8.5, 2020–2026)

Rolling 7-day windows, 1,124 total. Tier IM rates per the original Phase 8 design (10% at tier 1, rising to 50% at tier 5). Liquidation = position equity hits MM before window end.

| Tier | Max size (\$ vega-notional) | IM | Max lev | Long liq rate | Short liq rate |
|---|---|---|---|---|---|
| 1 | 50,000 | 10% | 10.0× | 67% | **80.3%** |
| 2 | 250,000 | 15% | 6.67× | 52% | 71% |
| 3 | 1,000,000 | 25% | 4.0× | 33% | 54% |
| 4 | 5,000,000 | 35% | 2.86× | 21% | 39% |
| 5 | 25,000,000 | 50% | 2.0× | 11% | 24% |

**Conclusion:** the uncapped design is not leverageable at any tier at Phase 8 IM rates; the short side in particular fails catastrophically at tier 1. Worst observed 60-day short excursion: **+979%** of entry index.

## B.7 Historical replay — capped design, Phase 8.6 (payoff cap $c = 2.5$)

Same 1,124 rolling 7-day windows. Per-position cap: short PnL clipped at $-2.5 \cdot K_{\text{entry}}$ relative to each position's own entry strike. Recalibrated IM rates and new tier caps.

| Tier | Max size (\$ vega-notional) | IM | MM | Max lev | Long liq rate | Short liq rate |
|---|---|---|---|---|---|---|
| 1 | 50,000 | 67% | 5% | **1.50×** | 0.36% | 3.65% |
| 2 | 250,000 | 80% | 5% | 1.25× | 0.18% | 2.93% |
| 3 | 1,000,000 | 100% | 5% | 1.00× | 0.09% | 2.22% |
| 4 | 5,000,000 | 125% | 5% | 0.80× | 0.00% | 1.51% |
| 5 | 25,000,000 | 150% | 5% | 0.67× | 0.00% | 0.80% |

**Gate:** both long and short liquidation rates < 5% at every tier. **PASS.** Headline number: **1.5× max leverage at tier 1** — the honest leverage ceiling reported in §10.9.

## B.8 Monte Carlo vs historical replay (the gap)

Same tier-1 10× leverage point; both calibrations executed against the 60-day-hold Phase 5/8.5 specification.

| Calibration | Model | Reported liquidation rate at 10× | Verdict |
|---|---|---|---|
| Phase 5 Monte Carlo | Hull-White log-normal SV, $\theta=-7.32$, $\kappa=0.00363$, $\eta=0.06$, $\rho=-0.08$ | **0.0%** | Model too benign |
| Phase 8.5 Historical replay | BTC 2020–2026 | **80.3%** | Ground truth |

The gap between these two numbers is the paper's methodological anchor (§10).

## B.9 Insurance fund sizing

| Parameter | Value |
|---|---|
| Seed at launch | \$500,000 |
| 90-day replenishment target | \$5,000,000 |
| Liquidation penalty rate | 0.5% of vega-notional |
| Worst-case tier-1 residual (simultaneous liq) | ≈ \$125,000 |
| Auto-deleverage priority | most-profitable counterparty first |

## B.10 Data coverage

| Source | Asset | Granularity | Coverage | Primary use |
|---|---|---|---|---|
| Binance | BTCUSDT | 1-minute klines | 2020-01-28 → 2026-03-31 | RV30 computation (primary) |
| Hyperliquid | BTC-USD perp | Trade stream | 2023-06 → 2026-03-31 | Cross-check, launch target |
| Deribit | DVOL | Daily | 2020-01 → 2026-03-31 | VRP measurement |

Binance vs HL deviation over the 2023-06 → 2026-03-31 overlap: median absolute 1-minute return difference < 2 bps, 99th percentile < 8 bps. The two streams are sufficiently close that a Binance-calibrated index transfers to HL without material recalibration.

## B.11 Reproducibility

Every table in this appendix regenerates from a single script invocation:

| Table | Script | Output |
|---|---|---|
| B.3 | `scripts/analyze_rv30_summary.py` | `analysis/rv30_summary.json` |
| B.4 | `scripts/analyze_regimes.py` | `analysis/regime_events.csv` |
| B.5 | `scripts/analyze_manipulation_cost.py` | `analysis/kyle_attack_table.csv` |
| B.6 | `scripts/analyze_margin_uncapped.py` | `analysis/phase_8_5_uncapped.csv` |
| B.7 | `scripts/analyze_margin_capped.py` | `analysis/phase_8_6_capped.csv` |
| B.8 | `scripts/compare_mc_vs_replay.py` | `analysis/mc_vs_replay.json` |

A full reproduction from a clean machine: `pytest` then `make paper-tables`.
