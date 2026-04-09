# Paper Numbers Sheet

Every empirical number cited in the paper, with its source doc, source script, and raw value. Use this to prevent citation drift between design docs, code, and paper prose. When a number here changes, update the paper in the same commit.

Format: `value — source file:section — script/code`.

---

## §4 Index spec (from `docs/design/01-index-spec.md`)

| Quantity | Value | Source §  |
|---|---|---|
| Return interval | 1 minute | §2.1, §3 |
| Log return | `r_t = ln(P_t / P_{t-1})` | §2.1 |
| Gap threshold | 65 seconds (1.5× interval) | §2.1, §3 |
| Cap form | `sign(r) × min(|r|, σ_cap · σ̂_t)` | §2.2 |
| MAD estimator | `σ̂_t = 1.4826 × median(|r_{t-1440..t-1}|)` | §2.2 |
| MAD lookback | 1,440 obs (1 day) | §2.2, §3 |
| Cap multiplier | `σ_cap = 4.0` | §2.2, §3 |
| Clip fraction under GBM | ≈0.006% of observations | §3 |
| Window length | 30 calendar days | §2.3, §3 |
| Expected window obs | 43,200 | §2.3, §2.5 |
| Validity threshold | ≥90% (38,880 obs) | §2.5, §3 |
| Outage tolerance | ~4.3 h per 30-day window | §3 |
| Annualization factor | 365 / 30 | §2.4, §3 |
| Display vol | `RVOL = sqrt(RV × 365/30)` | §2.4 |

Numerical example (§5): σ = 0.001/min ⇒ E[RV30] = 0.0432, annualized 0.526, vol ≈ 72.5%. Plausible BTC range `RV30 ∈ [0.0049, 0.123]` (≈ 20–100% ann vol).

Code: `rvol/index/{spec,returns,filters,variance}.py`.

---

## §5 Data

| Source | Dataset | Span | Script |
|---|---|---|---|
| Binance | BTCUSDT 1m klines + 8h funding | 2020-01-01 → present | `scripts/fetch_binance.py` |
| Hyperliquid | BTC trades (S3 + SDK) | 2023-03 → present | `scripts/fetch_hyperliquid.py` |
| Deribit | BTC-DVOL daily | 2019-01 → present | `scripts/fetch_deribit_dvol.py` |

BTC RV30 empirical range (from Phase 8.5 replay): **18.7%–147.7%** annualized vol across **54,120 hourly observations** spanning **2020-01-28 → 2026-03-31**. Source: `docs/design/07-phase-8.5-findings.md` §"What we ran".

---

## §6 RV30 empirical properties (from `docs/design/02-funding-mechanism.md` §"Parameter selection")

| Quantity | Value |
|---|---|
| RV30 daily change std | **6.1%** |
| RV30 AR(1) | **0.996** (highly persistent) |
| Mean VRP (short-side carry) | **+14 pp annualized** |

Source script: `scripts/analyze_data_quality.py`, `scripts/analyze_vrp.py`.

---

## §7 Funding mechanism (from `docs/design/02-funding-mechanism.md` + `docs/hip3/PARAMETERS.md`)

| Parameter | Value | Source |
|---|---|---|
| Interval | 1 hour | 02 §Functional form |
| Formula | `f_t = clip(B_t/k, −c, +c)` with `B_t=(M−I)/I` | 02 §Functional form |
| Dampening `k` | **100** | PARAMETERS.md (grid-searched, Phase 4) |
| Cap `c` | **0.001** (0.1%/h) | PARAMETERS.md |
| Sign convention | positive f → longs pay shorts | 02 §Notation |

Convergence results (from `PROPOSAL.md` §2.3): **50% basis shock half-life = 31h**; **cap-binding fraction < 1%** of hours in simulated regime (gate <5%).

Script: `scripts/simulate_funding.py`. Code: `rvol/funding/{rate,simulate}.py`.

Note: design doc's worked example suggested `k≈333, c≈0.005`; the **final calibrated values are `k=100, c=0.001`** from Phase 4 grid search (source of truth: `docs/hip3/PARAMETERS.md` and `PROPOSAL.md`). The paper must use the final values.

---

## §8 Manipulation resistance (from `docs/design/05-manipulation-cost.md` + `docs/hip3/PROPOSAL.md` §3.3)

Cost model: Kyle linear impact.
- **Depth calibration: $5,000,000 crosses the book to move BTC-USDT perp 1%** (5 bps slippage per $1M). Source: 05 §"Cost model".
- Round-trip factor: 2.

### Headline table (PROPOSAL §3.3, from `scripts/analyze_manipulation.py`)

Real BTC 1-minute return history: **3.28M minutes**.

| Attack | Uncapped Δvol_pt | Capped Δvol_pt | Cap reduction | Cost |
|---|---:|---:|---:|---:|
| Single 10σ spike | 0.030 | 0.005 | 84% | $24k |
| Single 20σ spike | 0.119 | 0.005 | 96% | $48k |
| Sustained 15σ × 1h | 3.84 | 0.27 | 93% | $4.3M |
| Sustained 15σ × 12h | 34.9 | 11.7 | 72% | $52M |

**Headline claim:** moving the index by 1 annualized vol point costs **≥ $3.7M** across all tested geometries. A 5 vol-pt move costs **> $20M**. Cap reduces single-spike impact by **≥ 93%** for k ≥ 15σ.

Gate check (from PARAMETERS.md §"Gate values"):
- Cap reduces large spike (k≥15σ) ≥90% → observed **≥93%** ✓
- Cost/vol-pt sustained ≥$1M → observed **≥$3.7M** ✓

Outputs: `data/processed/manipulation_cost_table.csv`, `manipulation_attack_curves.png`.

---

## §9 Margin & liquidation — **use the Phase 8.6 capped table**, not the original Phase 8 table

⚠️ The original Phase 8 design doc (`06-margin-liquidation.md`) lists a table claiming up to **10× leverage** with IM 10%/MM 5% at tier 1. **This table is superseded and empirically false** — Phase 8.5 replay demonstrated it produces 80% liquidation rates. The paper must use the **Phase 8.6** table below.

### Final tier table (source: `rvol/margin/tiers.py`, `docs/hip3/PARAMETERS.md`, `PROPOSAL.md` §5.1)

| Tier | Max V (USD/vol pt) | IM | MM | Max leverage |
|---:|---:|---:|---:|---:|
| 1 |    50,000 |  67% | 5% | **1.50×** |
| 2 |   250,000 |  75% | 5% | 1.33× |
| 3 | 1,000,000 |  80% | 5% | 1.25× |
| 4 | 5,000,000 | 100% | 5% | 1.00× |
| 5 |         ∞ | 150% | 5% | 0.67× (overcollateralized) |

- **Payoff cap:** `PAYOFF_CAP = 2.5` applied per position as `clip((I−I_entry)/I_entry, −1, 1.5)`.
- Short-side max loss: **1.5 × notional**. Long-side max loss: ≈−1.0 × notional.

### Calibration (from `scripts/analyze_margin_capped.py`, 7-day hold, 1,124 rolling windows, MM fixed at 5%)

| Tier | Long liq rate | Short liq rate |
|---:|---:|---:|
| 1 | 0.36% | 3.65% |
| 2 | 0.00% | 2.67% |
| 3 | 0.00% | 1.42% |
| 4 | 0.00% | 0.98% |
| 5 | 0.00% | 0.80% |

Both sides clear the **<5% gate** at every tier. Max long liq: **0.36%**. Max short liq: **3.65%**. Source: `docs/design/07-phase-8.5-findings.md` §"Phase 8.6 — resolution".

### Insurance fund (from `PARAMETERS.md`, `PROPOSAL.md` §5.2)

| Parameter | Value |
|---|---|
| Seed | $500,000 |
| Liquidation penalty | 0.5% × notional × entry index |
| Replenishment target | $5M within 90 days |

---

## §10 Historical stress test — the negative result (from `docs/design/07-phase-8.5-findings.md`)

Data: 54,120 hourly RV30 observations, 2020-01-28 → 2026-03-31, index range 18.7%–147.7% ann vol.

### Uncapped rolling-sweep liquidation rates (60-day hold, 314 windows)

| Tier | Leverage | Long liq | Short liq |
|---:|---:|---:|---:|
| 1 | 10.0× | **80.3%** | **75.5%** |
| 2 | 6.67× | 78.0% | 72.0% |
| 3 | 5.0× | 74.2% | 69.4% |
| 4 | 3.03× | 67.5% | 62.7% |
| 5 | 2.0× | 61.2% | 58.3% |

### COVID event study
Position opened **2020-02-11** at 45.8% ann vol. Over the next 60 days, vol peaks at **147.7%** (≈ 14× move in variance). Even tier-5 (2×, 25% MM) short loses.

### Empirical survival calibration (314 rolling 60-day windows)

| Survival | Worst long draw | Worst short draw | MM req | Max leverage |
|---|---:|---:|---:|---:|
| 90%  | −0.676 | +2.645 | 2.645 | 0.25× |
| 95%  | −0.779 | +3.435 | 3.435 | 0.19× |
| 99%  | −0.894 | +7.961 | 7.961 | 0.08× |
| 100% | −0.902 | +9.789 | **9.789** | 0.07× |

Worst observed 60-day short-side move: **+979%** of entry index. Survival-of-100% max leverage: **0.07×** (must post 15× notional in collateral).

### The fix (Phase 8.6)
Adopt Demeterfi-Derman-Kamal-Zou (1999) capped variance swap convention with `cap = 2.5`. Short-side max loss becomes `(2.5 − 1) × notional = 1.5 × notional`. Recalibrated tier table passes <5% liquidation gate at every tier (see §9). Leverage ceiling drops from claimed 10× (MC-calibrated, false) to **1.5×** (historical-calibrated, honest).

Outputs: `margin_historical_events.csv`, `margin_historical_rolling.csv`, `margin_historical_calibration.csv`.

---

## §11 On-chain realization (from `PROPOSAL.md` §4)

| Quantity | Value |
|---|---|
| Update cost | O(1) per minute |
| Buffer storage | 43,200 × 8 bytes ≈ **345 kB** |
| Funding cadence | 1 hour |
| Failure: no trades in minute | bucket NaN, excluded |
| Failure: obs <90% | `is_valid=False`, funding paused, new positions rejected |
| Circuit breaker | freeze funding if >5 missed updates |

Reference contract: `contracts/src/RvolIndex.sol` (TBD).

---

## §12 Applications — no empirical numbers yet (framework section)

Concepts only: gas-vol perp, correlation perp, depeg-risk perp, pre-IPO self-referential derivatives. Cite as "oracle-free perpetual on any computable statistic."

---

## Launch caps (from `PARAMETERS.md`)

Position size capped at **50% of tier maximum** for the first **14 days**. Lifted to full after basis + funding observed healthy.

---

## MC calibration parameters (from Phase 5, for §10 methodological note)

Log-normal SV model calibrated to BTC:
- θ = **−7.32**
- κ = **0.00363**
- η = **0.06**
- ρ = **−0.08**

This model produced **0% liquidations at tier-max leverage** in Phase 8, which is the failure that motivated the Phase 8.5 historical-replay gate. Paper methodological lesson: SV MC is insufficient as a margining calibration — real regimes are fatter than calibrated SV reproduces.

---

## Citations cross-reference

| Paper claim | Cite |
|---|---|
| Linear variance payoff, static replication | Carr & Lee (2003); Neuberger (1994); Demeterfi et al. (1999) |
| 2.5× cap convention | Demeterfi, Derman, Kamal, Zou (Goldman, 1999) |
| RV as Σ r² → QV | Andersen, Bollerslev, Diebold, Labys (2001, 2003); Barndorff-Nielsen & Shephard (2002) |
| MAD estimator factor 1.4826 | standard robust stats (normal consistency) |
| Kyle linear impact | Kyle (1985) |
| VRP positivity | Carr & Wu (2009); Bakshi & Kapadia (2003); Alexander & Imeraj (2021) for BTC |
| Log-normal SV | Hull & White (1987) |
| Squeeth / power perps | Opyn/Paradigm (2021) |
| HIP-3 venue | Hyperliquid HIP-3 spec |
| Oracle manipulation motivation | Eskandari et al. (2024) survey; BIS Bulletin 76 (2023) |
