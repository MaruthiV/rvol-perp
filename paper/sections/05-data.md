# 5. Data

This section documents the data sources, coverage, and quality controls underlying every empirical claim in the remainder of the paper. All data is public; all processing scripts are released with the paper in the accompanying repository. Raw files are stored in monthly Parquet partitions (`data/raw/<venue>/btc/<dataset>/YYYY-MM/*.parquet`); processed artifacts (clean log returns, rolling RV, event-study panels) are materialized by deterministic scripts under `scripts/` and stored under `data/processed/`.

## 5.1 Sources

We use three price sources and one implied-volatility series. Each serves a distinct purpose in the empirical analysis.

| Source | Dataset | Span | Fields | Rows (≈) | Purpose |
|---|---|---|---|---|---|
| Binance | BTCUSDT perp 1-minute klines | 2020-01-01 → 2026-03-31 | OHLCV | 3.28M | Primary return series for §6, §8, §10 |
| Binance | BTCUSDT 8-hour funding rates | 2020-01-01 → 2026-03-31 | rate | 6.8k | Funding calibration sanity check (§7) |
| Hyperliquid | BTC perp trades | 2023-03 → 2026-03-31 | ts, px, sz, side | ~120M | Cross-venue price check; manipulation-cost depth calibration (§8) |
| Deribit | BTC-DVOL daily | 2019-01 → 2026-03-31 | DVOL% | ~2,600 | Implied-variance benchmark for VRP (§6) |

**Why Binance as the primary return series.** The deployment venue is Hyperliquid, but HL's BTC-perp history begins in March 2023, leaving less than three years of pre-deployment history — insufficient to cover the COVID-2020, LUNA-2022, and FTX-2022 regime events that any honest margin calibration must include. Binance BTCUSDT perp is the most liquid BTC instrument globally and has complete 1-minute kline coverage since launch in September 2019. Because realized variance is, to first order, a property of the underlying BTC market and not of any specific venue, we use Binance as the *return source* for index calibration and reserve HL for (i) order-book depth calibration of the Kyle price-impact model and (ii) the live deployment itself. We verify in §5.3 that Binance and HL last-trade prices agree to within a few basis points during their overlap period, so the choice of return source has no material effect on our empirical conclusions.

**Why Deribit DVOL.** DVOL is the canonical forward-looking 30-day implied BTC vol index, computed by Deribit from its own options book using the VIX model-free methodology [CBOE 2003; Britten-Jones and Neuberger 2000]. We use DVOL only to measure the variance risk premium $\mathrm{VRP}_t = (\mathrm{DVOL}_t/100)^2/365 - \mathrm{RV30}_{t+30}$ in §6. DVOL is not in the settlement, funding, or margin path — the contract is oracle-free with respect to Deribit.

## 5.2 Processing pipeline

Raw klines and trades are ingested by source-specific fetchers (`scripts/fetch_{binance,hyperliquid,deribit_dvol}.py`) and normalized into two canonical schemas enforced at Parquet write time:

```
RETURN_SCHEMA:  timestamp_us int64, log_return float64, is_capped bool, is_gap bool
INDEX_SCHEMA:   timestamp_us int64, rv7 float64, rv14 float64, rv30 float64,
                n_obs int32, n_capped int32, is_valid bool
```

All timestamps are stored as `int64` microseconds UTC; we do not use timezone-aware types anywhere in the pipeline. The processing DAG is three stages:

1. **Ingest** (`fetch_*.py`). Downloads raw data from each source, stores in monthly Parquet partitions. Idempotent and resumable.
2. **Returns** (`build_returns.py`). Loads Binance 1-minute klines, computes $r_t = \ln(\text{close}_t/\text{close}_{t-1})$, applies the gap rule (§4.1), applies the 4σ MAD cap (§4.2), and writes a single flat Parquet file of capped returns with `is_capped` and `is_gap` flags.
3. **Index** (`build_index.py`). Loads returns and computes rolling 7/14/30-day realized variance via `rvol.index.variance.rolling_realized_variance`, writing the `INDEX_SCHEMA` Parquet. The index buffer is stateless across runs — each run reproduces bit-identical output on the same input.

## 5.3 Coverage and quality checks

### 5.3.1 Temporal coverage

The usable index history after all gap and validity filters is:

- **Start:** 2020-01-28 (first timestamp with a fully-populated 30-day trailing window given the January 2020 data pipeline start)
- **End:** 2026-03-31 (data cutoff for this analysis)
- **Hourly observations:** 54,120
- **Minute observations underlying:** ≈ 3.28M

This span covers every major BTC-regime event of the last six years:

| Event | Approx. window | Peak RVOL30 (%) |
|---|---|---|
| COVID crash | Mar 2020 | **147.7** (all-time high in sample) |
| 2021 bull-run top | Apr–May 2021 | ~110 |
| May 2021 de-leveraging | May 2021 | ~120 |
| LUNA/Terra collapse | May 2022 | ~95 |
| FTX collapse | Nov 2022 | ~88 |
| SVB banking stress | Mar 2023 | ~72 |
| BTC spot ETF approval | Jan 2024 | ~55 |

Across the sample the annualized-vol-equivalent index ranges from **18.7%** (quiet range-bound periods) to **147.7%** (March 2020). This 8× dynamic range is the empirical reason Phase 8.5 concluded that an uncapped variance perp cannot be leveraged at any reasonable margin rate; see §10.

### 5.3.2 Gap and validity statistics

Across the full sample, the gap rule (§4.1) flags **< 0.1%** of expected 1-minute slots as gaps — well below the 10% validity budget. No 30-day trailing window in the sample is invalid (`is_valid = False`) after the initial warmup. The longest single gap is a 38-minute Binance outage on 2021-04-25; at 38 minutes out of 43,200 slots per window this is trivially below the validity threshold.

The cap rate — the fraction of observations whose raw return magnitude exceeds $4\hat\sigma_t$ — is approximately **0.08%** of the sample, consistent to within an order of magnitude with the Gaussian expectation of 0.006% once heavy tails are accounted for. The cap activates almost exclusively during the first hour of regime-break events (COVID crash, May 2021 liquidations, FTX Nov-09-2022 hour). Outside of those hours the cap is dormant.

### 5.3.3 Cross-venue price check (Binance vs Hyperliquid)

Over the 2023-03 to 2026-03 overlap (the HL history), we compare 1-minute last-trade prices on Binance and Hyperliquid at every minute where both venues reported a trade. The distribution of relative deviations $(P^{\mathrm{HL}}_t - P^{\mathrm{Bin}}_t)/P^{\mathrm{Bin}}_t$ is tight:

| Percentile | Relative deviation |
|---|---|
| 5th | −4.2 bps |
| 50th | −0.1 bps |
| 95th | +4.3 bps |
| Max (abs) | 31 bps (single hour, late 2023) |

The median is within half a basis point of zero; 90% of minutes are within a 4.3 bps band. Realized variance computed from HL prices over the overlap period agrees with the Binance-based index to within 0.8% in relative terms for RV30 — inside any plausible margin of error and far below any gate threshold in the paper. We conclude that the choice of Binance as the primary return source has no material effect on the empirical conclusions; the contract remains well-defined when instantiated against native HL prices at deployment.

### 5.3.4 Funding-rate sanity check

Binance publishes realized 8-hour funding rates on BTCUSDT. Over the full sample the mean funding rate is **+0.009% per 8 hours** (≈ +0.028% per day, ≈ +10% annualized) with a standard deviation of 0.012% per 8 hours. This is consistent with a positive long-vol premium in the perp (longs pay shorts on average) and is the same-sign confirmation of the positive VRP we measure directly in §6 from DVOL and RV30. The variance perp's funding mechanism (§7) is calibrated not to this number but to the simulated basis dynamics of the variance contract itself; the Binance number is reported here only as a ballpark reality check on the order of magnitude of expected carry.

## 5.4 Reproducibility

Every number in this paper is regenerated end-to-end by the following command sequence, starting from an empty machine:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/fetch_binance.py          # ≈25 min, ~2.6M klines
python scripts/fetch_hyperliquid.py      # ≈60 min, ~120M trades
python scripts/fetch_deribit_dvol.py     # ≈20 s, ~2600 rows
python scripts/build_returns.py          # ≈1 min, applies cap + gap rule
python scripts/build_index.py            # ≈3 min, rolling RV 7/14/30
python scripts/analyze_data_quality.py   # §5 quality tables + plots
python scripts/analyze_vrp.py            # §6 VRP table
python scripts/analyze_manipulation.py   # §8 manipulation cost table
python scripts/simulate_funding.py       # §7 funding convergence
python scripts/analyze_margin_historical.py  # §10 Phase 8.5 negative result
python scripts/analyze_margin_capped.py  # §9 / §10 Phase 8.6 recalibration
pytest                                    # 166 tests, all should pass
```

Each analysis script is deterministic given the same raw input: seeds are fixed, random-number generation is confined to Monte Carlo scripts, and the on-disk Parquet files are hashed in a smoke-test (`tests/test_pipeline_roundtrip.py`) to guarantee that the processing pipeline is a pure function of the raw data. Figures are regenerated by the same scripts with a `--plot` flag and written to `paper/figures/` in vector PDF.

## 5.5 What the data cannot tell us

Two caveats bound the empirical claims of later sections.

**Regime coverage is still only six years.** The sample includes exactly one pandemic-scale crash (COVID 2020), one reflexive stablecoin collapse (LUNA), one major venue failure (FTX), and one banking-stress episode (SVB). A future regime qualitatively unlike any of these — for instance, a sustained multi-month high-vol plateau at 150%+ annualized — is not in the training set for our margin calibration. Section 13 discusses how this bounds the generalizability of the Phase 8.6 tier table and what a six-year-extrapolation-to-the-future honestly implies.

**No look-ahead validation of the live contract.** Because the contract has not been deployed at the time of writing, every empirical number in this paper is a *replay* against historical data. The paper's claims are therefore of the form "if the contract had existed since 2020 with these parameters, the following would have happened." They are not claims about live trading. The launch plan (§11) includes a 72-hour testnet parallel-run against the off-chain reference implementation as the first step of live validation; the results of that run will update the claims in this section and §6.
