# 11. On-chain realization

This section specifies how the index, funding mechanism, and margin system of the preceding sections are implemented on-chain. The contract is designed for deployment via HIP-3 on Hyperliquid, but the design is venue-agnostic — anything required of the host chain is a standard capability (read access to the venue's own trade state, persistent contract storage, scheduled or triggered execution). We describe the oracle computation, the state layout, the update cost, the failure modes, and the launch sequence.

## 11.1 The contract stack

The contract consists of three logical modules, each with distinct state and distinct update cadence:

1. **`RvolIndex`** — the oracle. Maintains the MAD estimator, the capped-return ring buffer, and the running sum that together define $\mathrm{RV}_T$. Updates every minute.
2. **`RvolFunding`** — the funding module. Computes the hourly funding rate from the current mark and index and applies it to every open position. Updates every hour.
3. **`RvolMargin`** — the position manager. Stores open positions with their entry strikes, applies the capped payoff of §9.2, handles liquidations, and interacts with the insurance fund.

Under HIP-3, all three live inside a single deployer-controlled contract bundle that presents as a custom perpetual market to the Hyperliquid matching engine. The matching engine handles orderbook, mark computation, and position accounting as it does for native HL perps; the custom modules override only the settlement index, the funding formula, and the margin rules specific to the variance payoff.

## 11.2 The oracle contract: state and updates

The oracle contract is the single most important module from a security perspective — every other guarantee in the paper depends on it producing correct $\mathrm{RV}_T$ values deterministically from on-chain inputs. Its state layout is tight:

```
contract RvolIndex:
    // Rolling windows
    int64[43200]  capped_sq_buffer   // ring buffer of (r_capped)^2, int64 fixed-point
    int64[1440]   abs_return_buffer  // ring buffer of |r_t|, for MAD computation
    uint32        buffer_head        // ring write pointer (minute-aligned)

    // Aggregates
    int256        running_sum_rv     // Σ capped_sq_buffer, kept incrementally
    int64         mad_sigma_cached   // 1-day rolling MAD × 1.4826

    // State
    uint64        last_update_ts     // unix seconds of last successful update
    uint32        valid_obs_count    // count of non-NaN entries in the RV buffer
    uint64        last_price         // last-trade price used in return computation
    bool          is_valid           // true iff valid_obs_count >= 38,880
    uint32        missed_updates     // for circuit breaker
```

Storage footprint: $43{,}200 \times 8$ bytes (RV buffer) $+\ 1{,}440 \times 8$ bytes (MAD buffer) $+\ O(64)$ bytes of scalar state $\approx$ **350 kB**. This is a single-contract state budget on any EVM-compatible chain and comfortably within Hyperliquid's per-contract storage limits.

### 11.2.1 Per-minute update

At each minute boundary the contract performs the following operations in order:

1. **Read price.** Look up the last-trade price of the host-venue BTC-USD perp. On Hyperliquid this is a precompile read from the matching-engine state, not an external call.
2. **Gap check.** If the wall-clock gap from `last_update_ts` exceeds 65 seconds, mark the new observation as NaN and skip to step 8 (validity update + circuit-breaker check). Otherwise continue.
3. **Compute return.** $r_t = \ln(P_t / P_{t-1})$ in fixed-point with 18 decimals of precision, implemented via a lookup-table-assisted natural log routine.
4. **Update MAD buffer.** Write $|r_t|$ into `abs_return_buffer[head % 1440]`. Recompute the MAD via an incremental median — either by maintaining a sorted structure (treap / sorted doubly linked list) that supports O(log n) insertion and deletion, or by periodically re-sorting the buffer when needed. The cached `mad_sigma_cached` is refreshed at every minute.
5. **Apply cap.** Compute $r_t^{\text{capped}} = \operatorname{sign}(r_t)\cdot\min(|r_t|,\ 4\cdot \text{mad\_sigma\_cached})$, then square.
6. **Roll RV buffer.** Subtract the evicted entry at `capped_sq_buffer[head % 43200]` from `running_sum_rv`; overwrite it with $(r_t^{\text{capped}})^2$; add the new entry to `running_sum_rv`.
7. **Increment counters.** `buffer_head += 1`; update `valid_obs_count` by $+1$ if writing a non-NaN and $-1$ if evicting a non-NaN; refresh `last_price` and `last_update_ts`.
8. **Validity update.** Set `is_valid = (valid_obs_count >= 38800)`. If the update was skipped (gap or price-read failure), increment `missed_updates`; otherwise reset it to zero.
9. **Circuit breaker.** If `missed_updates >= 5`, emit a `CircuitBreakerTripped` event; funding and new-position creation are paused until a governance reset.

The running sum is the only O(1) read used by every downstream module: any external caller querying the index simply reads `running_sum_rv` directly, with no iteration over the buffer. This is the single most important performance property of the oracle — it means every position mark-to-market, every liquidation check, and every funding computation can reference the current index in constant time with no gas amplification.

**Gas cost estimate.** Per-minute update is dominated by the incremental median in step 4. With a treap-backed sorted structure the amortized gas cost is approximately 50–80k gas per update, or roughly 72M–115M gas per day — one to two orders of magnitude below the per-day gas budget of a typical HL custom-market contract. Optimizations are possible (periodic sort instead of incremental, fixed-point approximations of MAD) but unnecessary.

### 11.2.2 Determinism and cross-implementation verification

The on-chain contract must produce exactly the same $\mathrm{RV}_T$ as the off-chain reference Python implementation (`rvol/index/variance.py`) given the same input price history. This is verified in two places:

- **Unit-level:** `tests/test_variance.py::test_rv_converges_to_true_variance_gbm` runs both the in-repo implementation and the `arch`-library realized variance estimator against the same GBM path and asserts they agree to machine precision.
- **Cross-implementation (planned):** A test harness replays historical Binance minute-returns through both the Python implementation and the Solidity contract (via Foundry's fork-mode) and asserts bit-identical output. This is part of the launch-gate checklist (§11.5).

Fixed-point arithmetic is the main source of potential divergence between Python (double precision IEEE-754) and Solidity (integer fixed-point with 18 decimals). We standardize on 18-decimal fixed-point throughout and use the same natural-log implementation in both — a lookup-table Taylor series with guaranteed 1-ulp error — to eliminate the floating-vs-fixed discrepancy.

## 11.3 Funding and margin modules

These are comparatively simple once the oracle is in place.

**`RvolFunding.accrueHour()`** reads the current index from `RvolIndex.running_sum_rv`, reads the current mark from the matching engine, computes $B_t = (M_t - I_t)/I_t$ and $f_t = \operatorname{clip}(B_t/k, -c, +c)$ with $k = 100$, $c = 0.001$, and applies the funding payment to every open position in proportion to its vega-notional. If `RvolIndex.is_valid` is false the function is a no-op.

**`RvolMargin.markToMarket(positionId)`** reads the current index, the position's stored `entry_index`, and computes the clipped payoff fraction of §9.2. It then updates the position's equity and checks the liquidation rule of §9.4. Positions are stored in a simple mapping keyed by a monotone position ID; the total number of open positions at steady state is bounded by the order-book-driven flow on the HL venue and is not expected to exceed a few thousand in the first year — well within sensible gas budgets for bulk mark-to-market.

**Insurance fund** is a separate contract with a single ERC-20 balance (USDC) that accepts penalty payments on liquidation and pays out residual losses when `RvolMargin` calls `drawShortfall(amount)`. Auto-deleverage is a fallback when `drawShortfall` returns insufficient funds; the ADL priority order (most-profitable-counterparty first) is standard and we inherit the HL convention.

## 11.4 Failure modes

| Failure | Detection | Response |
|---|---|---|
| No trades in a 1-minute bucket | `block.timestamp - last_update_ts > 65s` at update | Mark minute as NaN; exclude from RV sum; increment `missed_updates` |
| Observation count drops below 90% | `valid_obs_count < 38880` | `is_valid = false`; funding paused; new positions rejected |
| Oracle stalls (5+ missed updates in a row) | `missed_updates >= 5` | `CircuitBreakerTripped` event; funding frozen; governance reset required |
| Matching-engine mark feed unavailable | `RvolFunding` read fails | Funding hour skipped; positions mark to last valid index |
| Insurance fund exhausted | `drawShortfall` returns 0 | ADL triggered in reverse-profit order |
| Cross-implementation hash mismatch | Off-chain monitor comparing on-chain state to Python replay | On-chain: no action (the chain is authoritative); off-chain: alert governance immediately, assume a bug in one of the two implementations |

The key design principle: no failure mode should be able to produce an **incorrect** index value. All failures either pause the system (validity flag, circuit breaker) or fall back to a last-known-good state (mark to last valid index). The contract never attempts to repair state automatically — ambiguity is resolved conservatively in favor of stopping rather than guessing.

## 11.5 Launch sequence

Deployment proceeds in four gated phases, each with an explicit go/no-go criterion:

**Phase L0 — Testnet parallel-run (72 hours).**
The contract is deployed on HL testnet with a testnet BTC-USD feed. In parallel, an off-chain service runs the Python reference implementation against the same trade stream. At every minute boundary the off-chain service records the Python `RV_T` and compares it to the on-chain `running_sum_rv`. The launch gate is: **absolute deviation between the two implementations is less than 1 basis point of $\mathrm{RV}_T$ at every minute over the 72-hour window.** Any excursion above 1 bp blocks promotion to L1 and triggers a cross-implementation debug cycle.

**Phase L1 — Historical replay gate.**
Before mainnet, rerun the Phase 8.6 sweep (`scripts/analyze_margin_capped.py`) against the now-final contract parameters (the k, c, tier table, cap, and MM values actually baked into the deployed code). The gate: **maximum observed long and short liquidation rates across all 1,124 rolling 7-day windows must each be below 5%.** This catches any drift between the paper's reported parameters and the deployed parameters.

**Phase L2 — Mainnet soft launch (14 days).**
The contract goes live on HL mainnet with **50% of tier-maxima** position caps. The tier 1 cap is \$25,000 instead of \$50,000, tier 2 is \$125,000 instead of \$250,000, and so on. The 14-day window is used to observe: real basis behavior (is the simulated basis process realistic?), real funding accrual (does the VRP carry materialize as predicted?), and real liquidation events (are the rates within the historical-replay gate?). The gate to promote to L3: **observed 14-day liquidation rate below 2× the historical-replay gate**, and **no circuit-breaker events**.

**Phase L3 — Full launch.**
Tier maxima restored to their full values. Insurance fund replenishment targets the $5M goal within 90 days from this point. Ongoing monitoring includes a weekly recalibration review of the empirical distribution of 7-day excursions, with a discretionary recalibration trigger if observed rates exceed 2× the historical-replay gate over any rolling 90-day window.

Each phase has an explicit rollback path: between L1 and L2, the contract can be paused by governance with no open positions affected (none exist yet); between L2 and L3, the contract can be rolled back to position caps at 50% or lower without closing existing positions; in L3, parameter changes require at least a 14-day advance notice and a governance vote, following the HL norm for custom perpetual markets.

## 11.6 What makes this deployment novel

The on-chain realization completes the paper's central claim: **the entire settlement and margin system runs without any external oracle**. Every other on-chain volatility product in existence depends on at least one of the following: a Chainlink feed of DVOL or VIX-equivalent; a wrapped spot price from a centralized source; an external option-book price feed; or a cross-chain bridge reading another chain's state. The variance perp described here depends on none of these. Its inputs are (a) the host venue's own BTC-USD trade history, read through a precompile, and (b) the matching engine's own mark price, read through the same precompile. Both are state the chain already maintains for native HL purposes and that require no trusted third party to produce.

This is the direct generalization of the Panoptic and InfinityPools principle — that an on-chain derivative can be settled entirely from venue state — from spot-dependent products (options on current price) to path-dependent products (variance over a 30-day window). The generalization is non-trivial because a path-dependent product requires maintaining bounded O(1) state that evolves deterministically per observation over the window, which the ring-buffer construction of §11.2 provides. Nothing in this construction is specific to realized variance; any on-chain-computable statistic whose minute-level update can be expressed as a bounded state transition can be traded with the same machinery. Section 12 takes this observation seriously and sketches the broader product family it enables.
