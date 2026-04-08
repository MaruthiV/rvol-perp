# Phase 0 Design Doc: On-Chain Realized Variance Perpetual

## What we're building

A perpetual futures contract whose underlying index is the **realized variance** of BTC/USD price returns, computed entirely from on-chain price history. No external oracle. No centralized data source.

---

## Why variance, not volatility

The natural question is: why trade realized *variance* (squared returns) rather than realized *volatility* (their square root)?

**Variance is linear in the payoff.** A variance swap in TradFi has payoff:

```
PnL = N × (RV_realized − K_var)
```

where `N` is notional and `K_var` is the variance strike. This is a **linear** function of realized variance. Linearity means:

1. **Static replication**: A variance swap can be replicated by a static portfolio of vanilla options at all strikes (the log contract). This is the result of Carr & Lee (2003). It gives variance perps a theoretical foundation that vol perps lack.
2. **Clean funding mechanism**: The funding rate `(index − mark)` is also linear, which makes mark-to-index convergence a tractable no-arbitrage argument.
3. **No convexity adjustment needed**: A vol perp would have `payoff ∝ √RV`, which introduces a convexity term. Arbitrageurs would need to hedge that convexity dynamically, increasing friction.

The tradeoff: variance is less intuitive than volatility to most traders. We address this by displaying annualized vol-equivalent values in the UI (`annualized_vol = √(RV × 365)`), while the contract settlement is in variance units.

---

## Payoff structure

```
Mark-to-market PnL for long position over interval [t, t+1]:
  ΔPnL = notional × (RV_index[t+1] − RV_index[t]) − funding_paid[t]

Funding:
  funding_rate[t] = (mark_price[t] − index[t]) / index_scaling_factor
  funding_paid[t] = position_size × funding_rate[t]   (long pays, short receives if mark > index)

Settlement:
  Continuous (perpetual — no expiry). Mark-to-index convergence enforced by funding.
```

The index is the 30-day rolling realized variance:

```
RV_T = Σ_{i ∈ window_T} r_i²    (post manipulation-cap)
```

where `r_i = ln(P_i / P_{i-1})` at 1-minute intervals.

---

## Why oracle-free

The settlement index is computed from prices that already exist on-chain (Hyperliquid's own price feed). No external oracle is needed because:

1. Hyperliquid records every trade on-chain with timestamps.
2. Realized variance is a deterministic function of those on-chain prices.
3. An on-chain contract can verify any RV computation by replaying the price history.

This is the key design thesis borrowed from Panoptic (oracle-free options) and InfinityPools — but applied to a volatility product for the first time.

**Manipulation resistance** is the critical challenge. We address it with:
- Return-contribution caps (4σ MAD-based — see `01-index-spec.md`)
- TWAP mark price (prevents funding-time sniping)
- Minimum observation count (≥90% of expected 1-minute slots)

---

## Comparison to existing products

| Product | Underlying | Oracle | Settlement |
|---|---|---|---|
| VIX futures (TradFi) | 30-day *implied* variance | CBOE computation | Cash, monthly expiry |
| Volmex BVIV | BTC implied vol index | Centralized (Deribit) | Not directly tradeable as perp |
| Opyn Squeeth | ETH² (power perp) | Chainlink | Continuous (perp) |
| **This product** | BTC 30-day realized variance | On-chain price history | Continuous (perp), no expiry |

The key differentiators: (1) realized vs implied — we measure what actually happened, not what the market expects; (2) oracle-free — the index is self-contained.

---

## Target users

**Natural longs** (want to buy realized vol exposure):
- Options dealers hedging their vega books — they are naturally short realized vol via their options inventory
- Funds running tail-risk strategies — long RV as a crisis hedge

**Natural shorts** (want to sell realized vol exposure):
- BTC miners — their business is structurally short BTC vol (they benefit from stable prices)
- Structured product issuers — earn carry from the variance risk premium
- Systematic vol sellers — the VRP has historically been positive in crypto (implied > realized on average)

**Arbitrageurs** (what makes the market efficient):
- Options market makers who can trade Deribit options to arbitrage RV perp vs options-implied variance
- Statistical arb funds tracking basis convergence

---

## Deployment target: Hyperliquid HIP-3

Hyperliquid's HIP-3 protocol allows permissionless listing of custom perpetual markets. It provides:
- Deep existing liquidity infrastructure
- Hourly funding rate mechanism (matches our design)
- On-chain settlement
- An existing BTC price feed we can use as input to the RV computation

The HIP-3 deployment proposal (Phase 9) will specify the oracle source, funding parameters, margin schedule, and listing config.

---

## Open questions to resolve before Phase 1

1. **Return interval**: 1-minute chosen for robustness. Sub-minute introduces bid-ask bounce. Confirm with data in Phase 2.
2. **Annualization convention**: Use 365 calendar days (crypto convention, not 252 trading days). Lock this in `01-index-spec.md`.
3. **Notional denomination**: USDC-denominated. 1 contract = $1 notional exposure to realized variance. Scaling factor TBD in Phase 4.
4. **Funding frequency**: 1-hour, matching Hyperliquid's standard. Validate in Phase 4 simulation.
