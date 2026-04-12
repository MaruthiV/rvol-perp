# rvol-perp

An oracle-free perpetual futures contract on Bitcoin's realized variance, built for Hyperliquid.

## What is this?

Bitcoin's price swings wildly sometimes and barely moves other times. Traders, miners, and funds all care about how bumpy the ride is — but right now there's no clean way to bet on that bumpiness directly on-chain. You either have to mess around with options (complicated, expensive) or trust a centralized data feed (defeats the point of DeFi).

This project builds a new financial product that lets you trade Bitcoin's volatility as a simple perpetual contract. The key idea: instead of relying on an outside price feed (an "oracle"), the contract computes everything from the exchange's own trade history. No middleman, no external data to manipulate.

We track **realized variance** — basically, how much BTC's price actually moved over the last 30 days, measured every minute. Variance instead of volatility because the math works out cleaner: the payoff is linear, hedging is straightforward, and the margin system stays simple.

## What we built

**A Python library (`rvol/`)** that handles the core math:
- Computing rolling realized variance from 1-minute BTC price data
- A manipulation-resistant filter (4σ MAD cap) that ignores fake price spikes but lets real moves through
- Funding rate mechanics that keep the contract price anchored to reality
- Margin tiers calibrated against real historical data

**A data pipeline (`scripts/`)** that pulls years of BTC trading data from Binance, Hyperliquid, and Deribit, processes it into clean 1-minute returns, and builds the variance index.

**Simulations** — Monte Carlo and agent-based models to stress-test the design.

**An academic paper (`paper/`)** documenting the full design, calibration, and an important failure we found along the way.

**A deployment proposal (`docs/hip3/`)** for listing this as a real tradeable product on Hyperliquid via their HIP-3 protocol.

## The big finding

Our initial Monte Carlo simulations said traders could safely use 10x leverage on this contract. Then we ran the same design against six years of actual Bitcoin history — including COVID, LUNA, FTX, and every other crash — and found that **80% of positions at 10x would have been liquidated**.

The problem: variance is bounded at zero but has no ceiling. When volatility spikes (like March 2020, when BTC variance jumped roughly 14x), the short side of an uncapped contract gets destroyed. Standard stochastic volatility models miss these tail events entirely.

The fix comes from traditional finance — a payoff cap (the same one Goldman Sachs used for OTC variance swaps back in 1999). With a 2.5x cap applied, both sides of the contract survive historical stress tests, and the honest maximum leverage is **1.5x**. Not flashy, but actually safe. That collapse from 10x down to 1.5x is the most important number in the whole project.

## How manipulation resistance works

Since the index is computed from on-venue trades, someone could theoretically try to push fake prices into the index. We handle this with:

1. **A 4σ MAD cap** on each minute's return contribution — outlier moves get clipped using a robust statistical threshold that cheaters can't inflate
2. **30-day averaging** — any single minute is one of 43,200, so spiking one minute barely registers
3. **Minimum observation requirements** — if too much data is missing (exchange outage, etc.), the index pauses

Bottom line: moving the index by 1 volatility point costs an attacker at least $3.7 million in slippage. A 5-point move costs over $20M.

## Who would use this

- **Vol sellers** collecting the variance risk premium (implied vol tends to be higher than realized — that spread is the carry)
- **BTC miners** hedging against price instability (their business suffers when BTC swings hard)
- **Options desks** recycling vega exposure without the complexity of managing a full options book
- **Tail-risk funds** wanting a clean, linear payoff that scales with market chaos

## Quick start

```bash
# Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest

# Fetch data and build the index
python scripts/fetch_binance.py
python scripts/build_returns.py
python scripts/build_index.py
```

## Project structure

```
rvol/           Core library — index math, funding, margin, manipulation filters
scripts/        Data fetching, index building, analysis, figure generation
simulations/    Monte Carlo + agent-based models
paper/          Academic paper (sections, references, key numbers)
docs/hip3/      Hyperliquid HIP-3 deployment proposal
tests/          Test suite
```
