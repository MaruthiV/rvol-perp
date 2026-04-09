# 12. Applications and extensions

The contract described in sections 4–11 is a single instance of a more general construction. In this section we make that generalization explicit and sketch four worked examples: a gas-volatility perpetual, a correlation perpetual, a depeg-risk perpetual, and a self-referential pre-IPO-style instrument. The examples are not exhaustive — they are chosen to demonstrate that the same machinery (on-chain-computable index, capped payoff, empirically calibrated margin, oracle-free settlement) applies across qualitatively different statistics and asset classes.

## 12.1 The general construction

Let $\Phi$ be any statistic computable from the host chain's own state as a deterministic function of a rolling window of observations. That is:

$$ \Phi_T \;=\; f\bigl(X_{T-W},\, X_{T-W+1},\, \ldots,\, X_T\bigr), $$

where $X_t$ is an on-chain observation at time $t$, $W$ is the window length, and $f$ is implementable on-chain in bounded O(1) per-update state. Then, subject to four conditions, $\Phi$ can be the settlement index of a capped oracle-free perpetual.

**The four conditions.**

**C1. On-chain computability.** $f$ is expressible as an O(1) state transition: each new observation updates the running value of $\Phi_T$ by subtracting the evicted entry and adding the new one, or by an equivalent streaming update. This rules out statistics that require re-sorting the entire window each step (e.g. the sample median of the whole window, though rolling-median variants like MAD that can be maintained with a sorted auxiliary structure are fine).

**C2. Manipulation-resistant estimator.** Any single-observation excursion must be cap-able without destroying the estimator's signal in the common case. For realized variance this was the 4σ MAD cap; for other statistics it may be a different robust transform. The test is that under normal data the cap is almost never active, and under adversarial data it bounds the per-observation contribution to a known constant.

**C3. Economically meaningful payoff under a cap.** The capped payoff must still be linear in $\Phi_T$ over the band where the cap is inactive, and the cap itself must be chosen so that both sides of the book have finite maximum losses that can be covered by finite margin rates. For variance this was $c = 2.5$ inherited from Demeterfi et al. (1999); for other statistics the cap level is chosen empirically from the historical distribution of worst-case excursions, using the Phase 8.5/8.6 procedure of §10.

**C4. Natural counterparty flow.** There must be at least one structurally long and one structurally short population with economic reasons to take the opposing sides of the book — otherwise the contract has no sustainable volume regardless of the calibration. This is the softest condition and the one most easily waved away with hypothetical counterparties; the four examples below are chosen specifically because they have identifiable, unhypothetical flow on both sides.

The general construction then gives a product class: **an oracle-free capped perpetual on any $\Phi$ satisfying C1–C4**, with the same on-chain architecture (§11), the same funding mechanism (§7), and the same empirical-replay calibration procedure (§§9, 10). The realized variance perp is the prototype. The examples below are instantiations.

## 12.2 Example 1: Gas-volatility perpetual

**The statistic.** Let $G_t$ be the median gas price paid in block $t$ on the host chain. Define the 7-day rolling realized gas variance

$$ \Phi_T^{\text{gas}} \;=\; \sum_{t \in W_T} \bigl(\log G_t - \log G_{t-1}\bigr)^2, $$

with $W_T$ a trailing 7-day window of block timestamps (not minutes — the natural granularity is block-level). This is the direct gas-fee analogue of the BTC realized variance of §4, with returns computed on log gas price and the same MAD cap and validity machinery.

**Conditions.** C1: trivially satisfied — gas prices are first-class on-chain state and the rolling sum is an O(1) update per block. C2: the same 4σ MAD cap works; gas prices are fat-tailed but not more so than BTC returns. C3: gas prices cannot go below zero, so the same capped-variance-swap design applies with a per-position cap analogous to $c = 2.5$ (to be calibrated). C4: discussed below.

**Who takes each side.** The long side is populated by **anyone whose P&L is hurt by unexpected gas spikes**: automated market-makers rebalancing LP positions, arbitrageurs whose execution costs are dominated by gas, NFT minters running botted drops, and MEV searchers whose bids become uneconomic at high fees. The short side is populated by **anyone whose P&L benefits from persistent gas price elevation**: block builders, validators, and gas-token holders (the original use case of Chi/GST2-style gas tokens). Both populations exist at scale, and a venue to hedge gas volatility directly — rather than indirectly through an L2 bridge fee or a merged block auction — does not exist today.

**Why it needs oracle-free construction.** Gas prices are chain-native; they do not exist on any external oracle and cannot be faithfully produced by one. Chainlink or a similar oracle network could publish a gas-price feed, but it would be a feed **about** the chain the contract is deployed on, which is a strange category error — why would you pay for a trust-minimized feed of data the chain already has? The oracle-free construction is not just convenient here; it is the only sensible architecture.

**Deployment path.** A gas-vol perp on Ethereum mainnet has the largest addressable pool of natural counterparties. On L2s the absolute gas-variance numbers are smaller but the same construction applies — and because L2 gas fees are derived from the L1 base fee plus L2-specific components, a gas-vol perp on Arbitrum or Base is a natural tail-hedge instrument for anyone running a cross-layer strategy.

## 12.3 Example 2: Pair correlation perpetual

**The statistic.** For two on-chain-priced assets $A$ and $B$ (say BTC and ETH, both with deep perpetual markets on Hyperliquid), define the rolling 30-day realized correlation

$$ \rho_T^{AB} \;=\; \frac{\sum_{t\in W_T} r^A_t \cdot r^B_t}{\sqrt{\sum_{t\in W_T} (r^A_t)^2}\,\sqrt{\sum_{t\in W_T} (r^B_t)^2}}, $$

with $r^A_t$ and $r^B_t$ the 1-minute log returns on each asset. The index is bounded in $[-1, +1]$ and therefore has a naturally bounded payoff without requiring an explicit cap — condition C3 is satisfied automatically.

**Conditions.** C1: three running sums (two realized variances and one realized covariance), each O(1) per update — straightforward. C2: the MAD cap of §4.2 is applied per-asset before the sums are computed, and the bounded output of the correlation operator prevents adversarial inflation of the numerator. C3: the bounded $[-1, +1]$ range makes both sides have finite maximum loss equal to 2 in the worst case; no payoff cap is needed. C4: see below.

**Who takes each side.** The long side (betting correlation rises) is populated by **dispersion desks** running short-correlation structures on option volatilities; they hedge their correlation exposure by buying a realized-correlation claim. The short side is populated by **relative-value funds** running pairs trades on BTC and ETH; their positions lose when correlation rises (their long-short hedge breaks down) and they naturally want to hedge that risk with a short-correlation claim. Both populations are large and active in crypto; a pairs trader on BTC-ETH is the single most common systematic strategy in the space, and dispersion desks have grown dramatically with the build-out of crypto options markets.

**Why it's hard to do off-chain.** A correlation feed from Chainlink would require trusted computation of a sum-of-products over a 30-day minute-level window from two separate price sources. The attack surface is large: either source can be manipulated, the aggregator can be manipulated, and the aggregation logic itself becomes a trust root. The oracle-free construction collapses the trust surface to the chain itself, which is the minimum achievable.

**A subtle point.** Correlation is invariant under positive rescaling of either input, which means the MAD cap (which scales with $\hat\sigma_t$) does not change the correlation numerator in the same way it changes the variance denominator — the operator is more manipulation-resistant than variance, because a large move in one asset's return is partially self-cancelling in the correlation output. This is a desirable property and should make the manipulation-cost analysis of §8 even more favorable in the correlation case than in the variance case.

## 12.4 Example 3: Stablecoin depeg-risk perpetual

**The statistic.** For a stablecoin $S$ with nominal peg value \$1, define the 30-day rolling absolute-deviation variance

$$ \Phi_T^{\text{depeg}} \;=\; \sum_{t\in W_T} \bigl(\log P^S_t\bigr)^2, $$

where $P^S_t$ is the 1-minute last-trade price of $S$ in USDC (or another reference stable). This is a measure of how far $S$ has drifted from its peg over the window, integrated quadratically. The statistic is near-zero when $S$ holds its peg (log of a price very close to 1 is very small, squared is smaller still) and spikes sharply during depeg events.

**Conditions.** C1: O(1) update per minute, same as variance. C2: the 4σ MAD cap works but the MAD estimator is nearly degenerate in calm periods (log price is near zero, median of absolute values is near zero, so any small deviation looks like a 1000σ event); we modify the cap to use a floor MAD of $\sigma_{\text{floor}} = 10^{-4}$ (corresponding to a 1 bp per-minute baseline depeg volatility) below which the cap does not scale. This is the stablecoin-specific robustification of the general construction. C3: the integrand is bounded below at zero (it is a sum of squares) and unbounded above, so the same Demeterfi-style capped payoff applies with the cap calibrated to the historical worst-case depeg (e.g., the USDC-SVB 2023 event: peak excursion to \$0.88, or $\log 0.88 \approx -0.13$, squared is 0.017 per minute at the worst). C4: see below.

**Who takes each side.** The long side (betting depeg risk rises) is populated by **treasuries holding large stablecoin balances** (DeFi protocols, DAOs, exchanges) for whom a depeg event is a first-order P&L catastrophe. The short side is populated by **the stablecoin issuer itself**, and by **vol-sellers who believe the peg is safe**. The issuer is a natural short because selling depeg insurance generates carry that is income against the issuer's float, and because taking the other side of a market-implied depeg probability is a strong public signal of confidence in the peg. The vol-sellers are a natural short because historically, depeg events are rare and the carry is attractive — the same VRP argument as for variance.

**Why on-chain matters more here.** A stablecoin depeg perpetual that runs on an oracle fed by centralized spot data is itself a centralization vector: a successful attack on the oracle can trigger artificial liquidation cascades that could themselves cause a real depeg. The only safe place to run a depeg-insurance instrument is on the venue where the stablecoin itself trades, with the index computed directly from that venue's own order flow. This is the strongest version of the oracle-free argument in the paper: for depeg risk, oracle freedom is not an optimization, it is a prerequisite for the instrument not to create the very risk it is supposed to hedge.

## 12.5 Example 4: Self-referential pre-IPO instruments

The three examples above are all strict generalizations of the realized variance perp: they substitute a different statistic $\Phi$ but keep the same architecture. The fourth example is a structural departure — it uses the same oracle-free principle in a qualitatively different direction.

**Motivating example.** In the two years preceding this writing, informal prediction markets on private company equity — most prominently the various "OpenAI pre-IPO" instruments — have grown from novelties into non-trivial venues trading hundreds of millions of dollars of implied valuation. These instruments share a common problem: they reference the valuation of an off-chain private company, which has no tradeable spot price until the IPO event itself. The settlement mechanism of existing pre-IPO markets is therefore either (a) a trusted oracle that publishes a "reference valuation" at cadence, with all the attendant manipulation risk, or (b) a peer-prediction market where the price is whatever traders agree to trade at — which is self-referential but not rigorously so.

**The oracle-free construction for a self-referential instrument.** Let $M_t$ be the traded mark price of a perpetual on a host venue, and let the contract's "index" be defined not by any external reference but by the time-averaged mark price itself over a trailing window:

$$ I_T \;=\; \operatorname{clip-capped-average}\!\bigl(M_{T-W},\, M_{T-W+1},\, \ldots,\, M_T\bigr). $$

This is self-referential: the index is a function of the same traded prices the contract settles against. The funding mechanism is then $f_t = \operatorname{clip}(B_t/k, -c, +c)$ with $B_t = (M_t - I_T)/I_T$ as before, except that since the index is a lagged average of the mark, $B_t$ measures the deviation of the spot mark from its own trailing average. Funding pulls the mark toward its own trailing average, which sounds circular but is in fact a well-defined mean-reversion mechanism: the mark is free to move in response to news and trading, but the funding applies a restoring force against rapid excursions.

**Why this is interesting.** For a genuinely private asset (OpenAI stock, a company that has not yet priced its IPO, any valuation that does not have a public spot market), no external oracle can give a trust-minimized signal. The traditional answer is to run a peer-prediction market and hope the collective bets approximate truth. The self-referential funding mechanism above adds structure: it explicitly penalizes short-horizon excursions from the trailing average, which damps manipulation games while allowing genuine news to move the price over the funding time constant. It does not introduce any new trust root — the contract's inputs are entirely its own traded state.

**What it does not do.** This construction does not create information. It cannot price an instrument whose true value is genuinely unknowable; it can only create a tradeable perpetual whose price is the emergent consensus of traders, with a mean-reversion pull built into the funding. It is therefore a structural improvement over unanchored prediction markets, not a substitute for real price discovery. The intended comparison is not "oracle-based pre-IPO markets" — those do not work either — but "peer-prediction markets without any anchoring mechanism," which are worse than the self-referential alternative for exactly the reason that the self-referential version has a defined convergence behavior.

**Conditions C1–C4.** C1: trivially satisfied, the index is a rolling average of traded prices which is O(1) to maintain. C2: the cap is a floor-and-ceiling on per-minute mark moves that the running average is willing to absorb, bounding any single trader's ability to move the index. C3: self-evident if the running average is clipped. C4: the long and short sides of a pre-IPO instrument are the same as any other valuation perp — believers and skeptics of the target company's current implied worth. The variance-perp construction adapts cleanly.

**Caveat.** This example is the most speculative in the section and we present it explicitly as a design sketch, not as a calibrated instrument. It requires separate analysis of the manipulation-resistance properties (self-referential oracles are subject to attack patterns the variance-perp analysis of §8 does not cover), the convergence of the funding mechanism in a market with no external anchor, and the regulatory status of perpetuals on private-company valuations. We flag it to show that the general construction extends beyond straightforward generalizations of the variance perp into qualitatively new product classes, and to motivate follow-up work.

## 12.6 What unifies the four examples

Each example is a point in a three-dimensional space: (statistic $\Phi$, cap level $c$, host venue). The variance perp sits at (realized variance, 2.5, host venue's BTC perp). The gas-vol perp sits at (realized gas variance, TBD, host chain's block state). The correlation perp sits at (realized correlation, 1, host venue's BTC and ETH perps). The depeg perp sits at (squared log peg deviation, TBD, host venue's stablecoin pair). The pre-IPO instrument sits outside the first two dimensions — it uses a self-referential statistic and no external cap — but still inside the third.

This three-dimensional view suggests a simple product-design heuristic: pick an on-chain-computable statistic with natural long and short flow, inherit the variance-perp architecture, calibrate the cap empirically on the historical distribution of worst-case excursions, and deploy. The paper contributes the architecture, the calibration procedure, and the reference implementation for the variance case. The rest of the product family is left as a direct application.

We do not claim that each of these products is equally near-term. The gas-vol perp and the BTC-ETH correlation perp are the two most immediately deployable — both have clear counterparty populations, unambiguous statistics, and host chains (Ethereum mainnet and Hyperliquid respectively) on which the architecture of §11 is directly applicable. The depeg perp requires an additional regulatory analysis that we do not undertake here. The self-referential pre-IPO instrument is a design sketch. In all four cases, the contribution of this paper is showing that the same machinery works — and that the oracle-free property, far from being a niche feature of the variance case, is a prerequisite for an entire family of products that the existing on-chain derivatives stack cannot currently offer.
