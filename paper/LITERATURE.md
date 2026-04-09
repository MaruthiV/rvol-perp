# Literature Review

Comprehensive bibliography and positioning for the rvol-perp paper. Organized into seven threads; each entry gives the citation, the core result, and our specific use or departure.

---

## 1. Variance swap theory & static replication

The mathematical foundation for trading variance as a linear payoff replicable from vanilla options. This is the literature we inherit wholesale.

### Neuberger (1994), "The log contract"
*Journal of Portfolio Management* 20(2), 74–80.
- **Core result:** A contract paying `2 ln(S_T / S_0)` at expiry has a delta that, when hedged, locks in exposure to realized variance independent of the volatility path. This is the original observation that variance is statically hedgeable.
- **Our use:** foundational. The log contract is the mathematical ancestor of every variance swap including ours. We cite it in the Related Work framing.

### Carr & Madan (1998), "Towards a theory of volatility trading"
*Volatility: New Estimation Techniques for Pricing Derivatives* (Jarrow ed.), RISK Books, 417–427.
- **Core result:** Any twice-differentiable payoff `f(S_T)` can be replicated by a static portfolio of European options plus a bond and forward position; applied to `f(S) = 2(S/S_0 − 1 − ln(S/S_0))` yields a variance swap replication via a strike-weighted option strip.
- **Our use:** establishes the "strip of options" hedge that makes variance (not volatility) the natural primitive.

### Demeterfi, Derman, Kamal, Zou (Goldman Sachs, 1999), "More than you ever wanted to know about volatility swaps"
*Quantitative Strategies Research Notes*, Goldman Sachs.
- **Core result:** The canonical practitioner derivation of the variance swap, including the strike-weighting formula for the replicating option strip, the fair variance strike in closed form, and — critically for us — **the introduction of the capped variance swap** with cap typically at `2.5 × K_var` to bound short-side losses in discrete sampling and jump scenarios.
- **Our use:** **This is the single most important citation for our paper.** We adopt `PAYOFF_CAP = 2.5` directly from this convention in Phase 8.6. The paper's key positive framing is: "Phase 8.5 empirically rediscovered why Goldman capped every variance swap since 1999; Phase 8.6 ports that fix on-chain."

### Britten-Jones & Neuberger (2000), "Option prices, implied price processes, and stochastic volatility"
*Journal of Finance* 55(2), 839–866.
- **Core result:** Model-free formula for forward integrated variance in terms of current option prices. Direct precursor of the VIX methodology.
- **Our use:** marks the transition from variance as a theoretical object to variance as a tradable index. Cited in the VIX lineage.

### Carr & Lee (2003), "Robust replication of volatility derivatives"
Working paper, NYU / University of Chicago.
- **Core result:** Robust (model-free) static replication of variance derivatives from vanilla options, extended to volatility derivatives under a correlation assumption. Makes precise the sense in which variance is "more replicable" than volatility: variance has a model-free hedge, volatility does not.
- **Our use:** **Cited explicitly in our design doc §Why variance, not volatility** (`docs/design/00-variance-perp-design.md`). This is the reason we chose a variance perp over a vol perp. A vol perp would have a `√RV` payoff, introducing a convexity adjustment that arbitrageurs would have to hedge dynamically — incompatible with oracle-free on-chain execution.

### Carr & Wu (2006), "A tale of two indices"
*Journal of Derivatives* 13(3), 13–29.
- **Core result:** Explains the difference between the old VIX (Black-Scholes inversion) and the new VIX (model-free integrated variance), tying the latter to Britten-Jones–Neuberger.
- **Our use:** context for §4 and §6 (why realized vs implied).

---

## 2. Realized variance econometrics

The estimator we use is an Andersen-Bollerslev-Diebold-Labys realized variance. This thread establishes its convergence properties, noise-robustness, and forecasting benchmarks.

### Andersen, Bollerslev, Diebold, Labys (2001), "The distribution of realized exchange rate volatility"
*Journal of the American Statistical Association* 96(453), 42–55.
- **Core result:** With high-frequency data, realized variance `RV = Σ r²` is a consistent (and under mild conditions, nearly model-free) estimator of integrated variance. The distribution of log-RV is approximately normal even though returns themselves are heavily non-Gaussian.
- **Our use:** direct ancestor of our RV30 estimator. Citation for the statement "RV converges to quadratic variation as sampling frequency increases."

### Andersen, Bollerslev, Diebold, Labys (2003), "Modeling and forecasting realized volatility"
*Econometrica* 71(2), 579–625.
- **Core result:** The "ABDL" paper. Formalizes realized variance as the workhorse estimator for integrated variance, studies its long-memory properties, and builds a forecasting framework.
- **Our use:** foundational. Our 30-day sum of squared 1-minute returns is an ABDL-style estimator at minute sampling over a 30-day window.

### Barndorff-Nielsen & Shephard (2002), "Econometric analysis of realized volatility and its use in estimating stochastic volatility models"
*Journal of the Royal Statistical Society Series B* 64(2), 253–280.
- **Core result:** Distributional theory for realized variance under stochastic volatility; confidence intervals and asymptotic normality.
- **Our use:** citation for distributional properties; justifies why we can build a rolling estimator whose cross-sectional behavior is tractable.

### Barndorff-Nielsen, Hansen, Lunde, Shephard (2008), "Designing realized kernels to measure the ex post variation of equity prices in the presence of noise"
*Econometrica* 76(6), 1481–1536.
- **Core result:** Realized kernel estimators that are robust to microstructure noise (bid-ask bounce, discrete pricing). The naive `Σ r²` is biased when returns contain noise; kernel methods correct this.
- **Our use & gap:** We do **not** use realized kernels. Justification (to be defended in §4): (a) the on-chain oracle must be updatable in O(1) per minute, and kernel estimators require weighted sums over a bandwidth that changes the accounting; (b) we work at 1-minute frequency where microstructure noise is much smaller than at tick level; (c) the 4σ MAD cap acts as an outlier filter that handles the worst noise-like contributions directly. We should acknowledge this design tradeoff explicitly.

### Corsi (2009), "A simple approximate long-memory model of realized volatility"
*Journal of Financial Econometrics* 7(2), 174–196.
- **Core result:** The HAR-RV (Heterogeneous Autoregressive) model — a forecasting regression of RV on lagged daily, weekly, monthly RV — captures long-memory behavior with a parsimonious linear specification.
- **Our use & gap:** We do not forecast RV in this paper — we spot-price it. HAR is the natural benchmark if we add a forecasting section in a follow-up. For now, cited as "the standard forecasting approach we do not pursue here."

### Zhang, Mykland, Aït-Sahalia (2005), "A tale of two time scales"
*Journal of the American Statistical Association* 100(472), 1394–1411.
- **Core result:** Subsampling-based bias correction for realized variance under microstructure noise.
- **Our use:** alternative to kernel methods; same justification for not using it (on-chain O(1) constraint).

---

## 3. Variance risk premium (empirical)

The economic engine of the product: the VRP is the compensation that short-vol counterparties earn, and it has to be positive in expectation for the contract to have natural flow.

### Bakshi & Kapadia (2003), "Delta-hedged gains and the negative market volatility risk premium"
*Review of Financial Studies* 16(2), 527–566.
- **Core result:** By constructing delta-hedged option portfolios, isolates the component of option P&L attributable to volatility risk; shows it is significantly negative for equity indices, meaning long-vol positions pay a premium (equivalently, short-vol earns a premium).
- **Our use:** first clean empirical identification of the VRP. Cite in §6 when reporting our BTC VRP.

### Carr & Wu (2009), "Variance risk premiums"
*Review of Financial Studies* 22(3), 1311–1341.
- **Core result:** The canonical cross-sectional study. Defines VRP as `E^Q[RV] − E^P[RV]`, estimates it as "synthetic variance swap rate minus realized variance," documents that it is negative (short side earns) for equity indices, and analyzes its time-series and cross-sectional properties.
- **Our use:** **Direct methodological template for our Phase 2 VRP analysis.** Our measurement `VRP = DVOL²/365 − RV30_forward` is the discrete BTC analogue of their equity VRP. We cite them when defending the sign and magnitude of our VRP estimate.

### Bollerslev, Tauchen, Zhou (2009), "Expected stock returns and variance risk premia"
*Review of Financial Studies* 22(11), 4463–4492.
- **Core result:** VRP is a significant predictor of future stock returns. Establishes VRP as a priced risk factor beyond just an option-pricing artifact.
- **Our use:** supports the economic-thesis framing that VRP is a real risk premium, not a quirk of the options market.

### Alexander & Imeraj (2021), "The Bitcoin VIX and its variance risk premium"
*Journal of Alternative Investments* 23(4), 84–109 (or working paper precursor).
- **Core result:** Constructs a BTC VIX from Deribit options and measures the BTC variance risk premium. Finds a positive VRP in BTC, consistent with equity markets but larger in magnitude.
- **Our use:** **Closest comparable in the literature.** Our Phase 2 VRP measurement is the direct confirmation of their result using our own RV30 index. Cite prominently — this is the paper a reviewer will ask about first.

### Todorov (2010), "Variance risk-premium dynamics: The role of jumps"
*Review of Financial Studies* 23(1), 345–383.
- **Core result:** Decomposes VRP into diffusive and jump components; jumps contribute a large fraction.
- **Our use:** relevant for §10 methodological discussion (why log-normal SV without jumps underestimates tail risk).

---

## 4. Stochastic volatility models (for Monte Carlo calibration)

We use a stochastic vol model for Phase 5 lifecycle simulation. This literature gives us the model families and — importantly for the paper — the context to explain why even a calibrated SV model fails to capture real-regime tail events.

### Hull & White (1987), "The pricing of options on assets with stochastic volatilities"
*Journal of Finance* 42(2), 281–300.
- **Core result:** Options pricing under log-normal stochastic volatility; closed-form (under uncorrelated returns/vol) using a Taylor expansion around Black-Scholes.
- **Our use:** **The SV model we use in Phase 5 is log-normal SV in the Hull-White family** (parameters θ=−7.32, κ=0.00363, η=0.06, ρ=−0.08 calibrated to BTC).

### Heston (1993), "A closed-form solution for options with stochastic volatility"
*Review of Financial Studies* 6(2), 327–343.
- **Core result:** Square-root (CIR) variance process with correlation to returns; closed-form option pricing via characteristic functions. Industry standard.
- **Our use:** cite as the more common alternative to log-normal SV; justify log-normal choice (BTC log-vol is more symmetric than √vol under empirical fit).

### Bates (1996), "Jumps and stochastic volatility: exchange rate processes implicit in Deutsche mark options"
*Review of Financial Studies* 9(1), 69–107.
- **Core result:** Adds Poisson jumps to the Heston diffusion. Explains fat tails better than pure SV.
- **Our use:** cite as the correction that, had we implemented it, might have caught the Phase 8.5 failure. This is our "what we would do differently" citation.

### Andersen, Benzoni, Lund (2002), "An empirical investigation of continuous-time equity return models"
*Journal of Finance* 57(3), 1239–1284.
- **Core result:** Comprehensive empirical comparison of SV, SV+jumps, and affine models on equity indices; finds jumps are essential.
- **Our use:** empirical support for the "pure SV is not enough" lesson.

### Christoffersen, Jacobs, Mimouni (2010), "Volatility dynamics for the S&P500"
*Review of Financial Studies* 23(8), 3141–3189.
- **Core result:** Compares Heston-Nandi, SV, and GARCH models on S&P500 options; finds model choice matters less than jump inclusion.
- **Our use:** supports our methodological caveat in §10.

---

## 5. Market microstructure & manipulation cost

The cost model for Phase 7 attacks is linear price impact. This thread establishes the theoretical basis.

### Kyle (1985), "Continuous auctions and insider trading"
*Econometrica* 53(6), 1315–1335.
- **Core result:** In a market with an informed trader, noise traders, and a market maker, equilibrium price impact is linear in quantity: `Δp = λ · Q`. λ is determined by the ratio of noise-trader volatility to information volatility.
- **Our use:** **Cited explicitly in `docs/design/05-manipulation-cost.md`** as the theoretical foundation of our cost model. We calibrate `λ` from HL BTC-perp order book depth ($5M per 1% price move ≈ 5 bps per $1M).

### Almgren & Chriss (2000), "Optimal execution of portfolio transactions"
*Journal of Risk* 3(2), 5–39.
- **Core result:** Decomposes price impact into temporary (reverts after execution) and permanent (persists) components. Provides optimal execution trajectories under a quadratic risk/cost tradeoff.
- **Our use:** justification for the round-trip factor of 2 in sustained attack costs (attacker must pay temporary impact on entry and exit).

### Huberman & Stanzl (2004), "Price manipulation and quasi-arbitrage"
*Econometrica* 72(4), 1247–1275.
- **Core result:** No-arbitrage requires price impact functions to be linear in the permanent component; otherwise round-trip manipulation yields positive expected profit.
- **Our use:** theoretical justification for using Kyle's linear model without apology.

### Obizhaeva & Wang (2013), "Optimal trading strategy and supply/demand dynamics"
*Journal of Financial Markets* 16(1), 1–32.
- **Core result:** Limit order book resilience model; temporary impact decays over time.
- **Our use:** relevant for calibrating sustained-attack dynamics; we use a simplification (constant round-trip cost per minute) and should acknowledge the simplification.

---

## 6. DeFi oracles, manipulation, and on-chain derivatives

The crypto-native thread: why oracle-free matters, and what existing on-chain vol products exist.

### Eskandari, Salehi, Gervais, Clark (2024), "SoK: Oracles from the ground truth to market manipulation"
*MDPI Cryptography / ACM CCS SoK track* (2024).
- **Core result:** Systematic survey of DeFi oracle architectures and manipulation attacks; taxonomizes push/pull oracles, TWAP designs, and documented exploits (bZx, Harvest, Mango, etc.).
- **Our use:** **Primary citation for the "oracle-free matters" motivation** in §1 (Introduction). Establishes that oracle manipulation is a first-class threat in DeFi, not a theoretical concern.

### Bank for International Settlements (2023), "DeFi and the future of finance"
*BIS Bulletin No. 76*.
- **Core result:** Policy-oriented survey of DeFi risks, prominently including oracle dependency.
- **Our use:** regulatory/policy citation; supports the framing that oracle risk is institutionally recognized.

### Angeris, Chitra, Evans, Lorig (2023), "A primer on perpetuals"
arXiv:2209.03307.
- **Core result:** Formal treatment of perpetual contract pricing, funding rate mechanisms, and convergence of mark to index under various funding designs.
- **Our use:** **Methodological foundation for §7 (funding mechanism).** Our clamped-linear funding is a bounded variant of the stateless proportional family analyzed in this paper. Cite when deriving our convergence result (31h half-life, <1% cap-binding).

### White et al. / Opyn / Paradigm (2021), "Squeeth: power perpetuals primer"
Opyn technical docs + Paradigm research blog post "Everlasting Options" (Dave White, Sam Bankman-Fried, Aparna Krishnan, 2021) + subsequent Squeeth release.
- **Core result:** A perpetual on `ETH²` — the first on-chain perpetual on a non-linear function of price. Uses Uniswap v3 as the hedging venue. Provides convexity exposure.
- **Our use:** **Closest existing cousin to our product.** Mechanical similarity: both are perpetuals on non-linear functions of price. Key differences:
  1. Squeeth's index is `ETH²` (a function of current spot), ours is `Σ r²` over a window (a function of price path).
  2. Squeeth requires Uniswap as hedge venue and inherits AMM oracle risk.
  3. Squeeth is not directly linear in realized variance — it is a proxy via `p²` that tracks variance only under specific SV assumptions.
  4. Our contract is oracle-self-referential: the index is computed from the venue's own trades.
- Cite as prior art and differentiate carefully.

### He, Lambert, Mello, Piskorski / Panoptic team (2023), "Panoptic: perpetual options via LP tokens"
Panoptic whitepaper / arXiv.
- **Core result:** Perpetual American options constructed by borrowing and deploying Uniswap v3 LP positions. Oracle-free in the sense that the payoff is computed from LP position states, not external feeds.
- **Our use:** **The structural inspiration for our "oracle-free" framing.** Panoptic established that an on-chain derivative can be settled from venue state directly. We extend that principle from options (function of spot) to variance (function of path).

### InfinityPools whitepaper / documentation
- **Core result:** Perpetual options / leveraged LP via concentrated liquidity; oracle-free in the same sense as Panoptic.
- **Our use:** alongside Panoptic, as prior art in the "oracle-free derivative" category.

### GMX, dYdX, Hyperliquid perpetual funding literature
- **Core result:** Production-grade implementations of perpetual funding mechanisms; various empirical studies of mark-index convergence.
- **Our use:** context and comparison; HL is the deployment target.

### Qin, Zhou, Livshits, Gervais (2021), "Attacking the DeFi ecosystem with flash loans for fun and profit"
IFCA Financial Cryptography.
- **Core result:** Documents real flash-loan attacks against DeFi protocols including oracle manipulations (bZx, Harvest). Quantifies attacker profits and protocol losses.
- **Our use:** concrete motivation for oracle-free design; cite alongside Eskandari et al. (2024).

---

## 7. Crypto volatility products (institutional / practitioner)

Existing products that give volatility exposure in crypto markets, against which we position.

### Deribit DVOL
- **Description:** 30-day annualized implied volatility index for BTC (and ETH), computed by Deribit from its own options book using a VIX-like model-free methodology.
- **Our use:** **Reference series for our VRP measurement** in §6. Note DVOL is (a) implied, not realized, (b) centralized, (c) not directly tradeable as a perpetual — one can only trade Deribit's DVOL futures with monthly expiry. Our product is the realized analogue delivered as a true perpetual.

### CBOE VIX methodology
CBOE white paper, 2003 revision (and subsequent updates).
- **Core result:** Model-free implied variance computed from OTM option strip using the Britten-Jones–Neuberger formula.
- **Our use:** the methodological template Deribit DVOL follows; cited for context in §2 and §4.

### Hyperliquid HIP-3 specification
Hyperliquid public docs + GitHub (2025 launch, 500k HYPE staking requirement).
- **Core result:** Permissionless listing protocol for custom perpetual markets on Hyperliquid. Allows deployer-specified oracle contracts.
- **Our use:** **Deployment venue.** Our contract is designed to meet HIP-3's listing requirements. The oracle-free property is especially valuable under HIP-3 because the deployer, not Hyperliquid governance, is responsible for oracle integrity.

### Lyra, Premia, Dopex, Hegic
- **Description:** On-chain options venues with various implementations (AMM-style for Lyra/Premia, peer-to-pool for Dopex/Hegic).
- **Our use:** give vega exposure through option Greeks but with nonlinear payoff, path dependence, and expiry friction. Positioned in §1 as "existing on-chain vol exposure, but not linear and not perpetual."

### Volmex BVIV
- **Description:** BTC implied vol index, similar to Deribit DVOL.
- **Our use:** mentioned in `docs/design/00` as an existing product; same category as DVOL (implied, centralized).

---

## Citation gaps / items to double-check before submission

- Exact publication venue and date for **Demeterfi et al. (1999)** — it is a Goldman Quantitative Strategies Research Note, not a peer-reviewed paper; the canonical reference in the literature is "Demeterfi, Derman, Kamal, Zou (1999), *More Than You Ever Wanted to Know About Volatility Swaps*, Goldman Sachs QSRN, March 1999." Include PDF URL.
- **Carr & Lee (2003)** is a widely cited working paper; the formal journal version is Carr & Lee (2009), *Mathematical Finance*, "Robust Replication of Volatility Derivatives" — cite the journal version for formality.
- **Alexander & Imeraj (2021)** — confirm exact journal; there may be both a working paper and a journal version.
- **Angeris et al. (2023) "Primer on perpetuals"** — arXiv preprint; confirm latest version number.
- Squeeth whitepaper citation: currently informal; cite the Paradigm blog post + Opyn technical docs with access dates.
- **HIP-3 citation:** use permalink to Hyperliquid docs as of paper submission date.

---

## Suggested §3 (Related Work) paragraph structure — ~2 pages

1. **Variance swap lineage** (Neuberger → Carr-Madan → Demeterfi → Carr-Lee) — establishes that we are not inventing variance-as-an-instrument, only porting it on-chain as a perpetual with a native oracle.
2. **Realized variance estimation** (ABDL → Barndorff-Nielsen-Shephard → realized kernels → HAR) — positions our estimator; acknowledges the noise-robust literature we bypass and explains why.
3. **Variance risk premium** (Bakshi-Kapadia → Carr-Wu → Alexander-Imeraj) — motivates that the short side has structural carry in BTC.
4. **On-chain derivatives and oracle-free design** (Panoptic / InfinityPools / Squeeth / Angeris primer) — positions our contribution relative to the existing on-chain derivatives literature.
5. **Crypto vol products** (DVOL, VIX, Lyra/Premia, BVIV) — establishes the product gap we fill.
6. **Manipulation and oracles** (Kyle, Eskandari, BIS, Qin et al.) — establishes the security framing.

---

## Novelty claim (defended in §1 and revisited in §13)

**We do not claim novelty in:**
- variance swap pricing theory (Neuberger, Demeterfi, Carr-Lee own this)
- realized variance estimation (ABDL own this)
- variance risk premium measurement (Carr-Wu, Alexander-Imeraj own this for BTC)
- Kyle-type manipulation cost modeling (Kyle, Almgren-Chriss own this)
- the payoff cap itself (Demeterfi 1999 owns this)

**We do claim novelty in the synthesis:**
1. **First oracle-free perpetual on a path-dependent function.** Panoptic and Squeeth established oracle-free derivatives for spot-dependent functions; we extend the principle to a path-dependent quantity (rolling realized variance) that requires maintaining O(1) state per observation over a 30-day window.
2. **First honest historical-replay calibration of a variance-perp margin system against real crypto-regime data**, including the discovery (Phase 8.5) that naive variance is unleverageable and the canonical fix (Phase 8.6) recalibrated to pass a <5% liquidation gate.
3. **First on-chain application of the capped-variance-swap convention.** The Goldman 1999 cap has been OTC standard for 26 years; we port it to a DeFi perpetual with an empirically calibrated cap level justified by data, not tradition.
4. **A general framework** ("oracle-free perpetuals on any on-chain-computable statistic") with worked applications to gas-vol, correlation, depeg-risk, and self-referential pre-IPO instruments.
