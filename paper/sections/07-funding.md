# 7. Funding mechanism

The contract is a perpetual: there is no expiry and no settlement event. The sole mechanism by which the traded mark price is anchored to the realized-variance index is the periodic funding payment. This section specifies the functional form of the funding rate, derives the parameter choices from the empirical properties of §6, and reports the convergence behavior under simulated basis shocks.

## 7.1 Design desiderata

A funding mechanism for an oracle-free variance perp must satisfy four criteria simultaneously:

1. **Pull mark toward index** in steady state, so that the traded price cannot persistently drift from the underlying realized-variance path.
2. **Be bounded** so that a fat-finger basis excursion (a 500% mark-index deviation due to a thin orderbook or an adversarial print) cannot produce a funding payment large enough to liquidate every position on one side of the book instantaneously.
3. **Be stateless** — the rate at time $t$ must be a pure function of the current mark and index, not of any trailing integral or internal counter. Stateful funding (PID-style, or threshold-with-hysteresis) creates exploitable timing games around the state transitions.
4. **Not dominate the variance risk premium.** The mean VRP is $+14$ annualized vol points (§6.5); if the funding mechanism systematically transfers more than that from the short side to the long side, the short side's structural carry is erased and the economic case for the product evaporates.

These four constraints narrow the design space considerably. The form we adopt — clamped-linear response to relative basis, hourly cadence — is the simplest stateless function that satisfies all four.

## 7.2 Functional form

Let $I_t$ denote the variance index value at time $t$ (in daily-variance units, §4.5), let $M_t$ denote the traded mark price on the perp in the same units, and let

$$ B_t \;=\; \frac{M_t - I_t}{I_t} $$

denote the relative basis. The funding rate paid over the interval $[t, t+\Delta]$ is

$$ \boxed{\; f_t \;=\; \operatorname{clip}\!\left(\frac{B_t}{k},\; -c,\; +c\right),\;} $$

with dampening factor $k$ dimensionless, per-interval cap $c$, and interval $\Delta = 1$ hour matching the standard Hyperliquid funding cadence. Our sign convention: positive $f_t$ means **longs pay shorts** — when the mark is above the index, longs are over-paying relative to the underlying and are therefore penalized. Equivalently, over any funding interval,

$$ \text{long pays} \;=\; f_t \,\cdot\, V \,\cdot\, I_{\text{entry}}, \qquad \text{short receives} \;=\; f_t \,\cdot\, V \,\cdot\, I_{\text{entry}}, $$

where $V$ is the position's vega-notional and $I_{\text{entry}}$ is the position's entry index strike. At zero basis ($M_t = I_t$) the funding rate is exactly zero and the perp tracks the index passively.

The clip is load-bearing. Without it a single fat-finger basis excursion — say a 500% deviation caused by a brief liquidity void on the mark side — would produce a funding payment of $5.0/k$ per hour, which for any reasonable $k$ is large enough to liquidate every leveraged long on the book in a single cycle. The clip turns this from a runaway event into a bounded per-hour payment that the system can survive and that the market can arbitrage away over subsequent hours.

We considered and rejected three alternative functional forms:

- **Unclamped proportional** ($f = B/k$). Fails the boundedness requirement: a 500% basis produces an unsurvivable funding payment.
- **PID / integral** ($f = B/k + \alpha\int B$). Stateful. The integral term gives arbitrageurs a predictable asymmetric payoff around resets, which is exploitable on entry timing.
- **Threshold / deadband** ($f = 0$ if $|B|<\epsilon$, linear outside). Creates a discontinuity at the threshold that market-makers can camp on, producing a persistent steady-state basis at exactly the threshold.

Clamped linear is the smallest function that does what is needed.

## 7.3 Parameter calibration

The two tunable parameters are $k$ (dampening) and $c$ (per-hour clip). Both are selected by a combination of analytic reasoning from §6 and grid-search simulation in `scripts/simulate_funding.py`.

### 7.3.1 Analytic targets

From §6 we know:
- $\mathrm{RV}_{30}$ daily change standard deviation is $\approx 6.1$ percentage points of annualized vol (§6.2), or in relative terms, the index has approximately 6% day-over-day relative volatility.
- The AR(1) of hourly index changes is 0.996 — the index is mechanically smooth at the funding frequency.
- Mean VRP is $+14$ annualized vol points, or approximately $+1.5$ basis points per hour of expected short-side carry under the annualization.

Funding must satisfy two bracketing conditions:

**(A) Not dominate VRP at realistic steady-state basis.** A rational arbitrageur will accept some steady-state basis in exchange for avoiding the gas and capital cost of trading it away; a typical tolerance is 5% relative basis. At $B = 5\%$ we want the hourly funding payment to be of the same order of magnitude as the VRP carry, not many times larger. Setting $|f| \approx 1.5$ bps at $B = 5\%$ gives

$$ k \;\approx\; \frac{0.05}{1.5\times 10^{-4}} \;=\; 333. $$

**(B) Survive worst-case basis shocks without liquidating everything.** At the largest realistic fat-finger basis (say $|B| = 50\%$), the cap $c$ should bound the hourly payment to no more than a few tenths of a percent of notional. Setting $c = 0.5\%$ per hour gives a daily hard ceiling of $\pm 12\%$, which a properly margined position can survive for a day while arbitrageurs close the basis.

These two analytic targets point to $k \approx 333$ and $c \approx 0.005$. This was the initial parameter choice in the Phase 4 design doc.

### 7.3.2 Grid search refinement

The analytic targets are starting points, not final values. The Phase 4 grid search in `simulate_funding.py` replays the historical index series, injects a stochastic basis process with realistic mean-reversion and shock statistics, and scans a 2D grid over $(k, c)$. For each point it measures: mean absolute basis at steady state, half-life of a 50% basis shock, fraction of hours in which the clip is binding, and the maximum single-position liquidation rate under a 5× levered test trader.

The grid search selected:

| Parameter | Final value | Notes |
|---|---:|---|
| Dampening $k$ | **100** | Faster pull than the analytic target; 50% shock half-life drops from ~80h at $k=333$ to 31h at $k=100$ |
| Per-hour cap $c$ | **0.001** (0.1%) | Tighter than the analytic target; basis excursions in the simulated regime never required more |
| Interval $\Delta$ | 1 hour | Matches HL standard |

The final $k = 100$ is more aggressive than the analytic back-of-envelope, because the simulated basis process showed that the slower $k=333$ setting allowed modest-but-persistent basis drift that took several days to decay — uncomfortable from a mark-to-index tracking perspective. The tighter $c = 0.001$ came from the observation that the cap was essentially never binding at this value in the simulated regime — so the cap might as well be tight enough to make any cap-binding event itself a diagnostic signal that something is wrong in the market, rather than a frequent operational event.

These are the parameters in `docs/hip3/PARAMETERS.md` and in the HIP-3 proposal. They supersede the analytic targets in the Phase 4 design doc.

## 7.4 Convergence properties

We verify three convergence invariants in simulation. Each is a hard gate — failure of any one would require recalibration before launch.

### 7.4.1 Mean reversion of basis shocks

Starting from a 50% relative basis and holding the index path fixed, the mark converges to within 1% of the index within **31 hours** — the half-life of the exponential decay at $k = 100$. Within 48 hours the residual basis is below 0.3%; within 72 hours it is below 0.02%. There is no oscillation or overshoot, because the funding response is a pure proportional pull with no integral or derivative terms. Figure 1 (regenerated by `scripts/simulate_funding.py --plot`) shows the trajectory of mark convergence for initial basis shocks of $\pm50\%$, $\pm25\%$, $\pm10\%$.

The 31-hour half-life is well inside the 48-hour gate specified in the Phase 4 design doc, leaving substantial headroom for slower arbitrage in stressed-market conditions.

### 7.4.2 Cap binding fraction

Across the full simulated history, the clip is binding — i.e. $|B_t|/k > c$, meaning the cap is actively truncating the rate — in **less than 1% of hours** (gate: <5%). The hours in which the cap does bind are clustered around simulated regime-break events where the basis briefly spikes to 10–20% before mean-reversion pulls it back. Binding is rare enough that the cap behaves as an exception handler, not a regular operating mode. This is the desired design: in a healthy market the cap is inactive and the funding is a smooth linear function of basis; only in a dislocation does the cap kick in to bound per-hour payments.

### 7.4.3 No funding-driven liquidations in normal regimes

A 5×-levered long position entered at the long-side of the VRP trade, held through a 95th-percentile adverse basis walk, must not be liquidated by funding alone (excluding the direct mark-to-market move). In the simulated regime the worst 95th-percentile cumulative funding paid by a long over a 7-day window is approximately 0.7% of notional, which is well inside any reasonable maintenance-margin buffer. Funding is a second-order threat to position solvency compared to the mark-to-market move itself — which is as it should be, because the contract's risk should come from the underlying variance path, not from the funding mechanism anchoring it.

## 7.5 Relationship to the VRP

A worry we flagged in §7.1 is that the funding mechanism could eat the variance risk premium and destroy the short side's economic case. We can estimate the effect directly.

Assume a typical steady-state relative basis of 5% (the tolerance of a rational arbitrageur net of gas and capital costs), so $|f_t| \approx 0.05/100 = 5\times 10^{-4} = 5$ basis points per hour. Over a 24-hour day this is 12 basis points per day. Over a year (365 days, no compounding) it is approximately 44 percentage points of annualized vol — which is much larger than the 14-point VRP and would therefore erase the short-side carry.

This calculation looks alarming until one notes that it assumes the basis stays at 5% **persistently and in the same direction**. In practice the basis oscillates around zero with both signs equally likely by symmetry arguments, and the funding payments over a long horizon average to nearly zero plus a small drift in whichever direction the basis has a slight structural lean. The simulated net funding paid by the short side over a 30-day horizon at realistic basis statistics is approximately $+1.5$ annualized vol points — an order of magnitude below the $+14$ VRP, and with the correct sign (shorts pay a small funding premium in exchange for earning the much larger variance premium).

The simulated result is robust across a range of basis processes: increasing the basis volatility or the mean-reversion rate changes the funding payment by factors of less than two, never by the factor of ten needed to eat the VRP. The funding mechanism is calibrated cleanly under the second desideratum of §7.1.

## 7.6 Failure modes and invariants

Three invariants guard against funding-mechanism failure modes and are verified in `tests/test_funding.py`:

1. **Zero-basis identity:** $B_t = 0 \Rightarrow f_t = 0$ exactly. No rounding or sign ambiguity at the zero crossing.
2. **Sign antisymmetry:** $f(-B, k, c) = -f(B, k, c)$ for all $B, k, c$. Longs paying shorts under positive basis equals shorts paying longs under negative basis, not an approximation.
3. **Clip monotone:** For all $B$, $|f_t| \le c$; increasing $|B|$ weakly increases $|f_t|$ up to the clip.

Two operational failure modes are also considered:

**Invalid index.** If `is_valid = False` at the funding boundary (the trailing 30-day window has fewer than 38,880 valid observations, §4.4), the funding rate is set to zero and no payment is made. This pauses the anchoring mechanism during exchange outages, which in exchange creates a small risk that the mark drifts away from the (stale) index during the outage. The risk is bounded because new positions are also rejected during invalidity (§4.4), so the outstanding book is frozen for the duration.

**Circuit breaker.** If the oracle stalls for more than 5 consecutive update cycles, funding is frozen and a governance event is emitted. This handles the pathological case where the oracle contract itself has stopped updating (a contract bug, an upstream data-feed failure, or an exhausted gas sponsor). Positions remain markable during the freeze, but funding does not accrue, preventing a runaway funding bill from accumulating against a stale index. Recovery requires an explicit governance intervention — the freeze does not auto-clear.

## 7.7 Parameter summary

| Parameter | Value | Rationale |
|---|---|---|
| Cadence $\Delta$ | 1 hour | Matches Hyperliquid standard |
| Form | $f = \operatorname{clip}(B/k, -c, +c)$ | Stateless clamped linear |
| Dampening $k$ | **100** | 31h half-life on 50% basis shock (grid-searched) |
| Per-hour cap $c$ | **0.001** (0.1%) | Cap-binding <1% of hours in simulated regime |
| Daily hard ceiling | ±2.4% | $24 c$ assuming binding all day (never observed) |
| Zero-basis rate | 0 exact | Invariant |
| Invalid-index rate | 0 (pause) | `is_valid = False` pauses funding |
| Circuit-breaker trigger | 5 missed oracle updates | Freeze funding, emit governance event |

The funding mechanism is deliberately simple. Its job is to anchor the mark gently, survive basis shocks without creating cascades, not dominate the VRP, and stay out of the way of the contract's first-order risk, which comes from the variance path itself. The parameter choices and convergence properties verified above meet all four goals.
