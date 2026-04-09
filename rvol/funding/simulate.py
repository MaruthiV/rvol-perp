"""Mark-convergence simulation for the variance perp funding mechanism.

Given an exogenous index series (RV30 replayed from history), simulate a
traded mark price whose basis `b_t = (mark_t - index_t) / index_t` evolves
as a mean-reverting AR(1) process with funding feedback:

    b_{t+1} = rho * b_t + sigma * eps_t - gamma * f_t

where `f_t` is the current-interval funding rate (longs pay when positive),
and `gamma` represents the arbitrage reaction strength — arbs see a funding
payment and move capital to close the basis faster than passive mean
reversion alone.

This is intentionally a reduced-form model. A full agent-based convergence
model with explicit market makers and arbs lives in Phase 6 under
`simulations/abm/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rvol.funding.rate import FundingParams, funding_rate_series


@dataclass(frozen=True)
class BasisProcess:
    """Parameters of the reduced-form basis process.

    Attributes:
        rho: AR(1) persistence of the basis absent funding (0 < rho < 1).
        sigma: innovation std of basis shocks per interval.
        gamma: arbitrage feedback — how strongly current-interval funding
            pulls basis toward zero next interval. gamma=0 disables arb.
        initial_basis: seed basis at t=0.
    """

    rho: float = 0.90
    sigma: float = 0.005
    gamma: float = 10.0
    initial_basis: float = 0.0


@dataclass
class SimulationResult:
    index: pd.Series            # input index
    mark: pd.Series             # simulated traded mark
    basis: pd.Series            # (mark - index) / index
    funding: pd.Series          # per-interval funding rate (long→short)
    long_pnl: pd.Series         # cumulative pnl of a passive unit long
    short_pnl: pd.Series        # cumulative pnl of a passive unit short
    cap_hit_fraction: float     # fraction of intervals where |f| == cap
    mean_abs_basis: float
    max_abs_basis: float


def simulate_mark_convergence(
    index: pd.Series,
    funding_params: FundingParams,
    basis_process: BasisProcess,
    seed: int | None = 0,
) -> SimulationResult:
    """Simulate the mark price path and resulting funding flows.

    PnL accounting (per unit notional of index at t=0):
      - Position value: long earns (index_{t+1} - index_t) per interval
        (variance payoff tracks the index), minus funding paid to short.
      - Short symmetric.
      - Funding is paid in index units (notional × rate).

    Returns a SimulationResult with aligned series on the input index's
    timestamps.
    """
    if len(index) < 2:
        raise ValueError("index must have at least 2 observations")

    rng = np.random.default_rng(seed)
    idx_vals = index.to_numpy(dtype=np.float64)
    n = len(idx_vals)

    basis = np.zeros(n, dtype=np.float64)
    funding = np.zeros(n, dtype=np.float64)
    mark = np.zeros(n, dtype=np.float64)

    basis[0] = basis_process.initial_basis
    mark[0] = idx_vals[0] * (1.0 + basis[0])
    # funding at t=0: based on current basis
    funding[0] = float(np.clip(
        basis[0] / funding_params.dampening,
        -funding_params.cap,
        funding_params.cap,
    ))

    noise = rng.normal(0.0, basis_process.sigma, size=n)
    for t in range(1, n):
        b_prev = basis[t - 1]
        f_prev = funding[t - 1]
        b_t = basis_process.rho * b_prev + noise[t] - basis_process.gamma * f_prev
        basis[t] = b_t
        mark[t] = idx_vals[t] * (1.0 + b_t)
        funding[t] = float(np.clip(
            b_t / funding_params.dampening,
            -funding_params.cap,
            funding_params.cap,
        ))

    mark_s = pd.Series(mark, index=index.index, name="mark")
    basis_s = pd.Series(basis, index=index.index, name="basis")
    funding_s = pd.Series(funding, index=index.index, name="funding")

    # PnL: long unit notional. Payoff from index change plus funding received.
    # Long pays funding when funding > 0.
    d_index = np.diff(idx_vals, prepend=idx_vals[0])
    long_inc = d_index - funding * idx_vals
    short_inc = -d_index + funding * idx_vals
    long_pnl = pd.Series(np.cumsum(long_inc), index=index.index, name="long_pnl")
    short_pnl = pd.Series(np.cumsum(short_inc), index=index.index, name="short_pnl")

    cap_hit = np.isclose(np.abs(funding), funding_params.cap).mean()
    abs_b = np.abs(basis)

    return SimulationResult(
        index=index,
        mark=mark_s,
        basis=basis_s,
        funding=funding_s,
        long_pnl=long_pnl,
        short_pnl=short_pnl,
        cap_hit_fraction=float(cap_hit),
        mean_abs_basis=float(abs_b.mean()),
        max_abs_basis=float(abs_b.max()),
    )


# Re-export for convenience
__all__ = ["BasisProcess", "SimulationResult", "simulate_mark_convergence", "funding_rate_series"]
