"""Market step loop for the basis-convergence ABM.

At each hourly step:
  1. Noise trader emits a signed order at the current mid
  2. Arb computes desired order from the current basis, submits it
  3. MM takes the opposite side of all flow at its quoted price
  4. Mark moves under linear (Kyle) price impact from net signed flow
  5. Funding is paid using the Phase 4 `funding_rate()`
  6. Positions are marked to the new index change; cash PnL accrued
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rvol.funding import FundingParams, funding_rate

from simulations.abm.agents import Arbitrageur, MarketMaker, NoiseTrader


@dataclass(frozen=True)
class MarketParams:
    """Global market parameters."""

    price_impact: float = 0.01     # lambda in Kyle-style impact
    funding_params: FundingParams = field(default_factory=lambda: FundingParams(
        dampening=100.0, cap=0.001
    ))


@dataclass
class MarketState:
    index: float                   # exogenous index (RV30)
    mark: float                    # traded mark of the perp

    @property
    def basis(self) -> float:
        if self.index <= 0:
            return 0.0
        return (self.mark - self.index) / self.index


@dataclass
class StepLog:
    index: float
    mark: float
    basis: float
    funding: float
    arb_position: float
    arb_cash: float
    mm_position: float
    mm_cash: float
    noise_flow: float
    net_flow: float


def simulate(
    index_path: np.ndarray,
    market_params: MarketParams,
    arb: Arbitrageur,
    mm: MarketMaker,
    noise: NoiseTrader,
    initial_basis: float = 0.0,
) -> list[StepLog]:
    """Run the ABM over a fixed exogenous index path.

    Returns a list of StepLog rows, one per step.
    """
    n = len(index_path)
    if n < 2:
        raise ValueError("index_path must have at least 2 observations")

    state = MarketState(
        index=float(index_path[0]),
        mark=float(index_path[0]) * (1.0 + initial_basis),
    )
    logs: list[StepLog] = []

    prev_index = float(index_path[0])
    for t in range(n):
        new_index = float(index_path[t])
        # Mark inherits the proportional index move automatically. This is
        # the "passive arbitrage" of the underlying level — in real markets
        # the perp tracks the underlying's level changes without needing
        # the on-book arbs to push each tick. The residual basis process
        # captures only deviations on top of that.
        if prev_index > 0:
            state.mark = state.mark * (new_index / prev_index)
        state.index = new_index
        prev_index = new_index

        # 1. Noise order
        noise_order = noise.step_flow()

        # 2. Arb order (based on pre-step basis)
        arb_order = arb.step_order(state.basis)

        # 3. MM is counterparty — takes the opposite side at its quoted price
        net_flow = noise_order + arb_order
        bid, ask = mm.quote(state.index)
        fill_price = ask if net_flow > 0 else bid

        arb.settle(arb_order, fill_price)
        # MM also fills the noise trader at the same price
        mm.settle(-net_flow, fill_price)

        # 4. Price impact — mark shifts in the direction of net flow
        impact = market_params.price_impact * net_flow
        new_mark = state.mark * (1.0 + impact)

        # 5. Funding payment on the post-impact basis
        post_state = MarketState(index=state.index, mark=new_mark)
        f = funding_rate(new_mark, state.index, market_params.funding_params)
        # Long (positive position) pays funding when f > 0
        arb.cash -= arb.position * f * state.index
        mm.cash -= mm.position * f * state.index

        state.mark = new_mark

        logs.append(StepLog(
            index=state.index,
            mark=state.mark,
            basis=state.basis,
            funding=f,
            arb_position=arb.position,
            arb_cash=arb.cash,
            mm_position=mm.position,
            mm_cash=mm.cash,
            noise_flow=noise_order,
            net_flow=net_flow,
        ))

    return logs


def logs_to_arrays(logs: list[StepLog]) -> dict[str, np.ndarray]:
    """Convert list of StepLog rows to a dict of 1-D numpy arrays."""
    keys = StepLog.__dataclass_fields__.keys()
    return {k: np.array([getattr(l, k) for l in logs]) for k in keys}


def agent_mtm_pnl(arr: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mark-to-market PnL of arb and MM over time.

    PnL = cash + position × index. Returns (arb_pnl_series, mm_pnl_series).
    """
    arb_pnl = arr["arb_cash"] + arr["arb_position"] * arr["index"]
    mm_pnl = arr["mm_cash"] + arr["mm_position"] * arr["index"]
    return arb_pnl, mm_pnl
