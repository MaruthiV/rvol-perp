"""Agent classes for the basis-convergence ABM.

Agents are plain dataclasses with update methods. No inheritance, no
registry — the market loop in `market.py` owns composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Arbitrageur:
    """Basis arbitrageur.

    Targets a signed position inversely proportional to the basis. Cannot
    rebalance more than `max_turnover_per_step` units per step — this is an
    ABSOLUTE cap on flow, not a fraction of capital, so that bigger arb
    books don't produce pathological price impact.

    Position is clipped to [−capital, +capital] (unit notional in index units).
    """

    alpha: float = 5.0
    max_turnover_per_step: float = 0.2
    capital: float = 1.0
    position: float = 0.0
    cash: float = 0.0

    def target_position(self, basis: float) -> float:
        return float(np.clip(-self.alpha * basis * self.capital, -self.capital, self.capital))

    def step_order(self, basis: float) -> float:
        """Return the signed order the arb wants to place this step."""
        target = self.target_position(basis)
        delta = target - self.position
        return float(np.clip(delta, -self.max_turnover_per_step, self.max_turnover_per_step))

    def settle(self, fill: float, fill_price: float) -> None:
        """Apply an executed order. `fill_price` is absolute (index units)."""
        self.position += fill
        self.cash -= fill * fill_price


@dataclass
class MarketMaker:
    """Inventory-aware market maker.

    Quotes a two-sided market skewed against its own inventory. Trades
    happen at quoted price (half-spread around the skewed mid).
    """

    beta: float = 0.5
    half_spread: float = 0.0005
    inventory_cap: float = 0.5
    position: float = 0.0
    cash: float = 0.0

    def mid(self, index: float) -> float:
        """Skewed mid: pulled toward zero inventory."""
        skew = self.beta * (self.position / self.inventory_cap)
        return index * (1.0 - skew)

    def quote(self, index: float) -> tuple[float, float]:
        m = self.mid(index)
        return m * (1.0 - self.half_spread), m * (1.0 + self.half_spread)

    def settle(self, fill: float, fill_price: float) -> None:
        self.position += fill
        self.cash -= fill * fill_price


@dataclass
class NoiseTrader:
    """Uninformed signed order flow with AR(1) bias.

    `bias_t = rho_bias * bias_{t-1} + eps`
    `flow_t ~ Normal(bias_t, sigma_noise)` — net signed size per step.
    """

    rho_bias: float = 0.6
    sigma_bias: float = 0.02
    sigma_noise: float = 0.05
    bias: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def step_flow(self) -> float:
        self.bias = self.rho_bias * self.bias + self.rng.normal(0.0, self.sigma_bias)
        return float(self.bias + self.rng.normal(0.0, self.sigma_noise))
