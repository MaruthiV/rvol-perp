"""Tests for the agent-based basis convergence model (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from simulations.abm.agents import Arbitrageur, MarketMaker, NoiseTrader
from simulations.abm.market import (
    MarketParams,
    MarketState,
    agent_mtm_pnl,
    logs_to_arrays,
    simulate,
)


# --- Arbitrageur ---

class TestArbitrageur:
    def test_target_is_counter_basis(self) -> None:
        a = Arbitrageur(alpha=5.0, capital=1.0)
        assert a.target_position(0.1) < 0    # positive basis → short target
        assert a.target_position(-0.1) > 0
        assert a.target_position(0.0) == 0.0

    def test_target_clipped_to_capital(self) -> None:
        a = Arbitrageur(alpha=100.0, capital=1.0)
        assert a.target_position(10.0) == -1.0
        assert a.target_position(-10.0) == 1.0

    def test_turnover_cap(self) -> None:
        a = Arbitrageur(alpha=5.0, max_turnover_per_step=0.2, capital=1.0)
        # Big basis → desired delta > cap
        order = a.step_order(basis=1.0)
        assert abs(order) == pytest.approx(0.2)

    def test_settle_updates_position_and_cash(self) -> None:
        a = Arbitrageur()
        a.settle(fill=0.5, fill_price=100.0)
        assert a.position == 0.5
        assert a.cash == -50.0


# --- Market maker ---

class TestMarketMaker:
    def test_mid_skews_against_inventory(self) -> None:
        mm = MarketMaker(beta=0.5, inventory_cap=1.0, position=0.5)
        mid = mm.mid(index=100.0)
        # Positive inventory → mid pulled down (MM wants to sell)
        assert mid < 100.0

    def test_zero_inventory_mid_equals_index(self) -> None:
        mm = MarketMaker(beta=0.5, inventory_cap=1.0, position=0.0)
        assert mm.mid(index=100.0) == 100.0

    def test_quote_wraps_mid(self) -> None:
        mm = MarketMaker(half_spread=0.001, position=0.0)
        bid, ask = mm.quote(index=100.0)
        assert bid < 100.0 < ask
        assert ask - bid == pytest.approx(2 * 0.001 * 100.0)


# --- Noise trader ---

class TestNoiseTrader:
    def test_determinism_under_seeded_rng(self) -> None:
        n1 = NoiseTrader(rng=np.random.default_rng(42))
        n2 = NoiseTrader(rng=np.random.default_rng(42))
        a = [n1.step_flow() for _ in range(10)]
        b = [n2.step_flow() for _ in range(10)]
        assert a == b


# --- Market state ---

class TestMarketState:
    def test_basis(self) -> None:
        s = MarketState(index=100.0, mark=105.0)
        assert s.basis == pytest.approx(0.05)

    def test_degenerate_index(self) -> None:
        s = MarketState(index=0.0, mark=1.0)
        assert s.basis == 0.0


# --- Full simulation ---

class TestSimulation:
    @pytest.fixture
    def flat_index(self) -> np.ndarray:
        """Flat index at 0.002 daily variance (~55% ann vol), 30 days hourly."""
        return np.full(30 * 24, 0.002)

    def test_requires_two_observations(self) -> None:
        with pytest.raises(ValueError):
            simulate(
                np.array([0.002]),
                MarketParams(),
                Arbitrageur(),
                MarketMaker(),
                NoiseTrader(),
            )

    def test_zero_basis_zero_noise_stays_zero(self, flat_index: np.ndarray) -> None:
        """With silent noise trader and zero initial basis, basis stays ~0."""
        mp = MarketParams()
        arb = Arbitrageur()
        mm = MarketMaker()
        # Noise with zero variance → no flow
        noise = NoiseTrader(sigma_bias=0.0, sigma_noise=0.0, rng=np.random.default_rng(0))
        logs = simulate(flat_index, mp, arb, mm, noise, initial_basis=0.0)
        arr = logs_to_arrays(logs)
        assert np.abs(arr["basis"]).max() < 1e-10

    def test_initial_basis_decays(self, flat_index: np.ndarray) -> None:
        """With sufficient arb capital × price impact, basis decays toward zero.

        Steady-state basis under flow-driven convergence is
        b_ss = b_0 / (1 + λ·α·capital), so we need (λ·α·capital) >> 1.
        """
        mp = MarketParams(price_impact=0.05)
        arb = Arbitrageur(alpha=20.0, max_turnover_per_step=0.3, capital=20.0)
        mm = MarketMaker(inventory_cap=20.0)
        noise = NoiseTrader(sigma_bias=0.0, sigma_noise=0.0, rng=np.random.default_rng(0))
        logs = simulate(flat_index, mp, arb, mm, noise, initial_basis=0.1)
        arr = logs_to_arrays(logs)
        # 0.1 / (1 + 0.05·20·20) = 0.1/21 ≈ 0.0048
        assert abs(arr["basis"][-1]) < 0.01

    def test_arb_inventory_is_counter_basis(self, flat_index: np.ndarray) -> None:
        """Arb position and basis should have strong negative correlation."""
        mp = MarketParams()
        arb = Arbitrageur()
        mm = MarketMaker()
        noise = NoiseTrader(rng=np.random.default_rng(0))
        logs = simulate(flat_index, mp, arb, mm, noise, initial_basis=0.0)
        arr = logs_to_arrays(logs)
        # Skip first few steps (arb hasn't built position yet)
        corr = np.corrcoef(arr["arb_position"][10:], arr["basis"][10:])[0, 1]
        assert corr < -0.3

    def test_position_conservation(self, flat_index: np.ndarray) -> None:
        """Total book: arb + mm + cumulative_noise_flow = 0.

        MM is always the direct counter to (arb + noise) net flow, so
        mm.position = -cumsum(noise) - arb.position by construction.
        """
        mp = MarketParams()
        arb = Arbitrageur()
        mm = MarketMaker()
        noise = NoiseTrader(rng=np.random.default_rng(0))
        logs = simulate(flat_index, mp, arb, mm, noise, initial_basis=0.0)
        arr = logs_to_arrays(logs)
        noise_cumulative = np.cumsum(arr["noise_flow"])
        total = arr["arb_position"] + arr["mm_position"] + noise_cumulative
        np.testing.assert_allclose(total, 0.0, atol=1e-9)

    def test_mark_tracks_index_under_noise_only(self, flat_index: np.ndarray) -> None:
        """With active noise + arb, mark should stay within a few % of index."""
        mp = MarketParams()
        arb = Arbitrageur()
        mm = MarketMaker()
        noise = NoiseTrader(rng=np.random.default_rng(0))
        logs = simulate(flat_index, mp, arb, mm, noise, initial_basis=0.0)
        arr = logs_to_arrays(logs)
        assert np.abs(arr["basis"]).max() < 0.10  # never more than 10%
        assert np.abs(arr["basis"]).mean() < 0.03

    def test_shock_convergence_from_plus20(self, flat_index: np.ndarray) -> None:
        """From +20% initial basis, well-capitalized arbs bring it near zero."""
        mp = MarketParams(price_impact=0.05)
        arb = Arbitrageur(alpha=20.0, max_turnover_per_step=0.3, capital=20.0)
        mm = MarketMaker(inventory_cap=20.0)
        noise = NoiseTrader(sigma_bias=0.0, sigma_noise=0.0, rng=np.random.default_rng(0))
        logs = simulate(flat_index, mp, arb, mm, noise, initial_basis=0.20)
        arr = logs_to_arrays(logs)
        # Steady state: 0.2 / 21 ≈ 0.0095
        assert abs(arr["basis"][48]) < 0.05
        assert abs(arr["basis"][-1]) < 0.015
