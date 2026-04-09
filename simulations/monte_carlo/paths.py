"""JAX stochastic-volatility price path generator.

Log-normal SV model (see docs/design/03-monte-carlo-lifecycle.md):

    v_{t+1}  = v_t + κ(θ − v_t) Δt + η √Δt · ε^v_t        (OU log-variance)
    r_{t+1}  = √(exp(v_t) · Δt_day) · ε^s_t                (log return)
                                                            corr(ε^v, ε^s) = ρ

All time units are days. Δt is typically 1/1440 (one minute).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SVParams:
    """Log-normal stochastic volatility parameters (units: days).

    Defaults calibrated to BTC RV30 empirics from Phase 2/3:
      - theta: ln(0.000662)  ≈ -7.32   (median daily variance)
      - kappa: ln(2) / 191   ≈ 0.00363 (half-life of vol shock = 191 days)
      - eta:   0.06                    (vol-of-log-vol per √day)
      - rho:  -0.08                    (leverage effect)
      - v0:    theta                   (start at long-run mean)
      - s0:    1.0                     (arbitrary price numeraire)
    """

    theta: float = -7.32
    kappa: float = 0.00363
    eta: float = 0.06
    rho: float = -0.08
    v0: float = -7.32
    s0: float = 1.0


def simulate_paths(
    params: SVParams,
    n_paths: int,
    n_steps: int,
    dt_days: float = 1.0 / 1440.0,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Simulate `n_paths` paths of `n_steps` log-price increments.

    Returns:
        prices: (n_paths, n_steps+1) float32 — price levels including t=0
        log_var: (n_paths, n_steps+1) float32 — log-variance process
    """
    key = jax.random.PRNGKey(seed)
    key_v, key_s = jax.random.split(key)

    sqrt_dt = jnp.sqrt(dt_days)
    rho = params.rho
    sqrt_one_minus_rho2 = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 0.0))

    # Independent shocks: shape (n_paths, n_steps)
    eps_v = jax.random.normal(key_v, shape=(n_paths, n_steps), dtype=jnp.float32)
    eps_s_indep = jax.random.normal(key_s, shape=(n_paths, n_steps), dtype=jnp.float32)
    eps_s = rho * eps_v + sqrt_one_minus_rho2 * eps_s_indep

    kappa = jnp.float32(params.kappa)
    theta = jnp.float32(params.theta)
    eta = jnp.float32(params.eta)
    dt = jnp.float32(dt_days)

    def step(carry, inp):
        v_prev, log_s_prev = carry
        ev, es = inp
        v_new = v_prev + kappa * (theta - v_prev) * dt + eta * sqrt_dt * ev
        # Return std at 1-min: sqrt(exp(v_prev) * dt)
        ret_std = jnp.sqrt(jnp.exp(v_prev) * dt)
        log_return = ret_std * es
        log_s_new = log_s_prev + log_return
        return (v_new, log_s_new), (v_new, log_s_new)

    v0 = jnp.full((n_paths,), params.v0, dtype=jnp.float32)
    log_s0 = jnp.full((n_paths,), jnp.log(params.s0), dtype=jnp.float32)

    # Transpose shocks so scan iterates over the time axis
    eps_v_T = eps_v.T  # (n_steps, n_paths)
    eps_s_T = eps_s.T

    _, (v_path, log_s_path) = jax.lax.scan(step, (v0, log_s0), (eps_v_T, eps_s_T))

    # v_path, log_s_path are (n_steps, n_paths) — transpose to (n_paths, n_steps)
    v_path = v_path.T
    log_s_path = log_s_path.T

    # Prepend t=0
    v_full = jnp.concatenate([v0[:, None], v_path], axis=1)
    log_s_full = jnp.concatenate([log_s0[:, None], log_s_path], axis=1)

    prices = jnp.exp(log_s_full)
    return prices, v_full
