"""Funding mechanism for the variance perp.

See docs/design/02-funding-mechanism.md for the design rationale.
"""

from rvol.funding.rate import FundingParams, funding_rate, funding_rate_series
from rvol.funding.simulate import BasisProcess, SimulationResult, simulate_mark_convergence

__all__ = [
    "FundingParams",
    "funding_rate",
    "funding_rate_series",
    "BasisProcess",
    "SimulationResult",
    "simulate_mark_convergence",
]
