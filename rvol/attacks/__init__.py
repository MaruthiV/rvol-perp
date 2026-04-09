"""Adversarial attack simulation for RV30 index — Phase 7.

See docs/design/05-manipulation-cost.md for the threat model.
"""

from rvol.attacks.inject import (
    AttackParams,
    AttackResult,
    inject_single_spike,
    inject_sustained,
    estimate_cost_usd,
)

__all__ = [
    "AttackParams",
    "AttackResult",
    "inject_single_spike",
    "inject_sustained",
    "estimate_cost_usd",
]
