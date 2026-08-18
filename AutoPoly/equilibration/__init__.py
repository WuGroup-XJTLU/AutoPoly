# -*- coding: utf-8 -*-
"""
Growth-assisted / connectivity-altering equilibration for polymer melts.

This package implements the "grow-relax-rewire" equilibration method
(see ``equilibration_method.md`` in the project root): molecular-weight
annealing through oligomer joining, followed by fixed-N connectivity
annealing, with Monte Carlo moves executed by the Rust ``autopoly_mc``
kernel.

Phase 0 scope
-------------
* :class:`MeltState` — the mutable melt container (positions, per-chain
  ordered bead contours, cubic PBC box) with structural invariants.
* KG bead-spring support (harmonic / FENE bonds, cosine angle potential,
  LJ / WCA pairs) via :class:`MCRunner` over the Rust engine.
* Certification helper: mean-square internal distance (MSID) access.

Connectivity-altering moves (join, segment exchange) arrive in Phase 1.

Created on 2026-08-18
@author: zwu
"""

from .state import MeltState
from .runner import MCRunner, MCParams

__all__ = [
    "MeltState",
    "MCRunner",
    "MCParams",
]
