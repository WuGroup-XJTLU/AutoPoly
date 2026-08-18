# -*- coding: utf-8 -*-
"""
Python driver over the Rust ``autopoly_mc.McEngine``.

Holds the simulation parameters (force-field flavor, move mix, RNG seed)
and keeps the Python :class:`MeltState` synchronized with the engine.

Created on 2026-08-18
@author: zwu
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import autopoly_mc

from ..core.system import logger
from .state import MeltState


@dataclass
class MCParams:
    """Force-field and MC parameters for a KG bead-spring melt."""

    temperature: float = 1.0
    # Pair potential (LJ; pair_wca=True selects the WCA variant)
    pair_epsilon: float = 1.0
    pair_sigma: float = 1.0
    pair_cutoff: float = 2.5
    pair_shifted: bool = False
    pair_wca: bool = False
    # Bonds: "harmonic" (bond_k, bond_r0) or "fene" (bond_k, fene_r0max)
    bond_model: str = "harmonic"
    bond_k: float = 100.0
    bond_r0: float = 1.0
    fene_r0max: float = 1.5
    # Angle potential k*(1-cos(theta)); 0 disables
    angle_k: float = 0.0
    # Move step sizes
    max_displacement: float = 0.5
    max_angle: float = 0.3
    # Move mixture
    move_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "displacement": 0.4,
            "crankshaft": 0.1,
            "pivot": 0.15,
            "reptation": 0.1,
            "translation": 0.15,
            "rotation": 0.1,
        }
    )

    @classmethod
    def kremer_grest(cls, angle_k: float = 0.0) -> "MCParams":
        """Canonical KG melt: FENE + WCA, T=1."""
        return cls(
            pair_wca=True,
            bond_model="fene",
            bond_k=30.0,
            fene_r0max=1.5,
            angle_k=angle_k,
        )


class MCRunner:
    """Drives MC equilibration of a :class:`MeltState` via the Rust kernel."""

    def __init__(
        self,
        state: MeltState,
        params: Optional[MCParams] = None,
        seed: int = 12345,
    ) -> None:
        self.state = state
        self.params = params or MCParams()
        self.seed = seed
        self._engine: Optional[Any] = None
        self._build_engine()

    def _build_engine(self) -> None:
        p = self.params
        positions, chains, box = self.state.to_engine_args()
        self._engine = autopoly_mc.McEngine(
            positions,
            chains,
            box,
            seed=self.seed,
            pair_epsilon=p.pair_epsilon,
            pair_sigma=p.pair_sigma,
            pair_cutoff=p.pair_cutoff,
            pair_shifted=p.pair_shifted,
            pair_wca=p.pair_wca,
            bond_model=p.bond_model,
            bond_k=p.bond_k,
            bond_r0=p.bond_r0,
            fene_r0max=p.fene_r0max,
            angle_k=p.angle_k,
            temperature=p.temperature,
            max_displacement=p.max_displacement,
            max_angle=p.max_angle,
            move_weights=p.move_weights,
        )

    def run(self, n_steps: int) -> Dict[str, float]:
        """Run `n_steps` MC steps, sync state back, return acceptance rates."""
        self._engine.run(n_steps)
        self.state.update_from_engine(self._engine)
        return self._engine.acceptance_rates()

    def energy(self) -> float:
        return self._engine.energy()

    def recompute_energy(self) -> float:
        return self._engine.recompute_energy()

    def validate(self) -> bool:
        self.state.validate()
        return self._engine.validate()

    def msid(self, max_s: Optional[int] = None):
        """Mean-square internal distance <R^2(s)> as a (s, value) list."""
        return self._engine.msid(max_s)

    def bookkeeping_offset(self) -> float:
        """
        incremental energy - exact energy. Should be ~0 (|offset| << kT)
        for production runs started from overlap-free configurations; a
        finite offset after a *cold* (overlapping) start is a known
        float64 bookkeeping artifact of the hard-core regime and does not
        affect acceptance decisions (which use exact local ΔE).
        """
        return self.energy() - self.recompute_energy()
