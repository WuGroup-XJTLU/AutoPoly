# -*- coding: utf-8 -*-
"""
Atomistic melt state for the equilibration module: wraps
``AtomisticSystem`` (loader output) + the Rust ``AtomisticMC`` engine,
with serialization and certification helpers (backbone MSID).

Created on 2026-08-20
@author: zwu
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.system import logger
from .atomistic_loader import AtomisticSystem, backbone_chains, load_lammps_system


class AtomisticMelt:
    """Atomistic melt under MC equilibration (typed, OPLS-AA subset)."""

    def __init__(self, system: AtomisticSystem, seed: int = 12345,
                 kbt: float = 0.9, **engine_kw) -> None:
        import autopoly_mc
        self._engine_kw = engine_kw

        self.system = system
        self.kbt = kbt
        self.seed = seed
        self._chains = backbone_chains(system)
        n_types = int(system.type_ids.max())
        masses_list = [0.0] + [system.masses.get(t, 12.0) for t in range(1, n_types + 1)]
        self._engine = autopoly_mc.AtomisticMC(
            positions=system.positions.tolist(),
            box_size=system.box_size,
            types=system.type_ids.tolist(),
            charges=system.charges.tolist(),
            mols=system.mol_ids.tolist(),
            bonds=system.bonds,
            angles=system.angles,
            dihedrals=system.dihedrals,
            chains=self._chains,
            pair_eps=system.pair_eps.tolist(),
            pair_sig=system.pair_sig.tolist(),
            bond_k=system.bond_k.tolist(),
            bond_r0=system.bond_r0.tolist(),
            angle_k=system.angle_k.tolist(),
            angle_t0=system.angle_t0.tolist(),
            dih_k=[list(map(float, row)) for row in system.dih_k],
            masses_by_type=masses_list,
            lj_cut=system.lj_cut,
            coul_cut=system.coul_cut,
            scale14_lj=system.scale14_lj,
            scale14_coul=system.scale14_coul,
            temperature=kbt,
            seed=seed,
            **engine_kw,
        )

    @classmethod
    def from_lammps(cls, data: str | Path, settings: str | Path,
                    kbt: float = 0.9, seed: int = 12345,
                    engine_kw: Optional[Dict[str, Any]] = None, **kw) -> "AtomisticMelt":
        system = load_lammps_system(data, settings, **kw)
        return cls(system, seed=seed, kbt=kbt, **(engine_kw or {}))

    # ------------------------------------------------------------------ #
    @property
    def n_atoms(self) -> int:
        return self.system.n_atoms

    @property
    def backbone_chains(self) -> List[List[int]]:
        return self._chains

    @property
    def chain_lengths(self) -> List[int]:
        return [len(c) for c in self._chains]

    def run(self, n_steps: int) -> Dict[str, float]:
        self._engine.run(n_steps)
        return self._engine.acceptance_rates()

    def energy(self) -> float:
        return self._engine.energy()

    def recompute_energy(self) -> float:
        return self._engine.recompute_energy()

    def sync_positions(self) -> None:
        self.system.positions = np.asarray(self._engine.positions(), dtype=float)

    def msid_backbone(self, max_s: Optional[int] = None) -> np.ndarray:
        """Backbone MSID as (s, R^2) array."""
        return np.asarray(self._engine.msid(max_s), dtype=float)

    def bookkeeping_offset(self) -> float:
        return self.energy() - self.recompute_energy()

    # ------------------------------------------------------------------ #
    # HMC-healed double-bridging (Option B)
    # ------------------------------------------------------------------ #
    def hmc_bridge_attempt(self, healer, temperature_k: float = 450.0) -> bool:
        """
        One HMC-healed double-bridge attempt. Returns True if a swap was
        accepted (topology + healed positions adopted).
        """
        import autopoly_mc  # noqa: F401
        n_chains = self._engine.n_chains()
        if n_chains < 2:
            return False
        a = int(np.random.randint(n_chains))
        cands = self._engine.enumerate_bridge_candidates(a)
        n_fwd = len(cands)
        if n_fwd == 0:
            return False
        a, b, s, flip = cands[int(np.random.randint(n_fwd))]
        old_pos = np.asarray(self._engine.positions(), dtype=float)
        u_old = self._engine.recompute_energy()
        if not self._engine.apply_bridge_proposal(a, b, s, flip):
            return False
        res = healer.heal(
            self.system,
            old_pos,
            self._engine.bonds(),
            self._engine.angles(),
            self._engine.dihedrals(),
            u_old,
            temperature_k,
            seed=int(np.random.randint(1, 2**31)),
        )
        if res.healed_positions is None:
            # trajectory failed; revert topology and geometry
            self._engine.apply_bridge_proposal(a, b, s, flip)
            self._engine.set_positions(old_pos.tolist())
            return False
        # reverse-candidate count in the healed new state
        self._engine.set_positions(res.healed_positions.tolist())
        n_rev = len(self._engine.enumerate_bridge_candidates(a))
        logp = -res.delta_h / self.kbt + np.log(n_fwd / max(n_rev, 1))
        if res.delta_h <= 0.0 or np.random.random() < np.exp(logp):
            return True
        # reject: revert topology (involution) and restore old geometry
        self._engine.apply_bridge_proposal(a, b, s, flip)
        self._engine.set_positions(old_pos.tolist())
        return False

    def anneal_with_bridges(
        self,
        local_steps: int,
        healer,
        bridge_every: int = 10_000,
        temperature_k: float = 450.0,
    ) -> Dict[str, Any]:
        """
        Anneal: `local_steps` of kernel MC, interleaved with one
        HMC-healed bridge attempt per `bridge_every` local steps.
        Returns statistics.
        """
        stats = {"bridge_attempts": 0, "bridge_accepted": 0}
        done = 0
        chunk = bridge_every
        while done < local_steps:
            chunk = min(bridge_every, local_steps - done)
            self.run(chunk)
            done += chunk
            stats["bridge_attempts"] += 1
            if self.hmc_bridge_attempt(healer, temperature_k):
                stats["bridge_accepted"] += 1
        stats["acceptance"] = (
            stats["bridge_accepted"] / stats["bridge_attempts"]
            if stats["bridge_attempts"] else 0.0
        )
        stats["energy_offset"] = self.bookkeeping_offset()
        return stats

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        self.sync_positions()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            positions=self.system.positions,
            type_ids=self.system.type_ids,
            charges=self.system.charges,
            mol_ids=self.system.mol_ids,
            chains=np.array([np.asarray(c, dtype=np.int64) for c in self._chains],
                            dtype=object),
            box_size=np.array([self.system.box_size]),
        )
        meta = {
            "n_atoms": self.n_atoms,
            "chain_lengths": self.chain_lengths,
            "box_size": self.system.box_size,
            "kbt": self.kbt,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved AtomisticMelt to {path.with_suffix('.npz')}")
        return path
