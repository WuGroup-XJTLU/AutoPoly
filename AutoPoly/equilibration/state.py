# -*- coding: utf-8 -*-
"""
Melt state container for equilibration.

``MeltState`` owns the mutable simulation state that connectivity-altering
Monte Carlo operates on. Unlike the contiguous-slice representation used
by :class:`AutoPoly.models.bead_spring_system.BeadSpringSystem`, chains
here are *ordered lists of bead ids* along the contour, so join / cut /
segment-exchange moves can rewrite chain membership freely.

The Python object is the serializable, inspectable truth; the Rust
``autopoly_mc.McEngine`` is constructed from it for computation.

Created on 2026-08-18
@author: zwu
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.system import logger


class MeltState:
    """
    Mutable bead-spring melt state.

    Attributes:
        positions: (n_beads, 3) array, wrapped into the box.
        chains: per-chain ordered bead ids along the contour.
        box_size: cubic periodic box edge length.
    """

    def __init__(
        self,
        positions: np.ndarray,
        chains: Sequence[Sequence[int]],
        box_size: float,
    ) -> None:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (N, 3), got shape {positions.shape}"
            )
        if box_size <= 0:
            raise ValueError(f"box_size must be positive, got {box_size}")
        self.positions = positions
        self.chains: List[List[int]] = [list(map(int, c)) for c in chains]
        self.box_size = float(box_size)
        self.validate()

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_bead_spring_system(cls, system: Any) -> "MeltState":
        """
        Build from a generated
        :class:`~AutoPoly.models.bead_spring_system.BeadSpringSystem`.
        """
        if system._positions is None:
            raise ValueError(
                "BeadSpringSystem has no generated positions; call "
                "generate()/saw_generate() first"
            )
        positions = np.asarray(system._positions, dtype=float)
        chains = [
            list(range(start, end)) for start, end in system._chain_indices
        ]
        return cls(positions, chains, float(system._calculate_box_size()))

    @classmethod
    def random_walk(
        cls,
        n_chains: int,
        n_per_chain: int,
        density: float = 0.85,
        bond_length: float = 1.0,
        seed: int = 0,
    ) -> "MeltState":
        """
        Random-walk initial configuration (for tests and cold starts).

        Note: random walks contain bead overlaps at melt density; relax
        (e.g. displacement-only MC or push-off) before production MC —
        see the documentation on the cold-start energy regime.
        """
        rng = np.random.default_rng(seed)
        n = n_chains * n_per_chain
        box = (n / density) ** (1.0 / 3.0)
        positions = np.empty((n, 3))
        chains: List[List[int]] = []
        b = 0
        for _ in range(n_chains):
            chain = []
            p = rng.uniform(0.0, box, 3)
            for _ in range(n_per_chain):
                chain.append(b)
                positions[b] = p % box
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                p = p + bond_length * d
                b += 1
            chains.append(chain)
        return cls(positions, chains, box)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def n_beads(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    @property
    def chain_lengths(self) -> List[int]:
        return [len(c) for c in self.chains]

    def monodisperse(self) -> bool:
        lengths = self.chain_lengths
        return len(set(lengths)) <= 1

    def bonds(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for chain in self.chains:
            out.extend((chain[i], chain[i + 1]) for i in range(len(chain) - 1))
        return out

    # ------------------------------------------------------------------ #
    # Invariants
    # ------------------------------------------------------------------ #
    def validate(self) -> bool:
        """
        Check structural invariants. Raises ValueError on violation.

        * every bead appears in exactly one chain, exactly once;
        * every chain has >= 2 beads;
        * positions are finite and inside the box.
        """
        n = self.n_beads
        seen = np.zeros(n, dtype=bool)
        for c, chain in enumerate(self.chains):
            if len(chain) < 2:
                raise ValueError(f"chain {c} has {len(chain)} beads (need >= 2)")
            for bead in chain:
                if bead < 0 or bead >= n:
                    raise ValueError(
                        f"chain {c} references bead {bead} out of range [0, {n})"
                    )
                if seen[bead]:
                    raise ValueError(f"bead {bead} appears in more than one chain")
                seen[bead] = True
        missing = int((~seen).sum())
        if missing:
            raise ValueError(f"{missing} beads do not belong to any chain")
        if not np.isfinite(self.positions).all():
            raise ValueError("positions contain non-finite values")
        return True

    def assert_monodisperse(self) -> None:
        if not self.monodisperse():
            raise ValueError(
                f"expected monodisperse melt, got lengths "
                f"{sorted(set(self.chain_lengths))}"
            )

    # ------------------------------------------------------------------ #
    # Engine interop
    # ------------------------------------------------------------------ #
    def to_engine_args(self) -> Tuple[List[List[float]], List[List[int]], float]:
        """(positions, chains, box_size) as plain lists for the Rust engine."""
        return (
            self.positions.tolist(),
            [list(c) for c in self.chains],
            self.box_size,
        )

    def update_from_engine(self, engine: Any) -> None:
        """Pull positions/chains back from a Rust ``McEngine``."""
        self.positions = np.asarray(engine.positions(), dtype=float)
        self.chains = [list(map(int, c)) for c in engine.chains()]

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, meta: Optional[Dict[str, Any]] = None) -> Path:
        """Save to ``<path>.npz`` + ``<path>.json`` metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            positions=self.positions,
            chains=np.array(
                [np.asarray(c, dtype=np.int64) for c in self.chains],
                dtype=object,
            ),
            box_size=np.array([self.box_size]),
        )
        info: Dict[str, Any] = {
            "n_beads": self.n_beads,
            "n_chains": self.n_chains,
            "chain_lengths": self.chain_lengths,
            "box_size": self.box_size,
            "monodisperse": self.monodisperse(),
        }
        if meta:
            info["meta"] = meta
        path.with_suffix(".json").write_text(json.dumps(info, indent=2))
        logger.info(f"Saved MeltState to {path.with_suffix('.npz')}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MeltState":
        """Load from a ``.npz`` written by :meth:`save`."""
        path = Path(path)
        npz = np.load(path.with_suffix(".npz"), allow_pickle=True)
        positions = np.asarray(npz["positions"], dtype=float)
        chains = [list(map(int, c)) for c in npz["chains"]]
        box_size = float(npz["box_size"][0])
        return cls(positions, chains, box_size)
