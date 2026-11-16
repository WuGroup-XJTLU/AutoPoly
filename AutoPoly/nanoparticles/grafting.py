#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covalent polymer grafting onto nanoparticle surface sites.

    >>> from AutoPoly.nanoparticles import Nanoparticle, Grafting
    >>> brush = Grafting(polymer=Polymer(chain_num=1, sequence=PE10),
    ...                  density=0.8)                 # chains / nm^2
    >>> np_ = Nanoparticle(material="SiO2", radius=12.0, grafts=brush)

Model (silica, linker "silyl_ether"):

- Sites: N_g = round(sigma_g * A_accessible) silanol oxygens (or an
  explicit ``count``), spread by farthest-point sampling, at most one per
  silicon.
- Chemistry: Si-OH + H-C(head) -> Si-O-C(head): the silanol H and one H of
  the chain's head atom (the first heavy atom of the first monomer, which
  must carry H) are removed and the surface O bonds to the head atom.  The
  grafted O ("OG") keeps the silanol-O Lennard-Jones parameters and carries
  the silanol group charge q(OH) + q(HO) plus the charge of the removed
  organic H, so the total charge of particle + grafts is conserved exactly.
- Conformations: every graft grows monomer by monomer from its anchor with
  the obstacle-aware grower (packing.chain_regrowth): the head atom sits on
  the anchor's vacated O-H direction at the O-C bond length, every atom
  clears the core (``core_clearance``) and earlier grafts; grafted chains
  are part of the particle class, not free matrix molecules.
- Bonded linkage terms (host OPLS-AA only): bond O-C (OPLS CT-OS), angle
  Si-O-C (CT-OS-Si, wlj silyl ether), angles O-C-X (X-CT-OS) and
  dihedrals O-C-C-X (X-CT-CT-OS), with coefficients read from the bundled
  oplsaa.lt by the OPLS bonded classes of the typed atoms -- never
  invented.  Si-O-C-X torsions are not defined for silica-bound ethers in
  OPLS-AA and are omitted (documented: low barrier); terms whose atoms all
  belong to the rigid core are irrelevant and omitted.

Created on 2026-11-13
@author: zwu
"""
import copy
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import ValidationError
from ..core.system import logger

if TYPE_CHECKING:
    from ..materials.structure import AtomicStructure

#: O-C bond length of the silyl-ether linkage (A; OPLS CT-OS r0).
SILYL_ETHER_OC = 1.41
#: Supported linkers -> host force fields their bonded terms come from.
LINKERS = {"silyl_ether": ("oplsaa",)}


class GraftGrowthError(ValidationError):
    """A graft could not be grown from one surface site."""

    def __init__(self, message: str, site: int = -1) -> None:
        super().__init__(message)
        self.site = site


@dataclass
class Grafting:
    """
    Grafted-chain specification of a nanoparticle species.

    Args:
        polymer: Polymer model of one grafted chain (its ``chain_num`` is
            replaced by the number of grafts).  The head atom -- first heavy
            atom of the first monomer -- must carry at least one H.
        density: Target grafting density (chains/nm^2 of accessible area).
        count: Explicit number of grafts per particle (overrides density).
        linker: Linkage template ("silyl_ether").
        seed: Site-selection and growth seed.
        core_clearance: Minimum distance (A) of graft atoms from core atoms
            other than the anchor O and its Si.
        attempts: Independent anchored growth attempts per graft.
    """
    polymer: Any
    density: Optional[float] = None
    count: Optional[int] = None
    linker: str = "silyl_ether"
    seed: int = 0
    core_clearance: float = 2.0
    attempts: int = 60

    def __post_init__(self) -> None:
        if (self.density is None) == (self.count is None):
            raise ValidationError(
                "Grafting needs exactly one of density= (chains/nm^2) or "
                "count="
            )
        if self.density is not None and self.density <= 0:
            raise ValidationError("Grafting density must be > 0")
        if self.count is not None and self.count < 1:
            raise ValidationError("Grafting count must be >= 1")
        if self.linker not in LINKERS:
            raise ValidationError(
                f"Unknown linker '{self.linker}'; available: {sorted(LINKERS)}"
            )
        if getattr(self.polymer, "topology", "linear") != "linear":
            raise ValidationError("Grafted chains must be linear polymers")

    def n_grafts(self, area_nm2: float) -> int:
        if self.count is not None:
            return int(self.count)
        n = int(round(self.density * area_nm2))
        if n < 1:
            raise ValidationError(
                f"Grafting density {self.density}/nm^2 on {area_nm2:.2f} nm^2 "
                "gives no chain; raise the density or use count="
            )
        return n

    def chain_model(self, n: int):
        """Copy of the graft polymer with chain_num = n."""
        model = object.__new__(type(self.polymer))
        model.__dict__.update(copy.deepcopy(self.polymer.__dict__))
        model.chain_num = int(n)
        model.set_Sequence()
        return model


# ----------------------------------------------------------------------
# Stage 1: anchored growth
# ----------------------------------------------------------------------
def head_and_cap(template: Dict[str, Any]) -> Tuple[int, int]:
    """(head heavy-atom index, index of one H on it) of a first monomer."""
    el = template["elements"]
    head = next((i for i, e in enumerate(el) if e != "H"), None)
    if head is None:
        raise ValidationError("Graft head monomer has no heavy atom")
    for i, j in template["bonds"]:
        for a, b in ((i, j), (j, i)):
            if a == head and el[b] == "H":
                return head, b
    raise ValidationError(
        "The graft head atom (first heavy atom of the first monomer) carries "
        "no hydrogen; the silyl-ether linker needs an H to replace"
    )


def grow_grafts(core: "AtomicStructure", anchors: Dict[int, Dict],
                chains: List[Dict],
                variants: Dict[str, Any], grafting: Grafting,
                cap_index: int) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Anchored growth of one chain per anchor around a rigid core.

    Args:
        core: Core AtomicStructure (anchor O already without H).
        anchors: new-O index -> {"si", "direction", ...}.
        chains: geometry.json chain entries (one per anchor, same order).
        variants: geometry.json variants.
        cap_index: Head-H index (in the first monomer) removed at grafting.

    Returns:
        Per graft, the list of monomer transforms (R, t) in the particle
        frame.
    """
    from ..packing.chain_regrowth import ChainTemplate, ObstacleAwareGrower
    from ..packing.collision_geometry import PeriodicVoxelGrid
    from ..pipeline.unit_coords import chain_template_data

    pos = core.positions
    templates = [ChainTemplate(chain_template_data(c.get("id", "g"), c,
                                                   variants))
                 for c in chains]
    # finite cluster in a periodic grid large enough to never wrap
    reach = templates[0].n * 3.0 + 10.0
    half = float(np.linalg.norm(pos, axis=1).max()) + reach
    L = np.full(3, 2.0 * half)
    grid = PeriodicVoxelGrid(-0.5 * L, L, 0.3)
    exclude = set()
    for o, info in anchors.items():
        exclude.add(o)
        exclude.add(info["si"])
    mask = np.array([i not in exclude for i in range(core.n_atoms)])
    is_h = np.array([e == "H" for e in core.elements])
    grid.mark(pos[mask & ~is_h], grafting.core_clearance)
    grid.mark(pos[mask & is_h], min(grafting.core_clearance, 1.6))
    rng = np.random.default_rng(grafting.seed)
    grower = ObstacleAwareGrower(grid, L)
    out = []
    head = None
    for k, ((o, info), tpl) in enumerate(zip(anchors.items(), templates)):
        if head is None:
            head, cap = head_and_cap(tpl.info(0))
            if cap != cap_index:
                raise ValidationError("inconsistent graft cap atom")
        anchor = {"position": pos[o], "direction": info["direction"],
                  "length": SILYL_ETHER_OC, "head": head, "cap": cap_index,
                  # head substituents vs the anchor O (1-3) and Si (1-4)
                  "partners": ((pos[o], 1.85), (pos[info["si"]], 2.4)),
                  "partners_far": ((pos[o], 2.0), (pos[info["si"]], 2.4)),
                  "max_tilt": 15.0}
        # this graft must also clear every *other* anchor O / Si (they
        # are absent from the shared grid so their own grafts can attach)
        others = [x for oo, inf in anchors.items() if oo != o
                  for x in (oo, inf["si"])]
        grower.grid = grid.copy()
        if others:
            grower.grid.mark(pos[others], grafting.core_clearance)
        transforms = None
        for _ in range(grafting.attempts):
            transforms = grower.grow(tpl, rng, anchor=anchor)
            if transforms is not None:
                break
        if transforms is None:
            raise GraftGrowthError(
                f"Could not grow graft {k + 1}/{len(anchors)} from its surface "
                "site without overlaps (crowded brush or too short spacing): "
                "lower the grafting density, use shorter chains, or reduce "
                "core_clearance", site=o
            )
        world = np.vstack([tpl.local(m) @ R.T + t
                           for m, (R, t) in enumerate(transforms)])
        # mark this graft (minus the removed cap H) for the next ones
        cap_global = cap_index
        keep = np.ones(len(world), dtype=bool)
        keep[cap_global] = False
        els = np.concatenate([tpl.info(m)["elements"]
                              for m in range(tpl.n)])
        is_h = els == "H"
        grid.mark(world[keep & ~is_h], 1.7)
        grid.mark(world[keep & is_h], 1.2)
        out.append(transforms)
    logger.info(f"Grew {len(out)} grafted chains from surface anchors")
    return out


# ----------------------------------------------------------------------
# Stage 2: linkage terms (OPLS-AA)
# ----------------------------------------------------------------------
@lru_cache(maxsize=4)
def _opls_tables(ff_lt: str):
    """(type -> classes (b, a, d, i), coeff lines by kind) from oplsaa.lt."""
    text = Path(ff_lt).read_text()
    classes = {}
    for m in re.finditer(
            r"replace\{\s*@atom:(\S+)\s+@atom:\S+?_b(\S+?)_a(\S+?)_d(\S+?)"
            r"_i(\S+?)\s*\}", text):
        classes[m.group(1)] = (m.group(2), m.group(3), m.group(4),
                               m.group(5))
    coeffs = {"bond": {}, "angle": {}, "dihedral": {}}
    for kind in coeffs:
        for m in re.finditer(rf"^\s*{kind}_coeff @{kind}:(\S+)\s+([^#\n]+)",
                             text, re.M):
            coeffs[kind].setdefault(m.group(1), m.group(2).split())
    return classes, coeffs


def opls_coefficients(kind: str, names: Sequence[str], ff_lt: Path):
    """Coefficient tokens of an OPLS bonded term by class names (either
    direction); None if undefined."""
    _, coeffs = _opls_tables(str(ff_lt))
    for key in ("_".join(names), "_".join(reversed(names))):
        if key in coeffs[kind]:
            return coeffs[kind][key]
    return None


def opls_class(atom_type: str, slot: str, ff_lt: Path) -> str:
    """Bonded class ('b', 'a', 'd') of an OPLS @atom type (e.g. '81')."""
    classes, _ = _opls_tables(str(ff_lt))
    t = atom_type.split(":")[-1]
    if t not in classes:
        raise ValidationError(f"OPLS-AA type {atom_type} has no bonded classes")
    return classes[t]["bad".index(slot)]
