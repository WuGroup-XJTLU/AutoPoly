#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coarse-grained (CG) spherical fillers for the bead-spring backend.

A CG filler is a rigid (or frozen) particle made of LJ beads -- a solid
FCC cut ("fcc") or a Fibonacci surface shell ("shell") of one filler bead
type -- in the reduced LJ units of the bead-spring models:

    >>> from AutoPoly.models.bead_spring import BeadType
    >>> from AutoPoly.nanoparticles import CGFiller
    >>> bss = BeadSpringSystem("nc", system, bead_types=[BeadType("A"),
    ...                        BeadType("F", epsilon=1.0)], pair_style="lj",
    ...                        density=0.5)
    >>> bss.add_species(arch.linear([("A", 50)]), n_chains=40)
    >>> bss.add_filler(CGFiller(radius=4.0, count=2, bead_type="F"))
    >>> bss.generate(backend="direct")

Construction: fillers are placed first (random positions and
orientations, periodic minimum-image gap ``min_gap`` between particle
envelopes), then the chains grow by SAW around them (beads may not enter
the filler envelope, radius + half a filler bead).  The run script holds
the fillers during minimization, integrates each rigid filler as one body
(fix rigid/nvt molecule) or keeps frozen fillers in place, skips
intra-filler pair forces, and reports a matrix-only temperature.

CG fillers are generic LJ particles: their mapping onto a real material
(bead size, polymer-filler attraction epsilon, see
BeadSpringSystem.set_pair_coeff) is the user's modelling choice and is not
part of the atomistic material qualification matrix.

Created on 2026-11-13
@author: zwu
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

STRUCTURES = ("fcc", "shell")
MOTIONS = ("rigid", "frozen")


@dataclass
class CGFiller:
    """
    Coarse-grained spherical filler species (LJ units).

    Args:
        radius: Particle radius (sigma); every bead center lies within it.
        count: Number of particles.
        bead_type: Filler BeadType name; must be in the system's bead types
            and not used by any chain species.
        structure: "fcc" (solid FCC cut, default) or "shell" (beads on the
            sphere surface only).
        spacing: Nearest-neighbor bead distance (sigma).
        motion: "rigid" (one rigid body per particle) or "frozen".
        name: Species label.
    """
    radius: float
    count: int = 1
    bead_type: str = "F"
    structure: str = "fcc"
    spacing: float = 1.0
    motion: str = "rigid"
    name: str = "filler"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"CGFiller radius must be > 0, got {self.radius}")
        if self.count < 1:
            raise ValueError(f"CGFiller count must be >= 1, got {self.count}")
        if self.spacing <= 0:
            raise ValueError("CGFiller spacing must be > 0")
        if self.radius < self.spacing:
            raise ValueError(
                f"CGFiller radius ({self.radius}) is smaller than the bead "
                f"spacing ({self.spacing}): the particle would be a single "
                "bead; use a plain bead type (or a smaller spacing) instead")
        if self.structure not in STRUCTURES:
            raise ValueError(
                f"CGFiller structure must be one of {STRUCTURES}, "
                f"got '{self.structure}'")
        if self.motion not in MOTIONS:
            raise ValueError(
                f"CGFiller motion must be one of {MOTIONS}, got "
                f"'{self.motion}' (flexible CG fillers are not supported)")

    def bead_positions(self) -> np.ndarray:
        """(n, 3) bead centers of one particle, centered at the origin."""
        if self.structure == "fcc":
            pos = fcc_sphere(self.radius, self.spacing)
        else:
            pos = sphere_shell(self.radius, self.spacing)
        return pos

    @property
    def n_beads(self) -> int:
        return len(self.bead_positions())


def fcc_sphere(radius: float, spacing: float) -> np.ndarray:
    """FCC lattice sites (nearest-neighbor distance ``spacing``) within
    ``radius`` of a lattice site at the origin."""
    a = spacing * np.sqrt(2.0)
    n = int(np.ceil(radius / a)) + 1
    r = np.arange(-n, n + 1)
    cells = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    keep = np.linalg.norm(pos, axis=1) <= radius + 1e-9
    return pos[keep]


def sphere_shell(radius: float, spacing: float) -> np.ndarray:
    """Near-uniform beads on a sphere of ``radius``, about ``spacing``
    apart (hexagonal area per bead, Fibonacci lattice)."""
    from ..materials.topology import fibonacci_sphere
    area = 4.0 * np.pi * radius * radius
    n = max(1, int(round(area / (np.sqrt(3.0) / 2.0 * spacing * spacing))))
    return radius * fibonacci_sphere(n)


def _random_rotation() -> np.ndarray:
    """Uniform random rotation from numpy's global RNG (the bead-spring
    generators are seeded through np.random.seed)."""
    q = np.random.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class PlacedFiller:
    """One placed CG particle."""
    filler: CGFiller
    center: np.ndarray
    positions: np.ndarray      # world bead centers
    envelope: float            # outermost bead distance + half a bead


def place_fillers(fillers: Sequence[CGFiller], box_size: float,
                  bead_sigma: Dict[str, float], min_gap: float = 1.0,
                  max_attempts: int = 5000) -> List[PlacedFiller]:
    """
    Random sequential placement of all CG particles in the periodic cubic
    box [-L/2, L/2)^3 with envelope gaps >= ``min_gap`` (minimum image).

    Raises:
        ValueError: A particle does not fit the box or no free position
            was found (box too small for the requested count).
    """
    placed: List[PlacedFiller] = []
    for spec in fillers:
        local = spec.bead_positions()
        r_out = float(np.linalg.norm(local, axis=1).max())
        envelope = r_out + 0.5 * bead_sigma[spec.bead_type]
        if 2.0 * envelope + min_gap > box_size:
            raise ValueError(
                f"CG filler '{spec.name}' (envelope diameter "
                f"{2 * envelope:.2f} sigma + gap {min_gap}) does not fit the "
                f"periodic box ({box_size:.2f} sigma): use a larger box or a "
                "smaller radius")
        for k in range(spec.count):
            for _ in range(max_attempts):
                c = np.random.uniform(-box_size / 2, box_size / 2, 3)
                ok = True
                for other in placed:
                    d = c - other.center
                    d -= box_size * np.round(d / box_size)
                    if np.linalg.norm(d) < envelope + other.envelope + min_gap:
                        ok = False
                        break
                if ok:
                    break
            else:
                raise ValueError(
                    f"Could not place CG filler '{spec.name}' {k + 1}/"
                    f"{spec.count} without overlap after {max_attempts} "
                    "attempts: lower the filler count or enlarge the box")
            pos = local @ _random_rotation().T + c
            placed.append(PlacedFiller(spec, c, pos, envelope))
    return placed


def filler_volume(fillers: Sequence[CGFiller],
                  bead_sigma: Dict[str, float]) -> float:
    """Total envelope volume of all CG particles (sigma^3)."""
    v = 0.0
    for f in fillers:
        r = f.radius + 0.5 * bead_sigma[f.bead_type]
        v += f.count * 4.0 / 3.0 * np.pi * r ** 3
    return v


def motion_groups(fillers: Sequence[CGFiller]) -> Dict[str, List[str]]:
    """motion -> filler bead-type names (each type has one motion)."""
    by_type: Dict[str, str] = {}
    for f in fillers:
        prev = by_type.setdefault(f.bead_type, f.motion)
        if prev != f.motion:
            raise ValueError(
                f"CG filler bead type '{f.bead_type}' is used with motions "
                f"'{prev}' and '{f.motion}': give each motion its own bead "
                "type")
    groups: Dict[str, List[str]] = {}
    for t, m in by_type.items():
        groups.setdefault(m, []).append(t)
    return groups


# ----------------------------------------------------------------------
# LAMMPS run-script fragments (shared by the direct and moltemplate
# bead-spring backends)
# ----------------------------------------------------------------------
def group_lines(groups: Dict[str, List[str]]) -> List[str]:
    """LAMMPS group commands; type tokens are numeric ids (direct backend)
    or @atom: names (moltemplate In Settings)."""
    every = [t for m in MOTIONS for t in groups.get(m, [])]
    lines = [f"group fillers type {' '.join(every)}"]
    for m in MOTIONS:
        if m in groups:
            lines.append(f"group {m}_fillers type {' '.join(groups[m])}")
    lines.append("group matrix subtract all fillers")
    return lines


SETUP = (
    "# CG fillers: skip intra-particle pair forces; matrix-only temperature\n"
    "neigh_modify    exclude molecule/intra fillers\n"
    "compute         tmatrix matrix temp\n"
)
THERMO = "thermo_style    custom step temp c_tmatrix pe ke etotal press vol density\n"
HOLD = "fix             hold fillers setforce 0.0 0.0 0.0\n"
UNHOLD = "unfix           hold\n"


def production_lines(groups: Dict[str, List[str]]) -> str:
    """Integrators: matrix NVT, one rigid body per rigid particle, frozen
    particles at rest."""
    out = "fix             1 matrix nvt temp 1.0 1.0 0.1 tchain 3\n"
    if "rigid" in groups:
        # rigid/nvt, not rigid/nvt/small: the /small variant combined with
        # "neigh_modify exclude molecule/intra" aborts in LAMMPS 29Aug2024
        out += ("fix             2 rigid_fillers rigid/nvt molecule "
                "temp 1.0 1.0 1.0\n")
    if "frozen" in groups:
        out += ("velocity        frozen_fillers set 0.0 0.0 0.0\n"
                "fix             3 frozen_fillers setforce 0.0 0.0 0.0\n")
    return out


def compression_hint(groups: Dict[str, List[str]]) -> str:
    """Commented NPT block: the filler-aware SAW packs below melt density
    (periodic overlap checks), so the box is usually compressed first."""
    if "rigid" in groups:
        baro = ("# fix             2 rigid_fillers rigid/npt molecule temp 1.0 "
                "1.0 1.0 iso 0.0 5.0 10.0 dilate all\n")
        undo = "# unfix           2\n"
    else:
        baro = ("# fix             1 matrix npt temp 1.0 1.0 0.1 iso 0.0 5.0 "
                "1.0 dilate all\n")
        undo = "# unfix           1\n"
    return ("\n# (optional) compression toward melt density before production:\n"
            "# one barostat controls the box and dilates all beads (rigid\n"
            "# fillers by their centers of mass); P ~ 5 suits a Kremer-Grest\n"
            "# melt at T = 1. Uncomment and place before the production run.\n"
            + undo + baro + "# run             50000\n")


def summary(placed: Sequence[PlacedFiller], box_size: float) -> Dict:
    """Filler block of BeadSpringSystem.get_system_info()."""
    species: Dict[str, Dict] = {}
    for p in placed:
        s = species.setdefault(p.filler.name, {
            "count": 0, "radius": p.filler.radius,
            "structure": p.filler.structure, "bead_type": p.filler.bead_type,
            "motion": p.filler.motion, "beads_per_particle": len(p.positions),
            "envelope_radius": p.envelope})
        s["count"] += 1
    v = sum(4.0 / 3.0 * np.pi * p.envelope ** 3 for p in placed)
    return {"species": species, "n_particles": len(placed),
            "n_beads": int(sum(len(p.positions) for p in placed)),
            "volume_fraction": v / box_size ** 3 if box_size else None}
