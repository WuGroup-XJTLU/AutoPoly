#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base machinery for built-in crystalline substrate slab builders.

SlabBuilder implements the carve-and-functionalize pipeline shared by all
surface builders (following mBuild's SilicaInterface recipe, mosdef-hub/
mbuild, BSD-3; adapted to crystalline bulks generated in code):

1. Replicate a periodic surface cell to cover the requested lateral area
   and thickness.
2. Cleave the slab: the z-shift is chosen automatically so that the cut
   runs along the natural cleavage plane (fewest broken bonds) while
   keeping every Si tetrahedrally coordinated.
3. Bond perception by Si-O distance cutoff (lateral minimum image).
4. Strip disconnected fragments (largest connected component).
5. Optionally form Si-O-Si bridges between dangling top-face oxygens to
   reduce the silanol density toward a target (only where the surface
   geometry allows it).
6. Cap remaining dangling oxygens with H (top and, by default, bottom).
7. Type atoms by role and emit a self-contained moltemplate .lt class.

Since the inorganic-framework refactoring this module is a stable public
facade over the shared ``AutoPoly.materials`` foundation: steps 1-6 are
``materials.chemistry.silica.build_silica_slab`` (force-field agnostic,
returns an AtomicStructure), step 7 is a silica force-field adapter
(``materials.forcefields.silica``) plus the shared moltemplate writer
(``AutoPoly.exporters.moltemplate``).  Constructor arguments, defaults,
``build()`` and the generated .lt files are unchanged.

Subclasses provide the surface cell (fractional coordinates + lattice
constants) via class attributes.

Created on 2026-08-12
@author: zwu
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.exceptions import ValidationError
from ..core.system import logger
from ..exporters.moltemplate import mean_bond_lengths, write_inorganic_lt
from ..materials.chemistry import silica as silica_chem
from ..materials.chemistry.silica import (  # noqa: F401  (public constants)
    BRIDGE_SI_SI_MAX,
    ROLE_MASSES,
    SI_O_BOND_CUTOFF,
    Z_MARGIN,
)
from ..materials.forcefields.registry import (
    InorganicForceField,
    ParameterAssignment,
    RoleParameters,
)
from ..materials.forcefields.silica import SLAB_FF_ADAPTERS
from ..materials.structure import AtomicStructure
from .ff_tables import OH_BOND_LENGTH, SLAB_FF_REGISTRY, SLAB_ROLES


@dataclass
class SlabAtom:
    """One atom of a built slab."""
    name: str          # unique atom name (e.g. "si_3", "oh_7")
    role: str          # "Si", "OB", "OH" or "HO"
    type_name: str     # moltemplate @atom type
    charge: float
    position: np.ndarray
    epsilon: float = 0.0   # LJ epsilon (kcal/mol) of type_name
    sigma: float = 0.0     # LJ sigma (A) of type_name

    # Backwards-compatible private aliases (pre-refactoring attribute names)
    @property
    def _epsilon(self) -> float:
        return self.epsilon

    @_epsilon.setter
    def _epsilon(self, value: float) -> None:
        self.epsilon = value

    @property
    def _sigma(self) -> float:
        return self.sigma

    @_sigma.setter
    def _sigma(self, value: float) -> None:
        self.sigma = value


@dataclass
class SurfaceSlab:
    """
    A built crystalline substrate slab.

    Attributes:
        class_name: Moltemplate class name (single instance per system).
        lt_filename: File name the slab .lt is written to.
        lx, ly: Lateral slab dimensions (A); integer multiples of the
                surface cell.  The simulation box lateral sides must equal
                these values (seamless periodic slab).
        thickness: Requested slab envelope thickness (A).  The physical
                   slab (including hydroxyl caps) is guaranteed to fit
                   inside [0, thickness].
        z_extent: Actual physical z-extent of the slab atoms (A).
        atoms: Slab atoms (roles/types/charges/positions), coordinates
               centered laterally and at the slab mid-plane in z.
        n_silanol_top / n_silanol_bottom: silanol counts per face.
        slab_ff: Force-field table name used ("interface"/"clayff"/"custom").
        pair_substyle: LAMMPS pair sub-style prefix for the pair_coeff
                lines (e.g. "lj/charmm/coul/long"); required when the
                film force field uses a hybrid pair style, must be None
                otherwise.  Set automatically by BoxPacker from the film
                force field.
        bond_substyle: same for the bond_coeff lines ("harmonic" when
                the film force field uses a hybrid bond style).
        bonds: Si-O / O-H bond topology (atom index pairs).  Emitted
                with zero-force-coefficient harmonic terms: the slab is
                meant to be frozen, so the bonds add no forces but make
                special_bonds exclude the intra-slab 1-2/1-3/1-4
                nonbonded interactions (otherwise bonded O-H pairs at
                0.945 A would contribute enormous spurious LJ/Coulomb
                terms).
        structure: The force-field-agnostic AtomicStructure the slab was
                typed from (None for slabs constructed by hand).
        assignment: The ParameterAssignment (types/charges/LJ) applied.
    """
    class_name: str
    lt_filename: str
    lx: float
    ly: float
    thickness: float
    z_extent: float
    atoms: List[SlabAtom]
    n_silanol_top: int
    n_silanol_bottom: int
    slab_ff: str
    pair_substyle: Optional[str] = None
    bond_substyle: Optional[str] = None
    bonds: Optional[List[Tuple[int, int]]] = None
    structure: Optional[AtomicStructure] = None
    assignment: Optional[ParameterAssignment] = None

    # ------------------------------------------------------------------
    def lt_text(self) -> str:
        """The self-contained moltemplate .lt text of the slab."""
        from ..exporters.moltemplate import inorganic_lt_text
        return inorganic_lt_text(**self._lt_arguments())

    def write_lt(self, directory) -> Path:
        """Write the slab as a self-contained moltemplate .lt file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.lt_filename
        write_inorganic_lt(path, **self._lt_arguments())
        logger.info(f"Slab written to {path} ({len(self.atoms)} atoms)")
        return path

    def _lt_arguments(self) -> dict:
        header = [
            "crystalline substrate slab built by AutoPoly",
            f"lateral {self.lx:.4f} x {self.ly:.4f} A, envelope "
            f"thickness {self.thickness:.2f} A (physical z-extent "
            f"{self.z_extent:.2f} A)",
            f"silanols: top {self.n_silanol_top}, "
            f"bottom {self.n_silanol_bottom}; slab_ff='{self.slab_ff}'",
            "self-contained: masses and pair coefficients included;",
            "no bonded terms (slab is meant to be held rigid/frozen)",
        ]
        type_masses = {a.type_name: ROLE_MASSES[a.role] for a in self.atoms}
        type_lj = {a.type_name: (a.epsilon, a.sigma) for a in self.atoms}
        positions = np.array([a.position for a in self.atoms])
        args = dict(
            class_name=self.class_name,
            labels=[a.name for a in self.atoms],
            type_names=[a.type_name for a in self.atoms],
            charges=[a.charge for a in self.atoms],
            positions=positions,
            type_masses=type_masses,
            type_lj=type_lj,
            header=header,
            pair_substyle=self.pair_substyle,
            bond_substyle=self.bond_substyle,
        )
        if self.bonds:
            # Bond types: Si-O ("*_si_o") and O-H ("*_o_h")
            prefix = self.class_name.lower()
            bond_types = [
                f"{prefix}_" + silica_chem.role_bond_type(
                    self.atoms[i].role, self.atoms[j].role)
                for i, j in self.bonds
            ]
            order = sorted(set(bond_types), reverse=True)  # si_o before o_h
            # equilibrium length: mean length of bonds of this type
            # (lateral minimum image: bonds crossing the slab's periodic
            # edges would otherwise appear stretched)
            r0 = mean_bond_lengths(positions, self.bonds, bond_types, order,
                                   lx=self.lx, ly=self.ly)
            args.update(
                bonds=self.bonds,
                bond_types=bond_types,
                bond_coeffs={t: (0.0, r0[t]) for t in order},
            )
        return args


def _mass_table(atoms) -> Dict[str, float]:
    return {a.type_name: ROLE_MASSES[a.role] for a in atoms}


def _lj_table(atoms) -> Dict[str, Tuple[float, float]]:
    return {a.type_name: (a.epsilon, a.sigma) for a in atoms}


class SlabBuilder:
    """
    Base class for crystalline substrate slab builders.

    Class attributes provided by subclasses:
        CELL_FRACTIONAL: ((elem, (fx, fy, fz)), ...) surface-cell atoms.
        CELL_A / CELL_B: lateral lattice constants (A).
        CELL_C: surface-cell z period (A).
        BUILDER_NAME: registry name (e.g. "alpha_quartz").
        DEFAULT_CLASS_NAME / DEFAULT_LT_FILENAME: moltemplate identifiers.

    Args:
        lx_target: Requested lateral x size (A); snapped to a multiple of
                   CELL_A.
        ly_target: Requested lateral y size (A); snapped to a multiple of
                   CELL_B.
        thickness: Slab envelope thickness (A).  The physical slab
                   including hydroxyl caps fits inside this extent; the Si
                   core occupies thickness - ~5.4 A.
        oh_density: Target top-surface silanol density (sites/nm^2).  The
                    builder attempts to reduce the intrinsic dangling-O
                    density to the target by forming Si-O-Si bridges; if
                    the surface geometry does not allow enough bridges
                    (or the target exceeds the intrinsic density), all
                    dangling oxygens are hydroxylated and a warning is
                    logged.
        hydroxylate_bottom: Cap bottom-face dangling oxygens with H
                    (default True; both faces are exposed in the
                    fully-periodic slab model).
        slab_ff: "interface", "clayff", or "custom".
        slab_types / slab_charges / slab_lj: per-role overrides of the
                    built-in tables (roles "Si", "OB", "OH", "HO"); all
                    three are required when slab_ff="custom".
        seed: RNG seed for the bridge pairing.
        pair_substyle: LAMMPS pair sub-style for the pair_coeff lines
                    (e.g. "lj/charmm/coul/long"); required when the film
                    force field uses a hybrid pair style.  Normally left
                    to BoxPacker, which sets it from the film force field.
        class_name / lt_filename: moltemplate identifiers for the slab.
    """

    CELL_FRACTIONAL: Tuple = ()
    CELL_A: float = 0.0
    CELL_B: float = 0.0
    CELL_C: float = 0.0
    BUILDER_NAME: str = "base"
    ATOM_PREFIX: str = "slab"
    DEFAULT_CLASS_NAME: str = "SurfaceSlab"
    DEFAULT_LT_FILENAME: str = "surface_slab.lt"

    def __init__(
        self,
        lx_target: float,
        ly_target: float,
        thickness: float,
        oh_density: float = 4.6,
        hydroxylate_bottom: bool = True,
        slab_ff: str = "interface",
        slab_types: Optional[Dict[str, str]] = None,
        slab_charges: Optional[Dict[str, float]] = None,
        slab_lj: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: int = 42,
        pair_substyle: Optional[str] = None,
        bond_substyle: Optional[str] = None,
        class_name: Optional[str] = None,
        lt_filename: Optional[str] = None,
    ) -> None:
        self.lx_target = float(lx_target)
        self.ly_target = float(ly_target)
        self.thickness = float(thickness)
        self.oh_density = float(oh_density)
        self.hydroxylate_bottom = bool(hydroxylate_bottom)
        self.slab_ff = slab_ff
        self.seed = seed
        self.pair_substyle = pair_substyle
        self.bond_substyle = bond_substyle
        self.class_name = class_name or self.DEFAULT_CLASS_NAME
        self.lt_filename = lt_filename or self.DEFAULT_LT_FILENAME
        self._table = self._resolve_table(
            slab_ff, slab_types, slab_charges, slab_lj
        )
        self._adapter, self._role_table = self._resolve_adapter(
            slab_ff, self._table
        )
        self._validate()

    # ------------------------------------------------------------------
    # Validation / parameter resolution
    # ------------------------------------------------------------------
    @property
    def min_thickness(self) -> float:
        return 2.0 * Z_MARGIN + self.CELL_C / 2.0

    @staticmethod
    def _resolve_table(slab_ff, slab_types, slab_charges, slab_lj):
        """Role table (role -> type/charge/epsilon/sigma) incl. overrides."""
        if slab_ff == "custom":
            missing = [
                r for r in SLAB_ROLES
                if not (slab_types and slab_types.get(r)
                        and slab_charges and r in slab_charges
                        and slab_lj and r in slab_lj)
            ]
            if missing:
                raise ValidationError(
                    f"slab_ff='custom' requires slab_types, slab_charges "
                    f"and slab_lj covering all roles {SLAB_ROLES}; "
                    f"missing: {missing}"
                )
            return {
                r: {
                    "type": slab_types[r],
                    "charge": float(slab_charges[r]),
                    "epsilon": float(slab_lj[r][0]),
                    "sigma": float(slab_lj[r][1]),
                }
                for r in SLAB_ROLES
            }
        if slab_ff not in SLAB_FF_REGISTRY:
            raise ValidationError(
                f"Unknown slab_ff '{slab_ff}'; choose from "
                f"{sorted(SLAB_FF_REGISTRY)} or 'custom'"
            )
        base = {r: dict(e) for r, e in SLAB_FF_REGISTRY[slab_ff].items()}
        # per-role overrides on top of a built-in table
        for role, tname in (slab_types or {}).items():
            if role in base:
                base[role]["type"] = tname
        for role, q in (slab_charges or {}).items():
            if role in base:
                base[role]["charge"] = float(q)
        for role, lj in (slab_lj or {}).items():
            if role in base:
                base[role]["epsilon"], base[role]["sigma"] = (
                    float(lj[0]), float(lj[1])
                )
        return base

    @staticmethod
    def _resolve_adapter(slab_ff, table):
        """(adapter, role table) for the resolved per-role table."""
        role_table = {
            r: RoleParameters(
                type_name=str(e["type"]),
                mass=ROLE_MASSES[r],
                charge=float(e["charge"]),
                epsilon=float(e["epsilon"]),
                sigma=float(e["sigma"]),
            )
            for r, e in table.items()
        }
        adapter = SLAB_FF_ADAPTERS.get(slab_ff)
        if adapter is None:  # "custom": user-supplied, unvalidated table
            adapter = InorganicForceField(
                name="custom_silica",
                version="user",
                label="user-supplied silica role table",
                material_ids=("sio2",),
                roles=role_table,
                combining_rule="host",
                motions=("frozen", "rigid"),
                notes="slab_ff='custom' (not validated by AutoPoly)",
            )
        return adapter, role_table

    def _validate(self) -> None:
        if self.lx_target <= 0 or self.ly_target <= 0:
            raise ValidationError(
                f"{self.BUILDER_NAME} builder requires positive lateral "
                f"targets, got ({self.lx_target}, {self.ly_target})"
            )
        if self.thickness < self.min_thickness:
            raise ValidationError(
                f"Slab thickness must be >= {self.min_thickness:.1f} A "
                f"(envelope incl. hydroxyl coatings), got {self.thickness}"
            )
        if self.oh_density < 0:
            raise ValidationError(
                f"oh_density must be >= 0, got {self.oh_density}"
            )

    # ------------------------------------------------------------------
    # Public build
    # ------------------------------------------------------------------
    @property
    def oh_length(self) -> float:
        """O-H cap length for the selected slab force field (A)."""
        return OH_BOND_LENGTH.get(self.slab_ff, 0.98)

    def build_structure(self) -> AtomicStructure:
        """Steps 1-6: the force-field-agnostic hydroxylated slab."""
        return silica_chem.build_silica_slab(
            self.CELL_FRACTIONAL, self.CELL_A, self.CELL_B, self.CELL_C,
            self.lx_target, self.ly_target, self.thickness,
            oh_density=self.oh_density,
            hydroxylate_bottom=self.hydroxylate_bottom,
            seed=self.seed,
            oh_length=self.oh_length,
            name=self.BUILDER_NAME,
            phase=self.BUILDER_NAME,
        )

    def build(self) -> SurfaceSlab:
        structure = self.build_structure()
        return self.type_structure(structure)

    def type_structure(self, structure: AtomicStructure) -> SurfaceSlab:
        """Step 7: type a slab structure and wrap it as a SurfaceSlab."""
        assignment = self._adapter.assign(
            structure.roles, self._role_table,
            notes=[f"slab_ff={self.slab_ff}"],
        )
        atoms = self._assign_atoms(structure, assignment)
        total_q = assignment.net_charge
        if abs(total_q) > 1e-4:
            logger.warning(
                f"{self.BUILDER_NAME} slab net charge {total_q:+.4f} e "
                "(expected ~0 for a fully hydroxylated slab)"
            )
        meta = structure.metadata
        return SurfaceSlab(
            class_name=self.class_name,
            lt_filename=self.lt_filename,
            lx=float(meta["lx"]),
            ly=float(meta["ly"]),
            thickness=self.thickness,
            z_extent=float(structure.descriptors["z_extent"]),
            atoms=atoms,
            n_silanol_top=int(meta["n_silanol_top"]),
            n_silanol_bottom=int(meta["n_silanol_bottom"]),
            slab_ff=self.slab_ff,
            pair_substyle=self.pair_substyle,
            bond_substyle=self.bond_substyle,
            bonds=list(structure.bonds),
            structure=structure,
            assignment=assignment,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assign_atoms(self, structure: AtomicStructure,
                      assignment: ParameterAssignment) -> List[SlabAtom]:
        """SlabAtom records (names by role counter, types, charges, LJ)."""
        from ..exporters.moltemplate import role_labels
        names = role_labels(structure.roles, self.ATOM_PREFIX)
        atoms = []
        for name, role, tname, q, p in zip(
                names, structure.roles, assignment.type_names,
                assignment.charges, structure.positions):
            eps, sig = assignment.type_lj[tname]
            atoms.append(SlabAtom(
                name=name, role=role, type_name=tname, charge=float(q),
                position=np.array(p, dtype=float),
                epsilon=eps, sigma=sig,
            ))
        return atoms


def _coordination(n, bonds):
    return silica_chem.coordination(n, bonds)
