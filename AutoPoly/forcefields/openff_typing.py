#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenFF (SMIRNOFF) typing support — optional force field backend.

This module parameterizes whole molecules/polymer chains with the OpenFF
toolkit (SMIRNOFF direct chemical perception, e.g. the Sage force field) and
assigns partial charges (NAGL GNN by default, AM1-BCC or Gasteiger as
fallbacks). Unlike AutoPoly's six SMARTS-table force fields, SMIRNOFF assigns
parameters per molecule rather than per fixed atom type, so the result is a
per-system parameter table plus explicit per-term assignments:

- :class:`OpenFFTyper` parameterizes an RDKit molecule and accumulates every
  parameter actually used; :meth:`OpenFFTyper.write_force_field_lt` renders
  those parameters as a moltemplate ``openff.lt`` file (LAMMPS ``real`` units).
- ``pipeline/typing.py`` consumes :class:`ParameterizedMol` to write variant
  and chain ``.lt`` files with explicit ``Data Bonds``/``Data Angles``/
  ``Data Dihedrals``/``Data Impropers`` sections (no "By Type" inference).

Unit/convention mappings (verified against the SMIRNOFF spec and the LAMMPS
docs):

- bonds/angles: SMIRNOFF uses the ``k/2`` convention -> LAMMPS ``harmonic``
  gets ``K = k/2`` (kcal/mol/A^2 or kcal/mol/rad^2).
- proper torsions -> ``dihedral_style fourier``: leading term count ``m``,
  then ``(K, n, d)`` triplets with ``K = k`` unchanged and phases in degrees.
- improper torsions: SMIRNOFF lists the central atom SECOND and averages over
  the three trefoil orderings (``idivf=3``); LAMMPS ``cvff`` lists the central
  atom first, so each match is emitted as three entries with
  ``K = k/idivf``, ``d = +-1`` (phase 0 or 180 degrees only — exact for Sage,
  whose impropers are all periodicity 2 / phase 180; anything else raises).
- vdW: sigma/epsilon taken directly (``rmin_half`` is converted when present);
  1-4 scalings and cutoffs are read from the loaded handlers rather than
  hardcoded.

Both OpenFF packages are optional dependencies and are imported lazily: using
``force_field="openff"`` without them raises a GenerationError with install
instructions. NAGL additionally requires Python <= 3.12 (DGL dependency).

Created on 2026-09-23
@author: zwu
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from ..core.exceptions import GenerationError
from ..core.system import logger

# Newest-first candidates for the default SMIRNOFF force field (Sage family);
# the first one bundled with the installed openff-toolkit wins.
PREFERRED_OFFXML = (
    "openff-2.3.0.offxml",
    "openff-2.2.1.offxml",
    "openff-2.1.1.offxml",
    "openff-2.0.0.offxml",
)

# Newest-first NAGL charge model candidates (downloaded/shipped by the
# openff-nagl-models package).
NAGL_MODEL_CANDIDATES = (
    "openff-gnn-am1bcc-1.0.0.pt",
    "openff-gnn-am1bcc-0.1.0-rc.3.pt",
    "openff-gnn-am1bcc-0.1.0-rc.2.pt",
)

VALID_CHARGE_METHODS = ("nagl", "am1bcc", "gasteiger")

OPENFF_INSTALL_HINT = (
    "force_field='openff' requires the optional OpenFF packages. Install with "
    "'conda install -c conda-forge openff-toolkit openff-nagl' (recommended; "
    "openff-nagl needs Python <= 3.12 because of its DGL dependency) or "
    "'pip install openff-toolkit openff-nagl'. On Python 3.13 you can still "
    "use charge_method='gasteiger' with openff-toolkit alone."
)


@dataclass
class ParameterizedMol:
    """
    SMIRNOFF assignment for one molecule/chain, in RDKit atom indexing.

    Attributes:
        n_atoms: Number of atoms (== rdmol.GetNumAtoms()).
        vdw_ids: SMIRNOFF vdW parameter id per atom index.
        charges: Partial charge per atom index (elementary charge).
        bonds: (i, j, param_id) per bond.
        angles: (i, j, k, param_id) per angle.
        propers: (i, j, k, l, param_id) per proper torsion.
        impropers: (i, j, k, l, param_id) per improper torsion, in SMIRNOFF
            order (the CENTRAL atom is the second tuple element).
        formal_charge: Total formal charge of the molecule.
    """
    n_atoms: int
    vdw_ids: List[str]
    charges: List[float]
    bonds: List[Tuple[int, int, str]]
    angles: List[Tuple[int, int, int, str]]
    propers: List[Tuple[int, int, int, int, str]]
    impropers: List[Tuple[int, int, int, int, str]]
    formal_charge: int = 0


def sanitize_param_id(param_id: str) -> str:
    """Make a SMIRNOFF parameter id safe for moltemplate @token names."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in str(param_id))


class OpenFFTyper:
    """
    Parameterize molecules with an OpenFF SMIRNOFF force field.

    Accumulates every parameter used across ``parameterize()`` calls; the
    collected tables are rendered as a per-system ``openff.lt`` by
    ``write_force_field_lt()``.

    Args:
        offxml: SMIRNOFF force field file/name. None (default) tries the
            newest Sage bundled with the installed toolkit.
        charge_method: "nagl" (default, GNN, needs openff-nagl), "am1bcc"
            (AmberTools backend) or "gasteiger" (RDKit backend).
        nagl_model: Explicit NAGL model file name; None tries known models
            newest-first.
        neutralize: Uniformly shift charges so each parameterized molecule's
            total equals its formal charge (default True).
    """

    def __init__(
        self,
        offxml: Optional[str] = None,
        charge_method: str = "nagl",
        nagl_model: Optional[str] = None,
        neutralize: bool = True,
    ) -> None:
        if charge_method not in VALID_CHARGE_METHODS:
            raise GenerationError(
                f"Invalid OpenFF charge_method '{charge_method}'. "
                f"Must be one of: {list(VALID_CHARGE_METHODS)}"
            )
        self.charge_method = charge_method
        self.nagl_model = nagl_model
        self.neutralize = neutralize
        self.nagl_model_used: Optional[str] = None
        self.offxml_name: str = ""  # set by _load_force_field

        self._tk, self._unit = self._import_toolkit()
        self.force_field = self._load_force_field(offxml)

        # Electrostatics/vdW global settings, read back from the handlers so
        # custom offxml files are honored (do not hardcode Sage values).
        vdw = self.force_field.get_parameter_handler("vdW")
        elec = self.force_field.get_parameter_handler("Electrostatics")
        self.vdw_scale14 = float(getattr(vdw, "scale14", 0.5))
        self.coul_scale14 = float(getattr(elec, "scale14", 1.0 / 1.2))
        self._init_cutoffs(vdw, elec)

        # Parameter tables accumulated across all parameterized molecules.
        self._vdw: Dict[str, Tuple[float, float, float]] = {}  # id -> (eps, sigma, mass)
        self._bonds: Dict[str, Tuple[float, float]] = {}       # id -> (K, r0)
        self._angles: Dict[str, Tuple[float, float]] = {}      # id -> (K, theta0)
        self._propers: Dict[str, List[Tuple[float, int, float]]] = {}
        self._impropers: Dict[str, Tuple[float, int, int]] = {}  # id -> (K, d, n)

        logger.info(
            f"OpenFF typing initialized: {self.offxml_name}, "
            f"charge_method={self.charge_method}"
        )

    # ------------------------------------------------------------------
    # Imports and force field loading
    # ------------------------------------------------------------------
    @staticmethod
    def _import_toolkit():
        """Lazy-import the openff toolkit; clear error when not installed."""
        try:
            import openff.toolkit as tk
            from openff.units import unit
        except ImportError as e:
            raise GenerationError(
                f"Could not import openff-toolkit ({e}). {OPENFF_INSTALL_HINT}"
            ) from e
        return tk, unit

    def _load_force_field(self, offxml: Optional[str]):
        """Load the requested offxml, or the newest bundled Sage."""
        if offxml is not None:
            try:
                self.offxml_name = str(offxml)
                return self._tk.ForceField(offxml)
            except Exception as e:
                raise GenerationError(
                    f"Could not load OpenFF force field '{offxml}': {e}"
                ) from e
        last_err = None
        for candidate in PREFERRED_OFFXML:
            try:
                ff = self._tk.ForceField(candidate)
                self.offxml_name = candidate
                return ff
            except Exception as e:  # not bundled with this toolkit version
                last_err = e
        raise GenerationError(
            f"No default Sage offxml ({', '.join(PREFERRED_OFFXML)}) loadable "
            f"by the installed openff-toolkit: {last_err}"
        )

    def _init_cutoffs(self, vdw: Any, elec: Any) -> None:
        """Resolve pair style and cutoffs from the vdW/Electrostatics handlers."""
        unit = self._unit
        # Periodic-system treatment; SMIRNOFF 0.3 handlers expose
        # periodic_method ("cutoff" for Sage). LJPME/PME-style methods are not
        # representable by the simple LAMMPS pair styles written here.
        method = str(getattr(vdw, "periodic_method", "cutoff")).lower()
        if method not in ("cutoff",):
            raise GenerationError(
                f"OpenFF vdW periodic method '{method}' is not supported by "
                "the AutoPoly LAMMPS writer (only 'cutoff')."
            )
        cutoff = vdw.cutoff.m_as(unit.angstrom) if getattr(vdw, "cutoff", None) is not None else 9.0
        switch = getattr(vdw, "switch_width", None)
        switch = switch.m_as(unit.angstrom) if switch is not None else 0.0
        coul_cut = getattr(elec, "cutoff", None)
        coul_cut = coul_cut.m_as(unit.angstrom) if coul_cut is not None else cutoff
        self.vdw_cutoff = float(cutoff)
        self.vdw_switch = float(switch)
        self.coul_cutoff = float(coul_cut)
        # With a switching window, lj/charmm switches LJ from inner to outer.
        self.lj_inner = self.vdw_cutoff - self.vdw_switch if self.vdw_switch > 0 else self.vdw_cutoff
        self.lj_outer = self.vdw_cutoff
        self.pair_substyle = (
            "lj/charmm/coul/long" if self.vdw_switch > 0 else "lj/cut/coul/long"
        )

    # ------------------------------------------------------------------
    # Parameterization
    # ------------------------------------------------------------------
    def parameterize(self, rdmol: Chem.Mol) -> ParameterizedMol:
        """
        Assign SMIRNOFF parameters and partial charges to an RDKit molecule.

        The molecule must be sanitized with explicit hydrogens. Atom indices
        of the returned assignment match the RDKit atom indices (verified).

        Args:
            rdmol: RDKit Mol (explicit Hs; dummy atoms are NOT allowed).

        Returns:
            ParameterizedMol with per-atom types/charges and per-term
            parameter id assignments.
        """
        tk = self._tk
        for atom in rdmol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                raise GenerationError(
                    "OpenFF parameterization does not accept dummy atoms "
                    "(check the molecule/chain construction)"
                )

        offmol = tk.Molecule.from_rdkit(
            rdmol, allow_undefined_stereo=True, hydrogens_are_explicit=True
        )
        self._assert_index_alignment(rdmol, offmol)

        self._assign_charges(offmol)

        topology = tk.Topology.from_molecules([offmol])
        labels = self.force_field.label_molecules(topology)[0]

        n = rdmol.GetNumAtoms()
        vdw_ids = [""] * n
        for key, param in labels.get("vdW", {}).items():
            (i,) = key
            vdw_ids[i] = param.id
            self._register_vdw(param, rdmol.GetAtomWithIdx(i))
        if any(not v for v in vdw_ids):
            missing = [i for i, v in enumerate(vdw_ids) if not v]
            raise GenerationError(
                f"OpenFF vdW typing left {len(missing)} atoms unassigned "
                f"(indices {missing[:10]}...) — chemistry not covered by "
                f"{self.offxml_name}?"
            )

        charges = [float(q) for q in offmol.partial_charges.m_as(self._unit.elementary_charge)]
        formal = int(sum(a.formal_charge.m_as(self._unit.elementary_charge) for a in offmol.atoms))
        if self.neutralize:
            charges = self._neutralize(charges, formal)

        bonds = []
        for (i, j), param in sorted(labels.get("Bonds", {}).items()):
            bonds.append((i, j, param.id))
            self._register_bond(param)

        angles = []
        for (i, j, k), param in sorted(labels.get("Angles", {}).items()):
            angles.append((i, j, k, param.id))
            self._register_angle(param)

        propers = []
        for (i, j, k, l), param in sorted(labels.get("ProperTorsions", {}).items()):
            propers.append((i, j, k, l, param.id))
            self._register_proper(param)

        impropers = []
        for (i, j, k, l), param in sorted(labels.get("ImproperTorsions", {}).items()):
            # Defensive: SMIRNOFF lists the central atom second. Verify and
            # adapt loudly rather than emitting a silently-wrong trefoil.
            central_ok = all(
                rdmol.GetBondBetweenAtoms(j, x) is not None for x in (i, k, l)
            )
            if not central_ok:
                raise GenerationError(
                    f"OpenFF improper {(i, j, k, l)}: atom {j} is not the "
                    "central atom (bonded to the other three); the toolkit's "
                    "ordering convention is not the expected SMIRNOFF one."
                )
            impropers.append((i, j, k, l, param.id))
            self._register_improper(param)

        return ParameterizedMol(
            n_atoms=n,
            vdw_ids=vdw_ids,
            charges=charges,
            bonds=bonds,
            angles=angles,
            propers=propers,
            impropers=impropers,
            formal_charge=formal,
        )

    @staticmethod
    def _assert_index_alignment(rdmol: Chem.Mol, offmol: Any) -> None:
        """Guard the RDKit<->OpenFF atom index correspondence."""
        if rdmol.GetNumAtoms() != offmol.n_atoms:
            raise GenerationError(
                "OpenFF conversion changed the atom count "
                f"({rdmol.GetNumAtoms()} -> {offmol.n_atoms})"
            )
        for i, atom in enumerate(rdmol.GetAtoms()):
            if atom.GetAtomicNum() != offmol.atom(i).atomic_number:
                raise GenerationError(
                    f"OpenFF conversion changed atom ordering at index {i}; "
                    "cannot map SMIRNOFF assignments back to the RDKit molecule"
                )

    def _assign_charges(self, offmol: Any) -> None:
        """Assign partial charges with the configured backend."""
        if self.charge_method == "nagl":
            self._assign_nagl_charges(offmol)
            return
        method = self.charge_method  # "am1bcc" or "gasteiger"
        try:
            offmol.assign_partial_charges(partial_charge_method=method)
        except Exception as e:
            raise GenerationError(
                f"OpenFF charge assignment with method '{method}' failed: {e}. "
                "am1bcc needs AmberTools; gasteiger only needs RDKit."
            ) from e

    def _assign_nagl_charges(self, offmol: Any) -> None:
        """NAGL GNN charges via the toolkit's NAGLToolkitWrapper."""
        try:
            try:
                from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
            except ImportError:
                from openff.toolkit.utils.toolkits import NAGLToolkitWrapper
            wrapper = NAGLToolkitWrapper()
        except Exception as e:
            raise GenerationError(
                f"NAGL charge backend unavailable ({e}). {OPENFF_INSTALL_HINT}"
            ) from e

        models = [self.nagl_model] if self.nagl_model else list(NAGL_MODEL_CANDIDATES)
        last_err = None
        for model in models:
            try:
                offmol.assign_partial_charges(
                    partial_charge_method=model, toolkit_registry=wrapper
                )
                self.nagl_model_used = model
                return
            except Exception as e:
                last_err = e
        raise GenerationError(
            f"NAGL charge assignment failed for models {models}: {last_err}. "
            "Install openff-nagl-models or pick charge_method='am1bcc'/'gasteiger'."
        )

    def _neutralize(self, charges: List[float], formal: int) -> List[float]:
        """Uniform shift so the total charge equals the formal charge."""
        total = float(np.sum(charges))
        drift = total - formal
        if abs(drift) > 1e-6 and charges:
            shift = drift / len(charges)
            logger.info(
                f"OpenFF charges: shifting by {-shift:+.3e} e/atom to reach "
                f"formal charge {formal} (raw total {total:+.4f})"
            )
            charges = [q - shift for q in charges]
        return charges

    # ------------------------------------------------------------------
    # Parameter table registration (SMIRNOFF -> LAMMPS real-unit conversion)
    # ------------------------------------------------------------------
    def _register_vdw(self, param: Any, rdatom: Chem.Atom) -> None:
        unit = self._unit
        pid = sanitize_param_id(param.id)
        eps = param.epsilon.m_as(unit.kilocalories_per_mole)
        sigma = getattr(param, "sigma", None)
        if sigma is not None:
            sig = sigma.m_as(unit.angstrom)
        else:  # rmin_half convention: rmin = 2^(1/6) * sigma
            sig = 2.0 * param.rmin_half.m_as(unit.angstrom) / (2.0 ** (1.0 / 6.0))
        mass = Chem.GetPeriodicTable().GetAtomicWeight(rdatom.GetAtomicNum())
        entry = (float(eps), float(sig), float(mass))
        if pid in self._vdw and not np.allclose(self._vdw[pid], entry, rtol=1e-6):
            raise GenerationError(
                f"Conflicting vdW parameters registered under id '{pid}'"
            )
        self._vdw[pid] = entry

    def _register_bond(self, param: Any) -> None:
        unit = self._unit
        pid = sanitize_param_id(param.id)
        # SMIRNOFF E = (k/2)(r - r0)^2 -> LAMMPS harmonic K = k/2
        k = param.k.m_as(unit.kilocalories_per_mole / unit.angstrom**2) / 2.0
        r0 = param.length.m_as(unit.angstrom)
        self._bonds.setdefault(pid, (float(k), float(r0)))

    def _register_angle(self, param: Any) -> None:
        unit = self._unit
        pid = sanitize_param_id(param.id)
        # SMIRNOFF E = (k/2)(theta - theta0)^2 -> LAMMPS harmonic K = k/2
        k = param.k.m_as(unit.kilocalories_per_mole / unit.radian**2) / 2.0
        theta0 = param.angle.m_as(unit.degree)
        self._angles.setdefault(pid, (float(k), float(theta0)))

    def _register_proper(self, param: Any) -> None:
        unit = self._unit
        pid = sanitize_param_id(param.id)
        terms = []
        ks = param.k if isinstance(param.k, (list, tuple)) else [param.k]
        ns = param.periodicity if isinstance(param.periodicity, (list, tuple)) else [param.periodicity]
        ps = param.phase if isinstance(param.phase, (list, tuple)) else [param.phase]
        idivfs = getattr(param, "idivf", None)
        if not isinstance(idivfs, (list, tuple)):
            idivfs = [idivfs] * len(ks)
        for k, n, p, idivf in zip(ks, ns, ps, idivfs):
            # ProperTorsions: idivf "auto"/None resolves to 1 (OFF-EP 10).
            div = 1.0 if idivf in (None, "auto") else float(idivf)
            terms.append((
                float(k.m_as(unit.kilocalories_per_mole)) / div,
                int(n),
                float(p.m_as(unit.degree)),
            ))
        self._propers.setdefault(pid, terms)

    def _register_improper(self, param: Any) -> None:
        unit = self._unit
        pid = sanitize_param_id(param.id)
        ks = param.k if isinstance(param.k, (list, tuple)) else [param.k]
        ns = param.periodicity if isinstance(param.periodicity, (list, tuple)) else [param.periodicity]
        ps = param.phase if isinstance(param.phase, (list, tuple)) else [param.phase]
        if len(ks) != 1:
            raise GenerationError(
                f"OpenFF improper '{param.id}' has {len(ks)} terms; only "
                "single-term impropers are supported (LAMMPS cvff)."
            )
        phase = float(ps[0].m_as(unit.degree)) % 360.0
        if abs(phase) < 1e-3 or abs(phase - 360.0) < 1e-3:
            d = 1
        elif abs(phase - 180.0) < 1e-3:
            d = -1
        else:
            raise GenerationError(
                f"OpenFF improper '{param.id}' has phase {phase} deg; only "
                "0/180 are representable with LAMMPS improper_style cvff."
            )
        # idivf: SMIRNOFF averages over the trefoil ("auto" -> 3 terms).
        idivf = getattr(param, "idivf", None)
        if isinstance(idivf, (list, tuple)):
            idivf = idivf[0] if idivf else None
        div = 3.0 if idivf in (None, "auto") else float(idivf)
        k = float(ks[0].m_as(unit.kilocalories_per_mole)) / div
        self._impropers.setdefault(pid, (k, d, int(ns[0])))

    # ------------------------------------------------------------------
    # openff.lt rendering
    # ------------------------------------------------------------------
    def atom_type_name(self, vdw_id: str) -> str:
        """Moltemplate atom type for a SMIRNOFF vdW parameter id."""
        return f"@atom:of_{sanitize_param_id(vdw_id)}"

    def bond_type_name(self, param_id: str) -> str:
        return f"@bond:ofb_{sanitize_param_id(param_id)}"

    def angle_type_name(self, param_id: str) -> str:
        return f"@angle:ofa_{sanitize_param_id(param_id)}"

    def dihedral_type_name(self, param_id: str) -> str:
        return f"@dihedral:ofd_{sanitize_param_id(param_id)}"

    def improper_type_name(self, param_id: str) -> str:
        return f"@improper:ofi_{sanitize_param_id(param_id)}"

    def write_force_field_lt(self, path) -> None:
        """
        Render the accumulated parameters as a per-system ``openff.lt``.

        Contains no "By Type" sections on purpose: every bonded term is
        written explicitly by the caller's molecule files, keyed by the
        @bond/@angle/@dihedral/@improper type names defined here.
        """
        path = Path(path)
        charge_desc = self.charge_method
        if self.charge_method == "nagl" and getattr(self, "nagl_model_used", None):
            charge_desc = f"nagl ({self.nagl_model_used})"

        with open(path, "w") as f:
            f.write("# OPENFF (SMIRNOFF) force field — generated by AutoPoly\n")
            f.write(f"# Source force field: {self.offxml_name}\n")
            f.write(f"# Charges: {charge_desc}\n")
            f.write("# All bonded terms are assigned explicitly in the molecule\n")
            f.write("# files (no 'By Type' inference; direct chemical perception).\n\n")
            f.write("OPENFF {\n\n")

            f.write('  write_once("In Init") {\n')
            f.write("    units           real\n")
            f.write("    atom_style      full\n")
            f.write("    bond_style      hybrid harmonic\n")
            f.write("    angle_style     hybrid harmonic\n")
            f.write("    dihedral_style  hybrid fourier\n")
            f.write("    improper_style  hybrid cvff\n")
            f.write(
                f"    pair_style      hybrid {self.pair_substyle} "
                f"{self.lj_inner:.4f} {self.lj_outer:.4f} {self.coul_cutoff:.4f}\n"
            )
            f.write("    kspace_style    pppm 0.0001\n")
            f.write("    pair_modify     mix arithmetic\n")
            f.write(
                f"    special_bonds   lj 0.0 0.0 {self.vdw_scale14:.6f} "
                f"coul 0.0 0.0 {self.coul_scale14:.6f}\n"
            )
            f.write("  }  # In Init\n\n")

            f.write('  write_once("Data Masses") {\n')
            for pid in sorted(self._vdw):
                _, _, mass = self._vdw[pid]
                f.write(f"    {self.atom_type_name(pid)} {mass:.4f}\n")
            f.write("  }  # Data Masses\n\n")

            f.write('  write_once("In Settings") {\n')
            for pid in sorted(self._vdw):
                eps, sig, _ = self._vdw[pid]
                t = self.atom_type_name(pid)
                f.write(
                    f"    pair_coeff {t} {t} {self.pair_substyle} "
                    f"{eps:.6f} {sig:.6f}\n"
                )
            for pid in sorted(self._bonds):
                k, r0 = self._bonds[pid]
                f.write(
                    f"    bond_coeff {self.bond_type_name(pid)} harmonic "
                    f"{k:.6f} {r0:.6f}\n"
                )
            for pid in sorted(self._angles):
                k, t0 = self._angles[pid]
                f.write(
                    f"    angle_coeff {self.angle_type_name(pid)} harmonic "
                    f"{k:.6f} {t0:.4f}\n"
                )
            for pid in sorted(self._propers):
                terms = self._propers[pid]
                parts = " ".join(f"{k:.6f} {n} {d:.4f}" for k, n, d in terms)
                f.write(
                    f"    dihedral_coeff {self.dihedral_type_name(pid)} "
                    f"fourier {len(terms)} {parts}\n"
                )
            for pid in sorted(self._impropers):
                k, d, n = self._impropers[pid]
                f.write(
                    f"    improper_coeff {self.improper_type_name(pid)} "
                    f"cvff {k:.6f} {d} {n}\n"
                )
            f.write("  }  # In Settings\n\n")

            f.write("}  # OPENFF\n")

        logger.info(
            f"Wrote {path}: {len(self._vdw)} atom types, "
            f"{len(self._bonds)} bond / {len(self._angles)} angle / "
            f"{len(self._propers)} dihedral / {len(self._impropers)} improper "
            "parameter sets"
        )
