#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versioned inorganic force-field (parameter) adapters.

An InorganicForceField maps the chemical *roles* assigned by a material's
chemistry adapter (e.g. silica "Si"/"OB"/"OH"/"HO") onto moltemplate atom
types, masses, charges and 12-6 Lennard-Jones parameters, and records what
the parameter set may be used for:

- which materials it is defined for,
- the combining rule it was parameterized with (explicit cross terms are
  generated with it; see compatibility.py),
- the allowed motion treatments (built-in sets are nonbonded-only, so
  only "frozen" and "rigid" are permitted -- never "flexible"),
- the O-H (or other cap) bond length used when terminating surfaces,
- citations and provenance.

Applying an adapter to an AtomicStructure yields a ParameterAssignment
(the "typed structure"); the structure itself is never modified, so one
geometry can be typed under several compatible force fields.

Created on 2026-11-13
@author: zwu
"""
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...core.exceptions import ValidationError

#: Motion treatments a finite/periodic inorganic component can request.
MOTIONS = ("frozen", "rigid", "flexible")

#: Combining rules for explicit cross interactions.
COMBINING_RULES = ("arithmetic", "geometric", "host")


@dataclass(frozen=True)
class RoleParameters:
    """Nonbonded parameters of one chemical role."""
    type_name: str
    mass: float
    charge: float
    epsilon: float
    sigma: float

    @staticmethod
    def from_table(entry: Mapping) -> "RoleParameters":
        return RoleParameters(
            type_name=str(entry["type"]),
            mass=float(entry["mass"]),
            charge=float(entry["charge"]),
            epsilon=float(entry["epsilon"]),
            sigma=float(entry["sigma"]),
        )


@dataclass
class ParameterAssignment:
    """
    Force-field parameters attached to an AtomicStructure (typed structure).

    Attributes:
        force_field: Adapter name (e.g. "interface_silica").
        version: Adapter version string.
        roles: Role per atom (copied from the structure).
        type_names: Moltemplate @atom type per atom.
        charges: (N,) per-atom charges (e).
        type_masses: type -> mass (amu).
        type_lj: type -> (epsilon kcal/mol, sigma A).
        combining_rule: Rule for explicit cross terms.
        motions: Motion treatments the parameters are valid for.
        notes: Free-form provenance notes (overrides applied, ...).
    """
    force_field: str
    version: str
    roles: List[str]
    type_names: List[str]
    charges: np.ndarray
    type_masses: Dict[str, float]
    type_lj: Dict[str, Tuple[float, float]]
    combining_rule: str = "host"
    motions: Tuple[str, ...] = ("frozen", "rigid")
    notes: List[str] = field(default_factory=list)

    @property
    def net_charge(self) -> float:
        return float(np.sum(self.charges))

    def per_atom_lj(self) -> List[Tuple[float, float]]:
        return [self.type_lj[t] for t in self.type_names]

    def type_table(self) -> Dict[str, Dict[str, float]]:
        """type -> {mass, epsilon, sigma, charges...} summary (sorted)."""
        out = {}
        for t in sorted(self.type_lj):
            qs = sorted({round(float(q), 6) for q, tn in
                         zip(self.charges, self.type_names) if tn == t})
            eps, sig = self.type_lj[t]
            out[t] = {"mass": self.type_masses[t], "epsilon": eps,
                      "sigma": sig, "charges": qs,
                      "count": sum(tn == t for tn in self.type_names)}
        return out


@dataclass
class InorganicForceField:
    """
    One named, versioned inorganic parameter set (see module docstring).

    Attributes:
        name: Registry name (e.g. "interface_silica").
        version: Parameter-set version.
        label: Short human-readable label.
        material_ids: Material IDs the set is defined for.
        roles: role -> RoleParameters.
        combining_rule: "arithmetic" (Lorentz-Berthelot), "geometric", or
            "host" (follow the organic host force field's rule).
        motions: Allowed motion treatments.
        cap_bond_length: Cap (e.g. O-H) bond length used for termination.
        aliases: Alternative names (e.g. "interface").
        citations: Literature references.
        notes: Provenance and usage notes.
    """
    name: str
    version: str
    label: str
    material_ids: Tuple[str, ...]
    roles: Dict[str, RoleParameters]
    combining_rule: str = "host"
    motions: Tuple[str, ...] = ("frozen", "rigid")
    cap_bond_length: Optional[float] = None
    aliases: Tuple[str, ...] = ()
    citations: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.combining_rule not in COMBINING_RULES:
            raise ValidationError(
                f"Inorganic force field '{self.name}': unknown combining "
                f"rule '{self.combining_rule}'"
            )
        bad = [m for m in self.motions if m not in MOTIONS]
        if bad:
            raise ValidationError(
                f"Inorganic force field '{self.name}': unknown motions {bad}"
            )

    @property
    def role_names(self) -> Tuple[str, ...]:
        return tuple(self.roles)

    def with_overrides(
        self,
        types: Optional[Mapping[str, str]] = None,
        charges: Optional[Mapping[str, float]] = None,
        lj: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> Dict[str, RoleParameters]:
        """Role table with per-role overrides applied (unknown roles ignored)."""
        table = dict(self.roles)
        for role, params in list(table.items()):
            type_name, q = params.type_name, params.charge
            eps, sig = params.epsilon, params.sigma
            if types and role in types:
                type_name = str(types[role])
            if charges and role in charges:
                q = float(charges[role])
            if lj and role in lj:
                eps, sig = float(lj[role][0]), float(lj[role][1])
            table[role] = RoleParameters(type_name, params.mass, q, eps, sig)
        return table

    def assign(
        self,
        roles: Sequence[str],
        table: Optional[Mapping[str, RoleParameters]] = None,
        notes: Optional[List[str]] = None,
    ) -> ParameterAssignment:
        """
        Type a structure from its per-atom roles.

        Args:
            roles: Chemical role per atom.
            table: Optional role table (e.g. from with_overrides()).

        Raises:
            ValidationError: If a role is not covered by the table, or two
                roles map one type name onto different parameters.
        """
        table = dict(table or self.roles)
        missing = sorted({r for r in roles if r not in table})
        if missing:
            raise ValidationError(
                f"Inorganic force field '{self.name}' has no parameters for "
                f"role(s) {missing}; covered roles: {sorted(table)}"
            )
        type_masses: Dict[str, float] = {}
        type_lj: Dict[str, Tuple[float, float]] = {}
        for role in dict.fromkeys(roles):
            p = table[role]
            lj = (p.epsilon, p.sigma)
            if p.type_name in type_lj and (
                    type_lj[p.type_name] != lj
                    or type_masses[p.type_name] != p.mass):
                raise ValidationError(
                    f"Inorganic force field '{self.name}': type "
                    f"'{p.type_name}' is used by roles with different "
                    "mass/LJ parameters"
                )
            type_masses[p.type_name] = p.mass
            type_lj[p.type_name] = lj
        return ParameterAssignment(
            force_field=self.name,
            version=self.version,
            roles=list(roles),
            type_names=[table[r].type_name for r in roles],
            charges=np.array([table[r].charge for r in roles], dtype=float),
            type_masses=type_masses,
            type_lj=type_lj,
            combining_rule=self.combining_rule,
            motions=self.motions,
            notes=list(notes or []),
        )


_FF_REGISTRY: Dict[str, InorganicForceField] = {}
_FF_ALIASES: Dict[Tuple[str, str], str] = {}


def register_inorganic_ff(ff: InorganicForceField) -> None:
    """Register an inorganic force-field adapter (and its aliases)."""
    if ff.name in _FF_REGISTRY:
        raise ValidationError(
            f"Inorganic force field '{ff.name}' is already registered"
        )
    _FF_REGISTRY[ff.name] = ff
    for material_id in ff.material_ids:
        for alias in ff.aliases:
            _FF_ALIASES[(material_id, alias.lower())] = ff.name


def get_inorganic_ff(name: str,
                     material_id: Optional[str] = None) -> InorganicForceField:
    """
    Resolve an inorganic force field by registry name, or by alias in the
    context of a material (e.g. ("interface", "sio2") -> interface_silica).

    Raises:
        ValidationError: Unknown name, or the set is not defined for the
            material.
    """
    key = name
    if key not in _FF_REGISTRY and material_id is not None:
        key = _FF_ALIASES.get((material_id, name.lower()), name)
    if key not in _FF_REGISTRY:
        available = (
            sorted(n for n, ff in _FF_REGISTRY.items()
                   if material_id in ff.material_ids)
            if material_id else sorted(_FF_REGISTRY)
        )
        raise ValidationError(
            f"Unknown inorganic force field '{name}'"
            + (f" for material '{material_id}'" if material_id else "")
            + f"; available: {available}"
        )
    ff = _FF_REGISTRY[key]
    if material_id is not None and material_id not in ff.material_ids:
        raise ValidationError(
            f"Inorganic force field '{ff.name}' is not defined for material "
            f"'{material_id}' (defined for {list(ff.material_ids)})"
        )
    return ff


def registered_inorganic_ffs(material_id: Optional[str] = None) -> List[str]:
    """Sorted adapter names (optionally only those for one material)."""
    return sorted(
        n for n, ff in _FF_REGISTRY.items()
        if material_id is None or material_id in ff.material_ids
    )
