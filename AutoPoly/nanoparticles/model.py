#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nanoparticle: a first-class finite inorganic component.

A Nanoparticle is passed to generate() alongside Polymer/Molecule models.
It is *not* an RDKit macromolecule and *not* a planar substrate: its
geometry comes from the material's registered shape builder (or an
imported structure), its surface termination from the material's
chemistry adapter, and its interaction parameters -- attached only in
stage 2 -- from a registered inorganic force-field adapter.

    >>> from AutoPoly.nanoparticles import Nanoparticle, SilicaSurface
    >>> np_ = Nanoparticle(material="SiO2", structure="beta_cristobalite",
    ...                    shape="sphere", radius=12.0, count=2,
    ...                    surface=SilicaSurface(), motion="rigid")
    >>> generate(system, "ps_silica", [ps, np_], force_field="oplsaa")

Created on 2026-11-13
@author: zwu
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core.exceptions import ValidationError
from ..materials.material import Material
from ..materials.registry import MaterialSpec, get_material
from ..materials.structure import AtomicStructure
from .shapes import Imported, Shape, Surface, make_shape, make_surface

#: Motion treatments accepted by the nanocomposite pipeline.
NANOPARTICLE_MOTIONS = ("frozen", "rigid", "flexible")


def _component_id(name: str) -> str:
    cid = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")
    if not cid or not (cid[0].isalpha()):
        cid = f"np_{cid}"
    return cid


class Nanoparticle:
    """
    A nanoparticle species (``count`` identical instances).

    Args:
        material: Material ID/formula/alias ("SiO2", "Al2O3", "TiO2", "C",
            "Au", "Ag") or a Material object.
        structure: Phase / structure source (e.g. "alpha_quartz",
            "beta_cristobalite", "alpha_corundum", "anatase", "rutile",
            "graphene", "cnt", "fcc"); None = material default (or the
            Material's crystal).  ``phase`` is an alias.
        shape: "sphere", "sheet", "tube", "cuboctahedron", "octahedron",
            "wulff", or a Shape object (Sphere, Sheet, Tube, Polyhedron).
        count: Number of identical instances to place (>= 1).
        motion: "rigid" (default; one LAMMPS rigid body per instance),
            "frozen" (excluded from time integration), or "flexible"
            (only with a force field that provides validated bonded
            terms; none of the built-in sets do).
        surface: Termination ID or Surface object (e.g.
            SilicaSurface(oh_density=4.6)); None = material default.
        force_field: Inorganic force-field adapter name or alias (e.g.
            "interface", "clayff"); None = material default.
            ``atom_types`` is an alias.
        name: Component name (unique per system); default
            "<material>_<shape>".
        seed: Seed for stochastic surface steps (condensation pairing).
        allow_unqualified: Permit MD export although the exact combination
            is not in the qualification matrix (recorded as user-qualified).
        grafts: Optional Grafting specification (covalently grafted chains;
            silica, host OPLS-AA).
        **shape_kwargs: Shape size arguments (A) when ``shape`` is a name:
            radius, center_offset (sphere, polyhedra), length, width,
            layers (sheet), n, m, length (tube).
    """

    _is_inorganic_component = True

    def __init__(
        self,
        material: Union[str, Material] = "SiO2",
        structure: Optional[str] = None,
        shape: Union[str, Shape] = "sphere",
        count: int = 1,
        motion: str = "rigid",
        surface: Union[str, Surface, None] = None,
        force_field: Optional[str] = None,
        name: Optional[str] = None,
        seed: int = 42,
        allow_unqualified: bool = False,
        atom_types: Optional[str] = None,
        phase: Optional[str] = None,
        grafts: Optional[Any] = None,
        **shape_kwargs: Any,
    ) -> None:
        if isinstance(material, Material):
            spec = material.spec
            structure = structure or phase or material.crystal or material.phase
            force_field = force_field or atom_types or material.force_field
            if surface is None and material.surface is not None:
                surface = material.surface
        else:
            spec = get_material(material)
            structure = structure or phase
            force_field = force_field or atom_types
        self.material: MaterialSpec = spec
        self._imported: Optional[AtomicStructure] = None
        if isinstance(shape, Imported):
            self.shape = shape
        else:
            self.shape = make_shape(shape, **shape_kwargs)
        self.phase = (spec.resolve_phase(structure).name
                      if structure is not None or spec.default_phase
                      else None)
        self.surface: Surface = make_surface(surface)
        state = self.surface.state or spec.default_surface
        if state is not None:
            spec.resolve_surface(state)
        self.surface_state = state
        if not isinstance(count, int) or count < 1:
            raise ValidationError(
                f"Nanoparticle count must be a positive int, got {count!r}"
            )
        self.count = count
        if motion not in NANOPARTICLE_MOTIONS:
            raise ValidationError(
                f"Nanoparticle motion must be one of {NANOPARTICLE_MOTIONS}, "
                f"got '{motion}'"
            )
        self.motion = motion
        self.force_field = force_field or spec.default_force_field
        if self.force_field is not None and self.force_field != "custom":
            from ..materials.forcefields import get_inorganic_ff
            get_inorganic_ff(self.force_field, spec.id)  # validates name
        self.seed = int(seed)
        self.allow_unqualified = bool(allow_unqualified)
        self.component_id = _component_id(
            name or f"{spec.id}_{self.shape.name}")
        self._structure: Optional[AtomicStructure] = None
        self._validate_shape_for_phase()
        self.grafts = grafts
        self._graft_plan = None
        if grafts is not None:
            if spec.id != "sio2" or self.surface_state != "hydroxylated":
                raise ValidationError(
                    "Grafting is implemented for hydroxylated silica "
                    f"particles (silyl-ether linkage), not {spec.id} / "
                    f"'{self.surface_state}'"
                )
            if motion == "flexible":
                raise ValidationError("Grafted particles must be rigid or "
                                      "frozen")

    # ------------------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        material: Union[str, Material] = "SiO2",
        name: Optional[str] = None,
        count: int = 1,
        motion: str = "rigid",
        surface=None,
        force_field: Optional[str] = None,
        atom_types: Optional[str] = None,
        structure: Optional[str] = None,
        perceive_bonds: Optional[bool] = None,
        allow_unqualified: bool = False,
        center: bool = True,
    ) -> "Nanoparticle":
        """
        Nanoparticle from a validated atomistic cluster file.

        Supported formats: ``.json`` (AtomicStructure dict, as written by
        Nanoparticle.write_structure), ``.pdb`` with CONECT records, and
        ``.xyz``.  XYZ files carry no bonds: pass ``perceive_bonds=True``
        to perceive them with the material's distance cutoffs (the result
        is then validated against the material's valence rules).  Charges
        and parameters always come from the inorganic force-field adapter
        through the roles the chemistry adapter assigns from connectivity.

        Args:
            path: Structure file.
            material: Material of the cluster.
            structure: Phase label (default: "amorphous" for silica if
                registered, else the material default).
            center: Translate the cluster so its centroid is at the origin.
        """
        from .io import read_structure

        spec = material.spec if isinstance(material, Material) \
            else get_material(material)
        s = read_structure(path, spec, perceive_bonds=perceive_bonds)
        if center:
            s.translate(-s.positions.mean(axis=0))
        phase = structure
        if phase is None:
            phase = "amorphous" if "amorphous" in spec.phases \
                else spec.default_phase
        obj = cls(
            material=material, structure=phase,
            shape=Imported(source=str(path)), count=count, motion=motion,
            surface=surface, force_field=force_field, atom_types=atom_types,
            name=name or Path(path).stem, allow_unqualified=allow_unqualified,
        )
        obj._imported = s
        return obj

    @classmethod
    def from_structure(cls, structure: AtomicStructure,
                       material: Union[str, Material] = "SiO2",
                       **kwargs) -> "Nanoparticle":
        """Nanoparticle from an in-memory AtomicStructure (validated)."""
        from .io import prepare_imported

        spec = material.spec if isinstance(material, Material) \
            else get_material(material)
        s = prepare_imported(structure.copy(), spec, perceive_bonds=None)
        phase = kwargs.pop("structure", None) or (
            "amorphous" if "amorphous" in spec.phases else spec.default_phase)
        obj = cls(material=material, structure=phase,
                  shape=Imported(source="in-memory"), **kwargs)
        obj._imported = s
        return obj

    # ------------------------------------------------------------------
    def _validate_shape_for_phase(self) -> None:
        if isinstance(self.shape, Imported):
            return
        spec = self.material
        phase = spec.phases.get(self.phase) if self.phase else None
        shape_name = self.shape.name if self.shape.name != "polyhedron" \
            else self.shape.kind
        if phase is None or shape_name not in phase.shapes:
            supported = {p.name: list(p.shapes) for p in spec.phases.values()}
            raise ValidationError(
                f"No built-in '{shape_name}' builder for {spec.id} phase "
                f"'{self.phase}'. Supported (phase: shapes): {supported}; "
                "imported structures: Nanoparticle.from_file()"
            )
        lo, hi = spec.radius_range
        size = getattr(self.shape, "radius", None) or \
            getattr(self.shape, "size", None)
        if size is not None and not (lo <= size <= hi):
            raise ValidationError(
                f"{spec.id} particle size {size} A outside the supported "
                f"range [{lo}, {hi}] A"
            )

    @property
    def geometry_type(self) -> str:
        return self.shape.geometry_type

    @property
    def cap_length(self) -> Optional[float]:
        """Termination cap (O-H / C-H) length from the selected adapter."""
        if self.force_field in (None, "custom"):
            return None
        from ..materials.forcefields import get_inorganic_ff
        return get_inorganic_ff(self.force_field, self.material.id).cap_bond_length

    def build_structure(self) -> AtomicStructure:
        """The force-field-agnostic structure (built once, cached)."""
        if self._structure is not None:
            return self._structure
        if self._imported is not None:
            s = self._imported
        else:
            builder = self.material.builder(
                self.shape.name if self.shape.name != "polyhedron"
                else self.shape.kind)
            s = builder(
                material=self.material,
                phase=self.material.phases[self.phase],
                shape=self.shape,
                surface=self.surface,
                surface_state=self.surface_state,
                seed=self.seed,
                cap_length=self.cap_length,
            )
        s.metadata.setdefault("material_id", self.material.id)
        s.metadata["phase"] = self.phase
        s.metadata.setdefault("termination", self.surface_state)
        self._structure = s
        return s

    # ------------------------------------------------------------------
    # Grafting
    # ------------------------------------------------------------------
    def graft_plan(self, exclude=()) -> Optional[Dict[str, Any]]:
        """Sites + anchor structure of the grafted particle (cached)."""
        if self.grafts is None:
            return None
        if self._graft_plan is None or exclude:
            from ..materials.chemistry.silica import (
                graft_anchor_structure,
                select_graft_sites,
            )
            s = self.build_structure()
            area = float(s.descriptors.get("accessible_area_nm2") or 0.0)
            n = self.grafts.n_grafts(area)
            sites = select_graft_sites(s, n, seed=self.grafts.seed,
                                       exclude=exclude)
            core, anchors = graft_anchor_structure(s, sites)
            # anchor index in the core -> original silanol O index
            kept = [i for i in range(s.n_atoms)
                    if not (s.elements[i] == "H" and any(
                        o in sites for o in s.neighbors()[i]))]
            original = {new: old for new, old in enumerate(kept)}
            self._graft_plan = {"n": n, "area_nm2": area, "core": core,
                                "anchors": anchors, "sites": sites,
                                "original": original,
                                "excluded": list(exclude)}
        return self._graft_plan

    def graft_models(self) -> List[Any]:
        """Polymer models of the grafted chains (built in stage 1)."""
        plan = self.graft_plan()
        if plan is None:
            return []
        return [self.grafts.chain_model(plan["n"])]

    def _grafted_entry(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        from ..pipeline.unit_coords import placement_from_transform
        from .grafting import grow_grafts, head_and_cap

        plan = self.graft_plan()
        chains = [c for c in geometry["chains"]
                  if c.get("graft_of") == self.component_id]
        if len(chains) != plan["n"]:
            raise ValidationError(
                f"Expected {plan['n']} graft chains for '{self.component_id}', "
                f"geometry has {len(chains)}"
            )
        if any(len(c["placements"]) < 2 for c in chains):
            raise ValidationError("Grafted chains need at least 2 monomers")
        variants = geometry["variants"]
        first = variants[chains[0]["placements"][0]["variant"]]
        tpl0 = {"elements": [a["element"] for a in first["atoms"]],
                "bonds": first["bonds"]}
        head, cap = head_and_cap(tpl0)
        from .grafting import GraftGrowthError
        excluded = list(plan.get("excluded", []))
        for _ in range(8):
            try:
                transforms = grow_grafts(plan["core"], plan["anchors"],
                                         chains, variants, self.grafts, cap)
                break
            except GraftGrowthError as e:
                # crowded site: exclude it and re-select the sites
                excluded.append(plan["original"][e.site])
                plan = self.graft_plan(exclude=excluded)
        else:
            raise ValidationError(
                f"Grafting '{self.component_id}': {len(excluded)} sites were "
                "too crowded to grow chains from; lower the grafting density "
                "or use shorter grafts"
            )
        for chain, tr in zip(chains, transforms):
            chain["placements"] = [
                placement_from_transform(p["variant"], R, t)
                for p, (R, t) in zip(chain["placements"], tr)]
            first_name = chain["placements"][0]["variant"]
            variants[first_name]["graft_cap_atom"] = cap
        return {
            "linker": self.grafts.linker,
            "n": plan["n"],
            "density_target": self.grafts.density,
            "density_achieved": plan["n"] / plan["area_nm2"]
            if plan["area_nm2"] else None,
            "anchors": [{"o": int(o), "si": int(info["si"])}
                        for o, info in plan["anchors"].items()],
            "excluded_sites": [int(x) for x in excluded],
            "chain_ids": [c["id"] for c in chains],
            "head_atom": int(head),
            "cap_atom": int(cap),
        }

    # ------------------------------------------------------------------
    def geometry_entry(self, component_id: Optional[str] = None,
                       geometry: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        """The geometry.json inorganic entry of this species."""
        grafts = None
        if self.grafts is not None:
            if geometry is None:
                raise ValidationError("Grafted particles need the stage-1 "
                                      "geometry (graft chains)")
            grafts = self._grafted_entry(geometry)
            s = self._graft_plan["core"]
        else:
            s = self.build_structure()
        entry = {
            "id": self.component_id,
            "component": "nanoparticle",
            "role": "filler",
            "count": self.count,
            "motion": self.motion,
            "material": self.material.id,
            "phase": self.phase,
            "surface": self.surface_state,
            "surface_params": dict(self.surface.params),
            "geometry_type": s.geometry_type,
            "shape": _jsonable_shape(self.shape),
            "force_field": {"name": self.force_field},
            "allow_unqualified": self.allow_unqualified,
            "structure": s.to_dict(),
        }
        if grafts is not None:
            entry["grafts"] = grafts
        return entry

    def summary(self) -> Dict[str, Any]:
        """Composition, size and surface report of one instance."""
        s = self.build_structure()
        d = s.descriptors
        return {
            "id": self.component_id,
            "material": self.material.id,
            "phase": self.phase,
            "shape": _jsonable_shape(self.shape),
            "count": self.count,
            "motion": self.motion,
            "force_field": self.force_field,
            "n_atoms": s.n_atoms,
            "formula": s.formula(),
            "mass_amu": s.total_mass(),
            "envelope": d.get("envelope"),
            "core_volume_A3": d.get("core_volume"),
            "accessible_area_nm2": d.get("accessible_area_nm2"),
            "termination": s.metadata.get("termination"),
            "metadata": {k: v for k, v in s.metadata.items()},
        }

    def write_structure(self, path: Union[str, Path]) -> Path:
        """Write the structure (.json AtomicStructure, .xyz or .pdb)."""
        from .io import write_structure
        return write_structure(self.build_structure(), path)

    def __repr__(self) -> str:
        return (f"Nanoparticle({self.material.id}, phase={self.phase!r}, "
                f"shape={self.shape!r}, count={self.count}, "
                f"motion={self.motion!r}, force_field={self.force_field!r})")


def _jsonable_shape(shape: Shape) -> Dict[str, Any]:
    out = {"type": type(shape).__name__}
    for k, v in shape.__dict__.items():
        if isinstance(v, tuple):
            v = list(v)
        if isinstance(v, dict):
            v = {str(key): val for key, val in v.items()}
        out[k] = v
    return out
