#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Manifest Contract for AutoPoly

This module defines the file-based contract between pipeline stage 2
(force-field typing) and stage 3 (box packing):

    UnitTyper  --writes-->  build/<ff>/units.json  --reads-->  BoxPacker

A "unit" is one packable entity: a single built polymer chain (each chain
has its own MC conformation, so chains are genuinely distinct units), a
small-molecule species with an instance count, or an inorganic component
(a built-in substrate slab or a nanoparticle species with an instance
count).

Packing strategies read ONLY this manifest (plus the referenced .lt files
by name, and the optional per-unit coordinate files); they never parse .lt
contents.

Unit contract (format_version 2): three independent axes

- kind:          "polymer" | "molecule" | "inorganic"   (what it is made of)
- role:          "film" | "substrate" | "filler"        (where it goes)
- geometry_type: "slab" | "cluster" | "sheet" | "rod" | "imported"
                 (inorganic only; a geometry descriptor)

``role`` is never interpreted as the chemical type.  Version-1 manifests
(no geometry_type/material/motion/coords_file fields) remain readable and
are migrated in memory by migrate_manifest().

Created on 2026-07-30
@author: zwu
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import ValidationError

FORMAT_VERSION = 2
MANIFEST_FILENAME = "units.json"

UNIT_KIND_POLYMER = "polymer"
UNIT_KIND_MOLECULE = "molecule"
UNIT_KIND_INORGANIC = "inorganic"
VALID_UNIT_KINDS = (UNIT_KIND_POLYMER, UNIT_KIND_MOLECULE, UNIT_KIND_INORGANIC)

UNIT_ROLE_FILM = "film"
UNIT_ROLE_SUBSTRATE = "substrate"
UNIT_ROLE_FILLER = "filler"
#: Chains covalently grafted onto a filler: instantiated by the filler's
#: composite class, never placed by a strategy.
UNIT_ROLE_GRAFT = "graft"
VALID_UNIT_ROLES = (UNIT_ROLE_FILM, UNIT_ROLE_SUBSTRATE, UNIT_ROLE_FILLER,
                    UNIT_ROLE_GRAFT)

#: Geometry types of inorganic units (descriptors, not chemistry).
VALID_GEOMETRY_TYPES = ("slab", "cluster", "sheet", "rod", "imported")
#: Motion treatments of inorganic units.
VALID_MOTIONS = ("frozen", "rigid", "flexible")


@dataclass
class UnitSpec:
    """
    One packable unit.

    Attributes:
        id: Unique unit identifier ("poly_1", "water", ...).
        kind: "polymer" (a built chain with its own .lt class) or
              "molecule" (a species instantiated `count` times).
        lt_file: .lt file (relative to the build dir) defining the class
                 that packing strategies instantiate.
        count: Number of instances to place (1 for polymer chains).
        topology: "linear" | "ring" | None (polymers only).
        n_monomers: Chain length in monomers (polymers only).
        radius: Bounding/collision radius in Angstrom used by placement.
        anchors: Head/tail atom references for future grafting strategies,
                 e.g. {"head": "monomer[0]/C1", "tail": "monomer[49]/C2"}.
                 Populated by UnitTyper; not consumed by current strategies.
        role: "film" (default), "substrate" or "filler". Substrate units
              are packed into the slab region by the "on_substrate"
              strategy; filler units are placed first by the
              "nanocomposite_mc" strategy.
        monomer_files: Constituent monomer .lt files this unit depends on
                       (used for force-field subsetting).
        lt_class: Moltemplate class defined by lt_file when it differs
                  from the file stem (built-in slabs: "QuartzSlab").
        geometry_type: Inorganic geometry descriptor (None for organic
                       units).
        motion: Inorganic motion treatment ("frozen"/"rigid"/"flexible").
        material: Inorganic identity and parameterization record
                  (material id, phase, surface, inorganic force field,
                  version, combining rule, qualification id, ...).
        envelope: Collision/loading descriptors (shape, radius, core
                  volume A^3, accessible area A^2, lateral dims, ...).
        mass: Mass of one instance (amu), when known.
        coords_file: File (relative to the build dir) with the unit's
                     class-local atom coordinates (element x y z, XYZ
                     format) used by atom-level packing strategies.
        chain_file: Polymer units: per-monomer template sidecar (JSON:
                    monomer variant sequence, local coordinates, bonds,
                    connection atoms, junction lengths) used by strategies
                    that regrow chains in place.
    """
    id: str
    kind: str
    lt_file: str
    count: int = 1
    topology: Optional[str] = None
    n_monomers: Optional[int] = None
    radius: float = 3.0
    anchors: Dict[str, str] = field(default_factory=dict)
    role: str = UNIT_ROLE_FILM
    monomer_files: List[str] = field(default_factory=list)
    lt_class: Optional[str] = None
    geometry_type: Optional[str] = None
    motion: Optional[str] = None
    material: Optional[Dict[str, Any]] = None
    envelope: Optional[Dict[str, Any]] = None
    mass: Optional[float] = None
    coords_file: Optional[str] = None
    chain_file: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_UNIT_KINDS:
            raise ValidationError(
                f"Invalid unit kind '{self.kind}' for unit '{self.id}'. "
                f"Must be one of: {list(VALID_UNIT_KINDS)}"
            )
        if self.role not in VALID_UNIT_ROLES:
            raise ValidationError(
                f"Invalid unit role '{self.role}' for unit '{self.id}'. "
                f"Must be one of: {list(VALID_UNIT_ROLES)}"
            )
        if self.count < 1:
            raise ValidationError(
                f"Unit '{self.id}' has invalid count {self.count} (must be >= 1)"
            )
        if self.kind == UNIT_KIND_INORGANIC:
            if self.geometry_type not in VALID_GEOMETRY_TYPES:
                raise ValidationError(
                    f"Inorganic unit '{self.id}' needs a geometry_type in "
                    f"{list(VALID_GEOMETRY_TYPES)}, got {self.geometry_type!r}"
                )
            if self.motion not in VALID_MOTIONS:
                raise ValidationError(
                    f"Inorganic unit '{self.id}' needs a motion in "
                    f"{list(VALID_MOTIONS)}, got {self.motion!r}"
                )
        elif self.role == UNIT_ROLE_FILLER:
            raise ValidationError(
                f"Unit '{self.id}': role 'filler' is reserved for inorganic "
                f"units (kind '{self.kind}')"
            )
        if self.role == UNIT_ROLE_GRAFT and self.kind != UNIT_KIND_POLYMER:
            raise ValidationError(
                f"Unit '{self.id}': role 'graft' is for polymer chains"
            )

    @property
    def class_name(self) -> str:
        """Moltemplate class name instantiated by strategies."""
        return self.lt_class or Path(self.lt_file).stem

    @property
    def is_inorganic(self) -> bool:
        return self.kind == UNIT_KIND_INORGANIC

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UnitSpec":
        known = {
            "id", "kind", "lt_file", "count", "topology", "n_monomers",
            "radius", "anchors", "role", "monomer_files", "lt_class",
            "geometry_type", "motion", "material", "envelope", "mass",
            "coords_file", "chain_file",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return UnitSpec(**filtered)


@dataclass
class UnitLibrary:
    """
    The units.json manifest: a collection of typed, packable units plus the
    monomer .lt files they depend on.

    Attributes:
        force_field: Force field the units were typed with.
        units: List of UnitSpec (one per built chain, one per molecule species).
        monomer_files: All typed monomer/molecule .lt files in the build dir
                       (drives force-field subsetting in stage 3).
        build_config: Build-time configuration echoed from stage 1/2
                      (offset, rotate, densities, ...) for strategy use.
        geometry_source: Relative path to the source geometry.json.
        format_version: Manifest format version.
    """
    force_field: str
    units: List[UnitSpec] = field(default_factory=list)
    monomer_files: List[str] = field(default_factory=list)
    build_config: Dict[str, Any] = field(default_factory=dict)
    geometry_source: Optional[str] = None
    format_version: int = FORMAT_VERSION
    # Runtime-only: directory holding the .lt files (not serialized).
    source_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "force_field": self.force_field,
            "geometry_source": self.geometry_source,
            "build_config": self.build_config,
            "monomer_files": list(self.monomer_files),
            "units": [u.to_dict() for u in self.units],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UnitLibrary":
        data = migrate_manifest(data)
        return UnitLibrary(
            force_field=data["force_field"],
            units=[UnitSpec.from_dict(u) for u in data.get("units", [])],
            monomer_files=list(data.get("monomer_files", [])),
            build_config=dict(data.get("build_config", {})),
            geometry_source=data.get("geometry_source"),
            format_version=int(data.get("format_version", FORMAT_VERSION)),
        )

    def save(self, directory: Union[str, Path]) -> Path:
        """Write units.json into `directory`. Returns the manifest path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_FILENAME
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def load(source: Union[str, Path]) -> "UnitLibrary":
        """
        Load a UnitLibrary from a build directory or a units.json path.

        Raises:
            ValidationError: If the manifest is missing, unreadable, or has
                             an unsupported format_version.
        """
        source = Path(source)
        path = source / MANIFEST_FILENAME if source.is_dir() else source
        if not path.is_file():
            raise ValidationError(f"units.json manifest not found: {path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid units.json at {path}: {e}") from e
        library = UnitLibrary.from_dict(data)
        if library.format_version > FORMAT_VERSION:
            raise ValidationError(
                f"Unsupported units.json format_version {library.format_version} "
                f"(this AutoPoly supports up to {FORMAT_VERSION})"
            )
        library.source_dir = str(path.parent)
        return library

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, directory: Union[str, Path]) -> None:
        """
        Validate the manifest against files on disk.

        Checks that every referenced .lt file (monomer files and per-unit
        lt files) exists in `directory`.

        Raises:
            ValidationError: Listing all missing files.
        """
        directory = Path(directory)
        missing = []
        referenced = list(self.monomer_files)
        for unit in self.units:
            referenced.append(unit.lt_file)
            referenced.extend(unit.monomer_files)
            if unit.coords_file:
                referenced.append(unit.coords_file)
            if unit.chain_file:
                referenced.append(unit.chain_file)
        for rel in dict.fromkeys(referenced):  # dedup, keep order
            if not (directory / rel).is_file():
                missing.append(rel)
        if missing:
            raise ValidationError(
                f"units.json references {len(missing)} missing file(s) in "
                f"{directory}: {missing}"
            )

    # ------------------------------------------------------------------
    # Convenience views used by packing strategies
    # ------------------------------------------------------------------
    @property
    def polymer_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.kind == UNIT_KIND_POLYMER]

    @property
    def molecule_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.kind == UNIT_KIND_MOLECULE]

    @property
    def substrate_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.role == UNIT_ROLE_SUBSTRATE]

    @property
    def film_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.role == UNIT_ROLE_FILM]

    @property
    def inorganic_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.kind == UNIT_KIND_INORGANIC]

    @property
    def filler_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.role == UNIT_ROLE_FILLER]

    @property
    def organic_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.kind != UNIT_KIND_INORGANIC]

    @property
    def matrix_units(self) -> List[UnitSpec]:
        """Organic units placed by strategies (grafted chains excluded)."""
        return [u for u in self.units if u.kind != UNIT_KIND_INORGANIC
                and u.role != UNIT_ROLE_GRAFT]

    @property
    def graft_units(self) -> List[UnitSpec]:
        return [u for u in self.units if u.role == UNIT_ROLE_GRAFT]

    def total_particle_count(self) -> int:
        """
        Total monomers + molecule instances, used for density box sizing
        (inorganic components are not monomers and are excluded).
        """
        total = 0
        for unit in self.units:
            if unit.kind == UNIT_KIND_POLYMER:
                total += (unit.n_monomers or 1) * unit.count
            elif unit.kind == UNIT_KIND_MOLECULE:
                total += unit.count
        return max(total, 1)

    def max_chain_length(self) -> int:
        """Largest n_monomers across polymer units (1 if none)."""
        return max((u.n_monomers or 1 for u in self.polymer_units), default=1)


def migrate_manifest(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explicit in-memory migration of a units.json dict to the current
    format.

    Version 1 -> 2: inorganic fields (lt_class, geometry_type, motion,
    material, envelope, mass, coords_file, chain_file) did not exist; they default to
    None, which is exactly the v1 meaning (organic units only).  The
    kind/role values of v1 ("polymer"/"molecule", "film"/"substrate") are
    unchanged in v2.  The original format_version is preserved so callers
    can tell where a manifest came from.
    """
    version = int(data.get("format_version", 1))
    if version < 1:
        raise ValidationError(f"Invalid units.json format_version {version}")
    return dict(data)
