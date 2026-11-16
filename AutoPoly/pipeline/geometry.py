#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry Builder — Stage 1 of the AutoPoly three-stage pipeline.

Force-field-agnostic coordinate generation. Consumes Polymer/Molecule
models and produces a single ``geometry.json`` artifact:

    models ──▶ GeometryBuilder ──▶ <system out>/<name>/geometry/geometry.json

The artifact stores:
- one chain graph per distinct polymer model (mapped SMILES + atom count),
- one entry per unique monomer variant (atoms with chain atom-map numbers,
  bonds, connection atoms, coordinates; plus mirrored _T1 stereochemistry
  variants),
- one entry per built chain with per-monomer placements (rotation +
  translation), either from MC self-avoiding chain growth, deterministic
  fallback, or circular ring placement,
- small molecules with embedded conformers,
- (format_version 2) an ``inorganic`` list: one entry per inorganic
  component (built-in substrate slab, nanoparticle species) holding the
  force-field-agnostic AtomicStructure (elements, coordinates, cell/PBC,
  bonds with image shifts, chemical roles, surface sites, descriptors,
  provenance) plus the component's identity/placement request (material,
  phase, surface, count, motion, requested inorganic force field).

Artifacts without inorganic components are still written as
format_version 1 (unchanged); readers accept versions 1 and 2.

Atom-map numbers are the join key to stage 2 (UnitTyper): every variant
atom carries the map number of its atom in the full chain graph, so types
and charges assigned on the whole chain transfer by lookup, never by
geometric inference.

Created on 2026-07-30
@author: zwu
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from ..core.system import logger
from ..core.exceptions import GenerationError
from ..mc import (
    AtomData,
    CollisionDetector,
    ChainGrowthMC,
    MonomerTemplate,
    calculate_box_size,
)
from ..monomers.monomer_generator import (
    GEOM_MAP_PROP,
    BackboneAligner,
    ChainBuilder,
    ChainSplitter,
    ConformerGenerator,
    compute_lt_atom_order,
)

GEOMETRY_FORMAT_VERSION = 2
#: Version written when no inorganic section is present (legacy layout).
GEOMETRY_FORMAT_VERSION_ORGANIC = 1
GEOMETRY_DIRNAME = "geometry"
GEOMETRY_FILENAME = "geometry.json"

# Base for per-atom chain map numbers. Must stay clear of the marker ranges
# used in monomer_generator (1/2 connection, 100+ inter-bond, 10000+ charges,
# 20000+ cap H). SMILES atom maps handle 7-digit numbers fine.
GEOM_MAP_BASE = 1_000_000

# Retry budget for MC chain growth before deterministic fallback
MAX_CHAIN_RETRIES = 5


@dataclass
class GeometryConfig:
    """
    Configuration for force-field-agnostic geometry generation.

    Attributes:
        use_mc_chain_growth: Use MC self-avoiding random walk per chain
            (False = deterministic linear placement).
        mc_max_attempts: Max placement attempts per monomer in chain growth.
        mc_bond_angle_min: Minimum deflection angle (deg) for chain growth.
        mc_bond_angle_max: Maximum deflection angle (deg) for chain growth.
        mc_intrachain_exclude_neighbors: Neighbors excluded from intra-chain
            collision checks.
        offset: Nominal monomer-monomer spacing (Angstrom).
        offset_spacing: Extra spacing added to measured monomer length in
            deterministic placement.
        rotate: Rotation angle (deg) applied to every second monomer in
            deterministic placement.
        rng_seed: Optional seed for reproducible MC chain growth.
    """
    use_mc_chain_growth: bool = True
    mc_max_attempts: int = 10000
    mc_bond_angle_min: float = 50.0
    mc_bond_angle_max: float = 90.0
    mc_intrachain_exclude_neighbors: int = 2
    offset: float = 4.0
    offset_spacing: float = 2.0
    rotate: float = 90.0
    rng_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeometryResult:
    """Handle to a written geometry artifact."""
    directory: Path
    data: Dict[str, Any]

    @property
    def dir(self) -> str:
        """Geometry directory path (string form for convenience)."""
        return str(self.directory)

    @property
    def json_path(self) -> Path:
        return self.directory / GEOMETRY_FILENAME


def is_molecule_model(model: object) -> bool:
    """Detect Molecule models (vs Polymer) by their type flag."""
    return bool(getattr(model, "_is_molecule", False))


def is_inorganic_component(model: object) -> bool:
    """
    Detect inorganic stage-1 components (nanoparticles, built-in slab
    adapters): objects flagged ``_is_inorganic_component`` that provide
    ``geometry_entry(component_id) -> dict``.
    """
    return getattr(model, "_is_inorganic_component", False) is True


class GeometryBuilder:
    """
    Stage 1: build force-field-agnostic geometry for a set of models.

    Example:
        >>> geom = GeometryBuilder(system, name="peo", config=GeometryConfig())
        >>> result = geom.build([polymer])
        >>> result.dir
        '<out>/peo/geometry'
    """

    def __init__(
        self,
        system: object,
        name: str,
        config: Optional[GeometryConfig] = None,
    ) -> None:
        """
        Args:
            system: System object providing get_folder_path().
            name: Project name; geometry goes to <out>/<name>/geometry/.
            config: GeometryConfig (defaults to MC chain growth).
        """
        self.system = system
        self.name = name
        self.config = config or GeometryConfig()
        self.project_dir = Path(system.get_folder_path()) / name
        self.geometry_dir = self.project_dir / GEOMETRY_DIRNAME

        # RDKit helpers (typing-free: no force field involved at this stage)
        self._chain_builder = ChainBuilder(verbose=False)
        self._conformer_gen = ConformerGenerator(verbose=False)
        self._chain_splitter = ChainSplitter(self._conformer_gen, verbose=False)
        self._aligner = BackboneAligner(verbose=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, models: List[object],
              substrate_models: Optional[List[object]] = None,
              inorganic: Optional[List[object]] = None) -> GeometryResult:
        """
        Build geometry for all models and write geometry.json.

        Args:
            models: List of Polymer and/or Molecule model objects (film),
                optionally mixed with inorganic components (e.g.
                Nanoparticle), which are routed to the inorganic section.
            substrate_models: Optional list of models forming a physical
                substrate slab. Processed identically to film models, but
                their chains/molecules are tagged role="substrate" so the
                packing stage can place them into the slab region.
            inorganic: Optional extra inorganic components (e.g. the
                built-in slab adapter generate() creates for
                SubstrateSpec(builder=...)).

        Returns:
            GeometryResult with the geometry directory and data dict.

        Raises:
            GenerationError: On SMILES/embedding/splitting failures.
        """
        if self.config.rng_seed is not None:
            np.random.seed(self.config.rng_seed)

        components = [m for m in models if is_inorganic_component(m)]
        components += list(inorganic or [])
        models = [m for m in models if not is_inorganic_component(m)]

        chain_graphs: Dict[str, Any] = {}
        variants: Dict[str, Any] = {}
        chains: List[Dict[str, Any]] = []
        molecules: List[Dict[str, Any]] = []

        partitioned = (
            [(m, "film", None) for m in models]
            + [(m, "substrate", None) for m in (substrate_models or [])]
        )
        # grafted chains of inorganic components: regular chains tagged
        # role "graft" + owner; the owner anchors them in its entry
        for comp in components:
            for gm in getattr(comp, "graft_models", lambda: [])():
                partitioned.append((gm, "graft", comp.component_id))

        poly_model_idx = 0
        for model, role, owner in partitioned:
            if is_molecule_model(model):
                molecules.append(self._build_molecule_entry(model, role))
            else:
                model_id = f"model_{poly_model_idx}"
                base_name = f"monomer_{poly_model_idx}"
                poly_model_idx += 1
                n_before = len(chains)
                self._build_polymer_entries(
                    model, model_id, base_name,
                    chain_graphs, variants, chains,
                    role=role,
                )
                if owner is not None:
                    for k, chain in enumerate(chains[n_before:]):
                        chain["graft_of"] = owner
                        chain["graft_index"] = k

        # Inorganic components last: their builders use private RNGs, so
        # the organic geometry is identical with or without them.
        inorganic_entries = self._build_inorganic_entries(
            components, {"chains": chains, "variants": variants})

        data = {
            "format_version": (GEOMETRY_FORMAT_VERSION if inorganic_entries
                               else GEOMETRY_FORMAT_VERSION_ORGANIC),
            "created_with": {"autopoly_version": self._autopoly_version()},
            "mc_config": self.config.to_dict(),
            "chain_graphs": chain_graphs,
            "variants": variants,
            "chains": chains,
            "molecules": molecules,
        }
        if inorganic_entries:
            data["inorganic"] = inorganic_entries

        self.geometry_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.geometry_dir / GEOMETRY_FILENAME
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Geometry written to {json_path}: "
            f"{len(chains)} chains, {len(variants)} variants, "
            f"{len(molecules)} molecule species"
            + (f", {len(inorganic_entries)} inorganic components"
               if inorganic_entries else "")
        )
        return GeometryResult(directory=self.geometry_dir, data=data)

    @staticmethod
    def _build_inorganic_entries(components: List[object],
                                 geometry: Optional[Dict[str, Any]] = None
                                 ) -> List[Dict[str, Any]]:
        """geometry.json inorganic entries with unique component IDs."""
        from ..core.exceptions import ValidationError

        entries: List[Dict[str, Any]] = []
        used = set()
        for k, comp in enumerate(components):
            default_id = getattr(comp, "component_id", None) or f"inorganic_{k + 1}"
            if getattr(comp, "grafts", None) is not None:
                entry = comp.geometry_entry(default_id, geometry=geometry)
            else:
                entry = comp.geometry_entry(default_id)
            if entry["id"] in used:
                raise ValidationError(
                    f"Duplicate inorganic component id '{entry['id']}'; give "
                    "each nanoparticle species a unique name"
                )
            used.add(entry["id"])
            entries.append(entry)
        return entries

    @staticmethod
    def load(geometry_dir: str) -> Dict[str, Any]:
        """Load a geometry.json artifact and check its format version."""
        from ..core.exceptions import ValidationError

        path = Path(geometry_dir) / GEOMETRY_FILENAME
        if not path.is_file():
            raise ValidationError(f"geometry.json not found: {path}")
        with open(path) as f:
            data = json.load(f)
        version = int(data.get("format_version", 0))
        data.setdefault("inorganic", [])
        if version > GEOMETRY_FORMAT_VERSION:
            raise ValidationError(
                f"Unsupported geometry.json format_version {version} "
                f"(supported up to {GEOMETRY_FORMAT_VERSION})"
            )
        return data

    # ------------------------------------------------------------------
    # Polymer processing
    # ------------------------------------------------------------------
    def _build_polymer_entries(
        self,
        model: object,
        model_id: str,
        base_name: str,
        chain_graphs: Dict[str, Any],
        variants: Dict[str, Any],
        chains: List[Dict[str, Any]],
        role: str = "film",
    ) -> None:
        """Build chain graph, variants, and per-chain placements for a Polymer."""
        topology = getattr(model, "topology", "linear")
        dop = model.dop

        # 1. Assemble the chain graph from complement SMILES
        try:
            chain_mol, inter_bond_markers = self._chain_builder.build_chain(
                model.sequence
            )
        except Exception as e:
            raise GenerationError(
                f"Chain building failed for {model_id} "
                f"(sequence={model.sequence}): {e}"
            ) from e

        # 2. Give every atom a unique chain-level map number. Inter-bond
        #    marker atoms keep their existing (already unique) numbers.
        next_map = GEOM_MAP_BASE
        for atom in chain_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(next_map)
                next_map += 1

        chain_graphs[model_id] = {
            "smiles_mapped": Chem.MolToSmiles(chain_mol),
            "atom_count": chain_mol.GetNumAtoms(),
        }

        # 3. Embed conformers + split into per-position variants (no typing)
        try:
            if inter_bond_markers:
                position_variants = self._chain_splitter.split_chain(
                    chain_mol, inter_bond_markers, base_name, force_field="none"
                )
                position_variants = [
                    self._aligner.align_for_lt_file(v) for v in position_variants
                ]
            else:
                # DOP=1: the whole chain is a single standalone monomer
                position_variants = [self._single_variant(chain_mol, base_name)]
        except Exception as e:
            raise GenerationError(
                f"Chain splitting/embedding failed for {model_id}: {e}"
            ) from e

        # 4. Deduplicate variants by (variant type, mapless SMILES) so
        #    chemically identical positions share one .lt file.
        unique, by_position = self._deduplicate_variants(
            position_variants, topology, base_name
        )

        # 5. Serialize unique variants (plus mirrored _T1 variants)
        for variant in unique.values():
            name = self._variant_filename(variant, base_name)
            variants[name] = self._serialize_variant(variant, name, model_id)
            t1_name = f"{name}_T1"
            variants[t1_name] = self._serialize_variant(
                variant, t1_name, model_id, t1_of=name
            )

        # 6. Per-chain placements
        for chain_idx, chain_smiles in enumerate(model.sequenceSet):
            position_names = []
            for pos_idx, smiles in enumerate(chain_smiles):
                has_t1 = "_T1" in smiles
                name, name_t1 = by_position[pos_idx]
                position_names.append(name_t1 if has_t1 else name)

            placements = self._compute_chain_placements(
                position_names, variants, topology, chain_idx
            )
            chains.append({
                "id": f"chain_{len(chains)}",
                "model_id": model_id,
                "topology": topology,
                "placements": placements,
                "n_monomers": len(position_names),
                "radius": self._chain_radius(topology, dop),
                "role": role,
            })

    def _single_variant(self, chain_mol: Chem.Mol, base_name: str):
        """
        Build a 'single' MonomerVariant for a DOP=1 chain.

        The chain molecule (already built from SMILES with explicit Hs and
        per-atom map numbers) gets a conformer; atom map numbers are stashed
        in the geom_map property for the stage-2 type join.
        """
        from ..monomers.monomer_generator import MonomerVariant

        mol = self._conformer_gen.generate_conformer(Chem.Mol(chain_mol))
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                atom.SetIntProp(GEOM_MAP_PROP, map_num)
        return MonomerVariant(
            base_name=base_name,
            variant_type="single",
            mol=mol,
            smiles=Chem.MolToSmiles(mol),
            connection_atoms=(0, mol.GetNumAtoms() - 1),
            force_field="none",
            position=0,
        )

    def _deduplicate_variants(
        self,
        position_variants: list,
        topology: str,
        base_name: str,
    ) -> Tuple[Dict[tuple, object], List[Tuple[str, str]]]:
        """
        Deduplicate per-position variants and build the position -> name map.

        Mirrors the legacy monomer_processing behavior: chemically distinct
        monomers of the same variant type each get their own file; ring
        topologies use middle variants for every position.

        Returns:
            (unique_variants, by_position) where by_position maps each chain
            position to (name, name_t1).
        """
        unique: Dict[tuple, object] = {}
        for variant in position_variants:
            key = (variant.variant_type, self._mapless_smiles(variant.mol))
            if key not in unique:
                unique[key] = variant

        if topology == "ring":
            ring_variants = {
                ("ring", key[1]): v for key, v in unique.items() if key[0] == "middle"
            }
            if not ring_variants:
                ring_variants = {
                    ("ring", key[1]): v for key, v in unique.items() if key[0] == "first"
                }
            if ring_variants:
                unique = ring_variants

        # Name each unique variant once (name derives from first position
        # where that chemistry appears, exactly like the legacy naming).
        name_lookup: Dict[tuple, Tuple[str, str]] = {}
        for key, variant in unique.items():
            name = self._variant_filename(variant, base_name)
            name_lookup[key] = (name, f"{name}_T1")

        by_position = []
        for variant in position_variants:
            vt = "ring" if topology == "ring" else variant.variant_type
            key = (vt, self._mapless_smiles(variant.mol))
            if key not in name_lookup:
                # Ring: first/last end chemistry has no middle variant; use
                # the (single) ring variant for every position, matching the
                # legacy "middle variants for all ring positions" policy.
                name_lookup[key] = next(iter(name_lookup.values()))
            by_position.append(name_lookup[key])

        return unique, by_position

    @staticmethod
    def _variant_filename(variant, base_name: str) -> str:
        """Deterministic, FF-independent variant class/file name."""
        suffix_map = {"first": "le", "last": "re", "single": "single"}
        suffix = suffix_map.get(variant.variant_type, "i")
        return f"{base_name}_{variant.position}{suffix}"

    def _serialize_variant(
        self,
        variant,
        name: str,
        model_id: str,
        t1_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Serialize a MonomerVariant into geometry.json form.

        Atoms are stored in LT output order (connection atoms first), each
        carrying its chain-level map number. For _T1 variants the coordinates
        are mirrored in z.
        """
        mol = variant.mol
        conn_left, conn_right = variant.connection_atoms
        order = compute_lt_atom_order(
            mol, variant.variant_type, conn_left, conn_right
        )
        position_of = {atom_idx: pos for pos, atom_idx in enumerate(order)}

        conf = mol.GetConformer(0) if mol.GetNumConformers() > 0 else None
        atoms = []
        for atom_idx in order:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.HasProp(GEOM_MAP_PROP):
                map_num = atom.GetIntProp(GEOM_MAP_PROP)
            else:
                map_num = atom.GetAtomMapNum()
            if conf is not None:
                pos = conf.GetAtomPosition(atom_idx)
                coords = [pos.x, pos.y, pos.z]
            else:
                coords = [0.0, 0.0, 0.0]
            if t1_of is not None:
                coords[2] = -coords[2]
            atoms.append({
                "map_num": int(map_num),
                "element": atom.GetSymbol(),
                "coords": coords,
            })

        bonds = [
            [position_of[b.GetBeginAtomIdx()], position_of[b.GetEndAtomIdx()]]
            for b in mol.GetBonds()
        ]

        # Singles/molecules have no polymerization connection points
        has_connections = variant.variant_type not in ("single", "molecule")
        return {
            "model_id": model_id,
            "variant_type": variant.variant_type,
            "atoms": atoms,
            "bonds": bonds,
            "connection_atoms": {
                "left": position_of.get(conn_left) if has_connections else None,
                "right": position_of.get(conn_right) if has_connections else None,
            },
            "t1_variant_of": t1_of,
        }

    # ------------------------------------------------------------------
    # Chain placements
    # ------------------------------------------------------------------
    def _compute_chain_placements(
        self,
        position_names: List[str],
        variants: Dict[str, Any],
        topology: str,
        chain_idx: int,
    ) -> List[Dict[str, Any]]:
        """Compute per-monomer transforms for one chain."""
        n = len(position_names)
        if n == 1:
            # Single monomers are placed as whole units at box-packing time
            return [self._placement(position_names[0], None, None)]
        if topology == "ring":
            return self._ring_placements(position_names)
        if self.config.use_mc_chain_growth:
            return self._mc_placements(position_names, variants, chain_idx)
        return self._deterministic_placements(position_names, variants)

    def _ring_placements(self, position_names: List[str]) -> List[Dict[str, Any]]:
        """Circular placement for ring topologies."""
        n = len(position_names)
        radius = self.config.offset * n / (2 * np.pi)
        placements = []
        for i, name in enumerate(position_names):
            angle = 2 * np.pi * i / n
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            rotation_angle = float(np.degrees(angle)) + 90.0
            placements.append(self._placement(
                name,
                {"angle_deg": rotation_angle, "axis": [0.0, 0.0, 1.0]},
                [x, y, 0.0],
            ))
        return placements

    def _deterministic_placements(
        self,
        position_names: List[str],
        variants: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Deterministic linear placement (also the MC fallback)."""
        placements = []
        offset_cum = 0.0
        current_offset = self.config.offset
        for i, name in enumerate(position_names):
            if i == 0:
                placements.append(self._placement(name, None, None))
            else:
                placements.append(self._placement(
                    name,
                    {"angle_deg": self.config.rotate * (i % 2), "axis": [1.0, 0.0, 0.0]},
                    [offset_cum, 0.0, 0.0],
                ))
            # Legacy evaluate_offset: spacing after this monomer is the
            # distance between its first two (connection) atoms + margin.
            atoms = variants[name]["atoms"]
            if len(atoms) >= 2:
                c0 = np.array(atoms[0]["coords"])
                c1 = np.array(atoms[1]["coords"])
                current_offset = float(np.linalg.norm(c0 - c1)) + self.config.offset_spacing
            offset_cum += current_offset
        return placements

    def _mc_placements(
        self,
        position_names: List[str],
        variants: Dict[str, Any],
        chain_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        MC self-avoiding chain growth with retries and deterministic fallback
        (same policy and warnings as the legacy workflow).
        """
        cfg = self.config
        n = len(position_names)
        estimated_chain_length = n**0.6 * 4.0
        box_size = max(
            estimated_chain_length * 3.0,
            calculate_box_size(n, monomer_density=0.05),
        )
        half_box = box_size / 2
        box_bounds = ((-half_box, half_box),) * 3

        templates = [
            self._template_from_variant(name, variants[name])
            for name in position_names
        ]

        for retry in range(MAX_CHAIN_RETRIES):
            collision_detector = CollisionDetector(box_bounds, cell_size=5.0)
            chain_mc = ChainGrowthMC(
                collision_detector,
                cfg.mc_max_attempts,
                bond_angle_min=cfg.mc_bond_angle_min,
                bond_angle_max=cfg.mc_bond_angle_max,
                intrachain_exclude_neighbors=cfg.mc_intrachain_exclude_neighbors,
            )
            try:
                mc_placements = chain_mc.grow_chain_from_templates(
                    templates, chain_id=chain_idx
                )
                return [
                    self._placement(
                        p.template.monomer_name,
                        None if np.allclose(p.rotation_matrix, np.eye(3)) else {
                            "angle_deg": p.rotation_axis_angle[0],
                            "axis": list(p.rotation_axis_angle[1:]),
                        },
                        list(p.position),
                    )
                    for p in mc_placements
                ]
            except RuntimeError as e:
                logger.warning(
                    f"MC chain growth attempt {retry + 1}/{MAX_CHAIN_RETRIES} "
                    f"failed for chain {chain_idx}: {e}"
                )

        logger.warning(
            "=" * 70 + "\n"
            "WARNING: MC chain growth failed for chain %d after %d attempts.\n"
            "Falling back to DETERMINISTIC placement. The resulting chain\n"
            "coordinates are a crude linear/grid arrangement that may contain\n"
            "atomic clashes — inspect the output system.data before running MD,\n"
            "or increase the box size / mc_max_attempts and regenerate.\n"
            + "=" * 70,
            chain_idx, MAX_CHAIN_RETRIES,
        )
        return self._deterministic_placements(position_names, variants)

    @staticmethod
    def _template_from_variant(name: str, record: Dict[str, Any]) -> MonomerTemplate:
        """Build an in-memory MonomerTemplate from a geometry variant entry."""
        atoms = [
            AtomData(
                atom_id=f"{a['element']}{i + 1}",
                element=a["element"],
                coords=np.array(a["coords"], dtype=float),
                atom_type="",
                charge=0.0,
            )
            for i, a in enumerate(record["atoms"])
        ]
        conn = record["connection_atoms"]
        left_idx, right_idx = conn["left"], conn["right"]
        return MonomerTemplate(
            lt_file=name,
            monomer_name=name,
            monomer_type=record["variant_type"],
            atoms=atoms,
            left_conn_coords=atoms[left_idx].coords if left_idx is not None else None,
            right_conn_coords=atoms[right_idx].coords if right_idx is not None else None,
            left_conn_id=atoms[left_idx].atom_id if left_idx is not None else None,
            right_conn_id=atoms[right_idx].atom_id if right_idx is not None else None,
        )

    @staticmethod
    def _placement(
        variant: str,
        rotation: Optional[Dict[str, Any]],
        translation: Optional[List[float]],
    ) -> Dict[str, Any]:
        return {
            "variant": variant,
            "rotation": rotation,
            "translation": translation,
        }

    def _chain_radius(self, topology: str, dop: int) -> float:
        """SAW collision radius (same formulas as the legacy workflow)."""
        if topology == "ring":
            return self.config.offset * np.sqrt(dop / 12) * 2.0 + 2.0
        return self.config.offset * np.sqrt(dop / 6) * 2.0 + 2.0

    # ------------------------------------------------------------------
    # Molecule processing
    # ------------------------------------------------------------------
    def _build_molecule_entry(self, model: object,
                              role: str = "film") -> Dict[str, Any]:
        """Embed a small molecule conformer and serialize it (no typing)."""
        smiles = model.Smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise GenerationError(f"Invalid molecule SMILES: {smiles}")
        mol = Chem.AddHs(mol)

        # Map numbers = join key for stage-2 typing lookups
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
        smiles_mapped = Chem.MolToSmiles(mol)

        try:
            mol = self._conformer_gen.generate_conformer(mol)
        except Exception as e:
            raise GenerationError(
                f"Conformer generation failed for molecule '{model.molecule_name}' "
                f"({smiles}): {e}"
            ) from e

        # LT output order: heavy atoms first, then hydrogens
        order = [
            i for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        ] + [
            i for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        ]
        position_of = {atom_idx: pos for pos, atom_idx in enumerate(order)}

        conf = mol.GetConformer(0)
        atoms = []
        for atom_idx in order:
            atom = mol.GetAtomWithIdx(atom_idx)
            pos = conf.GetAtomPosition(atom_idx)
            atoms.append({
                "map_num": atom.GetAtomMapNum(),
                "element": atom.GetSymbol(),
                "coords": [pos.x, pos.y, pos.z],
            })

        bonds = [
            [position_of[b.GetBeginAtomIdx()], position_of[b.GetEndAtomIdx()]]
            for b in mol.GetBonds()
        ]

        return {
            "name": model.molecule_name,
            "smiles": smiles,
            "smiles_mapped": smiles_mapped,
            "count": model.Count,
            "atoms": atoms,
            "bonds": bonds,
            "role": role,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mapless_smiles(mol: Chem.Mol) -> str:
        """Canonical SMILES without atom map numbers (for variant dedup)."""
        m = Chem.Mol(mol)
        for atom in m.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(m)

    @staticmethod
    def _autopoly_version() -> str:
        try:
            from AutoPoly import __version__
            return __version__
        except Exception:
            return "unknown"
