#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Typer — Stage 2 of the AutoPoly three-stage pipeline.

Force-field assignment on top of stored geometry:

    geometry/geometry.json + force_field ──▶ build/<ff>/*.lt + units.json

Typing is graph-based: the full chain molecule is rebuilt from the stored
mapped SMILES (no coordinates needed), typed with priority-based SMARTS
matching, and charges are assigned (Gasteiger for GAFF/GAFF2 — computed on
the full chain, and on each small molecule — .fdefn charge tables otherwise).
Every stored variant atom carries its chain atom-map number, so types/charges
transfer by lookup — variant atoms *are* chain atoms.

The optional "openff" force field works differently: the chain/molecule is
parameterized with the OpenFF toolkit (SMIRNOFF direct chemical perception;
NAGL GNN charges by default), and every bonded term is written explicitly into
the variant/poly .lt files (no "By Type" inference). See _type_openff().

The same geometry can be typed under multiple force fields into separate
build/<ff>/ directories (typing is the cheap stage).

Created on 2026-07-30
@author: zwu
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..core.system import logger
from ..core.conf import FORCE_FIELD_REGISTRY
from ..core.exceptions import ValidationError
from .geometry import GEOMETRY_DIRNAME, GEOM_MAP_BASE, GeometryBuilder
from ..monomers.monomer_generator import (
    INTER_BOND_MAP_START,
    AtomTypingError,
    SMARTSTyper,
    write_lt_footer,
    write_lt_header,
)
from ..monomers.monomer_processing import read_lt_end_atoms
from .units import UnitLibrary, UnitSpec

BUILD_DIRNAME = "build"

# Collision radius derivation for molecule-like units: bounding sphere over
# the embedded coords (max |r - center|) plus padding, with a floor for
# (near-)single-atom species.
MOLECULE_RADIUS_PADDING = 1.0
MIN_MOLECULE_RADIUS = 1.5


def molecule_radius(atoms: List[Dict[str, Any]]) -> float:
    """
    Collision radius from stored atom coords: max |r - center| + padding.

    Replaces a fixed default so small molecules (water) don't waste volume
    and large solvents don't silently overlap during MC placement.
    """
    coords = np.asarray([a["coords"] for a in atoms], dtype=float)
    center = coords.mean(axis=0)
    max_dist = float(np.linalg.norm(coords - center, axis=1).max())
    return max(max_dist + MOLECULE_RADIUS_PADDING, MIN_MOLECULE_RADIUS)


def mol_from_mapped_smiles(smiles_mapped: str) -> Chem.Mol:
    """
    Rebuild an RDKit Mol from a mapped SMILES, preserving explicit Hs.

    The default MolFromSmiles strips explicit hydrogens (and their map
    numbers), which would break the map-number join to stored geometry, so
    parsing goes through sanitize=False + explicit property cache update.
    """
    mol = Chem.MolFromSmiles(smiles_mapped, sanitize=False)
    if mol is None:
        return None
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    return mol


class UnitTyper:
    """
    Stage 2: assign force-field types/charges onto stored geometry.

    Example:
        >>> UnitTyper(geom.dir, "oplsaa").type()   # build/oplsaa/
        >>> UnitTyper(geom.dir, "gaff2").type()    # build/gaff2/ — same geometry
    """

    def __init__(
        self,
        geometry_dir: Union[str, Path],
        force_field: str,
        output_dir: Optional[Union[str, Path]] = None,
        typer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            geometry_dir: Directory containing geometry.json (stage 1 output).
            force_field: One of oplsaa, lopls, gaff, gaff2, dreiding, compass,
                openff.
            output_dir: Build directory (default: <project>/build/<ff>).
            typer_options: Force-field-specific typer options. Currently only
                used for "openff" (forwarded to OpenFFTyper: offxml,
                charge_method, nagl_model, neutralize).

        Raises:
            ValidationError: Unknown force field, or unsupported/missing
                             geometry artifact.
        """
        if force_field not in FORCE_FIELD_REGISTRY:
            raise ValidationError(
                f"Invalid force_field '{force_field}'. "
                f"Must be one of: {list(FORCE_FIELD_REGISTRY)}"
            )
        self.geometry_dir = Path(geometry_dir)
        self.geometry = GeometryBuilder.load(self.geometry_dir)
        self.force_field = force_field
        self.typer_options = dict(typer_options or {})
        if output_dir is None:
            output_dir = self.geometry_dir.parent / BUILD_DIRNAME / force_field
        self.output_dir = Path(output_dir)
        if force_field == "openff":
            # Fail fast: ring chain graphs are stored as linear capped
            # molecules, so faithful SMIRNOFF ring typing needs a dedicated
            # assembly path (not yet implemented).
            for chain in self.geometry["chains"]:
                if chain["topology"] == "ring":
                    raise ValidationError(
                        "force_field='openff' does not support topology='ring' "
                        "yet (ring chain graphs are stored as linear capped "
                        "molecules; SMIRNOFF typing of the closure chemistry "
                        "requires a dedicated ring assembly path)."
                    )
            from ..forcefields.openff_typing import OpenFFTyper
            self.typer = OpenFFTyper(**self.typer_options)
        else:
            if self.typer_options:
                raise ValidationError(
                    "typer_options is only supported for force_field='openff' "
                    f"(got force_field='{force_field}')"
                )
            self.typer = SMARTSTyper(force_field, verbose=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def type(self) -> UnitLibrary:
        """
        Type all variants and molecules; write .lt files and units.json.

        Returns:
            The UnitLibrary manifest (also saved to the build directory).

        Raises:
            AtomTypingError: If chain/molecule typing fails.
            ValidationError: If variant map numbers are missing from the
                             typed chain.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.force_field == "openff":
            return self._type_openff()

        # 1. Type every chain graph (whole chain, for correct environments)
        typed_chains = self._type_chain_graphs()

        # 2. Write typed monomer variant .lt files
        variant_files = self._write_variant_files(typed_chains)

        # 3. Write poly_N.lt per multi-monomer chain and collect polymer units
        units, dop1_counts = self._write_chain_files()

        # 4. Type and write small-molecule .lt files; collect molecule units
        molecule_files = self._write_molecule_files(units)

        # 5. DOP=1 chains become molecule-like units grouped by variant
        self._add_single_monomer_units(units, dop1_counts)

        library = UnitLibrary(
            force_field=self.force_field,
            units=units,
            monomer_files=sorted(set(variant_files + molecule_files)),
            build_config={
                "use_mc_chain_growth": self.geometry["mc_config"].get(
                    "use_mc_chain_growth", True
                ),
                "offset": self.geometry["mc_config"].get("offset", 4.0),
                "rotate": self.geometry["mc_config"].get("rotate", 90.0),
            },
            geometry_source=str(
                Path("..") / GEOMETRY_DIRNAME / "geometry.json"
            ),
        )
        library.save(self.output_dir)
        library.validate(self.output_dir)
        library.source_dir = str(self.output_dir)
        logger.info(
            f"Typing complete ({self.force_field}): {len(units)} units, "
            f"{len(library.monomer_files)} monomer files -> {self.output_dir}"
        )
        return library

    # ------------------------------------------------------------------
    # Chain graph typing
    # ------------------------------------------------------------------
    def _type_chain_graphs(self) -> Dict[str, Chem.Mol]:
        """
        Rebuild and type the full chain molecule per polymer model.

        Returns:
            Dict of model_id -> typed RDKit Mol (map numbers intact).
        """
        typed = {}
        for model_id, graph in self.geometry["chain_graphs"].items():
            mol = mol_from_mapped_smiles(graph["smiles_mapped"])
            if mol is None:
                raise AtomTypingError(
                    f"Could not rebuild chain for {model_id} from mapped SMILES"
                )
            try:
                self.typer.assign_atom_types(mol)
            except Exception as e:
                raise AtomTypingError(
                    f"Atom typing failed for {model_id} "
                    f"(force field {self.force_field}): {e}"
                ) from e

            # Gasteiger charges on the FULL chain (before any splitting),
            # matching the legacy behavior for GAFF/GAFF2.
            if self.force_field in ("gaff", "gaff2"):
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                except Exception as e:
                    logger.error(
                        f"Failed to compute Gasteiger charges for {model_id}: {e}"
                    )
            typed[model_id] = mol
        return typed

    def _chain_lookup(self, chain_mol: Chem.Mol, model_id: str) -> Dict[int, int]:
        """Map number -> atom index lookup for a typed chain."""
        lookup = {}
        for atom in chain_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                lookup[map_num] = atom.GetIdx()
        if not lookup:
            raise ValidationError(
                f"Chain graph for {model_id} has no atom map numbers; "
                "cannot join types onto geometry"
            )
        return lookup

    # ------------------------------------------------------------------
    # Variant .lt files
    # ------------------------------------------------------------------
    def _write_variant_files(self, typed_chains: Dict[str, Chem.Mol]) -> List[str]:
        """Write one typed .lt file per geometry variant. Returns file names."""
        lookups = {
            model_id: self._chain_lookup(mol, model_id)
            for model_id, mol in typed_chains.items()
        }
        written = []
        for name, variant in self.geometry["variants"].items():
            chain_mol = typed_chains[variant["model_id"]]
            lookup = lookups[variant["model_id"]]
            rows = self._typed_atom_rows(variant, chain_mol, lookup, name)
            self._write_monomer_lt(name, variant, rows)
            written.append(f"{name}.lt")
        return written

    def _typed_atom_rows(
        self,
        variant: Dict[str, Any],
        chain_mol: Chem.Mol,
        lookup: Dict[int, int],
        name: str,
    ) -> List[Dict[str, Any]]:
        """
        Join variant atoms onto the typed chain via map numbers.

        Returns one row per atom: atom_id, atom_type, charge, coords.
        """
        rows = []
        for i, a in enumerate(variant["atoms"]):
            map_num = a["map_num"]
            idx = lookup.get(map_num)
            if idx is None:
                raise ValidationError(
                    f"Variant '{name}' atom {i} references map number "
                    f"{map_num}, which is missing from the typed chain "
                    f"({variant['model_id']})"
                )
            chain_atom = chain_mol.GetAtomWithIdx(idx)
            atom_type = self._atom_type(chain_atom, a["element"])
            charge = self._atom_charge(chain_atom, atom_type)
            rows.append({
                "atom_id": f"{a['element']}{i + 1}",
                "atom_type": atom_type,
                "charge": charge,
                "coords": a["coords"],
            })
        return rows

    def _atom_type(self, atom, element: str) -> str:
        """Atom type property with the legacy element fallback."""
        try:
            return atom.GetProp("AtomType")
        except KeyError:
            return f"@atom:{element.lower()}"

    def _atom_charge(self, atom, atom_type: str) -> float:
        """Per-atom charge: Gasteiger (gaff/gaff2) or .fdefn charge table."""
        if self.force_field in ("gaff", "gaff2"):
            try:
                charge = float(atom.GetProp("_GasteigerCharge"))
                if charge != charge or charge in (float("inf"), float("-inf")):
                    return 0.0
                return charge
            except (KeyError, ValueError):
                return 0.0
        return self.typer.charge_dict.get(atom_type, 0.0)

    def _write_monomer_lt(
        self, name: str, variant: Dict[str, Any], rows: List[Dict[str, Any]]
    ) -> None:
        """Write one typed monomer variant .lt file."""
        path = self.output_dir / f"{name}.lt"
        with open(path, "w") as f:
            write_lt_header(f, name, self.force_field)
            f.write('  write("Data Atoms") {\n')
            for row in rows:
                x, y, z = row["coords"]
                f.write(
                    f"\t$atom:{row['atom_id']} $mol:... {row['atom_type']} "
                    f"{row['charge']:.4f}    {x:.3f}   {y:.3f}   {z:.3f}\n"
                )
            f.write("  }\n\n")
            f.write("  write('Data Bond List') {\n")
            for i, j in variant["bonds"]:
                id1 = rows[i]["atom_id"]
                id2 = rows[j]["atom_id"]
                f.write(f"\t$bond:{id1}{id2}\t$atom:{id1}\t$atom:{id2}\n")
            f.write("  }\n")
            write_lt_footer(f, name)

    # ------------------------------------------------------------------
    # poly_N.lt chain files
    # ------------------------------------------------------------------
    def _write_chain_files(self):
        """
        Write poly_N.lt for every multi-monomer chain.

        Returns:
            (units, dop1_counts): polymer UnitSpec list, and a dict of
            (variant name, role) -> chain count for DOP=1 chains.
        """
        entry = FORCE_FIELD_REGISTRY[self.force_field]
        units: List[UnitSpec] = []
        dop1_counts: Dict[tuple, int] = {}

        poly_index = 0
        for chain in self.geometry["chains"]:
            role = chain.get("role", "film")
            placements = chain["placements"]
            n = len(placements)
            if n <= 1:
                variant_name = placements[0]["variant"]
                key = (variant_name, role)
                dop1_counts[key] = dop1_counts.get(key, 0) + 1
                continue

            poly_index += 1
            poly_id = f"poly_{poly_index}"
            variant_names = [p["variant"] for p in placements]
            monomer_files = sorted({f"{v}.lt" for v in variant_names})

            self._write_poly_lt(
                self.output_dir / f"{poly_id}.lt",
                poly_id, entry, placements,
                ring=(chain["topology"] == "ring"),
            )

            anchors = self._chain_anchors(variant_names, n)
            units.append(UnitSpec(
                id=poly_id,
                kind="polymer",
                lt_file=f"{poly_id}.lt",
                count=1,
                topology=chain["topology"],
                n_monomers=chain["n_monomers"],
                radius=chain["radius"],
                anchors=anchors,
                role=role,
                monomer_files=monomer_files,
            ))

        return units, dop1_counts

    def _write_poly_lt(
        self,
        path: Path,
        poly_id: str,
        ff_entry: Dict[str, str],
        placements: List[Dict[str, Any]],
        ring: bool,
    ) -> None:
        """Write one poly_N.lt from stored placements (verbatim transforms)."""
        n = len(placements)
        variant_names = [p["variant"] for p in placements]

        with open(path, "w") as f:
            f.write(f'import "{ff_entry["lt_file"]}"\n')
            for name in dict.fromkeys(variant_names):
                f.write(f'import "{name}.lt"\n')
            f.write("\n")
            f.write(f"{poly_id} inherits {ff_entry['inherits']} {{\n\n")
            f.write("    create_var {$mol}\n\n")

            for i, p in enumerate(placements):
                f.write(f"    monomer[{i}] = new {p['variant']}")
                if p["rotation"] is not None:
                    angle = p["rotation"]["angle_deg"]
                    ax, ay, az = p["rotation"]["axis"]
                    f.write(f".rot({angle:.4f},{ax:.4f},{ay:.4f},{az:.4f})")
                if p["translation"] is not None:
                    x, y, z = p["translation"]
                    f.write(f".move({x:.4f},{y:.4f},{z:.4f})")
                f.write("\n")

            f.write("\n    write('Data Bond List') {\n")
            bond_index = 0
            for i in range(n if ring else n - 1):
                next_i = (i + 1) % n
                _, second_atom = read_lt_end_atoms(
                    self.output_dir / f"{variant_names[i]}.lt"
                )
                first_atom, _ = read_lt_end_atoms(
                    self.output_dir / f"{variant_names[next_i]}.lt"
                )
                bond_index += 1
                f.write(
                    f"      $bond:b{bond_index}  "
                    f"$atom:monomer[{i}]/{second_atom}  "
                    f"$atom:monomer[{next_i}]/{first_atom}\n"
                )
            f.write("    }\n")

            f.write(f"\n}} # {poly_id}\n")

    def _chain_anchors(
        self, variant_names: List[str], n: int
    ) -> Dict[str, str]:
        """
        Head/tail atom references (reserved for future grafting strategies).
        head = first atom of the first monomer; tail = second atom of the last.
        """
        first_atom, _ = read_lt_end_atoms(
            self.output_dir / f"{variant_names[0]}.lt"
        )
        _, second_atom = read_lt_end_atoms(
            self.output_dir / f"{variant_names[-1]}.lt"
        )
        return {
            "head": f"monomer[0]/{first_atom}",
            "tail": f"monomer[{n - 1}]/{second_atom}",
        }

    # ------------------------------------------------------------------
    # Small molecules
    # ------------------------------------------------------------------
    def _write_molecule_files(self, units: List[UnitSpec]) -> List[str]:
        """Type and write small-molecule .lt files; append molecule units."""
        written = []
        for entry in self.geometry["molecules"]:
            name = entry["name"]
            mol = mol_from_mapped_smiles(entry["smiles_mapped"])
            if mol is None:
                raise AtomTypingError(
                    f"Could not rebuild molecule '{name}' from mapped SMILES"
                )
            try:
                self.typer.assign_atom_types(mol)
            except Exception as e:
                raise AtomTypingError(
                    f"Atom typing failed for molecule '{name}' "
                    f"(force field {self.force_field}): {e}"
                ) from e

            # GAFF/GAFF2 have no charge table — Gasteiger on the molecule,
            # mirroring the full-chain policy for polymers.
            if self.force_field in ("gaff", "gaff2"):
                try:
                    AllChem.ComputeGasteigerCharges(mol)
                except Exception as e:
                    logger.error(
                        f"Failed to compute Gasteiger charges for molecule "
                        f"'{name}': {e}"
                    )

            lookup = {
                atom.GetAtomMapNum(): atom.GetIdx()
                for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0
            }
            rows = []
            for i, a in enumerate(entry["atoms"]):
                idx = lookup.get(a["map_num"])
                if idx is None:
                    raise ValidationError(
                        f"Molecule '{name}' atom {i} references missing map "
                        f"number {a['map_num']}"
                    )
                atom = mol.GetAtomWithIdx(idx)
                atom_type = self._atom_type(atom, a["element"])
                charge = self._atom_charge(atom, atom_type)
                rows.append({
                    "atom_id": f"{a['element']}{i + 1}",
                    "atom_type": atom_type,
                    "charge": charge,
                    "coords": a["coords"],
                })

            self._write_monomer_lt(name, {"bonds": entry["bonds"]}, rows)
            lt_file = f"{name}.lt"
            written.append(lt_file)
            units.append(UnitSpec(
                id=name,
                kind="molecule",
                lt_file=lt_file,
                count=entry["count"],
                radius=molecule_radius(entry["atoms"]),
                anchors={},
                role=entry.get("role", "film"),
                monomer_files=[lt_file],
            ))
        return written

    def _add_single_monomer_units(
        self, units: List[UnitSpec], dop1_counts: Dict[tuple, int]
    ) -> None:
        """DOP=1 polymer chains pack as molecule-like units per variant."""
        for (variant_name, role), count in dop1_counts.items():
            lt_file = f"{variant_name}.lt"
            units.append(UnitSpec(
                id=variant_name if role == "film" else f"{variant_name}__{role}",
                kind="molecule",
                lt_file=lt_file,
                count=count,
                radius=molecule_radius(self.geometry["variants"][variant_name]["atoms"]),
                anchors={},
                role=role,
                monomer_files=[lt_file],
            ))

    # ------------------------------------------------------------------
    # OpenFF (SMIRNOFF) typing
    #
    # SMIRNOFF assigns parameters per molecule (direct chemical perception),
    # not per fixed atom type, so moltemplate's "By Type" inference cannot
    # represent it. Instead every bonded term is written explicitly: variant
    # .lt files carry typed Data Bonds/Angles/Dihedrals/Impropers for their
    # internal terms, and poly_N.lt carries the boundary-spanning terms as
    # $atom:monomer[i]/ID references (the same pattern as moltemplate's own
    # genpoly_lt.py). The per-system parameter table lives in openff.lt,
    # rendered by OpenFFTyper.write_force_field_lt().
    # ------------------------------------------------------------------
    def _type_openff(self) -> UnitLibrary:
        """
        Type all variants/molecules with OpenFF and write explicit-term files.

        Returns:
            The UnitLibrary manifest (also saved to the build directory).
        """
        typer = self.typer  # OpenFFTyper
        geom = self.geometry

        # 1. Parameterize every chain graph once (chains of a model share the
        #    same graph, hence the same SMIRNOFF assignment).
        chain_mols: Dict[str, Chem.Mol] = {}
        typed_chains = {}
        for model_id, graph in geom["chain_graphs"].items():
            mol = mol_from_mapped_smiles(graph["smiles_mapped"])
            if mol is None:
                raise AtomTypingError(
                    f"Could not rebuild chain for {model_id} from mapped SMILES"
                )
            try:
                typed_chains[model_id] = typer.parameterize(mol)
            except Exception as e:
                raise AtomTypingError(
                    f"OpenFF parameterization failed for {model_id}: {e}"
                ) from e
            chain_mols[model_id] = mol

        # 2. Decompose chains into monomer occurrences, resolve variant
        #    assignment groups (splitting into per-position files when
        #    SMIRNOFF assigns position-dependent parameters), neutralize the
        #    written composition, then write the variant files.
        variant_files: List[str] = []
        decomp_by_model = {}
        for model_id, mol in chain_mols.items():
            pm = typed_chains[model_id]
            decomp = self._openff_decompose(model_id, mol, pm)
            self._openff_resolve_variant_groups(decomp, pm)
            self._openff_neutralize_composition(decomp, pm)
            variant_files.extend(self._openff_write_variants(decomp, pm))
            decomp_by_model[model_id] = decomp

        # 3. poly_N.lt per multi-monomer chain (with boundary terms)
        units, dop1_counts = self._openff_write_chain_files(decomp_by_model)

        # 4. Small molecules: parameterize and write fully explicit files
        molecule_files = self._openff_write_molecule_files(units)

        # 5. DOP=1 chains become molecule-like units grouped by variant
        self._add_single_monomer_units(units, dop1_counts)

        # 6. The per-system force-field file, rendered from the accumulated
        #    parameter tables; shipped via monomer_files so stage 3 copies it.
        ff_lt = FORCE_FIELD_REGISTRY["openff"]["lt_file"]
        typer.write_force_field_lt(self.output_dir / ff_lt)

        library = UnitLibrary(
            force_field=self.force_field,
            units=units,
            monomer_files=[ff_lt] + sorted(set(variant_files + molecule_files)),
            build_config={
                "use_mc_chain_growth": self.geometry["mc_config"].get(
                    "use_mc_chain_growth", True
                ),
                "offset": self.geometry["mc_config"].get("offset", 4.0),
                "rotate": self.geometry["mc_config"].get("rotate", 90.0),
            },
            geometry_source=str(
                Path("..") / GEOMETRY_DIRNAME / "geometry.json"
            ),
        )
        library.save(self.output_dir)
        library.validate(self.output_dir)
        library.source_dir = str(self.output_dir)
        logger.info(
            f"Typing complete ({self.force_field}): {len(units)} units, "
            f"{len(library.monomer_files)} files -> {self.output_dir}"
        )
        return library

    # ------------------------------------------------------------------
    # Chain decomposition (OpenFF)
    # ------------------------------------------------------------------
    def _openff_decompose(self, model_id: str, mol: Chem.Mol, pm) -> Dict[str, Any]:
        """
        Split a typed chain into monomer occurrences and classify its terms.

        Inter-monomer bonds are recovered from the inter-bond marker map
        numbers (INTER_BOND_MAP_START..GEOM_MAP_BASE), paired by bond
        adjacency. Removing them yields the monomer fragments; chain order
        follows the marker numbers.

        Returns:
            Dict with "occurrences" (per-position dicts with a
            chain_idx -> lt_idx map), "internal" (per-position internal terms
            in lt indexing), "boundary" (terms spanning monomers, as
            (kind, [(pos, lt_idx)...], param_id)) and "resolved" (filled by
            _openff_write_variants: position -> variant file base name).
        """
        geom = self.geometry
        chains = [c for c in geom["chains"] if c["model_id"] == model_id]
        if not chains:
            raise ValidationError(f"No chains reference {model_id}")
        placements = chains[0]["placements"]
        n = len(placements)

        map_of = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}

        def is_marker(m: int) -> bool:
            return INTER_BOND_MAP_START <= m < GEOM_MAP_BASE

        def is_boundary_bond(i: int, j: int) -> bool:
            """
            True for inter-monomer bonds. ChainBuilder marks boundary b with
            (100+2b on the left monomer's right connection, 101+2b on the
            right monomer's left connection), so a boundary bond pairs an
            even-offset marker with the consecutive odd one. A bond between a
            monomer's own two connection atoms (e.g. [*]CC([*])...) pairs an
            odd marker with the NEXT even one and is correctly excluded.
            """
            m1, m2 = map_of[i], map_of[j]
            if not (is_marker(m1) and is_marker(m2)):
                return False
            lo, hi = min(m1, m2), max(m1, m2)
            return (lo - INTER_BOND_MAP_START) % 2 == 0 and hi == lo + 1

        boundary_bonds = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            for b in mol.GetBonds()
            if is_boundary_bond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        ]
        if len(boundary_bonds) != max(n - 1, 0):
            raise ValidationError(
                f"Chain {model_id}: expected {max(n - 1, 0)} inter-monomer "
                f"marker bonds, found {len(boundary_bonds)}"
            )

        if boundary_bonds:
            rw = Chem.RWMol(mol)
            for i, j in boundary_bonds:
                rw.RemoveBond(i, j)
            frags = Chem.GetMolFrags(rw.GetMol(), sanitizeFrags=False)
        else:
            frags = (tuple(range(mol.GetNumAtoms())),)

        if len(frags) != n:
            raise ValidationError(
                f"Chain {model_id}: decomposition gave {len(frags)} fragments "
                f"for {n} placements"
            )

        def frag_key(frag) -> int:
            ms = [map_of[i] for i in frag if is_marker(map_of[i])]
            return min(ms) if ms else 0

        ordered = sorted(frags, key=frag_key)

        occurrences = []
        for pos, frag in enumerate(ordered):
            name = placements[pos]["variant"]
            entry = geom["variants"][name]
            base = entry.get("t1_variant_of")
            if base:
                name, entry = base, geom["variants"][base]
            occ_map, is_rep = self._openff_occurrence_map(mol, frag, entry)
            occurrences.append({
                "pos": pos,
                "base_name": name,
                "entry": entry,
                "indices": tuple(frag),
                "map": occ_map,
                "is_rep": is_rep,
            })

        comp_of: Dict[int, int] = {}
        lt_of: Dict[int, int] = {}
        for occ in occurrences:
            for ci, lt in occ["map"].items():
                comp_of[ci] = occ["pos"]
                lt_of[ci] = lt
        if len(comp_of) != pm.n_atoms:
            raise ValidationError(
                f"Chain {model_id}: decomposition covers {len(comp_of)} of "
                f"{pm.n_atoms} atoms"
            )

        internal = [
            {"bonds": [], "angles": [], "dihedrals": [], "impropers": []}
            for _ in range(n)
        ]
        boundary: List[Tuple[str, List[Tuple[int, int]], str]] = []

        def put(kind: str, idxs: Tuple[int, ...], pid: str) -> None:
            poss = [comp_of[c] for c in idxs]
            lts = [lt_of[c] for c in idxs]
            if len(set(poss)) == 1:
                internal[poss[0]][kind].append((*lts, pid))
            else:
                boundary.append((kind, list(zip(poss, lts)), pid))

        for i, j, pid in pm.bonds:
            put("bonds", (i, j), pid)
        for i, j, k, pid in pm.angles:
            put("angles", (i, j, k), pid)
        for i, j, k, l, pid in pm.propers:
            put("dihedrals", (i, j, k, l), pid)
        for i, j, k, l, pid in pm.impropers:
            put("impropers", (i, j, k, l), pid)

        return {
            "occurrences": occurrences,
            "internal": internal,
            "boundary": boundary,
            "resolved": {},
        }

    def _openff_occurrence_map(
        self, mol: Chem.Mol, frag, entry: Dict[str, Any]
    ) -> Tuple[Dict[int, int], bool]:
        """
        chain_idx -> lt atom index for one monomer occurrence, plus whether
        this is the representative occurrence (its atoms carry the map numbers
        stored in the geometry entry).
        """
        stored = {a["map_num"]: j for j, a in enumerate(entry["atoms"])}
        frag_maps = {mol.GetAtomWithIdx(i).GetAtomMapNum() for i in frag}
        if set(stored) == frag_maps:
            # Representative occurrence: stored map numbers are its own.
            return {
                idx: stored[mol.GetAtomWithIdx(idx).GetAtomMapNum()]
                for idx in frag
            }, True
        return self._openff_match_occurrence(mol, frag, entry), False

    def _openff_match_occurrence(
        self, mol: Chem.Mol, frag, entry: Dict[str, Any]
    ) -> Dict[int, int]:
        """
        Map a non-representative occurrence onto its variant by an anchored
        topology match (elements + connectivity, all bonds single). Connection
        atoms pin the match: on the fragment side they carry the inter-bond
        markers (even marker = right connection, odd = left); on the variant
        side the geometry entry stores their LT indices in connection_atoms.

        Residual automorphisms (e.g. the two H of a CH2) are harmless: atoms
        related by a graph automorphism get identical SMIRNOFF parameters and
        identical NAGL charges.
        """
        vt = entry["variant_type"]
        conns = entry.get("connection_atoms") or {}
        anchor_lt: Dict[str, int] = {}
        if vt in ("middle", "ring", "last") and conns.get("left") is not None:
            anchor_lt["left"] = conns["left"]
        if vt in ("middle", "ring", "first") and conns.get("right") is not None:
            anchor_lt["right"] = conns["right"]

        anchor_frag: Dict[str, int] = {}
        for idx in frag:
            m = mol.GetAtomWithIdx(idx).GetAtomMapNum()
            if INTER_BOND_MAP_START <= m < GEOM_MAP_BASE:
                side = "right" if (m - INTER_BOND_MAP_START) % 2 == 0 else "left"
                anchor_frag[side] = idx
        if set(anchor_lt) != set(anchor_frag):
            raise ValidationError(
                f"Connection-atom mismatch for variant "
                f"'{entry.get('variant_type', '?')}': lt anchors "
                f"{sorted(anchor_lt)} vs fragment anchors {sorted(anchor_frag)}"
            )

        query = self._topology_mol_from_entry(entry)
        target, local_of_chain = self._topology_mol_from_fragment(mol, frag)
        matches = target.GetSubstructMatches(
            query, uniquify=False, useChirality=False
        )
        for match in matches:
            if all(
                local_of_chain[match[lt_idx]] == anchor_frag[side]
                for side, lt_idx in anchor_lt.items()
            ):
                return {
                    local_of_chain[match[lt_idx]]: lt_idx
                    for lt_idx in range(len(entry["atoms"]))
                }
        raise ValidationError(
            "Could not map a monomer occurrence onto its variant entry "
            "(no anchored topology match); variant chemistry differs between "
            "chain positions?"
        )

    @staticmethod
    def _topology_mol_from_entry(entry: Dict[str, Any]) -> Chem.Mol:
        """Topology-only RDKit mol of a geometry variant entry (single bonds)."""
        pt = Chem.GetPeriodicTable()
        rw = Chem.RWMol()
        for a in entry["atoms"]:
            rw.AddAtom(Chem.Atom(pt.GetAtomicNumber(a["element"])))
        for i, j in entry["bonds"]:
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        return rw.GetMol()

    @staticmethod
    def _topology_mol_from_fragment(mol: Chem.Mol, frag):
        """Topology-only subgraph mol of a chain fragment (single bonds)."""
        rw = Chem.RWMol()
        local_of_chain = {}
        pos_of = {}
        for li, ci in enumerate(frag):
            rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(ci).GetAtomicNum()))
            local_of_chain[li] = ci
            pos_of[ci] = li
        frag_set = set(frag)
        for ci in frag:
            for bond in mol.GetAtomWithIdx(ci).GetBonds():
                cj = bond.GetOtherAtomIdx(ci)
                if cj in frag_set and cj > ci:
                    rw.AddBond(pos_of[ci], pos_of[cj], Chem.BondType.SINGLE)
        return rw.GetMol(), local_of_chain

    # ------------------------------------------------------------------
    # Variant .lt files (OpenFF)
    # ------------------------------------------------------------------
    @staticmethod
    def _openff_canon(kind: str, idxs) -> Tuple:
        """Canonicalize a term tuple for cross-occurrence comparison."""
        if kind == "bonds":
            return tuple(sorted(idxs))
        if kind == "angles":
            return (idxs[1], min(idxs[0], idxs[2]), max(idxs[0], idxs[2]))
        if kind == "dihedrals":
            return min(tuple(idxs), tuple(reversed(idxs)))
        # impropers: SMIRNOFF order, central atom is second; arms canonicalized
        return (idxs[1], tuple(sorted((idxs[0], idxs[2], idxs[3]))))

    def _openff_occurrence_signature(self, occ, decomp, pm):
        """
        (atom type ids in LT order, canonicalized internal term set) — the
        equivalence key for sharing one variant .lt across occurrences.
        Charges are deliberately NOT part of the signature: NAGL charges
        differ slightly (~1e-3 e) between occurrences of a deduplicated
        variant by construction; the representative's charges are reused
        (the same sharing the SMARTS-typed paths already do).
        """
        inv = {lt: ci for ci, lt in occ["map"].items()}
        n_atoms = len(occ["entry"]["atoms"])
        types = tuple(pm.vdw_ids[inv[j]] for j in range(n_atoms))
        terms = frozenset(
            (kind, self._openff_canon(kind, idxs), pid)
            for kind, lst in decomp["internal"][occ["pos"]].items()
            for *idxs, pid in lst
        )
        return (types, terms)

    def _openff_resolve_variant_groups(self, decomp, pm) -> None:
        """
        Group each variant's occurrences by SMIRNOFF assignment signature and
        decide which files will be written. Fills decomp entries:

        - "resolved": position -> variant file base name (no _T1 suffix)
        - "groups": list of {"file_base", "base_name", "use_occ", "positions"}

        A deduplicated variant normally yields one group; SMIRNOFF can assign
        position-dependent parameters (e.g. near block junctions), in which
        case non-representative groups get <name>_p<pos> files. Grouping is by
        exact atom-type + parameter-id signature only — never by charge deltas
        (see _openff_occurrence_signature).
        """
        by_variant: Dict[str, List] = {}
        for occ in decomp["occurrences"]:
            by_variant.setdefault(occ["base_name"], []).append(occ)

        groups_out = []
        for base_name, occs in by_variant.items():
            reps = [o for o in occs if o["is_rep"]]
            if len(reps) != 1:
                raise ValidationError(
                    f"Variant '{base_name}' has {len(reps)} representative "
                    "occurrences (expected 1)"
                )
            rep = reps[0]
            groups: Dict[Any, List] = {}
            for occ in occs:
                sig = self._openff_occurrence_signature(occ, decomp, pm)
                groups.setdefault(sig, []).append(occ)
            if len(groups) > 1:
                logger.warning(
                    f"OpenFF: variant '{base_name}' got {len(groups)} distinct "
                    "parameter assignments across chain positions; writing "
                    "separate .lt files per assignment group"
                )
            for members in groups.values():
                if any(o["pos"] == rep["pos"] for o in members):
                    file_base, use_occ = base_name, rep
                else:
                    file_base = f"{base_name}_p{members[0]['pos']}"
                    use_occ = members[0]
                positions = [o["pos"] for o in members]
                for pos in positions:
                    decomp["resolved"][pos] = file_base
                groups_out.append({
                    "file_base": file_base,
                    "base_name": base_name,
                    "use_occ": use_occ,
                    "positions": positions,
                })
        decomp["groups"] = groups_out

    def _openff_neutralize_composition(self, decomp, pm) -> None:
        """
        Neutralize the *written* chain composition to the formal charge.

        parameterize() neutralizes the typed chain's own charges, but variant
        files share one occurrence's charges across all positions of a group,
        so the composition sum drifts by the inter-occurrence charge delta
        (~1e-3 e per monomer). Fix by shifting the charges of exactly the
        atoms whose values get written (each group's representative atoms),
        once per written atom, so the composed chain total is exact.
        """
        total = 0.0
        n_atoms = 0
        for g in decomp["groups"]:
            occ = g["use_occ"]
            inv = {lt: ci for ci, lt in occ["map"].items()}
            qs = [pm.charges[inv[j]] for j in range(len(occ["entry"]["atoms"]))]
            total += len(g["positions"]) * float(np.sum(qs))
            n_atoms += len(g["positions"]) * len(qs)
        drift = total - pm.formal_charge
        if abs(drift) < 1e-6 or n_atoms == 0:
            return
        shift = drift / n_atoms
        for g in decomp["groups"]:
            for ci in g["use_occ"]["map"]:
                pm.charges[ci] -= shift
        logger.info(
            f"OpenFF: composition-level charge neutralization by "
            f"{-shift:+.3e} e/atom (written-chain drift {drift:+.5f} e)"
        )

    def _openff_write_variants(self, decomp, pm) -> List[str]:
        """Write the variant .lt files decided by _openff_resolve_variant_groups."""
        geom = self.geometry
        written: List[str] = []
        for g in decomp["groups"]:
            file_base = g["file_base"]
            self._openff_write_variant_file(file_base, g["use_occ"], decomp, pm)
            written.append(f"{file_base}.lt")
            t1_entry = geom["variants"].get(f"{g['base_name']}_T1")
            if t1_entry is not None:
                self._openff_write_variant_file(
                    f"{file_base}_T1", g["use_occ"], decomp, pm,
                    atoms_override=t1_entry["atoms"],
                )
                written.append(f"{file_base}_T1.lt")
        return written

    def _openff_write_variant_file(
        self, file_base: str, occ, decomp, pm, atoms_override=None
    ) -> None:
        """Write one variant .lt from one occurrence's SMIRNOFF assignment."""
        entry = occ["entry"]
        atoms = atoms_override if atoms_override is not None else entry["atoms"]
        inv = {lt: ci for ci, lt in occ["map"].items()}
        n_atoms = len(entry["atoms"])
        vdw_ids = [pm.vdw_ids[inv[j]] for j in range(n_atoms)]
        charges = [pm.charges[inv[j]] for j in range(n_atoms)]
        terms = decomp["internal"][occ["pos"]]
        self._write_openff_monomer_lt(
            self.output_dir / f"{file_base}.lt", atoms, vdw_ids, charges, terms
        )

    def _write_openff_monomer_lt(
        self,
        path: Path,
        atoms: List[Dict[str, Any]],
        vdw_ids: List[str],
        charges: List[float],
        terms: Dict[str, List],
    ) -> None:
        """
        Write one explicit-terms .lt file (monomer variant or small molecule).

        Data Atoms rows keep the geometry entry's LT order (connection atoms
        first) and the f"{element}{i+1}" id convention. All bonded terms are
        explicit and typed; impropers are trefoil-expanded (3 cvff entries per
        SMIRNOFF improper, central atom first).
        """
        typer = self.typer
        name = path.stem

        def aid(idx: int) -> str:
            return f"{atoms[idx]['element']}{idx + 1}"

        with open(path, "w") as f:
            write_lt_header(f, name, "openff")
            f.write('  write("Data Atoms") {\n')
            for j, a in enumerate(atoms):
                x, y, z = a["coords"]
                f.write(
                    f"\t$atom:{aid(j)} $mol:... {typer.atom_type_name(vdw_ids[j])} "
                    f"{charges[j]:.6f}    {x:.3f}   {y:.3f}   {z:.3f}\n"
                )
            f.write("  }\n\n")

            if terms["bonds"]:
                f.write('  write("Data Bonds") {\n')
                for c, (i, j, pid) in enumerate(terms["bonds"], 1):
                    f.write(
                        f"\t$bond:b{c} {typer.bond_type_name(pid)} "
                        f"$atom:{aid(i)} $atom:{aid(j)}\n"
                    )
                f.write("  }\n\n")

            if terms["angles"]:
                f.write('  write("Data Angles") {\n')
                for c, (i, j, k, pid) in enumerate(terms["angles"], 1):
                    f.write(
                        f"\t$angle:a{c} {typer.angle_type_name(pid)} "
                        f"$atom:{aid(i)} $atom:{aid(j)} $atom:{aid(k)}\n"
                    )
                f.write("  }\n\n")

            if terms["dihedrals"]:
                f.write('  write("Data Dihedrals") {\n')
                for c, (i, j, k, l, pid) in enumerate(terms["dihedrals"], 1):
                    f.write(
                        f"\t$dihedral:d{c} {typer.dihedral_type_name(pid)} "
                        f"$atom:{aid(i)} $atom:{aid(j)} $atom:{aid(k)} "
                        f"$atom:{aid(l)}\n"
                    )
                f.write("  }\n\n")

            if terms["impropers"]:
                f.write('  write("Data Impropers") {\n')
                c = 0
                for i, j, k, l, pid in terms["impropers"]:
                    for order in self._openff_trefoil((i, j, k, l)):
                        c += 1
                        refs = " ".join(f"$atom:{aid(x)}" for x in order)
                        f.write(
                            f"\t$improper:i{c} {typer.improper_type_name(pid)} "
                            f"{refs}\n"
                        )
                f.write("  }\n\n")

            write_lt_footer(f, name)

    @staticmethod
    def _openff_trefoil(idxs) -> List[Tuple[int, int, int, int]]:
        """
        SMIRNOFF improper (i, j-central, k, l) -> the three same-handedness
        LAMMPS cvff orderings (central atom first, cyclic arm permutations).
        """
        i, j, k, l = idxs
        return [(j, i, k, l), (j, k, l, i), (j, l, i, k)]

    # ------------------------------------------------------------------
    # poly_N.lt chain files (OpenFF)
    # ------------------------------------------------------------------
    def _openff_write_chain_files(self, decomp_by_model):
        """
        Write poly_N.lt for every multi-monomer chain (explicit typed
        boundary terms instead of the untyped Data Bond List).

        Returns:
            (units, dop1_counts): polymer UnitSpec list, and the
            (variant name, role) -> count dict for DOP=1 chains.
        """
        units: List[UnitSpec] = []
        dop1_counts: Dict[tuple, int] = {}

        poly_index = 0
        for chain in self.geometry["chains"]:
            role = chain.get("role", "film")
            placements = chain["placements"]
            n = len(placements)
            if n <= 1:
                # DOP=1: single-occurrence variant, never split; name as-is.
                variant_name = placements[0]["variant"]
                key = (variant_name, role)
                dop1_counts[key] = dop1_counts.get(key, 0) + 1
                continue

            poly_index += 1
            poly_id = f"poly_{poly_index}"
            decomp = decomp_by_model[chain["model_id"]]
            resolved = decomp["resolved"]

            # Remap placement variant names onto the written files, keeping
            # the _T1 spelling where the geometry uses it.
            remapped = []
            for pos, p in enumerate(placements):
                name = p["variant"]
                is_t1 = bool(self.geometry["variants"][name].get("t1_variant_of"))
                file_base = resolved[pos]
                remapped.append({
                    **p, "variant": file_base + ("_T1" if is_t1 else "")
                })
            variant_names = [p["variant"] for p in remapped]
            monomer_files = sorted({f"{v}.lt" for v in variant_names})

            self._write_openff_poly_lt(
                self.output_dir / f"{poly_id}.lt",
                poly_id, remapped, decomp,
            )

            anchors = self._openff_chain_anchors(decomp, n)
            units.append(UnitSpec(
                id=poly_id,
                kind="polymer",
                lt_file=f"{poly_id}.lt",
                count=1,
                topology=chain["topology"],
                n_monomers=chain["n_monomers"],
                radius=chain["radius"],
                anchors=anchors,
                role=role,
                monomer_files=monomer_files,
            ))

        return units, dop1_counts

    def _write_openff_poly_lt(
        self,
        path: Path,
        poly_id: str,
        placements: List[Dict[str, Any]],
        decomp: Dict[str, Any],
    ) -> None:
        """Write one poly_N.lt: monomer instantiations + typed boundary terms."""
        typer = self.typer
        variant_names = [p["variant"] for p in placements]
        occurrences = decomp["occurrences"]

        def aref(pos: int, lt_idx: int) -> str:
            # Atom ids follow the base entry's LT order (identical across the
            # base/T1/split files of a variant), taken from the occurrence —
            # split names are not keys of geometry["variants"].
            entry = occurrences[pos]["entry"]
            a = entry["atoms"][lt_idx]
            return f"$atom:monomer[{pos}]/{a['element']}{lt_idx + 1}"

        with open(path, "w") as f:
            f.write('import "openff.lt"\n')
            for name in dict.fromkeys(variant_names):
                f.write(f'import "{name}.lt"\n')
            f.write("\n")
            f.write(f"{poly_id} inherits OPENFF {{\n\n")
            f.write("    create_var {$mol}\n\n")

            for i, p in enumerate(placements):
                f.write(f"    monomer[{i}] = new {p['variant']}")
                if p["rotation"] is not None:
                    angle = p["rotation"]["angle_deg"]
                    ax, ay, az = p["rotation"]["axis"]
                    f.write(f".rot({angle:.4f},{ax:.4f},{ay:.4f},{az:.4f})")
                if p["translation"] is not None:
                    x, y, z = p["translation"]
                    f.write(f".move({x:.4f},{y:.4f},{z:.4f})")
                f.write("\n")
            f.write("\n")

            sections = [
                ("bonds", "Data Bonds", "$bond:b", typer.bond_type_name),
                ("angles", "Data Angles", "$angle:a", typer.angle_type_name),
                ("dihedrals", "Data Dihedrals", "$dihedral:d", typer.dihedral_type_name),
                ("impropers", "Data Impropers", "$improper:i", typer.improper_type_name),
            ]
            for kind, section, prefix, type_name in sections:
                terms = [t for t in decomp["boundary"] if t[0] == kind]
                if not terms:
                    continue
                f.write(f'    write("{section}") {{\n')
                c = 0
                for _, atoms, pid in terms:
                    orders = (
                        self._openff_trefoil_pairs(atoms)
                        if kind == "impropers"
                        else [atoms]
                    )
                    for ordered in orders:
                        c += 1
                        refs = " ".join(aref(p, lt) for p, lt in ordered)
                        f.write(f"      {prefix}{c} {type_name(pid)} {refs}\n")
                f.write("    }\n\n")

            f.write(f"}} # {poly_id}\n")

    @staticmethod
    def _openff_trefoil_pairs(atoms):
        """Trefoil permutations of (pos, lt_idx) quadruples (central second)."""
        a, b, c, d = atoms
        return [(b, a, c, d), (b, c, d, a), (b, d, a, c)]

    def _openff_chain_anchors(
        self, decomp: Dict[str, Any], n: int
    ) -> Dict[str, str]:
        """Head/tail atom references, computed in-memory from the occurrences."""
        def aid_of(pos: int, lt_idx: int) -> str:
            entry = decomp["occurrences"][pos]["entry"]
            a = entry["atoms"][lt_idx]
            return f"{a['element']}{lt_idx + 1}"

        return {
            "head": f"monomer[0]/{aid_of(0, 0)}",
            "tail": f"monomer[{n - 1}]/{aid_of(n - 1, 1)}",
        }

    # ------------------------------------------------------------------
    # Small molecules (OpenFF)
    # ------------------------------------------------------------------
    def _openff_write_molecule_files(self, units: List[UnitSpec]) -> List[str]:
        """Parameterize and write small-molecule .lt files (fully explicit)."""
        typer = self.typer
        written = []
        for entry in self.geometry["molecules"]:
            name = entry["name"]
            mol = mol_from_mapped_smiles(entry["smiles_mapped"])
            if mol is None:
                raise AtomTypingError(
                    f"Could not rebuild molecule '{name}' from mapped SMILES"
                )
            try:
                pm = typer.parameterize(mol)
            except Exception as e:
                raise AtomTypingError(
                    f"OpenFF parameterization failed for molecule '{name}': {e}"
                ) from e

            lookup = {
                atom.GetAtomMapNum(): atom.GetIdx()
                for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0
            }
            n_atoms = len(entry["atoms"])
            lt_of = {}
            for j, a in enumerate(entry["atoms"]):
                ci = lookup.get(a["map_num"])
                if ci is None:
                    raise ValidationError(
                        f"Molecule '{name}' atom {j} references missing map "
                        f"number {a['map_num']}"
                    )
                lt_of[ci] = j
            vdw_ids = [pm.vdw_ids[lookup[a["map_num"]]] for a in entry["atoms"]]
            charges = [pm.charges[lookup[a["map_num"]]] for a in entry["atoms"]]

            terms = {"bonds": [], "angles": [], "dihedrals": [], "impropers": []}
            for i, j, pid in pm.bonds:
                terms["bonds"].append((lt_of[i], lt_of[j], pid))
            for i, j, k, pid in pm.angles:
                terms["angles"].append((lt_of[i], lt_of[j], lt_of[k], pid))
            for i, j, k, l, pid in pm.propers:
                terms["dihedrals"].append(
                    (lt_of[i], lt_of[j], lt_of[k], lt_of[l], pid)
                )
            for i, j, k, l, pid in pm.impropers:
                terms["impropers"].append(
                    (lt_of[i], lt_of[j], lt_of[k], lt_of[l], pid)
                )

            self._write_openff_monomer_lt(
                self.output_dir / f"{name}.lt",
                entry["atoms"], vdw_ids, charges, terms,
            )
            lt_file = f"{name}.lt"
            written.append(lt_file)
            units.append(UnitSpec(
                id=name,
                kind="molecule",
                lt_file=lt_file,
                count=entry["count"],
                radius=molecule_radius(entry["atoms"]),
                anchors={},
                role=entry.get("role", "film"),
                monomer_files=[lt_file],
            ))
        return written
