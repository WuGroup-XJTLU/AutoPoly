#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force Field Management Module for AutoPoly Package

This module provides the ForceFieldManager class for managing force field
parameters and generating force field files for LAMMPS molecular dynamics simulations.

The ForceFieldManager class handles:
- OPLS-AA force field parameter management
- GAFF (General AMBER Force Field) support
- OPLS-AA subset generation based on monomer types
- Alkyl dihedral parameter modification
- Force field file generation for Moltemplate

Key Features:
- Support for OPLS-AA, LOPLS, and GAFF force fields
- Automatic parameter subset generation
- Dihedral coefficient customization
- Integration with Moltemplate force field system

Dependencies:
- Moltemplate: For generating LAMMPS data files
- OPLS-AA/GAFF force field parameters
- GAFFAnalyzer: For GAFF-specific analysis

Created on 2026-01-06
@author: zwu
"""
import os
from pathlib import Path
import subprocess
import shutil
import numpy as np
from typing import List, Optional, Set, Dict, Tuple
from ..core.system import logger
from .gaff_analysis import GAFFAnalyzer
from ..core.conf import FORCE_FIELD_REGISTRY
from ..core.exceptions import GenerationError, WorkflowError
from ..monomers.monomer_processing import (
    collect_monomer_files,
    extract_atom_types_from_lt,
    parse_bond_graph,
)


class ForceFieldManager:
    """
    Manages force field parameters and force field file generation.

    This class handles all force field-related operations including parameter
    subset generation, force field file creation, and parameter modification
    for LAMMPS molecular dynamics simulations.

    Attributes:
        path_cwd (str): Current working directory for the project
        path_master (str): Path to external dependencies
        path_moltemplatesrc (str): Path to Moltemplate source
        force_field (str): Force field type ("oplsaa", "gaff", "gaff2", "lopls",
            "dreiding", "compass", or "openff")
        ff_modify_dihedral (np.ndarray): Modified dihedral parameters
        path_oplsaaprm (str): Path to OPLS-AA force field parameters
        gaff_analyzer (GAFFAnalyzer): GAFF analyzer instance (for GAFF/GAFF2 force field)
    """

    def __init__(self, path_cwd: str, path_master: str, path_moltemplatesrc: str,
                 force_field: str, ff_modify_dihedral: np.ndarray,
                 monomer_files: Optional[List[str]] = None) -> None:
        """
        Initialize the ForceFieldManager.

        Args:
            path_cwd (str): Current working directory for the project
            path_master (str): Path to external dependencies
            path_moltemplatesrc (str): Path to Moltemplate source
            force_field (str): Force field type ("oplsaa", "gaff", or "lopls")
            ff_modify_dihedral (np.ndarray): Modified dihedral parameters
            monomer_files (list, optional): Explicit monomer .lt file paths used
                for parameter subsetting. If None, files are collected from the
                `model` attribute (legacy behavior).
        """
        self.path_cwd = path_cwd
        self.path_master = path_master
        self.path_moltemplatesrc = path_moltemplatesrc
        self.force_field = force_field
        self.ff_modify_dihedral = ff_modify_dihedral
        self.monomer_files = monomer_files

        # Set force field parameter path based on force_field type
        # All force fields now use .lt files from moltemplate/force_fields/
        # (openff excepted: its per-system openff.lt is generated at typing
        # time, so no bundled file exists under force_fields/)
        self.path_oplsaaprm = str(
            Path(self.path_master) / "moltemplate" / "force_fields"
            / FORCE_FIELD_REGISTRY[force_field]["lt_file"]
        )

        if force_field == "openff":
            logger.info(
                "\n'you are now using a generated SMIRNOFF parameter set "
                "(openff.lt from the typing stage)\n"
            )
        else:
            logger.info(f"\n'you are now using parameter set of {self.path_oplsaaprm}\n")

        # Initialize GAFF analyzer if using GAFF/GAFF2 force field
        self.gaff_analyzer = None
        if force_field in ("gaff", "gaff2"):
            self.gaff_analyzer = GAFFAnalyzer(self.path_cwd, self.path_master, force_field)

    def _get_monomer_files(self) -> List[str]:
        """
        Monomer .lt files driving parameter subsetting.

        Uses the explicit file list when provided (stage-3 BoxPacker path);
        otherwise collects files from the `model` attribute (legacy path).

        Raises:
            GenerationError: If neither source is available.
        """
        if self.monomer_files is not None:
            return list(self.monomer_files)
        if hasattr(self, 'model'):
            return collect_monomer_files(self.model, self.path_cwd)
        raise GenerationError(
            "No monomer source. Set ff_manager.model or pass monomer_files=..."
        )

    def make_force_field_lt(self) -> None:
        """
        Creates the force field .lt file based on the force_field type.

        This is the main entry point for force field file generation. It delegates
        to the appropriate method based on the force field type:
        - For OPENFF: verifies the per-system openff.lt generated by UnitTyper
          is in place (nothing to subset — SMIRNOFF parameters are per-molecule)
        - For GAFF/GAFF2: creates a filtered subset using GAFFAnalyzer
        - For DREIDING/COMPASS: creates a filtered subset based on monomer topology
        - For OPLS-AA/LOPLS: copies .lt file directly (new moltemplate 2.22.5 format)

        The new OPLS-AA 2024 .lt files include replace{} directives that automatically
        convert simple atom types (e.g., @atom:54) to extended format for wildcard
        matching (e.g., @atom:54_bCT_aCT_dCT_iCT).

        Raises:
            SystemExit: If force field file generation fails
        """
        try:
            if self.force_field == "openff":
                self.make_openff_lt()
            elif self.force_field in ("gaff", "gaff2"):
                self.make_gaff_lt()
            elif self.force_field == "dreiding":
                # DREIDING: create subset based on topology
                self.make_dreiding_subset()
            elif self.force_field == "compass":
                # COMPASS: create subset based on topology
                self.make_compass_subset()
            else:  # oplsaa or lopls
                # Create filtered subset like DREIDING/COMPASS
                self.make_oplsaa_lt_subset()

        except Exception as e:
            raise WorkflowError(f"Error in make_force_field_lt: {str(e)}") from e

    def make_openff_lt(self) -> None:
        """
        Verify the generated openff.lt is in place (OpenFF path).

        Unlike the static force fields, openff.lt is not a bundled parameter
        file: SMIRNOFF assigns parameters per molecule, so the file is
        generated by UnitTyper (stage 2) into the build directory and copied
        here (the moltemplate working dir) by BoxPacker via the monomer_files
        manifest. There is nothing to subset — this method only checks the
        hand-off happened.

        Raises:
            GenerationError: If openff.lt is missing (e.g. the build dir was
                not produced by UnitTyper(force_field='openff')).
        """
        dst = Path(self.path_cwd) / "openff.lt"
        if not dst.exists():
            raise GenerationError(
                f"openff.lt not found in {self.path_cwd}. With "
                "force_field='openff' the parameter file is generated during "
                "typing (UnitTyper) and copied by BoxPacker — do not run the "
                "force-field stage from a non-OpenFF build directory."
            )
        logger.info(
            f"  Using generated OpenFF parameter file: {dst} "
            f"({dst.stat().st_size} bytes)"
        )

    def make_gaff_lt(self) -> None:
        """
        Create GAFF force field file using GAFFAnalyzer.

        This method delegates to the GAFFAnalyzer class to analyze the topology
        used in the monomer files and creates a filtered version of gaff.lt
        containing only the relevant parameters.

        The method:
        1. Validates that GAFF analyzer is initialized
        2. Calls GAFFAnalyzer.create_gaff_subset() to analyze monomers
        3. Generates gaff.lt with only necessary parameters

        Raises:
            SystemExit: If GAFF analyzer is not initialized or analysis fails

        Note:
            This requires force_field="gaff" to be set during initialization.
            The model attribute must be set before calling this method.
        """
        if self.gaff_analyzer is None:
            raise GenerationError(
                "GAFF analyzer not initialized. Set force_field='gaff' to use this method."
            )

        self.gaff_analyzer.create_gaff_subset(
            getattr(self, 'model', None),
            monomer_files=self._get_monomer_files(),
        )

    def make_dreiding_subset(self) -> None:
        """
        Creates a subset of DREIDING parameters based on the monomer topology.

        DREIDING uses atom types like C_3, O_3, H with dihedrals defined by
        the two central atoms (outer atoms use wildcards). This method:
        1. Extracts atom types used in monomers
        2. Filters masses, pair_coeffs, bonds, angles by used types
        3. Filters dihedrals by actual central bond pairs (the main source of over-inclusion)
        4. Keeps generic improper definitions (only 3 types)

        Falls back to full dreiding.lt if errors occur or model is not set.
        """
        dreiding_src = Path(self.path_oplsaaprm)
        dreiding_dst = Path(self.path_cwd) / "dreiding.lt"

        if not dreiding_src.exists():
            raise GenerationError(f"DREIDING force field file not found: {dreiding_src}")

        try:
            logger.info("Creating DREIDING parameter subset...")

            # Step 1: Collect monomer files and extract atom types
            monomer_files = list(dict.fromkeys(self._get_monomer_files()))
            logger.info(f"  Found {len(monomer_files)} monomer files")

            # Extract atom types from monomers
            atom_types: Set[str] = set()
            for filepath in monomer_files:
                try:
                    atom_types.update(extract_atom_types_from_lt(filepath))
                except Exception as e:
                    logger.warning(f"  Could not parse {filepath}: {e}")

            logger.info(f"  Found {len(atom_types)} unique atom types: {sorted(atom_types)}")

            if not atom_types:
                raise GenerationError("  No atom types found in monomer files. Cannot create force field subset.")

            # Step 2: Build bond graph and extract bond pairs for dihedral filtering
            bond_pairs: Set[Tuple[str, str]] = set()
            for filepath in monomer_files:
                try:
                    graph = self._build_dreiding_bond_graph(filepath)
                    # Extract actual bond pairs (central bonds for dihedrals)
                    for atom_id, (atom_type, neighbors) in graph.items():
                        for neighbor_id in neighbors:
                            neighbor_type = graph[neighbor_id][0]
                            # Normalize pair
                            if atom_type <= neighbor_type:
                                bond_pairs.add((atom_type, neighbor_type))
                            else:
                                bond_pairs.add((neighbor_type, atom_type))
                except Exception as e:
                    logger.warning(f"  Could not build graph from {filepath}: {e}")

            logger.info(f"  Found {len(bond_pairs)} unique bond pairs")

            # Step 3: Parse dreiding.lt and filter sections
            sections = self._parse_dreiding_lt_sections(str(dreiding_src))

            # Step 4: Write filtered output
            with open(dreiding_dst, 'w') as f:
                # Header comment
                f.write("# DREIDING Force Field Subset\n")
                f.write(f"# Generated by AutoPoly from {dreiding_src.name}\n")
                f.write(f"# Atom types used: {', '.join(sorted(atom_types))}\n")
                f.write(f"# Total atom types: {len(atom_types)}\n\n")

                f.write("DREIDING {\n\n")

                # Write Init section unchanged
                f.write("  write_once(\"In Init\") {\n")
                for line in sections['init']:
                    f.write(f"    {line}\n")
                f.write("  } # End of init\n\n")

                # Filter and write Masses
                f.write("  write_once(\"Data Masses\") {\n")
                masses_count = 0
                for line in sections['masses']:
                    if '@atom:' in line:
                        atom_type = line.split('@atom:')[1].split()[0]
                        if self._dreiding_type_matches(atom_type, atom_types):
                            f.write(f"\t{line}\n")
                            masses_count += 1
                    else:
                        f.write(f"\t{line}\n")
                f.write("  } # End of masses\n\n")
                logger.info(f"  Masses: {masses_count} (from {len(sections['masses'])})")

                # Filter and write Pair Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                pair_count = 0
                for line in sections['pair_coeffs']:
                    if 'pair_coeff' in line and '@atom:' in line:
                        parts = line.split('@atom:')
                        if len(parts) >= 3:
                            type1 = parts[1].split()[0]
                            type2 = parts[2].split()[0]
                            if self._dreiding_type_matches(type1, atom_types) and \
                               self._dreiding_type_matches(type2, atom_types):
                                f.write(f"\t{line}\n")
                                pair_count += 1
                    else:
                        f.write(f"\t{line}\n")
                f.write("  } # End of pair_coeffs\n\n")
                logger.info(f"  Pair coeffs: {pair_count} (from {len(sections['pair_coeffs'])})")

                # Filter and write Bond definitions
                f.write("  write_once(\"Data Bonds By Type\") {\n")
                bond_def_count = 0
                kept_bonds: Set[str] = set()
                for line in sections['bond_defs']:
                    if '@bond:' in line and '@atom:' in line:
                        parts = line.split('@atom:')
                        if len(parts) >= 3:
                            type1 = parts[1].split()[0]
                            type2 = parts[2].split()[0]
                            if self._dreiding_type_matches(type1, atom_types) and \
                               self._dreiding_type_matches(type2, atom_types):
                                bond_name = line.split('@bond:')[1].split()[0]
                                kept_bonds.add(bond_name)
                                f.write(f"\t{line}\n")
                                bond_def_count += 1
                f.write("  } # End of bond defs\n\n")
                logger.info(f"  Bond defs: {bond_def_count} (from {len(sections['bond_defs'])})")

                # Filter and write Bond Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                bond_coeff_count = 0
                for line in sections['bond_coeffs']:
                    if 'bond_coeff' in line and '@bond:' in line:
                        bond_name = line.split('@bond:')[1].split()[0]
                        if bond_name in kept_bonds:
                            f.write(f"\t{line}\n")
                            bond_coeff_count += 1
                    else:
                        f.write(f"\t{line}\n")
                f.write("  } # End of bond_coeffs\n\n")
                logger.info(f"  Bond coeffs: {bond_coeff_count}")

                # Filter and write Angle definitions
                f.write("  write_once(\"Data Angles By Type\") {\n")
                angle_def_count = 0
                kept_angles: Set[str] = set()
                for line in sections['angle_defs']:
                    if '@angle:' in line and '@atom:' in line:
                        parts = line.split('@atom:')
                        if len(parts) >= 4:
                            type1 = parts[1].split()[0]
                            type2 = parts[2].split()[0]
                            type3 = parts[3].split()[0]
                            if self._dreiding_type_matches(type1, atom_types) and \
                               self._dreiding_type_matches(type2, atom_types) and \
                               self._dreiding_type_matches(type3, atom_types):
                                angle_name = line.split('@angle:')[1].split()[0]
                                kept_angles.add(angle_name)
                                f.write(f"\t{line}\n")
                                angle_def_count += 1
                f.write("  } # End of angle defs\n\n")
                logger.info(f"  Angle defs: {angle_def_count} (from {len(sections['angle_defs'])})")

                # Filter and write Angle Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                angle_coeff_count = 0
                for line in sections['angle_coeffs']:
                    if 'angle_coeff' in line and '@angle:' in line:
                        angle_name = line.split('@angle:')[1].split()[0]
                        if angle_name in kept_angles:
                            f.write(f"\t{line}\n")
                            angle_coeff_count += 1
                    else:
                        f.write(f"\t{line}\n")
                f.write("  } # End of angle_coeffs\n\n")
                logger.info(f"  Angle coeffs: {angle_coeff_count}")

                # Filter Dihedral definitions - this is the key reduction
                # DREIDING dihedrals use @atom:* for outer atoms, so filter by central pair
                f.write("  write_once(\"Data Dihedrals By Type\") {\n")
                dihedral_def_count = 0
                kept_dihedrals: Set[str] = set()
                for line in sections['dihedral_defs']:
                    if '@dihedral:' in line and '@atom:' in line:
                        parts = line.split('@atom:')
                        if len(parts) >= 5:
                            # DREIDING format: @dihedral:X-Y @atom:* @atom:X @atom:Y @atom:*
                            # Central atoms are at positions 2 and 3 (1-indexed)
                            type2 = parts[2].split()[0]
                            type3 = parts[3].split()[0]
                            # Check if this central bond pair exists in our topology
                            if self._dreiding_dihedral_matches(type2, type3, bond_pairs, atom_types):
                                dihedral_name = line.split('@dihedral:')[1].split()[0]
                                kept_dihedrals.add(dihedral_name)
                                f.write(f"\t{line}\n")
                                dihedral_def_count += 1
                f.write("  } # End of dihedral defs\n\n")
                logger.info(f"  Dihedral defs: {dihedral_def_count} (from {len(sections['dihedral_defs'])})")

                # Filter and write Dihedral Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                dihedral_coeff_count = 0
                for line in sections['dihedral_coeffs']:
                    if 'dihedral_coeff' in line and '@dihedral:' in line:
                        dihedral_name = line.split('@dihedral:')[1].split()[0]
                        if dihedral_name in kept_dihedrals:
                            f.write(f"\t{line}\n")
                            dihedral_coeff_count += 1
                    else:
                        f.write(f"\t{line}\n")
                f.write("  } # End of dihedral_coeffs\n\n")
                logger.info(f"  Dihedral coeffs: {dihedral_coeff_count}")

                # Write Improper definitions (keep all - only 3 generic types)
                f.write("  write_once(\"Data Impropers By Type (cenIsortJKL.py)\") {\n")
                for line in sections['improper_defs']:
                    f.write(f"\t{line}\n")
                f.write("  } # End of improper defs\n\n")

                # Write Improper Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                for line in sections['improper_coeffs']:
                    f.write(f"\t{line}\n")
                f.write("  } # End of improper_coeffs\n\n")

                f.write("}  # DREIDING\n")

            # Calculate file size reduction
            original_size = dreiding_src.stat().st_size
            subset_size = dreiding_dst.stat().st_size
            reduction = (1 - subset_size / original_size) * 100

            logger.info(f"  Successfully created dreiding.lt subset")
            logger.info(f"  File size: {original_size} -> {subset_size} bytes ({reduction:.1f}% reduction)")

        except Exception as e:
            raise WorkflowError(f"Error in make_dreiding_subset: {str(e)}") from e

    def _build_dreiding_bond_graph(self, filepath: str) -> Dict[str, Tuple[str, List[str]]]:
        """Build a bond connectivity graph from a DREIDING monomer .lt file.

        Args:
            filepath: Path to monomer .lt file

        Returns:
            Dictionary mapping atom_id -> (atom_type, [neighbor_atom_ids])
        """
        return parse_bond_graph(filepath)

    def _parse_dreiding_lt_sections(self, dreiding_file: str) -> Dict[str, List[str]]:
        """Parse dreiding.lt file into sections.

        Args:
            dreiding_file: Path to dreiding.lt file

        Returns:
            Dict with section names as keys and list of content lines as values
        """
        sections = {
            'init': [],
            'masses': [],
            'pair_coeffs': [],
            'bond_defs': [],
            'bond_coeffs': [],
            'angle_defs': [],
            'angle_coeffs': [],
            'dihedral_defs': [],
            'dihedral_coeffs': [],
            'improper_defs': [],
            'improper_coeffs': []
        }

        current_section = None
        in_settings_for = None  # Track what type of coeffs we're in

        with open(dreiding_file, 'r') as f:
            for line in f:
                stripped = line.strip()

                # Skip empty lines and comments at top level
                if not stripped or stripped.startswith('#'):
                    continue

                # Section detection
                if 'write_once("In Init")' in stripped:
                    current_section = 'init'
                    continue
                elif 'write_once("Data Masses")' in stripped:
                    current_section = 'masses'
                    continue
                elif 'write_once("Data Bonds By Type")' in stripped:
                    current_section = 'bond_defs'
                    continue
                elif 'write_once("Data Angles By Type")' in stripped:
                    current_section = 'angle_defs'
                    continue
                elif 'write_once("Data Dihedrals By Type")' in stripped:
                    current_section = 'dihedral_defs'
                    continue
                elif 'write_once("Data Impropers By Type' in stripped:
                    current_section = 'improper_defs'
                    continue
                elif 'write_once("In Settings")' in stripped:
                    # Determine which coeff section based on what comes after
                    current_section = 'in_settings'
                    continue

                # End of section
                if stripped.startswith('}') and current_section:
                    if current_section == 'in_settings':
                        in_settings_for = None
                    current_section = None
                    continue

                # Content lines
                if current_section == 'init':
                    sections['init'].append(stripped)
                elif current_section == 'masses':
                    if '@atom:' in stripped:
                        sections['masses'].append(stripped)
                elif current_section == 'bond_defs':
                    if '@bond:' in stripped:
                        sections['bond_defs'].append(stripped)
                elif current_section == 'angle_defs':
                    if '@angle:' in stripped:
                        sections['angle_defs'].append(stripped)
                elif current_section == 'dihedral_defs':
                    if '@dihedral:' in stripped:
                        sections['dihedral_defs'].append(stripped)
                elif current_section == 'improper_defs':
                    if '@improper:' in stripped:
                        sections['improper_defs'].append(stripped)
                elif current_section == 'in_settings':
                    # Classify based on content
                    if 'pair_coeff' in stripped:
                        sections['pair_coeffs'].append(stripped)
                    elif 'bond_coeff' in stripped:
                        sections['bond_coeffs'].append(stripped)
                    elif 'angle_coeff' in stripped:
                        sections['angle_coeffs'].append(stripped)
                    elif 'dihedral_coeff' in stripped:
                        sections['dihedral_coeffs'].append(stripped)
                    elif 'improper_coeff' in stripped:
                        sections['improper_coeffs'].append(stripped)

        return sections

    def _dreiding_type_matches(self, pattern: str, used_types: Set[str]) -> bool:
        """Check if a DREIDING atom type pattern matches any used types.

        DREIDING uses wildcards like C_3*, O_3*, etc. in pair_coeff.

        Args:
            pattern: Atom type pattern (may contain * wildcard)
            used_types: Set of atom types actually used

        Returns:
            True if pattern matches at least one used type
        """
        if pattern == '*':
            return True  # Universal wildcard

        if '*' in pattern:
            # Pattern like C_3* matches C_3, C_32, C_33, etc.
            prefix = pattern.rstrip('*')
            for used_type in used_types:
                if used_type.startswith(prefix):
                    return True
            return False
        else:
            # Exact match
            return pattern in used_types

    def _dreiding_dihedral_matches(self, type2: str, type3: str,
                                   bond_pairs: Set[Tuple[str, str]],
                                   atom_types: Set[str]) -> bool:
        """Check if a DREIDING dihedral's central bond pair matches our topology.

        DREIDING dihedrals use wildcards for outer atoms, so we only need to check
        the central bond pair.

        Args:
            type2: Central atom type 2 (may have wildcard like C_3*)
            type3: Central atom type 3 (may have wildcard like O_3*)
            bond_pairs: Set of actual bond pairs in our topology
            atom_types: Set of atom types used

        Returns:
            True if this dihedral's central bond exists in our topology
        """
        # First check if types match any used atom types
        type2_matches = self._dreiding_type_matches(type2, atom_types)
        type3_matches = self._dreiding_type_matches(type3, atom_types)

        if not (type2_matches and type3_matches):
            return False

        # Check if any actual bond pair matches this pattern
        for pair in bond_pairs:
            p1, p2 = pair
            # Check both directions
            if (self._pattern_matches_type(type2, p1) and self._pattern_matches_type(type3, p2)) or \
               (self._pattern_matches_type(type2, p2) and self._pattern_matches_type(type3, p1)):
                return True
        return False

    def _pattern_matches_type(self, pattern: str, actual_type: str) -> bool:
        """Check if a pattern matches an actual atom type.

        Args:
            pattern: Pattern (may have * wildcard)
            actual_type: Actual atom type

        Returns:
            True if pattern matches the type
        """
        if pattern == '*':
            return True
        if '*' in pattern:
            prefix = pattern.rstrip('*')
            return actual_type.startswith(prefix)
        return pattern == actual_type

    def make_compass_subset(self) -> None:
        """
        Creates a subset of COMPASS parameters based on the monomer topology.

        COMPASS (class2 force field) uses atom types like c4, o2e, h1 with complex
        cross-terms. This method filters parameters by actual topology.

        Falls back to full compass_published.lt if errors occur or model is not set.
        """
        compass_src = Path(self.path_oplsaaprm)
        compass_dst = Path(self.path_cwd) / "compass_published.lt"

        if not compass_src.exists():
            raise GenerationError(f"COMPASS force field file not found: {compass_src}")

        try:
            logger.info("Creating COMPASS parameter subset...")

            # Step 1: Collect monomer files and extract atom types
            monomer_files = list(dict.fromkeys(self._get_monomer_files()))
            logger.info(f"  Found {len(monomer_files)} monomer files")

            # Extract atom types from monomers
            atom_types: Set[str] = set()
            for filepath in monomer_files:
                try:
                    atom_types.update(extract_atom_types_from_lt(filepath))
                except Exception as e:
                    logger.warning(f"  Could not parse {filepath}: {e}")

            logger.info(f"  Found {len(atom_types)} unique atom types")

            if not atom_types:
                raise GenerationError("  No atom types found in monomer files. Cannot create force field subset.")

            # Step 2: Build bond graph for topology filtering
            bond_pairs: Set[str] = set()  # Store as "type1~type2" for comparison
            for filepath in monomer_files:
                try:
                    graph = self._build_compass_bond_graph(filepath)
                    for atom_id, (atom_type, neighbors) in graph.items():
                        for neighbor_id in neighbors:
                            neighbor_type = graph[neighbor_id][0]
                            # Create normalized bond pair key
                            if atom_type <= neighbor_type:
                                bond_pairs.add(f"{atom_type}~{neighbor_type}")
                            else:
                                bond_pairs.add(f"{neighbor_type}~{atom_type}")
                except Exception as e:
                    logger.warning(f"  Could not build graph from {filepath}: {e}")

            logger.info(f"  Found {len(bond_pairs)} unique bond pairs")

            # Step 3: Parse and filter compass_published.lt
            # COMPASS uses complex extended atom type format with ~p, ~b, ~a, ~d, ~i fields
            # We need to match based on the 'p' (pair) field primarily
            sections = self._parse_compass_lt_sections(str(compass_src))

            # Extract the core pair types used (the ~p field)
            core_pair_types = self._extract_compass_core_types(atom_types)
            logger.info(f"  Core pair types: {sorted(core_pair_types)}")

            # Extract class mappings from replace{} block
            class_mappings = self._extract_compass_class_mappings(sections['replace_block'], core_pair_types)
            logger.info(f"  Class mappings: {class_mappings}")

            # Get the actual class sets for filtering
            bond_classes = self._get_compass_bond_classes(core_pair_types, class_mappings)
            angle_classes = self._get_compass_angle_classes(core_pair_types, class_mappings)
            dihedral_classes = self._get_compass_dihedral_classes(core_pair_types, class_mappings)
            improper_classes = self._get_compass_improper_classes(core_pair_types, class_mappings)

            logger.info(f"  Bond classes: {sorted(bond_classes)}")
            logger.info(f"  Angle classes: {sorted(angle_classes)}")
            logger.info(f"  Dihedral classes: {sorted(dihedral_classes)}")
            logger.info(f"  Improper classes: {sorted(improper_classes)}")

            # Step 4: Write filtered output
            with open(compass_dst, 'w') as f:
                # Header
                f.write("# COMPASS Force Field Subset\n")
                f.write(f"# Generated by AutoPoly from {compass_src.name}\n")
                f.write(f"# Total core atom types: {len(core_pair_types)}\n\n")

                # Write replace{} block (always include - needed for type expansion)
                if sections['replace_block']:
                    f.write("COMPASS {\n\n")
                    f.write("  # Atom type expansion rules\n")
                    f.write("  replace{\n")
                    for line in sections['replace_block']:
                        # Only include replace rules for used core types
                        if self._compass_replace_matches(line, core_pair_types):
                            f.write(f"    {line}\n")
                    f.write("  }\n\n")
                else:
                    f.write("COMPASS {\n\n")

                # Write Init section
                f.write("  write_once(\"In Init\") {\n")
                for line in sections['init']:
                    f.write(f"    {line}\n")
                f.write("  } # End of init\n\n")

                # Write Masses (filtered)
                f.write("  write_once(\"Data Masses\") {\n")
                masses_count = 0
                for line in sections['masses']:
                    if '@atom:' in line:
                        if self._compass_type_matches(line, core_pair_types):
                            f.write(f"    {line}\n")
                            masses_count += 1
                f.write("  } # End of masses\n\n")
                logger.info(f"  Masses: {masses_count} (from {len(sections['masses'])})")

                # Write Pair Coeffs (filtered)
                f.write("  write_once(\"In Settings\") {\n")
                pair_count = 0
                for line in sections['pair_coeffs']:
                    if 'pair_coeff' in line:
                        if self._compass_type_matches(line, core_pair_types):
                            f.write(f"    {line}\n")
                            pair_count += 1
                f.write("  } # End of pair_coeffs\n\n")
                logger.info(f"  Pair coeffs: {pair_count} (from {len(sections['pair_coeffs'])})")

                # Write Charge By Bond (filtered)
                if sections['charge_by_bond']:
                    f.write("  write_once(\"Data Charge By Bond\") {\n")
                    charge_count = 0
                    for line in sections['charge_by_bond']:
                        if self._compass_charge_matches_v2(line, bond_classes):
                            f.write(f"    {line}\n")
                            charge_count += 1
                    f.write("  } # End of charge by bond\n\n")
                    logger.info(f"  Charge by bond: {charge_count} (from {len(sections['charge_by_bond'])})")

                # Write Bond defs and coeffs (filtered)
                f.write("  write_once(\"Data Bonds By Type\") {\n")
                bond_count = 0
                kept_bonds: Set[str] = set()
                for line in sections['bond_defs']:
                    if '@bond:' in line:
                        if self._compass_bond_matches_v2(line, bond_classes):
                            bond_name = line.split('@bond:')[1].split()[0]
                            kept_bonds.add(bond_name)
                            f.write(f"    {line}\n")
                            bond_count += 1
                f.write("  } # End of bond defs\n\n")
                logger.info(f"  Bond defs: {bond_count} (from {len(sections['bond_defs'])})")

                f.write("  write_once(\"In Settings\") {\n")
                for line in sections['bond_coeffs']:
                    if 'bond_coeff' in line and '@bond:' in line:
                        bond_name = line.split('@bond:')[1].split()[0]
                        if bond_name in kept_bonds:
                            f.write(f"    {line}\n")
                f.write("  } # End of bond_coeffs\n\n")

                # Write Angle defs and coeffs (filtered)
                f.write("  write_once(\"Data Angles By Type\") {\n")
                angle_count = 0
                kept_angles: Set[str] = set()
                for line in sections['angle_defs']:
                    if '@angle:' in line:
                        if self._compass_angle_matches_v2(line, angle_classes):
                            angle_name = line.split('@angle:')[1].split()[0]
                            kept_angles.add(angle_name)
                            f.write(f"    {line}\n")
                            angle_count += 1
                f.write("  } # End of angle defs\n\n")
                logger.info(f"  Angle defs: {angle_count} (from {len(sections['angle_defs'])})")

                f.write("  write_once(\"In Settings\") {\n")
                for line in sections['angle_coeffs']:
                    if 'angle_coeff' in line and '@angle:' in line:
                        angle_name = line.split('@angle:')[1].split()[0]
                        if angle_name in kept_angles:
                            f.write(f"    {line}\n")
                f.write("  } # End of angle_coeffs\n\n")

                # Write Dihedral defs and coeffs (filtered)
                f.write("  write_once(\"Data Dihedrals By Type\") {\n")
                dihedral_count = 0
                kept_dihedrals: Set[str] = set()
                for line in sections['dihedral_defs']:
                    if '@dihedral:' in line:
                        if self._compass_dihedral_matches_v2(line, dihedral_classes):
                            dihedral_name = line.split('@dihedral:')[1].split()[0]
                            kept_dihedrals.add(dihedral_name)
                            f.write(f"    {line}\n")
                            dihedral_count += 1
                f.write("  } # End of dihedral defs\n\n")
                logger.info(f"  Dihedral defs: {dihedral_count} (from {len(sections['dihedral_defs'])})")

                f.write("  write_once(\"In Settings\") {\n")
                for line in sections['dihedral_coeffs']:
                    if 'dihedral_coeff' in line and '@dihedral:' in line:
                        dihedral_name = line.split('@dihedral:')[1].split()[0]
                        if dihedral_name in kept_dihedrals:
                            f.write(f"    {line}\n")
                f.write("  } # End of dihedral_coeffs\n\n")

                # Write Improper defs and coeffs (filtered)
                f.write("  write_once(\"Data Impropers By Type (cenJsortIKL)\") {\n")
                improper_count = 0
                kept_impropers: Set[str] = set()
                for line in sections['improper_defs']:
                    if '@improper:' in line:
                        if self._compass_improper_matches_v2(line, improper_classes):
                            improper_name = line.split('@improper:')[1].split()[0]
                            kept_impropers.add(improper_name)
                            f.write(f"    {line}\n")
                            improper_count += 1
                f.write("  } # End of improper defs\n\n")
                logger.info(f"  Improper defs: {improper_count} (from {len(sections['improper_defs'])})")

                f.write("  write_once(\"In Settings\") {\n")
                for line in sections['improper_coeffs']:
                    if 'improper_coeff' in line and '@improper:' in line:
                        improper_name = line.split('@improper:')[1].split()[0]
                        if improper_name in kept_impropers:
                            f.write(f"    {line}\n")
                f.write("  } # End of improper_coeffs\n\n")

                f.write("}  # COMPASS\n")

            # Calculate file size reduction
            original_size = compass_src.stat().st_size
            subset_size = compass_dst.stat().st_size
            reduction = (1 - subset_size / original_size) * 100

            logger.info(f"  Successfully created compass_published.lt subset")
            logger.info(f"  File size: {original_size} -> {subset_size} bytes ({reduction:.1f}% reduction)")

        except Exception as e:
            raise WorkflowError(f"Error in make_compass_subset: {str(e)}") from e

    def make_oplsaa_lt_subset(self) -> None:
        """
        Creates a subset of OPLS-AA/LOPLS .lt parameters based on monomer topology.

        OPLS-AA 2024 .lt format uses numbered atom types (54, 60, 180, etc.) with
        replace{} directives that expand to wildcard format for matching:
        - replace{ @atom:54   @atom:54_bCT_aCT_dCT_iCT }

        The wildcard format uses:
        - _b<class> for bonds
        - _a<class> for angles
        - _d<class> for dihedrals
        - _i<class> for impropers

        This method:
        1. Extracts atom types used in monomers (e.g., 54, 60, 180, 182, 185)
        2. Parses the replace{} block to get class mappings
        3. Filters In Charges, Data Masses, pair_coeff by used types
        4. Filters Bond/Angle/Dihedral/Improper defs and coeffs by class matching

        Falls back to full .lt file if errors occur or model is not set.
        """
        src_lt = Path(self.path_oplsaaprm)
        ff_dir = Path(self.path_master) / "moltemplate" / "force_fields"

        if self.force_field == "lopls":
            dst_lt = Path(self.path_cwd) / "loplsaa.lt"
            base_lt = ff_dir / "oplsaa2024.lt"
            ff_name = "LOPLSAA"
        else:
            dst_lt = Path(self.path_cwd) / "oplsaa.lt"
            base_lt = None
            ff_name = "OPLSAA"

        if not src_lt.exists():
            raise GenerationError(f"OPLS-AA force field file not found: {src_lt}")

        try:
            logger.info(f"Creating {ff_name} parameter subset from .lt file...")

            # Step 1: Collect monomer files and extract atom types
            monomer_files = list(dict.fromkeys(self._get_monomer_files()))
            logger.info(f"  Found {len(monomer_files)} monomer files")

            # Extract atom types from monomers (numeric OPLS-AA types, e.g. 54, 60)
            atom_types: Set[int] = set()
            for filepath in monomer_files:
                try:
                    atom_types.update(extract_atom_types_from_lt(filepath, int))
                except Exception as e:
                    logger.warning(f"  Could not parse {filepath}: {e}")

            logger.info(f"  Found {len(atom_types)} unique atom types: {sorted(atom_types)}")

            if not atom_types:
                raise GenerationError("  No atom types found in monomer files. Cannot create force field subset.")

            # Step 2: Parse oplsaa.lt and extract sections
            sections = self._parse_oplsaa_lt_sections(str(src_lt))

            # Step 3: Extract class mappings from replace{} block
            class_mappings = self._extract_oplsaa_class_mappings(sections['replace_block'], atom_types)

            # Get class sets for filtering
            bond_classes = self._get_oplsaa_bond_classes(atom_types, class_mappings)
            angle_classes = self._get_oplsaa_angle_classes(atom_types, class_mappings)
            dihedral_classes = self._get_oplsaa_dihedral_classes(atom_types, class_mappings)
            improper_classes = self._get_oplsaa_improper_classes(atom_types, class_mappings)

            logger.info(f"  Bond classes: {sorted(bond_classes)}")
            logger.info(f"  Angle classes: {sorted(angle_classes)}")
            logger.info(f"  Dihedral classes: {sorted(dihedral_classes)}")

            # Step 4: Write filtered output
            with open(dst_lt, 'w') as f:
                # Header
                f.write(f"# {ff_name} Force Field Subset\n")
                f.write(f"# Generated by AutoPoly from {src_lt.name}\n")
                f.write(f"# Atom types used: {', '.join(str(t) for t in sorted(atom_types))}\n")
                f.write(f"# Total atom types: {len(atom_types)}\n\n")

                # For LOPLS, import the base OPLSAA force field first
                if self.force_field == "lopls":
                    f.write('import "oplsaa2024.lt"  # Base OPLS-AA force field\n\n')
                    f.write("OPLSAA {\n\n")  # Augment OPLSAA with LOPLS parameters
                else:
                    f.write(f"{ff_name} {{\n\n")

                # Write replace{} rules for used types
                f.write("  # Atom type expansion rules\n")
                replace_count = 0
                for line in sections['replace_block']:
                    if self._oplsaa_replace_matches(line, atom_types):
                        f.write(f"  replace{{ {line} }}\n")
                        replace_count += 1
                f.write("\n")
                logger.info(f"  Replace rules: {replace_count} (from {len(sections['replace_block'])})")

                # Write Init section (unchanged)
                if sections['init']:
                    f.write("  write_once(\"In Init\") {\n")
                    for line in sections['init']:
                        f.write(f"    {line}\n")
                    f.write("  } # End of init\n\n")

                # Write filtered Charges
                f.write("  write_once(\"In Charges\") {\n")
                charges_count = 0
                for line in sections['charges']:
                    if self._oplsaa_type_matches_line(line, atom_types):
                        f.write(f"    {line}\n")
                        charges_count += 1
                f.write("  } # End of charges\n\n")
                logger.info(f"  Charges: {charges_count} (from {len(sections['charges'])})")

                # Write filtered Masses
                f.write("  write_once(\"Data Masses\") {\n")
                masses_count = 0
                for line in sections['masses']:
                    if self._oplsaa_type_matches_line(line, atom_types):
                        f.write(f"    {line}\n")
                        masses_count += 1
                f.write("  } # End of masses\n\n")
                logger.info(f"  Masses: {masses_count} (from {len(sections['masses'])})")

                # Write filtered Pair Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                pair_count = 0
                for line in sections['pair_coeffs']:
                    if self._oplsaa_pair_coeff_matches(line, atom_types, class_mappings):
                        f.write(f"    {line}\n")
                        pair_count += 1
                f.write("  } # End of pair_coeffs\n\n")
                logger.info(f"  Pair coeffs: {pair_count} (from {len(sections['pair_coeffs'])})")

                # Write filtered Bond Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                bond_coeff_count = 0
                kept_bonds: Set[str] = set()
                for line in sections['bond_coeffs']:
                    if self._oplsaa_bond_coeff_matches(line, bond_classes):
                        # Extract bond name for later filtering of bond defs
                        if '@bond:' in line:
                            bond_name = line.split('@bond:')[1].split()[0]
                            kept_bonds.add(bond_name)
                        f.write(f"    {line}\n")
                        bond_coeff_count += 1
                f.write("  } # End of bond_coeffs\n\n")
                logger.info(f"  Bond coeffs: {bond_coeff_count} (from {len(sections['bond_coeffs'])})")

                # Write filtered Bond defs
                f.write("  write_once(\"Data Bonds By Type\") {\n")
                bond_def_count = 0
                for line in sections['bond_defs']:
                    if self._oplsaa_bond_def_matches(line, bond_classes):
                        f.write(f"    {line}\n")
                        bond_def_count += 1
                f.write("  } # End of bond defs\n\n")
                logger.info(f"  Bond defs: {bond_def_count} (from {len(sections['bond_defs'])})")

                # Write filtered Angle Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                angle_coeff_count = 0
                kept_angles: Set[str] = set()
                for line in sections['angle_coeffs']:
                    if self._oplsaa_angle_coeff_matches(line, angle_classes):
                        if '@angle:' in line:
                            angle_name = line.split('@angle:')[1].split()[0]
                            kept_angles.add(angle_name)
                        f.write(f"    {line}\n")
                        angle_coeff_count += 1
                f.write("  } # End of angle_coeffs\n\n")
                logger.info(f"  Angle coeffs: {angle_coeff_count} (from {len(sections['angle_coeffs'])})")

                # Write filtered Angle defs
                f.write("  write_once(\"Data Angles By Type\") {\n")
                angle_def_count = 0
                for line in sections['angle_defs']:
                    if self._oplsaa_angle_def_matches(line, angle_classes):
                        f.write(f"    {line}\n")
                        angle_def_count += 1
                f.write("  } # End of angle defs\n\n")
                logger.info(f"  Angle defs: {angle_def_count} (from {len(sections['angle_defs'])})")

                # Write filtered Dihedral Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                dihedral_coeff_count = 0
                kept_dihedrals: Set[str] = set()
                for line in sections['dihedral_coeffs']:
                    if self._oplsaa_dihedral_coeff_matches(line, dihedral_classes):
                        if '@dihedral:' in line:
                            dihedral_name = line.split('@dihedral:')[1].split()[0]
                            kept_dihedrals.add(dihedral_name)
                        f.write(f"    {line}\n")
                        dihedral_coeff_count += 1
                f.write("  } # End of dihedral_coeffs\n\n")
                logger.info(f"  Dihedral coeffs: {dihedral_coeff_count} (from {len(sections['dihedral_coeffs'])})")

                # Write filtered Dihedral defs
                f.write("  write_once(\"Data Dihedrals By Type\") {\n")
                dihedral_def_count = 0
                for line in sections['dihedral_defs']:
                    if self._oplsaa_dihedral_def_matches(line, dihedral_classes):
                        f.write(f"    {line}\n")
                        dihedral_def_count += 1
                f.write("  } # End of dihedral defs\n\n")
                logger.info(f"  Dihedral defs: {dihedral_def_count} (from {len(sections['dihedral_defs'])})")

                # Write filtered Improper Coeffs
                f.write("  write_once(\"In Settings\") {\n")
                improper_coeff_count = 0
                for line in sections['improper_coeffs']:
                    if self._oplsaa_improper_coeff_matches(line, improper_classes):
                        f.write(f"    {line}\n")
                        improper_coeff_count += 1
                f.write("  } # End of improper_coeffs\n\n")
                logger.info(f"  Improper coeffs: {improper_coeff_count} (from {len(sections['improper_coeffs'])})")

                # Write filtered Improper defs
                improper_header = sections.get('improper_header', 'Data Impropers By Type (cenIsortJKL.py)')
                f.write(f"  write_once(\"{improper_header}\") {{\n")
                improper_def_count = 0
                for line in sections['improper_defs']:
                    if self._oplsaa_improper_def_matches(line, improper_classes):
                        f.write(f"    {line}\n")
                        improper_def_count += 1
                f.write("  } # End of improper defs\n\n")
                logger.info(f"  Improper defs: {improper_def_count} (from {len(sections['improper_defs'])})")

                if self.force_field == "lopls":
                    f.write("}  # OPLSAA (with LOPLS extensions)\n")
                else:
                    f.write(f"}}  # {ff_name}\n")

            # Calculate file size reduction
            original_size = src_lt.stat().st_size
            subset_size = dst_lt.stat().st_size
            reduction = (1 - subset_size / original_size) * 100

            logger.info(f"  Successfully created {dst_lt.name} subset")
            logger.info(f"  File size: {original_size} -> {subset_size} bytes ({reduction:.1f}% reduction)")

            # For LOPLS, also copy the base oplsaa2024.lt if needed
            if base_lt and base_lt.exists():
                shutil.copy(base_lt, Path(self.path_cwd) / base_lt.name)
                logger.info(f"  Copied dependency: {base_lt.name}")

        except Exception as e:
            raise WorkflowError(f"Error in make_oplsaa_lt_subset: {str(e)}") from e

    def _parse_oplsaa_lt_sections(self, oplsaa_file: str) -> Dict[str, List[str]]:
        """Parse oplsaa.lt file into sections.

        Args:
            oplsaa_file: Path to oplsaa.lt file

        Returns:
            Dict with section names as keys and list of content lines as values
        """
        sections = {
            'replace_block': [],
            'init': [],
            'charges': [],
            'masses': [],
            'pair_coeffs': [],
            'bond_coeffs': [],
            'bond_defs': [],
            'angle_coeffs': [],
            'angle_defs': [],
            'dihedral_coeffs': [],
            'dihedral_defs': [],
            'improper_coeffs': [],
            'improper_defs': [],
            'improper_header': 'Data Impropers By Type (cenIsortJKL.py)'
        }

        current_section = None

        with open(oplsaa_file, 'r') as f:
            for line in f:
                stripped = line.strip()

                # Handle replace{} directives (single line format)
                if stripped.startswith('replace{') and '}' in stripped:
                    content = stripped[8:-1].strip()  # Remove 'replace{' and '}'
                    if content and not content.startswith('#'):
                        sections['replace_block'].append(content)
                    continue

                # Section detection
                if 'write_once("In Init")' in stripped:
                    current_section = 'init'
                    continue
                elif 'write_once("In Charges")' in stripped:
                    current_section = 'charges'
                    continue
                elif 'write_once("Data Masses")' in stripped:
                    current_section = 'masses'
                    continue
                elif 'write_once("Data Bonds By Type")' in stripped:
                    current_section = 'bond_defs'
                    continue
                elif 'write_once("Data Angles By Type")' in stripped:
                    current_section = 'angle_defs'
                    continue
                elif 'write_once("Data Dihedrals By Type")' in stripped:
                    current_section = 'dihedral_defs'
                    continue
                elif 'write_once("Data Impropers By Type' in stripped:
                    # Capture the exact header format
                    start = stripped.find('write_once("') + 12
                    end = stripped.find('")')
                    if end > start:
                        sections['improper_header'] = stripped[start:end]
                    current_section = 'improper_defs'
                    continue
                elif 'write_once("In Settings")' in stripped:
                    current_section = 'in_settings'
                    continue

                # End of section
                if stripped.startswith('}') and current_section:
                    current_section = None
                    continue

                # Skip empty lines and pure comment lines
                if not stripped:
                    continue

                # Collect content
                if current_section == 'init':
                    sections['init'].append(stripped)
                elif current_section == 'charges':
                    if 'set type' in stripped and '@atom:' in stripped:
                        sections['charges'].append(stripped)
                elif current_section == 'masses':
                    if '@atom:' in stripped:
                        sections['masses'].append(stripped)
                elif current_section == 'bond_defs':
                    if '@bond:' in stripped:
                        sections['bond_defs'].append(stripped)
                elif current_section == 'angle_defs':
                    if '@angle:' in stripped:
                        sections['angle_defs'].append(stripped)
                elif current_section == 'dihedral_defs':
                    if '@dihedral:' in stripped:
                        sections['dihedral_defs'].append(stripped)
                elif current_section == 'improper_defs':
                    if '@improper:' in stripped:
                        sections['improper_defs'].append(stripped)
                elif current_section == 'in_settings':
                    if 'pair_coeff' in stripped:
                        sections['pair_coeffs'].append(stripped)
                    elif 'bond_coeff' in stripped:
                        sections['bond_coeffs'].append(stripped)
                    elif 'angle_coeff' in stripped:
                        sections['angle_coeffs'].append(stripped)
                    elif 'dihedral_coeff' in stripped:
                        sections['dihedral_coeffs'].append(stripped)
                    elif 'improper_coeff' in stripped:
                        sections['improper_coeffs'].append(stripped)

        return sections

    def _extract_oplsaa_class_mappings(self, replace_block: List[str], atom_types: Set[int]) -> Dict[int, Dict[str, str]]:
        """Extract class mappings from OPLS-AA replace{} block.

        replace{} content format: @atom:54   @atom:54_bCT_aCT_dCT_iCT

        This means type 54 has: bond class=CT, angle class=CT, dihedral class=CT, improper class=CT

        Args:
            replace_block: List of replace{} content strings
            atom_types: Set of atom types used in monomers

        Returns:
            Dict mapping atom type number -> {bond, angle, dihedral, improper class names}
        """
        mappings = {}
        for line in replace_block:
            # Parse: @atom:54   @atom:54_bCT_aCT_dCT_iCT
            parts = line.split('@atom:')
            if len(parts) >= 3:
                try:
                    simple_type = int(parts[1].split()[0].strip())
                except ValueError:
                    continue

                if simple_type in atom_types:
                    extended = parts[2].strip()
                    # Parse format: 54_bCT_aCT_dCT_iCT
                    class_map = {}
                    ext_parts = extended.split('_')
                    for p in ext_parts:
                        if p.startswith('b') and len(p) > 1:
                            class_map['bond'] = p[1:]
                        elif p.startswith('a') and len(p) > 1:
                            class_map['angle'] = p[1:]
                        elif p.startswith('d') and len(p) > 1:
                            class_map['dihedral'] = p[1:]
                        elif p.startswith('i') and len(p) > 1:
                            class_map['improper'] = p[1:]
                    if class_map:
                        mappings[simple_type] = class_map

        return mappings

    def _get_oplsaa_bond_classes(self, atom_types: Set[int], class_mappings: Dict) -> Set[str]:
        """Get bond classes for all used atom types."""
        bond_classes = set()
        for at in atom_types:
            if at in class_mappings and 'bond' in class_mappings[at]:
                bond_classes.add(class_mappings[at]['bond'])
        return bond_classes

    def _get_oplsaa_angle_classes(self, atom_types: Set[int], class_mappings: Dict) -> Set[str]:
        """Get angle classes for all used atom types."""
        angle_classes = set()
        for at in atom_types:
            if at in class_mappings and 'angle' in class_mappings[at]:
                angle_classes.add(class_mappings[at]['angle'])
        return angle_classes

    def _get_oplsaa_dihedral_classes(self, atom_types: Set[int], class_mappings: Dict) -> Set[str]:
        """Get dihedral classes for all used atom types."""
        dihedral_classes = set()
        for at in atom_types:
            if at in class_mappings and 'dihedral' in class_mappings[at]:
                dihedral_classes.add(class_mappings[at]['dihedral'])
        return dihedral_classes

    def _get_oplsaa_improper_classes(self, atom_types: Set[int], class_mappings: Dict) -> Set[str]:
        """Get improper classes for all used atom types."""
        improper_classes = set()
        for at in atom_types:
            if at in class_mappings and 'improper' in class_mappings[at]:
                improper_classes.add(class_mappings[at]['improper'])
        return improper_classes

    def _oplsaa_replace_matches(self, line: str, atom_types: Set[int]) -> bool:
        """Check if a replace{} line involves a used atom type."""
        # Line format: @atom:54   @atom:54_bCT_aCT_dCT_iCT
        parts = line.split('@atom:')
        if len(parts) >= 2:
            try:
                type_num = int(parts[1].split()[0])
                return type_num in atom_types
            except ValueError:
                pass
        return False

    def _oplsaa_type_matches_line(self, line: str, atom_types: Set[int]) -> bool:
        """Check if a line (charge or mass) references a used atom type.

        Charge format: set type @atom:54 charge -0.180
        Mass format: @atom:54   12.011
        """
        if '@atom:' in line:
            # Extract the atom type number
            parts = line.split('@atom:')
            if len(parts) >= 2:
                type_part = parts[1].split()[0]
                # Handle both simple (54) and extended (54_bCT_...) formats
                try:
                    type_num = int(type_part.split('_')[0])
                    return type_num in atom_types
                except ValueError:
                    pass
        return False

    def _oplsaa_pair_coeff_matches(self, line: str, atom_types: Set[int], class_mappings: Dict) -> bool:
        """Check if a pair_coeff line references used atom types.

        Format: pair_coeff @atom:54_bCT_aCT_dCT_iCT @atom:54_bCT_aCT_dCT_iCT 0.066 3.550
        """
        parts = line.split('@atom:')
        if len(parts) >= 3:
            # Extract both atom types
            try:
                type1 = int(parts[1].split('_')[0].split()[0])
                type2 = int(parts[2].split('_')[0].split()[0])
                return type1 in atom_types and type2 in atom_types
            except ValueError:
                pass
        return False

    def _oplsaa_bond_coeff_matches(self, line: str, bond_classes: Set[str]) -> bool:
        """Check if a bond_coeff line references used bond classes.

        Format: bond_coeff @bond:CT_HC 340.00 1.090
        """
        if '@bond:' in line:
            bond_name = line.split('@bond:')[1].split()[0]
            # Bond name format: CLASS1_CLASS2
            parts = bond_name.split('_')
            if len(parts) >= 2:
                # Check if both classes are in our set
                return parts[0] in bond_classes and parts[1] in bond_classes
        return False

    def _oplsaa_bond_def_matches(self, line: str, bond_classes: Set[str]) -> bool:
        """Check if a bond definition line references used bond classes.

        Format: @bond:CT_HC @atom:*_bCT*_a*_d*_i* @atom:*_bHC*_a*_d*_i*
        """
        parts = line.split('@atom:')
        if len(parts) >= 3:
            # Extract bond classes from atom patterns
            for part in parts[1:3]:
                # Find the _b<CLASS>* pattern
                if '_b' in part:
                    idx = part.find('_b')
                    end_idx = part.find('*', idx)
                    if end_idx == -1:
                        end_idx = part.find('_', idx + 2)
                    if end_idx == -1:
                        end_idx = len(part)
                    bond_class = part[idx+2:end_idx]
                    if bond_class not in bond_classes:
                        return False
            return True
        return False

    def _oplsaa_angle_coeff_matches(self, line: str, angle_classes: Set[str]) -> bool:
        """Check if an angle_coeff line references used angle classes.

        Format: angle_coeff @angle:CT_CT_CT 58.35 112.70
        """
        if '@angle:' in line:
            angle_name = line.split('@angle:')[1].split()[0]
            # Angle name format: CLASS1_CLASS2_CLASS3
            parts = angle_name.split('_')
            if len(parts) >= 3:
                return all(p in angle_classes for p in parts[:3])
        return False

    def _oplsaa_angle_def_matches(self, line: str, angle_classes: Set[str]) -> bool:
        """Check if an angle definition line references used angle classes.

        Format: @angle:CT_CT_CT @atom:*_b*_aCT*_d*_i* @atom:*_b*_aCT*_d*_i* @atom:*_b*_aCT*_d*_i*
        """
        parts = line.split('@atom:')
        if len(parts) >= 4:
            # Extract angle classes from atom patterns
            for part in parts[1:4]:
                # Find the _a<CLASS>* pattern
                if '_a' in part:
                    idx = part.find('_a')
                    end_idx = part.find('*', idx)
                    if end_idx == -1:
                        end_idx = part.find('_', idx + 2)
                    if end_idx == -1:
                        end_idx = len(part)
                    angle_class = part[idx+2:end_idx]
                    if angle_class not in angle_classes:
                        return False
            return True
        return False

    def _oplsaa_dihedral_coeff_matches(self, line: str, dihedral_classes: Set[str]) -> bool:
        """Check if a dihedral_coeff line references used dihedral classes.

        Format: dihedral_coeff @dihedral:CT_CT_CT_CT opls 0.650 -0.250 0.670 0.0
        Also handles wildcards like €€ which match any class.
        """
        if '@dihedral:' in line:
            dihedral_name = line.split('@dihedral:')[1].split()[0]
            # Dihedral name format: CLASS1_CLASS2_CLASS3_CLASS4
            parts = dihedral_name.split('_')
            if len(parts) >= 4:
                # Check each part - €€ or ?? are wildcards
                for p in parts[:4]:
                    if p not in ('€€', '??', '') and p not in dihedral_classes:
                        return False
                return True
        return False

    def _oplsaa_dihedral_def_matches(self, line: str, dihedral_classes: Set[str]) -> bool:
        """Check if a dihedral definition line references used dihedral classes.

        Format: @dihedral:€€_CT_CT_€€ @atom:*_b*_a*_d??*_i* @atom:*_b*_a*_dCT*_i* ...
        Wildcards ?? match any class.
        """
        parts = line.split('@atom:')
        if len(parts) >= 5:
            # Check each atom pattern for dihedral class
            for part in parts[1:5]:
                # Find the _d<CLASS>* pattern
                if '_d' in part:
                    idx = part.find('_d')
                    end_idx = part.find('*', idx)
                    if end_idx == -1:
                        end_idx = part.find('_', idx + 2)
                    if end_idx == -1:
                        end_idx = len(part)
                    dihedral_class = part[idx+2:end_idx]
                    # ?? is a wildcard
                    if dihedral_class not in ('??', '') and dihedral_class not in dihedral_classes:
                        return False
            return True
        return False

    def _oplsaa_improper_coeff_matches(self, line: str, improper_classes: Set[str]) -> bool:
        """Check if an improper_coeff line references used improper classes."""
        if '@improper:' in line:
            improper_name = line.split('@improper:')[1].split()[0]
            parts = improper_name.split('_')
            if len(parts) >= 4:
                for p in parts[:4]:
                    if p not in ('€€', '??', '*', '') and p not in improper_classes:
                        return False
                return True
        return False

    def _oplsaa_improper_def_matches(self, line: str, improper_classes: Set[str]) -> bool:
        """Check if an improper definition line references used improper classes."""
        parts = line.split('@atom:')
        if len(parts) >= 5:
            for part in parts[1:5]:
                if '_i' in part:
                    idx = part.find('_i')
                    end_idx = part.find('*', idx)
                    if end_idx == -1:
                        end_idx = part.find('_', idx + 2)
                    if end_idx == -1:
                        end_idx = len(part)
                    improper_class = part[idx+2:end_idx]
                    if improper_class not in ('??', '*', '') and improper_class not in improper_classes:
                        return False
            return True
        return False

    def _build_compass_bond_graph(self, filepath: str) -> Dict[str, Tuple[str, List[str]]]:
        """Build a bond connectivity graph from a COMPASS monomer .lt file."""
        return parse_bond_graph(filepath)

    def _parse_compass_lt_sections(self, compass_file: str) -> Dict[str, List[str]]:
        """Parse compass_published.lt file into sections."""
        sections = {
            'replace_block': [],
            'init': [],
            'masses': [],
            'pair_coeffs': [],
            'charge_by_bond': [],
            'bond_defs': [],
            'bond_coeffs': [],
            'angle_defs': [],
            'angle_coeffs': [],
            'dihedral_defs': [],
            'dihedral_coeffs': [],
            'improper_defs': [],
            'improper_coeffs': []
        }

        current_section = None

        with open(compass_file, 'r') as f:
            for line in f:
                stripped = line.strip()

                # COMPASS uses single-line replace{ ... } directives
                if stripped.startswith('replace{') and '}' in stripped:
                    # Extract the content between replace{ and }
                    content = stripped[8:-1].strip()  # Remove 'replace{' and '}'
                    if content and not content.startswith('#'):
                        sections['replace_block'].append(content)
                    continue

                # Section detection
                if 'write_once("In Init")' in stripped:
                    current_section = 'init'
                    continue
                elif 'write_once("Data Masses")' in stripped:
                    current_section = 'masses'
                    continue
                elif 'write_once("Data Charge By Bond")' in stripped:
                    current_section = 'charge_by_bond'
                    continue
                elif 'write_once("Data Bonds By Type")' in stripped:
                    current_section = 'bond_defs'
                    continue
                elif 'write_once("Data Angles By Type")' in stripped:
                    current_section = 'angle_defs'
                    continue
                elif 'write_once("Data Dihedrals By Type")' in stripped:
                    current_section = 'dihedral_defs'
                    continue
                elif 'write_once("Data Impropers By Type' in stripped:
                    current_section = 'improper_defs'
                    continue
                elif 'write_once("In Settings")' in stripped:
                    current_section = 'in_settings'
                    continue

                # End of section
                if stripped.startswith('}') and current_section:
                    current_section = None
                    continue

                # Collect content
                if not stripped or stripped.startswith('#'):
                    continue

                if current_section == 'init':
                    sections['init'].append(stripped)
                elif current_section == 'masses':
                    if '@atom:' in stripped:
                        sections['masses'].append(stripped)
                elif current_section == 'charge_by_bond':
                    if '@atom:' in stripped:
                        sections['charge_by_bond'].append(stripped)
                elif current_section == 'bond_defs':
                    if '@bond:' in stripped:
                        sections['bond_defs'].append(stripped)
                elif current_section == 'angle_defs':
                    if '@angle:' in stripped:
                        sections['angle_defs'].append(stripped)
                elif current_section == 'dihedral_defs':
                    if '@dihedral:' in stripped:
                        sections['dihedral_defs'].append(stripped)
                elif current_section == 'improper_defs':
                    if '@improper:' in stripped:
                        sections['improper_defs'].append(stripped)
                elif current_section == 'in_settings':
                    if 'pair_coeff' in stripped:
                        sections['pair_coeffs'].append(stripped)
                    elif 'bond_coeff' in stripped:
                        sections['bond_coeffs'].append(stripped)
                    elif 'angle_coeff' in stripped:
                        sections['angle_coeffs'].append(stripped)
                    elif 'dihedral_coeff' in stripped:
                        sections['dihedral_coeffs'].append(stripped)
                    elif 'improper_coeff' in stripped:
                        sections['improper_coeffs'].append(stripped)

        return sections

    def _extract_compass_core_types(self, full_types: Set[str]) -> Set[str]:
        """Extract core pair types from full COMPASS atom types.

        COMPASS types are like *~pc4~b*~a*~d*~i* where 'pc4' is the core pair type.

        Args:
            full_types: Set of full COMPASS atom type strings

        Returns:
            Set of core pair types (e.g., {'c4', 'o2e', 'h1'})
        """
        core_types = set()
        for full_type in full_types:
            # Parse ~p<type>~ pattern
            if '~p' in full_type:
                parts = full_type.split('~')
                for part in parts:
                    if part.startswith('p') and len(part) > 1:
                        core_types.add(part[1:])  # Remove 'p' prefix
            else:
                # Simple type without ~ notation
                core_types.add(full_type)
        return core_types

    def _extract_compass_class_mappings(self, replace_block: List[str], core_types: Set[str]) -> Dict[str, Dict[str, str]]:
        """Extract class mappings from COMPASS replace{} block.

        COMPASS replace{} content (after 'replace{' and '}' stripped):
        @atom:h1o @atom:h1o~ph1o~bh1~ah1~dh1~ih1

        This means h1o has: pair=h1o, bond=h1, angle=h1, dihedral=h1, improper=h1

        Args:
            replace_block: List of replace{} content (e.g., "@atom:h1o @atom:h1o~ph1o~...")
            core_types: Set of core types used (e.g., {'c4', 'h1o'})

        Returns:
            Dict mapping simple type -> {pair, bond, angle, dihedral, improper}
        """
        mappings = {}
        for line in replace_block:
            # Parse: @atom:h1o @atom:h1o~ph1o~bh1~ah1~dh1~ih1
            parts = line.split('@atom:')
            if len(parts) >= 3:
                simple_type = parts[1].split()[0].strip()
                if simple_type in core_types:
                    extended = parts[2].strip()
                    # Remove trailing } if present
                    extended = extended.rstrip('}').strip()
                    # Parse ~p..~b..~a..~d..~i.. format
                    class_map = {}
                    ext_parts = extended.split('~')
                    for p in ext_parts:
                        if p.startswith('p') and len(p) > 1:
                            class_map['pair'] = p[1:]
                        elif p.startswith('b') and len(p) > 1:
                            class_map['bond'] = p[1:]
                        elif p.startswith('a') and len(p) > 1:
                            class_map['angle'] = p[1:]
                        elif p.startswith('d') and len(p) > 1:
                            class_map['dihedral'] = p[1:]
                        elif p.startswith('i') and len(p) > 1:
                            class_map['improper'] = p[1:]
                    if class_map:
                        mappings[simple_type] = class_map
        return mappings

    def _get_compass_bond_classes(self, core_types: Set[str], class_mappings: Dict) -> Set[str]:
        """Get bond classes for all used core types."""
        bond_classes = set()
        for core in core_types:
            if core in class_mappings and 'bond' in class_mappings[core]:
                bond_classes.add(class_mappings[core]['bond'])
            else:
                # Default: bond class = core type
                bond_classes.add(core)
        return bond_classes

    def _get_compass_angle_classes(self, core_types: Set[str], class_mappings: Dict) -> Set[str]:
        """Get angle classes for all used core types."""
        angle_classes = set()
        for core in core_types:
            if core in class_mappings and 'angle' in class_mappings[core]:
                angle_classes.add(class_mappings[core]['angle'])
            else:
                angle_classes.add(core)
        return angle_classes

    def _get_compass_dihedral_classes(self, core_types: Set[str], class_mappings: Dict) -> Set[str]:
        """Get dihedral classes for all used core types."""
        dihedral_classes = set()
        for core in core_types:
            if core in class_mappings and 'dihedral' in class_mappings[core]:
                dihedral_classes.add(class_mappings[core]['dihedral'])
            else:
                dihedral_classes.add(core)
        return dihedral_classes

    def _get_compass_improper_classes(self, core_types: Set[str], class_mappings: Dict) -> Set[str]:
        """Get improper classes for all used core types."""
        improper_classes = set()
        for core in core_types:
            if core in class_mappings and 'improper' in class_mappings[core]:
                improper_classes.add(class_mappings[core]['improper'])
            else:
                improper_classes.add(core)
        return improper_classes

    def _compass_replace_matches(self, line: str, core_types: Set[str]) -> bool:
        """Check if a replace{} line involves used core types."""
        # Replace lines look like: @atom:c4 @atom:*~pc4~b*~a*~d*~i*
        for core in core_types:
            if f"@atom:{core}" in line or f"~p{core}~" in line:
                return True
        return False

    def _compass_type_matches(self, line: str, core_types: Set[str]) -> bool:
        """Check if a COMPASS line references used core types.

        Handles both simple types (e.g., @atom:c4 in masses) and
        extended types (e.g., @atom:*~pc4~b*~a*~d*~i* in pair_coeffs).
        """
        for core in core_types:
            # Match extended format with ~p prefix
            if f"~p{core}~" in line:
                return True
            # Match simple type format (used in masses)
            if f"@atom:{core} " in line or f"@atom:{core}\t" in line or line.endswith(f"@atom:{core}"):
                return True
        return False

    def _compass_bond_matches_v2(self, line: str, bond_classes: Set[str]) -> bool:
        """Check if a COMPASS bond line references used bond classes."""
        # Need both atom types to match their bond class
        parts = line.split('@atom:')
        match_count = 0
        for part in parts[1:]:  # Skip first split part
            for bc in bond_classes:
                if f"~b{bc}~" in part:
                    match_count += 1
                    break
        return match_count >= 2

    def _compass_angle_matches_v2(self, line: str, angle_classes: Set[str]) -> bool:
        """Check if a COMPASS angle line references used angle classes."""
        parts = line.split('@atom:')
        match_count = 0
        for part in parts[1:]:
            for ac in angle_classes:
                if f"~a{ac}~" in part:
                    match_count += 1
                    break
        return match_count >= 3

    def _compass_dihedral_matches_v2(self, line: str, dihedral_classes: Set[str]) -> bool:
        """Check if a COMPASS dihedral line references used dihedral classes."""
        parts = line.split('@atom:')
        match_count = 0
        for part in parts[1:]:
            for dc in dihedral_classes:
                if f"~d{dc}~" in part:
                    match_count += 1
                    break
        return match_count >= 4

    def _compass_improper_matches_v2(self, line: str, improper_classes: Set[str]) -> bool:
        """Check if a COMPASS improper line references used improper classes."""
        parts = line.split('@atom:')
        match_count = 0
        for part in parts[1:]:
            for ic in improper_classes:
                if f"~i{ic}~" in part:
                    match_count += 1
                    break
        return match_count >= 4

    def _compass_charge_matches_v2(self, line: str, bond_classes: Set[str]) -> bool:
        """Check if a COMPASS charge by bond line references used bond classes."""
        # Charge by bond uses bond classes
        parts = line.split('@atom:')
        match_count = 0
        for part in parts[1:]:
            for bc in bond_classes:
                if f"~b{bc}~" in part:
                    match_count += 1
                    break
        return match_count >= 2


    # Alkyl atom types: CH3 (methyl), CH2 (methylene), CH (methine)
    ALKYL_ATOM_TYPES = {80, 81, 82}

    def _is_alkyl_dihedral(self, line: str) -> bool:
        """
        Check if a line contains an alkyl dihedral specification.

        Args:
            line: The line to check

        Returns:
            bool: True if the line specifies a dihedral with all alkyl atoms
        """
        if '@dihedral:' not in line:
            return False

        try:
            dihedral_spec = line.split("@dihedral:")[1].split()[0]
            atom_types = [int(x) for x in dihedral_spec.split('-')]
            return all(atom in self.ALKYL_ATOM_TYPES for atom in atom_types)
        except (IndexError, ValueError):
            return False

    def _write_modified_dihedral_line(self, write_f, line: str) -> None:
        """
        Write a modified dihedral coefficient line for alkyl chains.

        Args:
            write_f: File object to write to
            line: Original line containing dihedral specification
        """
        dihedral_spec = line.split("@dihedral:")[1].split()[0]
        atom_types = [int(x) for x in dihedral_spec.split('-')]

        write_f.write("dihedral_coeff @dihedral:")
        write_f.write('-'.join(str(x) for x in atom_types))
        write_f.write(" opls")

        if hasattr(self, 'ff_modify_dihedral'):
            write_f.write(" " + " ".join(str(x) for x in self.ff_modify_dihedral))
        write_f.write("\n")

    def FFmodify_alkyl_dihedral_oplsaa(self) -> None:
        """
        Modifies alkyl dihedral coefficients in the OPLS-AA force field.

        This method modifies the dihedral coefficients for alkyl chains (CH3-CH2-CH2-CH3
        and similar) in the generated oplsaa.lt file. The modification uses custom
        dihedral parameters stored in the ff_modify_dihedral attribute.

        The process:
        1. Reads the generated oplsaa.lt file
        2. Identifies dihedral coefficients for alkyl chains (atom types 80, 81, 82)
        3. Replaces them with modified parameters
        4. Writes the modified file back

        Alkyl atom types:
        - 80: CH3 (methyl)
        - 81: CH2 (methylene)
        - 82: CH (methine)

        Raises:
            SystemExit: If oplsaa.lt file cannot be opened or modification fails

        Note:
            The ff_modify_dihedral attribute should contain 4 dihedral coefficients
            in the format [K1, K2, K3, K4] in units of kcal/mol.
        """
        input_file = Path(self.path_cwd) / "oplsaa.lt"
        output_file = Path(self.path_cwd) / "oplsaa_tmp.lt"

        if not input_file.exists():
            raise GenerationError(f"oplsaa.lt file cannot open: {input_file}")

        logger.info("Start modifying alkyl dihedral coefficients")

        try:
            with open(input_file, 'r') as read_f, open(output_file, 'w') as write_f:
                for line in read_f:
                    stripped = line.strip()
                    words = stripped.split()

                    # Handle write_once block start
                    if words and words[0] == 'write_once("In':
                        write_f.write(stripped + "\n")
                        next_line = next(read_f).strip()
                        next_words = next_line.split()

                        if next_words and next_words[0] == "dihedral_coeff":
                            if self._is_alkyl_dihedral(next_line):
                                self._write_modified_dihedral_line(write_f, next_line)
                            else:
                                write_f.write(next_line + "\n")
                        else:
                            write_f.write(next_line + "\n")
                        continue

                    # Handle lines inside dihedral_coeff block
                    if words and words[0] == "dihedral_coeff":
                        if self._is_alkyl_dihedral(stripped):
                            self._write_modified_dihedral_line(write_f, stripped)
                        else:
                            write_f.write(stripped + "\n")
                    else:
                        write_f.write(stripped + "\n")

            # Replace original file with modified version
            shutil.move(str(output_file), str(input_file))

        except Exception as e:
            raise WorkflowError(f"Error modifying alkyl dihedral coefficients: {str(e)}") from e
