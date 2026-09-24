#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# The functional-group SMARTS library and monomer eligibility rules in this file
# are adapted from AutoREACTER's functional_groups_library and
# functional_groups_detector modules.
#
# The adapted code originates from the AutoREACTER project
# (https://github.com/NanoCIPHER-Lab/AutoREACTER), developed by the
# NanoCIPHER Lab. AutoREACTER is distributed under the MIT License, which
# requires the following copyright notice and permission notice to be
# included in all copies or substantial portions of the software:
#
# MIT License
#
# Copyright (c) 2026 Janitha Mahanthe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The full license text is also reproduced in the LICENSE file in this
# directory.
# ---------------------------------------------------------------------------
"""
Functional Group Library and Detection for the AutoPoly Reactor

This module provides the functional-group inventory used to decide which
monomers can participate in reactor (LAMMPS ``fix bond/react``) reactions.
The SMARTS library and eligibility rules are adapted from the AutoREACTER
project (NanoCIPHER-Lab, MIT license) and reimplemented on plain RDKit
molecules without pandas/PIL dependencies.

Eligibility rules (same philosophy as AutoREACTER):

- ``mono``/``vinyl``: at least one match of the primary SMARTS is enough.
- ``di_identical``: at least two matches of the primary SMARTS are required
  (a single-site "A1" monomer would act as a chain stopper in step-growth
  polymerization and is rejected).
- ``di_different``: at least one match of EACH of the two SMARTS patterns
  (heterobifunctional A-B monomers such as amino acids or hydroxy acids).

Created on 2026-08-04
@author: zwu
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ..core.system import logger

# ---------------------------------------------------------------------------
# Functional group library
# ---------------------------------------------------------------------------
# Each entry:
#   functionality_type: "mono" | "vinyl" | "di_identical" | "di_different"
#   smarts_1: primary SMARTS (with a mapped anchor atom :1 where applicable)
#   smarts_2: secondary SMARTS for "di_different" entries
#   group_name: identifier matched against reaction library reactant names

FUNCTIONAL_GROUPS: Dict[str, Dict[str, Optional[str]]] = {
    # --- Hydroxy / carboxylic acid AB-type monomers -------------------------
    "hydroxy_carboxylic_acid_monomer": {
        "functionality_type": "di_different",
        "smarts_1": "[OX2H1;!$([O][C,S]=*):1]",
        "smarts_2": "[CX3:2](=[O])[OX2H1]",
        "group_name": "hydroxy_carboxylic_acid",
        "comments": None,
    },
    "hydroxy_acid_halides_monomer": {
        "functionality_type": "di_different",
        "smarts_1": "[OX2H1;!$([O][C,S]=*):1]",
        "smarts_2": "[CX3:2](=[O])[Cl,Br,I]",
        "group_name": "hydroxy_acid_halide",
        "comments": "Hydroxy acid halides are highly reactive and less common "
                    "than hydroxy carboxylic acids for polyesterification.",
    },
    # --- Alcohol / thiol functional monomers --------------------------------
    "diol_monomer": {
        "functionality_type": "di_identical",
        "smarts_1": "[OX2H1;!$([O][C,S]=*):1]",
        "smarts_2": None,
        "group_name": "diol",
        "comments": None,
    },
    "dithiol_monomer": {
        "functionality_type": "di_identical",
        "smarts_1": "[SX2H1;!$([S][C,S]=*):1]",
        "smarts_2": None,
        "group_name": "dithiol",
        "comments": None,
    },
    "hydroxy_thiol_monomer": {
        "functionality_type": "di_different",
        "smarts_1": "[OX2H1;!$([O][C,S]=*):1]",
        "smarts_2": "[SX2H1;!$([S][C,S]=*):2]",
        "group_name": "hydroxy_thiol",
        "comments": None,
    },
    # --- Amine / amino acid monomers ----------------------------------------
    "amino_acid_monomer": {
        "functionality_type": "di_different",
        "smarts_1": "[NX3;H2,H1;!$([N][C,S]=*):1]",
        "smarts_2": "[CX3:2](=[O])[OX2H1]",
        "group_name": "amino_acid",
        "comments": None,
    },
    "di_amine_monomer": {
        "functionality_type": "di_identical",
        "smarts_1": "[NX3;H2,H1;!$([N][C,S]=*):1]",
        "smarts_2": None,
        "group_name": "di_amine",
        "comments": None,
    },
    # --- Carboxylic acid / acid halide / ester monomers ---------------------
    "di_carboxylic_acid_monomer": {
        "functionality_type": "di_identical",
        "smarts_1": "[CX3:1](=[O])[OX2H1]",
        "smarts_2": None,
        "group_name": "di_carboxylic_acid",
        "comments": None,
    },
    "di_carboxylic_acid_halide_monomer": {
        "functionality_type": "di_identical",
        "smarts_1": "[CX3:1](=[O])[Cl,Br,I]",
        "smarts_2": None,
        "group_name": "di_carboxylic_acid_halide",
        "comments": None,
    },
    "carboxylic_acid_acid_halide_monomer": {
        "functionality_type": "di_different",
        "smarts_1": "[CX3:1](=[O])[OX2H1]",
        "smarts_2": "[CX3:2](=[O])[Cl,Br,I]",
        "group_name": "carboxylic_acid_acid_halide",
        "comments": "Mixed COOH/acid-halide AB monomer; forms "
                    "polyanhydride-type linkages, not polyester.",
    },
    "di_carboxylic_ester_monomer": {
        "functionality_type": "di_identical",
        "smarts_1": "[CX3:1](=[O])[OX2H0][#6]",
        "smarts_2": None,
        "group_name": "di_carboxylic_ester",
        "comments": None,
    },
    # --- Isocyanate monomers -------------------------------------------------
    "di_isocyanate_monomer": {
        "functionality_type": "di_identical",
        "smarts_1": "[NX2]=[CX2:1]=[OX1]",
        "smarts_2": None,
        "group_name": "di_isocyanate",
        "comments": None,
    },
}


@dataclass(frozen=True)
class FunctionalGroupInfo:
    """
    One detected functional group on a monomer.

    Attributes:
        functionality_type: "mono" | "vinyl" | "di_identical" | "di_different".
        fg_name: Group identifier (matched to reaction library reactants).
        fg_smarts_1: Primary SMARTS pattern.
        fg_count_1: Number of matches of the primary pattern.
        fg_smarts_2: Secondary SMARTS ("di_different" only).
        fg_count_2: Number of matches of the secondary pattern.
        matches: Atom-index tuples of all matches (both patterns).
    """
    functionality_type: str
    fg_name: str
    fg_smarts_1: str
    fg_count_1: int
    fg_smarts_2: Optional[str] = None
    fg_count_2: Optional[int] = None
    matches: Tuple[Tuple[int, ...], ...] = field(default_factory=tuple, compare=False)


@dataclass(frozen=True)
class MonomerRole:
    """
    A monomer annotated with its detected functional groups.

    Attributes:
        smiles: Monomer SMILES (as supplied).
        name: Monomer name.
        functionalities: Detected functional groups.
    """
    smiles: str
    name: str
    functionalities: Tuple[FunctionalGroupInfo, ...]

    def has_group(self, fg_name: str) -> bool:
        return any(fg.fg_name == fg_name for fg in self.functionalities)


def _count_matches(
    mol: Chem.Mol,
    functionality_type: str,
    smarts_1: str,
    smarts_2: Optional[str],
) -> Tuple[int, Optional[int], Optional[int], Tuple[Tuple[int, ...], ...]]:
    """
    Match one functional-group definition against a molecule.

    Returns:
        (functionality_count, count_1, count_2, matches) where
        functionality_count is 0 (not eligible), 1 (mono/vinyl eligible) or
        2 (di_identical/di_different eligible).
    """
    patt1 = Chem.MolFromSmarts(smarts_1)
    if patt1 is None:
        logger.warning(f"Invalid primary SMARTS skipped: {smarts_1}")
        return 0, None, None, ()

    if smarts_2:
        patt2 = Chem.MolFromSmarts(smarts_2)
        if patt2 is None:
            logger.warning(f"Invalid secondary SMARTS skipped: {smarts_2}")
            return 0, None, None, ()
        matches1 = mol.GetSubstructMatches(patt1)
        matches2 = mol.GetSubstructMatches(patt2)
        if len(matches1) >= 1 and len(matches2) >= 1:
            return 2, len(matches1), len(matches2), matches1 + matches2
        return 0, len(matches1), len(matches2), matches1 + matches2

    matches = mol.GetSubstructMatches(patt1)
    if functionality_type == "di_identical":
        if len(matches) >= 2:
            return 2, len(matches), None, matches
        return 0, len(matches), None, matches
    # mono / vinyl: a single site is sufficient
    if len(matches) >= 1:
        return 1, len(matches), None, matches
    return 0, 0, None, ()


def detect_functional_groups(
    mol: Chem.Mol,
    library: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> Tuple[FunctionalGroupInfo, ...]:
    """
    Detect all eligible functional groups on one molecule.

    Args:
        mol: RDKit Mol (hydrogens may be implicit; SMARTS match either way).
        library: Functional-group library (defaults to FUNCTIONAL_GROUPS).

    Returns:
        Tuple of FunctionalGroupInfo (empty if nothing detected).
    """
    if mol is None:
        return ()
    library = library or FUNCTIONAL_GROUPS
    detected: List[FunctionalGroupInfo] = []

    for entry in library.values():
        ftype = entry["functionality_type"]
        smarts_1 = entry["smarts_1"]
        smarts_2 = entry.get("smarts_2")

        count, count_1, count_2, matches = _count_matches(mol, ftype, smarts_1, smarts_2)
        if count > 0:
            detected.append(FunctionalGroupInfo(
                functionality_type=ftype,
                fg_name=entry["group_name"],
                fg_smarts_1=smarts_1,
                fg_count_1=count_1,
                fg_smarts_2=smarts_2,
                fg_count_2=count_2,
                matches=matches,
            ))
            if entry.get("comments"):
                logger.info(f"Note ({entry['group_name']}): {entry['comments']}")

    return tuple(detected)


def detect_monomer_roles(
    monomers: List[Tuple[str, str]],
    library: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> List[MonomerRole]:
    """
    Detect functional groups across a set of monomers.

    Args:
        monomers: List of (smiles, name) pairs.
        library: Functional-group library (defaults to FUNCTIONAL_GROUPS).

    Returns:
        List of MonomerRole for monomers with at least one detected group.

    Raises:
        ValidationError-style RuntimeError is avoided: an empty result simply
        means nothing reactive was found (callers decide how to react).
    """
    roles: List[MonomerRole] = []
    for smiles, name in monomers:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Skipping unparseable monomer SMILES: {smiles!r}")
            continue
        fgs = detect_functional_groups(mol, library)
        if fgs:
            roles.append(MonomerRole(smiles=smiles, name=name, functionalities=fgs))
            for fg in fgs:
                logger.debug(f"{name} ({smiles}): detected {fg.fg_name} "
                             f"(count={fg.fg_count_1})")
    return roles
