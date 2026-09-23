#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# The atom-mapping and template-walking algorithm in this file is adapted from
# AutoREACTER's prepare_reactions and walker modules.
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
Reaction Mapping and Template Walking for the AutoPoly Reactor

Executes a detected reaction with RDKit's reaction engine and derives the
atom bookkeeping needed for LAMMPS ``fix bond/react`` templates:

- full reactant->product atom-index mapping (isotope tracking survives the
  reaction engine, so every reactant atom is located in the product),
- *first shell* atoms (all atoms mapped < 999 in the reaction SMARTS,
  i.e. the reaction center carried into the product),
- the two *initiator* atoms (SMARTS map numbers 1 and 2) between which
  ``fix bond/react`` forms the new bond,
- *byproduct* atoms (the smallest product fragment, e.g. water/HCl) for
  reactions that delete atoms,
- the *template* subset: a breadth-first walk of ``max_bonds`` shells
  around the first shell, plus the *edge* atoms at the outermost shell.

The algorithm is adapted from AutoREACTER's ``PrepareReactions`` and
``walker`` modules (NanoCIPHER-Lab, MIT), reimplemented without pandas.

Created on 2026-08-04
@author: zwu
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from ..core.exceptions import GenerationError
from ..core.system import logger
from .detector import ReactionInstance


class ReactionMappingError(GenerationError):
    """Atom mapping between reactants and products failed or is inconsistent."""


@dataclass
class ReactionMetadata:
    """
    Complete atom bookkeeping for one executed reaction.

    All indices refer to the *combined* reactant/product molecules
    (reactant A followed by reactant B, 0-based), unless noted.

    Attributes:
        reaction_id: 1-based sequential identifier.
        instance: The ReactionInstance this metadata was built from.
        reactant_mol: Combined reactants (explicit Hs).
        product_mol: Combined products (explicit Hs).
        reactant_to_product: full mapping reactant idx -> product idx.
        product_to_reactant: inverse mapping product idx -> reactant idx.
        first_shell: reactant indices of reaction-center atoms.
        initiators: the two reactant indices of the initiator atoms.
        byproduct_indices: reactant indices of byproduct atoms (deleteIDs).
        template_atoms: reactant indices inside the template (walk result).
        edge_atoms: reactant indices at the template boundary (edgeIDs).
        template_reactant_to_product: mapping restricted to template atoms.
        delete_atom: whether this reaction deletes byproduct atoms.
    """
    reaction_id: int
    instance: ReactionInstance
    reactant_mol: Chem.Mol
    product_mol: Chem.Mol
    reactant_to_product: Dict[int, int]
    product_to_reactant: Dict[int, int]
    first_shell: List[int] = field(default_factory=list)
    initiators: List[int] = field(default_factory=list)
    byproduct_indices: List[int] = field(default_factory=list)
    template_atoms: List[int] = field(default_factory=list)
    edge_atoms: List[int] = field(default_factory=list)
    template_reactant_to_product: Dict[int, int] = field(default_factory=dict)
    delete_atom: bool = True


# ---------------------------------------------------------------------------
# Template walking (adapted from AutoREACTER walker.py)
# ---------------------------------------------------------------------------
def _new_neighbors(mol: Chem.Mol, atom_idx: int, shells: Dict[int, List[int]]) -> List[int]:
    """Neighbors of an atom not yet visited in any shell."""
    visited = {i for s in shells.values() for i in s}
    return [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if n.GetIdx() not in visited]


def reactant_atom_walker(
    mol: Chem.Mol,
    start_indices: List[int],
    max_bonds: int = 4,
) -> Tuple[List[int], List[int]]:
    """
    Breadth-first walk from the first shell up to ``max_bonds`` shells.

    Returns:
        (template_atoms, edge_atoms): all visited atom indices, and the
        outermost-shell indices (used as edgeIDs in the map file).
    """
    shells: Dict[int, List[int]] = {i: [] for i in range(1, max_bonds + 1)}
    shells[1] = list(start_indices)

    depth = 0
    while depth < max_bonds - 1:
        depth += 1
        for atom_idx in shells[depth]:
            for n in _new_neighbors(mol, atom_idx, shells):
                if n not in shells[depth + 1]:
                    shells[depth + 1].append(n)

    edge_atoms = shells[max_bonds]
    template_atoms = [i for s in shells.values() for i in s]
    return template_atoms, edge_atoms


# ---------------------------------------------------------------------------
# Reaction execution
# ---------------------------------------------------------------------------
def _assign_tracking_ids(r1: Chem.Mol, r2: Chem.Mol) -> None:
    """
    Tag reactant atoms with unique map numbers and isotopes.

    Isotopes survive RDKit's reaction engine, so original atom identity can
    be recovered in the products. Reactant 1 is tagged 1001+, reactant 2
    2001+.
    """
    for atom in r1.GetAtoms():
        tag = 1001 + atom.GetIdx()
        atom.SetAtomMapNum(tag)
        atom.SetIsotope(tag)
    for atom in r2.GetAtoms():
        tag = 2001 + atom.GetIdx()
        atom.SetAtomMapNum(tag)
        atom.SetIsotope(tag)


def _restore_map_numbers_from_isotopes(mol: Chem.Mol) -> None:
    """Restore atom map numbers from surviving isotope tags (and clear them)."""
    for atom in mol.GetAtoms():
        tag = atom.GetIsotope()
        if tag != 0:
            atom.SetAtomMapNum(tag)
            atom.SetIsotope(0)


def _reveal_template_map_numbers(mol: Chem.Mol) -> None:
    """Expose RDKit's internal 'old_mapno' (the SMARTS map numbers) as map numbers."""
    for atom in mol.GetAtoms():
        if atom.HasProp("old_mapno"):
            atom.SetAtomMapNum(atom.GetIntProp("old_mapno"))


def _clear_isotopes(*mols: Chem.Mol) -> None:
    for mol in mols:
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)


def _build_index_mapping(
    reactant: Chem.Mol, product: Chem.Mol
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Bidirectional reactant<->product index mapping via tracking map numbers."""
    product_map = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in product.GetAtoms() if a.GetAtomMapNum() != 0
    }
    forward: Dict[int, int] = {}
    for atom in reactant.GetAtoms():
        tag = atom.GetAtomMapNum()
        if tag != 0 and tag in product_map:
            forward[atom.GetIdx()] = product_map[tag]
    reverse = {v: k for k, v in forward.items()}
    return forward, reverse


def _validate_mapping(
    forward: Dict[int, int], reactant: Chem.Mol, product: Chem.Mol
) -> None:
    """Mapping must be complete and one-to-one."""
    if not forward:
        raise ReactionMappingError("Empty reactant->product mapping")
    r_idxs = list(forward.keys())
    p_idxs = list(forward.values())
    if len(r_idxs) != len(p_idxs):
        raise ReactionMappingError("Mapping size mismatch between reactant and product")
    if len(set(r_idxs)) != len(r_idxs) or len(set(p_idxs)) != len(p_idxs):
        raise ReactionMappingError("Duplicate indices in mapping (must be 1-to-1)")
    if any(i >= reactant.GetNumAtoms() for i in r_idxs):
        raise ReactionMappingError("Reactant index out of bounds in mapping")
    if any(i >= product.GetNumAtoms() for i in p_idxs):
        raise ReactionMappingError("Product index out of bounds in mapping")
    if len(r_idxs) != reactant.GetNumAtoms():
        raise ReactionMappingError(
            f"Incomplete mapping: {len(r_idxs)}/{reactant.GetNumAtoms()} reactant atoms mapped"
        )
    if len(p_idxs) != product.GetNumAtoms():
        raise ReactionMappingError(
            f"Incomplete mapping: {len(p_idxs)}/{product.GetNumAtoms()} product atoms mapped"
        )


def _assign_first_shell_and_initiators(
    reactant: Chem.Mol,
    product: Chem.Mol,
    reverse: Dict[int, int],
) -> Tuple[List[int], List[int]]:
    """
    Identify first-shell atoms (SMARTS map number < 999) and the two
    initiator atoms (map numbers 1 and 2), in reactant index space.

    Side effect: copies the SMARTS map numbers onto the reactant atoms.
    """
    first_shell: List[int] = []
    initiators: List[int] = []

    for p_atom in product.GetAtoms():
        map_num = p_atom.GetAtomMapNum()
        if map_num >= 999 or map_num == 0:
            continue
        p_idx = p_atom.GetIdx()
        if p_idx not in reverse:
            raise ReactionMappingError(f"Product atom {p_idx} missing from mapping")
        r_idx = reverse[p_idx]
        reactant.GetAtomWithIdx(r_idx).SetAtomMapNum(map_num)
        first_shell.append(r_idx)
        if map_num in (1, 2):
            initiators.append(r_idx)

    if len(initiators) != 2:
        raise ReactionMappingError(
            f"Expected 2 initiator atoms, got {len(initiators)}: {initiators}"
        )
    return first_shell, initiators


def _detect_byproducts(
    product: Chem.Mol,
    reverse: Dict[int, int],
    delete_atom: bool,
) -> List[int]:
    """
    Byproduct atoms = the smallest product fragment (e.g. water, HCl),
    mapped back to reactant index space. Empty when delete_atom is False.
    """
    if not delete_atom:
        return []
    frags = rdmolops.GetMolFrags(product)
    smallest = min(frags, key=len)
    return [reverse[i] for i in smallest if i in reverse]


def _canonical_combined(m1: Chem.Mol, m2: Chem.Mol) -> Chem.Mol:
    """Canonical combined reactant pair used for duplicate detection."""
    combined = Chem.CombineMols(Chem.Mol(m1), Chem.Mol(m2))
    for atom in combined.GetAtoms():
        atom.SetAtomMapNum(0)
    return combined


def _prepare_single_reaction(
    instance: ReactionInstance,
    reaction_id: int,
) -> Optional[ReactionMetadata]:
    """
    Execute one reaction instance and build its ReactionMetadata.

    Runs the reaction with explicit hydrogens on both reactant orderings
    when the reactants differ (first successful product set wins).

    Returns None if the reaction engine produced no products.
    """
    smiles_1 = instance.monomer_1.smiles
    smiles_2 = instance.monomer_1.smiles if instance.same_reactants or instance.monomer_2 is None \
        else instance.monomer_2.smiles

    m1 = Chem.MolFromSmiles(smiles_1)
    m2 = Chem.MolFromSmiles(smiles_2)
    if m1 is None or m2 is None:
        raise ReactionMappingError(
            f"Unparseable reactant SMILES: {smiles_1!r}, {smiles_2!r}"
        )
    m1 = Chem.AddHs(m1)
    m2 = Chem.AddHs(m2)

    rxn = AllChem.ReactionFromSmarts(instance.reaction_smarts)
    if rxn is None:
        raise ReactionMappingError(
            f"Invalid reaction SMARTS for {instance.reaction_name}: "
            f"{instance.reaction_smarts!r}"
        )

    # Try both orderings for hetero-reactions; one ordering for self-reactions
    orderings = [(m1, m2)] if (instance.same_reactants or m1 is m2 or smiles_1 == smiles_2) \
        else [(m1, m2), (m2, m1)]

    for ra, rb in orderings:
        ra = Chem.Mol(ra)
        rb = Chem.Mol(rb)
        _assign_tracking_ids(ra, rb)
        products = rxn.RunReactants((ra, rb))
        if not products:
            continue

        product_set = products[0]
        reactant_combined = Chem.CombineMols(ra, rb)
        product_combined = product_set[0] if len(product_set) == 1 \
            else Chem.CombineMols(*product_set)

        # Sanitize so ring info / property caches are available for typing
        try:
            Chem.SanitizeMol(reactant_combined)
            Chem.SanitizeMol(product_combined)
        except Exception:
            pass

        # Recover atom identity in the product, then build the full mapping
        _restore_map_numbers_from_isotopes(product_combined)
        forward, reverse = _build_index_mapping(reactant_combined, product_combined)

        # Switch the product's map numbers to the SMARTS template numbers
        _reveal_template_map_numbers(product_combined)

        _validate_mapping(forward, reactant_combined, product_combined)
        first_shell, initiators = _assign_first_shell_and_initiators(
            reactant_combined, product_combined, reverse
        )
        byproducts = _detect_byproducts(product_combined, reverse, instance.delete_atom)

        # Template subset: BFS walk around the first shell
        template_atoms, edge_atoms = reactant_atom_walker(reactant_combined, first_shell)
        template_map = {r: forward[r] for r in template_atoms if r in forward}

        # Restore normal chemistry for downstream consumers
        _clear_isotopes(reactant_combined, product_combined)

        return ReactionMetadata(
            reaction_id=reaction_id,
            instance=instance,
            reactant_mol=reactant_combined,
            product_mol=product_combined,
            reactant_to_product=forward,
            product_to_reactant=reverse,
            first_shell=first_shell,
            initiators=initiators,
            byproduct_indices=byproducts,
            template_atoms=template_atoms,
            edge_atoms=edge_atoms,
            template_reactant_to_product=template_map,
            delete_atom=instance.delete_atom,
        )

    logger.warning(f"Reaction SMARTS produced no products for {instance.description}")
    return None


def prepare_reactions(
    instances: List[ReactionInstance],
    max_bonds: int = 4,
) -> List[ReactionMetadata]:
    """
    Execute all detected reaction instances and build their metadata.

    Duplicate reactions (same reactant pair, same product) are dropped —
    the first occurrence is kept.

    Args:
        instances: Detected reaction instances.
        max_bonds: Template walk depth (LAMMPS recommends the template
                   cover all atoms whose topology changes; 4 matches
                   AutoREACTER's default).

    Returns:
        List of ReactionMetadata, one per unique reaction.
    """
    del max_bonds  # walk depth currently fixed to the walker default
    metadata: List[ReactionMetadata] = []
    seen_pairs = set()

    for instance in instances:
        meta = _prepare_single_reaction(instance, reaction_id=len(metadata) + 1)
        if meta is None:
            continue

        # Duplicate detection on canonical reactant+product SMILES
        for atom in meta.reactant_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        for atom in meta.product_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        key = (
            Chem.MolToSmiles(meta.reactant_mol),
            Chem.MolToSmiles(meta.product_mol),
        )
        if key in seen_pairs:
            logger.debug(f"Dropping duplicate reaction: {instance.description}")
            continue
        seen_pairs.add(key)

        meta.reaction_id = len(metadata) + 1
        metadata.append(meta)

    logger.info(f"Prepared {len(metadata)} unique reaction template set(s)")
    return metadata
