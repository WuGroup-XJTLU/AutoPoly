#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# The reaction-matching logic in this file is adapted from AutoREACTER's
# reaction_detector module.
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
Reaction Detection for the AutoPoly Reactor

Matches monomer functional-group inventories (see
:mod:`AutoPoly.reactor.functional_groups`) against the reaction library
(see :mod:`AutoPoly.reactor.reaction_library`) to enumerate every distinct
polymerization reaction that can occur in a monomer mixture. The matching
logic mirrors AutoREACTER's ``ReactionDetector`` (NanoCIPHER-Lab, MIT):

- ``same_reactants=True`` entries match any monomer carrying ``reactant_1``
  (A + A homo-polymerization).
- ``same_reactants=False`` entries match ordered monomer pairs where one
  carries ``reactant_1`` and the other ``reactant_2`` (A + B), as well as a
  single monomer carrying both groups (AB monomer self-condensation).

Created on 2026-08-04
@author: zwu
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.system import logger
from .functional_groups import FunctionalGroupInfo, MonomerRole
from .reaction_library import REACTIONS


@dataclass(frozen=True)
class ReactionInstance:
    """
    A specific reaction instance between identified monomers.

    Attributes:
        reaction_name: Name of the polymerization reaction (library key).
        reaction_smarts: Reaction SMARTS used to execute the transformation.
        delete_atom: Whether the reaction splits off a byproduct fragment.
        same_reactants: Whether this is an A + A homo-reaction.
        monomer_1: First participating monomer.
        functional_group_1: The reacting group on monomer 1.
        monomer_2: Second monomer (equals monomer_1 for self-reactions).
        functional_group_2: The reacting group on monomer 2.
        references: Literature pointers for the SMARTS / mechanism.
    """
    reaction_name: str
    reaction_smarts: str
    delete_atom: bool
    same_reactants: bool
    monomer_1: MonomerRole
    functional_group_1: FunctionalGroupInfo
    monomer_2: Optional[MonomerRole] = None
    functional_group_2: Optional[FunctionalGroupInfo] = None
    references: Dict[str, Any] = None

    @property
    def description(self) -> str:
        """Short human-readable summary of the instance."""
        m1 = f"{self.monomer_1.name}({self.functional_group_1.fg_name})"
        if self.monomer_2 is not None and self.functional_group_2 is not None:
            m2 = f"{self.monomer_2.name}({self.functional_group_2.fg_name})"
            return f"{self.reaction_name}: {m1} + {m2}"
        return f"{self.reaction_name}: {m1}"


def _matching_fgs(monomer: MonomerRole, group_name: str) -> List[FunctionalGroupInfo]:
    """Functional groups on a monomer matching a target group name."""
    return [fg for fg in monomer.functionalities if fg.fg_name == group_name]


def _seen_pair_key(
    reaction_name: str,
    monomer_1: MonomerRole,
    fg_1: FunctionalGroupInfo,
    monomer_2: Optional[MonomerRole] = None,
    fg_2: Optional[FunctionalGroupInfo] = None,
) -> Tuple:
    """Deduplication key for reaction instances (order-insensitive pairs)."""
    if monomer_2 is None or fg_2 is None:
        return (reaction_name, monomer_1.smiles, fg_1.fg_name)
    pair1 = (monomer_1.smiles, fg_1.fg_name)
    pair2 = (monomer_2.smiles, fg_2.fg_name)
    ordered = tuple(sorted([pair1, pair2]))
    return (reaction_name, ordered)


def detect_reactions(
    monomer_roles: List[MonomerRole],
    reaction_library: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ReactionInstance]:
    """
    Enumerate all distinct polymerization reactions between the monomers.

    Args:
        monomer_roles: Monomers annotated with functional groups.
        reaction_library: Reaction library (defaults to REACTIONS).

    Returns:
        Deduplicated list of ReactionInstance (empty if none found).
    """
    library = reaction_library or REACTIONS
    instances: List[ReactionInstance] = []
    seen: Set[Tuple] = set()

    for reaction_name, info in library.items():
        reactant_1 = info.get("reactant_1")
        reactant_2 = info.get("reactant_2")
        same_reactants = info.get("same_reactants", False)

        # CASE 1: homo-polymerization (A + A, single reactant group)
        if same_reactants and reactant_2 is None:
            for monomer in monomer_roles:
                for fg in _matching_fgs(monomer, reactant_1):
                    key = _seen_pair_key(reaction_name, monomer, fg)
                    if key not in seen:
                        seen.add(key)
                        instances.append(ReactionInstance(
                            reaction_name=reaction_name,
                            reaction_smarts=info["reaction"],
                            delete_atom=info["delete_atom"],
                            same_reactants=same_reactants,
                            monomer_1=monomer,
                            functional_group_1=fg,
                            references=info.get("reference"),
                        ))

        # CASE 2: co-polymerization between distinct monomers (A + B)
        else:
            for monomer_i in monomer_roles:
                fgs_i = _matching_fgs(monomer_i, reactant_1)
                if not fgs_i:
                    continue
                for fg_i in fgs_i:
                    for monomer_j in monomer_roles:
                        if monomer_i == monomer_j:
                            continue
                        for fg_j in _matching_fgs(monomer_j, reactant_2):
                            key = _seen_pair_key(
                                reaction_name, monomer_i, fg_i, monomer_j, fg_j
                            )
                            if key not in seen:
                                seen.add(key)
                                instances.append(ReactionInstance(
                                    reaction_name=reaction_name,
                                    reaction_smarts=info["reaction"],
                                    delete_atom=info["delete_atom"],
                                    same_reactants=same_reactants,
                                    monomer_1=monomer_i,
                                    functional_group_1=fg_i,
                                    monomer_2=monomer_j,
                                    functional_group_2=fg_j,
                                    references=info.get("reference"),
                                ))

        # CASE 2b: single monomer carrying BOTH groups (AB self-condensation)
        if not same_reactants and reactant_2 is not None:
            for monomer in monomer_roles:
                fgs_1 = _matching_fgs(monomer, reactant_1)
                fgs_2 = _matching_fgs(monomer, reactant_2)
                for fg_1 in fgs_1:
                    for fg_2 in fgs_2:
                        if fg_1 == fg_2:
                            continue
                        key = _seen_pair_key(
                            reaction_name, monomer, fg_1, monomer, fg_2
                        )
                        if key not in seen:
                            seen.add(key)
                            instances.append(ReactionInstance(
                                reaction_name=reaction_name,
                                reaction_smarts=info["reaction"],
                                delete_atom=info["delete_atom"],
                                same_reactants=same_reactants,
                                monomer_1=monomer,
                                functional_group_1=fg_1,
                                monomer_2=monomer,
                                functional_group_2=fg_2,
                                references=info.get("reference"),
                            ))

    logger.info(f"Detected {len(instances)} distinct reaction instance(s)")
    for inst in instances:
        logger.debug(f"  {inst.description}")
    return instances
