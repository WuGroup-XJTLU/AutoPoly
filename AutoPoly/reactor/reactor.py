#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# The reaction-orchestration workflow in this file is adapted in part from
# AutoREACTER.
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
Reactor Orchestrator for AutoPoly

The :class:`Reactor` adds AutoREACTER-style reactive-MD preparation to an
AutoPoly project: it scans the monomers/molecules packed into a generated
system, detects step-growth polymerization reactions between them, builds
the LAMMPS ``fix bond/react`` template triplet (pre/post molecule
templates + map file) for each unique reaction, and writes a ready-to-run
``in.bond_react`` script alongside the stage-3 outputs.

Typical use::

    from AutoPoly import System, Molecule, generate, Reactor

    system = System(out="polyester")
    eg = Molecule(Count=20, Smiles="OCCO", Name="eg")
    aa = Molecule(Count=20, Smiles="O=C(O)CCCCC(=O)O", Name="adipic")
    generate(system, "melt", [eg, aa], force_field="gaff2")

    reactor = Reactor(system.get_folder_path() + "/melt", monomers=[eg, aa])
    print(reactor.detect_reactions())
    result = reactor.build()   # writes reactor/ + in.bond_react

Created on 2026-08-04
@author: zwu
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.exceptions import GenerationError, ValidationError
from ..core.system import logger
from ..monomers.monomer_generator import SMARTSTyper
from .detector import ReactionInstance, detect_reactions
from .fftables import ForceFieldTables, load_force_field_tables
from .functional_groups import (
    MonomerRole,
    detect_monomer_roles,
    FUNCTIONAL_GROUPS,
)
from .lammps import write_bond_react_script, write_settings_reactor
from .mapper import ReactionMetadata, prepare_reactions
from .system_data import SystemData, parse_in_init
from .templates import TemplateBuilder, TemplateSet, TypeAssignment

REACTOR_DIRNAME = "reactor"


@dataclass
class ReactorResult:
    """
    Output manifest of :meth:`Reactor.build`.

    Attributes:
        output_dir: The ``reactor/`` directory holding template files.
        script: The ``in.bond_react`` LAMMPS input script.
        reactions: Human-readable summaries of the built reactions.
        template_files: All written template/map/settings file paths.
        template_sets: The per-reaction TemplateSet records.
        assignments: Per-reaction numeric type assignments.
        warnings: Non-fatal issues encountered during the build.
    """
    output_dir: Path
    script: Path
    reactions: List[str] = field(default_factory=list)
    template_files: List[Path] = field(default_factory=list)
    template_sets: List[TemplateSet] = field(default_factory=list)
    assignments: List[TypeAssignment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Reactor:
    """
    Prepare LAMMPS ``fix bond/react`` inputs for an AutoPoly system.

    Args:
        project_dir: AutoPoly project directory (contains ``system.data``,
            ``system.in.init`` and ``build/<force_field>/units.json``).
        monomers: Optional explicit monomer list (``Molecule``/``Polymer``
            models or ``(smiles, name)`` tuples). When omitted, monomers are
            read from ``build/<force_field>/units.json`` plus the geometry.
        force_field: Force field used to type the system. When omitted it is
            inferred from the single ``build/<ff>/`` directory present.
        functional_groups: Optional custom functional-group library.
        reaction_library: Optional custom reaction library.
    """

    def __init__(
        self,
        project_dir,
        monomers: Optional[List[Any]] = None,
        force_field: Optional[str] = None,
        functional_groups: Optional[Dict] = None,
        reaction_library: Optional[Dict] = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        if not self.project_dir.is_dir():
            raise ValidationError(f"Project directory not found: {self.project_dir}")

        self.functional_groups = functional_groups or FUNCTIONAL_GROUPS
        self.reaction_library = reaction_library

        self.data_path = self.project_dir / "system.data"
        if not self.data_path.is_file():
            raise ValidationError(
                f"system.data not found in {self.project_dir}; "
                "run generate() (stage 3) before building a reactor"
            )
        self.init_path = self.project_dir / "system.in.init"

        self.force_field = force_field or self._infer_force_field()
        self._monomer_inputs = monomers

        # Parsed lazily by build(); exposed for inspection/tests
        self.system_data: Optional[SystemData] = None
        self.ff_tables: Optional[ForceFieldTables] = None
        self.monomer_roles: Optional[List[MonomerRole]] = None
        self.reaction_instances: Optional[List[ReactionInstance]] = None
        self.reaction_metadata: Optional[List[ReactionMetadata]] = None

    # ------------------------------------------------------------------
    # Monomer / force-field resolution
    # ------------------------------------------------------------------
    def _infer_force_field(self) -> str:
        """Infer the force field from the single build/<ff>/ directory."""
        build_dir = self.project_dir / "build"
        candidates = []
        if build_dir.is_dir():
            candidates = [p.name for p in build_dir.iterdir()
                          if p.is_dir() and (p / "units.json").is_file()]
        if len(candidates) == 1:
            logger.info(f"Inferred force field '{candidates[0]}' from build directory")
            return candidates[0]
        if not candidates:
            raise ValidationError(
                f"No build/<ff>/units.json under {self.project_dir}; "
                "pass force_field=... explicitly"
            )
        raise ValidationError(
            f"Multiple typed force fields found {candidates}; "
            "pass force_field=... to choose one"
        )

    @staticmethod
    def _model_to_smiles_name(model: Any) -> Optional[Tuple[str, str]]:
        """(smiles, name) from a Molecule/Polymer model or (smiles, name) pair."""
        if isinstance(model, (tuple, list)) and len(model) == 2:
            return str(model[0]), str(model[1])
        smiles = getattr(model, "Smiles", None)
        name = getattr(model, "molecule_name", None) or getattr(model, "name", None)
        if smiles is None:
            return None
        return smiles, (name or smiles)

    def _resolve_monomers(self) -> List[Tuple[str, str]]:
        """
        Resolve (smiles, name) pairs for every reactive monomer.

        Explicit ``monomers=`` win; otherwise monomers are read from the
        stage-1 geometry (chain graph + molecule mapped SMILES, wildcards
        stripped) so that reactor can run with just a project directory.
        """
        if self._monomer_inputs is not None:
            pairs = []
            for model in self._monomer_inputs:
                pair = self._model_to_smiles_name(model)
                if pair is None:
                    logger.warning(f"Skipping monomer without SMILES: {model!r}")
                    continue
                pairs.append(pair)
            return pairs

        # Fall back to geometry.json
        geometry_path = self.project_dir / "geometry" / "geometry.json"
        if not geometry_path.is_file():
            raise ValidationError(
                f"No monomers given and no geometry.json under {self.project_dir}; "
                "pass monomers=... (Molecule/Polymer models or (smiles, name) pairs)"
            )
        import json
        geometry = json.loads(geometry_path.read_text())
        pairs: List[Tuple[str, str]] = []

        def _strip_mapping(smiles: str) -> str:
            """Remove atom-map numbers and wildcards from a mapped SMILES."""
            import re
            smiles = re.sub(r":\d+\]", "]", smiles)
            smiles = smiles.replace("[*]", "").replace("*", "")
            return smiles

        for model_id, graph in geometry.get("chain_graphs", {}).items():
            smiles = _strip_mapping(graph.get("smiles_mapped", ""))
            if smiles:
                pairs.append((smiles, graph.get("name", model_id)))
        for entry in geometry.get("molecules", []):
            smiles = _strip_mapping(entry.get("smiles_mapped", ""))
            if smiles:
                pairs.append((smiles, entry.get("name", smiles)))

        # Deduplicate by smiles while preserving order
        seen = set()
        unique = []
        for smiles, name in pairs:
            if smiles not in seen:
                seen.add(smiles)
                unique.append((smiles, name))
        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_reactions(self) -> List[ReactionInstance]:
        """
        Scan the monomers and enumerate every distinct possible reaction.

        Returns:
            The detected ReactionInstance list (also stored on the reactor).
        """
        monomer_pairs = self._resolve_monomers()
        self.monomer_roles = detect_monomer_roles(monomer_pairs, self.functional_groups)
        self.reaction_instances = detect_reactions(self.monomer_roles, self.reaction_library)
        return self.reaction_instances

    def build(
        self,
        temperature: float = 300.0,
        nevery: int = 100,
        rmin: float = 0.0,
        rmax: float = 3.5,
        prob: float = 1.0,
        run_steps: int = 1_000_000,
        stabilize_steps: int = 60,
        seed: int = 42,
        output_dir: Optional[Path] = None,
    ) -> ReactorResult:
        """
        Build all reaction templates and the fix bond/react input script.

        Args:
            temperature: Reaction-stage thermostat temperature (K).
            nevery: Reaction attempt frequency (timesteps).
            rmin / rmax: Initiator-atom distance cutoffs (Angstrom).
            prob: Reaction probability once constraints are met.
            run_steps: Reaction-stage run length (timesteps).
            stabilize_steps: nve/limit stabilization after each reaction.
            seed: RNG seed for template coordinate embedding.
            output_dir: Output directory (default <project_dir>/reactor).

        Returns:
            ReactorResult manifest.

        Raises:
            GenerationError: If no reactions are possible, or template
                             construction fails for every reaction.
        """
        if self.reaction_instances is None:
            self.detect_reactions()
        if not self.reaction_instances:
            raise GenerationError(
                "No polymerization reactions detected between the system monomers. "
                "Check that the monomers carry reactive functional groups covered "
                "by the reaction library."
            )

        out_dir = Path(output_dir) if output_dir else self.project_dir / REACTOR_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse the simulation's data + force-field tables
        self.system_data = SystemData(self.data_path)
        self.ff_tables = load_force_field_tables(self.force_field)
        typer = SMARTSTyper(self.force_field, verbose=False)
        builder = TemplateBuilder(typer, self.ff_tables, self.system_data)

        metadata_list = prepare_reactions(self.reaction_instances)
        if not metadata_list:
            raise GenerationError(
                "Reaction execution produced no usable templates for any "
                "detected reaction."
            )
        self.reaction_metadata = metadata_list

        template_sets: List[TemplateSet] = []
        assignments: List[TypeAssignment] = []
        warnings: List[str] = []
        for meta in metadata_list:
            try:
                ts, assignment = builder.build(meta, out_dir, seed=seed)
            except Exception as e:  # keep building the remaining reactions
                msg = (f"Skipping reaction {meta.reaction_id} "
                       f"({meta.instance.reaction_name}): {e}")
                logger.warning(msg)
                warnings.append(msg)
                continue
            template_sets.append(ts)
            assignments.append(assignment)

        if not template_sets:
            raise GenerationError(
                "Template construction failed for every detected reaction: "
                + "; ".join(warnings)
            )

        # Supplementary settings for reaction-introduced types (coeffs after
        # read_data; masses before read_data)
        settings_reactor = out_dir / "system.in.settings.reactor"
        masses_reactor = out_dir / "system.in.masses.reactor"
        has_reactor_settings = write_settings_reactor(
            settings_reactor, assignments, masses_path=masses_reactor)
        has_reactor_masses = masses_reactor.is_file()

        # Highest numeric type ids introduced across all templates (for
        # create_box sizing) and the simulation box bounds.
        max_new_types = {"atom": 0, "bond": 0, "angle": 0,
                         "dihedral": 0, "improper": 0}
        for a in assignments:
            for kind in max_new_types:
                max_new_types[kind] = max(max_new_types[kind],
                                          a.max_ids.get(kind, 0))
        box_bounds = self.system_data.box_bounds

        # The fix bond/react script lives next to system.data so its
        # relative includes resolve.
        script_path = self.project_dir / "in.bond_react"
        template_names = [ts for ts in template_sets]
        write_bond_react_script(
            script_path,
            template_names,
            temperature=temperature,
            nevery=nevery,
            rmin=rmin,
            rmax=rmax,
            prob=prob,
            stabilize_steps=stabilize_steps,
            run_steps=run_steps,
            data_file=self.data_path.name,
            init_file=self.init_path.name,
            settings_file="system.in.settings",
            charges_file=(
                "system.in.charges"
                if (self.project_dir / "system.in.charges").is_file() else None
            ),
            reactor_settings_file=(
                f"{REACTOR_DIRNAME}/system.in.settings.reactor"
                if has_reactor_settings else None
            ),
            reactor_masses_file=(
                f"{REACTOR_DIRNAME}/system.in.masses.reactor"
                if has_reactor_masses else None
            ),
            box_bounds=box_bounds if has_reactor_masses else None,
            max_new_types=max_new_types if has_reactor_masses else None,
            seed=seed,
        )

        # Copy templates referenced by the script into the reactor dir only;
        # the script references them by basename, so also copy next to it.
        import shutil
        for ts in template_sets:
            for f in (ts.pre_file, ts.post_file, ts.map_file,
                      ts.map_file_delete_ids):
                if f is not None and Path(f).is_file() and Path(f).parent != self.project_dir:
                    shutil.copy2(f, self.project_dir / Path(f).name)

        result = ReactorResult(
            output_dir=out_dir,
            script=script_path,
            reactions=[ts.reaction_name for ts in template_sets],
            template_sets=template_sets,
            assignments=assignments,
            warnings=warnings,
        )
        for ts in template_sets:
            result.template_files.extend(
                p for p in (ts.pre_file, ts.post_file, ts.map_file,
                            ts.map_file_delete_ids) if p is not None
            )
        if has_reactor_settings:
            result.template_files.append(settings_reactor)
        result.template_files.append(script_path)

        logger.info(
            f"Reactor build complete: {len(template_sets)} reaction template "
            f"set(s) -> {out_dir}; script: {script_path}"
        )
        return result
