#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box Packer — Stage 3 of the AutoPoly three-stage pipeline.

Packs typed units into a simulation box and runs moltemplate:

    build/<ff>/ (units.json + typed .lt) ──▶ moltemplate/ ──▶ system.data ...

The BoxPacker:
1. Accepts a UnitLibrary or a build-directory path (loads + validates units.json)
2. Copies the typed unit/monomer .lt files into <name>/moltemplate/ (keeps the
   self-contained layout; build/ remains the reusable source)
3. Generates the force-field files from the manifest's monomer files
   (no model objects needed — subsetting reads file paths)
4. Runs the selected packing strategy -> system.lt
5. Runs moltemplate (-nocheck), validates output, post-processes
   (get_rid_of_lj_cut_coul_long, mv_files)
6. Inorganic components: writes explicit organic-inorganic cross pair
   coefficients (fillers), component groups (system.in.groups) from the
   moltemplate instance assignments, a run template (in.run) and the
   force-field compatibility report (ff_compatibility.json/.md)

Created on 2026-07-30
@author: zwu
"""
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..core.system import logger
from ..core.conf import FORCE_FIELD_REGISTRY
from ..core.exceptions import WorkflowError
from ..core.file_management import get_rid_of_lj_cut_coul_long, mv_files
from ..forcefields.force_field import ForceFieldManager
from ..packing import (
    BoxSpec,
    PackingContext,
    PlacementResult,
    get_strategy,
)
from .units import UnitLibrary

MOLTEMPLATE_DIRNAME = "moltemplate"


class BoxPacker:
    """
    Stage 3: pack typed units into a box and produce LAMMPS input files.

    Example:
        >>> packer = BoxPacker(system, "peo", strategy="mc_random")
        >>> packer.pack("peo_run/peo/build/gaff2")
    """

    def __init__(
        self,
        system: object,
        name: str,
        strategy: str = "mc_random",
        box_size: Optional[float] = None,
        mc_max_attempts: int = 10000,
        monomer_density: float = 0.085,
        rng_seed: Optional[int] = None,
        run_moltemplate: bool = True,
        substrate: Optional[object] = None,
        subtract: Optional[list] = None,
        box_dims: Optional[tuple] = None,
        packing: Optional[object] = None,
    ) -> None:
        """
        Args:
            system: System object providing get_folder_path().
            name: Project name; output goes to <out>/<name>/moltemplate/.
            strategy: Registered packing strategy name ("mc_random", "grid",
                      "on_substrate", or a user-registered strategy).
            box_size: Explicit cubic box side (Angstrom); None = auto-size.
            mc_max_attempts: Max random placement attempts per instance.
            monomer_density: Target monomer density (monomers/A^3) for box sizing.
            rng_seed: Optional seed for reproducible stochastic strategies.
            run_moltemplate: Run moltemplate + post-processing (False = stop
                             after writing system.lt; useful for testing).
            substrate: Optional SubstrateSpec (required by the
                       "on_substrate" strategy).
            subtract: Optional list of CarveRegion specs (whole-instance
                      removal; supported by "mc_random" and "on_substrate").
            box_dims: Optional per-axis box sides (lx, ly, lz) in Angstrom;
                      any element may be None (auto-sized). Overrides
                      box_size per axis where given.
            packing: Optional strategy configuration (NanocompositePacking
                      for "nanocomposite_mc").

        Raises:
            ValidationError: If the strategy name is not registered.
        """
        self.system = system
        self.name = name
        self.project_dir = Path(system.get_folder_path()) / name
        self.moltemplate_dir = self.project_dir / MOLTEMPLATE_DIRNAME
        # Validate the strategy name eagerly (raises for unknown names)
        self.strategy = get_strategy(strategy)
        self.strategy_name = strategy
        self.box_size = box_size
        self.mc_max_attempts = mc_max_attempts
        self.monomer_density = monomer_density
        self.rng_seed = rng_seed
        self.run_moltemplate = run_moltemplate
        self.substrate = substrate
        self.subtract = subtract
        self.box_dims = box_dims
        self.packing = packing

        self.path_master = str(Path(__file__).parent.parent.resolve() / "extern/")
        self.path_moltemplatesrc = str(
            Path(self.path_master) / "moltemplate" / "scripts/"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pack(self, build: Union[str, Path, UnitLibrary]) -> PlacementResult:
        """
        Pack the units from a stage-2 build into the simulation box.

        Args:
            build: UnitLibrary, or a build directory / units.json path.

        Returns:
            PlacementResult from the strategy (instances + box bounds).

        Raises:
            WorkflowError: moltemplate failures, missing system.lt, missing
                           output files (same messages as the legacy workflow).
        """
        units = self._load_units(build)
        build_dir = self._resolve_build_dir(build)
        self._inorganic_only = bool(units.units) and not units.organic_units

        # 1. Fresh moltemplate dir + copy typed .lt files from build/<ff>/
        self._prepare_moltemplate_dir(units, build_dir)

        # 2. Force-field files, driven by the manifest's monomer files
        self._generate_force_field_files(units)

        # 3. Strategy -> placements -> system.lt
        result = self._run_strategy(units)
        self._write_system_lt(units, result)

        # 4. moltemplate + validation + post-processing
        if self.run_moltemplate:
            self._run_moltemplate_and_validate(units.force_field)
            self._patch_in_init_styles()
            logger.info("Processing output files")
            get_rid_of_lj_cut_coul_long(self.moltemplate_dir)
            cross = None
            if units.filler_units:
                cross = self._write_cross_terms(units)
            mv_files(self.moltemplate_dir)
            if self.substrate is not None:
                self._write_run_script(result)
            if units.filler_units:
                groups = self._write_component_groups(units, result)
                self._write_nanocomposite_run_script(units, result, groups)
            if units.inorganic_units:
                self._write_ff_report(units, result, cross)

        logger.info("Successfully completed polymer generation")
        return result

    # ------------------------------------------------------------------
    # Build dir / manifest loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_units(build: Union[str, Path, UnitLibrary]) -> UnitLibrary:
        if isinstance(build, UnitLibrary):
            return build
        return UnitLibrary.load(build)

    @staticmethod
    def _resolve_build_dir(build: Union[str, Path, UnitLibrary]) -> Optional[Path]:
        """Directory holding the typed .lt files (None if unknown)."""
        if isinstance(build, UnitLibrary):
            return Path(build.source_dir) if build.source_dir else None
        build = Path(build)
        return build if build.is_dir() else build.parent

    # ------------------------------------------------------------------
    # Step 1: moltemplate dir + file copying
    # ------------------------------------------------------------------
    def _prepare_moltemplate_dir(
        self, units: UnitLibrary, build_dir: Optional[Path]
    ) -> None:
        """
        Create the moltemplate dir and copy typed .lt files into it.

        Any previous moltemplate/ contents are removed first: everything in
        it is regenerated from build/<ff>/, and stale files (moved output of
        an earlier pack) would otherwise collide with the new run.
        """
        if self.moltemplate_dir.exists():
            shutil.rmtree(self.moltemplate_dir)
        self.moltemplate_dir.mkdir(parents=True, exist_ok=True)

        if build_dir is None:
            # In-memory library: .lt files must already be in place
            return

        units.validate(build_dir)
        # Source .lt files may be read-only (e.g. restaged from an immutable
        # artifact store); copy2 preserves that mode, so force each working
        # copy writable, and skip unit files already copied as monomer files
        # (a molecule is listed in both) — re-copying onto the read-only
        # destination would fail with EACCES.
        for rel in units.monomer_files:
            dst = self.moltemplate_dir / rel
            shutil.copy2(build_dir / rel, dst)
            dst.chmod(0o644)
        for unit in units.units:
            # the unit's class file plus any class it imports (e.g. the
            # core class of a grafted nanoparticle composite)
            for rel in [unit.lt_file] + list(unit.monomer_files):
                src = build_dir / rel
                dst = self.moltemplate_dir / rel
                if src.is_file() and not dst.exists():
                    shutil.copy2(src, dst)
                    dst.chmod(0o644)

        # External substrate slab: user-supplied .lt, imported as-is
        if self.substrate is not None and self.substrate.is_external:
            src = Path(self.substrate.lt_file)
            if not src.is_file():
                raise WorkflowError(
                    f"External substrate lt_file not found: {src}"
                )
            dst = self.moltemplate_dir / src.name
            shutil.copy2(src, dst)
            dst.chmod(0o644)

        # Built-in substrate slab: typed in stage 2 (inorganic substrate
        # unit, .lt already copied above), else generated at pack time
        # (legacy route for builds without the stage-1/2 slab).
        if self.substrate is not None and self.substrate.is_builder:
            typed = self._typed_substrate_unit(units)
            if typed is not None:
                from ..surfaces.component import SlabInfo
                self.substrate._built_slab = SlabInfo(typed)
                logger.info(
                    f"Using stage-2 typed substrate slab {typed.lt_file} "
                    f"({typed.class_name})"
                )
            else:
                self._build_substrate_slab(units.force_field)

    @staticmethod
    def _typed_substrate_unit(units: UnitLibrary):
        """The stage-2 typed built-in slab unit, if the build has one."""
        slabs = [u for u in units.units
                 if u.is_inorganic and u.role == "substrate"
                 and u.geometry_type == "slab"]
        if len(slabs) > 1:
            raise WorkflowError(
                f"Build contains {len(slabs)} substrate slab units; at most "
                "one substrate is supported"
            )
        return slabs[0] if slabs else None

    def _ff_lt_for_styles(self, force_field: str) -> Optional[Path]:
        """
        Path of the force field .lt whose style declarations to read.

        For openff this is the per-system file generated into the build dir
        and copied here by _prepare_moltemplate_dir (no bundled extern file
        exists). Returns None when unavailable.
        """
        if force_field == "openff":
            generated = self.moltemplate_dir / FORCE_FIELD_REGISTRY[force_field]["lt_file"]
            return generated if generated.is_file() else None
        return (
            Path(self.path_master) / "moltemplate" / "force_fields"
            / FORCE_FIELD_REGISTRY[force_field]["lt_file"]
        )

    def host_styles(self, force_field: str):
        """HostStyles of the film/matrix force field (shared parser)."""
        from ..materials.forcefields.compatibility import host_styles
        return host_styles(force_field, self._ff_lt_for_styles(force_field))

    def _film_bond_substyle(self, force_field: str) -> Optional[str]:
        """'harmonic' if the film force field uses a hybrid bond style."""
        return self.host_styles(force_field).bond_substyle

    def _film_pair_substyle(self, force_field: str) -> Optional[str]:
        """
        LAMMPS pair sub-style the slab's pair_coeff lines must carry.

        Read from the film force field's .lt pair_style line: with a
        hybrid pair style every pair_coeff needs a sub-style (the LJ
        one); with a plain pair style, None.  Class2 force fields store
        pair coefficients in the data file, which the slab writers do
        not support.
        """
        host = self.host_styles(force_field)
        if host.is_class2:
            raise WorkflowError(
                f"Built-in substrate builders do not support class2 "
                f"force fields ({force_field}); build the slab outside "
                "AutoPoly and use SubstrateSpec(lt_file=..., ...) instead"
            )
        if host.is_hybrid_pair:
            if host.pair_substyle is None:
                ff_lt = self._ff_lt_for_styles(force_field)
                raise WorkflowError(
                    f"Could not find an LJ sub-style in the pair_style line "
                    f"of {ff_lt.name if ff_lt else force_field}: "
                    f"'{host.pair_style}'"
                )
            return host.pair_substyle
        return None

    def _build_substrate_slab(self, force_field: str = "gaff") -> None:
        """
        Materialize a built-in substrate slab (SubstrateSpec.builder=...).

        Requires explicit lateral box_dims (the slab must be laterally
        periodic with the box); the box lateral sides are snapped to
        integer surface cells by the on_substrate strategy.  Writes the
        slab .lt into the moltemplate dir and stashes the SurfaceSlab
        on the spec (``spec._built_slab``) for the strategy.
        """
        from ..surfaces import SLAB_BUILDERS

        spec = self.substrate
        dims = self.box_dims or (None, None, None)
        if dims[0] is None or dims[1] is None:
            raise WorkflowError(
                "Builder substrates require explicit lateral box_dims "
                "(box_dims=(lx, ly, lz); lz may be None) so the slab can "
                "be built to match the box"
            )
        if spec.builder not in SLAB_BUILDERS:
            raise WorkflowError(
                f"Unknown substrate builder '{spec.builder}'"
            )
        builder = SLAB_BUILDERS[spec.builder](
            lx_target=dims[0],
            ly_target=dims[1],
            thickness=spec.thickness,
            oh_density=spec.oh_density,
            hydroxylate_bottom=spec.hydroxylate_bottom,
            slab_ff=spec.slab_ff,
            slab_types=spec.slab_types,
            slab_charges=spec.slab_charges,
            slab_lj=spec.slab_lj,
            seed=spec.slab_seed,
            pair_substyle=self._film_pair_substyle(force_field),
            bond_substyle=self._film_bond_substyle(force_field),
        )
        slab = builder.build()
        slab.write_lt(self.moltemplate_dir)
        spec._built_slab = slab

    # ------------------------------------------------------------------
    # Step 2: force-field files
    # ------------------------------------------------------------------
    def _generate_force_field_files(self, units: UnitLibrary) -> None:
        """Generate force field files from the manifest's monomer files."""
        force_field = units.force_field
        logger.info(f"Generating {force_field}.lt")
        if not units.monomer_files and units.inorganic_units:
            # inorganic-only system (e.g. a box of nanoparticles): nothing
            # to subset; the host file only provides the In Init styles
            self._write_init_only_force_field(force_field)
            return

        monomer_paths = [
            str(self.moltemplate_dir / rel) for rel in units.monomer_files
        ]
        ff_manager = ForceFieldManager(
            path_cwd=str(self.moltemplate_dir),
            path_master=self.path_master,
            path_moltemplatesrc=self.path_moltemplatesrc,
            force_field=force_field,
            ff_modify_dihedral=np.array(
                [0.6446926386, -0.2143420172, 0.1782194073, 0.0]
            ),
            monomer_files=monomer_paths,
        )
        ff_manager.make_force_field_lt()

        # Modify alkyl dihedral coefficients if needed (oplsaa only; LOPLS has
        # optimized alkyl dihedrals built in, other FFs use their own)
        if force_field == "oplsaa":
            ff_manager.FFmodify_alkyl_dihedral_oplsaa()

    def _write_init_only_force_field(self, force_field: str) -> None:
        """Host FF .lt with only its write_once("In Init") block."""
        import re

        entry = FORCE_FIELD_REGISTRY[force_field]
        src = self._ff_lt_for_styles(force_field)
        if src is None or not src.is_file():
            raise WorkflowError(
                f"Force-field file for '{force_field}' not found"
            )
        lines = src.read_text().splitlines()
        block, depth, inside = [], 0, False
        for line in lines:
            if not inside and re.search(r'write_once\("In Init"\)\s*\{', line):
                inside, depth = True, 1
                block.append(line)
                continue
            if inside:
                depth += line.count("{") - line.count("}")
                block.append(line)
                if depth <= 0:
                    break
        if not block:
            raise WorkflowError(
                f"No In Init block in {src.name}; cannot write the host "
                "styles for an inorganic-only system"
            )
        text = "\n".join([
            f"# {entry['lt_file']}: In Init styles of the {force_field} force "
            "field only (inorganic-only system; no organic types)",
            f"{entry['inherits']} {{", *block, "}", ""])
        (self.moltemplate_dir / entry["lt_file"]).write_text(text)

    # ------------------------------------------------------------------
    # Step 3: strategy + system.lt
    # ------------------------------------------------------------------
    def _run_strategy(self, units: UnitLibrary) -> PlacementResult:
        """Run the configured packing strategy."""
        offset = float(units.build_config.get("offset", 4.0))
        ctx = PackingContext(
            units=units,
            box=BoxSpec(
                requested_box_size=self.box_size,
                box_dims=self.box_dims or (None, None, None),
            ),
            mc_max_attempts=self.mc_max_attempts,
            rng_seed=self.rng_seed,
            offset=offset,
            monomer_density=self.monomer_density,
            substrate=self.substrate,
            subtract=self.subtract,
            nanocomposite=self.packing,
        )
        logger.info(f"Placing units with strategy '{self.strategy_name}'")
        return self.strategy.place(ctx)

    def _write_system_lt(
        self, units: UnitLibrary, result: PlacementResult
    ) -> Path:
        """Write system.lt: imports + placements + Data Boundary."""
        ff_lt = FORCE_FIELD_REGISTRY[units.force_field]["lt_file"]
        output = self.moltemplate_dir / "system.lt"

        with open(output, "w") as f:
            f.write(f'import "{ff_lt}"\n\n')

            # Unit imports (polymer chain files import their own monomers)
            seen = set()
            for unit in units.units:
                if unit.lt_file in seen:
                    continue
                seen.add(unit.lt_file)
                f.write(f'import "{unit.lt_file}"\n')

            # External substrate slab (already copied into this directory)
            if self.substrate is not None and self.substrate.is_external:
                slab_lt = Path(self.substrate.lt_file).name
                if slab_lt not in seen:
                    f.write(f'import "{slab_lt}"\n')

            # Built-in substrate slab (generated into this directory)
            if self.substrate is not None and self.substrate.is_builder:
                built = getattr(self.substrate, "_built_slab", None)
                if built is not None and built.lt_filename not in seen:
                    f.write(f'import "{built.lt_filename}"\n')

            # Strategy-generated classes (e.g. chains regrown in place)
            for fname, text in sorted(result.extra_lt_files.items()):
                (self.moltemplate_dir / fname).write_text(text)
                if fname not in seen:
                    f.write(f'import "{fname}"\n')
            f.write("\n")

            for record in result.records:
                f.write(record.lt_command + "\n")
            f.write("\n")

            (xmin, xmax), (ymin, ymax), (zmin, zmax) = result.box_bounds
            f.write('write_once("Data Boundary") {\n')
            f.write(f"   {xmin:.4f}  {xmax:.4f}  xlo xhi\n")
            f.write(f"   {ymin:.4f}  {ymax:.4f}  ylo yhi\n")
            f.write(f"   {zmin:.4f}  {zmax:.4f}  zlo zhi\n")
            f.write("}\n")

        return output

    def _write_run_script(self, result: PlacementResult) -> None:
        """
        Write a ready-to-run in.run for film-on-substrate systems.

        The substrate is frozen (fix setforce) and the film runs NVT at
        300 K with a film-only temperature compute: the default
        whole-system temperature would be biased low by the frozen slab.
        The slab molecule(s) are the first records (placed before the
        film), so molecule IDs 1..N_slab are the substrate.
        """
        n_slab = sum(
            1 for r in result.records
            if r.instance_name.startswith("substrate")
        )
        n_slab = max(n_slab, 1)
        slab_ids = " ".join(str(i) for i in range(1, n_slab + 1))
        script = f"""# AutoPoly film-on-substrate run script (generated)
# Frozen substrate + film NVT 300 K.  Edit as needed for production runs.
boundary        p p p

include         system.in.init
read_data       system.data
include         system.in.settings

# substrate = molecule(s) {slab_ids} (placed first in the data file)
group           slab molecule {slab_ids}
group           film subtract all slab

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes

# freeze the substrate (built slabs carry no force-bearing bonded terms)
fix             freeze slab setforce 0.0 0.0 0.0

minimize        1.0e-4 1.0e-6 1000 10000

velocity        film create 300.0 12345 dist gaussian
# film-only temperature: the frozen slab has zero kinetic energy and
# would bias the default whole-system temperature in the log
compute         tfilm film temp
fix             nvt film nvt temp 300.0 300.0 100.0
fix_modify      nvt temp tfilm

timestep        1.0
thermo          500
thermo_style    custom step c_tfilm pe ke etotal press
thermo_modify   temp tfilm

dump            d1 all custom 2000 dump.film.lammpstrj id mol type x y z

run             10000
"""
        out = self.project_dir / "in.run"
        out.write_text(script)
        logger.info(f"Run script written to {out}")

    # ------------------------------------------------------------------
    # Step 6: inorganic components (fillers): cross terms, groups, run
    # ------------------------------------------------------------------
    def _filler_policies(self, units: UnitLibrary):
        from ..materials.forcefields.compatibility import InorganicClassPolicy
        from ..materials.forcefields import get_inorganic_ff

        policies = {}
        for u in units.filler_units:
            mat = u.material or {}
            adapter = get_inorganic_ff(mat["inorganic_ff"], mat["id"])
            qual = mat.get("qualification") or {}
            # atom types live in the (core) particle class namespace; a
            # grafted composite GNP_x only instantiates NP_x + chains
            type_ns = mat.get("core_class") or u.class_name
            policies[type_ns] = InorganicClassPolicy(
                class_name=type_ns,
                adapter=adapter.name,
                native_rule=adapter.combining_rule,
                organic_rule=mat.get("cross_rule", adapter.combining_rule),
                qualification=qual.get("id") or qual.get("status"),
            )
        return policies

    def _write_cross_terms(self, units: UnitLibrary):
        """Explicit organic-inorganic pair_coeff lines (system.in.settings)."""
        from ..materials.forcefields.compatibility import (
            write_explicit_cross_terms,
        )
        host = self.host_styles(units.force_field)
        report = write_explicit_cross_terms(
            self.moltemplate_dir, self._filler_policies(units), host)
        logger.info(
            f"Wrote {report.n_pairs} explicit organic-inorganic cross "
            "pair coefficients to system.in.settings"
        )
        return report

    def _write_component_groups(self, units: UnitLibrary,
                                result: PlacementResult) -> dict:
        """
        system.in.groups: one group per filler instance (by molecule ID
        from the moltemplate assignments), per filler species, per motion,
        plus "fillers" and "matrix".  Never assumes instance order.
        """
        from ..materials.forcefields.compatibility import (
            read_instance_assignments,
        )
        mols, _ = read_instance_assignments(self.moltemplate_dir)
        by_unit = {u.id: u for u in units.filler_units}
        instances = {}
        for rec in result.records:
            if rec.unit_id in by_unit:
                # grafted fillers: the rigid body is the core molecule only
                key = rec.instance_name
                if key not in mols and f"{key}/core" in mols:
                    key = f"{key}/core"
                if key not in mols:
                    raise WorkflowError(
                        f"No molecule ID found for filler instance "
                        f"{rec.instance_name} in ttree_assignments.txt"
                    )
                instances.setdefault(rec.unit_id, []).append(
                    (rec.instance_name, mols[key]))
        lines = [
            "# AutoPoly component groups (generated from moltemplate's",
            "# output_ttree/ttree_assignments.txt: stable instance identities)",
        ]
        motions = {"rigid": [], "frozen": []}
        all_ids = []
        for uid, inst in instances.items():
            ids = [m for _, m in inst]
            all_ids += ids
            motions.setdefault(by_unit[uid].motion, []).extend(ids)
            for name, m in inst:
                lines.append(f"group {name} molecule {m}")
            lines.append(f"group filler_{uid} molecule "
                         + " ".join(str(m) for m in ids))
        lines.append("group fillers molecule "
                     + " ".join(str(m) for m in sorted(all_ids)))
        for motion, ids in motions.items():
            if ids:
                lines.append(f"group {motion}_fillers molecule "
                             + " ".join(str(m) for m in sorted(ids)))
        lines.append("group matrix subtract all fillers")
        out = self.project_dir / "system.in.groups"
        out.write_text("\n".join(lines) + "\n")
        logger.info(f"Component groups written to {out}")
        return {"instances": instances,
                "motions": {k: v for k, v in motions.items() if v}}

    def _write_nanocomposite_run_script(self, units: UnitLibrary,
                                        result: PlacementResult,
                                        groups: dict) -> None:
        """Minimal run template: relax matrix, rigid/frozen fillers, NVT."""
        motions = groups["motions"]
        if "flexible" in motions:
            raise WorkflowError(
                "Flexible fillers need a validated bonded inorganic model; "
                "no run template is generated"
            )
        species = "\n".join(
            f"#   {u.id}: {u.count} x {u.material['id']} "
            f"({u.material.get('phase')}, {u.geometry_type}), "
            f"{u.material['inorganic_ff']} v{u.material['ff_version']}, "
            f"motion '{u.motion}', qualification "
            f"{(u.material.get('qualification') or {}).get('id') or 'user'}"
            for u in units.filler_units
        )
        cfg = self.packing
        target = getattr(cfg, "target_polymer_density", None)
        dyn = []
        if "rigid" in motions:
            # rigid/nvt (not rigid/nvt/small): rigid/nvt/small together
            # with "neigh_modify exclude molecule/intra" aborts during setup
            # in LAMMPS 29Aug2024; particle counts are modest anyway
            dyn.append("fix             rigid rigid_fillers rigid/nvt "
                       "molecule temp 300.0 300.0 100.0")
        if "frozen" in motions:
            dyn.append("fix             freeze frozen_fillers setforce "
                       "0.0 0.0 0.0")
        dyn = "\n".join(dyn)
        has_matrix = bool(units.organic_units)
        charged = self._system_has_charges()
        kspace_fix = ("" if charged else
                      "# no charged atoms: PPPM needs an explicit g_ewald\n"
                      "kspace_modify   gewald 0.25\n")
        if has_matrix:
            matrix_dyn = (
                "velocity        matrix create 300.0 4928459 dist gaussian\n"
                f"{dyn}\n"
                "fix             nvt matrix nvt temp 300.0 300.0 100.0\n"
                "compute         tmatrix matrix temp\n"
                "thermo_style    custom step temp c_tmatrix pe ke etotal "
                "press density")
        else:
            matrix_dyn = (
                f"{dyn}\n"
                "thermo_style    custom step temp pe ke etotal press")
        compress = ""
        if target is not None:
            compress = f"""
# --- 3. (optional) compression toward the target matrix density
#        {target} g/cm^3: one barostat controls the box and dilates all
#        atoms (rigid bodies by their centers of mass). Uncomment to use.
# unfix           nvt
# unfix           rigid
# fix             rigid rigid_fillers rigid/npt molecule temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0 dilate all
# fix             nvt matrix nvt temp 300.0 300.0 100.0
# run             200000
"""
        script = f"""# AutoPoly nanocomposite run script (generated)
# Fillers:
{species}
# Matrix force field: {units.force_field}. Explicit organic-inorganic
# pair coefficients are appended to system.in.settings; component groups
# (per particle instance / species / motion) are in system.in.groups.
# Edit as needed; this template is a starting point, not an equilibration
# protocol.
boundary        p p p

include         system.in.init
read_data       system.data
include         system.in.settings
include         system.in.groups
{kspace_fix}
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
# rigid/frozen particles: skip intra-particle pair forces
neigh_modify    exclude molecule/intra fillers

# --- 1. relax the matrix around the (temporarily held) fillers
fix             hold fillers setforce 0.0 0.0 0.0
minimize        1.0e-4 1.0e-6 2000 20000
unfix           hold

# --- 2. dynamics: one rigid body per particle instance (or frozen
#        particles), matrix NVT; matrix-only temperature for the log
{matrix_dyn}
thermo          500
timestep        1.0
dump            d1 all custom 2000 dump.nanocomposite.lammpstrj id mol type x y z
run             10000
{compress}"""
        out = self.project_dir / "in.run"
        out.write_text(script)
        logger.info(f"Run script written to {out}")

    def _system_has_charges(self) -> bool:
        """True if system.data has any nonzero atomic charge."""
        data = self.project_dir / "system.data"
        if not data.is_file():
            data = self.moltemplate_dir / "output" / "system.data"
        if not data.is_file():
            return True
        in_atoms = False
        for line in data.read_text().splitlines():
            ls = line.split("#", 1)[0].strip()
            if ls.startswith("Atoms"):
                in_atoms = True
                continue
            if in_atoms:
                if not ls:
                    continue
                parts = ls.split()
                if not parts[0].isdigit():
                    break
                if len(parts) >= 4 and abs(float(parts[3])) > 0.0:
                    return True
        return False

    def _write_ff_report(self, units: UnitLibrary, result: PlacementResult,
                         cross) -> None:
        """ff_compatibility.json / .md for builds with inorganic units."""
        import json

        host = self.host_styles(units.force_field)
        comps = []
        for u in units.inorganic_units:
            mat = dict(u.material or {})
            comps.append({
                "id": u.id,
                "role": u.role,
                "geometry_type": u.geometry_type,
                "motion": u.motion,
                "count": u.count,
                "class": u.class_name,
                "material": mat.get("id"),
                "phase": mat.get("phase"),
                "surface": mat.get("surface"),
                "inorganic_ff": mat.get("inorganic_ff"),
                "ff_version": mat.get("ff_version"),
                "ff_citations": mat.get("ff_citations", []),
                "combining_rule": mat.get("combining_rule"),
                "cross_rule": mat.get("cross_rule", "host (pair_modify mix)"),
                "qualification": mat.get("qualification",
                                         {"status": "legacy substrate "
                                                    "(not blocked)"}),
                "net_charge": mat.get("net_charge"),
                "mass_amu": u.mass,
                "envelope": u.envelope,
            })
        report = {
            "host_force_field": host.to_dict(),
            "components": comps,
            "explicit_cross_terms": cross.to_dict() if cross else None,
            "placement": result.report,
            "strategy": self.strategy_name,
        }
        out = self.project_dir / "ff_compatibility.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        md = ["# AutoPoly force-field compatibility report", "",
              f"Host (matrix/film) force field: `{units.force_field}` "
              f"(`{host.pair_style}`, `{host.pair_modify or 'mix default'}`, "
              f"`{host.special_bonds}`)", "",
              "| component | role | material | phase | surface | inorganic FF "
              "| motion | qualification | cross rule | net charge |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in comps:
            q = c["qualification"] or {}
            md.append(
                f"| {c['id']} x{c['count']} | {c['role']} | {c['material']} "
                f"| {c['phase']} | {c['surface']} | {c['inorganic_ff']} "
                f"v{c['ff_version']} | {c['motion']} | "
                f"{q.get('id') or q.get('status')} | {c['cross_rule']} | "
                f"{c['net_charge']} |")
        if cross:
            md += ["", f"Explicit organic-inorganic pair coefficients: "
                       f"{cross.n_pairs} (appended to system.in.settings)."]
            for cls, rule in cross.rules.items():
                md.append(f"- `{cls}`: {rule}")
        if result.report:
            md += ["", "Placement checks (minimum image, A): "
                   + json.dumps(result.report.get("min_distance")),
                   "", "Loading: " + json.dumps(result.report.get("loading"),
                                               default=str)]
        (self.project_dir / "ff_compatibility.md").write_text(
            "\n".join(md) + "\n")
        logger.info(f"Force-field compatibility report written to {out}")

    # ------------------------------------------------------------------
    # Step 4a: in.init style patching (zero-count hybrid styles)
    # ------------------------------------------------------------------
    _STYLE_KINDS = (
        ("bond", "bond_style"),
        ("angle", "angle_style"),
        ("dihedral", "dihedral_style"),
        ("improper", "improper_style"),
    )

    def _patch_in_init_styles(self) -> None:
        """
        Replace hybrid bond/angle/dihedral/improper styles with their
        'none' form when system.data contains zero records of that kind.

        LAMMPS aborts when a hybrid sub-style is unused (e.g.
        'Improper hybrid sub-style cvff is not used'), which happens for
        any improper-free system (e.g. polyethylene films) because the
        force-field .lt always declares hybrid styles.
        """
        data_path = self.moltemplate_dir / "system.data"
        init_path = self.moltemplate_dir / "system.in.init"
        if not data_path.is_file() or not init_path.is_file():
            return
        counts = {}
        for line in data_path.read_text().splitlines()[:60]:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                counts[parts[1]] = int(parts[0])
        lines = init_path.read_text().splitlines()
        changed = False
        for k, line in enumerate(lines):
            stripped = line.strip()
            for kind, style_cmd in self._STYLE_KINDS:
                if (stripped.startswith(style_cmd)
                        and "hybrid" in stripped
                        and counts.get(kind + "s", 1) == 0):
                    logger.info(
                        f"system.data has 0 {kind}s: patching "
                        f"'{stripped}' -> '{style_cmd} none'"
                    )
                    lines[k] = f"{style_cmd}  none"
                    changed = True
        if changed:
            init_path.write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Step 4: moltemplate + validation + post-processing
    # ------------------------------------------------------------------
    def _run_moltemplate_and_validate(self, force_field: str) -> None:
        """Run moltemplate and validate the output."""
        logger.info("Running moltemplate")
        self.invoke_moltemplate()
        self._validate_moltemplate_output(force_field)
        self._check_required_files()
        self._log_gaff_charges_warning(force_field)

    def invoke_moltemplate(self) -> None:
        """
        Invoke moltemplate on system.lt (with -nocheck).

        Raises:
            WorkflowError: If system.lt is missing or moltemplate fails.
        """
        try:
            system_lt = self.moltemplate_dir / "system.lt"
            if not system_lt.exists():
                raise WorkflowError(
                    f"system.lt not found in {self.moltemplate_dir}"
                )

            moltemplate_sh = Path(self.path_moltemplatesrc) / "moltemplate.sh"
            process = subprocess.run(
                ["bash", str(moltemplate_sh), "-nocheck", "./system.lt"],
                cwd=self.moltemplate_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if process.returncode != 0:
                logger.error("Moltemplate execution failed with the following error:")
                logger.error(process.stderr)
                stdout_lines = process.stdout.splitlines()
                if stdout_lines:
                    logger.error("Last output lines:")
                    for line in stdout_lines[-5:]:
                        logger.error(line)
                raise WorkflowError("Moltemplate execution failed")

        except Exception as e:
            raise WorkflowError(f"Error running Moltemplate: {str(e)}") from e

    def _validate_moltemplate_output(self, force_field: str = "") -> None:
        """Validate that moltemplate created the required output files."""
        output_ttree_dir = self.moltemplate_dir / "output_ttree"
        if not output_ttree_dir.exists():
            raise WorkflowError(
                f"Moltemplate failed to create output_ttree directory at "
                f"{output_ttree_dir}"
            )

        # openff builds write typed "Data Bonds" sections (never the untyped
        # "Data Bond List"), so the required bond section differs by FF.
        # Inorganic classes always write typed "Data Bonds" (metal clusters
        # none at all): inorganic-only systems need no bond section.
        bond_section = "Data Bonds" if force_field == "openff" else "Data Bond List"
        required_data_files = ["Data Atoms"]
        if not getattr(self, "_inorganic_only", False):
            required_data_files.append(bond_section)
        missing_data_files = []

        for data_file in required_data_files:
            data_file_path = output_ttree_dir / data_file
            if not data_file_path.exists():
                missing_data_files.append(data_file)
            elif data_file_path.stat().st_size == 0:
                logger.error(f"Data file exists but is empty: {data_file}")
                missing_data_files.append(f"{data_file} (empty)")

        if missing_data_files:
            logger.error(
                f"Moltemplate failed to generate required data files: "
                f"{missing_data_files}"
            )
            raise WorkflowError(
                f"Critical data files missing in {output_ttree_dir}. "
                "Ensure write(\"Data Atoms\") sections are present in monomer files"
            )

    def _check_required_files(self) -> None:
        """Check that required LAMMPS input files were generated."""
        required_files = ["system.in.settings", "system.data"]
        missing_files = [
            f for f in required_files
            if not (self.moltemplate_dir / f).exists()
        ]

        if missing_files:
            logger.error(
                f"Moltemplate failed to generate required files: "
                f"{', '.join(missing_files)}"
            )
            raise WorkflowError(
                "Required LAMMPS input files not generated. "
                "Check that polymer .lt files and system.lt are properly formatted"
            )

    def _log_gaff_charges_warning(self, force_field: str) -> None:
        """Log the legacy warning about missing charges for GAFF."""
        if force_field == "gaff":
            charges_file = self.moltemplate_dir / "system.in.charges"
            if not charges_file.exists():
                logger.warning("Note: system.in.charges not generated for GAFF")
                logger.warning(
                    "GAFF requires manual charge calculation using AM1-BCC or RESP"
                )
                logger.warning("All atomic charges are currently set to 0.00")
