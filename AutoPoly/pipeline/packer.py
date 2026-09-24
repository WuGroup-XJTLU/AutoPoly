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
            mv_files(self.moltemplate_dir)
            if self.substrate is not None:
                self._write_run_script(result)

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
            src = build_dir / unit.lt_file
            dst = self.moltemplate_dir / unit.lt_file
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

        # Built-in substrate slab: generate the .lt at pack time
        if self.substrate is not None and self.substrate.is_builder:
            self._build_substrate_slab(units.force_field)

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

    def _film_bond_substyle(self, force_field: str) -> Optional[str]:
        """'harmonic' if the film force field uses a hybrid bond style."""
        if force_field == "openff" and self._ff_lt_for_styles(force_field) is None:
            # openff.lt always declares 'bond_style hybrid harmonic'.
            return "harmonic"
        ff_lt = self._ff_lt_for_styles(force_field)
        for line in ff_lt.read_text().splitlines():
            ls = line.strip()
            if ls.startswith("bond_style") and not ls.startswith("#"):
                return "harmonic" if "hybrid" in ls else None
        return None

    def _film_pair_substyle(self, force_field: str) -> Optional[str]:
        """
        LAMMPS pair sub-style the slab's pair_coeff lines must carry.

        Read from the film force field's .lt pair_style line: with a
        hybrid pair style every pair_coeff needs a sub-style (the LJ
        one); with a plain pair style, None.  Class2 force fields store
        pair coefficients in the data file, which the slab writers do
        not support.
        """
        if force_field == "openff" and self._ff_lt_for_styles(force_field) is None:
            # openff.lt always declares a hybrid pair style with an LJ
            # sub-style (lj/charmm/coul/long or lj/cut/coul/long).
            return "lj/charmm/coul/long"
        ff_lt = self._ff_lt_for_styles(force_field)
        pair_style = ""
        for line in ff_lt.read_text().splitlines():
            ls = line.strip()
            if ls.startswith("pair_style") and not ls.startswith("#"):
                pair_style = ls
                break
        if "class2" in pair_style:
            raise WorkflowError(
                f"Built-in substrate builders do not support class2 "
                f"force fields ({force_field}); build the slab outside "
                "AutoPoly and use SubstrateSpec(lt_file=..., ...) instead"
            )
        if "hybrid" in pair_style:
            for token in pair_style.split()[1:]:
                if token.startswith("lj/"):
                    return token
            raise WorkflowError(
                f"Could not find an LJ sub-style in the pair_style line "
                f"of {ff_lt.name}: '{pair_style}'"
            )
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
        bond_section = "Data Bonds" if force_field == "openff" else "Data Bond List"
        required_data_files = ["Data Atoms", bond_section]
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
