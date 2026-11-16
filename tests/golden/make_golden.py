#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record golden reference artifacts for the built-in silica substrates.

P0-R1 baseline of the inorganic-framework refactoring: run this script
against a *known-good* AutoPoly to (re)record

- the self-contained slab .lt files written by every built-in slab builder
  configuration in SLAB_CASES (plus the SurfaceSlab attributes), and
- the LAMMPS files of two complete film-on-substrate pipelines in
  PIPELINE_CASES (system.data, system.in.settings, system.in.init, in.run),
  plus their LAMMPS ``run 0`` energies when a LAMMPS binary is available.

tests/test_legacy_substrates_golden.py compares the current code against
these files for physical equivalence (coordinates, topology, types,
charges, pair/bond coefficients, energies).  Re-record only for an
intentional, documented output change:

    python tests/golden/make_golden.py            # slabs + pipelines
    python tests/golden/make_golden.py --slabs    # slab .lt files only

Recorded on 2026-11-13 from v2.0 @ 22c79cb (before the materials
refactoring).
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent.parent))

#: name -> (builder registry name, builder kwargs)
SLAB_CASES = {
    "quartz_interface": ("alpha_quartz", dict(
        lx_target=30.0, ly_target=34.0, thickness=12.0)),
    "quartz_clayff": ("alpha_quartz", dict(
        lx_target=30.0, ly_target=34.0, thickness=12.0, slab_ff="clayff")),
    "quartz_hybrid_styles_bare_bottom": ("alpha_quartz", dict(
        lx_target=25.0, ly_target=26.0, thickness=14.0,
        pair_substyle="lj/charmm/coul/long", bond_substyle="harmonic",
        hydroxylate_bottom=False)),
    "quartz_type_override": ("alpha_quartz", dict(
        lx_target=20.0, ly_target=17.0, thickness=12.0,
        slab_types={"Si": "my_si"}, slab_lj={"OB": (0.06, 3.1)})),
    "cristobalite_interface": ("beta_cristobalite", dict(
        lx_target=40.0, ly_target=53.0, thickness=12.0)),
    "cristobalite_bridged": ("beta_cristobalite", dict(
        lx_target=40.0, ly_target=53.0, thickness=14.0,
        oh_density=2.0, seed=7)),
    "cristobalite_custom": ("beta_cristobalite", dict(
        lx_target=21.0, ly_target=18.0, thickness=13.0, slab_ff="custom",
        slab_types={"Si": "xs", "OB": "xob", "OH": "xoh", "HO": "xho"},
        slab_charges={"Si": 1.2, "OB": -0.6, "OH": -0.7, "HO": 0.4},
        slab_lj={"Si": (0.1, 3.7), "OB": (0.05, 3.1),
                 "OH": (0.12, 3.1), "HO": (0.0, 0.0)})),
}

PE_SEQUENCE = ["CC[*]", "[*]CC[*]", "[*]CC"]

#: name -> generate() configuration
PIPELINE_CASES = {
    "pe_on_quartz_gaff": dict(
        builder="alpha_quartz", slab_ff="interface", force_field="gaff",
        box_dims=(30.0, 34.0, 40.0), rng_seed=3, thickness=12.0),
    "pe_on_cristobalite_oplsaa_clayff": dict(
        builder="beta_cristobalite", slab_ff="clayff", force_field="oplsaa",
        box_dims=(40.0, 53.0, 40.0), rng_seed=5, thickness=12.0),
}

PIPELINE_FILES = ("system.data", "system.in.settings", "system.in.init",
                  "in.run")

#: LAMMPS input evaluating the energy of the as-built system (no motion).
LAMMPS_RUN0 = """\
boundary p p p
include system.in.init
read_data system.data
include system.in.settings
thermo_style custom step pe evdwl ecoul elong ebond eangle edihed eimp
thermo_modify format float %20.10g
run 0
"""

THERMO_KEYS = ("pe", "evdwl", "ecoul", "elong", "ebond", "eangle",
               "edihed", "eimp")


def lammps_binary():
    """First LAMMPS executable found on PATH, or None."""
    for name in ("lmp", "lmp_serial", "lmp_mpi"):
        path = shutil.which(name)
        if path:
            return path
    return None


def run_lammps_energies(project_dir: Path, lmp: str) -> dict:
    """Run LAMMPS ``run 0`` in project_dir; return the thermo energies."""
    (project_dir / "in.golden_run0").write_text(LAMMPS_RUN0)
    proc = subprocess.run(
        [lmp, "-in", "in.golden_run0", "-log", "log.golden_run0",
         "-screen", "none"],
        cwd=project_dir, capture_output=True, text=True,
    )
    log = (project_dir / "log.golden_run0").read_text() \
        if (project_dir / "log.golden_run0").is_file() else ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed in {project_dir}:\n{proc.stdout}\n{proc.stderr}"
            f"\n{log[-2000:]}"
        )
    lines = log.splitlines()
    for k, line in enumerate(lines):
        parts = line.split()
        if parts[:2] == ["Step", "PotEng"] or (
                parts and parts[0] == "Step" and "PotEng" in parts):
            values = lines[k + 1].split()
            return {key: float(v) for key, v in zip(THERMO_KEYS, values[1:])}
    raise RuntimeError(f"No thermo output in {project_dir}/log.golden_run0")


def record_slabs():
    from AutoPoly.surfaces import SLAB_BUILDERS

    out = DATA / "slabs"
    out.mkdir(parents=True, exist_ok=True)
    for name, (builder, kwargs) in SLAB_CASES.items():
        slab = SLAB_BUILDERS[builder](**kwargs).build()
        with tempfile.TemporaryDirectory() as tmp:
            path = slab.write_lt(tmp)
            (out / f"{name}.lt").write_text(path.read_text())
        meta = {
            "builder": builder,
            "kwargs": {k: list(v) if isinstance(v, tuple) else v
                       for k, v in kwargs.items()},
            "class_name": slab.class_name,
            "lt_filename": slab.lt_filename,
            "lx": slab.lx,
            "ly": slab.ly,
            "thickness": slab.thickness,
            "z_extent": slab.z_extent,
            "n_atoms": len(slab.atoms),
            "n_bonds": len(slab.bonds or []),
            "n_silanol_top": slab.n_silanol_top,
            "n_silanol_bottom": slab.n_silanol_bottom,
            "slab_ff": slab.slab_ff,
            "net_charge": float(sum(a.charge for a in slab.atoms)),
            "roles": {r: sum(a.role == r for a in slab.atoms)
                      for r in ("Si", "OB", "OH", "HO")},
        }
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"slab {name}: {len(slab.atoms)} atoms")


def run_pipeline(case: dict, out_root: Path, name: str) -> Path:
    """Run one golden film-on-substrate pipeline; return the project dir."""
    from AutoPoly import Polymer, System, generate
    from AutoPoly.packing import SubstrateSpec
    from AutoPoly.pipeline.geometry import GeometryConfig

    system = System(out=str(out_root))
    film = Polymer(chain_num=2, sequence=PE_SEQUENCE,
                   tacticity="syndiotactic")
    spec = SubstrateSpec(builder=case["builder"],
                         thickness=case["thickness"], gap=3.0,
                         slab_ff=case["slab_ff"])
    generate(
        system, name, [film], force_field=case["force_field"],
        substrate=spec, box_dims=case["box_dims"],
        geometry_config=GeometryConfig(use_mc_chain_growth=False),
        rng_seed=case["rng_seed"],
    )
    return out_root / name


def record_pipelines():
    lmp = lammps_binary()
    for name, case in PIPELINE_CASES.items():
        out = DATA / "pipelines" / name
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            project = run_pipeline(case, Path(tmp) / "out", name)
            for fname in PIPELINE_FILES:
                shutil.copy2(project / fname, out / fname)
            energies = None
            if lmp is not None:
                energies = run_lammps_energies(project, lmp)
        import rdkit
        (out / "energies.json").write_text(
            json.dumps({"lammps": lmp is not None, "rdkit": rdkit.__version__,
                        "run0": energies}, indent=2) + "\n")
        print(f"pipeline {name}: energies {energies}")


def main():
    parser = argparse.ArgumentParser(
        description="Record golden reference artifacts for the built-in "
                    "silica substrates.")
    parser.add_argument("--slabs", action="store_true",
                        help="record the slab .lt goldens only")
    args = parser.parse_args()
    record_slabs()
    if not args.slabs:
        record_pipelines()


if __name__ == "__main__":
    main()
