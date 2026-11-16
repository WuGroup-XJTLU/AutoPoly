#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coarse-grained fillers for the bead-spring backend (P3 CG fillers):
particle geometry, obstacle-aware periodic SAW, data-file / moltemplate
output, run-script commands and (with LAMMPS) a smoke run of both
backends.
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from AutoPoly import System
from AutoPoly.mc.collision import CollisionDetector, ObstacleCollisionDetector
from AutoPoly.models import architectures as arch
from AutoPoly.models.bead_spring import BeadType, SAWConfig, saw_generate_graphs
from AutoPoly.models.bead_spring_system import BeadSpringSystem
from AutoPoly.nanoparticles import CGFiller
from AutoPoly.nanoparticles.cg import fcc_sphere, motion_groups, sphere_shell
from AutoPoly.pipeline.smoke import lammps_executable, run_smoke


def _system(tmp_path, fillers, density=0.3, n_chains=30, n_beads=20,
            backend="direct", bead_types=None, **kw):
    np.random.seed(1)
    bss = BeadSpringSystem(
        "nc", System(out=str(tmp_path / "out")),
        bead_types=bead_types or [BeadType("A"), BeadType("F"), BeadType("G")],
        bond_style="fene", pair_style="wca", density=density,
        backend=backend, **kw)
    bss.add_species(arch.linear([("A", n_beads)]), n_chains=n_chains)
    for f in fillers:
        bss.add_filler(f)
    return bss


def _read_data(path):
    text = path.read_text()
    lo, hi = (float(v) for v in
              next(l for l in text.splitlines() if "xlo" in l).split()[:2])
    atoms_block = text.split("Atoms")[1].split("\n\n", 1)[1]
    atoms_block = atoms_block.split("\nBonds")[0].strip()
    rows = np.array([[float(v) for v in l.split()]
                     for l in atoms_block.splitlines()])
    bonds = set()
    for l in text.split("Bonds\n\n")[1].strip().splitlines():
        if not l.strip():
            break
        _, _, i, j = (int(v) for v in l.split()[:4])
        bonds.add((min(i, j) - 1, max(i, j) - 1))
    return hi - lo, rows, bonds


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------
def test_fcc_sphere_geometry():
    pos = fcc_sphere(4.0, 1.0)
    assert np.linalg.norm(pos, axis=1).max() <= 4.0 + 1e-9
    d, _ = cKDTree(pos).query(pos, k=2)
    assert np.allclose(d[:, 1], 1.0)
    expected = np.sqrt(2.0) * 4.0 / 3.0 * np.pi * 4.0 ** 3
    assert abs(len(pos) - expected) / expected < 0.2
    assert np.allclose(pos.mean(axis=0), 0.0, atol=1e-9)


def test_shell_geometry():
    pos = sphere_shell(5.0, 1.0)
    assert np.allclose(np.linalg.norm(pos, axis=1), 5.0)
    d, _ = cKDTree(pos).query(pos, k=2)
    assert d[:, 1].min() > 0.7 and np.median(d[:, 1]) < 1.3


def test_cgfiller_validation():
    with pytest.raises(ValueError):
        CGFiller(radius=0.0)
    with pytest.raises(ValueError):
        CGFiller(radius=2.0, structure="cube")
    with pytest.raises(ValueError, match="flexible"):
        CGFiller(radius=2.0, motion="flexible")
    with pytest.raises(ValueError, match="single"):
        CGFiller(radius=0.5, spacing=1.0)
    assert np.allclose(np.linalg.norm(sphere_shell(0.3, 1.0), axis=1), 0.3)
    with pytest.raises(ValueError, match="motions"):
        motion_groups([CGFiller(radius=2.0, motion="rigid"),
                       CGFiller(radius=2.0, motion="frozen")])


# ----------------------------------------------------------------------
# obstacle-aware periodic SAW
# ----------------------------------------------------------------------
def test_obstacle_detector_periodic():
    L = 10.0
    det = ObstacleCollisionDetector(((-5, 5),) * 3, 2.0, [[4.5, 0, 0]], [2.0],
                                    period=L)
    # across the boundary: minimum image distance 1.5 < 2.0 + 0.5 - 0.1
    assert det.check_collision(np.array([-4.0, 0.0, 0.0]), 0.5)
    assert not det.check_collision(np.array([0.0, 0.0, 0.0]), 0.5)
    det.add_monomer(0, np.array([-4.9, 2.0, 2.0]), 0.5)
    # a bead outside the box overlaps the periodic image of bead 0
    assert det.check_collision(np.array([5.3, 2.0, 2.0]), 0.5)
    det.remove_monomer(0)
    assert not det.check_collision(np.array([5.3, 2.0, 2.0]), 0.5)
    # open detector behaves like the base class
    base = CollisionDetector(((-5, 5),) * 3, 2.0)
    base.add_monomer(0, np.array([-4.9, 2.0, 2.0]), 0.5)
    assert not base.check_collision(np.array([5.3, 2.0, 2.0]), 0.5)


def test_saw_without_obstacles_unchanged():
    archs = [arch.linear([("A", 15)])] * 5
    np.random.seed(7)
    a, _, sa = saw_generate_graphs(archs, 0.97, 12.0, SAWConfig())
    np.random.seed(7)
    b, _, sb = saw_generate_graphs(archs, 0.97, 12.0, SAWConfig(),
                                   obstacles=None)
    assert sa["success"] and sb["success"]
    assert np.array_equal(np.array(a), np.array(b))


# ----------------------------------------------------------------------
# direct backend
# ----------------------------------------------------------------------
def test_direct_backend_with_fillers(tmp_path):
    bss = _system(tmp_path, [CGFiller(radius=3.0, count=2, bead_type="F")])
    bss.set_pair_coeff("A", "F", 1.5, cutoff=2.5)
    bss.generate()
    project = tmp_path / "out" / "nc"
    L, rows, bonds = _read_data(project / "polymer.data")
    assert abs(L - bss._calculate_box_size()) < 1e-5
    info = bss.get_system_info()["fillers"]
    n_f = info["n_beads"]
    assert info["n_particles"] == 2 and len(rows) == 600 + n_f
    mol, typ, x = rows[:, 1], rows[:, 2], rows[:, 3:]
    assert sorted(set(mol[typ == 2])) == [31.0, 32.0]
    # no polymer bead inside a filler envelope; no cross-boundary clashes
    tree = cKDTree(np.mod(x + L / 2, L), boxsize=L)
    for i, j in tree.query_pairs(0.85):
        same_filler = typ[i] == typ[j] == 2 and mol[i] == mol[j]
        assert (min(i, j), max(i, j)) in bonds or same_filler
    for p in bss._placed_fillers:
        d = x[typ == 1] - p.center
        d -= L * np.round(d / L)
        assert np.linalg.norm(d, axis=1).min() > p.envelope + 0.5 - 0.1 - 1e-3
    script = (project / "in.polymer").read_text()
    for needle in ("group fillers type 2", "group matrix subtract all fillers",
                   "neigh_modify    exclude molecule/intra fillers",
                   "fix             hold fillers setforce 0.0 0.0 0.0",
                   "unfix           hold",
                   "fix             1 matrix nvt",
                   "rigid_fillers rigid/nvt molecule",
                   "pair_coeff      1 2 1.5000 1.0000 2.50000",
                   "c_tmatrix"):
        assert needle in script, needle
    assert script.index("fix             hold") < script.index("minimize")
    assert "fix             1 all nvt" not in script


def test_frozen_shell_filler_script(tmp_path):
    bss = _system(tmp_path, [CGFiller(radius=3.0, count=1, bead_type="G",
                                      structure="shell", motion="frozen")])
    bss.generate()
    script = (tmp_path / "out" / "nc" / "in.polymer").read_text()
    assert "group frozen_fillers type 3" in script
    assert "velocity        frozen_fillers set 0.0 0.0 0.0" in script
    assert "fix             3 frozen_fillers setforce" in script
    assert "rigid/nvt" not in script.split("# (optional)")[0]


def test_no_fillers_output_unchanged(tmp_path):
    bss = _system(tmp_path, [], bead_types=[BeadType("A")])
    bss.generate(backend="direct")
    project = tmp_path / "out" / "nc"
    script = (project / "in.polymer").read_text()
    assert "fix             1 all nvt temp 1.0 1.0 0.1 tchain 3" in script
    assert "fillers" not in script and "c_tmatrix" not in script
    box = next(l for l in (project / "polymer.data").read_text().splitlines()
               if "xlo" in l)
    assert len(box.split()[0].split(".")[1]) == 1     # legacy .1f bounds


def test_filler_errors(tmp_path):
    with pytest.raises(ValueError, match="not among"):
        _system(tmp_path, [CGFiller(radius=2.0, bead_type="X")])
    bss = _system(tmp_path, [CGFiller(radius=2.0, bead_type="A")])
    with pytest.raises(ValueError, match="own bead type"):
        bss.saw_generate()
    bss = _system(tmp_path, [CGFiller(radius=2.0, bead_type="F")],
                  generation_method="mc")
    with pytest.raises(ValueError):
        bss.generate_data_file()
    bss = _system(tmp_path, [CGFiller(radius=8.0, bead_type="F")],
                  box_size=12.0)
    with pytest.raises(ValueError, match="does not fit"):
        bss.saw_generate()


# ----------------------------------------------------------------------
# moltemplate backend
# ----------------------------------------------------------------------
@pytest.mark.integration
def test_moltemplate_backend_with_fillers(tmp_path):
    bss = _system(tmp_path, [CGFiller(radius=3.0, count=2, bead_type="F")],
                  backend="moltemplate")
    bss.generate()
    mtd = tmp_path / "out" / "nc" / "moltemplate"
    assert (mtd / "fillers.lt").is_file()
    settings = (mtd / "system.in.settings").read_text()
    assert "group fillers type 2" in settings
    assert "group rigid_fillers type 2" in settings
    script = (mtd / "in.polymer").read_text()
    assert script.index("include         system.in.init") < \
        script.index("read_data       system.data")
    assert "rigid_fillers rigid/nvt molecule" in script
    L, rows, _ = _read_data(mtd / "system.data")
    n_f = bss.get_system_info()["fillers"]["n_beads"]
    assert len(rows) == 600 + n_f
    assert len(set(rows[rows[:, 2] == 2][:, 1])) == 2


# ----------------------------------------------------------------------
# LAMMPS smoke
# ----------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("backend,motion", [("direct", "rigid"),
                                            ("direct", "frozen"),
                                            ("moltemplate", "rigid")])
def test_cg_filler_lammps_smoke(tmp_path, backend, motion):
    if lammps_executable() is None:
        pytest.skip("LAMMPS not available")
    bss = _system(tmp_path, [CGFiller(radius=3.0, count=2, bead_type="F",
                                      motion=motion)], backend=backend)
    bss.generate()
    project = tmp_path / "out" / "nc"
    if backend == "moltemplate":
        project = project / "moltemplate"
    smoke = run_smoke(project, steps=2000, script="in.polymer",
                      expected_rigid_bodies=2 if motion == "rigid" else None,
                      max_temperature=5.0)
    assert smoke.ok, smoke.errors
    assert smoke.thermo[0]["PotEng"] < 100.0       # clean start, no clashes
    assert 0.3 < smoke.thermo[-1]["c_tmatrix"] < 1.5   # heating from rest


@pytest.mark.integration
@pytest.mark.slow
def test_moltemplate_backend_runs_without_fillers(tmp_path):
    """Regression: in.polymer of the moltemplate backend must include
    system.in.init (atom_style full) before read_data."""
    if lammps_executable() is None:
        pytest.skip("LAMMPS not available")
    bss = _system(tmp_path, [], bead_types=[BeadType("A")],
                  backend="moltemplate")
    bss.generate()
    smoke = run_smoke(tmp_path / "out" / "nc" / "moltemplate", steps=500,
                      script="in.polymer", max_temperature=5.0)
    assert smoke.ok, smoke.errors
