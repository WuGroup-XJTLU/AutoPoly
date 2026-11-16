#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nanocomposite_mc packing: fillers first, exact counts, PBC clearances,
loading/box sizing, reproducibility, regrowth, and no silent fallback.

These tests stop before moltemplate (run_moltemplate=False) unless noted.
"""
import re

import numpy as np
import pytest

from AutoPoly import Polymer, System, generate
from AutoPoly.core.exceptions import ValidationError, WorkflowError
from AutoPoly.materials.topology import neighbor_pairs
from AutoPoly.nanoparticles import Nanoparticle
from AutoPoly.packing import (
    BoxSpec,
    NanocompositePacking,
    PackingContext,
    PackingError,
    SubstrateSpec,
    get_strategy,
)
from AutoPoly.pipeline.unit_coords import read_xyz, rotation_matrix
from AutoPoly.pipeline.units import UnitLibrary

PE = ["CC[*]"] + ["[*]CC[*]"] * 6 + ["[*]CC"]
CMD = re.compile(r"^(\S+) = new (\S+?)(?:\.rot\(([^)]*)\))?"
                 r"(?:\.move\(([^)]*)\))?$")


def _gen(tmp_path, nps, chains=10, name="nc", box_size=48.0, **packing):
    system = System(out=str(tmp_path / "out"))
    models = []
    if chains:
        models.append(Polymer(chain_num=chains, sequence=PE))
    models += nps
    packing.setdefault("seed", 11)
    res = generate(system, name, models, force_field="oplsaa",
                   packing=NanocompositePacking(**packing),
                   box_size=box_size, run_moltemplate=False, rng_seed=11)
    return tmp_path / "out" / name, res


def _world_atoms(project, res):
    """Atom coordinates of every placed instance from the lt commands."""
    lib = UnitLibrary.load(project / "build" / "oplsaa")
    units = {u.id: u for u in lib.units}
    out = []
    for rec in res.records:
        u = units[rec.unit_id]
        el, xyz = read_xyz(project / "build" / "oplsaa" / u.coords_file)
        m = CMD.match(rec.lt_command)
        assert m, rec.lt_command
        if m.group(2) != u.class_name:      # regrown chain: absolute coords
            continue
        R = np.eye(3)
        if m.group(3):
            a, x, y, z = map(float, m.group(3).split(","))
            R = rotation_matrix(a, (x, y, z))
        t = np.zeros(3) if not m.group(4) else \
            np.array([float(v) for v in m.group(4).split(",")])
        out.append((rec, u, xyz @ R.T + t))
    return out


def test_fillers_first_exact_count_unique_names(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0, count=3, name="silica")
    project, res = _gen(tmp_path, [np_], chains=8)
    names = [r.instance_name for r in res.records]
    assert names[:3] == ["silica_1", "silica_2", "silica_3"]
    assert len(set(names)) == len(names)
    assert sum(r.unit_id == "silica" for r in res.records) == 3
    assert res.report["loading"]["n_fillers"] == {"silica": 3}


def test_filler_gap_through_periodic_images(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0, count=4, name="silica")
    project, res = _gen(tmp_path, [np_], chains=4, min_gap=4.0,
                        box_size=50.0)
    L = np.array([hi - lo for lo, hi in res.box_bounds])
    placed = _world_atoms(project, res)
    fillers = [w for rec, u, w in placed if u.kind == "inorganic"]
    pos = np.vstack(fillers)
    owner = np.concatenate([np.full(len(w), k) for k, w in enumerate(fillers)])
    i, j, _, d = neighbor_pairs(pos - pos.min(0), 4.0, np.diag(L), (True,) * 3)
    assert not np.any(owner[i] != owner[j]), "filler-filler contact < gap"
    assert res.report["pbc_checked"] is True


def test_matrix_clearance_from_fillers(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0, count=1, name="silica")
    project, res = _gen(tmp_path, [np_], chains=10, filler_clearance=2.8,
                        matrix_placement="rigid", max_attempts=4000)
    L = np.array([hi - lo for lo, hi in res.box_bounds])
    placed = _world_atoms(project, res)
    filler = np.vstack([w for _, u, w in placed if u.kind == "inorganic"])
    matrix = [w for _, u, w in placed if u.kind != "inorganic"]
    for w in matrix:
        d = filler[:, None, :] - w[None, :, :]
        d -= L * np.round(d / L)
        assert np.sqrt((d ** 2).sum(-1)).min() >= 2.8 - 1e-4
    assert res.report["min_distance"]["filler_matrix"] >= 2.8


def test_box_from_initial_density_and_loading(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=7.0, count=2, name="silica")
    project, res = _gen(tmp_path, [np_], chains=40, initial_density=0.3,
                        box_size=None)
    assert res.report["box"]["sizing"].startswith("initial_density")
    load = res.report["loading"]
    vol = res.report["box"]["volume_A3"]
    v_matrix = load["matrix_mass_amu"] / 0.3 / 0.602214076
    assert vol == pytest.approx(v_matrix + load["filler_core_volume_A3"],
                                rel=1e-9)
    assert load["nominal_filler_volume_fraction"] == pytest.approx(
        load["filler_core_volume_A3"] / vol)
    w = load["filler_mass_amu"] / (load["filler_mass_amu"]
                                   + load["matrix_mass_amu"])
    assert load["filler_weight_fraction"] == pytest.approx(w)
    assert load["initial_matrix_density_gcm3"] == pytest.approx(0.3)


def test_explicit_box_dims(tmp_path):
    system = System(out=str(tmp_path / "out"))
    res = generate(system, "b", [Polymer(chain_num=4, sequence=PE),
                                 Nanoparticle(material="Au", radius=6.0)],
                   force_field="oplsaa", box_dims=(50.0, 45.0, 40.0),
                   run_moltemplate=False, rng_seed=2)
    L = [hi - lo for lo, hi in res.box_bounds]
    assert L == pytest.approx([50.0, 45.0, 40.0])
    assert res.report["box"]["sizing"] == "explicit"


def test_reproducible_with_seed(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0, count=2)
    _, a = _gen(tmp_path / "a", [np_], chains=6, seed=5)
    _, b = _gen(tmp_path / "b", [np_], chains=6, seed=5)
    _, c = _gen(tmp_path / "c", [np_], chains=6, seed=6)
    cmds = [r.lt_command for r in a.records]
    assert cmds == [r.lt_command for r in b.records]
    assert cmds != [r.lt_command for r in c.records]


def test_no_silent_fallback(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=10.0, count=12)
    system = System(out=str(tmp_path / "out"))
    with pytest.raises(WorkflowError, match="Could not place filler"):
        generate(system, "full", [np_], force_field="oplsaa", box_size=40.0,
                 run_moltemplate=False, rng_seed=1,
                 packing=NanocompositePacking(max_attempts=200))


def test_filler_larger_than_box(tmp_path):
    system = System(out=str(tmp_path / "out"))
    with pytest.raises(WorkflowError, match="cannot fit in the box"):
        generate(system, "big", [Nanoparticle(material="SiO2", radius=14.0)],
                 force_field="oplsaa", box_size=25.0, run_moltemplate=False)


def test_regrown_chains_are_typed_copies(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0, count=1)
    project, res = _gen(tmp_path, [np_], chains=6, matrix_placement="regrow")
    assert len(res.report["regrown_chains"]) == 6
    assert res.report["intrachain"]["overlapping_chains"] == 0
    for fname, text in res.extra_lt_files.items():
        cls = fname[:-3]
        assert f"{cls} inherits OPLSAA {{" in text
        src = (project / "build" / "oplsaa" /
               f"{cls.split('__')[0]}.lt").read_text()
        # identical except class name and monomer transforms
        strip = lambda t: [l for l in t.splitlines() if "monomer[" not in l
                           and "inherits" not in l and "} #" not in l]
        assert strip(text) == strip(src)
    lt = (project / "moltemplate" / "system.lt").read_text()
    for fname in res.extra_lt_files:
        assert f'import "{fname}"' in lt


def test_strategy_selection_and_conflicts(tmp_path):
    system = System(out=str(tmp_path / "out"))
    np_ = Nanoparticle(material="SiO2", radius=8.0)
    poly = Polymer(chain_num=2, sequence=PE)
    with pytest.raises(WorkflowError, match="filled film on a slab"):
        generate(system, "x", [poly, np_], force_field="gaff",
                 substrate=SubstrateSpec(builder="alpha_quartz",
                                         thickness=12.0),
                 box_dims=(30.0, 34.0, 60.0), run_moltemplate=False)
    with pytest.raises(WorkflowError, match="conflict with strategy"):
        generate(system, "y", [poly, np_], force_field="oplsaa",
                 strategy="grid", run_moltemplate=False)


def test_legacy_strategies_reject_inorganic_units(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0)
    project, res = _gen(tmp_path, [np_], chains=2)
    lib = UnitLibrary.load(project / "build" / "oplsaa")
    for name in ("mc_random", "grid"):
        ctx = PackingContext(units=lib, box=BoxSpec(requested_box_size=80.0))
        with pytest.raises(ValidationError, match="cannot place inorganic"):
            get_strategy(name).place(ctx)


def test_packing_config_validation():
    with pytest.raises(ValidationError, match="'raise'"):
        NanocompositePacking(packing_failure="grid")
    with pytest.raises(ValidationError, match="exceeds"):
        NanocompositePacking(initial_density=1.0, target_polymer_density=0.9)
    with pytest.raises(ValidationError, match="matrix_placement"):
        NanocompositePacking(matrix_placement="teleport")
    with pytest.raises(PackingError.__mro__[1]):
        NanocompositePacking(min_gap=0.0)


def test_nearest_filler_matrix_distance_minimum_image():
    """Exact filler-matrix distance reported when no contact lies within
    the verification cutoff; atoms of the filler's own body (grafts) are
    excluded."""
    from AutoPoly.packing.nanocomposite import NanocompositeMCStrategy

    L = np.array([20.0, 20.0, 20.0])
    pos = np.array([[9.5, 0.0, 0.0],      # filler core (body 0)
                    [9.0, 0.0, 0.0],      # own graft atom (body 0)
                    [-4.5, 0.0, 0.0],     # matrix (body 1): 6 A via image
                    [0.0, 0.0, 0.0]])     # matrix (body 1): 9.5 A direct
    owner = np.array([0, 0, 1, 1])
    is_filler = np.array([True, False, False, False])
    d = NanocompositeMCStrategy._nearest(pos, owner, is_filler, L)
    assert d == pytest.approx(6.0)
