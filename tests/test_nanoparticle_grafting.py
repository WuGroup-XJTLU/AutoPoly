#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Covalent grafting of polymer chains onto silica nanoparticles (P2)."""
import json

import numpy as np
import pytest

from AutoPoly import Polymer, System, generate
from AutoPoly.core.exceptions import ValidationError, WorkflowError
from AutoPoly.materials.chemistry import silica
from AutoPoly.materials.forcefields import get_inorganic_ff
from AutoPoly.nanoparticles import Grafting, Nanoparticle
from AutoPoly.pipeline.unit_coords import read_xyz
from AutoPoly.pipeline.units import UnitLibrary

GRAFT = ["CC[*]"] + ["[*]CC[*]"] * 3 + ["[*]CC"]
PE = ["CC[*]"] + ["[*]CC[*]"] * 6 + ["[*]CC"]


def _particle(**kw):
    if "count" not in kw:
        kw.setdefault("density", 0.5)
    brush = Grafting(polymer=Polymer(chain_num=1, sequence=GRAFT), seed=2,
                     **kw)
    return Nanoparticle(material="SiO2", radius=9.0, grafts=brush,
                        name="gsil")


class TestSites:
    def test_count_spread_and_one_per_silicon(self):
        np_ = _particle()
        plan = np_.graft_plan()
        area = plan["area_nm2"]
        assert plan["n"] == round(0.5 * area)
        s = np_.build_structure()
        nb = s.neighbors()
        si = [next(j for j in nb[o] if s.elements[j] == "Si")
              for o in plan["sites"]]
        assert len(set(si)) == len(si)
        pos = s.positions[plan["sites"]]
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        assert d.min() > 6.0       # farthest-point spread on R=9 A

    def test_anchor_structure_valid(self):
        plan = _particle(count=3).graft_plan()
        core = plan["core"]
        assert core.roles.count("OG") == 3
        silica.validate_silica(core)
        for o in plan["anchors"]:
            ne = [core.elements[j] for j in core.neighbors()[o]]
            assert ne == ["Si"]

    def test_exclusion_and_clearance(self):
        s = Nanoparticle(material="SiO2", radius=9.0).build_structure()
        ok = [i for i, r in enumerate(s.roles) if r == "OH"
              and silica.graft_site_clearance(s, i) >= 2.4]
        sites = silica.select_graft_sites(s, 2, exclude=ok[:1])
        assert ok[0] not in sites
        with pytest.raises(ValidationError, match="lower the grafting"):
            silica.select_graft_sites(s, 10_000)

    def test_spec_validation(self):
        poly = Polymer(chain_num=1, sequence=GRAFT)
        with pytest.raises(ValidationError, match="exactly one"):
            Grafting(polymer=poly)
        with pytest.raises(ValidationError, match="Unknown linker"):
            Grafting(polymer=poly, count=2, linker="glue")
        with pytest.raises(ValidationError, match="hydroxylated silica"):
            Nanoparticle(material="Au", radius=8.0,
                         grafts=Grafting(polymer=poly, count=2))

    def test_og_role_charges(self):
        for ff in ("interface", "clayff"):
            roles = get_inorganic_ff(ff, "sio2").roles
            assert roles["OG"].charge == pytest.approx(
                roles["OH"].charge + roles["HO"].charge)
            assert (roles["OG"].epsilon, roles["OG"].sigma) == \
                (roles["OH"].epsilon, roles["OH"].sigma)


def _build(tmp_path, ff="oplsaa", run=False):
    system = System(out=str(tmp_path / "out"))
    np_ = _particle()
    res = generate(system, "g", [Polymer(chain_num=8, sequence=PE), np_],
                   force_field=ff, box_size=60.0, rng_seed=3,
                   run_moltemplate=run)
    return tmp_path / "out" / "g", res, np_


def test_grafted_build(tmp_path):
    project, res, np_ = _build(tmp_path)
    build = project / "build" / "oplsaa"
    lib = UnitLibrary.load(build)
    (filler,) = lib.filler_units
    grafts = lib.graft_units
    rec = filler.material["grafts"]
    assert filler.class_name == "GNP_gsil" and len(grafts) == rec["n"]
    assert all(u.anchors["graft_of"] == "gsil" for u in grafts)
    # strategies never place graft chains themselves
    placed = {r.unit_id for r in res.records}
    assert not placed & {u.id for u in grafts}
    # linkage geometry: head C 1.41 A from its grafted O
    el, xyz = read_xyz(build / filler.coords_file)
    n_core = filler.envelope["n_core_atoms"]
    core_xyz, graft_xyz = xyz[:n_core], xyz[n_core:]
    entry = json.loads((project / "geometry" / "geometry.json")
                       .read_text())["inorganic"][0]
    for a in entry["grafts"]["anchors"]:
        d = np.sort(np.linalg.norm(graft_xyz - core_xyz[a["o"]], axis=1))
        assert d[0] == pytest.approx(1.41, abs=1e-3)   # O-C bond
        assert d[1] >= 1.85                            # 1-3 neighbors
    # grafts clear the core apart from the anchor neighborhood
    anchors = {a["o"] for a in entry["grafts"]["anchors"]} | \
        {a["si"] for a in entry["grafts"]["anchors"]}
    other = np.array([i not in anchors for i in range(n_core)])
    d = np.linalg.norm(graft_xyz[:, None] - core_xyz[other][None], axis=-1)
    assert d.min() > 1.6
    # charge: composite neutral, core carries the transferred cap charges
    assert abs(filler.material["net_charge"]) < 1e-6
    assert filler.material["core_net_charge"] == pytest.approx(
        rec["n"] * rec["cap_charge_transferred"], abs=1e-6)
    # coefficients come from the bundled OPLS-AA file
    coeffs = rec["linkage_coefficients"]
    assert coeffs["bond"]["gnp_gsil_ct_os"] == ["320.", "1.41"]
    assert coeffs["angle"]["gnp_gsil_ct_os_si"] == ["40.", "121.0"]
    assert "Si-OS-CT-CT" in rec["omitted_terms"]
    # removed cap H is gone from the head variant
    first = [u for u in grafts][0].monomer_files[0]
    head = (build / first).read_text()
    assert "$atom:H3 " not in head


def test_non_opls_host_refused(tmp_path):
    with pytest.raises(WorkflowError, match="host force field"):
        _build(tmp_path, ff="gaff2")


@pytest.mark.integration
def test_grafted_lammps_files(tmp_path):
    project, res, np_ = _build(tmp_path, run=True)
    groups = (project / "system.in.groups").read_text()
    assert "group rigid_fillers molecule 1" in groups
    data = (project / "system.data").read_text()
    q = 0.0
    section = None
    for line in data.splitlines():
        if line.startswith("Atoms"):
            section = "atoms"
            continue
        if section == "atoms":
            parts = line.split()
            if not parts:
                continue
            if not parts[0].isdigit():
                break
            q += float(parts[3])
    assert abs(q) < 1e-4
    report = json.loads((project / "ff_compatibility.json").read_text())
    assert report["explicit_cross_terms"]["n_pairs"] > 0
