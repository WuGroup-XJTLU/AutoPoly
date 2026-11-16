#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualification evidence: every md_ready combination of the material
qualification matrix is built end to end and (when a LAMMPS binary is on
PATH) smoke-tested: minimization with held fillers, then short rigid/frozen
+ matrix NVT dynamics.

The construction part (stages 1-3 + moltemplate, explicit cross terms,
component groups, run template, FF report) always runs; the LAMMPS part is
skipped without LAMMPS.  Each parametrized case ID is referenced by the
``evidence`` of a QualifiedCombination (tests/test_material_ff_
compatibility_matrix.py checks the cross references).
"""
import json

import pytest

from AutoPoly import Polymer, System, generate
from AutoPoly.nanoparticles import Grafting, Nanoparticle
from AutoPoly.packing import NanocompositePacking
from AutoPoly.pipeline.smoke import lammps_executable, run_smoke

PE = ["CC[*]"] + ["[*]CC[*]"] * 8 + ["[*]CC"]
PS = ["CC(c1ccccc1)[*]"] + ["[*]CC(c1ccccc1)[*]"] * 6 + ["[*]CC(c1ccccc1)"]
GRAFT = ["CC[*]"] + ["[*]CC[*]"] * 4 + ["[*]CC"]
PMMA = (["CC(C)(C(=O)OC)[*]"] + ["[*]CC(C)(C(=O)OC)[*]"] * 6
        + ["[*]CC(C)(C(=O)OC)"])

#: case -> (sequence, chains, host FF, Nanoparticle kwargs, qualification)
CASES = {
    "sio2_ps_oplsaa_rigid": (PS, 16, "oplsaa", dict(
        material="SiO2", structure="beta_cristobalite", radius=9.0, count=2,
        motion="rigid"), "sio2-interface-hydroxylated"),
    "sio2_pmma_oplsaa_rigid": (PMMA, 12, "oplsaa", dict(
        material="SiO2", structure="alpha_quartz", radius=9.0, count=1,
        motion="rigid"), "sio2-interface-hydroxylated"),
    "sio2_pe_oplsaa_frozen": (PE, 20, "oplsaa", dict(
        material="SiO2", radius=9.0, count=1, motion="frozen"),
        "sio2-interface-hydroxylated"),
    "sio2_pe_gaff_rigid": (PE, 20, "gaff", dict(
        material="SiO2", radius=9.0, count=1, motion="rigid"),
        "sio2-interface-hydroxylated"),
    "sio2_pe_gaff2_rigid": (PE, 20, "gaff2", dict(
        material="SiO2", radius=9.0, count=1, motion="rigid"),
        "sio2-interface-hydroxylated"),
    "sio2_imported_pe_oplsaa_rigid": (PE, 20, "oplsaa", "imported",
                                      "sio2-interface-hydroxylated"),
    "sio2_grafted_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="SiO2", radius=9.0, count=1, motion="rigid",
        name="gsilica",
        grafts=Grafting(polymer=Polymer(chain_num=1, sequence=GRAFT),
                        density=0.5, seed=2)),
        "sio2-interface-grafted"),
    "sio2clayff_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="SiO2", radius=9.0, count=1, motion="rigid",
        force_field="clayff"), "sio2-clayff-hydroxylated"),
    "al2o3_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="Al2O3", radius=8.0, count=1, motion="rigid"),
        "al2o3-clayff-hydroxylated"),
    "al2o3_pe_oplsaa_frozen": (PE, 20, "oplsaa", dict(
        material="Al2O3", radius=8.0, count=1, motion="frozen"),
        "al2o3-clayff-hydroxylated"),
    "al2o3dry_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="Al2O3", radius=9.0, count=1, motion="rigid",
        surface="stoichiometric"), "al2o3-clayff-stoichiometric"),
    "graphene_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="C", shape="sheet", length=22.0, width=20.0, count=1,
        motion="rigid"), "c-oplsaa-graphene"),
    "graphene_pe_oplsaa_frozen": (PE, 20, "oplsaa", dict(
        material="C", shape="sheet", length=22.0, width=20.0, count=1,
        motion="frozen"), "c-oplsaa-graphene"),
    "cnt_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="C", structure="cnt", shape="tube", n=6, m=6, length=20.0,
        count=1, motion="rigid"), "c-oplsaa-graphene"),
    "au_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="Au", radius=8.0, count=1, motion="rigid"),
        "au-interface-lj"),
    "au_pe_gaff2_rigid": (PE, 20, "gaff2", dict(
        material="Au", radius=8.0, count=1, motion="rigid"),
        "au-interface-lj"),
    "ag_pe_oplsaa_rigid": (PE, 20, "oplsaa", dict(
        material="Ag", radius=8.0, count=1, motion="rigid"),
        "ag-interface-lj"),
}


def _nanoparticle(spec, tmp_path):
    if spec == "imported":
        # round trip through a PDB with CONECT records
        src = Nanoparticle(material="SiO2", radius=9.0)
        pdb = src.write_structure(tmp_path / "silica_np.pdb")
        return Nanoparticle.from_file(pdb, material="SiO2", name="silica_imp",
                                      count=1, motion="rigid")
    return Nanoparticle(**spec)


def _build(name, tmp_path):
    seq, chains, ff, spec, qual = CASES[name]
    system = System(out=str(tmp_path / "out"))
    poly = Polymer(chain_num=chains, sequence=seq, tacticity="atactic")
    np_ = _nanoparticle(spec, tmp_path)
    result = generate(system, "nc", [poly, np_], force_field=ff,
                      packing=NanocompositePacking(initial_density=0.35,
                                                   seed=3),
                      rng_seed=3)
    return tmp_path / "out" / "nc", result, np_, qual


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CASES))
def test_lammps_smoke(name, tmp_path):
    project, result, np_, qual = _build(name, tmp_path)

    # --- construction checks (always) ---------------------------------
    report = json.loads((project / "ff_compatibility.json").read_text())
    (comp,) = report["components"]
    assert comp["qualification"]["id"] == qual
    if np_.grafts is not None:
        grafts = json.loads((project / "build" / "oplsaa" / "units.json")
                            .read_text())
        filler = [u for u in grafts["units"] if u["role"] == "filler"][0]
        rec = filler["material"]["grafts"]
        assert rec["n"] >= 1 and rec["linkage_coefficients"]["bond"]
    assert comp["qualification"]["status"] == "qualified"
    assert abs(comp["net_charge"]) < 1e-4
    assert report["explicit_cross_terms"]["n_pairs"] > 0
    settings = (project / "system.in.settings").read_text()
    assert "explicit organic-inorganic cross interactions" in settings
    groups = (project / "system.in.groups").read_text()
    assert "group fillers molecule" in groups
    assert f"group {np_.motion}_fillers molecule" in groups
    run = (project / "in.run").read_text()
    assert ("rigid/nvt molecule" in run) == (np_.motion == "rigid")
    assert ("setforce 0.0 0.0 0.0" in run)
    md = result.report["min_distance"]
    assert md["filler_matrix"] is None or md["filler_matrix"] >= 2.5 - 1e-6
    assert result.report["intrachain"]["overlapping_chains"] == 0
    assert sum(r.instance_name.startswith(np_.component_id)
               for r in result.records) == np_.count

    # --- LAMMPS smoke (when available) --------------------------------
    if lammps_executable() is None:
        pytest.skip("LAMMPS not available: construction checks only")
    smoke = run_smoke(project, steps=500,
                      expected_rigid_bodies=(np_.count if np_.motion ==
                                             "rigid" else None))
    assert smoke.ok, smoke.errors
    last = smoke.thermo[-1]
    assert 100.0 < last["Temp"] < 600.0
