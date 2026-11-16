#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nanoparticle typing and organic-inorganic force-field compatibility:
qualification enforcement, motions, charge neutrality, explicit cross
coefficients, host-style handling.
"""
import json
import math
import re

import pytest

from AutoPoly import Polymer, System, generate
from AutoPoly.core.exceptions import WorkflowError
from AutoPoly.materials.forcefields.compatibility import (
    combine,
    host_styles,
    read_self_pair_coeffs,
    read_type_assignments,
)
from AutoPoly.nanoparticles import Nanoparticle
from AutoPoly.pipeline.units import UnitLibrary

PE = ["CC[*]"] + ["[*]CC[*]"] * 6 + ["[*]CC"]


def _gen(tmp_path, np_, ff="oplsaa", run=False, **kw):
    system = System(out=str(tmp_path / "out"))
    kw.setdefault("box_size", 45.0)
    return generate(system, "t", [Polymer(chain_num=6, sequence=PE), np_],
                    force_field=ff, run_moltemplate=run, rng_seed=4, **kw)


def test_titania_is_geometry_only(tmp_path):
    np_ = Nanoparticle(material="TiO2", structure="anatase", radius=8.0)
    assert np_.force_field is None
    with pytest.raises(WorkflowError, match="geometry-only"):
        _gen(tmp_path, np_)
    # geometry export still works
    path = np_.write_structure(tmp_path / "tio2.pdb")
    assert path.read_text().count("HETATM") == np_.build_structure().n_atoms


def test_unqualified_combination_refused_with_hint(tmp_path):
    np_ = Nanoparticle(material="C", shape="sheet", length=20.0, width=18.0)
    with pytest.raises(WorkflowError, match="not md_ready") as exc:
        _gen(tmp_path, np_, ff="gaff2")
    assert "c-oplsaa-graphene" in str(exc.value)
    assert "allow_unqualified" in str(exc.value)


def test_allow_unqualified_is_recorded(tmp_path):
    np_ = Nanoparticle(material="C", shape="sheet", length=20.0, width=18.0,
                       allow_unqualified=True)
    _gen(tmp_path, np_, ff="gaff2")
    lib = UnitLibrary.load(tmp_path / "out" / "t" / "build" / "gaff2")
    (sheet,) = lib.filler_units
    assert sheet.material["qualification"]["status"] == "user-qualified"
    assert sheet.material["qualification"]["id"] is None


def test_flexible_motion_refused(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0, motion="flexible")
    with pytest.raises(WorkflowError, match="flexible particle needs"):
        _gen(tmp_path, np_)


def test_class2_host_refused(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0, allow_unqualified=True)
    with pytest.raises(WorkflowError, match="class2"):
        _gen(tmp_path, np_, ff="compass")


def test_filler_unit_record(tmp_path):
    np_ = Nanoparticle(material="Al2O3", radius=8.0, count=2, name="alumina")
    _gen(tmp_path, np_)
    lib = UnitLibrary.load(tmp_path / "out" / "t" / "build" / "oplsaa")
    (u,) = lib.filler_units
    assert (u.kind, u.role, u.geometry_type, u.motion) == \
        ("inorganic", "filler", "cluster", "rigid")
    assert u.count == 2 and u.class_name == "NP_alumina"
    m = u.material
    assert m["id"] == "al2o3" and m["inorganic_ff"] == "clayff_alumina"
    assert m["qualification"]["id"] == "al2o3-clayff-hydroxylated"
    assert m["cross_rule"] == "arithmetic"
    assert abs(m["net_charge"]) < 1e-6
    assert u.envelope["core_volume"] > 0 and u.mass > 0
    lt = (tmp_path / "out" / "t" / "build" / "oplsaa" / u.lt_file).read_text()
    assert "bond_coeff @bond:np_alumina_m_o 0.0" in lt  # exclusion-only


def test_hybrid_host_substyles_in_np_class(tmp_path):
    np_ = Nanoparticle(material="SiO2", radius=8.0)
    _gen(tmp_path, np_, ff="gaff2")
    lt = (tmp_path / "out" / "t" / "build" / "gaff2" /
          "NP_sio2_sphere.lt").read_text()
    assert re.search(r"pair_coeff @atom:i15_sc4 @atom:i15_sc4 "
                     r"lj/charmm/coul/long ", lt)
    assert "bond_coeff @bond:np_sio2_sphere_si_o harmonic 0.0" in lt


@pytest.mark.integration
@pytest.mark.parametrize("ff, material, kwargs, rule", [
    ("oplsaa", "SiO2", {}, "geometric"),            # host rule (OPLS)
    ("gaff2", "SiO2", {}, "arithmetic"),            # host rule (GAFF2)
    ("oplsaa", "SiO2", {"force_field": "clayff"}, "arithmetic"),  # CLAYFF
])
def test_explicit_cross_coefficients(tmp_path, ff, material, kwargs, rule):
    np_ = Nanoparticle(material=material, radius=8.0, **kwargs)
    _gen(tmp_path, np_, ff=ff, run=True)
    project = tmp_path / "out" / "t"
    mdir = project / "moltemplate" / "output"
    types = read_type_assignments(mdir)
    settings = project / "system.in.settings"
    selfc = read_self_pair_coeffs(settings)
    inorg = {t for t, (ns, _) in types.items() if ns.startswith("NP_")}
    organic = set(types) - inorg
    cross = {}
    for line in settings.read_text().splitlines():
        p = line.split("#")[0].split()
        if len(p) >= 5 and p[0] == "pair_coeff" and p[1] != p[2]:
            nums = [float(x) for x in p[3:] if re.match(r"^[-\d.eE+]+$", x)]
            cross[(int(p[1]), int(p[2]))] = nums[:2]
    # every organic-inorganic pair is explicit, with the declared rule
    for i in inorg:
        for j in organic:
            key = (min(i, j), max(i, j))
            assert key in cross, key
            eps, sig = combine(rule, selfc[i], selfc[j])
            assert cross[key][0] == pytest.approx(eps, rel=1e-5, abs=1e-12)
            assert cross[key][1] == pytest.approx(sig, rel=1e-5, abs=1e-12)
    report = json.loads((project / "ff_compatibility.json").read_text())
    assert report["explicit_cross_terms"]["n_pairs"] == len(cross)


def test_host_styles_parsing():
    from pathlib import Path
    root = Path(__file__).parent.parent / "AutoPoly" / "extern" / \
        "moltemplate" / "force_fields"
    opls = host_styles("oplsaa", root / "oplsaa.lt")
    assert opls.mixing_rule == "geometric" and opls.pair_substyle is None
    gaff = host_styles("gaff2", root / "gaff2.lt")
    assert gaff.mixing_rule == "arithmetic"
    assert gaff.pair_substyle == "lj/charmm/coul/long"
    assert gaff.bond_substyle == "harmonic"
    lopls = host_styles("lopls", root / "loplsaa.lt")   # via import
    assert lopls.lj_style == "lj/charmm/coul/long"
    assert host_styles("compass", root / "compass_published.lt").is_class2


def test_combining_rules():
    assert combine("geometric", (0.1, 4.0), (0.4, 1.0)) == \
        pytest.approx((0.2, 2.0))
    assert combine("arithmetic", (0.1, 4.0), (0.4, 1.0)) == \
        pytest.approx((0.2, 2.5))
    assert combine("arithmetic", (0.0, 0.0), (0.1, 3.0))[0] == 0.0
    assert math.isfinite(combine("geometric", (0.0, 0.0), (0.1, 3.0))[1])
