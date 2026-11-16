#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nanoparticle geometry: crystal sources, shape builders, imports."""
import json

import numpy as np
import pytest

from AutoPoly.core.exceptions import ValidationError
from AutoPoly.materials.crystals import CRYSTALS
from AutoPoly.materials.topology import neighbor_pairs
from AutoPoly.nanoparticles import Nanoparticle, Sheet, Sphere, Tube

#: crystal -> {element: bulk coordination}, neighbor cutoff
BULK_CN = {
    "alpha_quartz": ({"Si": 4, "O": 2}, 1.9),
    "beta_cristobalite": ({"Si": 4, "O": 2}, 1.9),
    "alpha_corundum": ({"Al": 6, "O": 4}, 2.1),
    "rutile": ({"Ti": 6, "O": 3}, 2.2),
    "anatase": ({"Ti": 6, "O": 3}, 2.2),
    "fcc_au": ({"Au": 12}, 3.2),
    "fcc_ag": ({"Ag": 12}, 3.2),
}

#: (crystal, expected density g/cm^3)
DENSITY = {"alpha_quartz": 2.65, "beta_cristobalite": 2.18,
           "alpha_corundum": 3.99, "rutile": 4.25, "anatase": 3.89,
           "fcc_au": 19.30, "fcc_ag": 10.49}


@pytest.mark.parametrize("name", sorted(BULK_CN))
def test_bulk_cells_fully_coordinated_under_3d_pbc(name):
    """The orthorhombic cells are genuine bulk cells (not surface cells)."""
    cell = CRYSTALS[name]
    expected, cut = BULK_CN[name]
    el, pos = cell.cartesian()
    # 3x3x3 supercell so the cutoff is below half the box
    reps = [np.array([i, j, k]) * np.asarray(cell.lengths)
            for i in range(3) for j in range(3) for k in range(3)]
    pos3 = np.vstack([pos + r for r in reps])
    el3 = el * 27
    i, j, _, _ = neighbor_pairs(pos3, cut, np.diag(np.asarray(cell.lengths)
                                                   * 3), (True,) * 3)
    cn = np.bincount(np.concatenate([i, j]), minlength=len(el3))
    for element, n in expected.items():
        assert set(cn[[e == element for e in el3]]) == {n}, element
    assert cell.density() == pytest.approx(DENSITY[name], rel=0.02)


def test_graphene_cell_bond_length():
    el, pos = CRYSTALS["graphene"].cartesian()
    d = np.linalg.norm(pos[1] - pos[0])
    assert d == pytest.approx(1.420, abs=2e-3)


@pytest.mark.parametrize("kw, formula_checks", [
    (dict(material="SiO2", structure="beta_cristobalite", radius=10.0), "si"),
    (dict(material="SiO2", structure="alpha_quartz", radius=10.0), "si"),
    (dict(material="Al2O3", radius=9.0), "al"),
    (dict(material="Al2O3", radius=10.0, surface="stoichiometric"), "al_dry"),
    (dict(material="TiO2", structure="anatase", radius=9.0), "ti"),
    (dict(material="TiO2", structure="rutile", radius=9.0), "ti"),
])
def test_oxide_spheres_neutral_and_coordinated(kw, formula_checks):
    s = Nanoparticle(**kw).build_structure()
    c = s.element_counts()
    if formula_checks == "si":
        roles = s.roles
        n = {r: roles.count(r) for r in ("Si", "OB", "OH", "HO")}
        assert 2 * n["OB"] + n["OH"] == 4 * n["Si"]
        assert n["HO"] == n["OH"]
    elif formula_checks == "al":
        assert 3 * c["Al"] - 2 * c["O"] + c.get("H", 0) == 0
    elif formula_checks == "al_dry":
        assert "H" not in c and 2 * c["O"] == 3 * c["Al"]
    else:
        assert 4 * c["Ti"] - 2 * c["O"] + c.get("H", 0) == 0
    # finite, centered, one connected cluster
    assert not s.is_periodic and s.geometry_type == "cluster"
    assert max(s.component_ids) == 0
    assert np.linalg.norm(s.positions.mean(axis=0)) < 2.0
    assert s.descriptors["core_volume"] > 0
    assert s.metadata["material_id"] == Nanoparticle(**kw).material.id


def test_silica_sphere_is_reproducible_and_si_centered_cut():
    a = Nanoparticle(material="SiO2", radius=10.0, seed=1).build_structure()
    b = Nanoparticle(material="SiO2", radius=10.0, seed=1).build_structure()
    np.testing.assert_array_equal(a.positions, b.positions)
    si = a.positions[[e == "Si" for e in a.elements]]
    assert np.linalg.norm(si, axis=1).max() <= 10.0 + 1e-9
    # no Q0/Q1 silicon left
    assert set(a.metadata["q_distribution"]) <= {"Q2", "Q3", "Q4"}


def test_silica_center_offset_changes_surface():
    a = Nanoparticle(material="SiO2", radius=9.0).build_structure()
    b = Nanoparticle(material="SiO2", shape=Sphere(radius=9.0,
                     center_offset=(0.7, 0.3, 0.2))).build_structure()
    assert a.n_atoms != b.n_atoms or not np.allclose(a.positions, b.positions)


def test_graphene_and_nanotube():
    g = Nanoparticle(material="C", shape="sheet", length=24.0,
                     width=20.0).build_structure()
    assert g.geometry_type == "sheet"
    assert np.ptp(g.positions[:, 2]) < 1e-6          # planar
    assert set(g.roles) == {"CA", "CE", "HE"}
    t = Nanoparticle(material="C", structure="cnt",
                     shape=Tube(n=8, m=0, length=20.0)).build_structure()
    assert t.geometry_type == "rod"
    c = t.positions[[e == "C" for e in t.elements]]
    r = np.hypot(c[:, 0], c[:, 1])
    expected = 2.46 * 8 / (2 * np.pi)
    assert r == pytest.approx(np.full(len(r), expected), abs=1e-6)
    bi = Nanoparticle(material="C", shape=Sheet(length=20, width=18,
                                                layers=2)).build_structure()
    assert len(set(np.round(bi.positions[:, 2], 3))) == 2


@pytest.mark.parametrize("metal", ["Au", "Ag"])
def test_metal_sphere(metal):
    s = Nanoparticle(material=metal, radius=9.0).build_structure()
    assert set(s.elements) == {metal} and s.bonds == []
    assert s.metadata["n_surface"] > 0
    assert min(int(k[2:]) for k in s.metadata["coordination"]) >= 3


def test_metal_polyhedra_need_ase():
    pytest.importorskip("ase")
    s = Nanoparticle(material="Au", shape="cuboctahedron",
                     radius=8.0).build_structure()
    assert s.n_atoms in (55, 147, 309, 561)   # magic cuboctahedra


def test_shape_phase_validation():
    with pytest.raises(ValidationError, match="No built-in 'sheet'"):
        Nanoparticle(material="SiO2", shape="sheet")
    with pytest.raises(ValidationError, match="requires radius"):
        Nanoparticle(material="SiO2", shape="sphere")
    with pytest.raises(ValidationError, match="outside the supported"):
        Nanoparticle(material="SiO2", radius=500.0)
    with pytest.raises(ValidationError, match="motion"):
        Nanoparticle(material="SiO2", radius=8.0, motion="wobbly")
    with pytest.raises(ValidationError, match="count"):
        Nanoparticle(material="SiO2", radius=8.0, count=0)
    with pytest.raises(ValidationError, match="Unknown surface"):
        Nanoparticle(material="SiO2", radius=8.0, surface="fluorinated")


class TestImport:
    def test_json_round_trip(self, tmp_path):
        np_ = Nanoparticle(material="SiO2", radius=8.0)
        path = np_.write_structure(tmp_path / "np.json")
        imp = Nanoparticle.from_file(path, material="SiO2")
        s, s0 = imp.build_structure(), np_.build_structure()
        assert imp.geometry_type == "imported"
        assert s.bonds == s0.bonds
        assert s.elements == s0.elements

    def test_pdb_conect_round_trip(self, tmp_path):
        np_ = Nanoparticle(material="SiO2", radius=8.0)
        path = np_.write_structure(tmp_path / "np.pdb")
        imp = Nanoparticle.from_file(path, material="SiO2", count=2)
        s = imp.build_structure()
        assert s.n_bonds == np_.build_structure().n_bonds
        assert imp.phase == "amorphous" and imp.count == 2

    def test_xyz_requires_explicit_perception(self, tmp_path):
        np_ = Nanoparticle(material="SiO2", radius=8.0)
        path = np_.write_structure(tmp_path / "np.xyz")
        with pytest.raises(ValidationError, match="no bond topology"):
            Nanoparticle.from_file(path, material="SiO2")
        imp = Nanoparticle.from_file(path, material="SiO2",
                                     perceive_bonds=True)
        assert imp.build_structure().n_bonds == np_.build_structure().n_bonds

    def test_invalid_import_rejected(self, tmp_path):
        np_ = Nanoparticle(material="SiO2", radius=8.0)
        s = np_.build_structure()
        # strip all hydrogens -> dangling silanol oxygens
        keep = [i for i, e in enumerate(s.elements) if e != "H"]
        bare = s.subset(keep)
        (tmp_path / "bare.json").write_text(json.dumps(bare.to_dict()))
        with pytest.raises(ValidationError, match="dangling"):
            Nanoparticle.from_file(tmp_path / "bare.json", material="SiO2")

    def test_wrong_material_elements(self, tmp_path):
        np_ = Nanoparticle(material="Au", radius=6.0)
        path = np_.write_structure(tmp_path / "au.json")
        with pytest.raises(ValidationError, match="not allowed"):
            Nanoparticle.from_file(path, material="SiO2")
