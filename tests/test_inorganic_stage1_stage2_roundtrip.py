#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1 inorganic geometry -> stage-2 typing round trips.

The geometry.json inorganic entry is force-field agnostic and reusable:
the same stored slab is typed under different film force fields (hybrid
vs plain pair styles) and under different inorganic tables, without
changing its coordinates or topology.
"""
import json

import numpy as np
import pytest

from AutoPoly import Polymer, System
from AutoPoly.core.exceptions import ValidationError
from AutoPoly.materials import AtomicStructure
from AutoPoly.packing import SubstrateSpec
from AutoPoly.pipeline import GeometryBuilder, GeometryConfig, UnitTyper
from AutoPoly.pipeline.unit_coords import read_xyz
from AutoPoly.pipeline.units import UnitLibrary
from AutoPoly.surfaces import SLAB_BUILDERS
from AutoPoly.surfaces.component import BuiltinSlabComponent

PE = ["CC[*]", "[*]CC[*]", "[*]CC"]
DIMS = (30.0, 34.0, 40.0)


@pytest.fixture(scope="module")
def geometry(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rt")
    system = System(out=str(tmp / "out"))
    spec = SubstrateSpec(builder="alpha_quartz", thickness=12.0)
    film = Polymer(chain_num=1, sequence=PE, tacticity="syndiotactic")
    result = GeometryBuilder(
        system, "rt", GeometryConfig(use_mc_chain_growth=False)
    ).build([film], inorganic=[BuiltinSlabComponent(spec, DIMS)])
    return result


def _atoms_block(lt_text):
    lines = lt_text.splitlines()
    start = lines.index('  write("Data Atoms") {')
    end = lines.index("  }", start)
    return [line.split() for line in lines[start + 1:end]]


def test_geometry_entry_is_ff_agnostic(geometry):
    data = json.loads(geometry.json_path.read_text())
    assert data["format_version"] == 2
    (entry,) = data["inorganic"]
    assert entry["component"] == "builtin_slab"
    assert entry["role"] == "substrate"
    assert entry["geometry_type"] == "slab"
    s = AtomicStructure.from_dict(entry["structure"])
    assert set(s.roles) == {"Si", "OB", "OH", "HO"}
    # no FF data stored in the structure
    assert "charges" not in entry["structure"]
    # identical to the builder's direct structure
    direct = SLAB_BUILDERS["alpha_quartz"](
        lx_target=DIMS[0], ly_target=DIMS[1], thickness=12.0
    ).build_structure()
    np.testing.assert_array_equal(s.positions, direct.positions)
    assert s.bonds == direct.bonds


def test_same_geometry_typed_under_two_film_force_fields(geometry, tmp_path):
    lib_gaff = UnitTyper(geometry.dir, "gaff",
                         output_dir=tmp_path / "gaff").type()
    lib_opls = UnitTyper(geometry.dir, "oplsaa",
                         output_dir=tmp_path / "oplsaa").type()
    lt_gaff = (tmp_path / "gaff" / "alpha_quartz_slab.lt").read_text()
    lt_opls = (tmp_path / "oplsaa" / "alpha_quartz_slab.lt").read_text()
    # hybrid GAFF pair/bond styles need sub-style keywords; OPLS does not
    assert "lj/charmm/coul/long" in lt_gaff and "harmonic 0.0" in lt_gaff
    assert "lj/charmm/coul/long" not in lt_opls
    assert _atoms_block(lt_gaff) == _atoms_block(lt_opls)
    for lib in (lib_gaff, lib_opls):
        (slab,) = lib.inorganic_units
        assert slab.kind == "inorganic" and slab.role == "substrate"
        assert slab.geometry_type == "slab" and slab.motion == "frozen"
        assert slab.class_name == "QuartzSlab"
        assert slab.material["inorganic_ff"] == "interface_silica"


def test_same_geometry_typed_under_two_inorganic_tables(geometry, tmp_path):
    a = UnitTyper(geometry.dir, "oplsaa", output_dir=tmp_path / "a").type()
    b = UnitTyper(geometry.dir, "oplsaa", output_dir=tmp_path / "b",
                  inorganic_force_fields={"substrate": "clayff"}).type()
    lt_a = (tmp_path / "a" / "alpha_quartz_slab.lt").read_text()
    lt_b = (tmp_path / "b" / "alpha_quartz_slab.lt").read_text()
    rows_a, rows_b = _atoms_block(lt_a), _atoms_block(lt_b)
    # same atoms/coordinates, different types and charges
    assert [r[0] for r in rows_a] == [r[0] for r in rows_b]
    assert [r[4:] for r in rows_a] == [r[4:] for r in rows_b]
    assert {r[2] for r in rows_b} == {"@atom:cff_st", "@atom:cff_ob",
                                      "@atom:cff_oh", "@atom:cff_ho"}
    assert b.inorganic_units[0].material["inorganic_ff"] == "clayff_silica"
    assert a.inorganic_units[0].material["inorganic_ff"] == "interface_silica"


def test_coords_file_and_manifest_round_trip(geometry, tmp_path):
    lib = UnitTyper(geometry.dir, "gaff", output_dir=tmp_path / "g").type()
    loaded = UnitLibrary.load(tmp_path / "g")
    assert loaded.format_version == 2
    slab = loaded.inorganic_units[0]
    elements, coords = read_xyz(tmp_path / "g" / slab.coords_file)
    entry = json.loads(geometry.json_path.read_text())["inorganic"][0]
    s = AtomicStructure.from_dict(entry["structure"])
    assert elements == s.elements
    np.testing.assert_allclose(coords, s.positions, atol=1e-6)
    # organic units carry coordinates too
    poly = [u for u in loaded.units if u.kind == "polymer"]
    assert poly and all(u.coords_file and u.mass > 0 for u in poly)
    assert [u.to_dict() for u in loaded.units] == \
        [u.to_dict() for u in lib.units]


def test_unknown_inorganic_override_rejected(geometry, tmp_path):
    with pytest.raises(ValidationError, match="unknown component"):
        UnitTyper(geometry.dir, "gaff", output_dir=tmp_path / "x",
                  inorganic_force_fields={"nope": "clayff"}).type()


def test_stage2_slab_equals_pack_time_slab(geometry, tmp_path):
    """Stage-2 typed slab .lt == legacy pack-time .lt (same substyles)."""
    UnitTyper(geometry.dir, "gaff", output_dir=tmp_path / "g").type()
    legacy = SLAB_BUILDERS["alpha_quartz"](
        lx_target=DIMS[0], ly_target=DIMS[1], thickness=12.0,
        pair_substyle="lj/charmm/coul/long", bond_substyle="harmonic",
    ).build()
    assert (tmp_path / "g" / "alpha_quartz_slab.lt").read_text() == \
        legacy.lt_text()
