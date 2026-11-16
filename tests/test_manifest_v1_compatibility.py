#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version-1 artifacts stay readable; the legacy pack-time slab route keeps
working and produces the same system as the stage-1/2 route.
"""
import json

import pytest

from AutoPoly import Polymer, System, generate
from AutoPoly.core.exceptions import ValidationError
from AutoPoly.packing import SubstrateSpec
from AutoPoly.pipeline import BoxPacker, GeometryBuilder, GeometryConfig, UnitTyper
from AutoPoly.pipeline.units import (
    FORMAT_VERSION,
    UnitLibrary,
    UnitSpec,
    migrate_manifest,
)

V1_MANIFEST = {
    "format_version": 1,
    "force_field": "oplsaa",
    "geometry_source": "../geometry/geometry.json",
    "build_config": {"offset": 4.0},
    "monomer_files": ["monomer_0_0le.lt"],
    "units": [
        {"id": "poly_1", "kind": "polymer", "lt_file": "poly_1.lt",
         "count": 1, "topology": "linear", "n_monomers": 3,
         "radius": 7.0, "anchors": {}, "role": "film",
         "monomer_files": ["monomer_0_0le.lt"]},
        {"id": "water", "kind": "molecule", "lt_file": "water.lt",
         "count": 10, "radius": 2.0, "role": "substrate"},
    ],
}


class TestUnitsV1:
    def test_v1_manifest_loads_with_defaults(self, tmp_path):
        (tmp_path / "units.json").write_text(json.dumps(V1_MANIFEST))
        lib = UnitLibrary.load(tmp_path)
        assert lib.format_version == 1
        poly, water = lib.units
        assert poly.kind == "polymer" and poly.role == "film"
        assert water.role == "substrate"
        for u in lib.units:
            assert u.geometry_type is None and u.coords_file is None
            assert u.material is None and u.motion is None
        assert lib.inorganic_units == []
        assert lib.total_particle_count() == 13

    def test_migration_is_explicit_and_lossless(self):
        migrated = migrate_manifest(dict(V1_MANIFEST))
        assert migrated["units"] == V1_MANIFEST["units"]
        with pytest.raises(ValidationError):
            migrate_manifest({"format_version": 0})

    def test_current_version_is_two(self):
        assert FORMAT_VERSION == 2

    def test_filler_role_requires_inorganic_kind(self):
        with pytest.raises(ValidationError, match="reserved for inorganic"):
            UnitSpec(id="x", kind="molecule", lt_file="x.lt", role="filler")

    def test_inorganic_kind_requires_geometry_and_motion(self):
        with pytest.raises(ValidationError, match="geometry_type"):
            UnitSpec(id="np", kind="inorganic", lt_file="np.lt",
                     role="filler", motion="rigid")
        with pytest.raises(ValidationError, match="motion"):
            UnitSpec(id="np", kind="inorganic", lt_file="np.lt",
                     role="filler", geometry_type="cluster", motion="bogus")


class TestGeometryV1:
    def test_polymer_only_geometry_stays_v1(self, tmp_path):
        system = System(out=str(tmp_path / "o"))
        poly = Polymer(chain_num=1, sequence=["CC[*]", "[*]CC[*]", "[*]CC"])
        res = GeometryBuilder(system, "g",
                              GeometryConfig(use_mc_chain_growth=False)
                              ).build([poly])
        data = json.loads(res.json_path.read_text())
        assert data["format_version"] == 1
        assert "inorganic" not in data
        loaded = GeometryBuilder.load(res.dir)
        assert loaded["inorganic"] == []


@pytest.mark.integration
def test_legacy_pack_time_slab_route_equals_stage_route(tmp_path):
    """
    A build without an inorganic substrate unit (as produced before the
    refactoring) still gets its slab built at pack time -- and the packed
    system is identical to the stage-1/2 route of generate().
    """
    pe = ["CC[*]", "[*]CC[*]", "[*]CC"]
    dims = (30.0, 34.0, 40.0)

    def spec():
        return SubstrateSpec(builder="alpha_quartz", thickness=12.0, gap=3.0)

    # stage-1/2 route
    sys_a = System(out=str(tmp_path / "a"))
    generate(sys_a, "p", [Polymer(chain_num=2, sequence=pe,
                                  tacticity="syndiotactic")],
             force_field="gaff", substrate=spec(), box_dims=dims,
             geometry_config=GeometryConfig(use_mc_chain_growth=False),
             rng_seed=3)

    # legacy route: polymer-only geometry + typing, slab at pack time
    sys_b = System(out=str(tmp_path / "b"))
    geom = GeometryBuilder(sys_b, "p", GeometryConfig(
        use_mc_chain_growth=False)).build(
        [Polymer(chain_num=2, sequence=pe, tacticity="syndiotactic")])
    lib = UnitTyper(geom.dir, "gaff").type()
    assert lib.inorganic_units == []
    BoxPacker(sys_b, "p", strategy="on_substrate", substrate=spec(),
              box_dims=dims, rng_seed=3).pack(lib)

    data_a = (tmp_path / "a" / "p" / "system.data").read_text()
    data_b = (tmp_path / "b" / "p" / "system.data").read_text()
    assert data_a == data_b
    for f in ("system.in.settings", "in.run"):
        assert (tmp_path / "a" / "p" / f).read_text() == \
            (tmp_path / "b" / "p" / f).read_text()
