#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Material registry: five families, six IDs, truthful capability states."""
import pytest

from AutoPoly.core.exceptions import ValidationError
from AutoPoly.materials import (
    CAPABILITY_STAGES,
    MATERIAL_REGISTRY,
    Material,
    MaterialSpec,
    get_material,
    register_material,
)
from AutoPoly.materials.forcefields import (
    get_inorganic_ff,
    registered_inorganic_ffs,
)

SIX_IDS = ("sio2", "al2o3", "tio2", "c", "au", "ag")


def test_six_ids_five_families():
    assert set(SIX_IDS) <= set(MATERIAL_REGISTRY)
    families = {get_material(m).family for m in SIX_IDS}
    assert families == {"oxide", "carbon", "metal"}
    # five families: SiO2, Al2O3, TiO2 (oxides), carbon, elemental metals
    assert len({(get_material(m).family, get_material(m).formula)
                for m in SIX_IDS if get_material(m).family == "oxide"}) == 3


@pytest.mark.parametrize("alias, mid", [
    ("SiO2", "sio2"), ("silica", "sio2"), ("Al2O3", "al2o3"),
    ("alumina", "al2o3"), ("TiO2", "tio2"), ("titania", "tio2"),
    ("C", "c"), ("graphene", "c"), ("Au", "au"), ("gold", "au"),
    ("Ag", "ag"), ("silver", "ag"),
])
def test_aliases(alias, mid):
    assert get_material(alias).id == mid


def test_unknown_material_is_actionable():
    with pytest.raises(ValidationError, match="Registered materials"):
        get_material("unobtainium")


@pytest.mark.parametrize("mid", SIX_IDS)
def test_metadata_complete(mid):
    spec = get_material(mid)
    assert spec.phases and spec.default_phase in spec.phases
    assert spec.surface_states and spec.default_surface in spec.surface_states
    assert spec.chemistry_module() is not None
    for shape, ref in spec.builders.items():
        assert callable(spec.builder(shape)), ref
    rep = spec.capability_report()
    assert set(rep) == set(CAPABILITY_STAGES)
    assert rep["registered"] and rep["geometry_ready"] and rep["surface_ready"]


def test_md_ready_is_derived_not_declared():
    for mid in SIX_IDS:
        spec = get_material(mid)
        assert "md_ready" not in spec.capabilities
    assert get_material("tio2").capability("md_ready") is False
    for mid in ("sio2", "al2o3", "c", "au", "ag"):
        assert get_material(mid).capability("md_ready") is True


def test_force_field_adapters_are_per_material():
    assert registered_inorganic_ffs("sio2") == ["clayff_silica",
                                                "interface_silica"]
    assert registered_inorganic_ffs("tio2") == []
    assert get_inorganic_ff("interface", "sio2").name == "interface_silica"
    assert get_inorganic_ff("clayff", "al2o3").name == "clayff_alumina"
    with pytest.raises(ValidationError, match="not defined for material"):
        get_inorganic_ff("clayff_alumina", "sio2")
    with pytest.raises(ValidationError, match="Unknown inorganic"):
        get_inorganic_ff("interface", "tio2")


def test_material_object():
    silica = Material(name="SiO2", crystal="alpha_quartz",
                      force_field="interface")
    assert silica.id == "sio2" and silica.resolved_phase == "alpha_quartz"
    assert silica.force_field_adapter.name == "interface_silica"
    with pytest.raises(ValidationError, match="geometry-only"):
        Material("TiO2").force_field_adapter
    with pytest.raises(ValidationError, match="Unknown phase"):
        Material("SiO2", crystal="stishovite").resolved_phase


def test_registration_rejects_alias_collisions():
    with pytest.raises(ValidationError, match="already"):
        register_material(MaterialSpec(id="silica2", name="x", formula="SiO2",
                                       family="oxide", elements=("Si", "O")))


def test_clayff_values_match_cygan_2004():
    """sigma = R0 / 2^(1/6): st 3.3020 A, ob/oh 3.1655 A, ao 4.2712 A."""
    from AutoPoly.materials.forcefields.tables import CLAYFF, CLAYFF_ALUMINA
    assert CLAYFF["Si"]["sigma"] == pytest.approx(3.3020, abs=2e-4)
    assert CLAYFF["OB"]["sigma"] == pytest.approx(3.1655, abs=2e-4)
    assert CLAYFF["OB"]["epsilon"] == pytest.approx(0.1554)
    assert CLAYFF_ALUMINA["Al"]["sigma"] == pytest.approx(4.2712, abs=2e-4)
    assert CLAYFF_ALUMINA["Al"]["charge"] == pytest.approx(1.575)


def test_metal_lj_values_match_heinz_2008():
    """Au: eps 5.29, r0 2.951 (sigma 2.629); Ag: eps 4.56, r0 2.955."""
    au = get_inorganic_ff("interface_metal_au").roles["Au"]
    ag = get_inorganic_ff("interface_metal_ag").roles["Ag"]
    assert au.epsilon == 5.29 and au.sigma == pytest.approx(2.629, abs=1e-3)
    assert ag.epsilon == 4.56 and ag.sigma == pytest.approx(2.633, abs=1e-3)
    assert au.charge == ag.charge == 0.0


def test_silica_tables_shared_not_duplicated():
    from AutoPoly.materials.forcefields import tables
    from AutoPoly.surfaces import ff_tables
    assert ff_tables.INTERFACE_FF is tables.INTERFACE_FF
    assert ff_tables.CLAYFF is tables.CLAYFF
    assert ff_tables.SLAB_FF_REGISTRY is tables.SLAB_FF_REGISTRY
