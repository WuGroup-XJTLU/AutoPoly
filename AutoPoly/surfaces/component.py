#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1/Stage-2 adapter for built-in substrate slabs.

``SubstrateSpec(builder="alpha_quartz"|"beta_cristobalite")`` keeps its
public behavior.  Internally, generate() now builds the slab geometry in
stage 1 (an "inorganic" entry of geometry.json holding the force-field
agnostic AtomicStructure plus the builder/force-field request), types it
in stage 2 against the film force field (writing the slab .lt and an
inorganic substrate unit into build/<ff>/units.json), and stage 3 only
places it.  The same stored geometry can therefore be typed under several
film force fields -- and under either silica table -- without rebuilding.

BoxPacker used directly on a build without an inorganic substrate unit
(e.g. a v1 build directory) still materializes the slab at pack time
(legacy route); both routes produce identical .lt files.

Created on 2026-11-13
@author: zwu
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ..core.exceptions import ValidationError, WorkflowError

if TYPE_CHECKING:
    from ..pipeline.units import UnitSpec

#: geometry.json "component" key of built-in slabs.
COMPONENT_KIND = "builtin_slab"


class BuiltinSlabComponent:
    """
    Inorganic stage-1 component wrapping a builder SubstrateSpec.

    Args:
        spec: SubstrateSpec with ``builder`` set.
        box_dims: generate() box_dims; the lateral sides are required (the
                  slab is built to match the box).
    """

    _is_inorganic_component = True
    component_id = "substrate"

    def __init__(self, spec, box_dims: Optional[Tuple] = None) -> None:
        if not getattr(spec, "is_builder", False):
            raise ValidationError(
                "BuiltinSlabComponent requires SubstrateSpec(builder=...)"
            )
        dims = box_dims or (None, None, None)
        if dims[0] is None or dims[1] is None:
            raise WorkflowError(
                "Builder substrates require explicit lateral box_dims "
                "(box_dims=(lx, ly, lz); lz may be None) so the slab can "
                "be built to match the box"
            )
        self.spec = spec
        self.lx_target = float(dims[0])
        self.ly_target = float(dims[1])

    # ------------------------------------------------------------------
    def builder(self, slab_ff: Optional[str] = None,
                pair_substyle: Optional[str] = None,
                bond_substyle: Optional[str] = None):
        """The SlabBuilder configured exactly as the legacy pack-time route."""
        return make_slab_builder(self.request(), slab_ff=slab_ff,
                                 pair_substyle=pair_substyle,
                                 bond_substyle=bond_substyle)

    def request(self) -> Dict[str, Any]:
        spec = self.spec
        return {
            "name": spec.builder,
            "lx_target": self.lx_target,
            "ly_target": self.ly_target,
            "thickness": float(spec.thickness),
            "oh_density": float(spec.oh_density),
            "hydroxylate_bottom": bool(spec.hydroxylate_bottom),
            "seed": spec.slab_seed,
            "slab_ff": spec.slab_ff,
            "slab_types": spec.slab_types,
            "slab_charges": spec.slab_charges,
            "slab_lj": (None if spec.slab_lj is None else
                        {r: list(v) for r, v in spec.slab_lj.items()}),
        }

    def geometry_entry(self, component_id: str = "substrate") -> Dict[str, Any]:
        """The geometry.json inorganic entry (FF-agnostic structure)."""
        builder = self.builder()
        structure = builder.build_structure()
        return {
            "id": component_id,
            "component": COMPONENT_KIND,
            "role": "substrate",
            "count": 1,
            "motion": "frozen",
            "material": "sio2",
            "phase": self.spec.builder,
            "surface": "hydroxylated",
            "geometry_type": "slab",
            "class_name": builder.class_name,
            "lt_filename": builder.lt_filename,
            "builder": self.request(),
            "structure": structure.to_dict(),
        }


def make_slab_builder(request: Dict[str, Any], slab_ff: Optional[str] = None,
                      pair_substyle: Optional[str] = None,
                      bond_substyle: Optional[str] = None):
    """SlabBuilder from a stored builder request (+ optional FF override)."""
    from . import SLAB_BUILDERS

    name = request["name"]
    if name not in SLAB_BUILDERS:
        raise WorkflowError(f"Unknown substrate builder '{name}'")
    ff = slab_ff or request["slab_ff"]
    overrides = {}
    if ff == request["slab_ff"]:
        # per-role overrides belong to the requested table only
        overrides = dict(
            slab_types=request.get("slab_types"),
            slab_charges=request.get("slab_charges"),
            slab_lj=request.get("slab_lj"),
        )
    return SLAB_BUILDERS[name](
        lx_target=request["lx_target"],
        ly_target=request["ly_target"],
        thickness=request["thickness"],
        oh_density=request["oh_density"],
        hydroxylate_bottom=request["hydroxylate_bottom"],
        slab_ff=ff,
        seed=request["seed"],
        pair_substyle=pair_substyle,
        bond_substyle=bond_substyle,
        **overrides,
    )


def type_builtin_slab(entry: Dict[str, Any], ctx) -> "UnitSpec":
    """
    Stage-2 typer of a built-in slab entry.

    Writes the slab .lt (+ coordinate file) into ctx.output_dir and
    returns its inorganic substrate UnitSpec.  ``ctx`` provides
    ``output_dir``, ``host`` (HostStyles of the film force field) and
    ``inorganic_force_field(entry_id)`` (optional stage-2 override of the
    silica table, e.g. "clayff").
    """
    from ..materials.structure import AtomicStructure
    from ..pipeline.units import UnitSpec
    from ..pipeline.unit_coords import write_xyz

    host = ctx.host
    if host.is_class2:
        raise WorkflowError(
            f"Built-in substrate builders do not support class2 "
            f"force fields ({host.force_field}); build the slab outside "
            "AutoPoly and use SubstrateSpec(lt_file=..., ...) instead"
        )
    if host.is_hybrid_pair and host.pair_substyle is None:
        raise WorkflowError(
            f"Could not find an LJ sub-style in the pair_style line "
            f"of the {host.force_field} force field: '{host.pair_style}'"
        )
    override = ctx.inorganic_force_field(entry["id"])
    builder = make_slab_builder(
        entry["builder"], slab_ff=override,
        pair_substyle=host.pair_substyle,
        bond_substyle=host.bond_substyle,
    )
    structure = AtomicStructure.from_dict(entry["structure"])
    slab = builder.type_structure(structure)
    out = Path(ctx.output_dir)
    slab.write_lt(out)
    coords_file = f"{Path(slab.lt_filename).stem}.xyz"
    write_xyz(out / coords_file, structure.elements, structure.positions,
              comment=f"{slab.class_name} (class-local coordinates)")
    adapter = builder._adapter
    return UnitSpec(
        id=entry["id"],
        kind="inorganic",
        lt_file=slab.lt_filename,
        lt_class=slab.class_name,
        count=1,
        radius=0.0,
        role="substrate",
        monomer_files=[],
        geometry_type="slab",
        motion=entry.get("motion", "frozen"),
        material={
            "id": "sio2",
            "phase": entry.get("phase"),
            "surface": entry.get("surface", "hydroxylated"),
            "component": COMPONENT_KIND,
            "inorganic_ff": adapter.name,
            "slab_ff": builder.slab_ff,
            "ff_version": adapter.version,
            "combining_rule": "host (LAMMPS pair_modify mix)",
            "net_charge": round(slab.assignment.net_charge, 6) + 0.0,
            "n_silanol_top": slab.n_silanol_top,
            "n_silanol_bottom": slab.n_silanol_bottom,
        },
        envelope={
            "shape": "slab",
            "lx": slab.lx,
            "ly": slab.ly,
            "thickness": slab.thickness,
            "z_extent": slab.z_extent,
        },
        mass=structure.total_mass(),
        coords_file=coords_file,
    )


class SlabInfo:
    """Minimal SurfaceSlab view rebuilt from a typed substrate unit."""

    def __init__(self, unit) -> None:
        env = unit.envelope or {}
        self.class_name = unit.class_name
        self.lt_filename = unit.lt_file
        self.lx = float(env["lx"])
        self.ly = float(env["ly"])
        self.thickness = float(env.get("thickness", 0.0))
        self.z_extent = float(env.get("z_extent", 0.0))
