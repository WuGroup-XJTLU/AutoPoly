#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 typing of nanoparticle components (NanoparticleTyper).

Separate from organic typing (UnitTyper/SMARTS): the stored
AtomicStructure is typed by the registered inorganic force-field adapter
selected for the component, never by organic SMARTS rules.  Before any
file is written the exact combination

    material + phase + geometry type + surface + inorganic FF
    + host (polymer) FF + motion

must be in the qualification matrix (or the component must be explicitly
``allow_unqualified``), the motion must be allowed by the adapter (built-in
sets are nonbonded-only: rigid/frozen), and the typed particle must be
charge-neutral.

Output per species: one self-contained moltemplate class
``NP_<id>`` (masses, pair coefficients with the host's hybrid sub-style,
atoms, exclusion-only zero-force bonds), its class-local XYZ coordinates,
and an inorganic "filler" UnitSpec carrying identity, parameterization,
qualification, envelope and loading descriptors.

Created on 2026-11-13
@author: zwu
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..core.exceptions import ValidationError
from ..core.system import logger
from ..exporters.moltemplate import (
    mean_bond_lengths,
    role_labels,
    write_inorganic_lt,
)
from ..materials.forcefields import get_inorganic_ff
from ..materials.qualification import require_md_ready
from ..materials.registry import get_material
from ..materials.structure import AtomicStructure

#: Max |net charge| (e) of a typed nanoparticle.
NET_CHARGE_TOLERANCE = 1e-4


def nanoparticle_class_name(component_id: str) -> str:
    return f"NP_{component_id}"


def type_nanoparticle(entry: Dict[str, Any], ctx):
    """Stage-2 typer of one nanoparticle species (see module docstring)."""
    from ..pipeline.unit_coords import write_xyz
    from ..pipeline.units import UnitSpec

    spec = get_material(entry["material"])
    structure = AtomicStructure.from_dict(entry["structure"])
    host = ctx.host
    if host.is_class2:
        raise ValidationError(
            f"Nanoparticles cannot be combined with class2 host force fields "
            f"({host.force_field}): no validated inorganic cross terms"
        )
    if host.lj_style is None:
        raise ValidationError(
            f"Host force field '{host.force_field}' pair style "
            f"'{host.pair_style}' has no 12-6 LJ (sub-)style for explicit "
            "inorganic cross interactions"
        )

    ff_name = ctx.inorganic_force_field(entry["id"]) or \
        (entry.get("force_field") or {}).get("name")
    if ff_name is None:
        raise ValidationError(
            f"Material '{spec.id}' has no validated inorganic force field "
            f"(component '{entry['id']}'): it is geometry-only. Export the "
            "structure (Nanoparticle.write_structure) or register a "
            "validated adapter before MD preparation."
        )
    adapter = get_inorganic_ff(ff_name, spec.id)
    motion = entry["motion"]
    if motion not in adapter.motions:
        raise ValidationError(
            f"Inorganic force field '{adapter.name}' supports motions "
            f"{list(adapter.motions)}, not '{motion}' (component "
            f"'{entry['id']}'): a flexible particle needs an independently "
            "validated bonded inorganic model"
        )

    grafts = entry.get("grafts")
    linker = grafts["linker"] if grafts else None
    if grafts:
        from .grafting import LINKERS
        if ctx.force_field not in LINKERS[linker]:
            raise ValidationError(
                f"Grafted particles with the '{linker}' linker need host "
                f"force field(s) {list(LINKERS[linker])} (the linkage bonded "
                f"terms come from them), got '{ctx.force_field}'"
            )
    allow = bool(entry.get("allow_unqualified")) or ctx.allow_unqualified
    try:
        qual = require_md_ready(
            spec.id, entry.get("phase"), entry["geometry_type"],
            entry["surface"], adapter.name, ctx.force_field, motion,
            linker=linker)
        qualification = {"id": qual.id, "status": "qualified",
                         "cross_rule": qual.cross_rule,
                         "evidence": list(qual.evidence)}
        cross_rule = qual.cross_rule
    except ValidationError:
        if not allow:
            raise
        qualification = {"id": None, "status": "user-qualified",
                         "cross_rule": adapter.combining_rule,
                         "evidence": []}
        cross_rule = adapter.combining_rule
        logger.warning(
            f"Nanoparticle '{entry['id']}': combination is NOT in the "
            "qualification matrix; proceeding because allow_unqualified=True "
            "(recorded as user-qualified)"
        )

    chem = spec.chemistry_module()
    roles = structure.roles or chem.assign_roles(structure.elements,
                                                 structure.bonds)
    assignment = adapter.assign(roles, notes=[f"component={entry['id']}"])
    graft_info = None
    unit_net = assignment.net_charge
    if grafts:
        graft_info = _graft_info(entry, ctx)
        # charge conservation: the grafted O also carries the charge of the
        # organic H the condensation removed
        for a in grafts["anchors"]:
            assignment.charges[a["o"]] += graft_info["cap_charge"]
        total = assignment.net_charge + sum(graft_info["chain_charges"])
        unit_net = total
        if abs(total) > NET_CHARGE_TOLERANCE * max(1, len(grafts["anchors"])):
            raise ValidationError(
                f"Grafted particle '{entry['id']}' has net charge "
                f"{total:+.5f} e (core + grafts)"
            )
    elif abs(assignment.net_charge) > NET_CHARGE_TOLERANCE:
        raise ValidationError(
            f"Nanoparticle '{entry['id']}' typed with '{adapter.name}' has "
            f"net charge {assignment.net_charge:+.4f} e; charged particles "
            "(and counter-ions) are not supported -- check the surface "
            "termination"
        )

    class_name = nanoparticle_class_name(entry["id"])
    lt_file = f"{class_name}.lt"
    labels = role_labels(roles, "np")
    prefix = class_name.lower()
    bond_types = [
        f"{prefix}_{chem.role_bond_type(roles[i], roles[j])}"
        for i, j in structure.bonds
    ]
    order = sorted(set(bond_types))
    r0 = mean_bond_lengths(structure.positions, structure.bonds,
                           bond_types, order)
    d = structure.descriptors
    env = dict(d.get("envelope") or {})
    center = np.asarray(env.get("center", [0.0, 0.0, 0.0]), dtype=float)
    positions = structure.positions - center
    env["center"] = [0.0, 0.0, 0.0]
    env["radius"] = float(np.linalg.norm(positions, axis=1).max())
    header = [
        f"nanoparticle '{entry['id']}' built by AutoPoly: {spec.id} "
        f"({entry.get('phase')}), {structure.geometry_type}, "
        f"{structure.n_atoms} atoms ({structure.formula()})",
        f"surface '{entry['surface']}'; inorganic FF {adapter.name} "
        f"v{adapter.version}; motion '{motion}'",
        "self-contained: masses + pair coefficients; bonds carry zero force "
        "constants (exclusions only) -- hold the particle rigid or frozen",
    ]
    out = Path(ctx.output_dir)
    write_inorganic_lt(
        out / lt_file,
        class_name=class_name,
        labels=labels,
        type_names=assignment.type_names,
        charges=assignment.charges,
        positions=positions,
        type_masses=assignment.type_masses,
        type_lj=assignment.type_lj,
        header=header,
        bonds=structure.bonds,
        bond_types=bond_types,
        bond_coeffs={t: (0.0, r0[t]) for t in order},
        pair_substyle=host.pair_substyle,
        bond_substyle=host.bond_substyle,
    )
    coords_file = f"{class_name}.xyz"
    write_xyz(out / coords_file, structure.elements, positions,
              comment=f"{class_name} (class-local coordinates)")
    unit_lt, unit_class, unit_coords = lt_file, class_name, coords_file
    unit_mass = structure.total_mass()
    n_core = structure.n_atoms
    graft_record = None
    if grafts:
        unit_class = f"G{class_name}"
        unit_lt = f"{unit_class}.lt"
        graft_record, g_el, g_xyz, g_mass = _write_grafted_composite(
            out, unit_class, class_name, lt_file, labels, roles, structure,
            positions, entry, graft_info, ctx, center=center)
        unit_coords = f"{unit_class}.xyz"
        write_xyz(out / unit_coords,
                  list(structure.elements) + g_el,
                  np.vstack([positions, g_xyz]),
                  comment=f"{unit_class}: core ({n_core} atoms) + grafts")
        unit_mass += g_mass
        allpos = np.vstack([positions, g_xyz])
        env["radius"] = float(np.linalg.norm(allpos, axis=1).max())
    # surface atoms (for grafting/analysis) by index
    surface_idx = [i for i, f in enumerate(structure.surface or []) if f]
    logger.info(
        f"Typed nanoparticle '{entry['id']}' ({adapter.name}, "
        f"{structure.n_atoms} atoms, net charge "
        f"{assignment.net_charge:+.2e} e) -> {lt_file}"
    )
    return UnitSpec(
        id=entry["id"],
        kind="inorganic",
        lt_file=unit_lt,
        lt_class=unit_class,
        count=int(entry["count"]),
        radius=env["radius"],
        role="filler",
        geometry_type=structure.geometry_type,
        motion=motion,
        material={
            "id": spec.id,
            "phase": entry.get("phase"),
            "surface": entry["surface"],
            "surface_params": entry.get("surface_params", {}),
            "component": "nanoparticle",
            "inorganic_ff": adapter.name,
            "ff_version": adapter.version,
            "ff_citations": list(adapter.citations),
            "combining_rule": adapter.combining_rule,
            "cross_rule": cross_rule,
            "qualification": qualification,
            "net_charge": round(unit_net, 8) + 0.0,
            "core_net_charge": round(assignment.net_charge, 8) + 0.0,
            "formula": structure.formula(),
            "n_atoms": structure.n_atoms,
            "types": assignment.type_table(),
            "metadata": structure.metadata,
            "n_surface_atoms": len(surface_idx),
            "grafts": graft_record,
            "core_class": class_name,
        },
        envelope={
            **env,
            "core_volume": d.get("core_volume"),
            "core_radius_equivalent": d.get("core_radius_equivalent"),
            "nominal_radius": d.get("nominal_radius"),
            "accessible_area_nm2": d.get("accessible_area_nm2"),
            "extent": [float(x) for x in np.ptp(positions, axis=0)],
            "n_core_atoms": n_core,
            "core_mass": structure.total_mass(),
        },
        mass=unit_mass,
        coords_file=unit_coords,
        monomer_files=[lt_file] if grafts else [],
    )


# ----------------------------------------------------------------------
# Grafted composites
# ----------------------------------------------------------------------
def _graft_info(entry, ctx) -> Dict[str, Any]:
    """Cap charge, graft units and per-chain charges from stage-2 typing."""
    grafts = entry["grafts"]
    units = ctx.graft_units.get(entry["id"], [])
    chains = {c["id"]: c for c in ctx.geometry["chains"]}
    by_chain = {u.anchors.get("chain_id"): u for u in units}
    ordered = [by_chain.get(cid) for cid in grafts["chain_ids"]]
    if None in ordered or len(ordered) != grafts["n"]:
        raise ValidationError(
            f"Graft chains of '{entry['id']}' were not typed as graft units"
        )
    cap = grafts["cap_atom"]
    chain_charges, cap_charge = [], None
    for cid in grafts["chain_ids"]:
        q = 0.0
        for k, p in enumerate(chains[cid]["placements"]):
            rows = ctx.variant_rows[p["variant"]]
            for i, row in enumerate(rows):
                if k == 0 and i == cap:
                    cap_charge = float(row["charge"])
                    continue
                q += float(row["charge"])
        chain_charges.append(q)
    return {"units": ordered, "cap_charge": cap_charge,
            "chain_charges": chain_charges, "chains": chains}


def _write_grafted_composite(out, unit_class, core_class, core_lt, labels,
                             roles, structure, positions, entry, info, ctx,
                             center=np.zeros(3)):
    """
    Write G<NP class>.lt (core + anchored graft chains + explicit OPLS-AA
    linkage terms); returns (record, graft elements, graft coords, mass).
    """
    from ..packing.chain_regrowth import ChainTemplate
    from ..pipeline.unit_coords import chain_coordinates, chain_template_data
    from .grafting import opls_class, opls_coefficients

    grafts = entry["grafts"]
    ff_lt = ctx.host_ff_lt
    head, cap = grafts["head_atom"], grafts["cap_atom"]
    variants = ctx.geometry["variants"]
    prefix = unit_class.lower()
    coeff = {"bond": {}, "angle": {}, "dihedral": {}}
    terms = {"bond": [], "angle": [], "dihedral": []}
    omitted = set()

    def lookup(kind, classes, key):
        tokens = opls_coefficients(kind, classes, ff_lt)
        if tokens is None:
            raise ValidationError(
                f"OPLS-AA defines no {kind} coefficient for "
                f"{'-'.join(classes)} (graft linkage of '{entry['id']}'); "
                "grafting this chain chemistry is not supported"
            )
        name = f"{prefix}_{key}"
        coeff[kind][name] = tokens
        return name

    g_el, g_xyz = [], []
    mass = 0.0
    from ..materials.structure import element_mass
    for g, (anchor, unit) in enumerate(zip(grafts["anchors"],
                                           info["units"])):
        chain = info["chains"][grafts["chain_ids"][g]]
        tpl = ChainTemplate(chain_template_data(unit.id, chain, variants))
        offs = tpl.offsets()
        el, xyz = chain_coordinates(chain["placements"], variants)
        keep = np.ones(len(el), dtype=bool)
        keep[offs[0] + cap] = False
        g_el += [e for e, k in zip(el, keep) if k]
        g_xyz.append(xyz[keep])
        mass += sum(element_mass(e) for e, k in zip(el, keep) if k)

        # chain-level graph (cap H removed), atom ids "monomer[k]/E<i+1>"
        def ref(global_idx):
            k = max(m for m in range(tpl.n) if offs[m] <= global_idx)
            i = global_idx - offs[k]
            e = tpl.info(k)["elements"][i]
            return f"$atom:graft[{g}]/monomer[{k}]/{e}{i + 1}", k, i
        nbrs: Dict[int, List[int]] = {}
        for a, b in tpl.chain_bonds():
            if offs[0] + cap in (a, b):
                continue
            nbrs.setdefault(a, []).append(b)
            nbrs.setdefault(b, []).append(a)

        def atom_type(k, i):
            return ctx.variant_rows[tpl.monomers[k]][i]["atom_type"]

        o_ref = f"$atom:core/{labels[anchor['o']]}"
        si_ref = f"$atom:core/{labels[anchor['si']]}"
        h_global = offs[0] + head
        h_ref, hk, hi = ref(h_global)
        h_type = atom_type(hk, hi)
        bname = lookup("bond", [opls_class(h_type, "b", ff_lt), "OS"],
                       f"{opls_class(h_type, 'b', ff_lt).lower()}_os")
        terms["bond"].append((bname, [o_ref, h_ref]))
        aname = lookup("angle", [opls_class(h_type, "a", ff_lt), "OS", "Si"],
                       f"{opls_class(h_type, 'a', ff_lt).lower()}_os_si")
        terms["angle"].append((aname, [si_ref, o_ref, h_ref]))
        for x in nbrs.get(h_global, []):
            x_ref, xk, xi = ref(x)
            x_type = atom_type(xk, xi)
            ca = [opls_class(x_type, "a", ff_lt),
                  opls_class(h_type, "a", ff_lt), "OS"]
            aname = lookup("angle", ca, "_".join(c.lower() for c in ca))
            terms["angle"].append((aname, [x_ref, h_ref, o_ref]))
            for y in nbrs.get(x, []):
                if y == h_global:
                    continue
                y_ref, yk, yi = ref(y)
                y_type = atom_type(yk, yi)
                cd = [opls_class(y_type, "d", ff_lt),
                      opls_class(x_type, "d", ff_lt),
                      opls_class(h_type, "d", ff_lt), "OS"]
                dname = lookup("dihedral", cd,
                               "_".join(c.lower() for c in cd))
                terms["dihedral"].append((dname, [y_ref, x_ref, h_ref,
                                                  o_ref]))
            omitted.add(f"Si-OS-{opls_class(h_type, 'd', ff_lt)}-"
                        f"{opls_class(x_type, 'd', ff_lt)}")

    lines = [
        f"# grafted nanoparticle '{entry['id']}' built by AutoPoly: core "
        f"{core_class} + {grafts['n']} chains ({grafts['linker']} linkage)",
        "# linkage bonded terms: OPLS-AA coefficients by bonded class; "
        "Si-O-C-X torsions omitted (undefined in OPLS-AA for silica)",
        f'import "{core_lt}"',
    ]
    for u in info["units"]:
        lines.append(f'import "{u.lt_file}"')
    lines += ["", f"{unit_class} {{", "", f"  core = new {core_class}"]
    # grafts were grown in the structure frame; the core class is centered
    shift = "" if not np.any(center) else \
        f".move({-center[0]:.6f},{-center[1]:.6f},{-center[2]:.6f})"
    for g, u in enumerate(info["units"]):
        lines.append(f"  graft[{g}] = new {u.class_name}{shift}")
    lines += ["", '  write_once("In Settings") {']
    for kind in ("bond", "angle", "dihedral"):
        for name, tokens in coeff[kind].items():
            lines.append(f"    {kind}_coeff @{kind}:{name} "
                         + " ".join(tokens))
    lines.append("  }")
    for kind, section, tag in (("bond", "Data Bonds", "b"),
                               ("angle", "Data Angles", "a"),
                               ("dihedral", "Data Dihedrals", "d")):
        if not terms[kind]:
            continue
        lines += ["", f'  write("{section}") {{']
        for n, (name, refs) in enumerate(terms[kind], start=1):
            lines.append(f"    ${kind}:g{tag}{n} @{kind}:{name} "
                         + " ".join(refs))
        lines.append("  }")
    lines += ["", f"}} # {unit_class}", ""]
    (Path(out) / f"{unit_class}.lt").write_text("\n".join(lines))
    record = {
        "linker": grafts["linker"],
        "n": grafts["n"],
        "density_target": grafts.get("density_target"),
        "density_achieved": grafts.get("density_achieved"),
        "chain_units": [u.id for u in info["units"]],
        "cap_charge_transferred": info["cap_charge"],
        "linkage_coefficients": {k: dict(v) for k, v in coeff.items()},
        "omitted_terms": sorted(omitted),
        "graft_mass": mass,
    }
    return record, g_el, np.vstack(g_xyz) - center, mass
