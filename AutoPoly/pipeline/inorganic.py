#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 typing of inorganic components (dispatch by component kind).

Each geometry.json inorganic entry names its component kind
("builtin_slab", "nanoparticle", ...).  The typer registered for that kind
turns the stored AtomicStructure into a typed moltemplate class and an
inorganic UnitSpec, using the film/matrix (host) force-field styles.  The
registry maps kinds to "module:function" references, so the pipeline
never branches on material or component names.

Created on 2026-11-13
@author: zwu
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.exceptions import ValidationError

#: component kind -> "module:function" stage-2 typer.
INORGANIC_TYPERS: Dict[str, str] = {
    "builtin_slab": "AutoPoly.surfaces.component:type_builtin_slab",
    "nanoparticle": "AutoPoly.nanoparticles.typing:type_nanoparticle",
}


def register_inorganic_typer(kind: str, ref: str) -> None:
    """Register a stage-2 typer for a new inorganic component kind."""
    INORGANIC_TYPERS[kind] = ref


@dataclass
class InorganicTypingContext:
    """
    What an inorganic typer needs from stage 2.

    Attributes:
        output_dir: build/<ff>/ directory.
        force_field: Host (organic) force-field name.
        host: HostStyles of the host force field.
        force_field_overrides: component id -> inorganic FF name/alias to
            use instead of the one requested at stage 1.
        allow_unqualified: Permit MD export of combinations that are not
            in the qualification matrix (recorded as user-qualified).
        geometry: The stage-1 geometry (grafted chains, variants).
        variant_rows: Typed atom rows per monomer variant (types,
            charges) -- grafting linkage terms and charge conservation.
        graft_units: filler id -> grafted chain UnitSpecs (in graft order).
        host_ff_lt: Host force-field .lt (bonded classes/coefficients).
    """
    output_dir: Path
    force_field: str
    host: Any
    force_field_overrides: Dict[str, str] = field(default_factory=dict)
    allow_unqualified: bool = False
    geometry: Optional[Dict[str, Any]] = None
    variant_rows: Dict[str, Any] = field(default_factory=dict)
    graft_units: Dict[str, List[Any]] = field(default_factory=dict)
    host_ff_lt: Optional[Path] = None

    def inorganic_force_field(self, component_id: str) -> Optional[str]:
        return self.force_field_overrides.get(component_id)


def type_inorganic_entries(entries: List[Dict[str, Any]],
                           ctx: InorganicTypingContext) -> List[Any]:
    """Type every inorganic entry; returns their UnitSpecs in order."""
    from ..materials.registry import resolve_callable

    unknown = sorted(set(ctx.force_field_overrides)
                     - {e["id"] for e in entries})
    if unknown:
        raise ValidationError(
            f"inorganic_force_fields names unknown component(s) {unknown}; "
            f"components: {[e['id'] for e in entries]}"
        )
    units = []
    for entry in entries:
        kind = entry.get("component")
        if kind not in INORGANIC_TYPERS:
            raise ValidationError(
                f"No stage-2 typer for inorganic component kind '{kind}' "
                f"(entry '{entry.get('id')}'); known: {sorted(INORGANIC_TYPERS)}"
            )
        typer: Callable = resolve_callable(INORGANIC_TYPERS[kind])
        units.append(typer(entry, ctx))
    return units
