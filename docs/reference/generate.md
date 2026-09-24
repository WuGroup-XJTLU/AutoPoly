# generate

One-shot convenience function composing the three pipeline stages
(`GeometryBuilder` → `UnitTyper` → `BoxPacker`). Use the stage classes
directly when you need control between stages (e.g. typing one geometry
under multiple force fields); use `generate()` for single-force-field
one-shot runs.

Beyond the bulk-melt default, `generate()` also accepts:

- `substrate=SubstrateSpec(...)` — build a film on a physical substrate
  slab (auto-selects the `on_substrate` strategy);
- `subtract=[CutAbove(...) / Cylinder(...) / ...]` — whole-instance carve
  regions applied after placement;
- `box_dims=(lx, ly, lz)` — rectangular boxes (per-axis auto-sizing with
  `None`);
- `typer_options={...}` — force-field-specific typing options. Used by
  `force_field="openff"`: `{"offxml": "openff-2.2.1.offxml",
  "charge_method": "nagl"}` (charge backends: `"nagl"` default,
  `"am1bcc"`, `"gasteiger"`).

See the [Substrates & Films guide](../guides/substrates.md) for the full
surface-simulation workflow.

::: AutoPoly.pipeline.workflow
