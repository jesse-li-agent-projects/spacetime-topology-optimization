# Post-run checks

A run can finish with every constraint's aggregate looking fine and still break a
condition the design is meant to meet. Example: the single mean start-point constraint
(`793db7b`) sits slightly elastic (multiplier at `mma_c`), so the base starts at up to
`3.5e-6` instead of `<= 1e-9`. Physically harmless, and it holds per element only
because `tPhys >= 0` by construction. Nothing reported it: it was found by hand while
profiling a slowdown.

A check that runs on a saved design (and at the end of `stto_cli`), and reports each
condition's margin, not just pass/fail. `stto` only; `seqopt` already starts from a
binarized, connected geometry.

## Decisions

- **The binarized design is what is checked**, since that is what gets printed:
  `physical_fields` at infinite projection sharpness, i.e. the filtered density
  thresholded at `eta` (not `xPhys` at 0.5, which differs from it for `eta != 0.5`).
- **Two hard checks.** Everything else is reported, not judged.
  - Start: every solid base element starts within `1e-5` of `t = 0`.
  - Support: every solid element off the base has an 8-connected solid neighbor
    printed strictly before it. This also covers connectivity: following earlier
    neighbors from any solid element ends on the base.
- **Reported on the binarized design:** every constraint row at the run's final
  settings (their smooth maxima are calibrated onto the true maximum, so the hotspot
  row is the true max severity against `Tcr`), compliance, volume fraction, and the
  grey fraction of the continuous design. The continuous design's constraint values are
  already in `iterations.jsonl`.
- **A failed hard check warns; it does not fail the run.** The design is saved either
  way. The report is written as `checks.json` beside the design so a sweep can read it.
- `subsolv` cap hits are a property of the run, not the design, and are left to
  `iterations.jsonl`.
