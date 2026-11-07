# Retire `stto` and `stto_heat`; pin the hotspot reference

Agreed with the user 2026-11-03, as the project wraps up: `stto_laplace` (Variant 2) is
the one space-time problem kept, `seqopt` stays as its sequential reference, and both read
the hotspot one way only. Not started. The options listed below exist to compare
alternatives that are now settled; dropping them removes the checks on how they combine
(e.g. `field_terms.py`'s `half_plane` needs `half_stencil`, review comment on #231).

## Settled hotspot reading

- **Constant reference:** `half_stencil` normalization, with the thin-layer divisor once
  the angular weight is on (`conductivity.thin_layer_denominator`, one number per
  iteration; PR #233, batch 55).
- **The plate as a solid half-plane** (`conductivity.substrate`), not `infinite_base`'s
  exemption (PR #230, batch 54).

## 1. Retire `stto` and `stto_heat`

Remove `stto.py`, `stto_cli.py`, `stto_heat.py`, `stto_heat_cli.py`, `RunConfig`,
`HeatRunConfig`, `configs/default.json`, `heat_default.json`, `continuation.md`, the
`sttopt` script in `pyproject.toml`, and their tests (`test_stto*.py` other than
Laplace, `test_e2e*.py`, `test_reference_sweep.py`, `test_robustness.py`,
`test_resolution_scaling.py`, and the MATLAB-loop fixtures).

Before deleting, untangle what `stto_laplace` and `seqopt` still borrow:
- `checks.py` and `viz.py` import `stto` (`stto.Problem`, `physical_fields`,
  `constraint_values`); `checks_cli.py` reads a `RunConfig`. Drop the `stto`-only checks,
  or point `checks_cli` at Laplace runs if the user still wants it.
- `compliance`, `constraints`, `gravity`, `filters`, `mma`, `fem` and the torch solvers
  are shared. Keep each with the tests that exercise its shared use, and drop the
  branches only `stto` reached.
- `SpaceTimeConfig` has one subclass left; fold it into `LaplaceRunConfig`.
- The package docstring, `conventions.md` and `conductivity_estimation_2d/structure.md`
  describe `stto` as the main problem.

`stto`'s open plans (`nucleation.md`, `length_scale_options.md`) are archived as closed.

## 2. Pin `stto_laplace` to the settled reading

Drop `hotspot_normalization`, `hotspot_plate` and `hotspot_reference` from
`LaplaceRunConfig`, and with them `Normalization.NEIGHBORHOOD`, `Plate`, `Reference`,
`directed_denominator`, and `infinite_base` (also `viz`'s `K_est == inf` handling) once
`seqopt` no longer needs them. Saved configs carrying the dropped keys still load:
`from_dict` warns and drops unknown keys. Rerun one validation cell to confirm nothing
moved.

## 3. Bring `seqopt` onto the settled reading

`seqopt` today uses the exemption and, with its angular weight on (`hotspot_kappa`
2.37), the per-element divisor. Give it the substrate and the thin-layer divisor, then
drop its `hotspot_normalization`.

Match the rest of its hotspot to `laplace_default.json` too, for now (user, 2026-11-03):
`q` 2 (not 3), `cooling` credit (not `sigmoid`), `capped_ratio` shielding (not
`absolute`), and the shaping term. `seqopt` has no fields for the last three yet.
Reading the hotspot this way changes its results, so rerun what Variant 2's sequential
reference relies on (`variant2_next_studies.md`).

## Order

1 first: it removes the other users of the options. 2 and 3 are then independent; do 3
before dropping anything both share.
