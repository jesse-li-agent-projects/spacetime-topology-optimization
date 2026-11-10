# Fold `stto` and `stto_heat` into variants of `stto_laplace`; pin the hotspot reference

Agreed with the user 2026-11-03, as the project wraps up, and amended 2026-11-09:
`stto_laplace` (Variant 2) is the one space-time problem, `seqopt` stays as its sequential
reference, and both read the hotspot one way only. `stto` and `stto_heat` are not deleted
but move toward variants of `stto_laplace`, not the other way round: their only reason to
stay is the "what changes with a different time-field parameterization" experiment
(`variant2_next_studies.md`, study 3), so they should share everything else with it.
Not finished. The options listed below exist to compare alternatives that are now
settled; dropping them removes the checks on how they combine (e.g. `field_terms.py`'s
`half_plane` needs `half_stencil`, review comment on #231).

## Settled hotspot reading

- **Constant reference:** `half_stencil` normalization, with the thin-layer divisor once
  the angular weight is on (`conductivity.thin_layer_denominator`, one number per
  iteration; PR #233, batch 55).
- **The plate as a solid half-plane** (`conductivity.substrate`), not `infinite_base`'s
  exemption (PR #230, batch 54).

## 1. Make `stto` and `stto_heat` variants of `stto_laplace`

Done (PR "time formulations"): `hotspot_constraint` (`RunConfig`, `HeatRunConfig`) leaves the
hotspot row out of the constraint stack, as `stto_laplace`'s own stack does;
`RunConfig.log_compliance`; `HeatRunConfig.field_move`; `--fixed-density` on all three
CLIs (`sttopt/fixed_density.py`); the S03bT06 settings of the three in
`configs/time_formulations/`.

Remaining:
- Make the Laplace-like settings the only ones for the two: no hotspot row, log
  compliance, the shared stack and objective. `stto` keeps its start-point constraint,
  which a directly optimized `t` needs.
- Share one driver (the step skeleton and CLI loop) across the three, so they differ in
  how `t` is produced and nothing else.
- Decide with the user whether `stto`'s MATLAB-fidelity tests and fixtures
  (`test_e2e*.py`, `test_reference_sweep.py`, `test_robustness.py`,
  `test_resolution_scaling.py`, the MATLAB-loop fixtures, `matlab_reference*.py`) stay;
  they pin the options the comparison turns off.
- `checks.py` and `viz.py` import `stto` (`stto.Problem`, `physical_fields`,
  `constraint_values`); `checks_cli.py` reads a `RunConfig`. Keep them working for all
  three.
- `SpaceTimeConfig` keeps its three subclasses.
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

1 first: it removes the other users of the options (the comparison study should run before
it drops any). 2 and 3 are then independent; do 3 before dropping anything both share.
