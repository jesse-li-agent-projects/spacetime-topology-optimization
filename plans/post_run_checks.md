# Post-run checks (not started)

A run can finish with every row's aggregate looking fine and still break a condition the
design is meant to meet. Example: the single mean start-point row (`793db7b`) sits slightly
elastic (multiplier at `mma_c`), so the base starts at up to `3.5e-6` instead of `<= 1e-9`.
Physically harmless, and it holds per element only because `tPhys >= 0` by construction.
Nothing reported it: it was found by hand while profiling a slowdown.

Idea: a check that runs on a saved design (and at the end of `stto_cli`), reading the
physical fields directly rather than the rows MMA saw, and reporting each condition's
margin, not just pass/fail.

Candidates:
- Start: every base element's `tPhys` within tolerance of 0, and the base is the
  earliest-printed material.
- Final row values: every `g <= tol`, with the elastic `y` of each row reported.
- Hotspot: true max severity against `Tcr` (not the smooth max).
- Volume fraction, grey fraction after projection.
- Connectivity: one solid component; no material printed before anything supports it.
- Solver health: `subsolv` iteration-cap hits.

Open questions: tolerances per check, and whether a failure should warn or fail the run.
