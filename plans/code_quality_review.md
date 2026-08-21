# Plan: Code quality improvements found during correctness review

> **Status (2026-08-19): open, living list.** This is not a phased implementation plan --
> it's a running list of code-quality issues (not correctness bugs) noticed while
> reviewing `sttopt/` for correctness against `resources/` and the MATLAB source (see
> `CLAUDE.local.md`, "(Transient) Current project state"). Append to it as review of more
> files continues; check items off (or move them to "Done") as they're fixed, ideally one
> per small commit per the repo's commit-size convention.

## Goal

Track cleanup/design items surfaced by manual review that are worth doing but are
independent of any single correctness fix -- so they don't get lost, and so they can be
picked off as small, separately-reviewable commits rather than bundled into whatever
correctness-fix commit happened to be in progress when they were noticed.

## Open items

### `compliance.py` -- analytic ground-truth test coverage

`whole_compliance` now has closed-form checks against elasticity theory (bar-in-tension
patch test, cantilever-beam vs. Timoshenko theory with a mesh-refinement convergence
check -- `tests/test_compliance.py`), independent of the MATLAB fixture. Two more would
close the remaining first-principles gaps:

- [ ] `gravity_compliance`'s self-weight path is untested against ground truth (only
  MATLAB-fixture and FD-self-consistency, same caveat as `conventions.md`'s framing:
  those can't catch a bug shared with the MATLAB source). A self-weight cantilever has a
  standard closed form: tip deflection `delta = w*L^4/(8*E*I)` for a uniformly
  distributed load `w` per unit length (here, self-weight via `gravity.gravity_load_matrix`
  at full density). Same slenderness/convergence-sweep approach as the tip-load
  cantilever test should carry over directly.
- [ ] Nothing exercises the `tPhys`/`time_mask` path against ground truth -- existing
  coverage (fixture + FD) never independently checks *what* a given `tPhys` field should
  produce, only that the Python port's numbers move consistently with themselves/MATLAB.
  Idea: a cantilever built up over deposition time (layer-by-layer through the depth, or
  section-by-section along the length) with a `tPhys` field encoding that build order, at
  a stage time `ti` before the whole beam is complete. The un-built portion should
  contribute ~nothing to stiffness (via `time_mask`'s sigmoid), so compliance at partial
  build should match the closed-form compliance of the *shorter* (or *thinner*) beam that
  is actually built by that stage -- e.g. a half-built cantilever (by length) should
  compliance-match a cantilever of half the length under the same tip load, modulo the
  sigmoid's transition sharpness (`lam`) softening the cutoff. Needs some thought on how
  to keep the sigmoid transition zone from dominating the comparison (either a very sharp
  `lam` at the cost of conditioning, or explicitly modeling the softened transition into
  the closed-form comparison).

### `timefield.py`

- [ ] `init_timefield`'s `variant` parameter is a bare `int` (1/2/3, `ValueError` on
  anything else) instead of an enum. Magic ints duplicated between the docstring and the
  `if/elif` dispatch (`sttopt/timefield.py:49-64`); an `IntEnum` (or plain `Enum`) would
  make call sites self-documenting and give a real type to check instead of prose.
- [ ] `_corner_distance_grid`/`timefield_edge` degenerate when `nelx==1` or `nely==1`
  (found while adding `test_timefield_variants_span_0_to_1`, `tests/test_timefield.py`):
  `nelx=nely=1` divides by `dist.max()==0`, producing `nan`/`nan` with a genuine
  `RuntimeWarning: invalid value encountered in divide`; a lone `nely=1` (or `nelx=1`)
  makes `timefield_opposite_corner` (or `timefield_edge`) never reach the corner/edge it
  normalizes against, so the field is constant `1` (or `0`) instead of spanning `[0, 1]`
  -- a consequence of the same `linspace(0, nel, nel)` endpoint-vs-single-sample behavior
  the module docstring already flags as intentional for `nel>1`. `nelx`/`nely` are always
  well above 1 in every real usage seen so far, so this is likely latent rather than
  live, but it's undocumented and unguarded. Worth a decision: reject `nelx<2`/`nely<2`
  explicitly (`ValueError`), or just document it as an assumption alongside the existing
  `_corner_distance_grid` docstring note.

### `optimize.py` / `conductivity.py`

- [ ] `step` (`sttopt/optimize.py:260`) inverts `fval` to recover `numer` and calls
  `hotspot_constraint` twice on refresh iterations (`sttopt/optimize.py:346-358`) --
  returning `numer`/`K_est` directly from `hotspot_constraint` would delete both the
  inversion and the second call, plus the paragraph of docstring explaining the
  workaround. A docstring paragraph justifying an awkward call pattern is itself a signal
  the interface is the wrong shape.
- [ ] `estimated_conductivity` (`sttopt/conductivity.py:119`) computes `FT_ba`/`DFT_ba`/
  `S1`/`S2` and throws them away -- only `K_est` survives. Either the computation should
  stop building values nothing uses, or (if `hotspot_constraint` genuinely needs to
  redo this work with the extra terms) the two functions should share it instead of
  `estimated_conductivity` duplicating a subset of `_conductivity_terms`'s work for
  nothing.
- [ ] `_conductivity_terms` (`sttopt/conductivity.py:75`) returns a positional 6-tuple.
  Call sites (`estimated_conductivity`, `hotspot_constraint`) have to unpack it
  positionally and remember what each slot means; a small `@dataclass` (or named tuple)
  would make call sites self-documenting and catch transposed-field bugs at write time
  instead of relying on correct unpack order.
- [ ] `Problem` (`sttopt/optimize.py:37`) has 30 fields, and `step`
  (`sttopt/optimize.py:260`) unpacks ~20 of them. Worth revisiting whether `Problem`
  should be decomposed into sub-groups (e.g. FEM assembly constants, filter/neighbor
  structures, MMA hyperparameters) that `step` can pass through instead of unpacking
  individually -- but see "Open questions" below before committing to a specific split.

### `cli.py`

- [ ] The module docstring (`sttopt/cli.py:1-36`) narrates review history in its last
  paragraph: `"Both mismatches were caught by an independent reviewer pass, not by any
  test -- tests/test_cli.py only checks that the PNG exists."` `CLAUDE.local.md`'s style
  section explicitly asks to avoid this kind of unimportant history in docs (contrast:
  history that explains a past *pitfall* to avoid repeating, which is fine to keep, and
  is arguably what the rest of this paragraph already does by explaining *why* `Obj.`/
  `Vol.` read from `prev_state` instead of `IterationRecord`). Trim to keep the
  why-it's-`prev_state`-not-`IterationRecord` reasoning and drop the meta-commentary
  about who caught it and how.
- [ ] Audit other module docstrings for the same pattern (this one was flagged by a
  previous review pass; grep for phrasing like "caught by", "reviewer", "previous
  agent" as a starting point) -- listed here as a to-check, not yet confirmed to
  recur elsewhere.

## Open questions

- Whether `Problem` decomposition is worth the churn given it's `frozen=True` and
  threaded through `build_problem`/`init_state`/`step`/tests -- needs a concrete
  sub-grouping proposal before starting, not just "it's big."
- Whether an `Enum` for `timefield` variant should also replace `Problem.tfield: int`
  (`sttopt/optimize.py:48`), or stay local to `init_timefield`'s parameter -- `tfield` is
  stored on `Problem` and round-trips through the CLI as a plain int argument, so making
  it an enum end-to-end touches more surface than just `timefield.py`.

## Done

(none yet)
