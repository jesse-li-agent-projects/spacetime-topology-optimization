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
check -- `tests/test_compliance.py`), independent of the MATLAB fixture. So does
`gravity_compliance`, closing the remaining first-principles gaps:

- [x] `gravity_compliance`'s self-weight path is now checked against a closed-form
  self-weight cantilever (`test_gravity_compliance_self_weight_cantilever`, plus an
  exact `Emax`-scaling check), Euler-Bernoulli only (no Timoshenko term -- a flat
  tolerance at L/H == 10 rather than a per-resolution shrinking one, since the
  uncorrected shear contribution doesn't vanish under mesh refinement).
- [x] The `tPhys`/`time_mask` path is now checked
  (`test_gravity_compliance_partial_build_matches_truncated_mesh`): a cantilever built
  up column-by-column along its length, stopped partway through the build, compared
  against an actual shorter mesh built to full density -- two independent FEM solves
  rather than a second closed form layered on top of beam theory, using a sharp `lam`
  and rescaling the truncated mesh's load to sidestep the sigmoid-softening question.

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

- [ ] `init_state` (`sttopt/optimize.py:226`) sets `t = tPhys.copy()` (raw == physical,
  unfiltered) instead of filtering, matching the MATLAB source's identical `t = tPhys`
  at init (e.g. `Space_Time_TopOpt_Robot.m:178-180`). But every `step()` iteration
  treats `tPhys` as *defined* by `tPhys = H @ t / Hs` (`sttopt/optimize.py:444`), and
  the sensitivity chain rule `dt = ... H @ (dct_g / Hs)` (`sttopt/optimize.py:321`) is
  only the correct gradient w.r.t. raw `t` under that definition. Since `init_timefield`
  is generally non-uniform (nonlinear for the corner-distance variants, `tfield=1`/`3`),
  `filter(t) != t` there, so iteration 1's `dt` doesn't reflect the actual (identity)
  forward map used to produce `state.tPhys` at init -- forward and backward pass
  disagree about what function was evaluated. Contrast `xTilde = x` at init
  (`sttopt/optimize.py:235`): also unfiltered by direct assignment, but numerically
  *equal* to `H @ x / Hs` anyway since `x` is spatially uniform (`volfrac`) and the
  filter preserves constants exactly -- no actual inconsistency there. This looks like a
  genuine (if minor, and iteration-1-only) bug in the reference algorithm rather than a
  deliberate design choice. Likely fix: seed `t = init_timefield(...)` as the raw
  variable and derive `tPhys = H @ t / Hs` at init too (mirroring `xPhys`'s treatment of
  `xTilde`) -- but this breaks exact bit-matching against `generate_fixtures.m`'s init,
  so needs a decision on whether fixture fidelity or algorithmic correctness wins here.
  Fixing it would also remove the asymmetry that currently justifies `xTilde`/`t` being
  threaded/commented as two different "unfiltered at init" special cases. `xTilde`
  should likely get the same treatment for consistency even though it's currently a
  numeric no-op (`x` uniform at init) -- init both `x`/`t` as the raw seed and derive
  `xTilde`/`tPhys` via `H @ (...) / Hs` uniformly, rather than special-casing which
  fields get filtered at init and which don't.

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
