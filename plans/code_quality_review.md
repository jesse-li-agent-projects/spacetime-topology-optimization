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

  **Decision: algorithmic correctness wins.** **Blocked on test coverage**, though:
  `tests/matlab_reference_loop.py` hardcodes the same `t = tPhys.copy()`, so this
  breaks the MATLAB-transliteration oracle *and* the `.mat` fixtures at once
  (`test_e2e.py`, `test_robustness.py`'s 1e-12 agreement check, both
  `test_reference_sweep.py` loop tests), leaving `optimize.py` with no correctness
  evidence at all -- see "Test coverage before fixture-breaking changes" below. Write
  items 1 and 2 of that section first.

- [ ] `Problem` (`sttopt/optimize.py:37`) has 30 fields, and `step`
  (`sttopt/optimize.py:260`) unpacks ~20 of them. Worth revisiting whether `Problem`
  should be decomposed into sub-groups (e.g. FEM assembly constants, filter/neighbor
  structures, MMA hyperparameters) that `step` can pass through instead of unpacking
  individually -- but see "Open questions" below before committing to a specific split.

### Array/tensor conventions and naming

- [ ] Audit for Fortran-ordered arrays (MATLAB's native layout) that got carried over
  during the port instead of being converted to NumPy's native C order -- check
  reshape/flatten calls and any explicit `order='F'` for whether row-major would be more
  idiomatic (and possibly faster, since most NumPy/PyTorch ops assume C order) now that
  there's no MATLAB fixture-bit-matching constraint forcing the layout.
- [ ] Audit for one-letter (or otherwise cryptic) variable names inherited from the
  MATLAB source that aren't documented or given a better name where the Python port
  would allow it -- MATLAB math code leans on terse single-letter names matching paper
  notation, but the Python port doesn't need to preserve that convention verbatim if a
  more descriptive name is available without hurting readability against the paper.

### Test coverage before fixture-breaking changes

Both the `init_state` fix above and the PyTorch port below invalidate MATLAB
comparisons, so the suite's non-MATLAB evidence is what has to carry them. An audit
found three evidence tiers: the `.mat` fixtures; the MATLAB-transliteration oracle
(`tests/matlab_reference*.py`, swept by `test_reference_sweep.py`), which is
independent of the `.mat` files but still "matches MATLAB"; and first-principles tests
(closed forms, patch tests, FD gradient checks, convergence under refinement).

The first-principles tier is strong where it exists: `fem.py`, `gravity.py`,
`compliance.py`, `constraints.py` and `filters.density_filter` all survive fixture
removal, and every sensitivity derived *inside* a module is FD-checked independently of
MATLAB (`gravity_compliance`'s `dcx_g`/`dct_g`, `hotspot_constraint`'s `df1`/`dt1`
including the exact-tie case, all four of `constraints.py`'s). Those FD tests are also
the natural acceptance criteria for an autodiff swap. The gaps:

- [ ] FD-check `step`'s *assembled* `df0dx`/`dfdx` w.r.t. the raw `[x; t]` vector, on a
  small grid at a couple of `Theta`/`nStage`/`tfield` points. `optimize.py`'s assembly
  -- `Theta` weighting, the `H @ (dct_g / Hs)` chain rule, constraint row stacking, the
  `m` count -- has no FD check anywhere and is fixture/oracle-only. Highest-value test
  in the repo: it gates the `init_state` fix and is the acceptance test for autodiff.
- [ ] Property test for `init_state` stating the *intended* invariant (raw seed ->
  `tPhys = H @ t / Hs`, leaning on the filter's constant-field fixed point
  `H @ 1 / Hs == 1`). Without it the `init_state` fix has no specification, only
  fixtures it will break.
- [ ] First-principles checks on `estimated_conductivity`/`hotspot_constraint`'s
  *values* (a hand-computable 2-3 element case, the `rouf -> inf` step-function limit,
  monotonicity in build order). The gradients are FD-checked against the code's own
  `fval`, and the only non-fixture golden is an `.npz` frozen from this same Python, so
  a smooth-but-wrong `K_est` currently passes everything.
- [ ] Property test for `filters.continuity_filter` (annihilates constant fields, row
  structure, symmetry) -- outside the fixtures it has only an `issparse()` type check.
- [ ] Fast non-`@slow` end-to-end check on `optimize.run` at small `nelx`/`nely`:
  objective descent, constraints satisfied at termination, mirror-symmetry invariance.
  `run()`'s only physics evidence today is the 180x60x800 `test_e2e_slow.py`
  reproduction, whose `f0val < 195` ceiling is hand-tuned rather than derived.

### Port to PyTorch / CUDA

- [ ] Consider porting `sttopt/` off NumPy/SciPy and onto PyTorch, with CUDA support --
  the FEM assembly and MMA-based optimization loop are dense linear-algebra-heavy and
  currently CPU-only. Needs a decision on scope (whole package vs. hot loop only) and a
  check that SciPy-only functionality in the current implementation (sparse solvers,
  etc.) has a suitable PyTorch/CUDA equivalent before committing.
  **Decision: start on float64** as torch's default dtype, so the suite's existing
  float64-calibrated tolerances (`conftest.assert_close`, `test_reference_sweep`'s
  `TIGHT`/`SOLVED`) carry over unchanged and the port is judged on correctness alone.
  Revisit only if float32/GPU throughput turns out to be the reason for porting.
- [ ] After a PyTorch port, evaluate replacing some of the hand-derived sensitivity
  (`dc`/`dt`/adjoint) code with autodiff, at least for the more mundane/straightforward
  derivative chains -- manually-written sensitivities are a common source of subtle
  correctness bugs (mismatched chain rule terms, stale derivatives after a forward-pass
  change) that autodiff sidesteps. Likely not a full replacement everywhere -- some
  sensitivities may be intentionally hand-optimized or awkward to express in autodiff --
  so this needs a per-case call, not a blanket switch.

## Open questions

- Whether `Problem` decomposition is worth the churn given it's `frozen=True` and
  threaded through `build_problem`/`init_state`/`step`/tests -- needs a concrete
  sub-grouping proposal before starting, not just "it's big."

## Done

- [x] `timefield.py`: `init_timefield`'s `variant` parameter is now `TimeField`, an
  `IntEnum` (`CORNER`/`EDGE`/`OPPOSITE_CORNER`), taken end-to-end through
  `Problem.tfield`/`build_problem`/`run` and the CLI's `--tfield` argument (parses into
  `TimeField` via `type=`/`choices=`) -- resolves the enum item and the open question
  about how far to take it. Existing plain-int call sites keep working since `IntEnum`
  compares equal to `int`.
- [x] `timefield.py`: `nelx < 2` or `nely < 2` (previously `nan` from the corner variants
  or a non-spanning constant field from the edge variant) is now rejected with
  `ValueError` via a shared `_check_grid_size` guard, covering all three variants and
  `init_timefield`.
- [x] `optimize.py`/`conductivity.py`: `hotspot_constraint` now returns `numer`/`K_est`
  directly (as a `HotspotConstraintResult` NamedTuple), so `step`'s `loop % 25 == 0`
  refresh no longer inverts `fval` to recover `numer`, calls `estimated_conductivity`
  separately for `K_est`, or calls `hotspot_constraint` a second time -- it rescales
  `fval`/`df1`/`dt1` exactly instead, since both are linear in `factor`. Verified the
  exact-rescale claim numerically (two `factor` values, random inputs): `fval` matched
  to 0, `df1`/`dt1` to ~1e-16/4e-17 relative.
- [x] `conductivity.py`: `_conductivity_terms` split into a cheap `_conductivity_core`
  (`K_est`/`Nsum3`) and the sensitivity-only extras, so `estimated_conductivity` no
  longer computes `FT_ba`/`DFT_ba`/`S1`/`S2` for nothing.
- [x] `conductivity.py`: `_conductivity_terms` returns a `NamedTuple`
  (`_ConductivityTerms`/`_ConductivityCore`) instead of a positional tuple; call sites
  use attribute access.
- [x] `cli.py`: trimmed the review-history sentence from the module docstring's last
  paragraph, keeping the why-it's-`prev_state`-not-`IterationRecord` reasoning. Audited
  the rest of `sttopt/` for the same pattern (`grep -rniE "caught by|reviewer|previous
  agent|review pass|phase [0-9]"`) and fixed the two dangling "Phase 8 handoff notes"
  references in `optimize.py` (rewritten to state their substance directly, since that
  document doesn't exist in the repo).
