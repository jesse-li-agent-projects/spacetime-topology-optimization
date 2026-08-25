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

### Port to PyTorch / CUDA

- [ ] Consider porting `sttopt/` off NumPy/SciPy and onto PyTorch, with CUDA support --
  the FEM assembly and MMA-based optimization loop are dense linear-algebra-heavy and
  currently CPU-only. Needs a decision on scope (whole package vs. hot loop only) and a
  check that SciPy-only functionality in the current implementation (sparse solvers,
  etc.) has a suitable PyTorch/CUDA equivalent before committing.
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
