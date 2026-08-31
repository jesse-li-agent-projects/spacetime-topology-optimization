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

### `optimize.py` / `conductivity.py`

- [x] **Done (PR #26).** `init_state` (`sttopt/optimize.py:226`) sets `t = tPhys.copy()` (raw == physical,
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

  **Decision: algorithmic correctness won.** Both blocking test-coverage items landed
  first (commit `e330344`), then the fix itself in PR #26: `init_state` now seeds `x`/`t`
  as the raw fields and derives `xTilde`/`tPhys` through the filter uniformly, exactly as
  `step` does. Resolution of the two fallout questions:

  * The **transliteration oracle follows the fix** (`tests/matlab_reference_loop.py`) --
    it is live code under our control, and an oracle that reproduces a known bug is not
    an oracle. Both `test_reference_sweep.py` loop tests still pass at the tight
    tolerance, so the fixed `optimize.py` and the separately-edited oracle agree.
  * The **`.mat` fixtures could not follow it**: they are frozen artifacts, and
    regenerating them needs MATLAB, which is currently broken in this environment
    (R2026a exits 1 with no output on `disp('hello')`). Rather than weaken or delete
    them, the tests that compare against them now *enter the trajectory at the
    fixtures' own starting state* via `conftest.matlab_init_state`, using the new
    `optimize.run_from_state`. They therefore keep checking exactly what they were
    written to check -- `step`'s per-iteration agreement with MATLAB -- while
    `init_state`'s own behaviour is specified by `test_optimize.py`'s three init
    invariant tests. `generate_fixtures.m:200-203` still contains the reference's
    unfiltered init and was deliberately left alone; regenerating it is optional
    follow-up, not a prerequisite.

  Note one small correction to the analysis above: the density half is *not* bit-for-bit
  unchanged by filtering at init, as originally claimed -- `H @ x / Hs` differs from
  uniform `x` by ~1.7e-16 (one ulp). Far inside every tolerance in the suite, but the
  claim is "equal to rounding", not "identical".

- [ ] **Deferred indefinitely, not being tackled soon.** `Problem`
  (`sttopt/optimize.py:37`) has 30 fields, and `step` (`sttopt/optimize.py:260`) unpacks
  ~20 of them. Worth revisiting whether `Problem` should be decomposed into sub-groups
  (e.g. FEM assembly constants, filter/neighbor structures, MMA hyperparameters) that
  `step` can pass through instead of unpacking individually -- but see "Open questions"
  below before committing to a specific split.

### Array/tensor conventions and naming

- [x] **Done.** Switched the element-enumeration convention from Fortran order
  (column-major, matching the MATLAB source) to NumPy's native C order (row-major):
  every `.flatten()`/`.reshape()` mirroring a grid-shaped `(nely, nelx)` field across
  `fem.py`, `filters.py`, `gravity.py`, `compliance.py`, `constraints.py`,
  `conductivity.py`, `optimize.py`, `cli.py` now uses the default order, and
  `conventions.md`'s "Array order" section documents the new `e // nelx, e % nelx`
  mapping. Node/dof numbering (`fem.element_dof_map`'s and `gravity.py`'s `nodenrs`)
  is a separate, deliberately-unchanged internal labeling scheme -- only *element*
  enumeration flipped, so `fixeddofs`/load-vector indices in `optimize.py` needed no
  change (confirmed by `test_assemble_and_solve` and the full FEM patch-test suite).
  One real bug the audit caught along the way: `optimize.build_problem`'s `Nei`
  (`start_point`'s print-origin elements) hardcoded the old column-major "first
  column" formula (`np.arange(nely)`) directly, not via a flatten/reshape call, and
  needed `np.arange(nely) * nelx` under the new convention.

  The live MATLAB-transliteration oracle (`tests/matlab_reference*.py`) and every
  from-scratch "independent reference" helper inside the test suite got the same
  flip, since they're code under our control, not frozen artifacts (same precedent as
  the `init_state` fix above). Only genuinely frozen `.mat`/`.npz` fixture data needed
  a real adapter: `tests/conftest.py` gained `fixture_element_perm`/
  `reindex_fixture`/`reindex_fixture_values`/`reindex_fixture_halves`, mapping a
  fixture's F-order element indices to the new C-order ones, used at every fixture
  comparison that's element-indexed (`edofMat` rows, `H`/`L`'s two element axes,
  `gravity_load_matrix`'s element columns, neighbor-pair COO indices, and the raw
  `[x; t]` MMA-variable halves of `df0dx`/`dfdx`/`xmma`/`low`/`upp`). Fixture-derived
  *grid*-shaped `(nely, nelx)` arrays (`xPhys_traj`, `tPhys_traj`, etc.) needed no
  reindexing -- a grid's physical layout doesn't depend on the flatten convention.
  All 327 non-slow tests pass; `test_e2e_slow.py` was audited for the same patterns
  and found clean (no `order='F'`, no hardcoded element-index formulas).
- [x] **Mostly done.** Audited `sttopt/` for one-letter/cryptic variable names inherited
  from the MATLAB source. Fixed: `optimize.py`'s `p = problem` local alias (collided with
  `Problem.p`, the hotspot p-norm exponent) -> `prob`; `xnew`/`s_new` -> `xt_new`/`x_new`
  to match the module's own `x`/`t` convention (`xt_new` folded away entirely by the
  later C-order rewrite of the split, which slices `xmma` directly). `cli.py`'s `T1` ->
  `hotspot_severity`, traced to MATLAB's `T1=(1-B).*XPhys` -- confirmed *not* a print
  time despite feeding `draw_combination1`'s `timing` parameter, but the same quantity as
  `hotspot_constraint`'s internal `T_val = 1 - K_est`, reused there only to piggyback on
  the generic per-element coloring plot; `viz.combination_plot`'s `tPhys` parameter
  renamed to `values` accordingly. `constraints.py`: `kk`/`A` in `time_field_continuity`
  -> `smoothness_weight`/`deviation` (the MATLAB source already comments `kk` as
  "controlling the smoothness of the time field" -- a tuning weight, not a paper symbol);
  `ss` in `start_point` kept (terse selector-matrix name reads fine against the math) but
  given an inline comment. `ft` (MATLAB's function name reused as a variable) ->
  `t_mask` in both `compliance.py`/`gravity_compliance` and
  `constraints.py`/`stage_volume_bounds`. Left alone on review: `fem.py`'s
  `A11/A12/B11/B12`/`iK/jK/sK` and `gravity.py`'s `I/J/S` (standard FEM/COO-triplet
  conventions, already commented); all of `mma.py` (deliberate verbatim port of
  Svanberg's reference code, per its own module docstring); `conductivity.py`'s
  `_pairwise_sigmoid_terms(a, b, ...)` vs. call sites' `e1`/`e2` (minor def/call-site
  naming mismatch, not worth churn); `gravity.py`'s return value documented as `C` in its
  own docstring but never named that internally (doc/code mismatch, not a bug).
  **Resolved (2026-08-25):** confirmed against Wang et al. 2019 (the actual STTO source
  paper, `resources/Wang2019_Space-Time-TO-Additive-Manufacturing.pdf`) that this
  sigmoid-sharpness parameter is their `beta_t` -- `rou`/`lamda` in the MATLAB source is
  just a caller/callee copy-paste split, not two different quantities (and unrelated to
  `rho`, which that paper reserves for the density field). Renamed throughout: generic
  sigmoid functions (`compliance.time_mask`/`time_mask_derivative`) take `beta`; the
  field-specific callers/state (`gravity_compliance`, `stage_volume_bounds`,
  `OptimizerState`) use `beta_t`, matching the existing `beta_d` (formerly `beta`) for the
  density-projection sharpness that was already living alongside it in `OptimizerState`/
  `Problem`. `Problem.rouf` (the unrelated hotspot/conductivity-selection sharpness, Das
  2023's zeta) was deliberately left alone -- different symbol, different meaning, despite
  the similar name.

### Test coverage before fixture-breaking changes

> **Status (2026-08-25): mostly done.** Both the `init_state` fix above and the PyTorch
> port below invalidate MATLAB comparisons, so the suite's non-MATLAB evidence is what
> has to carry them. An audit found three evidence tiers: the `.mat` fixtures; the
> MATLAB-transliteration oracle (`tests/matlab_reference*.py`, swept by
> `test_reference_sweep.py`), which is independent of the `.mat` files but still
> "matches MATLAB"; and first-principles tests (closed forms, patch tests, FD gradient
> checks, convergence under refinement).

The first-principles tier is strong where it exists: `fem.py`, `gravity.py`,
`compliance.py`, `constraints.py` and `filters.density_filter` all survive fixture
removal, and every sensitivity derived *inside* a module is FD-checked independently of
MATLAB (`gravity_compliance`'s `dcx_g`/`dct_g`, `hotspot_constraint`'s `df1`/`dt1`
including the exact-tie case, all four of `constraints.py`'s). Those FD tests are also
the natural acceptance criteria for an autodiff swap.

- [x] FD-check `step`'s *assembled* `df0dx`/`dfdx` w.r.t. the raw `[x; t]` vector --
  `test_optimize.py`'s `test_step_...` FD test (around line 252) checks the stacked
  `df0dx`/`dfdx` against central differences of `step`'s own `f0val`/`fval` across
  `Theta`/`nStage`/`tfield` points.
- [x] Property test for `init_state` stating the *intended* invariant (raw seed ->
  `tPhys = H @ t / Hs`, leaning on the filter's constant-field fixed point
  `H @ 1 / Hs == 1`) -- `test_init_state_seeds_the_raw_fields`,
  `test_init_state_density_half_is_derived_from_its_seed`, and
  `test_init_state_time_half_is_derived_from_its_seed` in `test_optimize.py`.
- [x] First-principles checks on `estimated_conductivity`/`hotspot_constraint`'s
  *values* -- `test_conductivity.py` now has closed-form hard/soft-gated 3-element
  cases, the `rouf -> 0`/`rouf -> inf` limits, monotonicity in build order/global time
  shift, and uniform-density closed forms, well beyond the gradient-only FD checks.
- [x] Property test for `filters.continuity_filter` (annihilates constant fields, row
  structure, closed forms on ramps/checkerboards) -- `test_filters.py`.
- [ ] Fast non-`@slow` end-to-end check on `optimize.run` at small `nelx`/`nely`:
  objective descent, constraints satisfied at termination, mirror-symmetry invariance.
  Still missing: `test_e2e.py`'s tests are all `.mat`-fixture comparisons (iteration-1
  assembly, MMA state threading, constraint stacking, a tiny fixture trajectory, the
  loop-25 hotspot refresh), and `run()`'s only fixture-free physics evidence remains the
  180x60x800 `test_e2e_slow.py` reproduction, whose `f0val < 195` ceiling is hand-tuned
  rather than derived.

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
- [x] After a PyTorch port, evaluate replacing some of the hand-derived sensitivity
  (`dc`/`dt`/adjoint) code with autodiff, at least for the more mundane/straightforward
  derivative chains -- manually-written sensitivities are a common source of subtle
  correctness bugs (mismatched chain rule terms, stale derivatives after a forward-pass
  change) that autodiff sidesteps. Likely not a full replacement everywhere -- some
  sensitivities may be intentionally hand-optimized or awkward to express in autodiff --
  so this needs a per-case call, not a blanket switch.
  **Done via `plans/archive/torch_port_part2.md`:** every sensitivity in
  `sttopt/compliance.py`, `sttopt/constraints.py`, `sttopt/conductivity.py`, and the FEM
  solve now comes from autograd, including the adjoint (`sttopt/torch_solve.py`'s
  `FemSolve`). No hand-derived call site was intentionally kept for performance --
  Phase 3.4's benchmark found plain autograd already faster than the hand-derived
  hotspot algebra at the production mesh. The hand-derived formulas survive only as an
  oracle in `tests/reference/`.

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
