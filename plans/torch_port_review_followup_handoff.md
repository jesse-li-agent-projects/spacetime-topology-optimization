# Handoff — torch_port_review_followup

Self-handoff for continuing `plans/torch_port_review_followup.md` (PR #53). Phases 1-2
are committed and done. Phase 3 is committed with a narrower scope than the plan
wrote down -- read "Phase 3: what changed" below before touching `step()` again.
Phase 4 is done, with one scope narrowing -- read "Phase 4: what changed" below
before touching `benchmarks/calibrate_cg_rtol.py`. Phase 5 is done for item 1 only
(items 2-3 need a decision, item 4 done) -- read "Phase 5: what changed" below
before touching fixture-loading/test conversion helpers. Phase 6 is untouched.

## Status

| Phase | State |
| --- | --- |
| 1 -- Docstring and boundary fixes | Done, committed (amended once mid-flight) |
| 2 -- Delete no-op detaches, mark gradient region | Done, committed |
| 3 -- Move beta updates to the tail | Done for items 1+3; item 2 (filter-pass removal) **dropped**, not deferred -- see below |
| 4 -- Move hand-derived formulas to `tests/reference/` | Done for items 1-4; item 5 (delete `calibrate_cg_rtol.py`) **not done, needs a decision** -- see below |
| 5 -- One tensor-boundary conversion helper | Done for items 1+4; items 2-3 (fixture-loading conversion, deleting `tt`/`tti`) **not done, needs a decision** -- see below |
| 6 -- Reply to review comments | Not started |

Commits so far on this branch (`torch-port-review-followup-impl`, PR #54):
1. Phase 1 (amended once -- see below)
2. Phase 2
3. Phase 3 (items 1+3 only)
4. Phase 4 (items 1-4; item 5 pending a decision)
5. Phase 5 (items 1+4 only; items 2-3 pending a decision)

## Phase 3: what changed from the plan, and why

The plan's item 2 said: "In the head, read `state.xTilde`/`state.xPhys`/`state.tPhys`
directly instead of re-filtering. This removes one full filter pass per iteration."

This turned out not to be implementable as a pure storage-reuse trick. Worked through
with the user in-session:

- `x`/`t` stay the autograd leaves (confirmed with the user) -- so the head's
  `xTilde = filter(x)` / `xPhys = heaviside(xTilde, beta_d)` pass is required every
  iteration regardless; it's not the redundant one.
- The only real candidate for removal was the **tail's** recompute of
  `xTilde_new`/`xPhys_new`/`tPhys_new` from `x_new`/`t_new` (post-MMA), which -- once
  the beta bump moved to the tail, making `beta_d` consistent across the boundary --
  is numerically identical to what the *next* iteration's head recomputes from the
  same `x_new` anyway. Storing the head's already-computed (pre-update) `xTilde`/
  `xPhys`/`tPhys` instead (skipping the tail recompute) looked like the fix.
- That breaks `test_step_output_state_is_self_consistent`
  (`tests/test_optimize.py:257`), which pins an explicit, load-bearing invariant:
  `state.xPhys`/`state.tPhys` always equal filtering `state.x`/`state.t` at
  `state.beta_d`. `_state_from_raw`'s docstring calls this "the invariant... i.e. the
  state `step` itself would have produced" and other tests build on it. Confirmed
  empirically: dropping the tail recompute broke 7 tests (`test_e2e.py`,
  `test_robustness.py`, `test_torch_fem.py`, `test_cli.py`, `test_optimize.py`) with a
  consistent one-iteration-stale pattern.
- So the tail recompute must stay for correctness, not just fidelity. Item 2 is
  reverted; the docstring/field comments were rolled back to not claim a lag that
  doesn't exist. **This is not a "removes one filter pass" win -- that part of the
  plan was wrong**, not merely unimplemented. Don't re-attempt it without a genuinely
  different mechanism (e.g. restructuring the objective's own differentiation to go
  through `xTilde`/`tPhys` as leaves with a filter-adjoint, the way the constraints
  already do via `_grad_rows_batched`/`_filter_adjoint` -- that's a real, larger
  refactor, not a quick win, and wasn't attempted here).

What's actually done: the three periodic updates (`beta_t`, `beta_d`, hotspot
`factor`) all now bump at the tail of `step()`, next to each other, taking effect
starting the next iteration -- exactly per the Decision at the top of the plan's
Phase 3. Fast suite (412 passed, 4 skipped) is green.

**Confirmed:** `test_e2e_slow.py::test_thesis_4_4_reproduction` (`nloop=800`) passed
(2375.78s, `1 passed`) -- the two boundaries this reordering actually touches
(`loop % 30`, `loop % 50`) hold under the new ordering. Phase 3 is done as scoped
(items 1+3; item 2 abandoned, see above).

The plan also flagged `tests/fixtures/torch_port_designs.npz` as expected to drift
slightly from this phase and said not to regenerate unless a benchmark looks wrong --
that guidance still stands, unchanged by the item-2 revert.

## Phase 4: what changed from the plan, and why

Items 1-4 are done as scoped: `tests/reference/{compliance,constraints,conductivity,fem}.py`
now hold the hand-derived predecessors (`whole_compliance`, `gravity_compliance`,
`batched_whole_and_gravity_compliance`, `time_mask_derivative`,
`global_volume_fraction`, `time_field_continuity`, `start_point`,
`stage_volume_bounds`, `hotspot_constraint` + its private `_conductivity_terms`/
`_ConductivityTerms`/`HotspotConstraintResult`, `assemble_stiffness`, `solve_fe`).
`sttopt/{compliance,constraints}.py`'s `*_value` functions were renamed to the plain
names their hand-derived predecessors vacated (`conductivity.hotspot_value` was
**not** renamed -- its signature/return shape never matched `hotspot_constraint`'s,
so there's no plain name for it to inherit). `test_fem.py`'s five
`assemble_stiffness`/`solve_fe`-dependent tests moved to the new
`tests/reference/test_fem.py`, alongside the functions. Every caller across
`tests/`/`benchmarks/` (11 files, plus 3 more not in the plan's list that turned out
to depend on the moved names: `test_e2e.py`, `test_robustness.py`,
`calibrate_cg_rtol_autograd.py`) was repointed. Fast suite: 412 passed, 4 skipped --
same count as Phase 3's baseline (pure reorg, no tests added or dropped).

**Item 5 (delete `benchmarks/calibrate_cg_rtol.py`) is not done -- needs a decision,
don't do it without confirming.** The plan's premise was that
`calibrate_cg_rtol_autograd.py` supersedes it. In fact three test files
(`test_torch_fem.py`, `test_compliance.py`, `test_torch_solve.py`) import `calib =
calibrate_cg_rtol` and lean on API `calibrate_cg_rtol_autograd.py` doesn't have at
all: `spsolve_backend`, `sensitivities`, `mesh_setup`, `FIXTURES`, `EMIN`/`EMAX`/
`PENAL`/`BETA_T`, `SENSITIVITY_TOL`, `elementwise_errors`, `finite_difference_check`.
`calibrate_cg_rtol_autograd.py` compares MGCG-at-a-candidate-`rtol` against
MGCG-at-`1e-12`, not against `spsolve` -- it's a different comparison, not a
drop-in replacement for the spsolve-vs-MGCG one these three files run. Deleting the
file as written would delete real fast-suite test coverage (four tests, all
currently green: `test_sensitivities_from_mgcg_match_spsolve_elementwise`,
`test_compliance_is_far_more_forgiving_than_its_sensitivities`,
`test_mgcg_sensitivity_matches_finite_difference`,
`test_whole_compliance_fixture_regression_through_mgcg`, plus the two
`test_adjoint_matches_hand_derived_*_near_binary` tests in `test_torch_solve.py`) --
not a safe mechanical step. What I did instead: kept the file, and repointed its own
internal `fem.assemble_stiffness`/`fem.solve_fe`/`compliance.whole_compliance`/
`gravity_compliance` calls at `tests/reference/`'s versions (the same rename Phase 4
did everywhere else), since Phase 4's renames would otherwise have silently changed
what `sensitivities()` computes (autograd `dcx=U` instead of hand-derived `dcx`).
Before deleting this file, either confirm with the user that the coverage
`calibrate_cg_rtol.py`-only tests provide is fine to drop, or migrate those tests to
compare against `calibrate_cg_rtol_autograd.py`'s MGCG-vs-tight-MGCG reference
instead (a real design decision about what those tests should assert once `spsolve`
is out of the picture, not mechanical).

## Phase 5: what changed from the plan, and why

Item 1 (`Problem`'s per-field `to_tensor` calls) is done: `torch_util.to_tensors`
batch-converts a dict of same-dtype arrays in one call, keyed by field name so it
splats straight into the dataclass constructor. `build_problem` calls it once for the
float-valued fields (`KE`/`F`/`Hs`/`w`) and once for the int-valued ones
(`edofMat`/`freedofs`/`e1`/`e2`/`Nei`), replacing 9 individual `to_tensor` calls with 2.
`init_state`'s own single `to_tensor` call (`init_timefield`'s output) is left alone --
it converts one array, not a group, so wrapping it in `to_tensors` would be pure
overhead for no DRY win. Item 4 (`to_numpy`'s docstring) is done: dropped the stale
"until later phases port them" clause -- the leaf math modules are torch now, per
`optimize.py`'s own module docstring -- and states plainly `cli.py`/`viz.py`/fixture
writing need it to stay public. Fast suite: 412 passed, 4 skipped -- unchanged.

**Items 2-3 (fixture-loading conversion, deleting `tt`/`tti`/`_K_est`/`_hotspot`) are
not done -- needs a decision, don't do it without confirming.** What I found working
through it:

- The plan's item 3 names `tt`/`tti` as living "in `test_constraints.py`" and calls
  them "per-test wrappers" -- both premises are stale. They live in `conftest.py`
  (centralized there by an earlier phase, plan text never updated) and are genuinely
  shared utilities, not per-test scaffolding: 197 call sites across 8 test files
  (`test_compliance.py`, `test_conductivity.py`, `test_constraints.py`, `test_mma.py`,
  `test_mma_toy_problems.py`, `test_reference_sweep.py`, `test_robustness.py`,
  `test_torch_fem.py`), most converting *locally-generated* NumPy (`rng.uniform(...)`
  etc.), not fixture data. `test_e2e.py`'s "equivalent" the plan asks to delete is
  already gone -- that file has no conversion wrapper today, so that part of item 3 is
  already satisfied (no-op).
- Item 2 ("fixture-loading fixtures return already-converted objects") means making
  `load_fixture_npz` hand back tensors instead of NumPy arrays -- there's no actual
  pytest fixture involved, it's a plain function called ~20 times across 9 files. This
  looked safe at first (`assert_close` already tensor-tolerates via `_as_numpy`, and
  `tt(already_a_tensor)` is a harmless no-op via `np.asarray`'s tensor support) but
  isn't: at least one call site feeds a fixture array straight into
  `scipy.sparse.csr_matrix(...)` (`test_compliance.py:54`,
  `sp.csr_matrix(grav["C"])`) -- unverified whether SciPy accepts a `torch.Tensor` in
  place of an `ndarray` there, and `torch_util.py`'s own module docstring already flags
  the adjacent hazard (`ndarray * tensor` raises `TypeError` rather than upcasting).
  Auditing every one of the ~20 `load_fixture_npz` call sites for this kind of
  numpy-only consumption is a real per-site check, not mechanical.
- Item 3's *named* wrapper deletions (`_K_est`/`_hotspot` in `test_conductivity.py`)
  are self-contained and don't depend on item 2, but aren't small either: ~40 call
  sites in a ~1050-line file, most feeding results into further NumPy-only code
  (`np.abs`, `np.testing.assert_allclose`, finite-difference perturbation loops) that
  would also need converting to stay `torch.allclose`-native. Comparable in size to
  item 2's audit, just confined to one file.

Before doing items 2-3: either confirm with the user that the ~20-site fixture-load
audit (item 2) and the ~40-site `test_conductivity.py` rewrite (item 3's named part)
are worth doing now, or scope them down further (e.g. only `test_conductivity.py`,
leaving the other 7 files' `tt`/`tti` call sites as the shared utility they actually
are -- item 3's literal text never asked for those).

## Phase 1 amendment (context for `git log`)

The first Phase 1 commit used a defensive `x0.detach()` inside `femsolve()`
(unconditionally detaching, silently). The user asked for this to be an assertion
instead ("the incoming warmstart seed, if present, should be detached") so a caller
bug shows up loudly rather than being silently papered over. Amended before Phase 2
started: `femsolve()` now asserts `x0 is None or not x0.requires_grad`; `optimize.py`'s
`U=U_new.detach()` is the (required, not redundant) thing that satisfies it.

## Next steps

1. Phase 4 item 5 -- decide `calibrate_cg_rtol.py`'s fate (see above), then act on
   the decision. Not mechanical; needs the user or a fresh look at the three
   dependent test files.
2. Phase 5 items 2-3 -- decide the fixture-loading/`tt`-`tti`/`_K_est`-`_hotspot`
   scope (see above), then act on the decision. Not mechanical; needs the user or a
   fresh per-call-site audit.
3. Phase 6 -- reply to and resolve the four review threads that need no code change
   (text already drafted in the plan).
4. Once all phases are done, move `plans/torch_port_review_followup.md` (and this
   handoff file) to `plans/archive/` and update `plans/CLAUDE.md`'s index, per that
   directory's own convention.
