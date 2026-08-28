# Torch port review follow-up

Executes the design-review comments left on PRs #44, #45, #47, #48, #52 (Phases 3.1-3.7
of `torch_port_part2.md`). Each phase below is one commit.

Decisions already taken by the user are recorded inline as **Decision**. Do not
re-litigate them.

## Findings that changed the plan

Two review comments turned out to need no code change, and one needed a different fix
than the comment proposed. Verified before writing this plan:

- `torch_fem.safe_div` is called from `torch_solve.py:117`, not only from `pcg`. It
  stays at module level. Reply to the comment; change nothing.
- `compliance.time_mask`'s `math.tanh(beta * ti)` takes Python floats (`beta: float`,
  `ti: float`). No gradient can flow through it, and `torch.tanh` would allocate a
  0-d tensor per call for nothing. The code is correct. Reply; change nothing.
- `state.x`/`state.t` do **not** track gradients today, contrary to the review comment
  on #52. Confirmed by inspecting `requires_grad`/`grad_fn` on every tensor field of
  `State` after two `step()` calls: all are `False`/`None`. The real defect is that
  five of the fourteen `.detach()` calls in `step()` are no-ops, which hides where the
  gradient-tracking region actually begins and ends. See Phase 2.

## Phase 1 -- Docstring and boundary fixes

Small, independent, no behaviour change beyond the `x0` detach.

1. `conductivity.py:154` `_safe_pmean`: rewrite the docstring in Sphinx format. State
   which variable the gradient is with respect to, and state the function's own
   contract rather than what `hotspot_constraint` needs from it. (#48)
2. `torch_solve.py`: detach `x0` inside `femsolve()` immediately before
   `FemSolve.apply`, so no caller can wire a warm start into the autograd graph.
   `FemSolve.backward` already returns `None` for `x0` unconditionally, so this is an
   invariant of `FemSolve`, not of any one caller. Detaching inside `forward` is too
   late -- `apply()` wires the graph edge before `forward` runs. (#52)
3. Keep the existing `U=U_new.detach()` in `optimize.py`. Belt and braces on a leak
   that cost ~180 MB/step is worth the redundancy; shorten its comment to one line
   pointing at commit 855eb76.

## Phase 2 -- Delete the no-op detaches, mark the gradient region

**Decision:** do not extract the region into a function. Delete the dead calls and
delimit the region with one comment at each end.

`optimize.py:522-523` builds the MMA bounds from `state.x`/`state.t`, not from the
`x`/`t` autograd leaves, so nothing downstream of it requires grad.

1. Delete five no-op `.detach()` calls: `state.x.detach()` and `state.t.detach()`
   (`:453-454`, keep `.clone().requires_grad_(True)`), `xval.detach()`,
   `xmin.detach()`, `xmax.detach()` (`:611-613`), and `xold1=xval.detach()` (`:641`).
2. Keep the nine real ones: `c_t`, `f0val_t`, `df0dx`, `fval`, `dfdx`, `numer_t`,
   `K_est_t`, `xPhys`, `U_new`. Each acts on a tensor built from the leaves.
3. Add one comment where `x`/`t` become leaves and one where the last gradient-carrying
   value is consumed, naming the region.

Sanity check: re-run the `requires_grad` sweep over `State`'s fields after two steps.
Every field must still be `False`.

## Phase 3 -- Move the beta updates to the end of `step()`

**Decision:** bump `beta_t`/`beta_d` at the tail of `step()`, mirroring the hotspot
`factor` refresh. The bump then takes effect on the next iteration.

The MATLAB source
(`Space_Time_TopOpt_Gravity_different_timefield.m:156`) bumps `beta` at the top of the
loop but does not recompute `xPhys`, which line 269 of the previous iteration built at
the old `beta`. So MATLAB's own value and sensitivity disagree on the doubling
iteration. There is no faithful ordering to copy; this phase picks the consistent one
that also costs less.

1. Move the `loop % 30` / `loop % 50` blocks from the head of `step()` to the tail,
   next to the `factor` refresh.
2. In the head, read `state.xTilde`/`state.xPhys`/`state.tPhys` directly instead of
   re-filtering. This removes one full filter pass per iteration.
3. Update `step()`'s docstring: replace the paragraph explaining the head recompute
   with the new rule, and fold the `factor` paragraph into it, since all three periodic
   updates now share one ordering.

Sanity checks:
- Fast tests must pass unchanged. The E2E fixture runs `nloop=3`, so no periodic update
  fires and no fixture needs regenerating.
- `test_e2e_slow.py::test_thesis_4_4_reproduction` (`nloop=800`) **must** be run
  explicitly. It crosses many `loop % 30` and `loop % 50` boundaries. Its assertions are
  loose bands (`185.0 < f0val < ceiling`, `tru_max` within 1% of 0.8), so a
  one-iteration lag should stay inside them, but this is the only real check that it
  does. Run it alone -- it is compute-intensive.
- `tests/fixtures/torch_port_designs.npz` drifts slightly, since it records a real run.
  It feeds benchmarks only (`bench_fem_solve.py`, `calibrate_cg_rtol.py`,
  `profile_step.py`), never a correctness assertion, so regenerating it is optional.
  Note the drift rather than regenerate, unless a benchmark result looks wrong.

## Phase 4 -- Move the hand-derived formulas out of `sttopt/`

**Decision (confirm before starting):** move, do not delete. Moving costs almost
nothing and keeps the autograd-vs-hand-derived agreement tests running.

Nothing in `sttopt/optimize.py` calls the hand-derived functions any more. Only tests,
fixtures, and benchmarks do. The independent MATLAB oracle is unaffected either way:
`tests/matlab_reference.py` is a standalone transliteration with its own `_assemble`,
`ref_whole_compliance`, `ref_gravity_compliance`, and `ref_hotspot`. It does not import
`sttopt`.

1. Create `tests/reference/`. Move into it:
   - the hand-derived `whole_compliance`, `gravity_compliance`,
     `_whole_compliance_from_U`, `_gravity_compliance_from_U`,
     `batched_whole_and_gravity_compliance`, and `time_mask_derivative` from
     `compliance.py`;
   - the hand-derived `global_volume_fraction`, `time_field_continuity`, `start_point`,
     `stage_volume_bounds` from `constraints.py`;
   - `hotspot_constraint` from `conductivity.py`;
   - `assemble_stiffness` and `solve_fe` from `fem.py`. Leave `plane_stress_KE`,
     `node_grid`, and `element_dof_map` in `sttopt/fem.py` -- the torch path still uses
     them for setup.
2. Rename each `*_value` function to the plain name it replaces.
3. Update the import lines in `test_compliance.py`, `test_constraints.py`,
   `test_conductivity.py`, `test_fem.py`, `test_reference_sweep.py`,
   `test_gravity.py`, `test_optimize.py`, `test_torch_fem.py`, `test_torch_solve.py`,
   `tests/fixtures/generate_fixtures.py`, and `benchmarks/bench_fem_solve.py`. The
   tests themselves need no logic change; `test_fem.py`'s four closed-form patch tests
   and one fixture test move with the functions unchanged.
4. `test_reference_sweep.py` keeps comparing against `matlab_reference.py`'s `ref_*`.
   Its other side becomes the autograd path, which makes the sweep a stronger check
   than it was.
5. Delete `benchmarks/calibrate_cg_rtol.py`. `calibrate_cg_rtol_autograd.py` supersedes
   it and Phase 3.6 already re-calibrated against autograd. (#45)

The `.npz` golden fixtures need no regeneration. They are frozen arrays on disk and
already hold the sensitivities (`dcx_whole_all`, `dcx_grav_all`, `dct_grav_all`,
`df1_all`, `dt1_all`, `dfdx_all`, `dcx0`). Autograd agreeing with them is exactly the
cross-validation, pinned permanently. `generate_fixtures.py` only needs its imports
repointed at `tests/reference/`, so re-running it stays reproducible.

## Phase 5 -- One conversion helper at the tensor boundary

**Decision:** a single `Problem`/`State` conversion entry point, and torch-native
tests.

1. Add one Array-to-Tensor conversion function in `torch_util.py` that converts a whole
   `Problem` (and one for `State`), replacing the per-field `to_tensor` calls now
   spread through `build_problem`/`init_state`.
2. In `tests/conftest.py`, have the fixture-loading fixtures return already-converted
   objects.
3. Delete the per-test wrappers that exist only to convert (`_K_est` and the
   `hotspot_constraint` wrapper in `test_conductivity.py:23,39`, the `tt`/`tti` helpers
   in `test_constraints.py`, and the equivalents in `test_e2e.py`). Assert with
   `torch.allclose` on tensors instead of round-tripping through `to_numpy()`.
4. `to_numpy` stays public. `cli.py`, `viz.py`, and fixture writing all need it, so it
   cannot become private. Update its docstring to say that, and drop the "until later
   phases port them" clause, which is now stale. (#44, #45)

## Phase 6 -- Reply to the review comments

Post replies on the four threads that need no code change, and resolve them:

- `torch_fem.py:147` (#47): `safe_div` is used by `torch_solve.py:117`; it stays at
  module level.
- `compliance.py:34` (#45): `beta`/`ti` are Python floats; `math.tanh` is correct and
  carries no gradient to lose.
- `compliance.py:35` (#45): decline `from torch import tanh, tensor`. The `torch.`
  prefix marks which operations enter the autograd graph, which is the distinction this
  module turns on.
- `#52` general comment on the gradient boundary: point at Phase 2, and note that the
  invariant already held -- the defect was the dead defensive calls.

Link the remaining threads to the phase that closes them.

## Not doing

- Extracting `step()`'s gradient region into its own function. Rejected as
  overengineering; Phase 2's comments do the job.
- Regenerating `torch_port_designs.npz` for Phase 3's drift, unless a benchmark result
  looks wrong.
