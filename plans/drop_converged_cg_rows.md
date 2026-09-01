# Drop converged rows from the batched CG solve, then delete the batched/unbatched split

## Goal

`optimize.step` currently has two FEM paths -- one batched `FemSolve` call and a
sequential loop over `whole_compliance` + `nStage` `gravity_compliance` calls -- selected
by `Problem.batch_fem_solves`, which `build_problem` defaults from mesh size. The split
exists only because batching *loses* on large meshes. Remove the cause, then remove the
split: always batch, and let `torch_fem.pcg` retire each batch row when that row reaches
`rtol` instead of running every row for as many iterations as the slowest row needs.

## Why batching loses today (measured, not assumed)

One `optimize.step` at production settings (`configs/default.json`, `nStage=8`, float64,
CUDA, iteration-800 near-binary snapshot from `tests/fixtures/torch_port_designs.npz`),
averaged over 3 steps with the previous step's `U` as warm start. "row-iterations" counts
CG iterations x rows: `9 * max(rows)` is what the batched path actually pays,
`sum(rows)` is what it would pay if each row stopped at its own convergence.

| mesh | batched ms/step | sequential ms/step | per-row convergence iteration | paid `9*max` | ideal `sum` | waste |
|---|---|---|---|---|---|---|
| 90x30 | 274 | -- | 20-26 | 225 | 208 | 8% |
| 180x60 | 514 | 571 | 13-28 | 261 | 173 | 34% |
| 360x120 | 2566 | 2344 | 16-45 | 378 | 223 | 41% |

Three facts follow:

- The waste grows with mesh size, which is exactly the shape of the 360x120 regression the
  flag was introduced to dodge. The premise behind early row retirement holds.
- The spread is structural, not noise: row 0 (the point-load whole-structure solve) and
  the last two gravity stages are consistently the hard rows; the middle stages converge
  in roughly half the iterations.
- **Every backward solve converges at iteration 0** at every mesh, on every row -- the
  self-adjoint warm start in `torch_solve.py` already works exactly as documented. Row
  retirement only has to help the forward solve; there is nothing left to win in the
  adjoint.

Scaling the measured batched cost per row-iteration to the ideal iteration count predicts
~1.5-1.7x against the current batched path and ~1.5x against the sequential path at
360x120 -- comfortably enough to make batching unconditionally the better path. The
prediction ignores compaction overhead and the rising per-row cost of a shrinking batch,
so Phase 3 below measures rather than assumes it.

## Design

### 1. Levels carry data, not closures

`torch_mg._Level.apply_A` is an opaque closure over `density` (level 0) or `keff` (coarse
levels), so a row subset cannot be taken. Replace it with the operator's data --
level 0 keeps `density`, coarse levels keep `keff` and their own `edofMat` -- and make
`apply_A` a method that dispatches on which is set. Then add:

```python
def select(self, rows: Int[Tensor, " k"]) -> "_Level":
```

returning the same level with every batched tensor (`density`/`keff`, `diag`, `chol`)
indexed along the batch dim, and every shared tensor (`mask`, `edofMat`) passed through
unchanged. A tensor that carries no batch dim (the broadcast case: one density field, many
right-hand sides) is passed through as-is.

`VCycle.select(rows)` is then one line: a new `VCycle` over the selected levels with the
same `omega`/`n_smooth`/`gamma`.

### 2. `pcg` retires converged rows

`pcg` keeps its current per-row residual test and gains a compaction step: rows at or below
`rtol` have their `x` written into a full-size output buffer and are removed from the
active set; the loop continues on the remaining rows with `apply_A.select(active)` and
`apply_M.select(active)`.

- **Operator interface.** `apply_A`/`apply_M` are `BatchedOperator`s -- callable, with
  `select`. Callers that pass a plain callable (the Jacobi-preconditioner tests) keep
  working: without `select`, `pcg` runs the current all-rows-together schedule.
- **When to compact.** Gathering `keff` costs roughly one V-cycle, so compact only when the
  active set has fallen to <=75% of its size at the last compaction (tune in Phase 3), and
  never below 2 rows.
- **Correctness.** CG's `alpha`/`beta`/`rz` are per-row scalars and the operator is
  block-diagonal across rows, so retiring a row changes no other row's arithmetic. Results
  are equal to the current path up to kernel-reduction order, not bitwise.
- **`CGConvergenceError`** must report *original* row indices; map the active index back
  before raising.
- The docstring's "frozen row" paragraph (a zero-`b` row kept in the batch and protected by
  `safe_div`) is superseded: such a row now converges at iteration 0 and is retired before
  the loop. Keep `safe_div` -- an exact-zero `rz` is still reachable -- but the paragraph
  shrinks to one sentence.

### 3. Delete the split

Once Phase 3 confirms the win, `optimize.step` calls
`compliance.batched_whole_and_gravity_compliance` unconditionally. Removed:
`Problem.batch_fem_solves`, `RunConfig.batch_fem_solves`, the key in `configs/*.json`, the
mesh-size default in `build_problem`, and the two-branch block in `step`. `State.U` stops
being `| None` except at `init_state`.

`compliance.whole_compliance` / `gravity_compliance` stay -- they are the single-solve API
`tests/reference` and the fixture tests use, and they are what the batched function is
checked against.

**Compatibility note (needs a decision):** `RunConfig.from_dict` rejects unknown keys, so
run configs already saved under `outputs/` that carry `batch_fem_solves` will fail to load
after the field is removed. Either accept the break (and say so in the PR) or ignore this
one retired key with a warning. Recommendation: accept the break; this is research code and
the key selects an implementation detail that no longer exists.

## Phases (one commit each)

1. **`_Level` refactor.** Data-carrying levels, `apply_A` as a method, `_Level.select` and
   `VCycle.select`. No behaviour change; existing `torch_mg`/`torch_fem` tests are the gate.
2. **Row retirement in `pcg`.** New tests: (a) batched solve with deliberately unequal
   per-row difficulty equals per-row separate solves to solver tolerance; (b) a zero-`b`
   row returns exactly zero and is retired at iteration 0; (c) an already-converged warm
   start retires before the first iteration; (d) `CGConvergenceError` names the correct
   original row indices when only some rows fail; (e) reported `n_iter` is the last active
   row's count. `sttopt/` behaviour with the flag on is unchanged elsewhere.
3. **Measure** (see below). Record the numbers in the PR.
4. **Delete the split.** `optimize.py`, `run_config.py`, `configs/*.json`, and the
   docstrings in `optimize.py`/`compliance.py`/`torch_solve.py` that describe the flag.
   `tests/test_optimize.py::test_batched_matches_sequential` moves down a level: it becomes
   a `test_compliance.py` check that `batched_whole_and_gravity_compliance` matches
   `whole_compliance` + `gravity_compliance`, which keeps the equivalence guarantee without
   the flag it currently toggles.

Phases 1-2 are shippable on their own: they speed up the batched path at every mesh even if
Phase 4 is deferred.

## Benchmarks

Primary, in order of authority:

- **`benchmarks/bench_fem_solve.py`** -- solve-only, cold and warm start, accuracy asserted
  at every timed point. This is the benchmark the win has to show up in first. Add a
  `720x240` mesh: 360x120 is where batching currently loses by 9%, and the trend must hold
  past it, not just at it. Add a reported column for row-iterations (`max` vs `sum` per
  solve) -- the iteration counts are what explain the times, and after Phase 2 the gap
  between those two numbers is the thing that shrank.
- **`benchmarks/profile_step.py`** -- end-to-end `step` breakdown at production settings, at
  180x60 (its default) and 360x120. Confirms the solve's share of step time falls and that
  no cost moved into the surrounding wiring.
- The per-row convergence diagnostic used above (monkeypatched `pcg` recording each row's
  first converged iteration) is worth folding into `bench_fem_solve.py` rather than kept as
  a throwaway.

**Acceptance criteria.** At 90x30, 180x60, 360x120 and 720x240, warm-started and cold:
batched-with-retirement must be faster than *both* the current batched path and the current
sequential path. The binding case is 360x120, where the target is >=1.3x against sequential;
anything less than parity there means Phase 4 must not land and the flag stays.

**Correctness gates.** `tests/test_reference_sweep.py` (sensitivities against the
hand-derived oracle), `tests/test_e2e_slow.py` (trajectory), and the `torch_mg`/`torch_fem`
solver tests. Fixture comparisons are tolerance-based, which they must be: compaction
changes batch sizes, and therefore kernel reduction order, so results move in the last bits.

## Risks

- **Compaction overhead eats the win at large meshes.** Gathering `keff` across the
  hierarchy is a real copy. Mitigated by the 75% threshold; if the threshold has to be
  tuned per mesh to break even, that is a negative result and Phase 4 does not land.
- **Retirement helps least where batching already wins** (90x30: 8% of iterations). Fine --
  the flag is not there for the small meshes.
- **Uneven rows may become even** at other operating points (early iterations, other
  `print_base` settings, other `nStage`). Measured spread is at the near-binary late-run
  design, which is where runtime is actually spent; a spot check at an early-iteration
  snapshot belongs in Phase 3.
