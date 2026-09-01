# Unify `optimize.step`'s sensitivity assembly

## Motivation

`optimize.step` currently assembles the MMA rows through two different helpers
(`_grad_row`, `_grad_rows_batched`) that differ in three ways at once: batched vs.
unbatched, different differentiation cut set, and different argument lists. The call
sites therefore have to know which helper a given constraint needs, and the row values
and the row sensitivities are accumulated into two parallel lists (`fval_parts`,
`dfdx_parts`) that must be kept index-aligned by hand.

The target shape is the one a reader would expect: build the constraint values, then
derive every row's sensitivity through **one** helper with **one** signature.

### Why the two helpers exist today

They are not two kinds of math. They are one kind of math with two cut sets:

- `_grad_row` differentiates all the way to the raw leaves `(x, t)`.
- `_grad_rows_batched` cannot, because `is_grads_batched`'s vmap has no batching rule
  for the sparse CSR matmul in `xTilde = H @ x / Hs`. So it cuts at `(xTilde, tPhys)`
  and applies the filter adjoint by hand (valid because `H` is symmetric by
  construction).

So the split is a consequence of batching, and it leaked into the call sites.

### Measured facts this plan is built on

All measured on this repo, mesh 180x60, `nStage=4`, float64. Reproduce before trusting
(see "Benchmarking" below).

1. Which outputs survive a vmapped `autograd.grad` (cut at `xTilde`/`tPhys`):

   | row | vmapped grad |
   |---|---|
   | global volume | OK |
   | time-field continuity | **FAIL** — `expand is unsupported for SparseCsc tensors` (the `L @ tPhys` inside the constraint) |
   | start point (k=nely) | OK |
   | stage volume bounds (k=nStage) | OK |
   | hotspot | OK |
   | whole / gravity compliance | **FAIL** — `Batching rule not implemented for aten::is_nonzero`, from the convergence test inside the adjoint CG that `FemSolve.backward` runs (`torch_fem.pcg`, the `if torch.all(rel_resid <= rtol)` line) |

   So a single `autograd.grad` over the whole stacked `[f0val, *g]` vector is **not
   available**. Do not attempt it.

2. Differentiating to the raw leaves is very expensive **on CPU**, and it is not the
   constraint math:

   | op (CPU, 180x60) | time |
   |---|---|
   | grad to `(xTilde, tPhys)` | 0.15 ms |
   | grad to the raw `(x, t)` leaves | 394 ms |
   | `H @ v` (CSR) | 2.1 ms |
   | `H.t() @ v` (CSC) | 197 ms |

   Autograd's backward for `H @ x` is `H.t() @ grad`, i.e. a CSC matvec, which PyTorch
   does very slowly on CPU. On GPU the same comparison is 1.31 ms vs 1.69 ms — i.e. the
   effect is CPU-only and does not matter for production GPU runs.

### The design this plan adopts

**Always cut at `(xTilde, tPhys)` and finish the filter adjoint by hand. Engage vmap
only when `k > 1`.**

This gives one helper and one cut set, and it has a useful property: the rows that
cannot survive vmap (continuity, the objective) are all `k == 1`, so they never enter
the vmapped path and their sparse/iterative ops are never a problem. Rows that need
batching (start point, stage bounds) are already vmap-safe.

Note that this is a coincidence of the current constraint set, not a guarantee. Say so
in the helper's docstring, and name the restriction (a `k > 1` constraint must avoid
sparse matmuls and `FemSolve`).

**Pre-verified, so you do not have to rediscover it.** On an 8x5 mesh, the proposed
`k == 1` branch (plain grad at the filtered cut, then `H @ (g / Hs)` by hand) reproduces
today's `_grad_row` output **bit for bit** — `max|old - new| == 0` — for the volume,
continuity and hotspot rows. The cut-set change is therefore expected to be exactly
value-preserving, not merely close. If your implementation shows a nonzero difference
on those three rows, you have a bug; do not reach for a looser tolerance.

## Non-goals

- **Do not** rewrite `H` or `L` as gather/`index_add` operators to make them vmap-safe.
  It is possible (`torch.zeros(n).index_add(0, rows, vals * v[cols])` passes
  `is_grads_batched`, verified), but pulling `H` inside the vmapped region would make
  the backward materialize `(k, nnz)` intermediates instead of one `(k, nel)` one,
  which is a peak-memory regression at the larger meshes. Out of scope; leave a
  follow-up note, not code.
- **Do not** change `FemSolve` or `torch_fem.pcg`. The objective is a single scalar, so
  there is nothing to batch there.
- **Do not** change any constraint's mathematical value. Row values must be unchanged.
- **Do not** touch `tests/reference/`, `tests/matlab_reference*.py`, or the fixtures.

## Implementation

### Phase 1 — one sensitivity helper

Replace `_grad_row` and `_grad_rows_batched` in `sttopt/optimize.py` with a single
helper. Suggested shape (adapt as you see fit, but keep one public entry point):

```python
def _sensitivity_rows(
    outputs: Float[Tensor, " k"],
    xTilde: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    H: Tensor,
    Hs: Float[Tensor, " nel"],
) -> Float[Tensor, "k n"]:
```

Behaviour:

- `outputs` is always 1-D. A scalar row is passed as `value[None]`; the helper never
  takes a 0-dim tensor. This is what removes the `[None, :]` / `[None]` noise from the
  call sites.
- Differentiate `outputs` w.r.t. `(xTilde, tPhys)`, with `retain_graph=True` and
  `allow_unused=True`.
  - `k == 1`: a plain `torch.autograd.grad` with no `grad_outputs` batching.
  - `k > 1`: `is_grads_batched=True` with a `torch.eye(k)` seed, exactly as
    `_grad_rows_batched` does today.
- Finish with the existing `_filter_adjoint` logic — `(H @ (g / Hs).T).T` — for both
  halves, and `torch.cat` them along dim 1. Keep it working for `k == 1` too (reshape
  to `(1, nel)`), so both branches share the same tail.
- `None` grads (a density-only constraint never touches `t`) become exact zeros, as
  today.

Keep the substance of `_grad_rows_batched`'s current docstring — why the cut is at the
filtered fields, why the hand-applied adjoint is `H` itself and not a reintroduced
hand-derived physics sensitivity. That reasoning is still exactly right and is the
thing a future reader will otherwise re-derive. Add the `k > 1` op restriction noted
above.

`x` and `t` must still be autograd leaves (`requires_grad_(True)`), because `xTilde`
and `tPhys` only carry grad if they are computed from tensors that do. The graph node
for the sparse matmul is simply never traversed.

### Phase 2 — one flattening convention

The `[density half; time half]` concatenation is currently written inline in five
places (`xmin`, `xmax`, `xval`, and once per half inside each grad helper). Add a small
module-level helper and use it everywhere:

```python
def _flatten_pair(density_part: Tensor, time_part: Tensor) -> Float[Tensor, " n"]:
```

This is the `flatten_design_vars` of the target design: it states once that the MMA
variable layout and the MMA gradient-column layout are the same layout by
construction. Use it in `_sensitivity_rows`' tail (concatenating the two `(k, nel)`
blocks — either give it a `dim` argument or keep the 2-D `cat` local, your call, but do
not leave two independent spellings of the ordering).

### Phase 3 — restructure `step`'s constraint block

Replace the parallel `fval_parts` / `dfdx_parts` accumulation with a single list of
value tensors, and derive `dfdx` from it:

```python
g_parts: list[Tensor] = [
    fv_vol[None],
    fv_cont[None],
    fv_start,          # (len(Nei),)
    stage_rows,        # (2 * nStage,), upper/lower interleaved
    fv_hotspot[None],
]
fval = torch.cat(g_parts)
dfdx = torch.cat([_sensitivity_rows(g, xTilde, tPhys, H, Hs) for g in g_parts], dim=0)
```

MMA row order is then stated exactly once, in one list, instead of being an invariant
of two lists appended to in lockstep. Keep a comment naming the order and pointing at
`tests/matlab_reference_loop.py` as the authority, as the current code does.

`df0dx` stays its own call: `_sensitivity_rows(f0val_t[None], ...)[0]`.

**Stage bounds.** Drop the hand-negation of the lower rows. Build the interleaved value
tensor and let autograd differentiate it:

```python
stage_upper = torch.stack([...])                       # (nStage,)
stage_rows = torch.stack([stage_upper, -stage_upper - 1.0e-5], dim=1).flatten()
```

`torch.stack(..., dim=1).flatten()` reproduces the current `upper_0, lower_0, upper_1,
lower_1, ...` interleaving — checked against the current construction and found
identical, but confirm it against the row-order test anyway. This costs `2 * nStage` seeds instead of `nStage` in one
vmapped call; measured at ~2 ms on CPU at 180x60, i.e. inside the noise of the FEM
solves.

Once the negation is gone, update `constraints.stage_volume_bounds`' docstring: it
currently instructs callers to build `fval_lower = -fval_upper - 1.0e-5` and
`dfdx_lower = -dfdx_upper` themselves. The value contract stays; the *sensitivity*
contract goes away.

### Phase 4 — documentation

- `sttopt/optimize.py` module docstring: it does not name the helpers, but check the
  "tensor boundary" paragraph still reads true.
- `benchmarks/bench_sensitivities.py:40` names `_grad_row`/`_grad_rows_batched`
  explicitly in its docstring. Update to the new helper name and keep the sentence's
  point (both sides of the comparison report a finished row in raw `x`/`t` space).
- Any `plans/archive/` reference is history — **do not** edit archived plans.
- Per `CLAUDE.local.md`: new and modified docstrings use Sphinx `:param:`/`:return:`
  format. Do not reformat docstrings you are not otherwise touching.

## What to test

Run with `pytest`. **Do not run the slow tests** (`tests/test_e2e_slow.py`) unless the
below gives you a reason to.

**Primary safety nets** — these are the tests that will actually catch a mistake:

1. `tests/test_optimize.py::test_step_assembled_sensitivities_match_finite_differences`
   — central differences against `step`'s own stacked `df0dx`/`dfdx`, parametrized over
   `nStage`/`tfield`/`Theta`. This is the single most important test in this refactor.
   It also asserts `record.dfdx.shape == (problem.m, problem.n)` and that no row is
   uniformly ~zero, so a mis-stacked or dropped row fails here.
2. `tests/test_e2e.py` — pins `fval`/`dfdx` at iteration 1 against `mma.npz`
   (`dfdx_1`, tier `algebraic`, rtol 1e-10) and the full trajectory against
   `dfdx_all`. This is the row-order and row-value regression net against the MATLAB
   source.
3. `tests/test_optimize.py::test_step_batched_matches_sequential_fem_solves` — the
   `batch_fem_solves` on/off paths must still agree on `dfdx`.
4. `tests/test_optimize.py::test_step_produces_no_nan_gradients_on_a_near_binary_snapshot`
   and `test_step_would_have_produced_nan_without_the_nan_safe_rewrite` — the cut set
   moves from the raw leaves to the filtered fields for four rows that previously used
   the leaf path, so re-confirm the nan-safety properties still hold.
5. `tests/test_constraints.py` — `stage_volume_bounds`' FD test asserts the lower row's
   sensitivity is exactly `-dfx`, `-dft`. That test exercises the *constraint function*,
   not `step`, so it should still pass unchanged. If you changed its docstring contract
   (Phase 3), check the test's comment at `tests/test_constraints.py:172` does not now
   describe something that no longer exists in `sttopt/`.

**Full run:** `pytest` (fast tests) must be green before and after. Capture the
before-state so a pre-existing failure is not mistaken for one of yours.

**Warnings are failures for review purposes.** Per `CLAUDE.local.md`, do not ignore new
warnings. The sparse-CSR beta warning and the CUDA-init warning already appear on this
machine; anything *new* must be explained.

**Extra check to add — bit-level agreement of the two branches.** Add a test that the
`k == 1` and `k > 1` branches of `_sensitivity_rows` agree, e.g. feed a 2-row output
and compare against two separate single-row calls, `assert_allclose` at a tight rtol.
This pins the property the whole design rests on: the dispatch on `k` is a performance
choice, not a semantic one.

**Device coverage.** The default device is CUDA when available. Run the suite on
whatever the machine offers, and additionally spot-check `test_step_assembled_
sensitivities_match_finite_differences` on CPU explicitly, since the CPU and GPU sparse
kernels differ (see the measurements above).

## What to benchmark

**The bar is "no substantial performance regression." Do not aim for a speedup and do
not report one as the goal.** A measured improvement is welcome but is not the
objective, and must not be traded against clarity.

Method (`benchmarks/bench_sensitivities.py`'s own docstring states the house rules —
follow them):

- `torch.cuda.synchronize()` around every timed region, warm-up discarded, median (or
  min) of several repeats.
- The machine must be idle. This cannot be verified from inside the script — say so
  explicitly when reporting, and per `CLAUDE.local.md` do not run this while another
  agent is running compute-intensive work.
- Same device, same dtype, same inputs on both sides. Do not compare CPU-before to
  GPU-after.

What to measure:

1. **`benchmarks/profile_step.py`** — wall time for a full `step`, before vs. after, on
   the default device. This is the number that matters.
2. Meshes: `90x30` and `180x60` at minimum. Add `360x120` if it runs without OOM;
   report `OOM` rather than aborting if it does not.
3. **Peak memory**, alongside wall time. The stage rows go from `nStage` to `2*nStage`
   vmap seeds, so peak memory in that one backward roughly doubles. Confirm it is
   immaterial next to the FEM solves rather than assuming it.
4. **Both `batch_fem_solves` settings**, since they produce different graphs.
5. Run on CPU as well as GPU. The cut-set change has a much larger effect on CPU (the
   CSC matvec above), so a CPU-only or GPU-only measurement will misrepresent the
   change in opposite directions.

Report a small before/after table. If any cell regresses by more than ~10%, stop and
report rather than proceeding — the refactor is not worth a real slowdown.

`benchmarks/bench_sensitivities.py` compares autograd against the hand-derived
reference rows and is *not* the right tool for this change's before/after; it is listed
here only because Phase 4 edits its docstring. Do not spend a long run on it.

## Risks and gotchas

- **Row order.** The single easiest way to break this silently is to get the stage
  upper/lower interleaving wrong. `test_e2e.py`'s `dfdx_all` comparison catches it;
  make sure that test actually ran.
- **`allow_unused`.** Rows that touch only one field must still produce an exact-zero
  block of the right shape, not a `None` that blows up in `torch.cat`.
- **`retain_graph`.** Every call still needs `retain_graph=True`; the graph is reused
  across every row and the objective.
- **Exact negation is gone.** `dfdx[lower] == -dfdx[upper]` will hold to floating-point
  rather than bit-for-bit. The fixture tolerance (rtol 1e-10) absorbs a ~1e-16 relative
  difference comfortably, but if any test asserts *exact* equality of those rows, that
  assertion is now testing the implementation and should be relaxed to `allclose` —
  flag it in the PR rather than silently loosening it.
- **`H` symmetry is now load-bearing for every row**, not just the batched ones. It is
  true by construction (`filters.density_filter`'s weight depends only on distance),
  and `filters` already has fixture tests. Consider a one-line assertion or a test that
  pins the symmetry, so the property is stated somewhere executable.
- If any Bash call fails with a sandbox/permission error, or a simple command is
  unexpectedly slow, treat it as a likely misconfiguration and report it as its own
  flagged line — do not quietly work around it.

## Definition of done

- One sensitivity helper; `_grad_row` and `_grad_rows_batched` are gone.
- `step` builds one list of constraint values and derives `fval` and `dfdx` from it.
- The stage lower rows come from autograd, not hand-negation.
- `_flatten_pair` (or equivalent) is the only place the `[density; time]` ordering is
  spelled out.
- Fast `pytest` suite green on CPU and on the default device, no new warnings.
- Before/after wall-time and peak-memory table, on CPU and GPU, showing no substantial
  regression.
- `step` is shorter than it was. If it is not, the refactor went wrong — say so rather
  than shipping it.
- Commits kept small and self-contained (Phase 1+2 as one, Phase 3 as another, docs as
  a third is a reasonable split). Open an undrafted PR.
