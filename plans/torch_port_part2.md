# Plan: PyTorch port, part 2 -- porting the optimization loop

> **Status (2026-08-27): not started; the user's open questions are answered** (see the
> final section). This plan replaces Phase 3 of
> `plans/archive/torch_port.md`, whose Phases 0a/0/1/2 are complete and whose GPU gate
> passed at 4.36x. Read that plan's *Results* sections for the measurements this one
> builds on; do not re-derive them.

## What part 1 established

- **The GPU solve wins.** `sttopt/torch_mg.py`'s MGCG beats `scipy.spsolve` by **4.36x**
  at 180x60 on a late near-binary design, warm-started, at matched accuracy (max
  element-wise `ce` error 1.1e-8 against `spsolve`). 12.85x at 360x120; a 0.72x *loss* at
  90x30.
- **Where the time goes**, per `optimize.step` at 180x60 / `nStage=8` on the `it0800`
  snapshot (3060 ms/step total): `spsolve` 41.1%, `mma.mmasub` 36.3%,
  `conductivity.hotspot_constraint` 17.6%, `fem.assemble_stiffness` 3.3%, everything else
  ~1.7%. So **MMA is nearly as expensive as the FEM solve** and hotspot is a real third.
- **Warm starting is worth ~25% of CG iterations**; batching the stage solves is worth
  1.3-1.4x at 90x30 and 180x60 and a small loss at 360x120.
- **Determinism was decided**: keep `index_add_`'s nondeterministic atomics (drift 1.4e-12
  on a full solve, far under the `solved` tier's 1e-6) and treat
  `torch.use_deterministic_algorithms(True)` as an opt-in debugging switch.

Existing torch code: `sttopt/torch_fem.py` (matrix-free operator, Jacobi-PCG, `pcg`),
`sttopt/torch_mg.py` (multigrid hierarchy, `VCycle`, `solve`), `tests/test_torch_fem.py`,
`benchmarks/{profile_step,bench_fem_solve,calibrate_cg_rtol}.py`. None of it is wired
into a production call site.

## Goal

Make `optimize.step` run end to end on the GPU in float64, with sensitivities from
autograd rather than hand-derived algebra, and with no NumPy round trip inside the loop.

**Done means:** `sttopt.cli` runs the production configuration (180x60, `nStage=8`,
`nloop=800`) entirely on the GPU; every existing test still passes (with the tolerance
changes this plan justifies below); the hand-derived sensitivity code is gone from
`sttopt/` and the independent oracles in `tests/` are what pin correctness.

### Non-goals

- **Any change to the physics or the algorithm.** This is a port. If the port surfaces a
  correctness question, log it in `plans/code_quality_review.md` and keep going -- do not
  fix it in a port commit (Phase 0a's NaN bug is the precedent: found here, fixed
  separately, documented separately).
- **A NumPy/torch dual backend in production.** See "Decisions taken up front".
- **Multi-GPU, float32, or mixed precision.** float64 throughout; the conditioning
  argument in part 1 has not changed.
- **Making 90x30 fast.** GPU MGCG loses there and that is accepted. The production mesh is
  180x60.

---

## Decisions taken up front

These are the choices that shape every commit below. Each is reversible, but flip-flopping
mid-port is not; settle them by reading the rationale rather than by re-litigating.

**1. Explicit `device`/`dtype` on `Problem`, not `torch.set_default_dtype`.**
Part 1's sketch proposed the global default. Reject it: a global dtype switch is a
side effect on the whole process, it fights this repo's CLI convention of deferring heavy
imports, and it makes a test that wants float32 or CPU impossible to write locally.
Instead `Problem` gains `device: torch.device` and `dtype: torch.dtype` fields, and every
tensor it holds is already on that device in that dtype. Nothing inside `step` should ever
call `.to(...)` -- if it needs to, a tensor was built in the wrong place.

**2. The NumPy production path is retired at the end, not kept as a backend.**
*Confirmed by the user (2026-08-27).* A machine without a GPU is not a reason to keep the
NumPy path: torch's own CPU device is the fallback, and Decision 1's explicit
`device`/`dtype` on `Problem` is what makes `device="cpu"` a first-class configuration
rather than a degraded mode. Nothing in this plan may assume CUDA is present -- the CPU
device must stay a working path for the whole loop, and it is what the non-GPU tests run
on. What is retired is the *second implementation*, not CPU support.

Keeping two full implementations doubles the maintenance surface and, worse, invites the
two to drift while both look tested. The correctness argument for deleting is that the
*oracle does not live in `sttopt/`*: `tests/matlab_reference.py` and
`tests/matlab_reference_loop.py` are an independent from-scratch transliteration of the
MATLAB source in its native 1-indexed F-order convention, and `tests/fixtures/*.npz` are
golden snapshots. Both survive the deletion untouched. So the sequence is: port, validate
against the oracle, then delete the hand-derived sensitivities in a *separate, later*
commit.

This is about the *production path*, not about hand-derived algebra as such: a hand-derived
backward inside a `torch.autograd.Function` is torch code on the GPU and stays if Phase
3.4's benchmark says it earns its place.

Two NumPy things stay on purpose:
- `sttopt/fem.py`'s `assemble_stiffness` / `solve_fe`. Small, and it is the solver oracle
  `tests/test_torch_fem.py` compares MGCG against. Deleting it would remove the only check
  that the matrix-free operator is the same operator.
- `sttopt/viz.py` and the fixture I/O. These are boundary code; they take arrays and should
  keep taking arrays.

**3. The FEM solve is a `torch.autograd.Function` with a hand-written adjoint, not
autograd through CG.** Differentiating through the CG iteration would be both wrong in
spirit (the iteration count is data-dependent) and ruinous in memory. The adjoint is
standard and is derived in Phase 3.3.

**4. `x` and `t` -- the raw MMA variables -- are the autograd leaves.** Today every
constraint function bakes the density-filter and Heaviside chain rules into its own
returned sensitivity (`H @ (... * dx / Hs)`, repeated in six places). Making the raw
variables the leaves deletes all six copies: the filter and the projection become ordinary
forward operations and autograd threads the chain rule. This is the single largest
simplification the port buys, and it is why `dx =
filters.heaviside_projection_derivative(...)` disappears entirely.

**5. Torch becomes a required dependency.** Move `torch` from
`[project.optional-dependencies]` to `dependencies` in `pyproject.toml` when Phase 3.2
lands, and delete the comment there explaining why it was optional.

---

## Risks

**Autograd reintroduces the Phase 0a NaN (high, and confirmed).** `hotspot_constraint`'s
forward computes `cond_p = (T_val * x**r) ** p` with `r = 0.05`. `d(x**r)/dx` is `inf` at
`x == 0`, and densities do reach exactly zero once `beta_d` saturates. The hand-derived
code sidesteps this by using the analytically cancelled diagonal form; **autograd will
not, and silently produces `nan` gradients on exactly the designs the optimizer spends its
late iterations on.** Measured on the naive transcription:

```
grad naive : [nan, 1.0415739507921168e-10, 1.4073748835532822e-10]
grad reform: [0.0, 1.0415739507921164e-10, 1.407374883553282e-10]
analytic   : [0.0, 1.0415739507921162e-10, 1.407374883553282e-10]
```

The fix is to write the forward in the already-cancelled form `T_val**p * x**(r*p)`, which
is identical in value (`1.3758776550327337e-10` vs `...34e-10`) and whose gradient is
finite because `r*p = 1.25 > 1`. `T_val = 1 - K_est` is in `[0, 1]` (`K_est` is a weighted
average of `x**q`), so the negative-base concern does not arise. The same rewrite applies
to `Tsub_pow`'s `(T_val[e2] * xb**r) ** (p - 1)`.

Generalize the lesson: **every `x**k` with `k < 1` in a forward pass is a NaN gradient
waiting for an exact zero.** Phase 3.4 must grep for them before trusting any autograd
output, and the guard already in `conductivity.py` (`if r * p < 1 and np.any(x == 0)`)
must survive the port, keyed to the quantity that is actually inconsistent.

**Autograd can be slower than the hand-derived algebra (high).** Reverse mode reuses
common subexpressions along the graph automatically, so the naive worry -- "the hand
version shares terms and autograd will not" -- is mostly unfounded. The real losses are
elsewhere, and this code has an instance of each:

- **Lost analytic cancellations.** These are asymptotic, not constant-factor, and they are
  the ones that actually hurt. The known case is `hotspot_constraint`'s diagonal
  self-heating term: the hand-derived form is `r * T_val**p * x**(r*p - 1)`, one
  *element*-sized expression (10800 entries), and getting there let Phase 0a delete a
  4.2M-entry *pair*-sized expansion that existed only to use 10800 of its entries.
  Mechanical differentiation of the pair block will not find that collapse, so it pays the
  full `npairs` cost on a term whose answer is `nel`-sized. Assume autograd is ~400x worse
  on this one term and check whether that matters in context.
- **Memory traffic for saved activations.** Every `npairs`-sized intermediate the forward
  saves is 34 MB read back during backward. The hand-derived version recomputes some of
  these and stores fewer.
- **A second linear solve per FEM solve.** Addressed in Phase 3.3, where it turns out to be
  avoidable.

Some loss is acceptable -- the GPU port is buying multiples, not percentages, and the
end-to-end target in Phase 3.7 is what the port is actually judged on. But it has to be
*measured* loss, not assumed-negligible loss. Phase 3.4 adds a dedicated benchmark and a
ranked set of escape hatches, of which "wrap the expensive block in an `autograd.Function`
with its hand-derived backward" is the main one: the choice is per-term, not
all-or-nothing.

**Full-Jacobian extraction (medium).** MMA needs a dense `dfdx` of shape `(m, n)` = `(79,
21600)` at production settings -- 79 rows of reverse-mode. The saving grace is that the
expensive graph nodes appear in exactly one row each (the FEM solves only in `f0val`, the
4.2M-pair hotspot algebra only in the hotspot row), and reverse mode only traverses the
subgraph reachable from the output being differentiated. So the *structure* is fine; the
question is constant factors. Plan for `torch.autograd.grad(..., is_grads_batched=True)`
per constraint group (one call covering the 60 start-point rows, one covering the 8
distinct stage-bound rows) rather than 79 separate calls.

**MMA's scalar control flow synchronizes the GPU (medium).** `subsolv` has a `while epsi >
epsimin` outer loop, a `while residumax > 0.9*epsi and ittt < 200` Newton loop, and an
inner backtracking line search whose condition is `resinew > residunorm` -- all reading
scalars that will live on the device. Each read is a sync. Expect on the order of a few
hundred syncs per `mmasub` call. At the ~20 us per sync these problem sizes suggest, that
is single-digit milliseconds and acceptable, but it is a floor that no amount of kernel
optimization removes. Do not try to make the loop conditions device-resident; the
resulting code would be unreadable for a few ms.

**Memory in the hotspot backward (medium).** `npairs = 4204240` at 180x60, so each saved
float64 pair-sized intermediate is 34 MB. The forward has roughly ten of them. ~340 MB of
saved activations against an 8 GB card is fine at 180x60, but 360x120 has 4x the pairs
(~1.4 GB) and is worth checking rather than assuming. `torch.utils.checkpoint` on the pair
block is the escape hatch if it bites.

**E2E trajectories will not reproduce bitwise (accepted, but must be handled
deliberately).** CG at `rtol = 1e-8` is not `spsolve`, atomics are nondeterministic at
1e-12, and autograd's summation orders differ from the hand-derived expressions'. The
existing `conftest.e2e_rtol` (one decade looser per iteration, capped at 1e-2) was written
for exactly this amplification and should absorb a 3-iteration E2E comparison. An 800-loop
comparison is a different question and is answered in Phase 3.7 by comparing *aggregate
behaviour*, not trajectories.

**The `[project.scripts]` entry point is already broken** (`sttopt = "sttopt.cli:main"`,
but `main` requires an `args` argument). Not caused by this port and not this port's
job -- log it in `plans/code_quality_review.md`.

---

## Execution notes

- **The GPU is not reachable from the normal sandbox.** Run anything CUDA through the
  `gpu-exec` MCP server, which executes as the `claude` user. Confirmed working from a
  worktree: `torch 2.12.1+cu132`, `NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU`.
- **One compute-heavy job at a time** (repo rule), and part 1 learned the hard way that a
  contended machine inflates a baseline. Any timing number in this plan's results sections
  must come from an otherwise idle box, and say so.
- **Small, self-contained commits.** Each numbered phase below is one or a few commits, not
  one "port" commit. Undrafted PRs.
- **The slow test runs exactly twice** (user, 2026-08-27): `pytest -m slow`
  (`tests/test_e2e_slow.py`, the 800-iteration thesis 4.4 reproduction) once **before the
  first port commit**, to capture the pre-port baseline, and once **at the end of Phase
  3.7**, to confirm no regression. Record both runs' assertion quantities *and* wall clock
  in Phase 3.7's results. Do not run it at phase boundaries or to debug: it is minutes to
  hours, it is deselected by default for that reason, and the per-phase check is the
  ordinary suite plus the fixture oracles. The baseline run is the one thing in this plan
  that must happen before Phase 3.1, so do it first and record the numbers even though
  nothing has changed yet -- after the port there is no way to go back and take it.

### Results: pre-port slow-test baseline (captured retroactively, 2026-08-28, commit 97ef372)

The baseline capture above was missed: Phases 3.1 through 3.6 were merged (through PR #50)
without it ever having been run. This section records it anyway, captured late rather than
lost, since git kept the exact pre-port code. `97ef372` (the merge just before Phase 3.1's
first commit, `26d89f9`) is byte-identical pre-port code, checked out detached in an isolated
worktree (no phase-3.x branches touched) on an otherwise idle machine (load average ~0.6 on a
16-core box; this pre-port code is NumPy/SciPy only and runs on CPU, so no GPU check applied).

`pytest -m slow -v tests/test_e2e_slow.py` at `97ef372`:

- **Result:** 1 passed.
- **Wall clock:** 1895.33s (0:31:35).
- The test only asserts bounds, not the values themselves, so a second, instrumented run
  (same parameters, same commit, calling `optimize.run` directly and printing
  `record.f0val` / `record.tru_max`) was made immediately after to get concrete numbers:
  - **Wall clock:** 1747.22s (0:29:07).
  - **f0val:** 192.84432887244273 (assertion window was 185 < f0val < 195).
  - **tru_max:** 0.8005839213907026 (assertion window was |tru_max - 0.8| <= 0.008).

Phase 3.7's end-of-port run should diff against `f0val ~ 192.84` and `tru_max ~ 0.8006`, not
just re-check the same pass/fail bounds -- those bounds are loose enough to hide a regression
this baseline's concrete numbers would catch.

---

## Phase 3.1: Device/dtype plumbing and the tensor boundary

The scaffolding every later phase needs, with no behaviour change to the optimizer.

- Add `device` and `dtype` to `Problem`. `build_problem` gains keyword-only `device="cpu"`,
  `dtype=torch.float64` so the default is CPU-and-float64 and every existing caller keeps
  working.
- Convert `Problem`'s array fields to tensors on that device: `KE`, `edofMat`, `freedofs`,
  `F`, `Hs`, `e1`, `e2`, `w`, `Nei`. `H`, `L`, `C` become sparse CSR tensors
  (`torch.sparse_csr_tensor`); note that torch's sparse CSR matmul supports the
  `sparse @ dense` and `sparse.T @ dense` forms this code uses, but check `H @ (nel, k)`
  for the start-point constraint specifically rather than assuming.
- Add `free_mask` (from `torch_fem.free_mask`) to `Problem` alongside `freedofs` -- the
  matrix-free path wants the mask, `fem.solve_fe` wants the index array, and both exist
  during the transition.
- `State`'s fields become tensors.
- Small conversion helpers for the boundary: fixtures load as arrays, `viz` and the CLI's
  printing take arrays. Put them somewhere obvious (`sttopt/torch_util.py`), not scattered.

**Tests.** No new physics tests. Assert that `build_problem` puts everything on the
requested device with the requested dtype, and that a `Problem` built on CUDA has no
lingering CPU tensor (walk the dataclass fields).

## Phase 3.2: Port the leaf math to torch, keeping the hand-derived sensitivities

This is the bulk of the mechanical work and the safest part of the port, because the
existing fixtures are an exact oracle: every function here is the same formula in a
different array library, so **`algebraic`-tier agreement (rtol 1e-10) is the bar**, not
`solved`.

Port, in roughly this order (each its own commit):

1. `filters.py` -- `heaviside_projection`, `heaviside_projection_derivative`. The filter
   *builders* (`density_filter`, `continuity_filter`) stay NumPy/SciPy and gain a
   conversion at the end; they run once, in `build_problem`, and SciPy's COO assembly is
   clearer than torch's.
2. `timefield.py`, `gravity.py` -- same treatment: builders stay NumPy, outputs convert.
3. `compliance.py`'s `time_mask` / `time_mask_derivative` / `_element_strain_energy`.
4. `constraints.py` -- all four constraints, sensitivities still hand-derived.
5. `conductivity.py` -- `_pairwise_sigmoid_terms`, `_conductivity_core`,
   `_conductivity_terms`, `estimated_conductivity`, `hotspot_constraint`. `np.add.at`
   becomes `Tensor.index_add_`. Keep the overflow-safe `exp(-|z|)` forms; keep the
   cancelled diagonal term.
6. `optimize.step`'s wiring.

`mma.py` and the FEM solve are deliberately *not* in this list -- they are Phases 3.5 and
3.3.

**A note on `np.add.at` -> `index_add_`.** Part 1 measured `np.add.at` at under 1% of step
time, so this is not a performance change; it is a portability one. But `index_add_`'s
atomics are the nondeterminism source part 1 characterized, so this is the commit that
introduces run-to-run drift into the constraint sensitivities. Expect it, and check the
drift is at the 1e-12 level rather than something larger.

**Tests.** Every existing test in `test_filters.py`, `test_compliance.py`,
`test_constraints.py`, `test_conductivity.py`, `test_gravity.py`, `test_timefield.py`
must pass with tensors flowing through, at their current tiers. The cheapest way to get
there is to make `conftest.assert_close` accept tensors (`np.asarray` already almost does;
a `.detach().cpu().numpy()` for tensor inputs finishes it) rather than converting at every
call site. `test_reference_sweep.py` -- the independent MATLAB oracle -- is the one that
matters most here; it must pass unchanged.

**Deliverable.** The loop runs on the GPU except for the FEM solve and MMA. Do not
benchmark this intermediate state and do not report a number from it: part 1's
round-trip argument says a hybrid loop is the worst of both worlds, and a timing from here
would be misleading.

## Phase 3.3: The FEM solve as an autograd Function

**The adjoint.** For `K U = F` with `K = sum_e d_e KE` symmetric positive definite on the
free dofs, and a downstream scalar `L(U)` with `g = dL/dU`:

```
lambda = K^-1 g                        (same operator, so the same MGCG hierarchy)
dL/dF   = lambda
dL/dd_e = -(lambda_e @ KE) . U_e       (elementwise contraction over the 8 element dofs)
```

Two consequences worth stating because they are free tests:

- For the compliance objectives, `lambda` is a multiple of `U` and the adjoint solve is
  free -- see below. `dL/dd_e` then reduces to `-ce`, which the existing fixtures already
  pin. **Assert this.**
- `gravity_compliance`'s "extra adjoint term" and its factor of 2 -- currently hand-derived
  as `dcx2`/`dct2` and explained in a paragraph of docstring -- fall out of `dL/dF` with no
  special-casing, because the gravity load is just another differentiable function of the
  density. That paragraph gets deleted, not ported.

**The adjoint's cost, and why it is nearly zero here.** A generic adjoint runs a second CG
solve per forward solve, which would take the loop from 9 solves per iteration to 18 --
the largest single performance risk the switch to autograd carries, and one that would
eat a good fraction of part 1's 4.36x. It does not apply here, because compliance is
self-adjoint. The scalar both compliance functions compute is `sum_e simp_e * ce_e` with
`ce_e = Ue^T KE Ue`, whose derivative with respect to `U` is `2 K U = 2 F` -- so
`lambda = 2 U` exactly, with no solve. Verified at 6x4 against the real
`assemble_stiffness`/`solve_fe` path:

```
max|g - 2F| : 4.66e-14   (against |2F| = 2.0)
alpha       : 2.0000000000001346
```

Rather than special-casing the compliance objective inside `FemSolve` -- which would be a
correctness trap the moment a non-compliance scalar is differentiated -- take the general
route that happens to be exact here: **warm-start the adjoint CG from the best multiple of
`U`**, `alpha = (U . g) / (U . F)`, which is the least-squares fit of `alpha*U` to
`K^-1 g` and costs two dot products. When `g` is parallel to `F` it lands on the exact
answer and CG returns at iteration zero (`pcg` already checks convergence *before* the
first iteration, for exactly this reason). When it is not, it is still a good warm start
and the solve is correct. Assert the iteration count is zero for the compliance case, so
a future non-self-adjoint objective shows up as a performance change rather than silently.

**Implementation.**

- `class FemSolve(torch.autograd.Function)` in a new `sttopt/torch_solve.py` (not in
  `torch_fem.py`, which is the operator layer and should stay free of autograd).
  `forward(density, F, ...)` -> `U`; `backward` runs one MGCG solve against the same
  operator, warm-started from the saved `U`. Save `U` and `density` for backward; rebuild
  or cache the hierarchy (see below).
- **Batch all nine solves in one call** at 180x60: `F` of shape `(9, ndof)` and `density`
  of shape `(9, nel)` -- `whole_compliance`'s row plus `nStage` gravity rows. `torch_fem`'s
  broadcasting contract already supports this. Part 1 measured batching as a 1.3-1.4x win
  at 180x60 and a small loss at 360x120, so make it a `Problem` field
  (`batch_fem_solves: bool`) defaulting to on at or below 180x60, rather than
  unconditional.
- **Warm start** from the previous iteration's `U`, carried on `State`. Part 1 measured
  ~25% of the iterations. The adjoint solve warm-starts from the forward `U`.
- **Hierarchy reuse.** The hierarchy build is 7.2 ms of a 30.4 ms solve (~24%). The forward
  and backward of one solve share the same `K`, so the backward must reuse the forward's
  hierarchy rather than rebuilding it -- that alone recovers ~24% of the adjoint cost.
  Across solves it cannot be reused (nine different densities), as part 1 established.
- **`CGConvergenceError` must propagate.** No silent fallback to `spsolve`, no returning an
  unconverged `U`. If the backward solve fails, that is a real signal.

**Tests.**

1. **`torch.autograd.gradcheck` on `FemSolve`** at a small mesh in float64 -- torch's
   built-in finite-difference check, and the direct test that the adjoint is right. It
   works, but only inside a narrow envelope; the four constraints below were each measured
   at 4x3 on the GPU rather than guessed, and a future agent that hits one of them should
   adjust the test rather than conclude gradcheck is unusable here.

   - **Passes at a moderate density field** with stock settings (`eps=1e-6`, `atol=1e-5`,
     `rtol=1e-3`) and the solver at its production `rtol = 1e-8`. No special tuning needed.
   - **Cannot validate a near-binary design, and this is a property of finite differences,
     not of the adjoint.** At the `Emin = 1e-9` contrast, even a 4x3 mesh gives
     `cond(K) = 1.4e10` and `|U|max = 2.3e9`, so the central-difference roundoff floor is
     `eps_machine * |U| / eps_step ~ 5e-1` absolute -- five orders above gradcheck's
     `atol`. Measured error is ~2e-1, on *solid* elements as much as void ones. Tightening
     the solver `rtol` does not help, because the solver is not the inaccurate side. So
     **gradcheck runs on a well-scaled field only**, and near-binary correctness is pinned
     by test 2 instead -- which is why test 2 is not redundant with this one and must not
     be dropped in favour of it.
   - **Run it with warm starting disabled.** gradcheck calls the forward many times and a
     warm start carried in external state makes it path-dependent; a configuration that
     passes cold fails warm.
   - **Leave `check_batched_grad` at its default `False`.** Opting in fails with
     `Batching rule not implemented for aten::is_nonzero`, because `pcg`'s
     `if torch.all(rel_resid <= rtol)` is Python control flow that vmap cannot trace. The
     same limit means Phase 3.4's `is_grads_batched=True` Jacobian extraction must not span
     a `FemSolve` backward -- it does not, as planned, since the batched groups
     (start-point and stage-bound rows) have no FEM dependence, but do not "optimize" the
     objective row into that path later.

2. Gradients from `FemSolve` against the hand-derived `dcx` from the *current*
   `compliance.whole_compliance` and `gravity_compliance`, element-wise on max relative
   error (not a norm -- part 1's Phase 1 made this point and it still holds; MMA reads
   every element). `solved` tier. **Run this one at the near-binary snapshots**, per test
   1's second bullet: it is the only check that covers the designs the optimizer actually
   spends its late iterations on.
3. `lambda == 2 U` for both compliance cases, to `algebraic` tier, **and the adjoint solve
   reports zero CG iterations** -- the check that the self-adjoint shortcut is actually
   being taken and not merely available.
4. The batched `(9, ndof)` path against nine sequential single solves. **Blocked on a
   prerequisite -- see below.**
5. Warm-started vs cold-started results agree to `solved` tier, and the warm one uses
   fewer iterations on a real consecutive snapshot pair (the fixture stores loop 799 next
   to loop 800 for exactly this).
6. A non-convergent case still raises through the autograd boundary.

**Prerequisite: fix `pcg`'s zero-`b` NaN first.** `torch_fem.pcg`'s docstring records a
"Known issue, not yet fixed": a batch member whose residual hits exact zero before the
others -- an all-zero `b` being the given example -- turns `alpha` into a `0/0` NaN that
poisons that member, and `CGConvergenceError` then misreports the failure as `nan` / `[]`
because NaN comparisons are always `False`. That reads like a corner case. It is not: it
**fires deterministically on any gradcheck of a batched `FemSolve`**, because gradcheck
seeds the backward with one-hot `grad_output`, which leaves *every other batch member*
with an all-zero right-hand side. Confirmed directly at 4x3 -- a two-member batch with
member 0's `b` zeroed raises with `worst relative residual nan` at every tolerance tried.

So test 4 cannot pass until the issue is fixed, and the fix is small: freeze a converged
member (zero its `alpha`) rather than dividing, or mask converged members out of the
update. Do it as its own commit before the batched work, not folded into it. Worth noting
this is also a latent production hazard, not only a test one -- an early gravity stage
whose time mask leaves nothing active has a near-zero load vector.

## Phase 3.4: Autograd replaces the hand-derived sensitivities

Now that the leaves are `x`/`t` and the solve is differentiable, delete derivative code.
One commit per module, each keeping the hand-derived version alive until the following
commit removes it.

**Before writing anything: apply the NaN-safe rewrite from the Risks section.** Grep the
forward passes for fractional powers. Known instances: `x**r` and `(T*x**r)**p` and
`(T[e2]*xb**r)**(p-1)` in `conductivity.py`. `simp_density`'s `x**penal` (penal 3) and
`xtJoint**(penal-1)` (power 2) are safe. Check `heaviside_projection` and the sigmoids for
saturation-induced zero gradients while you are there -- those are not NaN, but a silently
zero gradient at `beta_d = 128` is its own failure mode and is worth a test.

Order:

1. **Compliance** (`whole_compliance`, `gravity_compliance`). Smallest, and Phase 3.3 has
   already validated the hard part.
2. **The four constraints.** All four lose their `dx`/`H`/`Hs` arguments; `global_volume_fraction`
   and `stage_volume_bounds` lose their `dx` chain rule; `start_point` loses its one-hot
   `ss` matrix construction entirely (it becomes `tPhys[Nei]`). Note that
   `stage_volume_bounds`' lower row is exactly the negation of the upper -- keep that as an
   explicit negation rather than differentiating it twice.
3. **The hotspot constraint.** The big one. Port it, then read the benchmark below before
   deciding whether it keeps its autograd gradient. Do not port it on principle.

### Benchmark: autograd against the hand-derived sensitivities

**`benchmarks/bench_sensitivities.py`**, and it must be written and run *during* this
phase, not after it. Each commit here keeps the hand-derived version alive until the
following commit removes it, and that overlap is the only window in which the two can be
timed against each other on the same machine, the same device and the same inputs. Once
Phase 3.7 deletes the hand-derived code the measurement is no longer available, and "we
never checked" becomes permanent.

**Method.** For each of the six sensitivity-producing call sites -- `whole_compliance`,
`gravity_compliance`, the four constraints, and `hotspot_constraint` -- time the
hand-derived value-plus-sensitivity call against the autograd forward-plus-backward, at
90x30 / 180x60 / 360x120, on the `it0800` near-binary snapshots. Report forward and
backward separately: a slow backward and a slow forward have different fixes. Report peak
memory alongside, since the `npairs` activations are the other thing that can bite.
Standard hygiene from part 1 applies -- `torch.cuda.synchronize()` around every timing
region, discard warm-up, idle machine, and say the machine was idle.

**The methodological trap to avoid**, in the spirit of the two part 1 recorded: **both
sides must be torch, on the same device.** Timing autograd-on-GPU against the original
NumPy-on-CPU hand-derived code measures the port and the autodiff together and cannot
separate them -- and it would flatter autograd badly. Phase 3.2 exists in part to make
this comparison fair: it leaves a torch hand-derived implementation in place.

**What to do with the answer.** There is no pass/fail here; the gate is the end-to-end
number in Phase 3.7, and some loss is expected and acceptable given the GPU port is buying
multiples. The per-module table's job is to say *where* to spend effort if the end-to-end
number comes in under target. Escape hatches, in increasing order of how much hand-written
code they cost:

1. **`torch.compile` the forward.** Part 1 found `hotspot_constraint`'s cost is
   `_pairwise_sigmoid_terms` plus the surrounding `npairs`-sized elementwise algebra --
   exactly the shape inductor fuses well, and fusing the forward shrinks the saved
   activations too. Try this before writing anything by hand.
2. **A custom `autograd.Function` around one block, with its hand-derived backward.**
   This is the granular middle ground, and the reason the choice is per-term rather than
   all-or-nothing: `hotspot_constraint`'s diagonal self-heating term is a `nel`-sized
   analytic expression that mechanical differentiation will compute as a `npairs`-sized
   one, so wrapping the pair block and supplying the cancelled backward keeps autograd
   everywhere else while recovering the one place it is asymptotically worse. Phase 3.3's
   `FemSolve` is the same pattern and the template to copy.
3. **`torch.utils.checkpoint` on the pair block.** Trades recompute for activation memory
   -- the opposite trade to (1), and the right one only if memory rather than time is the
   binding constraint (likely at 360x120, where the pair activations are ~1.4 GB).
4. **Keep the hand-derived gradient for that row.** Always available, and not a failure.
   If it is taken, record the measurement that justified it here, so the decision is
   revisitable when a later torch version changes the arithmetic.

**Jacobian assembly.** `optimize.step` needs `dfdx` as a dense `(m, n)`. Write one helper
that takes a group of constraint outputs and the `(x, t)` leaves and returns the
corresponding rows, using `torch.autograd.grad(..., is_grads_batched=True)` with a batch of
one-hot seeds so the 60 start-point rows and the 8 stage rows are one call each rather than
68. `f0val` and the hotspot row are single `grad` calls. Assemble in the reference loop's
exact row order, which `optimize.step` already documents and `test_optimize.py` already
pins.

**Tests.**

- Every autograd sensitivity against its hand-derived predecessor, element-wise max
  relative error, `solved` tier for anything downstream of the solve and `algebraic` for
  anything not. This is the whole point of not deleting them in the same commit.
- Finite-difference checks at a small mesh for the hotspot row specifically, since its
  algebra is the least trivial and the FD check is independent of both implementations.
- **A near-binary regression test with exact zeros in `xPhys`**, asserting no `nan` in any
  returned gradient. This is the test that would have caught Phase 0a's bug, and it is the
  test that catches its autograd resurrection. Take the field from
  `tests/fixtures/torch_port_designs.npz`'s late snapshots.
- `test_reference_sweep.py` still passes.

### Results: `bench_sensitivities.py` (2026-08-28, machine idle, RTX PRO 1000 Blackwell
Laptop GPU, 8 GB)

An earlier run of this benchmark (recorded only in `plans/phase3.4_handoff.md`, since
superseded and deleted) had a fairness bug: the four constraints' and
`hotspot_constraint`'s hand-derived functions bake the density-filter/Heaviside chain
rule (`H`/`Hs`/`dx`) into their returned sensitivity, while their `*_value` autograd
counterparts deliberately stop at `d(.)/d(xPhys)` (Decision 4 makes the filter an
ordinary forward op for the *caller*, i.e. `optimize.step`, to differentiate through).
Timing the hand side's finished row against the autograd side's unfinished one made
autograd look better than it is. Fixed in `benchmarks/bench_sensitivities.py` by
finishing the same chain (`H @ (... * dx / Hs)` for density, `H @ (.../Hs)` for time)
after every autograd backward, timed in the same region -- matching what
`optimize.step`'s `_grad_row`/`_grad_rows_batched` actually compute for these rows.
`whole_compliance`/`gravity_compliance` needed no fix: `compliance.py`'s hand-derived
`dcx`/`dct` were never finished rows either, so both sides already stopped at the same
place. See the benchmark script's own docstring for the full reasoning.

While re-running, `hotspot_constraint`'s hand-derived *and* naive-autograd forward both
turned out to **OOM at 360x120 on this 8 GB card**, independent of hand vs. autograd:
`conductivity.neighbor_weights`'s `npairs` scales ~16.35x from 180x60 to 360x120
(`4204240` -> `68744592`), not the Risks section's assumed 4x, because both element
count and the conductivity filter radius double between those meshes. Each
`npairs`-sized float64 intermediate is therefore ~550 MB, and several are live at once
in the pairwise-sigmoid algebra regardless of which implementation is used -- not a
benchmark-script memory leak (a `_cleanup` between cells, with `gc.collect()` +
`torch.cuda.empty_cache()`, was added and ruled this out; memory is fully released
between meshes) and not fixed by `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(tried, made no difference -- the failure is a genuine peak-memory shortfall, not
fragmentation). Escape hatch 1 (`torch.compile` on `hotspot_value`'s forward) resolves
it completely -- inductor's fusion means the elementwise pair algebra never
materializes most of those intermediates -- so `bench_sensitivities.py` gained a third
`"compiled"` mode for the hotspot cell, and any cell that still OOMs prints `OOM`
instead of aborting the run.

```
=== 90x30 ===
site                  mode           fwd(ms) fwd+bwd(ms)  peak(MB)
------------------------------------------------------------------
whole_compliance      hand             47.43       47.47      91.8
whole_compliance      autograd         47.05       48.16     123.8
gravity_compliance    hand             27.50       27.98     123.9
gravity_compliance    autograd         27.04       28.93     123.9
global_volume_fraction hand             0.07        0.06      71.4
global_volume_fraction autograd         0.02        0.45      71.4
time_field_continuity hand              0.26        0.25      72.9
time_field_continuity autograd          0.05        0.79      72.9
start_point           hand              0.14        0.13      73.8
start_point           autograd          0.02        0.47      71.4
stage_volume_bounds   hand              0.17        0.16      71.5
stage_volume_bounds   autograd          0.06        0.59      71.4
hotspot_constraint    hand              2.34        2.33     101.7
hotspot_constraint    autograd          0.34        1.68     100.0
hotspot_constraint    compiled          0.26        0.84      71.8

=== 180x60 (production mesh) ===
site                  mode           fwd(ms) fwd+bwd(ms)  peak(MB)
------------------------------------------------------------------
whole_compliance      hand             51.78       50.54     225.4
whole_compliance      autograd         50.38       51.28     225.4
gravity_compliance    hand             46.66       45.82     225.8
gravity_compliance    autograd         45.71       46.70     225.9
global_volume_fraction hand             0.09        0.09     171.6
global_volume_fraction autograd         0.02        0.29     171.6
time_field_continuity hand              0.30        0.30     177.5
time_field_continuity autograd          0.05        0.79     177.5
start_point           hand              1.76        1.75     191.3
start_point           autograd          0.02        0.52     171.6
stage_volume_bounds   hand              0.21        0.21     171.9
stage_volume_bounds   autograd          0.06        0.66     171.7
hotspot_constraint    hand             44.24       44.18     657.0
hotspot_constraint    autograd          7.39       32.90     629.1
hotspot_constraint    compiled          2.26        7.00     176.1

=== 360x120 ===
site                  mode           fwd(ms) fwd+bwd(ms)  peak(MB)
------------------------------------------------------------------
whole_compliance      hand             96.28       96.34    1836.7
whole_compliance      autograd         96.28       97.83    1836.7
gravity_compliance    hand             87.12       87.43    1838.0
gravity_compliance    autograd         87.03       89.08    1838.6
global_volume_fraction hand             0.61        0.61    1777.1
global_volume_fraction autograd         0.03        1.04    1776.8
time_field_continuity hand              1.12        1.12    1800.7
time_field_continuity autograd          0.07        1.39    1800.7
start_point           hand             31.72       31.74    1936.8
start_point           autograd          0.02        1.04    1777.1
stage_volume_bounds   hand              1.25        1.35    1778.1
stage_volume_bounds   autograd          0.06        1.66    1777.4
hotspot_constraint    hand               OOM
hotspot_constraint    autograd           OOM
hotspot_constraint    compiled          29.41      106.62    1844.3
```

**Decision (2026-08-28): `optimize.step` keeps its plain `hotspot_value` autograd
call, no code change.** At the production mesh, 180x60, plain autograd already beats
the hand-derived function on every column that matters: forward 7.39 ms vs. 44.24 ms,
forward+backward 32.90 ms vs. 44.18 ms (~1.34x faster, reversing the Risks section's
"assume autograd is ~400x worse" prediction -- apparently the lost diagonal-term
cancellation is swamped by reverse-mode's automatic reuse and by the hand-derived path
materializing several pair-sized arrays autograd's graph does not), and peak memory is
comparable (629 MB vs. 657 MB). This is "good enough" per the standing instruction, so
no escape hatch is needed for the mesh the plan's Done criteria actually target, and
`sttopt/optimize.py::step` is left as-is (it already calls `hotspot_value`, wired in
commit `2b0a318`).

360x120 is a different story: neither hand-derived nor plain autograd runs there at
all on this 8 GB card (both OOM, as above), and `torch.compile` (escape hatch 1) is
what fixes it -- 29.41 ms / 106.62 ms fwd/fwdbwd at 1844 MB peak, no NaN. 360x120 is
not one of this plan's stated production settings (Phase 3.7's target is 180x60), so
this phase does not wire `torch.compile` into `optimize.step` -- doing so is a
one-line change (`torch.compile(conductivity.hotspot_value)`) with no downside
observed at 180x60 either (2.26 ms / 7.00 ms fwd/fwdbwd there, faster and lower-memory
than plain autograd), but it needs its own test pass (compiled numerics are not
guaranteed bit-identical, so the existing exact/near-exact comparisons against the
hand-derived predecessor and the FD checks would need re-running against the compiled
path) which is out of this phase's scope. Recorded here as a ready-made option for
whoever next needs 360x120 to run, or for Phase 3.6's performance pass, which already
lists `torch.compile`/fusing hotspot's pairwise algebra as an action item.

This did **not** trigger the "stop and report" condition: plain autograd is good
enough at the production mesh without any escape hatch, and 360x120's fix (`torch.compile`)
is escape hatch 1, which the standing instruction pre-approves.

## Phase 3.5: MMA to torch

`mma.py` is a verbatim port of Svanberg's code and stays verbatim -- translate array
library, change nothing else. It is 36.3% of step time and pure dense algebra, so this is
the second-largest win available.

- `np.linalg.solve` -> `torch.linalg.solve` on the `(m+1, m+1)` bordered system (80x80 at
  production -- small enough that it may be *faster* on the CPU; measure, and if so, that
  one solve is a defensible exception to the no-round-trips rule since it moves 51 KB).
- `np.diag(diaglamyi) + (GG * (1/diagx)) @ GG.T` is the dominant flop: `(79, 21600) @
  (21600, 79)`. Trivially GPU-friendly.
- Keep both the `m < n` and `m >= n` elimination branches. The `m >= n` branch is
  documented as smoke-tested only; do not "clean it up" while porting it.
- The `warnings.warn` on hitting the Newton cap stays.

**Scalar reads.** `residumax`, `residunorm`, `resinew`, `steg` and the `.max()` calls in
the fraction-to-the-boundary step all become device scalars read by Python control flow.
Let them sync; see the Risks section.

**Tests.** `test_mma.py` (MATLAB fixture) and `test_mma_toy_problems.py` unchanged, at
their current tiers. `subsolv` is an interior-point method with a line search, so if a
tolerance needs loosening, loosen it with a measurement and a sentence saying why -- an
unexplained tolerance bump here would hide a real translation error.

### Results: bordered-solve CPU-vs-GPU measurement (2026-08-28, machine idle, RTX PRO
1000 Blackwell Laptop GPU, 8 GB)

`torch.linalg.solve` on a random well-conditioned SPD-plus-border 80x80 float64 system
(the production `(m+1, m+1)` size), 3000 iterations after 20 discarded warm-up calls,
`torch.cuda.synchronize()` bracketing every timed region:

```
n=80  cpu-only solve:                 27.33 us
n=80  cuda-only solve:               172.82 us
n=80  cuda tensors, cpu solve+xfer:    63.27 us
```

**Decision: solve on the CPU, with the round trip.** At this size, kernel-launch
overhead dominates the GPU solve (172.82 us) -- more than 6x the CPU-only solve
(27.33 us) -- and even paying to move the ~51 KB `AA`/`bb` off the GPU and the ~640 B
result back (63.27 us total) is still under half the pure-GPU cost. `sttopt/mma.py`'s
`subsolv` therefore does `torch.linalg.solve(AA.cpu(), bb.cpu())` for both the `m < n`
and `m >= n` branches, per the plan's carve-out for a solve this small.

## Phase 3.6: Performance pass

Only after the loop is correct end to end. Part 1 left a ranked list of solver
optimizations; re-rank it against a fresh profile of the *torch* `step`, because the
proportions will have moved. Carried forward, in part 1's order of expected return:

1. **CUDA graphs / `torch.compile(mode="reduce-overhead")` on the CG iteration body.** At
   180x60 a V-cycle over 3 levels launches ~40-60 kernels per iteration on a problem whose
   arithmetic is microseconds, so launch overhead plausibly still sets the time. `pcg`'s
   in-place updates are already a precondition for capture. The data-dependent iteration
   count means capturing one iteration and replaying it, not the whole solve.
2. **Let converged batch members drop out.** `pcg` runs every member to the slowest
   member's count -- 22 iterations for stages that individually need 13 at 180x60, and 11
   to 35 at 90x30. This also fixes the "batching is a loss at 360x120" result properly,
   rather than by the per-mesh switch Phase 3.3 puts in. Note this is the same machinery
   as Phase 3.3's zero-`b` NaN prerequisite, approached from the performance side: freezing
   a converged member is what both need, so build it once there and extend it here rather
   than writing two variants.
3. **Re-tune multigrid against real designs.** `MAX_COARSE_ELEMENTS = 700`, `omega = 0.6`,
   `n_smooth = 2` were chosen against a *synthetic* hard 0/1 field at 90x30, and the real
   near-binary designs behave differently (90x30 is harder than 180x60, which the synthetic
   proxy did not predict). Sweep `(max_coarse_elements, n_smooth, omega, gamma)` against
   the `it0800` snapshots, timing as well as counting iterations -- the two disagree:
   `max_coarse_elements = 700` bottoms out at 45x15 and costs 6.9 ms/solve of hierarchy
   build against 0.4 ms at 200, but a coarser bottom cost 119 iterations against 31 in part
   1's measurement.
4. **A cheaper coarse solve.** The coarsest level is a dense Cholesky on a 1472-dof
   operator, per solve, per batch member. If the sweep wants to keep a large coarse grid,
   replacing it with a few CG iterations or a sparse factorization cuts most of the
   hierarchy build's 24% share.
5. **Fuse the pair-sized elementwise algebra in `hotspot_constraint`.** Part 1 found the
   cost is `_pairwise_sigmoid_terms` plus the surrounding `npairs`-sized elementwise work
   -- not the `np.add.at` reductions, contrary to the prior. That is exactly the shape
   `torch.compile` fuses well.

Also re-open the **CG tolerance** here. `rtol = 1e-8` was calibrated in part 1 against
sensitivity accuracy with the hand-derived formulas. The adjoint solve is a second solve
whose error compounds with the forward's, so re-run `benchmarks/calibrate_cg_rtol.py`'s
method against the autograd gradients before assuming 1e-8 still holds. It may need to be
tighter; it may (less likely) be loosenable, which would be a straight speed win.

**Status (2026-08-28): only the CG-tolerance re-evaluation below is done, per explicit
user direction to skip the performance pass for now.** Items 1-5 above (CUDA graphs,
dropping converged batch members, multigrid re-tuning, a cheaper coarse solve, hotspot
fusion) are all still open and unstarted; nothing in this phase's performance-pass list
has been touched. Phase 3.6 is **not** complete.

### Results: CG rtol re-calibration for autograd (2026-08-28, machine idle, RTX PRO 1000
Blackwell Laptop GPU, 8 GB)

`benchmarks/calibrate_cg_rtol.py`'s table only ever exercised the forward solve --
it reads the hand-derived `dcx`/`dct` off an already-solved `U`. `benchmarks/
calibrate_cg_rtol_autograd.py` re-runs the same method through
`compliance.whole_compliance_value`/`gravity_compliance_value` and
`torch.autograd.grad`, so both `FemSolve`'s forward *and* its adjoint `backward` run at
the candidate `rtol`. `spsolve` (the original's reference) detaches the autograd graph
and can't stand in as a gradient reference here, so the reference is MGCG at
`rtol=1e-12` instead (see the new script's docstring), cross-checked by a
solver-independent finite-difference test on `whole_compliance_value` itself.

Production mesh, `it0800` snapshot, `nStage=8`:

```
=== 90x30 (ndof=5642) ===
    rtol   fwd(min-max)   bwd(min-max)   dcx rel@act  dcxg rel@act  dctg rel@act
   1e-06     13-39           0-0            2.32e-06      3.98e-06      1.32e-05
   1e-07     16-41           0-0            1.85e-07      7.92e-07      7.92e-07
   1e-08     17-43           0-0            4.96e-08      2.04e-07      2.04e-07
   1e-09     19-44           0-0            1.93e-09      2.15e-08      1.81e-08

=== 180x60 (ndof=22082, production mesh) ===
    rtol   fwd(min-max)   bwd(min-max)   dcx rel@act  dcxg rel@act  dctg rel@act
   1e-06     15-26           0-0            1.18e-04      4.56e-05      5.20e-06
   1e-07     17-29           0-0            7.92e-06      5.81e-06      3.86e-07
   1e-08     18-31           0-0            1.50e-07      9.37e-08      9.04e-08
   1e-09     21-33           0-0            2.70e-08      5.53e-08      1.34e-08

FD check 24x16 (solver-independent, whole_compliance_value differenced directly):
  rtol=1e-06: max rel error 1.270e-07
  rtol=1e-08: max rel error 2.685e-07
  rtol=1e-10: max rel error 7.694e-08
```

(`rel@act` is the max-over-active-elements relative error against the `rtol=1e-12`
reference, `elementwise_errors`' first return value; the `abs/peak` column is omitted
above for brevity and tracked the same pattern.)

**The predicted compounding did not materialize: `bwd(min-max)` is `0-0` at every
`rtol`, on both objectives.** `FemSolve.backward`'s self-adjoint warm start (Phase 3.3)
lands on the exact answer at iteration zero whenever `dL/dU` is parallel to `F` --
true for `whole_compliance` by construction, and it turns out to hold for
`gravity_compliance` too, because `dL/dU = 2*K*U = 2*F` regardless of how `F` itself
depends on density (the extra `dcx2`/`dct2` adjoint term is downstream of `FemSolve`'s
own backward, in the differentiable `_gravity_load` graph, not a second CG solve). So
in this codebase the adjoint costs nothing extra, and every error row above comes from
the *forward* solve alone -- the same quantity part 1 already calibrated, just now read
through autograd's chain instead of the hand-derived formulas.

**Decision: `rtol=1e-8` is unchanged.** At the production mesh, `dcx`/`dcxg`/`dctg` sit
at `1.50e-07`/`9.37e-08`/`9.04e-08` -- 6.7x-11x inside `SENSITIVITY_TOL = 1e-6`
(`calibrate_cg_rtol.py`'s "solved"-tier bar), consistent with the margin the original
hand-derived calibration found. **Loosening is not tempting, let alone something to act
on unilaterally: `rtol=1e-7` already breaches the bar at 180x60** (`dcx rel@act =
7.92e-06 > 1e-6`), one order looser than the current default, so the one-decade-per-
step increments this table checks leave no room to loosen even if the user wanted to
consider it. No code change made in `sttopt/torch_solve.py` or `sttopt/torch_mg.py`
(both default `rtol=1e-8`, `torch_solve.femsolve`'s default is the one the production
call path actually uses). Nothing here needs the user's approval since neither
direction -- tightening or loosening -- is warranted by the numbers.

**Not done in this pass:** the five performance-pass items above the tolerance
paragraph remain untouched and open for a future pass, per explicit user direction to
skip them this time (correctness-only scope). `benchmarks/calibrate_cg_rtol_autograd.py`
was added as a new script rather than folded into `calibrate_cg_rtol.py`, since the two
differ in what they patch (`_solve_fe`/`_solve_fe_batched`'s *value*-returning autograd
counterparts vs. the hand-derived ones) and in their reference backend (`rtol=1e-12`
MGCG vs. `spsolve`) -- see the new script's module docstring for why `spsolve` can't
serve as an autograd reference.

## Phase 3.7: Validation, the end-to-end number, and the deletion

**Profile again.** Re-run `benchmarks/profile_step.py` against the torch `step` at 180x60 /
`nStage=8` on the `it0800` snapshot, machine idle. Report the same table as part 1's Phase
0 so the two are directly comparable.

**Target.** The NumPy baseline is **3060 ms/step**. Part 2's arithmetic, using part 1's
measured 4.36x on FEM and assuming a conservative 5x on MMA and hotspot:

| item | NumPy (ms) | projected (ms) |
|---|---|---|
| FEM solves | 1257 | ~30 |
| `mma.mmasub` | 1109 | ~220 |
| `hotspot_constraint` | 539 | ~110 |
| assembly | 102 | 0 (matrix-free) |
| everything else | ~53 | ~50 |
| **total** | **3060** | **~410** |

So **~7x is the honest expectation and >=5x (<= 610 ms/step) is the target**. A result
below 5x is not a failure to report quietly -- it means one of the three big items did not
port as well as projected, and the profile will say which. Record it here either way.

**The acceptance floor is 4x (<= 765 ms/step), set by the user (2026-08-27), and it is a
floor rather than a gate on the port's merit:** the user's position is that even ~4x --
which would be a slight regression against the projection -- is acceptable, because the
simplicity autodiff buys makes up for it and Phase 3.6-style optimization can happen
later. So do not hold the port back, revert to hand-derived sensitivities, or open a
performance investigation on a number in the 4-5x band; record it, note which row missed,
and move on. Below 4x, stop and report rather than deleting the hand-derived code in the
final commit -- at that point the profile has found something the projection did not
anticipate and the user should see it before the fallback disappears.

Note what the hotspot row assumes: **5x is the projection for the whole constraint
including its sensitivity, so it presumes autograd roughly matches the hand-derived
algebra there.** Phase 3.4's `bench_sensitivities.py` is what tests that presumption, and
its table is where to look first if this row misses. The FEM row assumes the self-adjoint
shortcut in Phase 3.3 holds -- without it that row roughly doubles.

**Correctness, at three scales:**

1. **Unit and fixture suites.** Everything under `tests/` except the slow marker, passing.
   Plus the second and final `pytest -m slow` run, against the baseline taken before Phase
   3.1 (see Execution notes): same assertions, and the wall-clock difference between the
   two runs is a second, independent end-to-end timing datapoint alongside the profile.
2. **Short E2E.** `test_e2e.py`'s `nloop=3` trajectory against the existing fixture, at
   `e2e` tier. This should just work; `e2e_rtol` was built for this.
3. **Long E2E, by aggregate rather than by trajectory.** Re-run the 90x30 / `nloop=800`
   configuration that `tests/fixtures/generate_torch_port_designs.py` already knows how to
   run, and compare against the NumPy run's recorded history on quantities that are
   *stable* rather than pointwise: final volume fraction, the compliance history's shape,
   final `tru_max`, the fraction of elements that are within 1e-3 of 0 or 1, and a
   structural-similarity-style comparison of the final density field. Bitwise agreement is
   not expected and is not the test; a design that converges somewhere qualitatively
   different is a real failure. **State the accepted divergence explicitly in the results
   section** so a future reader does not over-read the agreement.
   This run is also the natural end-to-end timing datapoint: part 1 recorded 35 minutes for
   a native 180x60 run of the same shape.

**Then delete.** In a final, separate commit: remove the hand-derived sensitivity code from
`sttopt/compliance.py`, `sttopt/constraints.py`, `sttopt/conductivity.py`, and the
now-unused `filters.heaviside_projection_derivative`. Keep `sttopt/fem.py` and the
`tests/matlab_reference*.py` oracles -- and keep any hand-derived backward that Phase
3.4's benchmark justified retaining inside an `autograd.Function`, with the measurement
that justified it recorded next to it. Take the `bench_sensitivities.py` numbers before
this commit lands; afterwards they cannot be reproduced. Any test that existed only to compare autograd
against the hand-derived version goes with it -- the oracle tests are what remain.

Sanity check on the diff: this port should make `sttopt/` *shorter*. If it has grown,
something has gone wrong (repo rule), most likely a compatibility shim that should have
been a deletion.

**Finally:** update `plans/CLAUDE.md`, move this plan to `plans/archive/`, and open the
follow-up items (`plans/code_quality_review.md`) rather than absorbing them.

---

## Answers from the user (2026-08-27)

All three questions are settled; the bodies above have been updated, and this section is
the record of *why*, not a second source of truth.

- **Retiring the NumPy path: confirmed** (Decision 2). The premise that a GPU-less machine
  needs the NumPy path was wrong -- torch's CPU device is the fallback. So the deletion
  goes ahead, and the obligation it creates is that `device="cpu"` stays a genuinely
  working configuration rather than a nominal one.
- **The end-to-end bar: 4x is acceptable, and it is a floor, not a gate** (Phase 3.7). The
  user's reasoning: ~4x is a slight regression against the projection and still worth it,
  because the simplicity autodiff buys is the point of the port and further optimization
  can happen later. >=5x remains the target to aim at; 4x is where the port stops being
  reportable-and-move-on and starts needing the user's eyes before the hand-derived code
  is deleted.
- **Slow-test cadence: twice only** -- once before Phase 3.1, once at the end of Phase 3.7.
  See Execution notes.

No open questions remain.
