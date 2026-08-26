# Plan: PyTorch port, gated on a GPU CG solver benchmark

> **Status (2026-08-26): open, not started. Do not begin any phase without the
> repository owner's explicit go-ahead.** Phases 0a-2 are investigation and produce a
> go/no-go decision. Phase 3 is the actual port and must not begin until Phase 2 says go.

## Goal

Port `sttopt` from NumPy/SciPy to PyTorch (float64 default), for two reasons:

1. **Autodiff.** Replace the hand-derived sensitivities (`compliance.py`,
   `constraints.py`, `conductivity.py`'s nontrivial pair algebra) with autograd.
2. **GPU.** Move the per-iteration cost onto the GPU.

Reason 1 is achievable regardless -- a `torch.autograd.Function` wrapping the FEM solve
with its already-derived adjoint is a small, contained amount of work. Reason 2 is not
established: the solve is `scipy.sparse.linalg.spsolve`, a sparse direct factorization,
and there is no GPU-resident drop-in for it. If the FEM solve has to stay on the CPU,
the GPU half of the motivation collapses and the port is worth far less.

So this plan front-loads the question that decides the project's value, and does not
write any production PyTorch code until that question is answered.

### Non-goal

Dense `torch.linalg.solve`. Ruled out: at 360x120, `ndof = 87362`, so a dense operator
is 61 GB in float64 against an 8 GB GPU. Not viable even at 180x60 (3.9 GB, plus
factorization workspace).

## Background: what the code does today

Per FEM solve (`fem.solve_fe`): assemble a CSR matrix from 64*nel COO triplets, extract
the free-dof submatrix by `K[np.ix_(freedofs, freedofs)]`, convert to CSC, `spsolve`.

Sizes, at `nu=0.3`, uniform density (measured):

| mesh | ndof | free dofs | nnz(K) | nnz/row |
|---|---|---|---|---|
| 90x30 | 5642 | 5580 | 71890 | 12.7 |
| 180x60 | 22082 | 21960 | 284170 | 12.9 |
| 360x120 | 87362 | 87120 | 1129930 | 12.9 |

Solve count: `optimize.step` performs `1 + nStage` solves per iteration -- one
`whole_compliance`, then one `gravity_compliance` per stage. At the production settings
(`cli.py` / `test_e2e_slow.py`: 180x60, `nStage=8`, `nloop=800`) that is **9 solves per
iteration, 7200 solves per run**.

Each solve uses a *different* `K`: `whole_compliance` assembles from `xPhys`, each
`gravity_compliance` from `xtJoint = xPhys * time_mask(ti)`. So no factorization is
reusable, either within an iteration or across iterations.

### Two structural facts the benchmark must not miss

- **The design moves slowly.** `move = tmove = 0.01`, so no design variable changes by
  more than 1% per iteration. The previous iteration's `U` is therefore an excellent CG
  initial guess. A direct solver cannot exploit this at all. Warm-started CG is the
  realistic operating mode and must be benchmarked as such.
- **The stage solves are independent.** The `nStage` gravity solves within one iteration
  share a sparsity pattern and differ only in element density. They can be batched into
  a single CG over an `(nStage, ndof)` right-hand side, turning a latency-bound GPU
  problem into a throughput-bound one. This is a structural GPU advantage with no CPU
  analogue short of multiprocessing.

## Risks, stated up front

**Conditioning (highest).** `Emin = 1e-9`, `Emax = 1.0`, and the Heaviside projection
(`beta_d` continuing to 128) drives `xPhys` to near-binary. The stiffness contrast is
then ~1e9, and `cond(K)` scales as contrast times the usual O(h^-2) mesh factor --
plausibly 1e12-1e13. Against float64 machine epsilon that leaves few reliable digits,
and CG's iteration count grows as `sqrt(cond)`. Consequences:

- Jacobi (diagonal) preconditioning is very unlikely to be sufficient. Expect to need
  geometric multigrid (MGCG -- the standard answer for topology optimization on regular
  grids; the mesh here is exactly the structured grid multigrid wants).
- **Benchmarking on uniform `x = volfrac` would be dishonest.** That is the easiest
  possible conditioning and it is not what the optimizer spends its time on. Every
  correctness and timing measurement must also be run at a realistic near-binary design
  -- which is what Phase 0a exists to produce, and why it comes first.

**Problem size (high).** 22082 dofs is small for a GPU. A sparse matvec at 284k nonzeros
is microseconds of actual work, so CG will be dominated by kernel-launch latency, not
bandwidth. The available GPU is a laptop part (RTX PRO 1000 Blackwell, 8 GB), so the
bandwidth advantage over the CPU is maybe 3-5x, not orders of magnitude. Mitigations if
this bites: CUDA graphs to eliminate per-iteration launch overhead, kernel fusion, and
the stage batching above.

**The FEM solve must not stay on the CPU (high).** Note that this is *not* an Amdahl
argument, and a small FEM share of runtime is not a reason to abandon the port.

Every other hot spot in the loop -- `hotspot_constraint`'s pair reductions, assembly,
the filter matvecs, MMA's dense algebra -- is ordinary array work that ports to torch
directly and runs on the GPU without any research. The FEM solve is the *only* piece
with no easy GPU story, because there is no GPU sparse direct solver in torch. So if
Phase 0 finds the solve is a small fraction of runtime, that is good news: most of the
runtime is in the easy-to-port half, and comparatively little rides on the risky CG
work.

But the solve still has to be faster on GPU+CG than on CPU+spsolve, for a reason
independent of its share: **a hybrid loop is actively bad.** Leaving the solve on the
CPU while the rest runs on the GPU means 9 GPU->CPU->GPU round trips per iteration
(7200 per production run), each a full synchronization that serializes the pipeline and
defeats the point of having the rest on the device. There is no "keep the solve on CPU
and accelerate everything else" configuration worth having.

So the gate is a direct comparison of the solve itself, not a runtime-fraction
threshold. Phase 0 still runs first -- it tells us where the easy wins are, and what
`hotspot_constraint` and assembly actually cost -- but its output informs the port's
*shape*, not its go/no-go.

**Honest prior.** For 2D problems at these sizes, sparse direct factorization is strong
-- nested dissection gives low fill-in and near-optimal work. My expectation is that GPU
CG loses at 90x30, is close at 180x60, and may win at 360x120. The benchmark is still
worth running, because warm-starting and stage batching are advantages the naive
comparison omits, and because the answer decides whether the GPU motivation survives.

---

## Phase 0a: Generate the realistic test designs (do this first)

Everything downstream -- Phase 0's profiling, Phase 1's correctness tests, Phase 2's
benchmark -- needs realistic near-binary designs rather than uniform `x = volfrac`.
Generating them takes a long-running job, so **start it before any implementation work**,
so the designs are on disk by the time an agent needs them.

**Method.** Run `test_e2e_slow.py`'s experiment at **half resolution (90x30)**, logging
intermediate `xPhys`/`tPhys` snapshots along the way. Then **upscale 2x and 4x** for the
180x60 and 360x120 cases. 90x30 has 4x fewer elements and dofs than 180x60, and a sparse
direct solve scales superlinearly, so the run is more than 4x cheaper.

Details that matter:

- **Snapshot across the whole run, not just the end.** Conditioning worsens as `beta_d`
  ramps (doubling every 50 iterations to the 128 cap) and the design binarizes. Keep
  early, middle, and late snapshots -- the late ones are the hard cases and the ones the
  go/no-go should turn on.
- **Log `tPhys` too, not just `xPhys`.** `gravity_compliance` needs both, and the
  benchmark covers the stage solves.
- **Upscale by nearest-neighbour block repeat** (`np.kron(x, np.ones((2, 2)))`), not
  interpolation. A near-binary design must stay near-binary; bilinear interpolation would
  manufacture intermediate densities and quietly make the conditioning easier -- exactly
  the property under test.
- **Scale the run's own parameters when generating**, since `rmin`, `lrmin`, and
  `rmin_cond` are in element units: at 90x30 they should be half the production values
  (2.0, 1.0, 6.0) for the design to be the same physical problem.

**Known limitation, accepted.** An upscaled 90x30 design has feature sizes twice as large
in elements as a natively-converged 180x60 design would. So it is not identical to what
the optimizer would really produce at 180x60 -- it is a coarser-featured design at fine
resolution. It preserves the property that actually drives the conditioning risk (hard
0/1 contrast and the void topology), at a small fraction of the cost, which is the right
trade here. Worth a sentence in the Phase 2 write-up so the numbers are not over-read.

**Deliverable:** a set of `(xPhys, tPhys)` snapshots at each of 90x30 / 180x60 / 360x120,
stored where both the tests and the benchmark can load them, plus the script that
regenerates them.

---

## Phase 0: Profile the current code

**Question:** where does `optimize.step` actually spend its time?

Not a go/no-go input -- see "The FEM solve must not stay on the CPU" above. This phase
maps where the easy GPU wins are and sizes the non-FEM work, so Phase 3 can be sequenced
sensibly and so any cheap NumPy-level wins get spotted early.

Profile `optimize.step` at production settings (180x60, `nStage=8`) over enough
iterations to be representative, with `cProfile` plus targeted wall-clock timers around
each suspect. Run at a Phase 0a near-binary design, not from `init_state`.

Suspects, with priors:

- `fem.solve_fe`'s `spsolve` -- the hypothesis under test.
- `K[np.ix_(freedofs, freedofs)]` -- sparse fancy indexing on both axes, allocating a
  new matrix every solve. Plausibly a large share of `solve_fe` and *separable* from the
  solve itself. If so it is a cheap independent win (apply boundary conditions by
  zeroing rows/columns and putting 1 on the diagonal, rather than re-indexing).
- `fem.assemble_stiffness` -- 691200 COO triplets per solve at 180x60, coalesced into
  CSR nine times per iteration.
- `conductivity.hotspot_constraint` -- **strong suspect.** `rmin_cond = 12` gives a
  23x23 neighbor window, so `npairs` is 4204240 at 180x60 (measured), and one call makes
  six `np.add.at` passes over arrays that size (`Nsum3`, `num`, `S1`, `S2`, `cond_arr1`,
  `cond_arr2`). `np.add.at` is unbuffered and is commonly 10-100x slower than
  `np.bincount` on the same reduction. This may rival or exceed the FEM solve.
- `mma.mmasub` -- `n = 21600` design variables, `m = 79` constraint rows; `dfdx` is a
  dense 79x21600. Probably not dominant, but measure rather than assume.

**Deliverable:** a runtime breakdown of `optimize.step`, and -- separately from the solve
itself -- how much of `fem.solve_fe` is the `np.ix_` reindex rather than `spsolve`. The
second matters for Phase 2: the baseline should be timed against `spsolve`'s real cost,
not against `spsolve` plus an artefact of how boundary conditions happen to be applied
today.

**Note:** if `np.add.at` or the `np.ix_` reindex turn out to be significant, those are
worth fixing on their own merits, independent of this plan's outcome. Log them in
`plans/code_quality_review.md` rather than absorbing them here.

---

## Phase 1: Implement and validate a GPU CG solver

Branch. No changes to production call sites yet -- build the solver alongside the
existing one, selectable, so the existing test suite can be run against both.

### Design

**Matrix-free, element-by-element matvec.** Do not assemble at all. Every element shares
the same `KE`; the elementwise operator is

```
gather U -> Ue (nel, 8);  Ue @ KE -> (nel, 8);  scale by density (nel, 1);  scatter-add
```

That is one `(nel, 8) @ (8, 8)` matmul plus an `index_add_`, both ideal on GPU. It
removes assembly cost entirely (which Phase 0 will have measured), and drops memory from
O(nnz) to O(nel).

Risk to note: `index_add_` uses atomics and is non-deterministic by default. See the
determinism test below -- this must be decided explicitly, not discovered later.

**Preconditioner, in escalation order:**

1. **Jacobi (diagonal).** Cheap, matrix-free (scatter-add of `diag(KE)` scaled by
   density). Build this first as the correctness scaffold -- it makes the solver easy to
   verify before adding multigrid's complexity. Expect its iteration count to be
   unacceptable at near-binary designs; measure that rather than assuming it.
2. **Geometric multigrid (MGCG).** The real target. V-cycle on the structured grid,
   coarsening by 2 per dimension (180x60 -> 90x30 -> 45x15 -> direct coarse solve), with
   a few Jacobi or damped-Jacobi smoothing sweeps. Typically converges in 5-20
   iterations largely independent of mesh size, which is exactly what the conditioning
   risk demands.
3. Incomplete Cholesky: **skip.** Its triangular solves are sequential and a poor GPU
   fit.

**Stopping criterion.** Relative residual `||r|| / ||b||`. Note that residual tolerance
is not solution error -- the error is amplified by `cond(K)`, which is the thing we are
worried about. Do not pick a tolerance by assumption; calibrate it (below).

**Failure must be loud.** If CG does not reach tolerance within `max_iter`, raise. Never
return a silently-unconverged `U` -- with 7200 solves per run, a quiet failure would
corrupt an entire optimization with no signal.

### Accuracy calibration

There is a real asymmetry here that the tests must respect:

- **Compliance is forgiving.** `U` minimizes the potential energy, so the energy (and
  hence `c`) is stationary at the solution and its error is second-order in the error of
  `U`. A sloppy solve yields a surprisingly accurate `c`.
- **Sensitivities are not.** `dcx` depends on per-element `ce = Ue^T KE Ue`, which is not
  globally stationary; element-level quantities carry first-order error.

So **the sensitivities set the tolerance, not the compliance.** Calibrating against `c`
would pick a tolerance that looks fine and quietly degrades the optimizer's search
direction.

### Tests

Reuse the existing suite wherever possible -- `tests/conftest.py`'s `assert_close` tiers
(`algebraic` rtol 1e-10, `solved` rtol 1e-6, `e2e` scaled) already encode this repo's
tolerance policy, and the existing fixtures are the strongest available oracle.

1. **Matvec vs assembled.** Matrix-free `K @ v` against `fem.assemble_stiffness(...) @ v`
   from the existing SciPy path. Random `v`, several density fields, several mesh sizes.
   Pure algebra -- `algebraic` tier.
2. **Diagonal vs assembled.** Matrix-free diagonal against `K.diagonal()`. `algebraic`
   tier.
3. **Solve vs `spsolve`.** `U` from CG against `U` from `fem.solve_fe`, at four density
   fields: uniform `volfrac`; random; **the Phase 0a near-binary snapshots** (late ones
   especially); and a near-all-void degenerate case. `solved` tier.
4. **Boundary conditions.** Fixed dofs are exactly zero on output, and the solution is
   unchanged by whatever sits in the fixed rows of `F`.
5. **Existing fixture regression.** Run `test_fem.py`, `test_compliance.py`, and the
   `nloop=3` E2E fixture with the CG backend substituted. These must pass unchanged.
   This is the single highest-value check in the list.
6. **Sensitivity accuracy (the tight one).** `dcx`/`dct` computed from CG-`U` against the
   same from `spsolve`-`U`, compared **element-wise on max relative error**, not by norm
   -- a norm hides a single bad element, and MMA reads every element. Plus a
   finite-difference check through the CG solve. This test is what pins the CG tolerance.
7. **Non-convergence raises.** Force a failure (tiny `max_iter`) and assert it raises
   rather than returning.
8. **Determinism.** Same input twice on GPU, bitwise identical. If `index_add_` atomics
   make this false, decide explicitly: either use a deterministic scatter, or accept
   nondeterminism and loosen the E2E trajectory comparisons. The repo's E2E tests compare
   trajectories, so this cannot be left undecided.
9. **Dtype.** float64 end to end; assert no silent float32 downcast anywhere in the
   solver.
10. **CPU/GPU parity.** The same CG on CPU and on GPU agree to `solved` tier.

---

## Phase 2: Benchmark

**Question:** is GPU CG faster than `scipy.spsolve`, at matched accuracy, on realistic
designs?

Scope is the **FEM solve alone** -- assemble plus solve, not the surrounding optimization
loop. Meshes 90x30, 180x60, 360x120. Repeat each configuration enough times for a stable
seconds-per-solve, with a total benchmark budget of roughly 5 minutes.

### Configurations

Three cells, not a 2x2 -- there is no direct-solver-on-GPU cell to fill, which is the
whole premise of this plan. And only one CPU direct solver: `spsolve` is the baseline
because it is what the code runs today, and it is already a well-optimized
factorization; a second CPU direct solver would not change any decision here.

| cell | what |
|---|---|
| `scipy.spsolve` (CPU) | the baseline, as `fem.solve_fe` runs today (net of the `np.ix_` cost Phase 0 measured) |
| torch CG (GPU) | the candidate |
| torch CG (CPU) | diagnostic control -- see below |

The third cell is cheap (the same code with `.to("cpu")`) and earns its place by making
a negative result *interpretable*. If GPU CG loses to `spsolve`, the next question is
immediately why:

- torch-CG-CPU also much slower than `spsolve` -> the problem is **algorithmic**: CG is
  taking too many iterations. Fixable with a better preconditioner, so a no-go verdict
  would be premature.
- torch-CG-CPU competitive with `spsolve`, but the GPU no faster -> the problem is
  **the device**: too small a problem, latency-bound. Fixable only with batching or CUDA
  graphs, if at all.

Without that cell a no-go tells you to stop without telling you whether stopping is
right.

Crossed with:

- **Density field:** uniform `volfrac` **and** the Phase 0a near-binary snapshots. Report
  both separately; the late near-binary number is the one that decides.
- **Start:** cold start **and** warm start from a design perturbed by one `move = 0.01`
  step, which is the real operating condition.
- **Batching:** sequential stage solves vs one batched `(nStage, ndof)` CG.

### Measurement hygiene

- `torch.cuda.synchronize()` around every GPU timing region.
- Discard warm-up iterations (allocator, cuBLAS autotune, CUDA graph capture).
- Report CG iteration counts alongside times -- a fast solve at a loose tolerance is not
  a win, and the iteration count is what explains a result.
- **Assert accuracy at every benchmark point.** Each timed configuration must also verify
  its result against `spsolve` at the Phase 1 sensitivity tolerance. Timings at unmatched
  accuracy are meaningless.

### Go / no-go

The criterion is the solve itself, not a runtime fraction: **at 180x60, on a late
near-binary Phase 0a design, warm-started, at matched accuracy, GPU CG must beat
`spsolve`.** Parity is not sufficient -- the CG solver is hand-rolled code this project
would then have to own and maintain, against a library routine that already works, so it
has to pay for itself. As a starting threshold, **>=2x**.

If the verdict is no-go: **stop and reconsider.** Record the numbers here, and read the
torch-CG-CPU cell first to see whether the failure is algorithmic (more preconditioner
work might change the answer) or hardware (it will not).

A no-go does not necessarily kill the port -- reason 1 (autodiff) stands on its own, and
Phase 0 may have shown that most of the runtime is in easily-ported non-FEM work.
But a GPU port whose FEM solve is slower than today's is not worth having (see the
round-trip argument above), so a no-go means the *GPU* motivation is dead and what
remains is a different, CPU-oriented plan -- to be written then, deliberately, not a
fallback to slide into.

Also worth capturing either way: the 360x120 result. If GPU CG wins only at the larger
mesh, that is useful information about where this approach becomes viable.

---

## Phase 3: Port the rest to PyTorch

**Only on a Phase 2 go.** Sketch only -- to be expanded into its own plan once Phase 2's
numbers are in and the solver's real shape is known.

- `torch.set_default_dtype(torch.float64)`; device plumbing through `Problem`.
- The FEM solve as a `torch.autograd.Function`. Forward: CG. Backward: for `K U = F`
  with `K` symmetric, the adjoint solve is against the same `K` -- warm-startable from
  the forward solution. The sensitivity algebra is already derived; this is one
  contained place to use it.
- Autodiff replaces the hand-derived sensitivities in `compliance.py`,
  `constraints.py`, `conductivity.py`. Keep the hand-derived versions as test oracles --
  they are validated against MATLAB fixtures and are the reason a switch to autograd can
  be checked at all. Do not delete them in the same commit that introduces autograd.
- `scipy.sparse` operators (`H`, `L`, `C`) to sparse torch tensors.
- `np.add.at` -> `index_add_` / `bincount`.
- jaxtyping annotations from `Float[np.ndarray, ...]` to `Float[Tensor, ...]`.
- Per this repo's commit convention, sequence this as many small self-contained commits,
  not one port commit.
