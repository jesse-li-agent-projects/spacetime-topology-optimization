# Plan: PyTorch port, gated on a GPU CG solver benchmark

> **Status (2026-08-26): open, not started.** Phases 0-2 are investigation and produce a
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
  correctness and timing measurement must also be run at a realistic near-binary design.

**Problem size (high).** 22082 dofs is small for a GPU. A sparse matvec at 284k nonzeros
is microseconds of actual work, so CG will be dominated by kernel-launch latency, not
bandwidth. The available GPU is a laptop part (RTX PRO 1000 Blackwell, 8 GB), so the
bandwidth advantage over the CPU is maybe 3-5x, not orders of magnitude. Mitigations if
this bites: CUDA graphs to eliminate per-iteration launch overhead, kernel fusion, and
the stage batching above.

**Amdahl (high).** If the FEM solve is not most of the runtime, no solver speedup
matters. Phase 0 exists to measure this before anything is built.

**Honest prior.** For 2D problems at these sizes, sparse direct factorization is strong
-- nested dissection gives low fill-in and near-optimal work. My expectation is that GPU
CG loses at 90x30, is close at 180x60, and may win at 360x120. The benchmark is still
worth running, because warm-starting and stage batching are advantages the naive
comparison omits, and because the answer decides the whole project.

---

## Phase 0: Profile the current code

**Question:** where does `optimize.step` actually spend its time, and what fraction is
the FEM solve?

Profile `optimize.step` at production settings (180x60, `nStage=8`) over enough
iterations to be representative, with `cProfile` plus targeted wall-clock timers around
each suspect. Run at a realistic near-binary design, not just from `init_state`.

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

**Deliverable:** a runtime breakdown, and the fraction `f` of per-iteration time spent
in the FEM solve. `f` sets the ceiling on the whole GPU effort (`1/(1-f)`), and becomes
an input to the Phase 2 go/no-go threshold.

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
   fields: uniform `volfrac`; random; **a converged near-binary design** (take an
   `xPhys` snapshot from a real run); and a near-all-void degenerate case. `solved` tier.
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

Measure a 2x2 of `{CG, direct} x {CPU, GPU}`, not just the two endpoints. Without the
torch-CPU-CG cell you cannot tell whether a result is "CG beats direct" or "GPU beats
CPU", and those have different implications.

| cell | what |
|---|---|
| `scipy.spsolve` (CPU) | the baseline, exactly as `fem.solve_fe` runs today |
| `scipy.spsolve`, BCs pre-applied | separates solve cost from the `np.ix_` reindex |
| torch CG (CPU) | isolates the CG-vs-direct axis from the device axis |
| torch CG (GPU) | the candidate |
| CHOLMOD (CPU), *optional* | `K` is SPD; this is the fair "best CPU" number. Worth knowing if the verdict is that the CPU wins |

Crossed with:

- **Density field:** uniform `volfrac` **and** a converged near-binary design. Report
  both separately; the near-binary number is the one that decides.
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

Combine with Phase 0's FEM fraction `f`. Proceed to Phase 3 only if, at 180x60 on the
near-binary design at matched accuracy, GPU CG beats `spsolve` by a margin large enough
that `f` makes it worth the port -- as a starting threshold, **>=2x on the solve, with
`f` >= 0.5**, giving >=1.3x end to end. Adjust the threshold once `f` is known.

If the verdict is no-go: **stop and reconsider.** Record the numbers here. Note that
Phase 0's findings and reason 1 (autodiff) may still justify a CPU-only PyTorch port, or
a narrower NumPy-level optimization pass -- but that is a different plan, to be written
then, not a fallback to slide into.

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
