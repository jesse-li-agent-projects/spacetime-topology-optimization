# Time field from a virtual heat equation

## Read this first

This is a large project. The design below was agreed in one long discussion, but it
has gaps that nobody has found yet. **Ask the user often.** If a step needs a design
decision that this plan does not make, or if a result conflicts with an assumption
here, stop and ask before you continue. Do not choose silently and do not work around
the problem. An early question costs less than an implementation built on an
assumption that the user would reject.

Values marked *tentative* are starting points. Change them when results give a reason,
and tell the user about the change and the evidence for it.

**Style review after each phase.** When a phase is complete, the implementing agent
starts a Sonnet subagent that runs the `style-review` skill on that phase's changes.
Apply its findings before the PR goes to review.

**Keep a margin on every calibrated value.** Do not set a value at the edge of what a
measurement allows. Results shift with the mesh, the geometry, the device and later
code changes. State each margin in terms of the quantity that fails (for example "T on
the part is ≥ 100× the solve error"), not as a distance in the setting. Record the
measured limit and the margin in the PR. This applies to β, `C`, the CG tolerances,
the test tolerances, and the schedule values.

## Goal

Replace the time field `t` as a direct design variable with the solution of a
*virtual* heat equation, where the design variable is the diffusivity. The virtual
equation has no relation to the real thermal process (the hotspot constraint models
that). Its only purpose is to give a time field that, by construction, has **no local
minima**. A local minimum is material that prints before everything around it, i.e.
without support.

There are two variants. Each is a new script at the level of `stto.py`, not a
configuration of `stto.py`:

- **Variant 1, `stto_heat.py` (Wu2025 §2.3).** `∇·(χ∇T) − α T = 0` with `χ = ρ·χ(μ)`,
  `T = 1` on the build plate, zero flux elsewhere. The drain makes `T` subharmonic, so
  `T` has no interior maximum, and so `t = f(T)` (with `f` decreasing) has no interior
  minimum. Because `χ` contains `ρ`, time goes *through the material*. The guarantee
  applies over the part.
- **Variant 2, `stto_laplace.py`.** `∇·(χ∇t) = 0` with `χ = χ(μ)` (no `ρ`), `t = 0` on
  the build plate, and optimizable positive Dirichlet data `b` on every other wall. The
  maximum principle gives no interior extrema. The field does not see the part, so the
  guarantee applies over the **domain only**. The user accepts this, with a post-run
  check over the part (Phase 5).

Saddles are not ruled out in either variant, and cannot be bounded by boundary data in
3D. The tool-radius constraint controls them. Phase 5 adds a saddle count as a
diagnostic.

## References

- `resources/Wu2025_PhDThesis_Space-Time-TO-Multiaxis-AM.pdf` §2.3 (Eq. 2.9-2.15, Fig.
  2.6) and appendix A.1 (adjoint).
- `resources/geodesics_with_heat.pdf` (Crane et al.), Algorithm 1 and §3. Wu's
  equation *is* Crane's step I: one backward-Euler heat step is a screened Poisson
  equation, with `χ/α` in place of Crane's `t`.
- Alessandrini & Magnanini: in 2D, `∇·(χ∇u) = 0` with `u = 0` on one connected
  boundary arc and unimodal data on the complement has no interior critical points,
  also for discontinuous `χ`. This is the reason for the unimodal wall data below. It
  is a 2D result only.

## Background the decisions depend on

**Discrete maximum principle.** The guarantee survives discretization only if the
system matrix is an M-matrix: all off-diagonal entries `≤ 0`. Then each row says "this
node is a positive-weight average of its neighbours", and no interior node can be a
strict extremum. The Q4 Laplacian on square elements (off-diagonals −1/6, −1/3) and
the Hex8 Laplacian on cubes (0, −1/12) have this property for any per-element `χ`. A
**lumped** drain matrix keeps it. A consistent mass matrix has positive off-diagonals
and breaks it. Do not use a consistent mass matrix.

**The mapping `T → t` in Variant 1, and β.** Three options:

| Option | Guarantee kept? | Wants β | Note |
|---|---|---|---|
| `1 − T` (Wu) | exact | small | Layers not uniform. **Only** for reproducing Wu. |
| `−log T` (Varadhan) | exact (monotone map) | large | One solve. Thick last layer under top surfaces. |
| normalize + Poisson (Crane) | **no** | large | Two solves. No thick last layer. The preferred candidate. |

As β → ∞, `−log T / k` approaches the geodesic distance in the metric `ds/√χ`
(Varadhan). A small β gives `t ∝ Ly − y²/2` instead. With near-insulating void, a free
surface has `∂t/∂n ≈ 0` under `−log T`, so the iso-lines meet it at a right angle and
the last layer under a *top* surface thickens over a depth of about `l_c/√β`.
Poisson sets `∂φ/∂n = X·n` and so has no such zone.

The Poisson step is a least-squares fit, with no maximum principle. At a local minimum
of `φ` we need `∇·X ≥ 0`, and `∇·X` (the iso-line curvature) is positive on a normal
diverging front. So the guarantee is lost in theory. The post-run check measures
whether it is lost in practice.

**Solver precision limits β.** On the part, `T` falls to about `exp(−√β·L/l_c)` (at
`χ = 1`, with `L` the longest path from the plate through the part), or faster where
`χ < 1`. `L/l_c` depends on the geometry: 1.7 for an `edge` plate on 180×60, and more
for a path around holes. CG stops on a *relative* residual, so where `T` is below the
absolute error there are no correct digits, and `−log T` or the direction of `∇T`
there is noise. The allowed β therefore depends on the CG tolerance and on `χ_min`.
Phase 4 measures it.

**Why `χ` is exponential.** Flux conservation gives `χ·|∇t|·(tube width) = const`. To
keep layers uniform where the fronts spread out by ×k, `χ` must fall by ×k. An
exponential parametrization reaches a wide range, and an MMA step in `μ` is then a
relative change of `χ` at any value.

## Settled decisions

### Discretization (both variants)

- `χ(μ) = C^(μ − ½)`, `μ ∈ [0, 1]`, `C = 10³` (*tentative*). So `χ ∈ [0.032, 32]`
  and the initial `μ = ½` gives `χ = 1`. A smaller `χ_min` reduces the β margin (see
  above), so change `C` and β together.
- Variant 1: `χ_e = ρ̃_e · χ(μ_e)`, with `ρ̃ = max(xPhys, 10⁻⁶)`. The lumped drain keeps
  the matrix SPD even at `χ = 0`. The floor keeps the conditioning and the multigrid
  sane.
- `T` and `t` are solved on the **nodes**. The element value is the mean of the 4
  element nodes (as in Wu). The rest of the code reads element `tPhys`, unchanged.
- The build plate is the nodes on the left edge (`edge`) or the bottom edge
  (`bottom_edge`). **No other print base is supported.** A near-point base has no
  physical use and gives a log singularity. Reject every other `print_base` value at
  config load.
- **No filter on `μ`** and no filter on `t`. A checkerboard in `χ` modulates `|∇t|`
  from element to element, which is the padding mode that `_gradient_cv` rewards
  (PR #91/#94). The roughness term works against it, as in `stto`. Log the
  roughness of `log χ` as a diagnostic. If a checkerboard shows, the fix is the
  density filter on `μ`, but only after the user agrees.
- The density field has the same filter and Heaviside projection as `stto`.

### Variant 1

- `α = β / l_c²`, with `l_c = √(design area)`, the same unit length as
  `timefield.unit_length`. It does not depend on the print base or on the geometry.
  Then `k·l_c = √β` at `χ = 1`. The config field is `drain_beta`.
- `T → t` is a config option: `one_minus`, `neg_log`, `poisson` (default; see the
  priority in Phase 8).
- `neg_log`: `t = −log(max(T, 0) + T_eps)`, with
  `T_eps = 10⁻³ · min{T_e : ρ_e ≥ ½·max(ρ)}` (detached; the same set as the
  normalization). A floor set from the actual minimum on the part cannot clip part
  values at any β or geometry. A fixed floor such as 10⁻⁶ can. The clamp is for the
  void: there `T` falls below the solve error and was measured negative (to −4e-8 on
  the street grid at β = 25), which gave NaN. `t` in deep void is noise either way.
- `poisson`:
  - `X = −∇T / |∇T|` at the 2×2 Gauss points (not only at the cell centre, which
    cannot see an hourglass mode), with a guard against 0/0 only. **Not** the
    `NORMAL_EPS` floor relative to the mean gradient: `T` falls exponentially, so that
    floor shortens `X` far from the plate. Measured on a flat front (exact answer
    linear): max error 0.14–0.58 over √β = 3–10 with the floor, 0 without.
  - Solve `∇·(w∇φ) = ∇·(wX)` with `w = ρ̃`, `φ = 0` on the plate, natural BCs
    elsewhere.
  - Sign: `T` is largest on the plate, so `−∇T` points away from it, and `φ`
    increases away from it.
- The default β is chosen by the Phase 4 sweep, not in advance.

### Variant 2

- `χ = χ(μ)` over the whole domain. No `ρ`.
- **Unimodal wall data.** Order the non-plate boundary nodes along the arc from one
  plate end to the other: `k = 0 … n_arc−1`. Then
  `b = minimum(cumsum(a), reverse_cumsum(c))`, with `a, c ∈ [0, 1]` and each increment
  scaled by `2h/l_c`. This is the minimum of an increasing and a decreasing sequence,
  so it is unimodal by construction. A ramp needs increments of `h/l_c` per node, so
  the ramp init sits inside the bounds: `a = ½` where the ramp rises along the arc,
  `c = ½` where it falls, and 0 on the far wall, where it is flat (`a = c = ½`
  everywhere would put a tent on the far wall). The code derives `a`, `c` from the
  target wall values. A plate node has priority at a shared corner.
- Use `torch.minimum`, not a softmin. Change to a softmin only if MMA chatters at the
  crossing node.
- The overall scale of `b` is a flat direction, because `t` is normalized. Accept this.
  Do not add a gauge constraint unless it causes trouble.
- The inhomogeneous Dirichlet data enters by lifting: `rhs = −K_fd·g`. `FemSolve`
  already returns `dL/dF`, so autograd carries the gradient to `a` and `c`.
- `T → t` options: `identity` (default; see the priority in Phase 8), `poisson`.
  `poisson` uses `w = 1`, `φ = 0` on the
  plate, and natural BCs on the walls, so the walls then act only through the
  direction of `X`.

### Normalization of `t` to [0, 1]

`tPhys = t / max{ t_e : ρ_e ≥ ½·max(ρ) }`, with the denominator detached. This set is
never empty. On a uniform grey start it is the whole domain, and on a binary design it
is the solid. (`max_e(ρ_e·t_e)` fails on the grey start: at `ρ = 0.5` it scales `t` up
to about 2.) This is not elegant, but no better option has been found.

### Initialization

- Both variants: `x = volfrac`, `μ = ½` (uniform `χ = 1`).
- Variant 2: `a` and `c` reproduce the `edge`/`bottom_edge` ramp exactly (uniform `χ`
  with linear wall data gives a linear `t`).
- Variant 1 at uniform `ρ` gives the 1D `cosh` profile, mapped by the chosen `T → t`.
- **Alternative, not the default (Variant 2):** solve for `χ` so that the initial `t`
  is a given target, e.g. a Euclidean distance. On a smooth `f` without critical
  points, `χ = e^w` with `∇w·∇f = −Δf` along the gradient flow. Implement this only
  if tuning shows that the ramp init hurts robustness.

### Optimization

- **Design vector:** `[x; μ]` (Variant 1, `n = 2·nel`), `[x; μ; a; c]` (Variant 2).
- **Objective:** the same terms as `stto`: compliance + `Theta`·gravity stages +
  `uniformity_weight`·`_gradient_cv` + roughness. `uniformity_weight = 0` is a valid
  setting. Phase 8 tests whether `neg_log`/`poisson` need it at all.
- **Constraints:** global volume, hotspot, tool radius, `min_gradient`,
  `gradient_smoothness`. The PDE and the Dirichlet plate make these unnecessary:
  `start_point`, `time_field_continuity`, the `t` filter. Stage volume is deprecated
  and is omitted.
- **One move limit.** Every design group is on [0, 1] with a comparable meaning per
  unit, so all groups use one scheduled `move`. The late taper that `stto` applies to
  `tmove` applies to `move` here. Add a per-group override only with evidence from a
  tuning run, and only after the user agrees.
- **Sensitivities:** one autograd pass per constraint part, as in `stto` (PR #173).
  Each row now also passes through one PDE adjoint (two with `poisson`). If profiling
  shows that this matters, cut the graph at `tPhys` and push all rows through one
  batched adjoint solve. Do not do this before you profile.

### Load cases

A new `sttopt/load_cases.py`, used by `stto` and both new scripts. Config field
`load_case`, default `cantilever`, so old `config.json` files still load. There is a
new field `support_length_m`, default 4 mm. Loads use the existing `load_length_m`.
All cases use 1 mm elements.

| Name | Domain | Supports | Loads |
|---|---|---|---|
| `cantilever` (C1) | 180×60 | whole left edge clamped | unit downward, bottom-right, over `load_length_m` (the current case) |
| `cantilever` (C2) | 120×90 | as C1 | as C1 |
| `corner_loads` (S) | 120×120 | whole left edge clamped | top-right corner: unit, down-right `(1, −1)/√2`; bottom-right corner: unit, down-left `(−1, −1)/√2`; each over `load_length_m` from its corner along the right edge |
| `edge_traction` (D) | 120×90 | clamped patches of `support_length_m`, centred on the left edge at 25% and 80% of the height and on the bottom edge at 30% of the width | uniform +x traction over the whole right edge, unit total |

State the positions as fractions of the domain and the spans in metres, not as node
indices. The code calls the current case "cantilever". Remove the name "MBB" wherever
it refers to this case.

### Pass criteria

On the **dev set**, a schedule passes when:
- the hotspot row is ≤ 0 on the binarized design (`checks.py` report),
- the tool-radius row is ≤ 0 on the binarized design (Variant 2 only; the user
  dropped it for Variant 1: the heat fronts meet in seams around holes, and the
  iso-lines there have near-zero concave radius),
- the start and support checks pass, and
- the compliance is ≤ 1.10 × the matched `stto` control.

Support (all three scripts, `checks.support_report`) = no unsupported element in the
part's interior (all 4 edge neighbours solid; outside the domain is void). An
unsupported element on the boundary reads as a steep overhang; the user accepts it as
printable (a heuristic). Every unsupported element measured in Phase 8 was on the
boundary, mostly ties within one layer. Variant 2 is judged on the domain-level check
instead; its part-level counts are reported.

On the validation matrix, report these values. Do not gate on them.

## Phases

Each phase is one or more PRs. Keep the commits small (see `CLAUDE.local.md`). Phases
2 and 3 can run in parallel: Phase 2 is compute, Phase 3 is code.

### Phase 0: documentation fixes (small PR)

- `configs/default.json`: `print_base` `opposite_corner` → `edge`. Since #170,
  `opposite_corner` means a true corner. The runs that `continuation.md` calls
  validated used the old meaning (the whole left column, i.e. `edge`).
- `configs/continuation.md`: correct its print-base statements to match, and change
  "MBB half-beam" to "cantilever".
- Search the tracked docs and docstrings for other statements that the #170 fix
  made wrong. Do not edit the artefacts in `output/`.

### Phase 1: shared refactor of `stto.py` (no change in behaviour)

- `load_cases.py` (above). `stto.build_problem` gets `F`/`freedofs` from it.
- Move out of `stto.py` the parts that the new scripts share:
  - the constraint stack (`constraint_values` and the smooth-max calibrations),
  - the objective terms after compliance (uniformity + roughness),
  - the MMA call with the trust region and the state plumbing,
  - `estimated_conductivity`, `_hotspot_diagnostics`, `_admissible_tool_radius_m`.
- Keep the per-part Jacobian rows exactly (PR #173). Measure a step before and after
  the change (`benchmarks/profile_step.py`), and put both numbers in the PR.
- Configs: `HeatRunConfig`, `LaplaceRunConfig` in `run_config.py`. Move the shared
  fields into a base class only if every `output/*/config.json` still loads. Check this
  with a script and report the result. If some files do not load, ask the user.
- Gate: all fixture and E2E tests pass without a fixture change.

### Phase 2: `stto` study at a 4 mm tool radius (compute)

- The dev set (below) with `stto`, `tool_radius_m` final value 4 mm, 800 iterations,
  one run per cell. Rerun a marginal cell (a hotspot or tool-radius row of the
  binarized-design report within 0.05 of 0) on the other device (CPU vs GPU).
- **Report the compliance change from iteration 600 to 800 in each run.** This
  indicates the expected compliance of a 600-iteration schedule.
- Also do one quick run of load case D, to check that its optimum is not trivial (for
  example, straight tension bars only). Show the design to the user.
- If all cells pass: change the default to 4 mm in its own PR, and use 4 mm for the
  new variants and for all controls. If a cell fails: report to the user, and keep
  2.5 mm until the user decides.

### Phase 3: scalar FEM

- Make `torch_mg` and `torch_solve.FemSolve` generic in dofs per node, with 1 for the
  scalar problems. Do not write a second solver.
- Add a scalar Q4 `KE` (diffusion) and a lumped mass (drain), assembled per element
  from `χ_e`.
- Add inhomogeneous Dirichlet by lifting (above), and batched right-hand sides.
- Give the CG tolerance of each scalar solve its own setting (see "solver precision").
- Warm-start every scalar solve from the previous iteration's solution, as `stto` does
  with `State.U`. The warm start must be detached (`femsolve` asserts this; an
  undetached `x0` once leaked the multigrid hierarchy every step, commit 855eb76).
  Check whether the adjoint solve is warm-started too; ask the user before you change
  it.
- **Tests:**
  - **Manufactured solutions**, with smooth `χ = exp(sin …)` sampled at the element
    centres. Choose `u` so that `∂u/∂n = 0` on the Neumann walls, so the production
    code needs no inhomogeneous Neumann support. Show O(h²) L2 convergence over 3
    refinements for each BC set:
    1. all Dirichlet;
    2. Dirichlet plate + zero-flux Neumann + drain (Variant 1);
    3. zero plate + nonuniform Dirichlet walls, with one Neumann wall for the mixed
       case (Variant 2).
  - **The M-matrix property** of the assembled matrix for random log-uniform `χ` and
    for a `χ` contrast of 10⁶. **The discrete max principle** on nodes *and* on element
    means. If the element means violate it, report to the user; do not relax the test.
    - **Measured:** the principle holds on the nodes but **not** on the element means.
      With random log-uniform `χ` and random wall data, 12/40 Laplace fields at 10³
      contrast and 26/40 at 10⁶ had a strict element-mean extremum (margins up to 8% of
      the range). With ramp wall data: 0/40 at 10³, 5/40 at 10⁶. Drain fields: 0/40
      maxima at both contrasts. An element mean is not a positive-weight average of its
      neighbours' means. The user chose to record this as a strict xfail and go on. So
      the guarantee is on the nodal field; `tPhys` and `checks.unsupported` can still
      see a local minimum where `χ` has element-scale contrast. Phase 5 measures it on
      real designs; the density filter on `μ` is the fix if it matters.
  - Multigrid-CG against a direct sparse solve, including the 10⁶ contrast.
  - FD checks of the adjoint w.r.t. `χ`, the Dirichlet values, and the rhs.
  - 1D analytic profiles: `cosh` (drain + Neumann) and linear (Laplace).

### Phase 4: `sttopt/virtual_heat.py`, the map `(ρ, μ[, a, c]) → tPhys`

- Both variants' forward maps, every `T → t` option, the unimodal wall data, and the
  normalization.
- **Tests:**
  - Reproduce Wu Fig. 2.6(e): the 1D profiles of `1 − T` for β ∈ {0.1, 1, 10}.
  - **Distance recovery** (Variant 1 at uniform `μ`; `neg_log` and `poisson`) on
    binary geometries: a rectangle (a flat front, so the result is exact), an
    annulus, a sine "worm" (a band of material around `y = A sin(ωx)`, with the plate
    at one end), and a street grid (genus ≥ 1). The reference is
    `timefield._solid_geodesic` on a raster 4× finer. Measure the errors first, then
    set the tolerances from them with a margin, and report both to the user.
  - **The β sweep:** `√β = k·l_c ∈ {3, 4, 5, 7, 10}`, on the geometries with the
    largest `L/l_c`: C1 with an `edge` plate, and the street grid. For each value,
    record the recovery error, the minimum `T` on the part, and the agreement of
    `−log T` and of `X` on the part with a direct sparse solve at the production CG
    tolerance. The default β must keep the minimum `T` on the part ≥ 100× the
    measured absolute solve error on the worst geometry. Take the largest β that
    meets that margin. Report the table.
    - **Result: β = 25 (√β = 5)** at CG rtol 10⁻⁸. min `T` on the part / absolute
      CG error, worst geometry (street grid): 7.0e4, 8.3e3, **1.0e3**, 13, 0.025 at
      √β = 3, 4, 5, 7, 10. The worm fails at √β = 7 too (72). Recovery at β = 25, max
      / mean: `neg_log` ≤ 0.156 / 0.049, `poisson` ≤ 0.089 / 0.023 (annulus, street
      grid); test tolerances 1.6–1.7× those. `X` from CG agrees with a direct solve
      to 1.6e-3 on the part, `−log T` to 3.5e-4.
  - Wu's β uses a different `l_c` (the longest diffusion path). Convert Wu's values
    to this `l_c` before you compare with Fig. 2.6.
  - Variant 2: the ramp init reproduces the ramp exactly; `b` is unimodal for random
    `a`, `c`; no interior extrema over the domain for random `χ`; and a count of
    interior saddles (the 2D theorem says zero). Report counts that are not zero.
    - **Measured** (24×16, 20 fields each, 8-ring sign changes, ties dropped): no
      saddle with uniform `χ`, with i.i.d. per-element `χ` up to 30× contrast, or
      with a smooth `χ` (σ = 1–2 elements) at the full 10³. With i.i.d. `χ`: 0.95
      per field at 100×, 3.85 at 300×, 9.5 at 10³. The theorem is continuous;
      element-scale jumps of ~100× give discrete saddles, as for the element means
      (Phase 3). Recorded as a strict xfail; the smooth case is a strict test.
  - Finite gradients everywhere, deep void included, for every option.
  - FD checks of the whole map w.r.t. `x`, `μ`, `a`, `c`.
  - Normalization: the threshold set on a uniform field, on a binary field, and on a
    field where every `ρ` is below `eta`.

### Phase 5: post-run checks

- The part-level local-minimum check exists already: `checks.unsupported` ("each solid
  element off the base has an earlier solid neighbour") is exactly this check. Reuse
  it. `plans/post_run_checks.md` is still open. Coordinate with it; do not duplicate
  it.
- Add a domain-level check for Variant 2 (the same test with every element counted as
  solid) and a saddle count (4 or more sign changes of `t − t_c` around the 8-ring).
  Both are reported, not judged. (Phase 8 made the domain check Variant 2's support
  gate; see "Pass criteria".)
- Log the saddle count, the local-minimum count, and the roughness of `log χ` at each
  snapshot.
- **Done in Phase 5:** `checks.saddles(values, where, among)` (ties dropped), and the
  saddle count over the part in `check_design`'s report. `plans/post_run_checks.md`'s
  start/support checks and report already exist in `checks.py`.
- **For Phases 6/7:** `check_design` is `stto`-specific, so each new script needs its
  own (on the binarized design, `tPhys` recomputed from it). The domain-level check is
  `checks.unsupported` with every element solid. The `log χ` roughness is
  `timefield.roughness(log χ)`, the RMS 5-point residual, which reads element-scale
  modes. The per-snapshot logging goes into the new CLIs.

### Phase 6: `stto_heat.py`, `stto_heat_cli.py`, a config

The same structure as `stto.py` (`Problem`/`State`/`IterationRecord`/`build_problem`/
`init_state`/`step`/`run`/`run_from_state`) on the Phase 1 modules. Follow the CLI
layout for fast `--help` from `CLAUDE.local.md`. `State` carries the last solution of
each scalar solve (`T`, and `φ` for `poisson`) as the warm start of the next. Add an
E2E smoke test (a small mesh,
a few iterations, finite values, and a decrease of the objective or of the
infeasibility).

### Phase 7: `stto_laplace.py`, `stto_laplace_cli.py`, a config

As Phase 6, with the extra design groups `a`/`c`.

### Phase 8: tuning

Use the `long-running-runs` skill. For compute: one GPU job, plus CPU runs at **3
runs × 4 threads** (measured: 2 × 8 contends badly). Launch with `PYTHONSAFEPATH=1`
from a snapshot, and log `sttopt.__file__`.

**Dev set** (each cell is also in the validation matrix):

| Load | volfrac | Base | Tcr | Why |
|---|---|---|---|---|
| C1 | 0.5 | edge | 0.8 | the known `stto` baseline |
| S | 0.5 | bottom_edge | 0.6 | the support and the print base are in conflict (at 0.3 the thin top bar sits at the hotspot limit: +0.005 heat, +0.62 `stto`) |
| D | 0.7 | edge | 0.8 | an asymmetric support set |

Steps (Variant 1 first, then Variant 2 on the same steps):
1. **Hotspot only** (tool radius, `min_gradient`, `gradient_smoothness` off). Get
   *any* schedule to pass on C1. Then pass on all 3 cells. Tune only the primary
   `T → t` option of each variant (see "`T → t` priority" below).
2. `uniformity_weight` 0 against > 0: is `_gradient_cv` still needed? (Variant 1
   only. Variant 2 `identity` needs it.)
3. Add back the tool radius (Variant 2 only), then `min_gradient`, then
   `gradient_smoothness`, one at a time. Re-tune after each one.
4. Variant 2 only: if the ramp init is not robust, try the alternative init (above).
5. Compress from 800 to 600 iterations (start by scaling the change points by 0.75).
   Use the same pass criteria. If it fails, report the gap to the user.

**`T → t` priority.** The user chose these on a hunch, to start tuning early:
- Variant 1: `poisson`.
- Variant 2: `identity`, with `_gradient_cv` (`uniformity_weight > 0`) for uniform
  layers.

The other options (`neg_log` for Variant 1, `poisson` for Variant 2) are *lower
priority, not rejected*. Phase 4 still implements and tests them. Tune them only
if the primary option fails, or if the user asks.

Judge a change against a matched control (the same schedule without the change).
Replicate the marginal cells on CPU and GPU. Watch the multipliers: rows at `mma_c`
mean a schedule conflict. Record the final schedule and its reasons in a
`configs/*.md` note, as `configs/continuation.md` does for `stto`.

### Phase 9: validation matrix

A Graeco-Latin 4×4 square: rows = load case, columns = volfrac (0.3, 0.5, 0.7, and
0.5 again as the reference level). Each cell is (base, Tcr), from `row XOR column`:
0 = (edge, 0.6), 1 = (edge, 0.8), 2 = (bottom_edge, 0.6), 3 = (bottom_edge, 0.8).
Each pair of factors is balanced. Interactions are aliased, so this is a robustness
test, not an interaction study.

| | 0.3 | 0.5 | 0.7 | 0.5 |
|---|---|---|---|---|
| **C1** | edge, 0.6 | edge, 0.8 | bottom, 0.6 | bottom, 0.8 |
| **C2** | edge, 0.8 | edge, 0.6 | bottom, 0.8 | bottom, 0.6 |
| **S** | bottom, 0.6 | bottom, 0.8 | edge, 0.6 | edge, 0.8 |
| **D** | bottom, 0.8 | bottom, 0.6 | edge, 0.8 | edge, 0.6 |

16 cells × {Variant 1, Variant 2, `stto` control} = 48 runs, about 5 hours at the
capacity above. The `stto` controls use `stto`'s validated schedule, not a
600-iteration version. Add CPU/GPU replicates on the marginal cells. Report a table
per variant: pass criteria, compliance ratio to the control, `central_difference_cv`,
local-minimum and saddle counts.

## Out of scope

- 3D. (The M-matrix argument carries over to Hex8. The saddle result does not.)
- A 600-iteration schedule for `stto` (a follow-up; Phase 2 gives the first data).
- A fixed-geometry (`seqopt`-style) counterpart.
- Corner and other near-point print bases.
