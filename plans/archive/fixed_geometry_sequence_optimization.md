# Fixed-geometry fabrication sequence optimization (`seqopt`)

## Goal

Optimize **only** the print-time field `t` for a **fixed, prescribed geometry**, to minimize
overheating subject to layer-uniformity and print-continuity constraints. The geometry comes
from a file, so any component can be studied: the Wu2025 thesis examples (Fig. 3.4's overhang
bracket, Fig. 2.13's C/V/L shapes) or a topology-optimized layout produced by a separate run
(Fig. 3.10).

This is a *second* optimization problem living beside the existing space-time topology
optimization (STTO), not a replacement. STTO co-optimizes density **and** time; `seqopt` fixes
density and optimizes time alone.

## Reference

- `resources/Wu2025_PhDThesis_Space-Time-TO-Multiaxis-AM.pdf`, chapter 3 (fabrication sequence
  optimization with layer geometry control) and §2.4.3-2.4.4. Eq. 3.23-3.28 is the problem
  statement Wu solves; ours differs in the objective (we minimize the Das2025 conductivity /
  overheating proxy, not thermal-induced distortion).
- `resources/Das2025_Overheating-Prevention-Geometric-Method-TO.pdf` for the overheating proxy
  itself, already implemented in `sttopt/conductivity.py`.

## Settled decisions

1. **Objective = overheating + layer uniformity. No FEM.** There is no compliance term, so
   there is no `K U = F` solve anywhere on the `seqopt` path, no boundary conditions to
   specify, and no `Emin`/`Emax`/`nu`/`penal`/`eta`/`beta_d` parameters. An iteration costs two
   sparse mat-vecs plus the pairwise-sigmoid sum. Self-weight compliance is a possible later
   addition; it would require boundary conditions in the geometry file, so it is out of scope.
2. **The uniformity metric must stay swappable.** Do not write docs, config field names, or
   variable names that assume any particular measure is *the* uniformity measure. There is one
   today, selected by name. Prefer `uniformity_*` naming over `Gamma`. (`RunConfig.Gamma` on
   the STTO side keeps its name; do not touch it.)
6. **Both objective terms are dimensionless and order 1**, so the weights transfer across
   resolutions and components and no run-time rescaling exists. See `GRADIENT_CV`.
3. **Geometry input is an `.npz` holding `xPhys`.** This makes the Fig. 3.10 case free: any
   STTO run's `output/<tag>/final_design.npz` is already a valid geometry file. A helper script
   generates the thesis geometries as `.npz`.
4. **Full rename for symmetry**: `sttopt/optimize.py` -> `sttopt/stto.py`, `sttopt/cli.py` ->
   `sttopt/stto_cli.py`, beside the new `sttopt/seqopt.py` / `sttopt/seqopt_cli.py`.
5. **`seqopt` does not filter the time field: `tPhys = t`.** See below.

## No smoothing of the time field

STTO computes `tPhys = H @ t / Hs`, reusing the *density* filter to smooth the time field. That
smooths **across void**: on a topology-optimized component, two branches separated by a gap get
their print times mixed purely because they are geometrically near, with no material between
them to carry heat. Wu2025 avoids this by solving the heat equation on the part alone.

`seqopt` avoids it by not smoothing the time field at all. `t` is used directly as the physical
time field.

**This is an educated guess, not a settled decision.** It is the starting point because it
removes the void coupling at its source, but whether the time field wants some other
regularization in the filter's place is an open empirical question. Do not write a test that
asserts the absence of smoothing, and do not build anything that assumes it is permanent.

What follows from it, and must be carried through every phase:

- `seqopt` never builds or uses the density filter. `filters.density_filter`, `H` and `Hs` do
  not appear in `sttopt/seqopt.py`, and `SeqRunConfig` has no `rmin`.
- The autograd leaf and the physical field are the same tensor, so there is no filter chain rule
  and no filter adjoint on the `seqopt` path. Sensitivities are plain gradients with respect to
  `t`.
- PR #26's `init_state` concern — that an unfiltered initial `tPhys` disagreed with the
  filtered map `step` differentiates — cannot arise here, because there is no map. Do not port
  that reasoning across.
- Regularization of `t` now rests entirely on the continuity constraint
  (`constraints.time_field_continuity`, the `L` operator) and the uniformity penalty. Losing the
  filter's implicit smoothing makes those two load-bearing rather than merely helpful. Watch the
  first real runs for a noisy or checkerboarded time field; if one appears, that is a signal
  about `lrmin` and the uniformity weight, not a reason to quietly reintroduce the filter.
- **Still open, do not decide unilaterally**: the continuity operator `L` is also a local
  neighbourhood average over the whole grid, so it too couples elements across a void gap,
  just over the smaller radius `lrmin`. Whether it should be density-weighted is a separate
  question from the density filter. Raise it with the user rather than changing
  `constraints.time_field_continuity`, which STTO shares.

Independent of all this, `t` outside the part is pinned only by the continuity constraint, so
the uniformity measure must still be **weighted by the density**, or meaningless void gradients
dominate it.

STTO keeps its existing filtered `tPhys` — it is a faithful port of the MATLAB source, and
changing it is a separate question from adding a new problem. Do not touch it in this work.

---

## Accepted: the time field over void stays a free design variable

**Decision: change nothing.** Void elements keep their time variables, and the optimizer and the
regularizer handle them, exactly as they would in STTO.

The reason is what `seqopt` is *for*. It is a proving ground for the full space-time approach,
and in that approach there is no fixed void set to special-case — density is a design variable,
so which elements are "void" changes every iteration. Any mechanism that freezes or slaves void
times by identity would work here and then fail to transfer, and would make results from the
proving ground say less about the thing being proved. Maximizing similarity to the eventual full
method is worth more than removing the artefact below.

What follows: the design vector stays `n = nel`, `build_problem` needs no solid/void
partitioning, and nothing anywhere keys behaviour off which elements hold material — density
enters only as a *weight* (as in `GRADIENT_CV`), never as a mask on the design space. Weighting
transfers to STTO; masking does not.

The rest of this section documents the artefact being accepted, because a known effect that is
never written down gets rediscovered as a bug.

### The artefact

Measured, not theorized. On a 30x20 L-shape with `rmin_cond=6`, holding the time field over the
**solid** elements fixed at a bottom-up ramp and changing only the values over **void**:

| time over void | hotspot `numer` | mean `K_est` over solid |
|---|---|---|
| follows the ramp | 0.3527 | 0.9712 |
| 0.0 (printed first) | 0.4862 | 0.8719 |
| 1.0 (printed last) | 0.2292 | 0.9981 |

A 35% swing in the objective, with no change whatever to the order the actual material is
deposited in. (One geometry and one radius — illustrative of the mechanism, not a universal
number.)

Mechanism: `K_est_i = sum_j x_j^q w_ij FT_ij / sum_j w_ij FT_ij`. A void neighbour contributes
nothing to the numerator but `w*FT` to the denominator, so driving void times late makes
`FT -> 0`, drops those neighbours out of the normalizer, and lifts `K_est` toward the
solid-only average. The optimizer is rewarded for relabelling empty space.

Das2025 Eq. (6) has no such freedom: `mu_i = sum_{j in S_i} rho_j w_j / sum_{j in S_i} w_j`,
where `j in S_i` iff `dist(i,j) <= kappa` **and** `y_j <= y_i`. That domain is fixed by the
build direction, so the denominator is a geometric constant. The space-time port replaces the
`y_j <= y_i` test with the differentiable time-order mask `FT`, and applies it to numerator and
denominator alike — which is what makes the normalizer design-dependent.

This is not a porting bug, and the `FT` in the denominator is not a port artefact either
(`conductivity_estimation_stto_main.m` line 388: `Nsum3(i)=sum(ti_e.*w_e)` with
`ti_e = FT_el{i}`). Once deposition order *is* a design variable, "restricted to what has
already been deposited" and "independent of the design" cannot both hold. The source picked one.
STTO carries the same property; `seqopt` is only where it is easiest to see, because with the
geometry frozen there is less else going on.

Note also what the artefact is *not*. Density weighting is not missing: the numerator has
`x_j^q`, and Das2025 Eq. (6) has no density in its denominator either. And void elements are
already all but excluded as *evaluation* points, since `cond_p = T^p * x^(r*p)` with `r*p = 1.25`
annihilates a void element's own contribution. Void must stay counted as a *neighbour* — drop it
from the denominator and a thin strut hanging in space reads as perfectly cool, inverting the
proxy.

### What to watch, since it is not being fixed

The continuity constraint is now the only thing limiting how far void times drift from their
solid neighbours — the density filter used to do some of that job implicitly, and `seqopt` no
longer has it. Two consequences for the first real runs:

- If the objective improves markedly while the time field over the *part* barely changes, this
  artefact is what improved. Compare the time field over solid between iteration 0 and the end
  before believing a large reduction.
- This raises the stakes on the still-open question of whether `constraints.time_field_continuity`
  should be density-weighted. Note that weighting `L` by density *would* transfer to STTO, so it
  does not fall foul of the reasoning above — unlike freezing or slaving void times, which
  would.

# Phase 0 — rename `optimize.py` -> `stto.py`, `cli.py` -> `stto_cli.py`

Mechanical and self-contained. **Must land before every other phase**, because later phases
edit the same files. One commit, no behaviour change.

Do the bulk with `git mv` plus a `sed` sweep rather than by hand:

```sh
git mv sttopt/optimize.py sttopt/stto.py
git mv sttopt/cli.py sttopt/stto_cli.py
git mv cli.sh stto.sh

# Import forms in use: `import sttopt.optimize as optimize`, `from sttopt.optimize import ...`,
# `sttopt.optimize`, and prose references like `optimize.build_problem` / `cli.py`.
files=$(git ls-files '*.py' '*.md' '*.toml' '*.sh')
sed -i 's/sttopt\.optimize/sttopt.stto/g; s/sttopt\.cli/sttopt.stto_cli/g' $files
```

Then fix up by hand what `sed` cannot safely do:

- The local alias. `import sttopt.stto as optimize` still binds the name `optimize`; rename the
  alias to `stto` and update every `optimize.` use site in that file. Do this per file, and
  check the result compiles — a blind `s/optimize\./stto./g` would also rewrite
  `scipy.optimize.` and `torch.optim` if either appears.
- `import sttopt.cli as cli` in `tests/test_cli.py` -> `import sttopt.stto_cli as stto_cli`,
  and rename that test file to `tests/test_stto_cli.py`.
- `pyproject.toml`: `[project.scripts]` `sttopt = "sttopt.stto_cli:main"`.
- Prose in docstrings and `.md` files that says `optimize.py` or `cli.py` — including
  `sttopt/conventions.md`, `sttopt/__init__.py`, `sttopt/viz.py`'s module docstring, and any
  `plans/archive/*.md` that a reader would follow. Archived plans are history; update them only
  where a stale name would actively mislead, and do not rewrite their narrative.

**Acceptance**: `rg -n 'sttopt\.optimize|sttopt\.cli\b'` returns nothing outside
`plans/archive/`; `pytest` passes with the same result count as before the rename;
`sh stto.sh --help` works.

---

# Phase 1a — `sttopt/geometry.py` and the geometry generator

New files only. Runs in parallel with Phase 1b.

## `sttopt/geometry.py`

Loading and interrogating a fixed density field. Keep it free of torch — it works in NumPy at
the same boundary `timefield.init_timefield` sits at.

```python
def load_geometry(path: Path) -> Float[np.ndarray, "nely nelx"]:
```
Reads an `.npz` and returns its `xPhys` array. Requirements:
- Require the key `xPhys`. If absent, raise with the list of keys actually present — a user
  pointing at the wrong `.npz` should learn what is in it.
- Validate: 2-D, finite, all values within `[0, 1]`. State the failure in terms of the
  quantity that is wrong (a value out of `[0, 1]` is not a density) rather than the file.
- **Require the field to be binary**, and reject it otherwise. The error must report how many
  elements are intermediate and their extreme values, so a user can tell three boundary
  elements from a wholly grey field. `--non-binarized` on the CLI waives the check and takes
  the field as given.
- Measured caveat that shapes the design: real STTO outputs are **~35% intermediate** at a
  `1e-3` tolerance (checked across three runs in `output/`, all `nelx*nely=10800`, min density
  `5.9e-11`, max `1.0`). The Heaviside projection at `beta_d=128` does not come close to
  binary. So feeding a topology-optimized layout — the Fig. 3.10 workflow, and the main reason
  this tool takes a file at all — trips the check every time.

  `seqopt` does **not** binarize. Converting a topology-optimized layout into a printable part
  is its own step with its own output file, not a hidden transform inside an optimization run —
  so the thresholded geometry is an artefact you can inspect, plot and reuse, and two runs
  naming the same geometry file always mean the same part. That conversion lives in the
  geometry CLI (below). `seqopt` has exactly two states: assert binary, or `--non-binarized` to
  study a grey field deliberately.

```python
def base_elements(
    xPhys: Float[np.ndarray, "nely nelx"],
    candidates: Int[np.ndarray, " k"],
    solid_threshold: float = 0.5,
) -> Int[np.ndarray, " j"]:
```
Filters a set of candidate boundary elements (from `timefield.base_elements`, Phase 1b) down to
those that actually hold material. This is the load-bearing piece: in Fig. 3.4(a) only the left
half of the bottom edge touches the build plate, so "the whole bottom row" is the wrong
print-start set. Raise if the result is empty — a geometry that does not touch its build plate
cannot be printed from it, and that is worth failing loudly rather than optimizing nonsense.

`solid_threshold` is a parameter, not a constant: a grey topology-optimized field and a
hand-drawn binary mask do not want the same cutoff.

## `sttopt/geometry_builders.py` + a CLI to write them out

Parametric builders for the thesis components, rasterized onto an `(nely, nelx)` grid from
millimetre dimensions. Each returns `Float[np.ndarray, "nely nelx"]` of 0.0/1.0, using the
repo's grid convention (row 0 is the **top**; see `sttopt/conventions.md` and note that
`timefield.OPPOSITE_CORNER` measures from `y = nely`, the bottom).

Build the raster by point-in-polygon on element centres, not by hand-written index arithmetic —
it keeps each shape a readable list of vertices.

Shapes, all with the build plate at the **bottom**:

- `overhang_bracket(nelx, nely)` — Wu2025 Fig. 3.4(a), same shape as Fig. 2.13(b) "V-shaped".
  100 mm x 100 mm, vertices in mm with the origin at the bottom-left and `y` up:
  `(0,0) -> (50,0) -> (100,50) -> (100,100) -> (50,100) -> (50,50) -> (0,50) -> close`.
  That is a 50x50 left block, a right column 50 wide and 100 tall, and a 45 degree cut removing
  the triangle below the line from `(50,0)` to `(100,50)`. Only `x` in `[0, 50]` sits on the
  build plate.
- `c_shape(nelx, nely)` — Wu2025 Fig. 2.13(a). 200 mm wide x 240 mm tall, with a rectangular
  slot open at the right edge: `x` in `[60, 200]`, `y` in `[80, 180]`.
- `l_shape(nelx, nely)` — Wu2025 Fig. 2.13(c). 200 mm x 200 mm; the union of a vertical arm
  (`x` in `[0, 80]`, full height) and a horizontal base (full width, `y` in `[0, 80]`).

Give each builder its physical dimensions as keyword arguments with the thesis values as
defaults, so a reader can vary them without editing the body. The aspect ratio fixes the natural
`nelx:nely`; make the CLI take one resolution number (elements along the taller side, say) and
derive the other, so a caller cannot silently distort the component.

CLI, following the repo's light-import pattern (`parse_args` before the heavy imports). Two
subcommands, because there are two ways a geometry comes into existence and both end in the same
kind of file:

```
python -m sttopt.geometry_builders generate --shape overhang_bracket --resolution 100 \
    --out geometries/overhang_bracket.npz

python -m sttopt.geometry_builders binarize output/<tag>/final_design.npz \
    --threshold 0.5 --out geometries/topopt_bracket.npz
```

`binarize` is the answer to the ~35% grey measurement above: it takes a topology-optimized
`xPhys`, thresholds it, and writes a clean binary geometry that `seqopt` then accepts without a
waiver. Record the threshold and the source path in the output `.npz`, so a geometry file always
says where it came from. Report the resulting solid fraction and how many elements changed side,
since a badly chosen threshold can sever a thin member — and print a warning if thresholding
leaves solid material with no path to the build plate, which is the failure that matters.

Writes an `.npz` with key `xPhys`, plus the shape name and the dimensions used, so a geometry
file is self-describing. Default output directory: `geometries/` at the repo root (create it,
and add a one-line `.gitignore`-style decision — commit the small generated `.npz` files or not
is the user's call; do not commit them by default).

## Tests — `tests/test_geometry.py`

- `load_geometry` round-trips an array; rejects a missing `xPhys` key, a 3-D array, and values
  outside `[0, 1]`.
- `base_elements` on `overhang_bracket` returns exactly the bottom-row elements whose column
  centre is left of the 50 mm line, and nothing to the right of it. This is the test that would
  catch "whole bottom edge" reasoning.
- `base_elements` raises for a geometry lifted clear of the build plate.
- Each builder: correct solid fraction to a tolerance set by the raster resolution, correct
  corners solid/void, and **`nelx != nely`** in at least one case (`conventions.md`: a square
  fixture can pass a transposed implementation undetected).

---

# Phase 1b — shared pieces: `sensitivity.py`, `timefield.py`, `run_config.py`

Runs in parallel with Phase 1a; the file sets are disjoint.

## `sttopt/sensitivity.py` — extract the batched-Jacobian helper

`seqopt` has no filter on its design variable, so it needs no filter adjoint. What the two
problems genuinely share is the smaller, subtler half of `stto.py`'s `_sensitivity_rows`: the
`k == 1` versus `k > 1` split, and the one-hot `is_grads_batched` trick that assembles `k`
gradient rows in one call. Extract exactly that, and no more:

```python
def jacobian_rows(
    outputs: Float[Tensor, " k"], leaves: Sequence[Tensor]
) -> tuple[Float[Tensor, "k nel"], ...]:
```

One `(k, numel)` block per leaf, in leaf order, zeros for a leaf the outputs do not depend on
(`allow_unused`). Carry across the load-bearing part of the existing docstring: that
`is_grads_batched`'s vmap has no batching rule for the sparse CSR matmul backward
(`RuntimeError: expand is unsupported for SparseCsc tensors`), so a `k > 1` output must not have
a sparse matmul or a `FemSolve` inside its own graph. That restriction now applies to both
callers and is the single reason this function is shared rather than written twice.

The filter adjoint stays in `stto.py`: `_sensitivity_rows` becomes `jacobian_rows(outputs,
(xTilde, tPhys))` followed by its existing `H`/`Hs` adjoint and `_flatten_pair`, and keeps the
part of its docstring explaining why the differentiation cut sits at the *filtered* fields and
why `H` is its own adjoint. That reasoning is STTO's, not generic, so it should not move.

`seqopt` calls `jacobian_rows(outputs, (t,))[0]` and is done.

Check the restriction holds for `seqopt`'s own `k > 1` rows before relying on it: `start_point`
indexes `t` directly and the stage rows go through `compliance.time_mask`, neither of which
touches a sparse matmul. `time_field_continuity` *does* (`L @ t`), but it is a single row and
takes the unbatched path.

**Acceptance**: STTO's existing tests pass unchanged, bit-for-bit where they were exact before.
This is a pure move.

## `sttopt/timefield.py` additions

1. `TimeField.BOTTOM_EDGE = 4` — a bottom-to-top ramp, for the build plate at the bottom, which
   is what every Wu2025 example uses and what none of the existing three variants give. In
   `init_timefield`: `np.tile(np.linspace(1, 0, nely)[:, None], (1, nelx))`, i.e. `t = 0` on the
   bottom row and `1` on the top. Extend the module docstring's note about lone-1 meshes to
   cover it. This is the planar-layer reference (Wu2025 Fig. 3.4d), kept as a baseline to
   compare against, **not** as `seqopt`'s default initialization.

1b. `TimeField.GEODESIC` — `seqopt`'s default: normalized geodesic distance from the build-plate
   elements, measured **through the material**, so distance respects the part's shape instead of
   its bounding box. In a C-shape the two arms then start from times that reflect the real path
   from the plate, which a planar ramp gets wrong.

   Unlike every other variant this one depends on the geometry, so it does not fit
   `init_timefield(nelx, nely, variant)`'s signature. Give it its own function taking `xPhys`
   and the build-plate element set, and have `seqopt` select it; do not contort the existing
   signature.

   Implementation: a single multi-source Dijkstra over the 8-connected element graph, via
   `scipy.sparse.csgraph.dijkstra` — scipy is already a dependency, so this needs no new one.
   Grid-graph metrication error of a few percent is irrelevant for an initialization; do not
   reach for fast marching.

   Edge cost is the step length (`1` or `sqrt(2)`), multiplied by a fixed penalty when the step
   enters a void element. **One graph over the whole mesh, not a solid-only graph**: the time
   field is defined over every element (void times are design variables — see above), so the
   initialization must be too. A penalized traversal gives that in one mechanism, with no
   infinities to patch, no second field glued on at the material boundary, and no discontinuity
   there for the continuity constraint to immediately fight. Void ends up later than the solid
   front beside it, which is the sensible reading of never being deposited.

   The penalty is a named module constant with a comment, not a config field — it shifts an
   initialization, not a result. Normalize the finished field to `[0, 1]`.

   Handle explicitly rather than by accident: solid elements with no path to the build plate.
   With a finite void penalty they are reachable rather than infinite, so this degrades
   gracefully — but say in the docstring that a disconnected island gets a late time by way of
   the surrounding void, since that is a modelling statement, not an implementation detail.

2. `base_elements(nelx, nely, variant) -> Int[np.ndarray, " k"]` — the candidate print-start
   elements for a variant: `[0]` for `CORNER`, `arange(nely) * nelx` (column 0) for `EDGE` and
   `OPPOSITE_CORNER`, `(nely - 1) * nelx + arange(nelx)` (bottom row) for `BOTTOM_EDGE`.
   `stto.build_problem` currently inlines this exact `Nei` logic — replace that inline block
   with a call, so the two problems cannot drift apart. Keep `build_problem`'s existing comment
   about `m` generalizing from `len(Nei)`.

3. A **selectable** uniformity measure. The point of this is that the user intends to try other
   measures; make adding one a matter of writing a function and adding an enum member.

   ```python
   class UniformityMetric(StrEnum):
       GRADIENT_CV = "gradient_cv"

   def uniformity_penalty(
       tPhys: Float[Tensor, "nely nelx"],
       metric: UniformityMetric,
       weights: Float[Tensor, "nely nelx"] | None = None,
   ) -> Float[Tensor, ""]:
   ```

   Dispatch through a small name -> function mapping. Document `uniformity_penalty` as "the
   selected layer-uniformity penalty", never as "the gradient-magnitude standard deviation" —
   that belongs in the individual metric's own docstring.

   Write exactly one metric. Do not add a second one speculatively, and do not build machinery
   for measures this signature cannot express — a measure that bounds the *range* of the
   gradient (Wu2025 Eq. 3.26-3.27 in spirit) would be a pair of constraint rows, not a scalar
   penalty, so it would change `Problem.m` and need its own design. That is a live future
   direction, not scope here; when it arrives, adding it beside this rather than inside it is
   the expected shape.

4. `GRADIENT_CV` — the density-weighted **coefficient of variation** of `|grad t|` over the
   mesh interior: `m = sum(w*g)/sum(w)`, `s = sqrt(sum(w*(g-m)^2)/sum(w))`, returning `s/m`.
   `weights` is `(nely, nelx)`, cropped to the interior internally to line up with
   `gradient_magnitude`'s output.

   Two separate reasons for this shape, both load-bearing:

   - **Weighted**, so the statistic is taken over the part rather than the bounding box. `t`
     over void is pinned by nothing physical, so unweighted gradients there are noise that
     would dominate the spread on any part that does not fill its box.
   - **Divided by the mean**, so the measure is invariant to mesh resolution. `gradient_
     magnitude` uses central differences in *element* units, so `|grad t|` scales as `1/nelx`
     for a field spanning `[0, 1]`; its raw spread does too. The ratio does not. That makes
     `uniformity_weight` a number that stays valid when the same component is rasterized at a
     different resolution — which is the whole point, since the geometry file sets the mesh.
     It is also directly interpretable: 0.1 means layer thickness varies by about 10% of its
     own mean.

   The hotspot term needs no equivalent treatment: `numer` is already a p-th-power *mean*
   followed by a p-th root, so it is intensive and lands in `[0, 1]` at any resolution. (The
   radii being in element units still moves hotspot values with resolution — that is the
   trade-off recorded under `SeqRunConfig`, and is not something a normalization can fix.)

   With both terms dimensionless and order 1, **there is no `normalize_objective`**: the
   weights are directly meaningful, no scale has to be latched from the initial field, and the
   objective stays comparable across runs, resolutions and components.

   **Do not change the existing `gradient_magnitude_std(tPhys)`** — STTO uses it and its
   fixtures pin it. `GRADIENT_CV` is a new, separate function; it is not a generalization of
   that one, and `weights=None` should not be made to reproduce it.

   Zero total weight (an all-void geometry), or a zero mean gradient (a constant time field),
   returns zero rather than dividing by zero.

## `sttopt/run_config.py` — `SeqRunConfig`

Add a sibling dataclass. **Keep the two configs flat and separate**; do not build an inheritance
hierarchy over the ~14 overlapping fields, because the two problems will not keep them in step.
Factor only `to_dict`/`from_dict` (including the unknown-key warning) into a small shared mixin
that both use.

```python
@dataclass(kw_only=True)
class SeqRunConfig:
    nloop: int

    print_base: str          # a TimeField member name, case-insensitive
    solid_threshold: float   # density above which an element counts as touching the build plate

    lrmin: float             # continuity filter radius
    rmin_cond: float         # conductivity-neighbourhood radius

    hotspot_weight: float
    uniformity_metric: str   # a UniformityMetric member
    uniformity_weight: float

    nStage: int              # per-stage deposition budgets; 0 disables. Also the stage count
                             # the plots draw boundaries for.
    p: float
    q: float
    r: float
    rouf: float

    a0: float
    mma_c: float
    tmove: float
```

No `nelx`/`nely`: the geometry file defines the mesh, so a mismatch between config and geometry
is not merely caught, it is unrepresentable. No `rmin` either: the time field is not filtered,
so there is no radius to set. Note both in the class docstring — a reader comparing against
`RunConfig` will otherwise assume they were forgotten.

`lrmin` and `rmin_cond` are counted in **elements**, as in `RunConfig`, not in the geometry's
physical units. Since the geometry file now sets the mesh, rasterizing the same component at a
different resolution changes what those radii mean physically, and they have to be rescaled by
hand to match. State this on both fields — it is the one place a reader can be caught out by
the mesh coming from somewhere else.

`configs/seq_default.json` holds the defaults. Reuse the STTO values for the shared numerics
(`p=25, q=3, r=0.05, rouf=100, lrmin=2, rmin_cond=12, a0=1, mma_c=2500, tmove=0.01`),
with `print_base="bottom_edge"`, `solid_threshold=0.5`, `nStage=8`,
`uniformity_metric="gradient_cv"`, and both weights `1.0`. Both objective terms are
dimensionless and order 1, so those weights mean what they say and carry over to other
components and resolutions unchanged.

## Tests

Extend `tests/test_timefield.py`: `BOTTOM_EDGE`'s range and orientation (0 at the bottom row);
`base_elements` for all four variants; `uniformity_penalty` dispatch, including that an unknown
metric name raises; weighted vs unweighted spread on a field where a masked-off region has a
deliberately different gradient, so the weighting demonstrably changes the answer; zero-weight
returns zero. Add a `SeqRunConfig` round-trip and unknown-key-warning test wherever the
`RunConfig` equivalents live.

---

# Phase 2 — the driver, CLI, and plots

Depends on Phase 0, 1a and 1b.

## `sttopt/seqopt.py`

Mirror `stto.py`'s shape (`Problem` / `State` / `IterationRecord` / `build_problem` /
`init_state` / `step` / `run` / `run_from_state`) so that a reader who knows one knows the
other. Do **not** try to share `step` between them with flags; the wiring is exactly what
differs, and a merged `step` full of conditionals is the god function the repo's style guide
warns against.

`Problem`: `config`, `device`, `dtype`, `xPhys` (the fixed geometry, as a tensor), `nelx`,
`nely` (derived from `xPhys.shape`, since the config has none), `L`, `e1`, `e2`, `w`, `Nei`,
`m`, `n`. **No `H`/`Hs`** — the time field is not filtered. Note `n = nel`, and
`m = 1 (continuity) + len(Nei) + 2*nStage`.

`build_problem(config, xPhys, *, device=None, dtype=torch.float64)` takes the geometry as an
argument — loading a file is the CLI's job, not the problem builder's.

`State`: `t`, `xold1`, `xold2`, `low`, `upp`, `loop`, `beta_t`, `factor`. No `beta_d`, no `U`,
no objective scales, and **no separate `tPhys`**: with no filter it would be a
second name for the same tensor, and two names for one field is how they drift apart. Use `t`
throughout `seqopt`. The physics functions name their own parameter `tPhys`; that is their
convention, and passing `t` positionally into it is correct.

`init_state`: the geodesic time field, used as-is. Nothing else to set up — both objective
terms are already dimensionless and order 1, so no scale is latched from the initial field.

`step`:
- `t` is the single autograd leaf, and is itself the physical time field. `xPhys` is a
  constant, and must never require grad.
- Objective: `hotspot_weight * numer + uniformity_weight * penalty`, where `numer, K_est = conductivity.hotspot_value(xPhys, t, e1, e2, w, p, q, r, rouf)` and
  `penalty = timefield.uniformity_penalty(t, metric, weights=xPhys)`.
- Constraints, in a fixed documented order: `constraints.time_field_continuity(t, L)`, then
  `constraints.start_point(t, Nei)`, then, when `nStage > 0`, the interleaved
  upper/lower stage rows exactly as `stto.step` builds them.

  Reuse `constraints.stage_volume_bounds` unchanged by passing **`volfrac = float(xPhys.mean())`**:
  its scale factor is `nelx*nely*volfrac`, which then equals `xPhys.sum()`, turning the row into
  "fraction *of the part* deposited by `t_stage`, versus `t_stage`" — the right statement for a
  fixed geometry. Put that one-line derivation in a comment; it is not obvious from the call
  site. Note that Wu2025 §3.2 deliberately omits per-layer volume constraints and discusses the
  consequence, which is why this is optional rather than always on.
- Move limits: `config.tmove` on `t` only, clamped to `[0, 1]`.
- Sensitivities via `sensitivity.jacobian_rows(outputs, (t,))[0]`. No filter adjoint.
- `mma.mmasub` with `n = nel`.
- Periodic updates, both deferred to take effect the *next* iteration, same convention and
  rationale as `stto.step`: `beta_t += 5` every 30 iterations capped at 50, and the hotspot
  `factor` refresh every 25 iterations. There is no Heaviside sharpening, since there is no
  density projection.

`IterationRecord`: `f`, `hotspot` (the raw `numer`), `uniformity` (the raw penalty), `tru_max`,
plus the MMA outputs the STTO record carries.

**Port `factor` and `tru_max` exactly as `stto.step` computes them**: `factor` refreshed every
25 iterations from `max((1 - K_est) * xPhys**r) / numer`, taking effect the following iteration,
and `tru_max = factor * numer`. It is a smooth, differentiable rescaling of the p-norm
aggregate, so it stays comparable to `Tcr` and to STTO runs, and it does not introduce the hard
maximum's kinks into anything downstream. Keep `state.factor` on `State`.

The objective itself minimizes **`numer`**, not `factor * numer`. Reason: `factor` steps
discontinuously every 25 iterations, which is harmless
in `tru_max` (a reported number) and harmless in STTO (a constraint row MMA re-linearizes
anyway), but as an objective it would rescale `df/dt` in a jump. If a `Tcr`-comparable objective
turns out to matter more than a smooth scale, multiplying by `factor` is a one-line change.

## `sttopt/seqopt_cli.py`

Model on `stto_cli.py`, including the light-import pattern. Flags: `--geometry` (required),
`--config` (required), `--tag`, `--tag-force`, `--device`.

Artefacts under `output/<tag>/`:
- **`seq_config.json`** — deliberately a different filename from STTO's `config.json`. `viz.py`
  dispatches on which one is present, which beats sniffing the key set of a JSON object.
- `geometry.npz` — a copy of the input `xPhys`, so a run directory reproduces itself without
  the original file.
- `design_it####.npz` every 50 iterations, and `final_design.npz`.

`final_design.npz` must carry the keys **`xPhys`, `tPhys`** (plus `loop` and the record
scalars), because `viz.py` already reads exactly those two — matching them is what makes the
whole plotting path work unchanged. Here `tPhys` is simply `t`, written under the name the
plotting code reads; say that in a one-line comment at the `savez` call, so nobody later hunts
for a filtering step that does not exist. Do not also write a separate `t` key holding the same
array.

Progress line, one per iteration:
`It.: %4d f: %10.4f hot: %8.5f unif: %8.5f Tm.: %7.3f`

## `sttopt/viz.py`

`_main` currently builds a full STTO `Problem` merely to reach `e1`/`e2`/`w`, and computes a
compliance for the plot titles. For a `seqopt` run directory:
- Detect it by the presence of `seq_config.json`.
- Get the conductivity neighbourhood straight from `conductivity.neighbor_weights(nelx, nely,
  rmin_cond)` with `nelx`/`nely` read off the saved `xPhys`. No `Problem`, no FEM.
- Pass `compliance=None` to the title helpers — `_timefield_title` already accepts it.
- `nStage == 0` must not draw stage boundaries; check `stage_boundary_plot` and
  `hotspot_severity_plot` handle that and guard at the call site if not.

Keep the STTO path byte-identical. If the two paths start fighting each other inside `_main`,
split the run-directory loading into two small functions returning the same tuple, rather than
threading conditionals through the plotting calls.

Add the new CLI to `pyproject.toml`'s `[project.scripts]` (`sttopt-seq`) and add `seqopt.sh`
beside `stto.sh`.

## Tests

- `tests/test_seqopt.py`: a tiny non-square mesh (e.g. 7x5 from a builder or a hand-made mask).
  - `step` returns finite values and a state whose fields are all finite and in bounds. Do
    **not** assert that the time field passes through unsmoothed: whether to smooth it is an
    open question, and a test would freeze a guess into a requirement.
  - `xPhys` never acquires a gradient, and `t` is the only design variable — assert the MMA
    vectors have length `nel`, not `2*nel`.
  - Constraint row count matches `problem.m`, for both `nStage=0` and `nStage>0`.
  - A finite-difference check of `df/dt` and of each constraint row against the autograd
    result, following whatever pattern `tests/test_optimize.py` already uses.
  - Over a handful of iterations from a deliberately bad initial field, the hotspot term goes
    down. Keep it small; per the repo rules nothing near production scale runs in tests.
  - Resolution invariance of the uniformity term: the same component rasterized at two
    resolutions gives close `GRADIENT_CV` values for the same *shape* of time field. That is
    the property `uniformity_weight`'s portability rests on, so it is worth pinning even
    loosely.
- `tests/test_seqopt_cli.py`: smoke test mirroring `tests/test_stto_cli.py` — 2 iterations on a
  tiny geometry, asserting `final_design.npz`, `seq_config.json` and `geometry.npz` all land and
  that `final_design.npz` carries the `xPhys`/`tPhys` keys `viz.py` needs.
- Extend `tests/test_viz.py` to cover regenerating plots from a `seqopt` run directory.

---

# Phase 3 — documentation and integration

- `sttopt/__init__.py`: describe both problems, not just STTO.
- `sttopt/conventions.md`: record that the two problems treat the time field differently — STTO
  filters it with the density filter, `seqopt` uses it raw — and why (smoothing couples print
  times across void). Keep it to a few lines; the reasoning lives here in the plan.
- `sttopt/conductivity.py`: two deviations from Das2025 are currently undocumented and should be
  named where the code implements them. The paper's weight is radial **times angular**, the
  angular part favouring the build direction; the port keeps only the radial, because the build
  direction is not fixed when the deposition order is a design variable — a known future
  direction, not an oversight. And the paper's Eq. (6) weights neighbours by `rho_j` where the
  port uses `x_j^q` with `q = 3`, a SIMP-style penalization of intermediate density that the
  paper does not have.
- A short section in the repo's structure notes describing when to use `stto_cli` versus
  `seqopt_cli`.
- Move this plan to `plans/archive/` and update `plans/CLAUDE.md`.

# Commits

One commit per phase, in phase order, each self-contained and passing tests on its own. The
Phase 0 rename is large in diff and trivial in logic; everything else is the reverse. Do not mix
them.
