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
   variable names that assume the gradient-magnitude standard deviation is *the* uniformity
   measure. It is *one* measure, selected by name. Prefer `uniformity_*` naming over `Gamma`.
   (`RunConfig.Gamma` on the STTO side keeps its name; do not touch it.)
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
- Do **not** threshold or binarize. A grey field from a topology optimization is a legitimate
  input; the conductivity proxy consumes densities, not a binary mask.

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

CLI, following the repo's light-import pattern (`parse_args` before the heavy imports):

```
python -m sttopt.geometry_builders --shape overhang_bracket --resolution 100 --out geometries/overhang_bracket.npz
```

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
   cover it.

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
       GRADIENT_STD = "gradient_std"

   def uniformity_penalty(
       tPhys: Float[Tensor, "nely nelx"],
       metric: UniformityMetric,
       weights: Float[Tensor, "nely nelx"] | None = None,
   ) -> Float[Tensor, ""]:
   ```

   Dispatch through a small name -> function mapping. Document `uniformity_penalty` as "the
   selected layer-uniformity penalty", never as "the gradient-magnitude standard deviation" —
   that belongs in the individual metric's own docstring.

4. Density weighting for `GRADIENT_STD`. `weights` is `(nely, nelx)`, cropped to the interior
   internally to line up with `gradient_magnitude`'s output. Weighted mean and spread:
   `m = sum(w*g)/sum(w)`, `sqrt(sum(w*(g-m)^2)/sum(w))`.

   **Do not change the existing `gradient_magnitude_std(tPhys)` signature or its `torch.std`
   (Bessel-corrected) result** — STTO uses it and its fixtures pin it. The weighted form is a
   different normalization, so `weights=None` should keep routing to the existing function
   rather than to a ones-weighted call that would differ by the `n` vs `n-1` factor. Say so in
   one line in the docstring; a silent mismatch between those two paths is exactly the kind of
   thing that wastes a day later.

   Zero total weight (an all-void geometry) returns zero, matching the existing
   fewer-than-two-samples behaviour.

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
    normalize_objective: bool  # scale each term by its value at the initial time field

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

`configs/seq_default.json` holds the defaults. Reuse the STTO values for the shared numerics
(`p=25, q=3, r=0.05, rouf=100, lrmin=2, rmin_cond=12, a0=1, mma_c=2500, tmove=0.01`),
with `print_base="bottom_edge"`, `solid_threshold=0.5`, `nStage=8`,
`uniformity_metric="gradient_std"`, `normalize_objective=true`, and both weights `1.0`.

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

`State`: `t`, `xold1`, `xold2`, `low`, `upp`, `loop`, `beta_t`, `factor`, and the two objective
scales. No `beta_d`, no `U`, and **no separate `tPhys`**: with no filter it would be a
second name for the same tensor, and two names for one field is how they drift apart. Use `t`
throughout `seqopt`. The physics functions name their own parameter `tPhys`; that is their
convention, and passing `t` positionally into it is correct.

`init_state`: `t` from `timefield.init_timefield`, used as-is. When
`config.normalize_objective`, evaluate both objective terms once at this initial `t` and store
them as `hotspot_scale` / `uniformity_scale`; otherwise store `1.0`. Doing it here rather than
latching on iteration 1 keeps `State` free of `None`-valued scales and keeps the scales a
deterministic function of the initial field. Guard against a zero scale.

`step`:
- `t` is the single autograd leaf, and is itself the physical time field. `xPhys` is a
  constant, and must never require grad.
- Objective: `hotspot_weight * numer/hotspot_scale + uniformity_weight * penalty/uniformity_scale`,
  where `numer, K_est = conductivity.hotspot_value(xPhys, t, e1, e2, w, p, q, r, rouf)` and
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

The objective itself minimizes **`numer`**, not `factor * numer`, and lets `normalize_objective`
handle its scale. Reason: `factor` steps discontinuously every 25 iterations, which is harmless
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
  - `normalize_objective=True` makes the initial objective equal `hotspot_weight +
    uniformity_weight`; that is a sharp, cheap assertion on the scaling.
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
- A short section in the repo's structure notes describing when to use `stto_cli` versus
  `seqopt_cli`.
- Move this plan to `plans/archive/` and update `plans/CLAUDE.md`.

# Commits

One commit per phase, in phase order, each self-contained and passing tests on its own. The
Phase 0 rename is large in diff and trivial in logic; everything else is the reverse. Do not mix
them.
