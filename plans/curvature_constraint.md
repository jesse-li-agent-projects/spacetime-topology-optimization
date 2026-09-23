# A tool-radius constraint on the time field's iso-line curvature

## The problem

A layer is deposited along an iso-line of `t`. Where that iso-line is concave as the tool
sees it, material already printed stands up on both sides of the tool, and once the
concave radius of curvature is smaller than the tool radius the tool collides with the
part. Printing successive shells of a sphere from the outside in fails this way; printing
them from the inside out never does.

`stto` has no term that sees this. This plan adds a hard MMA constraint: the concave
curvature of every iso-line in the part stays below `1 / R`, with `R` the tool radius.
`stto` only; `seqopt` is out of scope for now.

## Formulation

With `grad t` per unit length (`timefield.unit_length`), as every other `timefield` term:

```
n     = grad t / sqrt(|grad t|^2 + eps_n * gbar^2)
kappa = div n
g     = smax_e( R * -kappa_e * rho_e**r ) - 1  <=  0
```

- `n` points from printed toward unprinted material, so `kappa > 0` is a convex printed
  region and `kappa < 0` a concave one. Sphere shells outside-in: `n = -rhat`,
  `kappa = -1/r`, constrained. Inside-out: `kappa = +1/r`, free.
- **Concave only.** A convex front is never a collision, and a corner start
  (`TimeField.CORNER`) has `kappa = +1/r`, unbounded at the start point; a symmetric
  penalty would fight the start-point constraint.
- `gbar` is the density-weighted mean `|grad t|`, so `eps_n` is relative, as in
  `_gradient_cv`. See Risks for why `eps_n` is not `GRAD_EPS`.
- `smax` is the calibrated LogSumExp smooth maximum the hotspot row uses: scheduled
  sharpness `curvature_beta`, calibration offset refreshed on the hotspot's refresh rule
  (every `hotspot_refresh_period`, and whenever `curvature_beta` moves).
- `rho**r` uses the **hotspot severity's density exponent `r`**, with the same NaN-safe
  handling of exact-zero density. Void scores zero severity and can never violate.
- **`R` sits inside the aggregate** (as implemented; the first draft had it outside), so
  the severity is dimensionless, 1 on the bound, and `curvature_beta` reads like
  `hotspot_beta`. A change in `R` refreshes the calibration, as a change in `rouf` does.
- `R` is set in **elements** (`tool_radius`) and converted to unit lengths in one place;
  moving to physical units later is a change to that conversion only.

### Continuation

`tool_radius` is `Scheduled`. `R = 0` gives `g = -1`: a zero-radius tool cannot collide,
so this is the constraint's natural "off", not a data trick. A ramp `0 -> R_target` is the
partially applied constraint.

The row exists only when the `tool_radius` schedule is not identically zero, derived from
the setting rather than a separate flag (cf. `enable_continuity`). This keeps `m` fixed
for the whole run and leaves the MATLAB reference row order (`tests/
matlab_reference_loop.py`) untouched for existing configs. The row is appended after the
hotspot row.

## Discretization

A Q4 cell has `t_xx = t_yy = 0`, so the explicit second-derivative curvature formula does
not work on the existing Gauss stencil. Use the divergence form instead (standard
level-set practice): `n` on the dual grid from `_q4_gauss_gradient`, then its discrete
divergence back onto elements -- a compact 3x3 stencil. It is blind to the checkerboard
mode; `relative_roughness` already penalizes that (PR #94).

No finite-difference sensitivity test: every operation is a built-in torch op.

## Phases

Each phase is its own PR.

### Phase 1 -- `timefield.iso_curvature`

`iso_curvature(tPhys, weights) -> kappa` per element, per unit length. `weights` only
feeds `gbar`.

Tests:

- Concentric circles give `kappa = +-1/r` in the sphere-shell sense above.
- A linear ramp at any orientation gives zero.
- The same physical field at two resolutions gives the same `kappa` (cf.
  `test_relative_roughness_is_resolution_invariant`).
- A near-flat solid plateau with element-scale noise keeps `kappa` bounded. This test
  picks `eps_n`.

### Phase 2 -- a generic calibrated smooth maximum

Pull the calibrated LogSumExp out of `conductivity.LogSumExp` into a smooth maximum over
an arbitrary severity field, which both the hotspot row and the curvature row own an
instance of. Pure refactor: hotspot results bit-identical.

### Phase 3 -- the constraint row

- `run_config.RunConfig`: `tool_radius` (`Scheduled`, elements) and `curvature_beta`
  (`Scheduled`). `configs/default.json`: `tool_radius: 0`.
- `stto.step`: the row above, when enabled.
- Diagnostics: the true maximum concave `kappa` (in elements^-1, so it reads against
  `tool_radius` directly), the current `R`, and the calibration offset.
- `viz.py`: a `kappa` map, concave side highlighted.

Tests: `tool_radius: 0` leaves the row stack and the whole iteration identical to today;
a nonzero schedule adds exactly one row; `R = 0` mid-schedule gives `g = -1`.

### Phase 4 -- diagnose existing designs

On `output/stto_k2p37` and its `kappa = 0` control: `kappa` maps and the largest `R`
each final design already satisfies. Picks `R_target` and a starting `curvature_beta`.
No optimization in this phase.

#### Phase 4 results -- do not re-run

Final designs, severity `-kappa * xPhys**0.05` per element. Void is exactly 0 after
projection, so `r = 0.05` lets no void through. `R` is `1 / curvature`, in elements.

| run | max concave (1/el) | admissible `R` | `R` without worst 0.5% / 1% / 5% of solid |
|---|---|---|---|
| `stto_k2p37` | 1.742 | **0.57** | 0.99 / 1.46 / 3.52 |
| `stto_ctrl`  | 2.179 | **0.46** | 0.99 / 1.52 / 3.66 |

Share of solid elements that violate: `R = 1`: 0.5%, `R = 2`: ~1.9%, `R = 3`: 4.0%,
`R = 5`: ~6.7%, `R = 8`: ~12%. The angular weight changes almost none of this.

**The worst values are creases, not bends.** They lie on one-element-wide lines, many of
them exactly along grid rows (`plot/<tag>/iso_curvature.png`). Across one (`stto_k2p37`,
column 75, rows 42-58) `tPhys` is a zig-zag: the slope flips from -0.0055 to +0.0055 per
element in one step at row 48, and back at row 52. The raw `t` has a one-row trough
there (0.28 against about 0.40 on each side). The time filter's cone kernel makes a
one-row spike into a tent with a sharp apex, so `tPhys` keeps a crease. Raw `t` in the
solid has a median `|laplacian|` of 0.04 and a maximum of 0.54, against 0.0005 and 0.012
for `tPhys`: `relative_roughness` scores `tPhys` and never sees the spikes behind it.

Consequences for Phase 5:

- Any `R_target >= 1` element is active from the start. To remove these creases, the
  optimizer must smooth the raw `t`, not only bend fronts.
- Ramp up from `0` to a first target of 2-3 elements, where 2-4% of the solid violates.
  Start `curvature_beta` at 100, the same as `hotspot_beta`, because the severity is 1 on
  the bound in both rows.
- This probably relates to the current `seqopt` smooth-layers question: layer
  smoothness measured on `tPhys` can look good while the raw `t` is a sawtooth.

### Phase 5 -- tune the continuation (GPU)

From the `stto_k2p37` config: `tool_radius` ramps `0 -> R_target` at a few onsets, each
judged against a matched `tool_radius: 0` control with the same angular-weight and `Tcr`
schedules -- never against the best previous run. Report compliance, hotspot severity,
uniformity, roughness, and the true maximum concave `kappa`.

Open questions this phase answers:

- Onset relative to the `Tcr` onset (150-250).
- Whether the curvature row and the hotspot row compete for the same `t` freedom.
- `curvature_beta`: sharp enough to track the true maximum, soft enough to spread the
  sensitivity past one element.

#### Phase 5 results

Recipe, in `configs/stto_tool_radius.json` (the `stto_k2p37` config plus these three):

```json
"tool_radius":    {"points": [[0, 0.0], [300, 0.0], [400, 2.5]], "mode": "linear"},
"curvature_beta": {"points": [[0, 10.0], [500, 10.0], [650, 100.0]], "mode": "linear"},
"tmove":          {"points": [[0, 0.01], [450, 0.01], [550, 0.001]], "mode": "log"}
```

**`curvature_beta` is the knob that matters, not the `tool_radius` onset.** At beta 100
the row's linearization is wrong by an order of magnitude: on a real iterate, a step at
the move limit is predicted to lower the row by 0.63 and lowers the true maximum by
0.05. MMA then either leaves the row violated with a near-zero multiplier, or pins the
multiplier at `mma_c`, where the row outweighs the uniformity term and the optimizer
turns the concave ridges of `t` into convex troughs (layers become a patchwork). At beta
10 the prediction is 0.21 against 0.16. Because the calibration is refreshed every
iteration, a soft beta does not loosen the bound; it only spreads the gradient.

**A shrinking `tmove` is what makes the final beta 100 hold.** At beta 100 the gradient
sits on ~2 elements, so a full `tmove` step lets other near-bound elements drift over.
Without it, `R = 10` held while beta was soft and then collapsed to 6 or less; later or
gradual sharpening alone does not fix this. `tmove` is `Scheduled` for this.

**The late onset (300-400, after `Tcr`) keeps the topology.** On 180x60 the onset makes
no measurable difference; on 120x90 an onset during topology formation lands in a
different topology at +10% compliance, and the late one costs +1.8%.

Final designs, each against its matched `tool_radius: 0` control. `tail g` is the
largest curvature row over the last 50 iterations; `adm. R` the final admissible radius.

| run | problem | c | `true_max` | unif | adm. R | tail g |
|---|---|---|---|---|---|---|
| control | 180x60 | 180.64 | 0.800 | 0.035 | 0.58 | -- |
| recipe, R 2.5 | 180x60 | 180.91 | 0.800 | 0.036 | 2.50 | +0.013 |
| recipe, R 5 | 180x60 | 180.51 | 0.800 | 0.036 | 5.00 | +0.013 |
| recipe, R 10 | 180x60 | 181.52 | 0.800 | 0.040 | 9.58 | +0.066 |
| control | 120x90 | 34.16 | 0.805 | 0.060 | 0.36 | -- |
| recipe, R 2.5 | 120x90 | 34.78 | 0.800 | 0.080 | 2.50 | +0.002 |

What does not work, at R 2.5 on 180x60 unless stated:

| change from the recipe | c | adm. R | why |
|---|---|---|---|
| beta 100 throughout, onset 150-250 | 183.12 | 2.28 | row at `mma_c` for ~300 iterations; uniformity 0.31 |
| beta 100, constant R 2.5 | 190.81 | 1.00 | creases still form by iteration 25 |
| beta 30 during the ramp | 181.79 | 2.50 | works, but costs ~1 in `c` |
| constant `tmove`, R 10 | 182.98 | 6.11 | collapses after sharpening |
| constant `tmove`, R 10, sharpen 700-780 | 182.89 | 9.29 | still flickers to +0.45 |
| `hotspot_kappa` ramp 0 -> 2.37 over 250-450 | 180.87 | 2.35 | no gain: the two rows do not compete once beta is soft |

Other findings:

- **Creases form at iteration ~25** in the control, long before the `Tcr` onset.
- **Soft beta removes the creases rather than moving them.** The designs keep the
  control's topology on 180x60, and the raw `t` is no rougher (median `|laplacian|` in
  the solid 0.035-0.039 against 0.039).
- **The remaining cost is `subsolv` iteration-cap hits** in the small-`tmove` phase:
  0-1 at R 2.5 and 5, ~140 at R 10 and ~280 on 120x90, which also slow those runs to
  15-25 minutes (about 9 without the shrink). The designs are unaffected. The `t`
  asymptote clamps shrink with `tmove` while the `x` ones stay at `move`, which is the
  likely cause. A `tmove` floor of 0.002 halves the hits on 120x90 with the same
  design, but leaves R 10 at 9.57 instead of 9.72.
- R 10 reaches ~96% of the target with the same schedule. The residual is the same
  end-of-run flicker the hotspot row has at `hotspot_refresh_period = 1`, only larger.

## Risks

- **Plateau noise.** `GRAD_EPS = 1e-12` suits `|grad t|`, but in `n` it amplifies
  element-scale noise on a `t` plateau by up to ~1e6 in `kappa`, and a max-type
  constraint finds exactly those spikes. Settled in Phase 1: `timefield.NORMAL_EPS =
  1e-2` reduces the noise response from 49 to 0.03 per unit length, and changes circle
  curvature by 0.5%.
- **Resolution invariance of the design, not of the metric.** `kappa` is per unit length,
  but `time_filter_rmin` and `rmin` are in elements. The cone filter does **not** limit
  curvature to about `1 / r_filter`: a spike in the raw `t` keeps a crease at the apex,
  about 1 per element at any resolution (Phase 4). The optimized design is resolution
  invariant only if the filter radii scale with the mesh, the same limit the existing
  terms have.
- **Void gaps are ignored by design.** A notch spanning a void gap (two arms of a U
  leading the layer) is not seen. Accepted as out of scope.
