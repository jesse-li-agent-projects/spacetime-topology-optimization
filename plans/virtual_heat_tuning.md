# Virtual-heat time field: schedule tuning and validation

The code of `plans/archive/virtual_heat_timefield.md` (Phases 0–7) is complete:
`stto_heat.py` (Variant 1) and `stto_laplace.py` (Variant 2). This plan is the tuning of
their schedules (Phase 8) and the validation matrix (Phase 9). Read the archived plan
for the design, the variants and the settled decisions.

Its rules still apply: **ask the user often**, and **keep a margin on every calibrated
value**, stated in terms of the quantity that fails.

## Pass criteria

On the **dev set**, a schedule passes when:
- the hotspot row is ≤ 0.01 on the design as optimized (the `constraints_continuous`
  of the check report), i.e. the constraint holds at the end. The row is active at the
  optimum, and binarizing moves it by up to +0.0145 (a `stto` control), so the
  binarized row is reported, and the check warns above a rise of 0.02,
- the tool-radius row is ≤ 0 on the binarized design (Variant 2 only; the user
  dropped it for Variant 1: the heat fronts meet in seams around holes, and the
  iso-lines there have near-zero concave radius),
- the start and support checks pass, and
- the compliance is ≤ 1.10 × the matched `stto` control.

Start (the virtual-heat scripts, `virtual_heat.start_report`) = the part touches the
plate, and every node of a solid element except the plate's own has `t > 0`. It is
judged on the nodes, where `t` is solved and the plate has `t = 0` exactly. The first
ordering of element means (no element off the base before a base element) failed by
near-ties of 0.07–0.7 of a layer: a base element's mean is half the local rise of `t`
over one element, so it differs with the local gradient (user decision). The
element-mean ordering is reported. `stto` keeps its own start check, where `t` is per
element and the base is pinned to 0.

Support (all three scripts, `checks.support_report`) = no unsupported element in the
part's interior (all 4 edge neighbours solid; outside the domain is void). An
unsupported element on the boundary reads as a steep overhang; the user accepts it as
printable (a heuristic). Every unsupported element measured in Phase 8 was on the
boundary, mostly ties within one layer. The domain-level check of Variant 2 is
reported, not judged.

On the validation matrix, report these values. Do not gate on them.


## Phase 8: tuning

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
   - **Variant 1 keeps the hotspot only** (user, 2026-09-30). `gradient_smoothness`:
     91–95% of its violations sat on the part boundary, where the stencil reads the
     void `t`, which only follows the floor diffusivity; it pinned rows at `mma_c`,
     broke C1 into islands, and cost S 17% compliance. `min_gradient`: the Poisson map
     keeps `|grad t| ≥ 0.79 × median` on 99% of the part without it; the elements
     below its floor were merge points (wanted) and part-boundary stencil artefacts,
     and it drove the S conflict that split the 600-iteration runs.
4. Variant 2 only: if the ramp init is not robust, try the alternative init (archived plan, "Initialization").
5. Compress from 800 to 600 iterations (start by scaling the change points by 0.75).
   Use the same pass criteria. If it fails, report the gap to the user.

**`T → t` priority.** The user chose these on a hunch, to start tuning early:
- Variant 1: `poisson`.
- Variant 2: `identity`, with `_gradient_cv` (`uniformity_weight > 0`) for uniform
  layers.

The other options (`neg_log` for Variant 1, `poisson` for Variant 2) are *lower
priority, not rejected*. Phase 4 implemented and tested them. Tune them only
if the primary option fails, or if the user asks.

**Robustness means one schedule across design problems.** Passing replicates of one
problem is easy and tells little; what matters is that the same schedule passes on
different load cases, volume fractions, bases and Tcr. Judge a schedule change on
several cells, not on replicates of one (user, 2026-09-30).

Judge a change against a matched control (the same schedule without the change).
Replicate the marginal cells on CPU and GPU only to tell chance from a systematic
effect. Watch the multipliers: rows at `mma_c`
mean a schedule conflict. Record the final schedule and its reasons in a
`configs/*.md` note, as `configs/continuation.md` does for `stto`.

**Variant 1 schedule tuning (2026-10-01, set aside; see the last item).** Hotspot only, no gravity
stages (`Theta = 0`, `nStage = 0`; user decision, because stage solves failed CG at low
volfrac and the stages weigh 1–39% of the objective). Cells: the dev set and the three
hardest Phase 9 cells (volfrac 0.3: C1 edge/0.6, S bottom/0.6, D bottom/0.8). Ratio to
the matched `stto` control; D at 0.3 on the compliance as optimized, since binarizing
breaks its fully loaded traction edge for every method, `stto` included.
- The 600-iteration step schedule passes 5/6 and fails C1 at 0.3 (2.4×): when β_d
  steps 8 → 16 at iteration 150, the low-volfrac design loses its load path.
- O1, the same with one log ramp of β_d (1 → 128 by 262) in place of eight steps,
  passes 5/6 and fails C1 at 0.3 by less (1.25–1.42× over three samples).
- A, one window (β_d 8 → 128 over 360, Tcr over [0, 180]), passes 5/6 and fails S at
  0.5 (1.15×, both devices).
- Seven other single-window variants failed two or more cells.
- Mechanism: C1 at 0.3 needs a sharp projection and a formed layout before the hotspot
  acts, and then a slow Tcr ramp, which the hotspot row can follow (A: row ≈ 0
  throughout). S at 0.5 needs the hotspot later than A's, or it shapes a web of thin
  members before compliance sets the layout. S at 0.3 fails when the design is
  nearly binary before the hotspot acts (B, G).
- P, three stages (β_d log 1 → β_mid over 0–120; Tcr 1 → final over 120–300 at fixed
  β_d and penal 2; β_d → 128 and penal 2 → 3 over 300–420; nloop 540). β_mid = 4 is the
  user's base (best C1 0.3 topology). P4 passes C1 0.3 (1.075) and S 0.5 and fails S 0.3
  (1.30, hotspot +0.44). Every failing run holds the hotspot multiplier at `mma_c` for
  hundreds of iterations while the violation grows.
- On S 0.3 (P4, it 250) the `poisson` time field had a basin in the middle of the top
  bar: the leak through near-void that `penalize_conduction` is for. What is left at
  `p = 3`, β = 9 on that checkpoint is two one-node minima on the bar's border, 0.09%
  of the time range deep, at element-scale `μ` contrast. No `μ` filter for now (user,
  2026-10-01).
- **Set aside (user, 2026-10-01).** Variant 1 has difficulty defining the time at the
  top of the part accurately until the topology is sketched out. P4 on S 0.3, three
  conduction choices (compliance × `stto`):

  | `χ` density exponent | `drain_beta` | Run | Result |
  |---|---|---|---|
  | 1 | 25 | `w31_heat_S_P4` | 1.30, hotspot +0.44: near-void leaks heat, so the top bar prints from its middle |
  | 3 | 9 | `w32_heat_S_P4pc` | 31.6: on the grey early design `T` falls below the solve error on 31% of the part, so the time map's gradient is noise |
  | `penal` (2 → 3) | 4 | `w33_heat_S_P4pp` | 12.7, hotspot +0.61: gradient correct, but the volume falls from iteration 80; the uniformity term's gradient is 1.8× the compliance gradient (one checkpoint, not proved by a run) |

  Without penalization the grey design's time field is wrong at the top; with it, the
  time field is too sensitive to the grey design to steer by.

**Variant 2 status (paused 2026-09-30, to finish Variant 1 first).** Runs in
`/tmp/claude-1001/phase8/work/output/w19_*`–`w21_*`; gates as above (hotspot, tool
radius, `min_gradient` ≤ 0.01 as optimized; start on the nodes; interior support).
- `identity` with unimodal wall data failed S at step 1 (hotspot +0.13 to +0.62: the
  single peak went to the tip of a branch to the clamped wall). `poisson` passed step
  1 on all cells but failed S whenever the tool radius was on.
- `identity` with free, filtered wall data (current code): step 1 passes on C1, D, S
  (compliance 0.95–1.002 × `stto`). On S the wall data finds two peaks, the ends of
  the top bar, with a dip where the diagonal joins it. Steps 3a–3c pass on C1 and D.
  S passes 3b on both devices; it fails 3a on one of two runs (tool radius +0.15)
  and 3c on CPU (`min_gradient` +0.055, at the top bar's merge point, where
  `|grad t| → 0`). `stto` also fails S with all constraints (hotspot +0.59).
- Open: how to judge S with all constraints; the S 3c GPU replicate; step 5
  (600 iterations). Undulating wall data on D lies in the void only; the time field on
  the part is clean, so it is left as it is.

## Phase 9: validation matrix

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
