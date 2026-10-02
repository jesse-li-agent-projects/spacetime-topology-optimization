# S 0.3: a hotspot that needs material behind the print front

Ask the user before any design decision this plan does not make.

## Problem

`stto` on S 0.3 (`corner_loads`, 120×120, volfrac 0.3, `bottom_edge`, Tcr 0.6, no
gravity; run `w25_stto_S_hot`) prints the middle of the top bar (columns ~64–96) before
its ends: a shallow time basin (0.917 in the middle, 0.96–0.98 at the sides) with void
below it. The support check finds 7 interior unsupported elements there. The hotspot
row is active (+0.0008) **at one of those elements**, but scores it 0.60, not ~1.

Cause: the print-order sigmoid `sigmoid(rouf * (t_e - t_n))` gives a neighbor printed
slightly later about half credit. With `rouf` 100, one element layer of a [0, 1] field
over 120 elements is `rouf·Δt ≈ 0.8`, and the 177 solid elements in the stencil are all
0–0.041 later. A flat region and a layered one both collect about half the stencil
weight, so the optimizer lowers the hotspot by flattening `t`. The angular weight is not
the cause (κ = 0 scores 0.63). A sharper sigmoid only narrows the band: `rouf` 10⁴ gives
0.99, but a slightly tilted basin would pass again.

## Idea: material behind the front

A neighbor takes heat away only if it was deposited some time before and has cooled.
Count it only if it is at least δ earlier: `sigmoid(rouf * (t_e - t_n - δ))`.
- δ is a length behind the front, in metres (as `rmin_cond_m`), converted to time with a
  reference gradient: δ_t = δ · g_ref. It must stay well inside the stencil radius, or
  a layered field gets no credit.
- g_ref = the density-weighted **mean** `|grad t|` over the part, held out of the
  gradient (as the calibrations are). Not a nominal gradient: a serpentine and a straight
  bar would behave differently. Not the median: by the coarea formula the mean is
  (level-set length integrated over t) / area, so it cannot collapse, while a field of
  flat terraces has a median near 0, which would turn δ off (user, 2026-10-01).
- The HALF_STENCIL reference must use the element's own **direction** but g_ref as its
  **magnitude**. With the own magnitude, the reference shrinks with the numerator as the
  field flattens: offline, δ = 2 elements then moves the S offenders only 0.600 → 0.606.

Offline test on final `stto` designs (`/tmp/claude-1001/phase8/offset_sigmoid.tmp.py`),
max hotspot row with δ = s elements at the median gradient (the mean agrees within 1%):

| Design | Support | Now | s=0 | s=1 | s=2 | s=4 |
|---|---|---|---|---|---|---|
| S 0.3 | FAIL | +0.001 | +0.18 | +0.32 | +0.43 | +0.57 |
| S 0.5 | FAIL | +0.001 | +0.17 | +0.34 | +0.46 | +0.59 |
| D 0.3 | FAIL | 0.000 | +0.07 | +0.16 | +0.21 | +0.24 |
| C1 0.5 | pass | 0.000 | +0.004 | +0.011 | +0.027 | +0.08 |
| D 0.7 | pass | 0.000 | 0.000 | +0.020 | +0.045 | +0.09 |
| C1 0.3 | pass | 0.000 | +0.12 | +0.12 | +0.115 | +0.14 |

On every failing design the worst element is an offender. C1 0.3's rise is at a merge
point (a 2-element strut printed from both ends, `|grad t|` 0.19× the median), and comes
from the reference change, not from δ (s = 0 already +0.12). Merge points are wanted;
see step 1.

## Decisions (user, 2026-10-01)

- `stto` (t as a design variable). The virtual-heat variants are out of scope.
- No `min_gradient` floor: `|grad t|` cannot tell a minimum from a maximum, so a floor
  also forbids the merge maximum. The δ-hotspot is to forbid minima on its own (a
  minimum has nothing behind it; a merge has material behind it on all sides). If a
  floor is used again later, make it relative to the mean, as g_ref.
- `t` is closer to a process order than a strict time: the two legs should be free to
  merge where the design wants, which a uniform gradient can forbid. The hotspot still
  reads `t` as time. Uniformity is a light regularizer: set `uniformity_weight` so its
  gradient stays small against the compliance gradient (tentative: ≤ 0.2× the norm, at
  checkpoints). `w25` used 100.
- Implement δ and g_ref as explicit config options, default off, in `sttopt`.
- The plain-TO baseline turns the hotspot off with Tcr 2 (severity ≤ 1, so the row stays
  ≤ −0.5).
- Overnight run (2026-10-01/02): the agent decides s, Tcr, the schedule and any further
  option alone, records each decision and its evidence here, and keeps new behaviour
  behind options that default to off. Code goes to PR #190, never merged by the agent.
  The result: this plan, a dashboard under `plot/`, and one summary message.

## Steps

1. **Plain-TO baseline.** One `stto` run on S 0.3 with the `w25_stto_S_hot` config and
   every time-field term off (hotspot, uniformity, roughness, `min_gradient`, gravity).
   If `stto` has no explicit way to turn the hotspot off, ask. This shows roughly what the
   geometry "should" be (the user expects a skewed hourglass); thin features in it may
   overheat, so its compliance is a reference, not a target. **Done** (`s03_base_plainTO`,
   `plot/s03/baseline_vs_w25.png`): the skewed hourglass (bottom and top bars, two
   diagonals meeting at a node right of centre, short members to the right corners),
   compliance 109.6 binarized. `w25` has the same topology with a thicker top bar
   (110.3). The baseline's `t` is its initial layering, since nothing acts on it.
2. **κ = 0 offline check** of the C1 0.3 merge point: does the angular lobe, which credits
   only one side where the direction is undefined, cause the +0.12? **Yes** (2026-10-01).
   Severity at (12, 56): κ = 0 gives 0.554 at s = 0 and 0.508 at s = 2 (δ helps a merge,
   as it should); κ = 2.37 with the mean reference gives 0.673. Candidates for step 7: a
   κ continuation, or a lobe scale `g0` relative to g_ref, so that the lobe turns
   isotropic where `|grad t| ≪ g_ref`.
3. **Code** (new PR): the δ option, the g_ref reference, tests (δ = 0 with the own
   reference reproduces today's `K_est`; a flat field scores ~1 with δ > 0; a linear field
   at g_ref scores K = 1). Raise an error for δ > 0 with the own-magnitude reference.
4. **Uniformity weight** by the gradient-ratio rule, measured on `w25`-like checkpoints.
   **Done:** on `w25` (weight 100) the uniformity gradient is 0.07–0.23× the compliance
   gradient in `x`, but 1.0–1.6× the hotspot term (λ·∇g) in `t` once the hotspot acts
   (it 200 on). `t` is where it competes, so the weight is **20** (≈ 0.2× the hotspot
   term). Before the hotspot acts it is the only term on `t` at any weight.
5. **Tcr.** How Tcr maps from the old measure to the δ one is not known. Too high, the
   row is not active; too low, the problem becomes infeasible. Start high (e.g. 0.9) and
   lower it step by step. Expected order of effects: first the time-field gradients align
   with the bar directions, then bars that were too thin get thicker. Find the range where
   the row is active and the design still looks right.
6. **Runs.** S 0.3 at s = 1, 2, 4 elements, against a matched control (the same config,
   δ = 0). Then the chosen s on C1 0.3, C1 0.5 and D 0.7, to see that designs that pass
   now are not damaged.
7. **Schedule tuning.** The `stto` schedule was tuned for the old measure, so retune it
   with δ on: the Tcr ramp, β_d and `penal`, and a continuation of `hotspot_kappa` (the
   angular weight; κ = 0 is radial). The user's earlier idea: start mostly radial, while
   the topology and so the print directions are still undefined, and sharpen κ later.
   Same rules as Phase 8: one schedule across cells, judged against matched controls.

## Results (overnight 2026-10-01/02)

Runs in `/tmp/claude-1001/s03/work/output/`; code snapshot `snap_0409284`; all S 0.3,
`w25` config with uniformity 20 and no floor. "Interior unsup." = the support check.

| Run | δ | Tcr | Compliance (× plain TO) | Interior unsup. | Hotspot row | Top bar |
|---|---|---|---|---|---|---|
| `b1_ctrl_T06` | 0, own ref | 1 → 0.6 over 150–400 | 109.6 (1.00) | **17** | −0.001 | several basins, no merge |
| `b1_s2_T07` | 2 el. | 1 → 0.7 over 150–400 | 109.7 (1.00) | 0 | +0.001 | basins at 250–350, gone by 450; one merge at col ~48 |
| `b1_s2_T06` | 2 el. | 1 → 0.6 over 150–400 | 111.3 (1.02) | 0 | **+0.12** | basins until it 600, cleared only in the last 200 |
| `b1_s2_scan` | 2 el. | 1 → 0.5 over 150–700 | 109.7 (1.00) | 0 | −0.0005 | basins at 250–450, gone by 650; one merge at col ~55 |
| `b2_ctrl_scan` | 0, own ref | scan | 109.6 (1.00) | **22** (39 saddles) | +0.013 | many basins, max at the far right |
| `b2_s1_scan` | 1 el. | scan | 110.0 (1.00) | 0 (2 saddles) | −0.0004 | basins until 450, gone by 550; one merge at col 68 |
| `b2_s4_scan` | 4 el. | scan | 111.9 (1.02) | 3 (13 saddles) | **+0.91** | severity stuck at 0.955 from it 200 (underside of the top bar mid-span); design distorted |

- δ is what removes the basins: both matched controls (no δ) end with 17–22 interior
  unsupported elements, the δ runs at s = 1 and 2 with 0. The slow ramp alone does not.
- s = 4 elements (a third of the 12-element stencil radius) leaves too little of the
  stencil behind the front to credit; s = 1 and 2 both work. **Chosen: s = 2** (2 mm),
  the clearer separation in the offline test.
- The speed of the Tcr ramp matters more than its final value: the slow scan meets 0.5,
  the faster ramp to 0.6 by 400 ends 0.12 over it. Basins form while the hotspot
  multiplier sits at `mma_c` and the severity lags Tcr; they clear once it catches up.
- The scan schedule (Tcr 1 → 0.5 over 150–700) is the base for the next batches.

**Other cells**, s = 2 against a matched control (same scan schedule, no δ):

| Cell | δ: compliance / interior unsup. / saddles / row | control: same | `w25` stto |
|---|---|---|---|
| C1 0.3 | 267.4 / 1 / 5 / +0.38 | 266.6 / 3 / 8 / +0.46 | 283.8 / 0 / 0 (at its own Tcr 0.6) |
| C1 0.5 | 175.2 / 0 / 10 / 0.000 | 174.9 / 2 / 24 / +0.044 | 174.1 / 0 / 0 |
| D 0.7 | 2.13 / 0 / 0 / 0.000 | 2.13 / 0 / 0 / 0.000 | 2.22 / 0 / 0 |
| S 0.5 | 74.8 / 0 / 2 / 0.000 | (queued) | 75.1 / **5** / 6 |
| D 0.3 (as optimized) | **12.06** / 2 / 5 / **+0.90** | (queued) | 8.31 / 4 / 3 |

- C1 0.3 cannot reach Tcr 0.5 under either measure: the severity levels off at 0.70
  (δ) and 0.73 (control) from iteration ~500, with the multiplier at `mma_c`. Same
  topology in both; δ does no damage. A slow ramp to 0.75 is queued to see the design
  under a constraint it can meet.
- C1 0.5: same topology; δ meets 0.5 where the control misses. Below Tcr ~0.6 both
  runs bend the time contours on the left and add a thin member at the bottom right
  late (it 650–800): a low-Tcr effect, not a δ one.
- D 0.7: δ and its control end identical, and both simpler than `w25` (one lower hole,
  no web of thin members, no isolated hole at the top right) at 4% less compliance.
  The difference from `w25` is the schedule (slow ramp, uniformity 20), not δ.
- S 0.5: δ passes where `w25` fails support; `w25` also has a small isolated hole in
  its top bar, which the δ run does not.
- D 0.3 fails with δ: from it 200 (when the hotspot acts) the severity is stuck at
  0.95–0.97 on grey stubs hanging under the long top bar at cols ~20–25 (`w25` grows a
  solid bump at the same place). The stubs print before the bar above them, so nothing
  is behind them, and `x**0.05` gives a grey element almost full severity; the
  projection keeps them grey (grey fraction 0.029 from it 400) and they are never
  removed. A protrusion that is not optimized away.

## Judging a run

Metrics are symptoms; look at the checkpoints (every 50 iterations: density, time field,
severity map). A good run:
- has a topology near the plain-TO baseline, without the thin features that would
  overheat;
- leaves no stubs or protrusions, and no frozen grey web;
- builds `t` bottom-up, with one clean maximum where the fronts merge: on the top bar or
  at a corner, both plausible;
- never forms a local minimum or flat basin on the top bar. Once one forms the optimizer
  can hardly push it out (that is how the Variant 1 designs failed), so note the first
  checkpoint where one appears, even if it later shrinks.

Compliance within ~1.2× the baseline is the expectation for a good optimizer, not a gate.
