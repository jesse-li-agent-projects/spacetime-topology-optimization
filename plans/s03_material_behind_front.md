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

## Steps

1. **Plain-TO baseline.** One `stto` run on S 0.3 with the `w25_stto_S_hot` config and
   every time-field term off (hotspot, uniformity, roughness, `min_gradient`, gravity).
   If `stto` has no explicit way to turn the hotspot off, ask. This shows roughly what the
   geometry "should" be (the user expects a skewed hourglass); thin features in it may
   overheat, so its compliance is a reference, not a target.
2. **κ = 0 offline check** of the C1 0.3 merge point: does the angular lobe, which credits
   only one side where the direction is undefined, cause the +0.12?
3. **Code** (new PR): the δ option, the g_ref reference, tests (δ = 0 with the own
   reference reproduces today's `K_est`; a flat field scores ~1 with δ > 0; a linear field
   at g_ref scores K = 1). Raise an error for δ > 0 with the own-magnitude reference.
4. **Uniformity weight** by the gradient-ratio rule, measured on `w25`-like checkpoints.
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
