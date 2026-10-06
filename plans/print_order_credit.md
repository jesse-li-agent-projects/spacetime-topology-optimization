# Print-order credit: no credit for a neighbor printed at the same time

Follows `s03_material_behind_front.md` (PR #190). Ask the user before any design
decision this plan does not make.

## Problem

The hotspot's print-order sigmoid `sigmoid(rouf * (t_e - t_n))` gives a neighbor printed
at the same time half credit. A region printed all at once then scores like a layered
one, and the optimizer lowers the hotspot by flattening `t` (S 0.3's top-bar basin).

PR #190 counted a neighbor only once it was δ behind the front, and put δ into the
HALF_STENCIL reference too. Decisions (user, 2026-10-06):
- The normalization is a property of the stencil, not of the delay: the reference stays
  `master`'s directional denominator, delay-free. The delay is a regularizer against a
  flat `t`, not physics. Tcr need not match `master`'s; sensible geometry and time
  fields matter.
- That drops `ReferenceGradient`/PART_MEAN, the front offset and the denominator
  `detach`: with a delay-free reference, OWN is not cancelled by a flat region.
- Try two numerator credits, both behind one enum, and tune both.

## The two credits

Only the numerator's pair weight changes; `directed_denominator` is untouched.
- **SIGMOID with a tie credit** `p`: `sigmoid(rouf * dt + logit(p))`. `p = 0.5` is
  today's behaviour. A tie always gets `p`; the shift `-logit(p) / rouf` vanishes as a
  `rouf` continuation sharpens the sigmoid, and layered regions go back to `K ~ 1`.
- **COOLING** with a cooling time `tau` (in `t`): `1 - exp(-softplus(k dt / tau) / k)`,
  `k = 8`. No credit for a later neighbor, about `0.08` for a tie, saturating after a few
  `tau`. `tau` is schedulable.

Offline re-score of finished designs (delay-free denominator; max severity on solid
elements, elements over 0.6 in brackets):

| Credit | S 0.3 `w25` offenders | S 0.3 δ (merge) | C1 0.5 heat |
|---|---|---|---|
| sigmoid, rouf 100 | 0.594 | 0.573 (0) | 0.600 (0) |
| sigmoid, rouf 10⁴ | 0.989 | 0.572 (0) | 0.667 (9) |
| tie p 0.25, rouf 100 | 0.814 | 0.652 (563) | 0.683 (442) |
| cooling τ 0.0025 | 0.996 | 0.556 (0) | 0.578 (0) |

A flat region scores `K ~ 2 x` the tie credit, so it fails where `1 - 2p > Tcr`.

## Experiments (user, 2026-10-06)

Use the `long-running-runs` skill. GPU: one job at a time. CPU: 3 runs x 4 threads.
Launch from a snapshot with `PYTHONSAFEPATH=1`, and log `sttopt.__file__`. The base
schedule is `S03_stto_delta`'s (800 iterations, uniformity 20, no gradient floor, Tcr
1 -> target over 150-700), with δ off.

**Cells** (1 mm elements): S = `corner_loads` 120x120, D = `edge_traction` 120x90, C1 =
`cantilever` 180x60.

| Cell | Base | Tcr (control) |
|---|---|---|
| S 0.3 | bottom_edge | 0.6 |
| S 0.5 | bottom_edge | 0.6 |
| D 0.3 | bottom_edge | 0.8 |
| D 0.7 | edge | 0.8 |
| C1 0.3 | edge | 0.6 |
| C1 0.5 | edge | 0.8 |

**Dev set:** S 0.3 and D 0.3.
1. Stage 1, at Tcr 0.75: cooling τ in {0.0025, 0.005, 0.01}; `rouf` continuation 100 ->
   {1e3, 1e4} (log, iterations 300-700) x tie credit p in {0.5, 0.25}. 14 runs.
2. Stage 2: the best setting of each credit x Tcr in {0.6, 0.7, 0.8, 0.9}. 16 runs. Each
   credit takes the strictest Tcr that passes on both dev cells; if none passes D 0.3,
   pick on S 0.3 and say so.
3. Full set: the 6 cells x {cooling, `rouf` continuation} at their chosen Tcr, plus the
   `master` control (cell Tcr) and plain TO (hotspot off, Tcr 2). The controls do not
   depend on tuning, so they run on the CPU from the start.

"Best" in stage 1: passes the gates on both dev cells, then the lowest compliance ratio,
then the fewest saddles and local minima.

**Gates** (a run passes all):
- Hotspot row <= 0.01 at the end, at the run's own Tcr.
- At most 5 interior unsupported elements (support check); the hotspot should catch
  the rest.
- Start check passes.
- Compliance (binarized) <= 1.15x the cell's plain-TO run.

Report saddle and local-minimum counts without gating. The user judges the shapes.

**Output:** runs in `output/poc_<stage>_<cell>_<setting>/`; a dashboard under
`plot/print_order_credit/` in the format of `plot/s03/dashboard_CELLS.html`; results and
decisions recorded here.
