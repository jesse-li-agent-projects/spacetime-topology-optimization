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

**D cells dropped from future cell sets (user, 2026-10-07).** The D (`edge_traction`)
topologies are not interesting: plain TO gives a few straight, mostly horizontal bars
from the supports to the loaded edge (D 0.3), or two thick bars with a slot (D 0.7), and
already prints them in simple layers. On D 0.7 every variant ends within 0.3% of plain
TO. They test little that S and C1 do not. D 0.3's failures in this sweep came from a
notch in the loaded edge, not from the print order. The runs below still include D.

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

## Overnight log (2026-10-07)

- **01:50, agent decision: cooling onset sharpness.** At the fixed onset `k = 8`, the
  stage-1 cooling runs stalled: from iteration 200 the max severity stayed at 0.98
  (S 0.3, the top-bar basin, row 6) and 0.95 (D 0.3) while Tcr ramped to 0.75, with the
  hotspot multiplier at `mma_c`. The `rouf` continuation runs tracked Tcr down. Cause: a
  neighbor printed `dt` later gets a pull toward being earlier of about
  `exp(-k dt / tau)`-small; at `dt = -0.01` it is ~600x the source sigmoid's at rouf
  100. The design is not the problem: the δ-study S 0.3 design scores 0.556 under the
  same credit. Fix (commit `0bed51a`, default unchanged): `hotspot_cooling_sharpness`,
  schedulable. Added stage-1 settings `coolk0.01` and `coolk0.02`: k 1 -> 8 (log,
  iterations 300-700), the cooling analog of the `rouf` continuation. The driver was
  restarted on snapshot `0bed51a`; interrupted runs resumed from their checkpoints.
- **Compliance gate.** D 0.3 plain TO is 4.56. The stage-1 D 0.3 runs measure 1.03-1.29x
  it, so the 1.15x gate binds there but does not fail every run (the δ study's 8.3-12
  were a different schedule).
- **01:46-02:48, operations.** Stopping the first driver's shell left its Python process
  alive; it kept starting runs, so two drivers ran 8 jobs at once for an hour. Fixed by
  killing it; its finished runs (snapshot `72b5842`, identical code for their configs)
  are kept, its interrupted ones resumed. Run times in that window are inflated.
- **04:05, stage 1 and a selection fix.** `coolk0.01` passes every gate on S 0.3 (row
  -0.000, 0 unsupported, compliance 109.55 = plain TO, 1 saddle, 0 minima); no setting
  passes D 0.3. The driver first picked `rc1000p0.25`, because with nothing passing on
  both cells my sort key fell through to compliance. `rc1000p0.5` is closer to passing
  (row <= 0 on both cells; fails only the unsupported gate, 8 and 11; 20-22 saddles
  against 60-137), and the plan ranks gates first, so the key now counts gates passed
  before compliance. Restarted before any `rc1000p0.25` stage-2 run began. Stage 2:
  `coolk0.01` and `rc1000p0.5`.
- **D 0.3 broken binarization.** `coolk0.01`, `coolk0.02` and `rc10000p0.25` reach an
  as-optimized compliance of 4.67-4.71 (1.02x plain TO) but 161 300 binarized: elements
  (41-42, 119) on the traction edge sit at density 0.27-0.45, so binarized they leave
  part of the load on void. The gate catches a real defect (a notch in the loaded edge).
- **07:10, stage 2.** Passing Tcr per dev cell (all gates):

  | Setting | S 0.3 passes at | D 0.3 passes at | Chosen |
  |---|---|---|---|
  | `coolk0.01` | 0.6, 0.75 | 0.6, 0.7 | Tcr 0.6 |
  | `rc1000p0.5` | 0.9 | 0.7, 0.9 | Tcr 0.9 |

  Pass/fail is not monotone in Tcr: a stricter Tcr sometimes ends cleaner. With one run
  per point, a single pass is weak evidence; failures are mostly the unsupported gate
  (6-34 elements) with a matching rise in saddles and minima.

## Results (2026-10-07, all runs finished 08:32)

Full set, each credit at its stage-2 Tcr; the `master` control at the cell's Tcr. Pass =
all four gates. Compliance is binarized, as a ratio to the cell's plain-TO run.

| Cell | `master` control | `coolk0.01`, Tcr 0.6 | `rc1000p0.5`, Tcr 0.9 |
|---|---|---|---|
| S 0.3 | FAIL: 16 unsupported, 25 saddles | pass (1.000, 3 saddles) | pass (1.000, 2 unsupported, 8 minima) |
| S 0.5 | FAIL: 6 unsupported | pass (1.001, 1 saddle) | pass (1.000, 4 minima) |
| D 0.3 | FAIL: binarized load edge broken | pass (1.026, 2 saddles) | pass (1.003, 5 minima) |
| D 0.7 | pass (1.002) | pass (1.003) | pass (1.001) |
| C1 0.3 | FAIL: row +0.22 | FAIL: row +0.24 | pass (1.000) |
| C1 0.5 | pass (1.000) | pass (1.005, 4 saddles) | pass (1.000) |

- The cooling continuation passes 5 of 6 cells at the strictest Tcr of the night (0.6),
  with clean time fields (0 unsupported, 0-1 minima everywhere). On S 0.3 it prints the
  top bar from both ends to a merge near column 75, and the struts along their length;
  max severity 0.597.
- The `rouf` continuation passes 6 of 6, but at Tcr 0.9, a loose limit; on C1 0.3 its
  design equals plain TO. Its time fields keep a few minima (4-8 on S and D).
- C1 0.3 levels off above Tcr 0.6 under both `master` and cooling, as in the δ study.
- Caveats: one run per point, and stage 2 showed pass/fail is not monotone in Tcr, so a
  single cell's verdict is weak evidence. The cooling setting was added overnight (see
  the log above) and chosen on the same dev set.

Dashboards: `plot/print_order_credit/dashboard_full.html`, `dashboard_stage1.html`,
`dashboard_stage2.html`. Runs: `output/poc_*`; driver, configs, logs and
`decisions.json` in `output/poc/`.

## Next: a schedule experiment (agreed with the user, 2026-10-07)

Evidence from the overnight runs:
- **The time-field step limit binds.** On S 0.3 (`coolk0.01`, Tcr 0.6) the largest time step
  sits at `tmove` from iteration 200 to 800. `tmove` tapers 0.01 -> 0.001 over 450-550,
  while Tcr still tightens until 700 (the δ study moved the Tcr ramp end from 250 to 700
  but kept the taper). The top bar's minima (27 at 450) clear only by 650, at 0.001 per
  iteration. The density `move` is a flat 0.01 in these runs. A density taper was never
  shown to help: on 2026-09-30 the "flat move fixes D" result was confounded by the
  uniformity term, and the taper came back only as the default.
- **Tcr acts after the topology is fixed.** On C1 0.3, β_d is 8 at iteration 150 and 64
  at 300, while Tcr is still ~0.9 at 300. The severity levels off at 0.73-0.75 at a thin
  member near the loaded tip, as in the δ study (~0.70). Plain TO on C1 0.3 gives
  members too thin to print without overheating, so the hotspot must act while they
  can still thicken (user: by about iteration 150).
- **The time field must keep up.** If Tcr binds before the time field is aligned with
  the forming members, the result is too conservative; larger time steps help.
- Candidate P (2026-10-01, virtual-heat runs only): sketch, then hold β_d (and penal)
  while Tcr ramps, then binarize; P4 held β_d at 4 over 120-300 (user: "beta 4 feels like
  it has the best shot", on C1 0.3).

Design:
- Cells: S 0.3 (Tcr 0.6) and C1 0.3 (Tcr 0.6 and 0.75, the δ-study floor). Credit fixed:
  cooling, τ 0.01, sharpness k 1 -> 8 (place the k ramp after the β_d hold, with the
  binarization).
- (a) the δ-study schedule with a flat `tmove` (0.01).
- (b) P for `stto`: β_d 1 -> 4 by 120, held to 350, -> 128 by 450; Tcr 1 until 120, ->
  target by 350; flat `tmove` 0.02; penal as before (2 -> 3 over 150-350).
- (c) as (b), with penal held at 2 until 350, -> 3 by 450.
- References: the overnight `poc_S03_coolk0.01_T0.6`, `poc_C103_coolk0.01_T0.6`, and the
  plain-TO runs. 9 runs, about 2-3 h on GPU + 3 CPU slots.
- **Measure whether the time field responds to Tcr**: per iteration, the largest and
  mean time step against `tmove` (fraction of steps at the limit), |Δt| on solid
  elements between snapshots, the hotspot multiplier, and the lag between Tcr and the
  max severity. Save snapshots every 5 iterations.
- Dashboard: as `dashboard_full.html`, plus charts for `tmove` with the largest time
  step, |Δt| on solid elements, and the hotspot multiplier.

Reference for "Tcr changes the topology": `output/tcr_sweep/` (2026-09-26, commit
`9610316`): the 180x60 cantilever (then "MBB", now C1) at volfrac 0.5, print base
`opposite_corner`, Tcr 5 -> {1.0, 0.8, 0.6, 0.4} over 150-250. Topology overlap (IoU)
with the Tcr 1.0 design: 0.976, 0.949, 0.873; compliance +0.07%, +0.6%, +5.0%. Plots in
`plot/tcr_sweep/`.

## Results: schedule experiment (2026-10-07, runs 12:45-13:48)

Credit in all runs: cooling, τ 0.01, k 1 -> 8. Compliance is binarized, as a ratio to the
cell's plain-TO run; a failed run shows its hotspot row.

| Cell, Tcr | Overnight ref | (a) flat `tmove` | (b) P | (c) P, penal held |
|---|---|---|---|---|
| S 0.3, 0.6 | pass, 1.000 | pass, 1.000 | pass, 1.005 | pass, 1.006 |
| C1 0.3, 0.6 | FAIL, +0.243 | FAIL, +0.244 | pass, 1.087 | pass, 1.089 |
| C1 0.3, 0.75 | - | pass, 1.001 | pass, 1.028 | pass, 1.012 |

Fraction of iterations with the largest time step at `tmove`, and the iteration from
which the hotspot row stays <= 0.01:

| Run (C1 0.3, Tcr 0.6) | 120-350 | 350-500 | 500-700 | 700-800 | Row <= 0.01 from |
|---|---|---|---|---|---|
| Overnight ref | 0.05 | 0.68 | 1.00 | 0.77 | never |
| (a) | 0.11 | 0.63 | 1.00 | 1.00 | never |
| (b) | 0.54 | 0.60 | 0.12 | 0.04 | 442 |
| (c) | 0.57 | 0.61 | 0.14 | 0.03 | 447 |

- **P is the first schedule to pass C1 0.3 at Tcr 0.6.** The thin member at the loaded
  tip (1-2 elements in (a)) is 3-4 elements thick in (c), with material taken from the
  flanges near the support (+8.7-8.9% compliance).
- **A larger time step alone does not help once the topology is fixed:** in (a) the step
  is at `tmove` in every iteration from 500 on, and the row never clears.
- **Holding penal at 2 makes no clear difference** ((b) vs (c)). What matters is ramping
  Tcr under the β_d hold.
- **No pressure before ~150.** The max severity sits at 0.94-0.96 while the design is
  grey, so with Tcr 1 until 120 the hotspot multiplier is ~0 (2e-6) until Tcr falls below
  that, at ~150. At the end of the Tcr ramp (350) the row is still +0.07 (lag); it clears
  by ~445.
- **Flat `tmove` leaves the end unsettled on S 0.3:** the row crosses 0.01 until ~790 in
  (a)-(c) (the taper reference: 565), and the final margins are small (-0.004, -0.000,
  +0.001).
- Warnings: 3-8 MMA subsolver "iteration cap" per run (existing behaviour); print-support
  warnings on C1 0.6 (b), (c) are the 2 and 1 unsupported elements in the table.

Dashboard: `plot/sched/dashboard.html` (frames every 5 iterations; serve `plot/` on
localhost, see `agent_docs/dashboards.md`). Runs: `output/sched_*`; driver, configs, logs
in `output/sched/`.

User's notes (2026-10-07):
- Visually, (b) is the most promising, but needs further tuning.
- **Necking** must be avoided: a member narrows in the middle where the hotspot
  constraint discourages material. Visible, weakly, in (b)'s C1 0.3 Tcr 0.6 design: the
  diagonal about 1/3 from the right, bottom half (from ~(row 28, col 115) to ~(55, 135)),
  which sits at the severity limit along its length. It is an intermediate state: with
  more material or a looser Tcr it would not neck; with a tighter Tcr the member would
  eventually break (no material can keep it from overheating). Either way, the aim is
  that such a member does not form in the first place. These runs show it weakly because
  volfrac 0.3 leaves little material to give.
- By iteration ~80 some members have started to coalesce, so the hotspot should already
  apply pressure by then.
- Hunch: let the hotspot conductivity exponent `q` follow `penal` (the SIMP exponent)
  instead of a fixed 3.
- Replace C1 0.3 Tcr 0.75 with C1 0.5 Tcr 0.4. (Overnight, the cooling credit passed C1
  0.5 at Tcr 0.6 at 1.005 × plain TO, so Tcr 0.6 does not change its topology.)

## Next: tuning P (proposal, 2026-10-07, not yet agreed)

1. **Code: make `q` schedulable** (`Scheduled`, like `penal`), so "q follows penal" is a
   config choice (give `q` the penal schedule), not a special flag. `field_terms`
   evaluates it at the iteration; `seqopt` keeps a fixed value.
2. **A necking check**, to tune against instead of by eye: on the binarized design,
   the local thickness (twice the distance transform, sampled on the skeleton) along
   each member; flag a member whose thickness drops below a fraction of its own median
   between two joints. Report it per run and in the dashboard.
3. **Factors** (2 x 2, all from (b)):
   - `q`: 3 vs. the penal schedule. With `q` = 2 while grey, grey material counts more
     (0.3^2 = 0.09 vs 0.3^3 = 0.027 of a solid neighbor), so the hotspot reads a grey
     design less pessimistically and can act early without spurious pressure.
   - Tcr ramp start: 120 (as (b)) vs. ~50, starting at ~0.95 (the grey-phase severity
     level, so that it binds at once) and reaching the target by 350, so the hotspot
     acts by ~80.
4. **Fixed in all runs:** the `tmove` taper back after binarization (0.02 -> 0.002 over
   450-550), against the unsettled end on S 0.3.
5. **Cells:** S 0.3 Tcr 0.6, C1 0.3 Tcr 0.6, C1 0.5 Tcr 0.4. 12 runs (~3 h); plain-TO
   references exist (`poc_*_plainTO`).
