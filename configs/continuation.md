# The default continuation schedule

`default.json` holds the numbers; this is why they sit where they do, for whoever tunes
them next. Iterations are for `nloop` 800.

## Phases

Change points fall every 100 iterations from 150, so each phase does one job.

| Iterations | Phase | What moves |
|---|---|---|
| 0-150 | topology forms | `beta_d` 1 -> 8, `beta_t` 10 -> 35; `penal` held at 2; `Tcr` held at 1 (constraint inactive) |
| 150-250 | hotspot | `Tcr` 1 -> target; `beta_d` 8 -> 32; `beta_t` reaches 50 at 240 |
| 250-350 | freeze | `beta_d` 32 -> 128; `penal` reaches 3 (ramp from 150) |
| 350-450 | tool radius | `tool_radius_m` 0 -> 2.5 mm |
| 450-550 | settle | `tmove` 0.01 -> 0.001; `curvature_beta` 10 -> 100; roughness weight reaches its floor |
| 550-800 | converge | nothing |

The gradient floor and Hessian bound (`min_gradient_fraction`, `gradient_smoothness_m`)
are constant from iteration 0.

## Why

- **`Tcr` starts at 1.** The severity never exceeds 1, so a ramp from above 1 does nothing
  until it crosses 1 and then acts as a step. From 5.0 over 150-250 it fell 0.97 -> 0.4
  in about 10 iterations, leaving the hotspot ~100 iterations before the freeze.
- **The hotspot tightens over 150-250**: after a rough layout exists, before the freeze.
  An earlier onset costs compliance (about 3% from iteration 1 at `Tcr` 0.8, 7-8% for
  a 50-250 ramp at 0.4); ending after ~250 leaves too few mobile iterations and ends
  infeasible.
- **The gradient constraints are on from the start.** Ramped in later (over 150-350,
  250-400, 400-550), pits and saddles form first and cannot be removed once the topology
  freezes. Beta 20: at 50 the floor's gradient concentrates and wins over the hotspot.
- **`penal` is 2 until the hotspot is placed.** A softer SIMP penalty keeps the topology
  movable while the hotspot and the floor both act. 1 -> 3 is more reliable still but
  costs too much compliance, especially in shorter runs.
- **The tool radius waits for the freeze.** Iso-line curvature means little on a grey
  design; the topology is essentially fixed by 350 (under 0.4% of elements change later).
- **Settling sharpens together.** `curvature_beta` rises while `tmove` shrinks, and the
  roughness weight reaches its floor at the same time.

Aligning the change points was measured as neutral (within run-to-run noise).

## Validated on

MBB half-beam, `volfrac` 0.5, 1 mm elements, 800 iterations: 120x90 and 180x60, all four
`print_base` values at `Tcr` 0.4, spot checks at 0.8 and 0.6. It fails 180x60 at `Tcr`
0.4 for `bottom_edge` and `corner`; `bottom_edge` fails there even without the gradient
constraints. Runs up to PR #169 used the pre-#170 `opposite_corner` base (the whole left
column), so a true corner base is untested. Results: PR #169.

## When tuning

- **Replicate marginal cases.** 180x60 `opposite_corner` at `Tcr` 0.4 flipped between
  pass and fail on a device change alone (CPU vs GPU) before the `penal` ramp. Judge a
  change on several runs, CPU and GPU, not one.
- **Compare against a matched control**, the same schedule without the change.
- **Watch the multipliers.** A hotspot and floor constraint both at `mma_c` means MMA is
  violating both; that is a schedule conflict, not an infeasible problem.
- **Shorter runs**: scaling every change point by `nloop / 800` keeps `Tcr` 0.8 valid
  down to 400, but 180x60 at `Tcr` 0.4 rises above compliance 200 below 800.
- **Hotspot limits**: a straight vertical free wall scores ~0.43, so `Tcr` below that
  needs shaped edges everywhere. The lowest reached on 180x60 is ~0.40.
