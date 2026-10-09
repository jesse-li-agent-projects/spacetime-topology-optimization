# The default continuation schedule

`default.json` holds the numbers; this is why they sit where they do, for whoever tunes
them next. Iterations are for `nloop` 600. Settled 2026-10-08 by the tuning in
`plans/print_order_credit.md` (tuning P and its follow-ups).

## Phases

| Iterations | Phase | What moves |
|---|---|---|
| 0-120 | layout forms | `beta_d` 1 -> 4 (log); `beta_t` steps 10 -> 50 by 240; `Tcr` 1 -> 0.95 by 50, then toward the target; `penal`, `q` held at 2; `r` 0.5; `hotspot_weight` 50 acts from the start |
| 120-350 | grey hold | `beta_d` held at 4; `penal`, `q` 2 -> 3 over 150-350; `Tcr` reaches the target at 350 |
| 350-450 | binarize | `beta_d` 4 -> 32 (log); `r` 0.5 -> 0.05; cooling sharpness 1 -> 8 (log); `penal`, `q` continue toward 5 |
| 450-550 | settle | `penal`, `q` reach 5 at 550; `tmove` 0.04 -> 0.002 over 500-550 (log); the roughness weight reaches its floor at 550 |
| 550-600 | converge | nothing |

Off: gravity stages (`Theta` 0, `nStage` 0), the tool radius, `min_gradient_fraction`,
`gradient_smoothness_m`. They were off in every tuning run since the print-order credit
study and are not validated on with this schedule.

## Why

- **`r` 0.5 while grey, then 0.05.** Severity is `(1 - K_est) * xPhys**r`. At `r` 0.05
  near-void counts almost fully (0.02 -> 0.82), so the constraint shapes grey void and
  C1 0.5 compliance blows up over 120-350 (1.37 x plain TO at 350, against 1.15 at
  `r` >= 0.25). Kept large to the end, `r` lets grey material hide its own severity: up
  to 92% of the hottest elements after binarizing were grey, and the binarized row
  reached +0.86 at `r` 1. Ramping to 0.05 during binarization removes both.
- **`beta_d` ends at 32, not 128.** At 128 the projection gradient vanishes near the
  threshold, so islands and grey stubs freeze (an 11-element island persisted 200
  iterations). At 32 the geometry still moves after 450. The edges stay 1-3% grey, so
  gate on the binarized hotspot row, not the one as optimized.
- **`penal` and `q` go on to 5 by 550**, after binarization; `q` follows `penal`. Evidence
  is mixed (it fixed one binarized hotspot row, +0.21 -> +0.02, but cleared grey bars in
  only 1 of 4 runs); kept for the best chance, since a binary design is unaffected.
- **`Tcr` 1 -> 0.95 by 50 so the hotspot binds while the design is grey**, then to the
  target by 350. A start at 120 left the layout to form unconstrained. The two ramps
  have nearly the same slope at a 0.6 target: a candidate to merge into one.
- **`tmove` 0.04, then a taper.** At 0.02 S 0.3 stalled for ~200 iterations with its
  hotspot site flipping every iteration. Any level 0.03-0.06 passed; above ~0.04 the
  limit rarely binds. Shrinking it mid-run (one log decay) cost C1 compliance. The taper
  starts after binarization and damps late oscillation (late excursions no higher in
  12/12 pairs, lower in 11).
- **`calibration_rate` 0.1.** Re-measuring a smooth maximum's calibration every
  iteration made the row MMA sees jump: on grey designs the hotspot row and its
  multiplier fell into a 2-cycle (row sign flips in 198/200 iterations, multiplier
  switching between `mma_c` and ~30), which collapsed MMA's asymptotes and froze the
  design. Damped, the cycle loses ~95% of its alternating part (PR #202).
- **`hotspot_weight` 50** (2x the measured 1x of 25: the median ratio of the
  regularizers' to the hotspot's time-field gradient over 0-150). The time field then
  lowers the hotspot before `Tcr` binds: the row turns active at ~180 instead of ~60,
  the multiplier sits at `mma_c` far less (C1 0.3: 44 -> 8 iterations), S 0.3 ends ~20%
  under `Tcr`, and compliance moves under 0.01. 50 sits a factor 2 inside the passing
  range 12.5-100. Cost: more saddles in the time field (C1 0.5: 1 -> 8).
- **Uniformity weight 20 and the cooling credit** come from the print-order credit
  study; dropping the uniformity term wrecks the time field (binarized row +1.5).

## Validated on

The three dev cells, 600 iterations, one run per setting: S 0.3 (`Tcr` 0.6,
`bottom_edge`, corner loads, 120x120), C1 0.3 (`Tcr` 0.6) and C1 0.5 (`Tcr` 0.4), 180x60.
All gates pass, including the binarized hotspot row: compliance 1.007, 1.117 and 1.038 x
plain TO. Not yet run: the default `Tcr` 0.8, the validation matrix, replicates.

## When tuning

- **Read the MMA diagnostics** (`asy_floor_*`, `subsolv_cap_hits` in `iterations.jsonl`).
  A large share of variables at the asymptote floor means MMA has frozen them, whatever
  their gradient; it appeared on S 0.3 during the grey hold and on grey bars at the end.
- **Watch the multipliers.** A row at `mma_c` for long means MMA is violating it; with a
  2-cycle around it, suspect the calibration or the trust region before the schedule.
- **Compare against a matched control**, and judge on several cells, not replicates of
  one: C1 0.5 sits near several topologies and changes members on small schedule changes.
- **Hotspot limits**: a straight vertical free wall scores ~0.43, so `Tcr` below that
  needs shaped edges everywhere.
