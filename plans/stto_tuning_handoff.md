# Handoff: `stto` schedule tuning and optimization dynamics (as of 2026-10-14)

A quick start for the next agent. No session ran between 2026-10-10 02:20 and this note.
The history of each decision is in the files linked below, not here.
Dashboards (`plot/<study>/dashboard.html`) must be served, not opened from disk; see
`agent_docs/dashboards.md`.

## Where things stand

- **The schedule is settled and merged** (stack #206, last layer #205): `configs/default.json`,
  with the phases and the reason for each setting in `configs/continuation.md`. The
  process and the user's decisions are in `plot/pres_tuning/tuning_review.md` (untracked).
- **Validation matrix ran** (2026-10-10, one run per cell, 600 iterations, code snapshot
  `62dfdce` = `master` + #208). 16 of 18 cells pass every gate. Apart from C1 vf 0.3
  (1.093x), compliance is within 0.6% of plain TO. Outputs `output/val_*`, driver and
  logs in `output/val/`, dashboard `plot/val/dashboard.html`. **The user has not reviewed
  it yet.**

  | Failing cell | Tcr | C / TO | Binarized hotspot row | Interior unsupported |
  |---|---|---|---|---|
  | S vf 0.5, `bottom_edge` | 0.8 | 1.002 | +0.15 | 32 |
  | L middle (`l_bracket_middle`), `right_edge` | 0.6 | 1.005 | +0.61 | 32 |

  Both break when binarization starts (~350). The hotspot multiplier then sits at
  `mma_c`, and the hot spot is on a region printed all at once (a flat time basin).
  This is the same failure as S 0.3 under the simplification variants V2/V4 and the
  early-sharpness k runs (below).
- **Caveat on the matrix:** in many passing cells the hotspot row ends far from active
  (-0.13 to -0.63). Plain TO already satisfies them, so they test little. The cells
  where the row is active (about 0) are the informative ones: C1 0.3 and 0.5 at 0.6, C2 0.5,
  and L corner at 0.6.

## The open thread: grey "nucleation" under flat time fields

The user's view (2026-10-09) is that this "feels like a formulation issue". Severity is
`(1 - K_est) * x^r`, and the conductivity credit weights neighbours by `x^q`. With `r` 0.5
during the grey hold, light material can sit anywhere at little cost. The proxy credits
any earlier, nearby material without asking whether that material is printable. On a
flat `t` (the S top bar), this rewards growing "artificial support" under the bar.

- Derived condition: the hotspot gradient on void should push density down. For power
  laws this reduces to the ratio `q / r` (separate f = severity factor from g =
  conductivity weight: g ~ 0 wherever f is below the severity limit).
- Toys: scripts in `.claude/worktrees/schedule-simplify/toys.tmp/` (untracked), outputs in
  `output/nucleation/`, plots in `plot/nucleation/`. With the bar free to heal, no g
  nucleates. With the bar fixed, every g grows material under it.
- **Unanswered, and the next step to raise:** the user doubted that the toys are
  representative, and asked whether a g that is identically 0 on a band gives only
  neutral stability. The agent agreed (neutral) and proposed (1) a g that dips below 0 at
  small x, so that a perturbation is pushed back, and (2) toy v3: the bar attached to the
  base, a normal front, then a **flat plateau** of `t`. Ask whether to build v3 with the
  negative-dip g.
- A cheap test that was proposed but never run: decouple `q` from `penal` during the grey
  hold and hold it high (e.g. `q` 6), with `k` 4.

## Experiments since the schedule settled (no change adopted)

- **Simplification** (`output/simp/`, `eval.json`, `plot/simp/dashboard.html`):
  V1 (Tcr held at 1 to 120) passes but gains nothing. Starting `beta_d` at 4 (V2/V4)
  fails S 0.3 (+0.56). V3 (penal ramp from 0) fails C1 0.5 (+0.03).
  The user decided to keep the settled schedule.
  - **Remind the user:** they want to revisit `beta_d` starting at 1 (one fewer
    hyperparameter; their intuition is that projection only matters at the end).
  - Pending cleanup: `beta_t` does nothing in the settled schedule, so its staircase can
    go from `default.json`.
- **Cooling sharpness k / angular-weight kappa continuation** (`output/kcont/`,
  `plot/kcont/dashboard.html`): sharpening k early (held at 2 or 4) makes S 0.3 fail every
  time (a grey blob under the top bar). The kappa ramp is neutral. Settled is kept.
- **Presentation assets** are finished: `plot/pres_tuning/` (before/after on C1 0.5, Tcr
  0.4: 1.109x vs 1.038x plain TO; GIFs, since the container has no `ffmpeg`).

## Open PRs and decisions waiting on the user

- **#208** `domain-mask` (worktree `schedule-simplify`): non-rectangular domains and the
  L-bracket load cases. Ready for review. The validation sweep already used it.
- **#207** `worktree-laplace-s03-schedule`: Variant 2 (`stto_laplace`) on S 0.3. It passes only
  with chi = max(rho^penal, 1e-5) * chi(mu) and `field_move` 0.01 (600 iterations) or 0.02
  (800 iterations), and the window is narrow. C1 0.3 and C1 0.5 fail. Its base was
  `settled-schedule` (merged), so check it targets `master`. User decisions are pending
  on: which binarized hotspot reading to gate on (the binarized check recomputes `t`),
  800 iterations for Variant 2, and a density-style filter on mu.
- `plans/print_order_credit.md` still lists the validation matrix and Tcr 0.8 as open.
  Update it once the user has seen the results.

## Deferred, by user decision

- Minimum length scale / hinge filters: `plans/length_scale_options.md` ("explore other
  avenues first").
- Variant 1 (`stto_heat`): set aside 2026-10-01 (`plans/virtual_heat_tuning.md`).
- The MMA trust region collapses during the grey hold (49% of the density variables at the
  asymptote floor on S 0.3 over 150-350). Noted, never followed up.

## Housekeeping

- **Do not push** the stale local branches `print-order-credit`, `damped-calibration` or
  `mma-diagnostics`, or the `backup-1008/*` branches. They predate the #206 rebuild, and
  pushing them would undo it. They are candidates to prune; ask first.
- No background jobs are running. No vast.ai instance is in use.

## How the user likes this work done

- Judge a schedule on several cells against a matched control, not on replicates of one.
  Give a holistic view rather than a single-detail cause.
- Have a dashboard ready with every result (`agent_docs/dashboards.md`). Plot severity
  without an `xPhys` mask.
- Prefer a fast toy experiment to validate a theory before hour-long `stto` runs.
- Prefer a simpler schedule when it costs nothing, but only adopt a change on evidence.
