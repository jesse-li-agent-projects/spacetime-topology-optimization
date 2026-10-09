# Variant 2 on S 0.3: the settled `stto` schedule

Resumes Variant 2 (`stto_laplace.py`) from `virtual_heat_tuning.md`, paused 2026-09-30.
`stto`'s schedule was since retuned (`print_order_credit.md`); the user's hypothesis is
that it carries over, since Variant 2 is `stto` with a time field that cannot form a
class of local extrema. Ask the user before any design decision this plan does not make.

Caveat on that hypothesis: a harmonic `t` has no interior maximum either, so every merge
of print fronts sits on a wall. On S 0.3 `stto` merges the top bar near the top wall, so
this cell can reach it; cells with interior merges (C1 0.3's strut) may not.

## Decisions (user, 2026-10-08)

- MVP: S 0.3 only (`corner_loads` 120×120, volfrac 0.3, `bottom_edge`, Tcr 0.6, 1 mm).
- Base: the settled `stto` schedule (`configs/default.json`, `continuation.md`; run
  `output/tune_b_r0.5_S03_T0.6_w50`), ported: nloop 600, β_d log 1 → 4 by 120, held to
  350, → 32 by 450; Tcr 1 → 0.95 by 50, → 0.6 by 350; penal and q 2 → 3 over 150–350,
  → 5 by 550; r 0.5 → 0.05 over 350–450; cooling credit τ 0.01, k 1 → 8 over 350–450;
  `calibration_rate` 0.1; no gravity, tool radius, `min_gradient` or smoothness. `t` is
  normalized to max 1 on the part, so τ carries over.
- Keep `stto`'s change points; tune the levels. Shift a change point only on clear
  evidence, and record why. The final schedule should be synchronized and simple.
- Wall data: optimized directly per node, with the 4 mm arc filter. `identity` time map
  with `gradient_cv`. Ramp init only (on a `bottom_edge` base the distance init is the
  ramp).
- **Code:** a move limit for `[μ; wall]` apart from `x` (the analog of `tmove`); shared
  by μ and wall, split only if the data shows they need different steps. Other code
  changes allowed behind options that default to off, as PRs on the stack, never merged
  by the agent.
- If the move levels and the weights do not pass, try other knobs one at a time against
  a matched control (wall filter radius, `chi_contrast`, Tcr timing). `stto`'s settled
  values stay the base.
- Gates, as `stto`'s: binarized hotspot row ≤ 0.01 at Tcr 0.6; ≤ 5 interior unsupported
  elements; start check (on the nodes); compliance ≤ 1.15 × `poc_S03_plainTO`. Report
  necking (flag ≥ 0.7), saddles and minima without gating.
- If S 0.3 passes with time left: the winner unchanged on C1 0.3 (Tcr 0.6) and C1 0.5
  (Tcr 0.4), report only, against `tune_b_r0.5_*_w50` and plain TO.
- Overnight: stop launching ~8 h after the start, finish the dashboard, one push
  notification with the summary.

## Steps

1. **Code.** The `[μ; wall]` move limit (it also sets their MMA asymptotes, as `tmove`
   does), defaulting to `move`. Per-iteration diagnostics if missing: largest and mean
   `|Δt|` on solid elements, fraction of time-field variables at the move limit and at
   the asymptote floor.
2. **Calibrate on `t`, not on the variables.** One μ/wall step moves `t` over a region,
   so match the realized `|Δt|` on solid elements over 0–150 to `stto`'s at `tmove`
   0.04 (from the `tune_b` snapshots). Re-measure `hotspot_weight` by `stto`'s
   gradient-ratio rule and `uniformity_weight` by the ≤ 0.2 × hotspot-term rule: the
   objective takes the log of compliance, so `stto`'s 50 and 20 do not transfer.
3. **Batch 1.** The ported schedule at the μ/wall move level {½, 1, 2} × calibrated ×
   `hotspot_weight` {0, calibrated}; control: `laplace_default`'s schedule on S 0.3
   (×0.75 to 600 iterations, Tcr 0.6, the same terms off).
4. **Refine** the winner (taper level, move split if needed); CPU and GPU replicate.
5. **Fallback knobs** (see Decisions), then the C1 check.

Runs: `output/lap_<batch>_<cell>_<setting>/`; driver, configs, logs in `output/lap/`.
Dashboard: `plot/laplace_s03/`.

## Summary (overnight 2026-10-08/09)

Departures from the steps: step 2's calibration run was skipped (a 600-iteration run
took 7–20 min on GPU, so the move levels were swept directly), and `stto`'s weights were
divided by the cell's plain-TO compliance instead of re-measured.

1. **The settled schedule does not carry over as it is.** With χ blind to the density,
   every move level (0.002–0.04), weight, wall filter (12 mm), Tcr start and hotspot β
   fails S 0.3 the same way (+0.56 to +0.64): μ and the wall data give the middle of the
   top bar the void's print time (a flat dip in the wall data, fed by a high-μ channel
   through the void below), the hotspot accepts it while grey material below counts as
   support, and after the binarization the local gradient cannot undo it.
2. **Variant 2 cannot represent `stto`'s passing S 0.3 time field**, which puts a merge
   (an interior maximum) on the top bar's underside. A near-exact fit (rms 0.001) scores
   +0.61.
3. **χ = max(ρ^penal, 10⁻⁵) · χ(μ) (`chi_density_exponent`) is what makes it pass**: the
   time field then runs through the material, up the diagonals and along the top bar to
   merges at the corners. The floor matters: at 10⁻² μ can still make void out-conduct
   solid.
4. **At 600 iterations the pass is fragile**: B passes at `field_move` 0.01 on GPU and CPU
   (−0.142, −0.022), but 0.005, 0.0075, 0.015 and 0.02 fail, not monotonically. **At 800
   iterations** (taper 700–750) B at 0.02 passes on both devices with margin (−0.188,
   −0.193; compliance 1.014; 0 saddles), and 0.0075 passes; 0.01 and 0.015 pass as
   optimized but not the check.
5. **The check recomputes `t` on the binarized design**, and with χ ∝ ρ^e a one-element
   bridge left grey can re-route it (Δt 0.07). With `t` kept (`stto`'s reading) every B
   run at 800 iterations passes. Which reading gates is a user decision.
6. C1 (report only): C1 0.3 breaks at 600 iterations (8.6 × compliance); at 800 it ends
   +0.004 as optimized but 1.216 × compliance (gate 1.15; `stto` 1.117). C1 0.5 at 600:
   +0.011 as optimized, 1.058 × (`stto` 1.038); at 800 (0.02): +1.47, 16 saddles. The
   800-iteration setting that is robust on S 0.3 does not carry over to C1 0.5.

Open for the user:
- Gate on the check (`t` recomputed) or on `t` kept? (Item 5.)
- 800 iterations for Variant 2, against the 600 `stto` settled on?
- A density-style filter on μ (Variant 1's deferred fix): μ still alternates 0/1 at
  element scale in some runs.
- Variant 2 forbids merges inside the part; on C1 this costs compliance (item 6).

## Results

### Batch 1 (2026-10-09, `lap1_*`): the μ/wall move level

The ported schedule at `field_move` {0.002, 0.005, 0.01, 0.02, 0.04} (flat, then ÷20 over
500–550) × `hotspot_weight` {0, 0.456 (= 50 / plain-TO compliance)}, and the
pre-tuning Variant 2 schedule (×0.75) as a control. Every finished run fails the
hotspot gate:

| Run | Hotspot row (binarized) | Compliance / plain TO |
|---|---|---|
| old schedule | +0.085 | 1.092 |
| `field_move` 0.002, w 0 / w 0.456 | +0.643 / +0.641 | 1.054 / 1.048 |
| `field_move` 0.005, w 0 / w 0.456 | +0.596 / +0.599 | 1.035 / 1.038 |
| `field_move` 0.02, w 0.456 | +0.601 | 1.010 |

- **Mechanism.** S 0.3's top bar lies on the top wall, so its print time is the wall data
  itself. While the design is grey (75–250) the hotspot lowers the wall data next to its
  hottest element, so that the neighbors print earlier; grey material below the bar
  (counted at r 0.5, q 2) reads as support. This leaves a flat dip on the wall
  (columns 70–90, 20 elements wide) and a second one at column 27. Binarization
  (350–450) removes the grey support, and the dip prints with nothing before it:
  row +0.1 → +0.6, hottest element (5, 80) to the end.
- **After binarization the hotspot's gradient on the wall is local and alternates in
  sign** (columns 75/80/85: +0.8 / −1.2 / +1.0): it asks for a tilt next to the hottest
  element, not for the dip to become a merge. A local minimum of the optimization, not a
  step-size limit: the wall steps sit at 1–25% of `field_move`, and the level does not
  change the outcome. `stto` recovers from the same jump at binarization because each
  element's `t` moves on its own.
- The old schedule's milder row is the sigmoid credit (a tie scores half credit), not a
  better time field: its top bar has the same kind of dips.
- μ and wall asymptotes collapse to the floor (85–99%) while the hotspot is inactive
  (50–120) and again once its multiplier is at `mma_c` (from ~250).
- **The dip is a leak through the void (2026-10-09, `lap1_S03_fm0.02_w50s` final).** μ
  builds a column at μ = 1 (χ ≈ 32) from the middle of the top bar (columns 80–84) down
  into the void below it; along it `t` is flat at 0.756, the void's time. The middle of
  the bar takes its print time from the void and prints early: Variant 1's leak, made
  here by μ alone, since χ does not see the density. Hence `chi_density_exponent`
  (commit `30ce59c`, default 0): χ = max(ρ^e, floor) · χ(μ). Variant 2 has no drain, so
  `t` does not decay by decades as in Variant 1's `w32` (exponent 3 left `T` below the
  solve error on 31% of the part); a high exponent should stay resolved.
- **Variant 2 cannot represent `stto`'s passing S 0.3 time field** (`oracle_fit.tmp.py`,
  `output/lap/oracle_e*.log`). Fitting (μ, wall) so that the harmonic `t` matches the
  final `t` of `tune_b_r0.5_S03_T0.6_w50` on its binarized part reaches an rms error of
  0.001 (max 0.011, 1500 Adam steps), yet Variant 2's own hotspot scores the fit +0.61,
  against −0.23 for `stto`'s `t` under the same measure (exponent 1: +0.62). `stto`
  prints the top bar's underside last near column 52 (rows 8–9 at 0.993, `t` falling
  above and below): an interior maximum on the part, i.e. a merge inside the bar, which
  a harmonic field with `t` defined in the void cannot have. The fit flattens it, and
  the bar's lower edge reads hot (0.96–0.97). The hotspot resolves `t` to ~0.005 (τ =
  0.01), so small fit errors matter. Whether a *different* time field for this layout
  passes under Variant 2 is open; the optimized runs below test that.

### Batches 2–4: one change each against `lap1_S03_fm0.02_w50s` (+0.601)

| Run | Change | Hotspot row (binarized) | Compliance / plain TO |
|---|---|---|---|
| `lap2_S03_wf12` | wall filter 12 mm | +0.589 | 1.010 |
| `lap2_S03_chi1` | χ density exponent 1 | +0.617 | 1.011 |
| `lap3_S03_hb10` | hotspot β 10 until 350, → 100 by 450 | +0.570 | **19.9** (load path broken on binarizing) |
| `lap3_S03_q3` | q 3 | stopped by the agent for batch 5 | |

Every run jumps to +0.3–0.5 over the binarization (350–450), as the control does.
- With exponent 1 the leak through the void is smaller, but μ finds another route:
  inside the solid bar it alternates 0/1 element to element (χ 0.03–32). Low-μ cells
  insulate a segment of the bar (columns 68–95), high-μ channels join it to grey stubs
  below, and the segment sits on a flat plateau (`t` 0.848, rows 0–7), printed at once.
  The floor 10⁻² on ρ^e is also too high against a χ(μ) range of 1000: void at μ = 1
  (χ 0.32) still out-conducts solid at μ = 0 (0.032).
- So the lever is μ's element-scale contrast (Variant 1 saw the same 0/1 alternation).
  A density-style filter on μ needs the user's agreement (archived plan), so batch 5
  lowers `chi_contrast` instead (10, 100), a config knob.

### Batch 5–6: χ contrast, wide wall filter, and the density exponent with a low floor

- `chi_contrast` 10 and 100 (χ blind to the density) form the same dip at column ~80 by
  iteration 150–230; stopped at ~it 300 (agent). The dip comes from the wall data; μ's
  channels come after. The 40 mm wall filter (χ blind to the density) was stopped at its
  start, once the run below changed the base.
- **`lap5_S03_chip_f5`: exponent = `penal` (2 → 3 → 5), floor 10⁻⁵.** Row +0.319
  binarized (+0.118 as optimized), compliance 1.075: the first clear change. The time
  field now runs through the material: 0 along the bottom bar, up both diagonals, into
  the top bar where the left diagonal meets it (column ~20, `t` 0.16), then along the
  bar both ways, to the corners (merges on the wall). The rest of the violation is on the
  top bar near (0, 28), where `t` rises only ~0.002 per element against τ = 0.01, and the
  range is uneven (90% of the part below `t` 0.35; the top-right corner sets the
  maximum). Base `B` for batch 7.

### Batch 7: around B

- **`lap7_S03_B_fm0.01` passes every gate** (GPU): binarized row −0.142 (as optimized
  −0.137), compliance 1.023 × plain TO (`stto` settled: 1.007), 0 interior unsupported,
  start passes, 0 saddles, 0 local minima; row ≤ 0.01 from iteration 502; multiplier at
  `mma_c` for 139 iterations (B at 0.02: 288). Same skewed-hourglass layout as `stto`,
  top bar a little thicker. The row still jumps over the binarization (+0.48 at 400) and
  recovers by ~500, where B at `field_move` 0.02 recovers only to +0.12.
- `field_move` 0.04 on B stopped by the agent (0.02 already fails); batch 8 checks 0.01
  on CPU and its neighbors 0.005 and 0.015.
- **Binarizing moves `t` once χ follows the density.** The check recomputes `t` on the
  binarized design; with χ ∝ ρ^e the 1–2% grey edge elements (β_d ends at 32) switch
  between conducting and not, and re-route `t`. `stto`'s check keeps `t` (a design
  variable). Hotspot row three ways (`bin_keep_t.tmp.py`): binarized with the
  as-optimized `t` kept / the check (`t` recomputed) / as optimized:

  | Run | `t` kept | check | as optimized |
  |---|---|---|---|
  | B, `field_move` 0.02 (GPU) | +0.106 | +0.319 | +0.120 |
  | B, `field_move` 0.02 (CPU) | −0.019 | +0.028 | +0.001 |
  | B, `field_move` 0.01 | −0.138 | −0.142 | −0.137 |
  | B, `field_move` 0.005 | −0.156 | +0.564 | −0.082 |
  | B, exponent 3 (0.02) | −0.153 | +0.188 | −0.058 |

  Which one gates is the user's call; the plan's gate (the check) is kept, and both are
  reported. `lap8_S03_B_fm0.005_bd128` (β_d 32 → 128 over 450–550) tests whether full
  binarization closes the gap.
- `lap7_S03_B_u5` (uniformity ×5): +0.647, compliance 7.2 ×, 7.6% grey. Rejected.
- **Full binarization does not close the gap.** `lap8_S03_B_fm0.005_bd128` (β_d → 128 by
  550; grey 0.2%) ends as optimized −0.053, `t` kept −0.056, check +0.566. Where `t`
  moves most (Δt 0.07 at (107, 34)): a grey stub near the bottom bar (rows 104–110,
  columns 31–35, ρ 0.5–0.7) conducts weakly as optimized; binarized, a few of its elements
  turn solid and join it to the plate at full χ, so `t` there falls from 0.07 to 0.004 and
  the elements around it read hot. With χ ∝ ρ^e the recomputed `t` hangs on single-element
  connections: a fragility of the density coupling, not of the check.
- `lap7_S03_B_n800` (B at 0.02, 800 iterations, taper over 700–750): **passes**, check
  −0.188, compliance 1.014, 0 saddles. At 0.02, B needs more mobile iterations after the
  binarization than 600 leave.
- `lap8_S03_B_fm0.01_cpu`: **passes** (check −0.022, `t` kept −0.052, compliance 1.017,
  1 saddle). B at 0.01 passes on both devices, with less margin on CPU.
- **Narrow window at 600 iterations.** B at `field_move` 0.015 (CPU) fails as optimized
  (+0.533); exponent 3 at 0.01 (CPU) fails (+0.550) though exponent 3 at 0.02 (GPU) passed
  as optimized (−0.058). Not monotone in either knob: on S 0.3 single runs are weak
  evidence, and the passing setting has little margin around it.
- **C1 0.3, B at 0.01 unchanged (report only): fails** — +0.256 as optimized, +0.633
  binarized, compliance 8.56 × plain TO (binarizing breaks the load path), grey 3.8%.
  `stto` settled passes this cell at 1.117 ×.
- **C1 0.5, B at 0.01 unchanged (report only):** as optimized +0.011, `t` kept −0.012,
  check +0.508 (the recompute gap), compliance 1.058 × plain TO (`stto` settled 1.038).
- B at `field_move` 0.0075 fails (+0.466 as optimized): over 0.005–0.02 the 600-iteration
  outcome is not monotone in the level (0.005 passes as optimized, 0.0075 fails, 0.01
  passes on both devices, 0.015 fails, 0.02 marginal). Batch 10 tests whether 800
  iterations (taper 700–750), which passed with the most margin at 0.02, removes this.

### Batches 9–11: 800 iterations and the C1 check

800 iterations, `field_move` taper and roughness floor at 700–750, every other change
point as at 600. Hotspot row: check / `t` kept / as optimized; compliance ÷ plain TO;
necking max (flag ≥ 0.7).

| Run | check | `t` kept | as opt. | C / TO | necking |
|---|---|---|---|---|---|
| B 0.02, GPU (`lap7_S03_B_n800`) | **−0.188** | −0.207 | −0.174 | 1.014 | 0.29 |
| B 0.02, CPU (`lap9_S03_B_n800_cpu`) | **−0.193** | −0.216 | −0.189 | 1.014 | 0.29 |
| B 0.0075 (`lap10_…_fm0.0075_n800`) | **−0.071** | −0.190 | −0.146 | 1.051 | 0.38 |
| B 0.01, CPU (`lap9_…_fm0.01_n800`) | +0.064 | −0.044 | +0.001 | 1.042 | 0.78 |
| B 0.015, CPU (`lap10_…_fm0.015_n800`) | +0.113 | −0.016 | −0.010 | 1.100 | 0.66 |
| C1 0.3, B 0.02 (`lap11_C103_B_n800`) | +0.172 | −0.028 | +0.004 | 1.216 | 0.73 |
| C1 0.5, B 0.02 (`lap11_C105_B_n800`) | +1.472 | +1.490 | +1.473 | 1.140 | 0.81 |

Necking of the references: `stto` settled S 0.3 0.29, C1 0.3 0.68, C1 0.5 0.87; plain TO
S 0.3 0.27, C1 0.3 0.30, C1 0.5 0.74. The passing B runs at 600 iterations: 0.12–0.14.
