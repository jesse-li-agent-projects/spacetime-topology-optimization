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

## Results
