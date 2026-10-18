# Handoff: Variant 2 (`stto_laplace`) tuning, as of 2026-10-18

A quick start for the next agent. Details live in the files linked below; this note only
says where things stand and where to look. Read `laplace_s03_schedule.md` (this branch)
first, then `virtual_heat_tuning.md` for the pass criteria and the older history.

## Where things stand

- Variant 2 = `sttopt/stto_laplace.py`: `t` solves `∇·(χ∇t) = 0`, `t = 0` on the plate,
  optimizable Dirichlet data on the other walls (`archive/virtual_heat_timefield.md`).
  Variant 1 (`stto_heat`) is set aside.
- PR #207 (this branch, worktree `.claude/worktrees/laplace-s03-schedule`) adds
  `field_move` and `chi_density_exponent`/`chi_density_floor`, plus the Oct 8–9 overnight
  results. **Its base is still `settled-schedule` (merged as #205): retarget it to
  `master`** (ask first). It predates #208, so `stto_laplace` has no design-domain mask.
- The `stto` schedule was ported (`configs/default.json`, `configs/continuation.md`). It
  fails S 0.3 as is (a leak through the void). It passes with χ = max(ρ^penal, 1e-5)·χ(μ):
  setting **B** (`lap5_S03_chip_f5`, then `lap7_*`). Full results:
  `laplace_s03_schedule.md`, "Summary" and "Results".
- The time-only hotspot objective (`hotspot_weight`, commit `ed6ed9b`) is in
  `stto_laplace`. B has it on: `stto`'s 50 ÷ the cell's plain-TO compliance.

## User decisions, 2026-10-18

- **Gate on the binarized hotspot with `t` kept** (the as-optimized `t`, as `stto`
  reads it), not on `t` recomputed on the binarized design. The checks still recompute
  `t`, so compute this reading with `bin_keep_t.tmp.py TAG...` (worktree; cached in
  `output/lap/keep_t.json`). Report the check's reading next to it.
- **800 iterations are OK for development**, but the goal is to bring them back to 600.
- Validation cells: the rectangular ones, without S 0.7.
- **The user's hypothesis:** C1 0.3 fails because the hotspot acts only after the design
  has solidified. While `r` is 0.5 (until 350), a lower starting Tcr might work, even
  Tcr at its final value from iteration 0. Batch 13 tests this.
- **Next code task: L-bracket support in `stto_laplace`** (the `l_bracket_*` load cases,
  `load_cases.active_mask`; see how `stto` does it in #208, `archive/domain_mask.md`).
  Open design questions for the user: which walls carry the optimizable Dirichlet data
  (the inner walls of the re-entrant corner?), the re-entrant corner itself, and whether
  `t` is solved only on the domain or on the mesh with χ ≈ 0 outside it.

Under the `t`-kept gate, the earlier B runs read (`keep_t.json`): S 0.3 at 600 iterations
passes at `field_move` 0.01 (GPU −0.138, CPU −0.052), 0.005 (−0.156) and 0.02 on CPU
(−0.019), and fails at 0.02 on GPU (+0.106). At 800 iterations every B run on S 0.3
passes. C1 0.3 at 800 passes the hotspot gate (−0.028) but not compliance (1.216 × plain
TO). C1 0.5 at Tcr 0.4 passes at 600 iterations, `field_move` 0.01 (`lap9_C105_B_fm0.01`:
−0.012, compliance 1.058), and fails at 800 iterations, 0.02 (+1.49).

## Overnight 2026-10-18/19 (batches 12–13)

The driver is `lap_driver2.tmp.py` (GPU + 1 CPU × 4 threads, because the nucleation toys held
8 threads), and the batch scripts are `lap_batch12.tmp.py` and `lap_batch13.tmp.py`, all in
the worktree. The code snapshot is `output/lap/snapshot_30ce59c` (= this branch's code).
The queue, configs, logs and `driver.log` are in `output/lap/`, and the runs in `output/lap1[23]_*`.

- **Batch 12, generalization (report only):** B unchanged on S 0.5 b 0.8, C1 0.5 e 0.6,
  C2 0.5 e 0.6, C2 0.5x b 0.6, S 0.5x e 0.8. Each runs at 800 iterations (`field_move`
  0.02) and at 600 iterations (0.01). Weights rescaled by the cell's `val_*_plainTO`.
  Controls: `stto`'s `output/val_<cell>` and `val_<load>_<vf>_<base>_plainTO` (matrix in
  `output/val/`, dashboard `plot/val/`).
- **Batch 13, a lower starting Tcr:** B at 800 iterations on S 0.3, C1 0.3 and C1 0.5
  (Tcr 0.6). Every Tcr point is moved a fraction f ∈ {1/3, 2/3, 1} of the way to the
  final value, so f = 1 holds Tcr at its final value from iteration 0. Controls (f = 0):
  `lap7_S03_B_n800`, `lap11_C103_B_n800`, `lap12_C1_v0.5_e_T0.6_B_n800`.

## Overnight results (2026-10-18, one run per setting)

All 19 runs finished. Table: `lap_eval12.tmp.py` → `output/lap/eval12.json`. Dashboard:
`plot/laplace_s03/overnight_1018.html` (built by `lap_dashboard2.tmp.py
lap_spec_1018.tmp.json` with `LAP_SNAP=output/lap/snapshot_30ce59c`; refs from `val_*`,
hotspot gated with `t` kept). Hotspot = binarized row with `t` kept (gate ≤ 0.01);
C = compliance ÷ plain TO (gate ≤ 1.15). Every run passes start and support (0
interior unsupported).

**Batch 12: B generalizes to every cell, at some cost in compliance.** Both settings pass
all five cells. That includes S 0.5 b 0.8, which `stto` fails (+0.154); Variant 2 passes it
on both readings.

| Cell | 800 it, 0.02: hotspot / C | 600 it, 0.01: hotspot / C | `stto`: hotspot / C |
|---|---|---|---|
| S 0.5 b 0.8 | −0.273 / 1.011 | −0.216 / 1.014 | +0.154 / 1.002 |
| C1 0.5 e 0.6 | −0.018 / 1.021 | −0.016 / 1.012 | −0.027 / 1.004 |
| C2 0.5 e 0.6 | −0.057 / 1.059 | −0.114 / 1.026 | −0.038 / 1.003 |
| C2 0.5x b 0.6 | −0.089 / 1.093 | −0.110 / 1.082 | −0.027 / 0.999 |
| S 0.5x e 0.8 | −0.050 / 1.013 | −0.095 / 1.013 | −0.631 / 1.005 |

The check's reading (`t` recomputed) fails 8 of these 10 runs (+0.06 to +0.58). On these
cells the 600-iteration setting is as good as or better than 800.

**Batch 13: a lower starting Tcr.** f = the fraction of the way to Tcr final.

| f | C1 0.3: hotspot / C | S 0.3: hotspot / C | C1 0.5 (Tcr 0.6): hotspot / C |
|---|---|---|---|
| 0 (control) | −0.028 / **1.193** ✗ | −0.207 / 1.011 | −0.018 / 1.021 |
| 1/3 | −0.011 / 1.132 | −0.237 / 1.021 | −0.021 / 1.020 |
| 2/3 | −0.008 / **1.275** ✗ | −0.179 / 1.018 | +0.003 / 1.022 |
| 1 | +0.002 / 1.085 | **+0.663** / 1.186 ✗ | −0.018 / 1.004 |
| `stto` | −0.021 / 1.093 | −0.226 / 1.003 | −0.027 / 1.004 |

- f = 1/3 is the only f that passes all three cells. f = 1 gives the best C1 compliance
  (1.085, below `stto`'s 1.093) but breaks S 0.3. The result is not monotone on C1 0.3
  (f = 2/3 fails), so a single run per f does not separate f from chance there. Next:
  replicates on C1 0.3 (CPU and GPU), and f = 1/3 at 600 iterations.
- S 0.3 at f = 1: the hotspot multiplier sits at `mma_c` for 564 of 800 iterations, from
  iteration ~150. The row stays small while the design is grey (+0.04 at 300), then jumps
  over the binarization (+0.43 at 350, +0.63 at 450) and does not recover (+0.642 as
  optimized). This is the same jump as the Oct 9 failures. The layout has not been
  inspected yet; the frames are in the dashboard.

## Still open

- A density-style filter on μ (μ alternates 0/1 element to element in some runs). Not
  yet agreed with the user.
- C1: Variant 2 cannot place a merge of print fronts inside the part (a harmonic `t`
  has no interior maximum). Whether the C1 compliance cost is acceptable is the user's
  call, and so is whether batch 13 changes the picture.
- Two calibration steps skipped on Oct 8: re-measuring the weights on Variant 2 (they
  were scaled by plain-TO compliance instead) and the cooling time τ against the range of
  `t` under the `identity` map.

## Housekeeping

- The user trashed old runs on 2026-10-18 (`tune_b*`, `poc_*`). They can be recovered
  from an external backup if needed. `lap_jobs.tmp.py` reads them and so no longer works
  as is, but batches 12–13 take their templates from `output/lap/configs/`. The dashboards'
  frames for those runs are still in `plot/*/frames/`.
- Past sessions (transcripts under `/home/ubuntu/.claude/projects/`):
  - `01c0f687` (worktree `laplace-s03-schedule`): the Oct 8–9 overnight run and the user's
    decisions for it.
  - `653a5621`: the Oct 14 status report.
  - `9b945c74`: the Oct 6 recap of both variants.
  - `aed802b4`: the `stto` tuning handoff (PR #209, closed).
  - `recovered/transcripts/…phase8-own-t-check/639e312c…`: Variant 2 paused on Sep 30.
