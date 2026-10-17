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

Results: see "Overnight results" below, if the session got that far. Otherwise evaluate them
with the `t`-kept gate and extend the dashboard (`plot/laplace_s03/`, built by
`dash_tabs.tmp.py`, tested by `dash_test.tmp.py`; serve `plot/` per `agent_docs/dashboards.md`).

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
