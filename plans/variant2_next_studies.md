# Variant 2: next studies

Variant 2 (`stto_laplace`) passes all eight validation cells on one default
(`configs/laplace_default.json`, rationale in `configs/laplace_continuation.md`). The
tuning history is in `archive/scale_invariant_hotspot.md`. This plan lists what to run
next, in the user's priority as of 2026-10-23. Agree each study's setup with the user
before starting it.

Compute: CPU only until the user frees the GPU. A 180 × 60 run of 600 iterations takes
25–35 min at 4 threads, three at a time (`lap_driver4.tmp.py`, worktree
`laplace-s03-schedule`).

## 1. The plain-TO gap

Variant 2's plain TO (its schedule, Tcr held at 2) is coarser than `stto`'s: C1 0.5
1.007 ×, C1 0.3 1.043 × `stto`'s plain TO, with fewer members
(`plot/pres_weekly_1023/fig2_plainTO.png`). Most of C1 0.3's cost over `stto`'s plain
TO is this gap, not the hotspot. The 2026-10-14 `stto` plain TO is
`output/val_C1_v0.5_e_plainTO`.

Suspects, from the config diff:
- **The density continuation.** Variant 2 ramps penal 2 → 5 over 0–300 with β_d held at
  2 and a density move of 0.02. `stto` holds penal at 2 to 150, reaches 3 at 350 and 5
  at 550, with β_d 1 → 4 by 120, then → 32, and move 0.01.
- **The time-field objective terms.** Uniformity and roughness reach the density through
  χ = ρ⁴, so "plain TO" with Tcr at 2 still trades compliance for the time field.

Runs, C1 0.5 and C1 0.3:
- the control: plain TO on the current default;
- the time-field terms off;
- `stto`'s density schedule;
- both changes.

If one closes the gap, rerun the simultaneous C1 cells with it.

## 2. A fair sequential reference

The current one (`lap_seqref.tmp.py`, then `seqopt`) reads severity differently from the
gate:
- `seqopt`'s sigmoid credit is the gate's cooling credit at k = 1, while the gate uses
  k = 8;
- Tcr is not imposed, because the hotspot is only a weighted objective term.

So a failing sequential cell doesn't show that the density had to move.

- **2a, direct t** (the user prefers it for `seqopt`):
  - Add `hotspot_credit`, `hotspot_cooling_time` and `hotspot_cooling_sharpness` to
    `SeqRunConfig`, scheduled so it can ramp k 1 → 8. `conductivity` already supports
    them.
  - Make Tcr binding, either as a constraint or as a heavier weight. This is the user's
    call.
- **2b, Laplace t:** freeze the plain-TO density and optimize μ and the wall data
  against the gate (as `lap_seq_ref.tmp.py` did in batch 35).

Eight cells each. Report the gate, and compliance against the simultaneous design.

## 3. Ablation: virtual (Laplace) vs direct time field

Same density, same hotspot formulation and gate, only the time-field parameterization
differs.
- **Sequential:** 2a against 2b on the same plain-TO designs.
- **Simultaneous:** `stto` (direct t) against `stto_laplace`, with every shared setting
  equal (schedules, `capped_ratio`, the cooling credit). Agree with the user what can't
  be shared (`stto`'s `tmove` and its time-field regularizers against `field_move`, μ
  and the wall data).
- **Metrics:** gate, compliance, grey, the time field's saddles and unsupported elements,
  and the spread over repeats (runs are not bitwise reproducible).

## 4. C1 at a higher resolution

C1 0.5 Tcr 0.4 and C1 0.3 at 360 × 120 on the default. Lengths are in metres, so the
filters and stencil keep their physical size; τ is in units of t and does not scale.
Estimate memory before the first CPU run (four times the elements). Questions:
- Does the 2× run pass its own gate?
- Is its topology the 1× one?
- How does its compliance ratio compare?

The 2× reading of the 1× design fails (+0.29 to +0.47 at the worst element), and this
separates "the design isn't resolution-robust" from "the problem isn't".

## 5. The gate's resolution sensitivity

The gate reacts to t perturbations of 0.005–0.01, which is τ, about two elements of
print time (`roundtrip.tmp.py`, `t_sensitivity.tmp.py`). After 4, decide with the user
whether τ should be defined in physical units, or the gate read at a reference
resolution.

## Not on this list

- **Small enclosed holes:** tabled 2026-10-22.
- **Nucleation:** open, with questions for the user in `nucleation.md`.
- **The μ filter:** dropped.
- **Minimum length scale:** deferred, in `length_scale_options.md`.
