# A hotspot severity invariant to scaling every density

## Motivation

On C1 0.3, Variant 2's grey bars stay grey until forced to solidify, and often vanish
when they do (user, 2026-10-18). The suspected cause is the hotspot measure itself: with
`absolute` shielding, `K_e = Σ_j w F x_j^q / D` scales as `n^q` when every density is
scaled by `n`. While the bar is grey it reads as unshielded, its own density only ever
gets pushed down (`∂S_e/∂x_e = r x_e^(r-1) (1 - K_e) > 0`), and solidifying it helps only
once it is solid all at once. Removing the bar also clears the hotspot.

Requirement: `S_e(n x) = S_e(x)` for every element, i.e. `Σ_j x_j ∂S_e/∂x_j = 0`, so the
hotspot exerts no force along "scale every density".

## Formulations (`hotspot_shielding`)

Neighbor `j` contributes `c(u)` per unit weight, `u = (x_j + ε) / (x_e + ε)`:

- **A, `capped_ratio`:** `c = min(1, u)^q`. A neighbor at least as dense as `e` counts in
  full, so a grey element reads as it would once it and its support were solid.
- **B, `ratio`:** `c = u^q`, uncapped. One dense neighbor can make up for a missing
  direction, so a grey element on partial support reads as safe and solidifying it
  reveals the hotspot. Kept as a check that the cap is what matters.

Both agree with `absolute` on binary designs to within `ε`, so the binarized gates read
the same.

**The gate.** `S_e = 1 - K_e` alone fails: void at the mesh edge has part of its stencil
missing, reads as unshielded (~0.5 at a side wall) and no design change fixes it; and
void elements would be back in the constraint with a severity set by their own print
time, which PR #98 made drop out. So the severity keeps a gate, `x_e / (x_e + ε)`: zero
on void, flat above ~10ε, so it neither breaks invariance nor pushes grey density down.
ε = 1e-3. LogSumExp only (`ratio` is unbounded below, which `p_mean` cannot take).

## Batch 14 (2026-10-18)

Config B (`lap11_C103_B_n800`, `lap11_C105_B_n800`) at 800 iterations, `field_move`
0.02, with `hotspot_shielding` A or B, on C1 0.3 (Tcr 0.6) and C1 0.5 (Tcr 0.4), and
batch 13's sweep of f, the fraction of the way to the final Tcr at iteration 0
(`variant2_handoff.md`). No new controls: the user treats this as pass/fail. Batch 13's
C1 0.3 runs and `lap11_C105_B_n800` (f = 0) stand as the `absolute` reference.

Gates as in `variant2_handoff.md`: the binarized hotspot row with `t` kept ≤ 0.01, read
with `absolute` shielding (`bin_keep_t14.tmp.py`); compliance ≤ 1.15 × plain TO; ≤ 5
interior unsupported elements; print start. Driver `lap_driver2.tmp.py` (GPU + 3 CPU × 4
threads), batch script `lap_batch14.tmp.py`, code snapshot `output/lap/snapshot_4b2cedd`.
Dashboard: `lap_dashboard3.tmp.py`.

## Results

Pending.
