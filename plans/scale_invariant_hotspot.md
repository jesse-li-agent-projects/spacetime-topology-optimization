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

## Results (batch 14)

Hotspot row binarized with `t` kept (absolute shielding) / compliance ÷ plain TO. ✓ =
every gate passes. Every run: 0 interior unsupported, print start passes.

| cell | f | current (batches 11, 13) | A `capped_ratio` | B `ratio` |
|---|---|---|---|---|
| C1 0.3 | 0 | −0.028 / 1.193 | **−0.009 / 1.098 ✓** | +0.463 / 1.033 |
| C1 0.3 | 1/3 | −0.011 / 1.132 ✓ | +0.042 / 2.702 | +0.587 / 7.284 |
| C1 0.3 | 2/3 | −0.008 / 1.275 | −0.012 / 2.136 | +0.561 / 2.809 |
| C1 0.3 | 1 | +0.002 / 1.085 ✓ | −0.018 / 2.304 | +0.441 / 1.378 |
| C1 0.5 | 0 | +1.490 / 1.134 | +0.201 / 1.109 | +1.495 / 1.592 |
| C1 0.5 | 1/3 | – | +0.050 / 2.028 | +1.495 / 1.516 |
| C1 0.5 | 2/3 | – | −0.006 / 1.805 | +1.343 / 1.819 |
| C1 0.5 | 1 | – | +0.078 / 5.639 | +1.428 / 1.740 |

- **B fails everywhere**, as predicted: its as-optimized row is ≤ 0.03 in 6 of 8 runs
  while the binarized row is +0.44 to +1.50 — grey on partial support reads as safe and
  the hotspot appears on binarizing.
- **A at f = 0** passes C1 0.3 and cuts C1 0.5's binarized hotspot from +1.49 to +0.20.
- **A at f > 0** wrecks compliance (1.8–5.6 × plain TO). The hotspot sits on near-void
  wisps: on C1 0.3 at f = 1, iterations 100–300, the top-50 severities have median
  density 0.07–0.14 and 500–900 elements exceed Tcr, most with `x < 0.1`
  (`sev_density.tmp.py`); at f = 0 no element exceeds Tcr there. The `x / (x + ε)` gate
  counts anything above ~10ε as material, and scaling a wisp's neighborhood down is the
  very direction the invariance makes force-free, so only the gate could delete it, and
  it is flat there. The gradient that remains mostly raises the wisp's earlier
  neighbors, which grows more light material.
- Next: a gate that is a step at a meaningful density, e.g. `x^k / (x^k + η_g^k)` with
  `η_g ≈ 0.2–0.3`, so a grey bar sits on its flat part and a wisp on its slope.

Dashboard: `plot/laplace_s03/dashboard_shielding.html` (spec `lap_spec14.tmp.json`);
table `output/lap/eval14.json` (`lap_eval14.tmp.py`).
