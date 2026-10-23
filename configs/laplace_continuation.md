# The default Laplace continuation schedule

`laplace_default.json` holds the numbers for `stto_laplace`; this is why they sit where
they do. Iterations are for `nloop` 600. Evidence per batch is in
`plans/archive/scale_invariant_hotspot.md` (PR #207).

## Phases

| Iterations | Phase | What moves |
|---|---|---|
| 0-300 | layout forms | `hotspot_kappa`, `penal`, `beta_t` ramp linearly; cooling sharpness `k` 1 -> 8 (geometric); `Tcr` 1 -> target over 0-100; `beta_d` held at 2, `q` held at 1 |
| 300-400 | sharpen | `beta_d` 2 -> 8 (geometric) and `q` 1 -> 3, together |
| 400-600 | settle | nothing ramps; the roughness weight reaches its floor at 600 |

The move limits are fixed: 0.02 on the density, 0.002 on the time-field variables.

## Why

- **`q` follows `beta_d`.** The hotspot credits a grey neighbour `(x_b / x_a)^q`; on a
  binary design that is 0 or 1, so `q` only shapes the grey. `q` = 1 while the layout
  forms reads grey support generously, which gave the best compliance. Held to the end,
  it lets a grey design pass that fails once binarized (C1 0.3 by +0.04). Raising it to 3
  while `beta_d` sharpens makes the reading of grey support as strict as binarizing
  would be. Since the design still ends 6-13% grey, `q` matters to the end.
- **`beta_d` ends at 8, not 128.** Branched from 300 on the four hardest cells: -> 8
  passes all of them with equal or better compliance (C1 0.3 1.066 against 1.074 at
  128) and smooth 0.5 contours; -> 4 and held at 2 fail on C1. Ending grey means gating
  on the binarized hotspot row, not the one as optimized.
- **`beta_d` held at 2 until 300, then a gentle ramp.** Held low with `q` = 1, C1 got its
  best and most repeatable compliance; a steep ramp tears the design (C1 compliance
  peaks near 400 against about 210).
- **`k` geometric.** `k` is a softplus sharpness, so equal ratios are equal steps; as good
  as or better than linear on C1.
- **Ends at 600.** Every cell passes the binarized gate from 500 on in both validation
  batches, and the last 50 iterations move compliance by at most 0.3% (C1).
- **Fixed move limits.** Dropping the density limit's halving after 300 (together with
  the faster `q` ramp) cost nothing; dropping the time field's halving over 550-600 made
  no difference beyond run-to-run noise (compliance within 0.25% on every cell).
- **No `mu` filter** (`mu_filter_rmin_m`): it halves the diffusivity's roughness but
  leaves C1 grey and failing once binarized.

## Validated on

Eight cells, two runs each (batches 41 and 42; three on C1 0.5 Tcr 0.4): C1 0.5 at Tcr
0.4 and 0.6, C1 0.3, C2 0.5, S 0.3 bottom and edge, L middle and corner. All pass, at
1.006-1.060 x plain TO.
