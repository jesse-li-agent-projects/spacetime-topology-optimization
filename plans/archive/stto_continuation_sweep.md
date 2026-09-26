# stto continuation sweep — reaching compliance < 200 with the half-stencil hotspot measure

Question: can `stto` reach whole-structure compliance below 200 with smooth layers
(`uniformity_weight` 100) while the half-stencil hotspot severity stays at or below
`Tcr` = 0.8? The `stto_unif_w*` runs that preceded this sweep sat at 239–251 and broke
the hotspot limit.

Answer: yes — 180.8 with the severity held exactly at 0.800, which is 0.3 above the
compliance of the same problem with no hotspot constraint at all.

All runs: 180x60, 800 iterations, volfrac 0.5, `opposite_corner` print base unless the
run name says otherwise. Code: branch `worktree-exp-stto-continuation`, commit 3b50d55,
run from a frozen copy. The run artefacts (`output/stto_continuation_sweep/`) have been
deleted; settings below predate the physical-units configs.

## Recommended settings

```json
"hotspot_aggregation": "logsumexp_severity",
"hotspot_beta": 100.0,
"hotspot_refresh_period": 1,
"Tcr": {"points": [[1, 5.0], [150, 5.0], [250, 0.8]]},
"uniformity_weight": 100.0
```

Everything else stays at `configs/default.json`. This is run `p4_on150_b100_r1`.

Three changes matter, each for its own reason:

1. **`logsumexp_severity`** takes the smooth maximum of the severity `T * x**r` itself,
   which is what `p_mean` and the calibration always targeted. The previous `logsumexp`
   takes the smooth maximum of `T` alone with `x**s` as a separate weight.
2. **A sharp `hotspot_beta`.** The aggregate's gradient is shared by `n_eff` elements;
   at beta 4 that is ~4000 of 5400 solid elements, so the row pushes the *mean* severity
   down and thickens every member. Beta 25-100 concentrate it on 7-30 elements.
3. **`hotspot_refresh_period` 1.** The calibration offset is a constant in the row's
   gradient, so refreshing it every iteration costs nothing and removes a sawtooth: with
   the default 25, the true maximum drifts up to 0.83 between refreshes while the
   calibrated aggregate reads exactly 0.800.

**The delayed onset (`Tcr` held high, then ramped down) is what buys the compliance.**
Holding the row on from iteration 1 shapes the design during the grey phase and costs
~5.5; turning it on once the topology has formed costs 0.3.

## Results

`c` is whole-structure compliance, `bin` the same on the 0.5-thresholded design. `tail`
is the range of the true maximum severity over the last 50 iterations — an end-of-run
value alone hides the calibration sawtooth.

| run | what it changes | c (bin) | true max (bin) | tail | unif |
|---|---|---|---|---|---|
| `p0_A0` | no hotspot, no layer terms | 179.65 (179.42) | 0.886 (0.883) | 0.884–0.886 | 0.83 |
| `p0_A1` | + `uniformity_weight` 100 | 180.51 (180.30) | 0.901 (0.899) | 0.900–0.901 | 0.025 |
| `p1_b4` | hotspot on, beta 4 | stopped at it 74 | 0.934 | — | — |
| `p1_b12` | beta 12 | stopped at it 85 | 0.912 | — | — |
| `p1_b25` | beta 25 | 187.97 (187.91) | 0.768 (0.798) | 0.767–0.830 | 0.042 |
| `p1_b50` | beta 50 | 186.95 (186.79) | 0.799 (0.802) | 0.798–0.803 | 0.029 |
| `p1_b50_r1` | + refresh 1 | 187.25 (187.07) | 0.800 (0.800) | 0.7998–0.8001 | 0.031 |
| `p2_rep` | repeat of `p1_b50_r1` | 187.34 (187.17) | 0.800 (0.800) | 0.7999–0.8001 | 0.031 |
| `p2_tcr079` | `Tcr` 0.79 | 187.60 (187.42) | 0.790 (0.790) | 0.790–0.791 | 0.034 |
| `p2_b100_r1` | beta 100, refresh 1 | 186.22 (186.07) | 0.800 (0.799) | 0.798–0.801 | 0.028 |
| `p1_b25_on150` | onset 150→250, beta 25 | 181.78 (181.67) | 0.711 (0.744) | 0.711–0.841 | 0.041 |
| `p3_on150_b50` | onset, beta 50 | 181.05 (180.91) | 0.774 (0.795) | 0.773–0.830 | 0.032 |
| `p3_on150_b50_r1` | onset, beta 50, refresh 1 | 180.95 (180.81) | 0.800 (0.800) | 0.7995–0.8002 | 0.032 |
| `p3_on250_b50` | onset 250→350 | 180.68 (180.53) | **0.844** (0.802) | 0.755–0.844 | 0.030 |
| **`p4_on150_b100_r1`** | **onset, beta 100, refresh 1** | **180.77 (180.61)** | **0.800 (0.800)** | 0.7998–0.8002 | 0.029 |
| `p4_on100_b100_r1` | onset 100→200 | 181.40 (181.14) | 0.800 (0.800) | 0.7995–0.8005 | 0.030 |
| `p2_corner` | `corner` print base | 179.20 (179.19) | 0.800 (0.794) | 0.7998–0.800 | 0.029 |
| `p2_edge` | `edge` print base | 191.23 (191.05) | 0.800 (0.800) | 0.799–0.801 | 0.024 |

## What to be careful about

- **Run-to-run spread is about ±0.3 in compliance** (`p2_rep` against `p1_b50_r1`, same
  settings). Differences below that are noise; the ~5.5 the onset buys is not.
- **A low `hotspot_beta` stalls MMA in the grey phase.** At beta 4 and 12 the subsolver
  hit its Newton cap and iterations slowed from 0.7 s to 8-20 s, so both runs were
  stopped. It is the dilution (`n_eff` in the thousands), not sharpness, that hurts.
- **Delay the onset too far and the run ends infeasible**: `p3_on250_b50` ends at 0.844
  because the row turns on at 250-350 and never settles.
- Two tests (`test_cli_prints_full_objective_and_post_update_volume`,
  `test_enable_continuity_false_drops_the_continuity_constraint_row`) fail identically on
  unmodified `master`; they are unrelated to this work.
