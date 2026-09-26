# Handoff: tune the tool-radius constraint (Phase 5)

Read `plans/curvature_constraint.md` first. This file is the brief for its Phase 5, and it
corrects that plan where the two disagree.

## Task

Find a `tool_radius` continuation that reaches **`R_target = 2.5` elements** on the
`stto_k2p37` problem, and measure what it costs. 2.5 is a guess, not a physical tool
size: do not tune toward it as if it were a hard requirement, and report if a smaller
target is much cheaper.

## Corrections to the plan

- **"Any `R_target >= 1` element is active from the start" is wrong for a new run.**
  It is true for the Phase 4 final designs only. A new run starts from a smooth time
  field, which probably has no creases, so the row is probably inactive at first and
  becomes active as creases form. What matters is whether the ramp reaches `R` before
  the creases form. Measure `admissible_tool_radius` of the `R = 0` control over the
  iterations to see when they form.
- **The violating crease is a ridge of `t`, not the trough.** In Phase 4 (`stto_k2p37`,
  column 75) the trough at row 48 is a convex wedge tip (blue in the plot, harmless). The
  ridge at row 52, where two fronts meet in a V-shaped notch, is concave (red) and is
  the worst element. Both come from one-row spikes in the raw `t`, which the cone time
  filter keeps as sharp apexes.
- **Intuition for the numbers.** Curvature is the turning angle per unit length, and a
  crease puts its whole turn into about one element. For `R = 2.5` to be satisfied, an
  iso-line must turn less than 0.4 rad (23 deg) per element. The Phase 4 creases turn
  about 85 deg in one element.

## Setup

- **Config.** Old run records do not load, because they have no `tool_radius` and
  `curvature_beta`. Copy `output/stto_k2p37/config.json` and add the two fields, for
  example:

  ```json
  "tool_radius": {"points": [[0, 0.0], [150, 0.0], [250, 2.5]], "mode": "linear"},
  "curvature_beta": 100.0
  ```

  To plot an old run with `python -m sttopt.viz`, add
  `"tool_radius": 0.0, "curvature_beta": 100.0` to its `config.json` first.
- **Control.** Run a matched `tool_radius: 0.0` control in the same sweep, with the same
  code and settings. Do not reuse `output/stto_k2p37` as the control: it was made by
  older code. Compare each arm with its matched control, never with the best previous
  run.
- **Compute.** One run takes about 8 minutes on the GPU (800 iterations). Use `gpu-exec`
  and the `long-running-runs` skill. Run only one GPU job at a time. When you start a
  run from a snapshot, log `sttopt.__file__` so that you know which code it imported.

## Sweep

Start small and change one thing at a time. Suggested first matrix (5 runs, about 40
minutes):

| arm | `tool_radius` | notes |
|---|---|---|
| control | `0.0` | matched control |
| early | ramp 0 -> 2.5 over iterations 0-100 | constraint before creases form |
| mid | ramp 0 -> 2.5 over 150-250 | same window as the `Tcr` onset |
| late | ramp 0 -> 2.5 over 300-400 | after the `Tcr` onset |
| step | constant `2.5` | no continuation |

Keep `curvature_beta` at 100 for the first matrix: the severity is 1 on the bound, the
same scale as the hotspot row with `hotspot_beta = 100`. Tune it only after an onset is
chosen.

## What to report for each arm

From `iterations.jsonl` and `final_design.npz`:

- compliance `obj`, `true_max` (hotspot), `uniformity`, `roughness`
- final `admissible_tool_radius`, and the last `g` entry (the curvature row)
- the share of solid elements (`xPhys > 0.5`) with `-kappa * 2.5 > 1`, from
  `timefield.iso_curvature(tPhys, xPhys) / timefield.unit_length(tPhys)`, and the
  90/95/99th percentiles of the concave curvature. `admissible_tool_radius` is a hard
  maximum over one element, so it alone can hide a large improvement.
- `plot/<tag>/iso_curvature.png`, and the time-field plots

Reference values from the old `stto_k2p37` run: `obj` 180.71, `true_max` 0.800,
`uniformity` 0.0351, `roughness` 0.0579, admissible `R` 0.57 elements.

## Questions to answer

- Which onset reaches `R = 2.5` at the smallest cost in compliance and hotspot severity?
- Do the curvature row and the hotspot row compete for the same freedom in `t`?
- Does the constraint remove the creases by smoothing the raw `t`, or does it move them
  into void or onto the mesh border, where they are not measured? Look at the raw `t`
  (`final_design.npz["t"]`), not only at `tPhys`.

## Known gaps and traps

- **The curvature calibration offset is not logged** (the plan lists it, but Phase 3
  did not add it). If you need it to tune `curvature_beta`, add
  `problem.curvature.calibration` to the diagnostics in `stto.step`.
- **`admissible_tool_radius` is `inf`** when nothing is concave. `iterations.jsonl`
  writes it as `Infinity`, which Python's `json` reads but strict JSON parsers do not.
- **Border elements are not measured.** `iso_curvature` returns interior elements only.
  A crease on the mesh border is not constrained.
- **Do not ignore warnings.** MMA `subsolv` warnings from a hand-built `State` past
  iteration 1 are a test artefact (zero asymptotes), not a problem with the row. From a
  real run they are a finding: report them.
