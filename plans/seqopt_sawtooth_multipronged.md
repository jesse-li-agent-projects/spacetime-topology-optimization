# Removing the gradient-padding sawtooth from `seqopt`

## The problem

Optimizing `seqopt`'s layer-uniformity term alone on the c-shape produces a visibly
jagged time field. The benchmark run (`output/seq_c_shape_gradonly3`, 800 iterations,
`hotspot_weight=0`, `enable_continuity=false`, `nStage=0`) reports a uniformity CV of
**0.0273**, but its true layer non-uniformity is **~0.11**. The gap is a column-
alternating corrugation in `t` of amplitude ~2e-3 against a 5.6e-3 layer thickness --
roughly 35% of a layer.

`t` along row 100, a fully-solid row, in that run:

```
0.11444  0.11055  0.11463  0.11071  0.11482  0.11086  0.11502
```

## Why it happens

`_gradient_cv` measures the spread of `|grad t| = sqrt((dt/dx)^2 + (dt/dy)^2)`. Where the
print-direction rate falls short of the target layer thickness, a transverse gradient
makes up the deficit *in quadrature*, and a column-alternating pattern supplies transverse
gradient without transporting `t` sideways. The optimizer tunes the amplitude per location
to top thin layers up to a common thickness.

On an ideal ramp the mode is **exactly** invisible: a `(-1)^j` sawtooth puts `+/-2a` on
`dt/dx` at every Gauss point, and squaring erases the sign, leaving `|grad|` spread at
4e-17 and CV at `0.00000000` for amplitudes up to 0.1. On the real field it is not merely
invisible but *rewarded*, which is why the reported CV sits ~4x below the truth.

This is the metric's genuine preference, not a numerical artefact. PR #91's 2x2 Gauss
quadrature fixed the checkerboard and the along-print sawtooth; it cannot fix this one,
because this is the magnitude discarding a sign rather than a quadrature blindness.

## Ruled out -- do not re-investigate

- **More iterations.** The amplitude peaks at iteration 250 and then *decays*, but only
  ~10% per 550 iterations. Not counterproductive, just hopeless.
- **Reverting to central differences.** They are *exactly* blind to this mode, which is
  why they make a good independent diagnostic (below) and a terrible objective.
- **Excluding boundary-straddling cells from the metric.** The optimizer is supposed to
  fix those by moving void `t`, and it does -- straddle share of variance falls 62.6% ->
  0.5% by iteration 250. Masking discards a signal that works.
- **Initialization alone.** Restarting a pure-CV run from a converged *smooth* field
  (rw100's) regrew the sawtooth monotonically, 1.4e-5 -> 3.3e-4 over 400 iterations with
  no plateau. The smooth state is not a local minimum, so no starting point prevents this.
- **Continuation down to `rw = 0`.** Same result: the floor must be strictly positive.

The initialization is nonetheless a strong *accelerant*: from the geodesic init the
amplitude reaches 1.9e-3 by iteration 250, roughly 6x faster than from a smooth start.
And the restart reached true CV 0.0232 at 400 iterations versus 0.113 for a from-scratch
pure-CV run -- so a large part of the damage is a basin effect even though the basin is
not closed. Both prongs 1 and 2 are worth doing; neither is sufficient alone.

## Pre-check results (already run, do not repeat)

**P1, harmonic void initialization.** Replacing `init_geodesic_timefield`'s outward-growing
void pass with a harmonic extension (`grad^2 t = 0` on void, Dirichlet `t = t_solid` on the
interface, Neumann on the domain boundary):

| field | CV | interface jump, row 29/30 | straddle share of variance |
|---|---|---|---|
| geodesic | 2.21980 | 0.4108 | 99.7% |
| harmonic | 0.09682 | 0.0254 | 86.0% |

Viable. Note it does not reach the 0.036 of the fully-solid interior: a harmonic extension
makes `t` continuous across the interface but not `grad t`, so `|grad t|` still kinks
there. If that residual proves to matter, the C1 fix is a biharmonic extension matching
value *and* normal derivative.

There is a deliberate design intent being changed here. The current void pass encodes
"void prints after the solid it grows from" (see its docstring). A harmonic extension
interpolates instead, so void can land earlier than adjacent solid. That is defensible --
`seqopt`'s module docstring says `t` over void "is pinned by nothing physical" and it is a
free design variable, so for an *initialization* smoothness is the more useful property --
but it is a real change of intent, not a bug fix.

**P2, density-filter gain at the one-element modes** (interior element, full stencil):

| rmin | gain at kx = Nyquist | checkerboard |
|---|---|---|
| 1.5 | 0.301 | -0.041 |
| 2.0 | -0.041 | 0.041 |
| 3.0 | 0.048 | -0.003 |
| 4.0 | -0.0054 | 0.007 |

At rmin=4 the mode is attenuated ~200x: reproducing rw0's 2e-3 corrugation in `tPhys`
would need a 0.374-amplitude sawtooth in `t`, 37% of the whole `[0, 1]` range. Not
forbidden outright, but it would dominate the design variable and fight the move limit.
rmin=1.5 is too weak to be worth a run.

Caveats: the gain is computed for an interior element, so near the domain edge and the
solid/void boundary the `Hs` normalization differs and attenuation is weaker -- the mode
may survive in a boundary layer. Cross-void mixing is not a concern on this geometry (the
c-shape's mouth is ~50 elements against rmin=4), but the filter does pull void `t`, which
is pinned by nothing, into `tPhys` of solid within rmin of the boundary.

## Units warning

PR #94 normalized the roughness term: the objective uses `timefield.relative_roughness`
(roughness divided by the mean gradient magnitude), which is dimensionless and resolution-
invariant. **All weights below are in those normalized units**, where 0.06 leaves a 16%
wiggle, 0.18 flattens it, and the config default is 0.2. Earlier raw-`roughness` weights
from the first sweep (1/10/30/100) are ~5.6e-3 times these and must not be mixed in.
State the units in every config file written.

## Instrumentation

**The headline metric must be sawtooth-blind. The objective's own CV cannot be trusted --
that is the entire failure mode.** Use the central-difference CV, which annihilates the
mode exactly:

```
dx = (t[1:-1, 2:] - t[1:-1, :-2]) / 2
dy = (t[2:, 1:-1] - t[:-2, 1:-1]) / 2
g  = sqrt(dx**2 + dy**2)          # over elements whose 4 neighbours are all solid
true_cv = g.std() / g.mean()
```

It agreed to within 8% with an independent estimate (Q4 CV of the 1-2-1 column-filtered
field), so either is fine; use both if cheap.

Sawtooth amplitude: `mean(|t - smooth|)` over solid, where
`smooth[:, 1:-1] = (t[:, :-2] + 2*t[:, 1:-1] + t[:, 2:]) / 4`. Report it as a fraction of
the mean layer thickness, which is the number that says whether a print would show it.

Log per iteration: reported CV, true CV, `relative_roughness`, sawtooth amplitude, and the
existing MMA trust-region fields. **For prong 3, log the sawtooth amplitude in both `t` and
`tPhys`** -- the failure signature there is a smooth `tPhys` over a wildly jagged `t`.

Do **not** add a "padding pressure" diagnostic built from gradient inner products. One was
tried (directional derivative of CV along `grad relative_roughness`, deliberately
axis-free) and it reads +0.83 at rw100's converged field, predicting smoothing, when the
ground-truth restart from that exact field grew the sawtooth 24x. Both gradients are
dominated by whichever region has the largest residual, so the inner product reports on
that region rather than on the bulk mode.

## Run matrix

Common: c-shape, `hotspot_weight=0`, `enable_continuity=false`, `nStage=0`, 800
iterations, matched budget. ~8 minutes per run on the GPU.

| id | init | roughness_weight | filter | question |
|---|---|---|---|---|
| B1 | geodesic | 0 | none | baseline; **already exists** as `seq_c_shape_gradonly3` |
| B2 | geodesic | 0.2 constant | none | PR #94 default baseline |
| E1a | harmonic | 0 | none | does a clean init alone slow the mode |
| E1b | harmonic | 0.2 constant | none | does a clean init improve the regularized result |
| E2a-d | geodesic | 1.0 for it 1-300, then step to floor in {0, 0.02, 0.06, 0.18} | none | how low can the floor go |
| E3a | geodesic | 0 | rmin=2 | is padding eliminated or only damped |
| E3b | geodesic | 0 | rmin=4 | same, at the default radius |
| E4 | best of E1 | best of E2 | best of E3 | do the prongs compose |

Roughly 10 runs, ~1.5 h GPU.

E2 uses a **step**, not a smooth decay, so that regrowth is attributable to a known
iteration and the floor value is not confounded with a decay rate. The 500-iteration hold
after the step matters: regrowth in the restart run was visible by iteration 100-200 and
still climbing at 400, so a shorter hold can read as "no regrowth" purely from being too
short. E2a (floor 0) is largely answered already by the restart run and can be dropped if
GPU time is tight. If a floor works, schedule *shape* is a separate follow-up sweep, and
geometric decay is the better candidate there than cosine, since the weight spans a ~50x
range and linear-space schedules spend most of their iterations in the top decade.

Control geometry, once a winner is chosen: use `overhang_bracket`, not `l_shape`.
`l_shape` already converges to a true CV of ~1e-4 with no sawtooth at `rw=0`, so it only
tests "does no harm" -- the pathology needs a geometry whose layers cannot be uniform.

## Decision criteria

Rank by **true CV at 800 iterations**, with sawtooth amplitude as a tiebreak; ignore
reported CV except to quote the reported-vs-true gap as evidence of gaming. A prong earns
its place only if it beats B2 on true CV at equal budget -- B2 is already a decent result
(true CV ~0.031), so "removes the sawtooth" is not by itself a win.

## Traps

- **Never edit `sttopt/` while a sweep is running.** Adding a required config field
  mid-sweep kills every subsequent run in that sweep.
- `gpu-exec` kills backgrounded processes when the call returns, and aborts a call at its
  idle timeout. Run in the foreground, one or two runs per call, and arm a `Monitor` on
  the log file -- it survives a tool-call abort and is the only reliable completion signal.
- **Run the CLI with an explicit `PYTHONPATH`.** An editable install of the main checkout
  shadows a worktree, so `python3 sttopt/seqopt_cli.py` silently runs the wrong code.
- `claude` cannot write into the repo; write run output to `/tmp/claude-1001/...` (mode
  777) and copy back.
- Fixed iteration counts are not proof of convergence: one earlier run changed its CV 6%
  in a single MMA step. Check step size and oscillation before comparing finals.
- Determinism is established -- a re-run reproduced `seq_c_shape_gradonly3` bit-for-bit
  (max abs diff exactly 0.0) -- so any nonzero difference between identical configs is a
  bug, not noise.
- `viz.timefield_filled_contour_plot` at its default 30 contours puts levels 0.033 apart
  while the sawtooth is 2e-3, so it under-shows this defect as mild texture. For visual
  confirmation, crop to a solid block and use contour spacing ~0.004.
