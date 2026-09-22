# An angular weight for the conductivity stencil

## The problem

The hotspot term's neighbor stencil (`conductivity.neighbor_weights`) is purely radial.
Das2023 Ch. 3 weights a neighbor by radial *times* angular, the angular factor favoring
the direction opposite the build direction, because heat leaves through already-printed
material rather than sideways within a layer. Ch. 4 dropped the angular factor when the
method moved to space-time topology optimization: once deposition order is a design
variable there is no a-priori build direction to measure an angle against. The port
inherited that, and `conductivity.py`'s module docstring records it as a known future
direction.

The direction is not unavailable, though -- it is `grad t`, which the run already has.
This plan restores an angular factor computed from the local time-field gradient.

## Formulation

For element `i` with time-field gradient `g = grad t_i` (per unit length, the
`timefield._unit_length` convention), and a neighbor at offset `d` with unit offset
`dhat`:

```
w       = w_radial(|d|) * w_angular
w_angular = exp( -kappa * (g . dhat + |g|) / (|g| + g0) )
```

`w_angular` is a **von Mises lobe**: it is `1` when the neighbor lies straight behind the
print front (`dhat = -g/|g|`) and decays as the neighbor rotates toward, and then past,
the print direction.

Why this form rather than Das's `w_a = arctan(dy/dx)/psi`:

- **No branch cut and no kink.** Das's ramp is C0 at the in-layer direction and its
  `arctan` has a branch cut in exactly the same place, which is where sensitivities would
  be worst. This lobe is analytic in `g` and `d` everywhere `|g| > 0`.
- **`kappa = 0` is exactly the present behavior**, bit for bit, so the continuation has a
  start point that needs no separate justification and the reduction is a regression test.
- **The `1/|g|` of normalizing the direction cancels.** Written as above there is no
  division by the gradient magnitude anywhere. As `|g| -> 0` the exponent goes to `0` and
  the lobe goes to exactly isotropic -- which is the right answer where there is no
  layering to define a direction, and it is exact rather than an epsilon-guarded
  approximation. The only remaining non-smooth point is `|g|` itself at the origin, which
  the existing `timefield._GRAD_EPS` convention covers.

`kappa` sets the lobe width and is the continuation parameter. `g0` sets how much
layering must exist before a direction is believed, and is set relative to the domain's
median `|grad t|` so it carries across meshes and geometries rather than needing a
per-geometry value.

**`kappa = 2.37`** matches Das's ramp at half maximum (both reach half weight at 45
degrees from the build-opposite direction). At that width the forward hemisphere keeps
about 9% weight instead of Das's exact zero. That is deliberate: a wide lobe is the goal,
not a delta, and the residual tail is what buys the smoothness. `plot/angular_lobes.png`
(from `angular_lobes.tmp.py`) is the shape comparison.

The self-pair (`e1 == e2`, `d = 0`) has no direction; it takes `w_angular = 1`, the
"no direction, no penalty" rule.

It cancels between numerator and denominator on *uniform* material, but not in general:
the origin contributes its full weight to the numerator against `sigmoid(0) = 0.5` in the
denominator, a local ratio of 2, and it is the one offset the lobe never suppresses. So
it sets a floor on how far a narrow lobe can pull the saturated `K_est` down -- dropping
it takes the maximum from 1.040 to 1.016 at `kappa = 10`, `|grad t| = 0.7`. Giving the
origin the in-layer weight `exp(-kappa_eff)` instead would remove that floor and is
arguably the better statement of where the origin sits, but nothing yet depends on it.

## Normalization: `half_stencil` becomes directional

`Normalization.HALF_STENCIL`'s divisor is a constant, `half_stencil_weight(rmin_cond)`:
the stencil weight an infinite uniform layer schedule would have already deposited
around any element. It is exactly half the untruncated stencil total, by a pairing
argument that needs the radial weight to be symmetric under `d -> -d`.

**The angular factor destroys that symmetry**, so the constant is no longer the right
divisor -- measured, it leaves uniform material reading a severity of 0.41 instead of 0
(see Pre-check results). The divisor must therefore be computed per element, from the
same lobe, against the same reference:

> `denom_i` = the stencil sum `sum_offsets w_radial * w_angular(g_i) * sigmoid(-rouf * g_i . d)`,
> i.e. the same sum the numerator computes, evaluated on a synthetic reference of full
> density whose time field is exactly linear with element `i`'s own gradient.

`K_est = 1` then means "as well shielded as an ideal uniform layered fill printing in
this direction at this layer thickness", which is what `half_stencil` already meant --
this generalizes its physical anchor rather than adding a parallel mechanism.

Three properties that make this the design rather than a patch:

- At `kappa = 0` it reduces to `half_stencil_weight` **exactly, for every direction**, by
  the same pairing argument (`sigmoid(z) + sigmoid(-z) == 1`, equal radial weights). So
  the merge is behavior-preserving at the continuation start, and that is a unit test.
- The sum runs over the **untruncated** offset list, not the in-mesh COO pairs. That is
  what preserves `half_stencil`'s defining property: leaving the mesh costs exactly what
  an equal-sized internal void costs.
- It is **attached to autograd**, not detached. A detached divisor normalizes the value
  but leaves the discretization artefact in the sensitivity, which is the thing being
  removed. There is no gaming risk: the reference is what the severity is defined
  against, so a gradient that moves the field toward ideal layering is the intended
  behavior.

`Normalization.NEIGHBORHOOD` does not get an angular factor. It is slated for
deprecation over the correctness issues in its own docstring, and `kappa > 0` under it
is a combination with no defensible meaning.

## Where the build direction comes from

A new per-element central-difference gradient *vector* in `timefield.py`, a sibling of
`_central_difference_gradient` (which returns magnitudes over fully-solid stencils only
and is a diagnostic).

Central differences, not `_q4_gauss_gradient`, for two reasons: the angular weight needs
one vector per element rather than one per Gauss point, and central differencing is blind
to the one-element sawtooth. That blindness is why `_central_difference_gradient`'s
docstring forbids optimizing it, and it is exactly what is wanted here -- the lobe must
not chatter with a padding mode.

One-sided differences at the mesh border. The field over void is used as-is rather than
masked: `void_extension: harmonic` already makes it smooth, and every element that
carries severity needs a direction. If that proves to misbehave near free surfaces, the
fallback is to extend the solid-region gradient outward, but do not do that speculatively.

## Ruled out -- do not re-investigate

- **Dithering the stencil weights.** Specifically, this would normalize the weights by a constant to account for the 0.41 offset, then use random dithering on every iteration to smooth over an otherwise rough angular objective. Ruled out because of a lack of benefit over the per-cell normalization approach, which cancels out this error exactly, and is relatively inexpensive (measured at ~1% of an `stto` step, where the FEM solve dominates, and ~9% of a `seqopt` step, which has no FEM -- see Cost below).

  Two further findings, should anyone revisit it: the aggregation is a smooth *maximum*,
  which rectifies incoherent noise into a positive bias that grows with element count
  (measured +0.008 on only 49 interior elements), destroying the cancellation that makes
  uncorrelated error preferable in the first place; and a fixed-seed dither is frozen, so
  the optimizer sees a deterministic rough landscape rather than noise it can average
  over. Resampling per iteration would fix the second, at the cost of a stochastic
  objective -- which breaks MMA's subproblem, the hotspot calibration refresh, and run
  reproducibility.
- **Das's `arctan` ramp, or any hard-zero hemisphere.** Branch cut and kink at the
  in-layer direction; and a hard-zero variant cannot reach an isotropic limit, so the
  continuation would need a second blend parameter.
- **A detached reference divisor**, and the config flag that would have selected it. See
  above.
- **Keeping `half_stencil` and adding `directed_stencil` as a second member.** The
  constant divisor is now known to be wrong for any `kappa > 0`, so that member would be
  correct only at one value of another setting.
- **Worrying about the sigmoid print-order mask as an independent artefact source.** It
  does not ripple on its own; see Pre-check result 1.

## Pre-check results (already run, do not repeat)

`angular_ripple.tmp.py`, 29x29 fully solid patch, exactly linear time field, direction
swept over 3600 angles, settings from `output/stto_lse_default/config.json`
(`rmin_cond=12`, `q=3`, `rouf=100`, `r=0.05`, 437 offsets, `half_stencil_weight=75.3734`),
at the run's own `|grad t|` and `kappa = 3.0`.

| variant | mean severity | peak-to-peak | period |
|---|---|---|---|
| radial (today) | 0 | 5e-15 | -- |
| angular, constant divisor | 0.411 | 2.2e-04 | 90 deg |
| angular, directional divisor | 0 | 5e-15 | -- |
| angular, constant divisor + dither (0.1) | 0.411 | 8.0e-03 | none (broadband) |

1. **The present radial stencil has exactly zero angular ripple.** The `sigmoid(z) +
   sigmoid(-z) == 1` pairing makes the point-sampled mask cancel at every direction. Any
   ripple is introduced by the angular factor breaking that pairing, not by the mask.
2. **The constant divisor's offset, not its ripple, is the disqualifying problem**: 0.41
   against a `Tcr` target of 0.8, roughly 2000x the ripple.
3. **The directional divisor cancels both**, to the float64 noise floor. Caveat stated
   honestly: on uniform material with an exactly linear field this is near-tautological,
   since the divisor is built from the same sum. What it establishes is that the
   construction is self-consistent under discrete point sampling of both the lobe and the
   mask, which was the open question. It says nothing about voids or mesh boundaries.

`kappa = 0` reproducing the radial variant was asserted numerically: max |dS| over the
whole angle sweep = 0.000e+00.

**Layer-thickness sweep.** The same test at `kappa = 2.3666` over a 100x range of
`|grad t|`, reported by the width the print-order sigmoid takes to switch (the quantity
that sets the artefact; `1/|grad t|` is the build height, not a deposited layer):

| sigmoid width | constant divisor | directional divisor |
|---|---|---|
| 15 el | ripple 8.2e-07, offset 0.204 | 5e-15 |
| 5 el | ripple 1.4e-05, offset 0.290 | 5e-15 |
| 1.5 el (reference run) | ripple 1.3e-04, offset 0.359 | 5e-15 |
| 0.5 el | ripple 4.6e-04, offset 0.411 | 4e-15 |
| 0.15 el | ripple 1.0e-03, offset 0.507 | 5e-15 |

The directional divisor cancels exactly everywhere, including where the mask switches
inside a single element -- the maximally-quantized case. The constant divisor degrades in
both ripple and offset as layering sharpens.

**Cost, measured.** The directional divisor is a second `O(nel * stencil)` sum, attached,
so it costs one more forward and backward of the same shape:

- `stto` 180x60 (`benchmarks/profile_step.py`): step 406 ms, conductivity forward 4.1 ms
  = **1.0%**. The FEM solve is 66%. Irrelevant.
- `seqopt` c-shape 120x100 (no FEM): step ~210 ms; stencil forward 4.8 ms (2.4%); stencil
  forward+backward standalone 18 ms = **~9% of a step**.

So the ceiling is roughly +15-25% of a `seqopt` iteration once the lobe's own per-pair
`exp` and dot product are counted, and nothing measurable in `stto`. MMA and the filter
chain dominate `seqopt`, not the hotspot term.

## Phases

Each phase is its own PR.

### Phase 1 -- the lobe and the directional divisor

`conductivity.py`:

- Keep `neighbor_weights` returning the radial triplets; add the per-pair offset vectors
  alongside, since the lobe needs `dhat` per pair.
- `angular_weight(g_per_element, offsets, kappa, g0)` -> per-pair factor. One small
  function, so a different lobe is a one-place edit.
- `half_stencil_weight` becomes the per-element directional sum described above, with the
  cached constant kept as the `kappa == 0` fast path. `constant_denominator` and
  `_conductivity_core`'s `denom` parameter change shape from scalar to per-element.
- `_stencil_offsets` stays the single source of truth for which offsets are in the
  stencil; the reference sum reads it directly (untruncated).

`timefield.py`: the per-element central-difference gradient vector.

`run_config.py`: `hotspot_kappa` (a `Scheduled`, like `hotspot_beta`) and `hotspot_g0` on
both `RunConfig` and `SeqRunConfig`; `0` and the median-relative default in
`configs/default.json` and `configs/seq_default.json`. A scheduled `hotspot_kappa` must
also stale the hotspot calibration, the way scheduled `hotspot_beta`/`rouf` already do
(`hotspot_refresh_period`'s comment).

Tests:

- `kappa = 0` reproduces the current `K_est` field exactly, on a real snapshot.
- The directional divisor equals `half_stencil_weight(rmin_cond)` at `kappa = 0` for a
  spread of gradient directions and magnitudes.
- The lobe is isotropic at `|grad t| = 0`, and monotone in the angle for `kappa > 0`.
- `infinite_base`'s captured set is unchanged: reach is radial, and the lobe is strictly
  positive, so `base_weight > 0` fires on exactly the same elements.

### Phase 2 -- diagnostics and visualization

The build-direction field is now a thing a run has, and a run that goes wrong will go
wrong in it. A quiver or hue overlay on the time-field plot in `viz.py`, plus `|grad t|`
percentiles in the iteration record, so a `kappa` continuation can be read after the fact
rather than re-run.

### Phase 3 -- tune the continuation

`kappa` continuation `0 -> 2.37`, `Tcr` held at its current schedule, so the effect of
`kappa` alone is visible. Judge against a matched `kappa = 0` control run, not against
the best previous run.

Expect the calibrated severity level to move as `kappa` rises, which will interact with
the `Tcr` onset (currently 150-250) and with `hotspot_beta`. Retuning those is a second
pass, after the shape of the `kappa` effect is known. Note that the tuned `stto` settings
recorded so far are `kappa = 0` settings and are not assumed to survive.

Open questions this phase answers, not Phase 1:

- Does `g0` want to be a fraction of the median `|grad t|`, and which fraction?
- Does the continuation want to start before or after the `Tcr` onset?
- Does the lobe help or hurt the layer-smoothness objective that motivated `seqopt`'s
  current line of work?

#### Phase 3 results -- do not re-run

16 `seqopt` runs, 500 iterations, shipped `seq_default` weights, `Tcr` untouched. Ten
`kappa` arms on the c-shape plus a matched `kappa = 0` control, then control /
`kappa = 2.37` / `kappa = 5` repeated on the l-shape and the overhang bracket, neither of
which anything was tuned on. Run-to-run reproducibility was measured first, by running
one config twice: the objective agrees to 1.3e-11 relative (CUDA atomics), so anything
above ~1e-6 is real.

**A run's own `hotspot` number, and therefore its `f`, is not comparable across arms.**
`kappa` changes what `K_est` is measured against, so each arm reports a different
quantity and each is best at its own. Every design was re-scored under one fixed measure
instead (true maximum severity at `kappa = 0` and at `kappa = 2.37`), and the decision
rests on the diagnostics rather than on either severity.

**1. `g0` wants to be small, and is not a tuning axis.** Converged `|grad t|` is tight --
p5..p95 within +/-10% of the median on every geometry -- so a `g0` near the median
multiplies `kappa` by a near-constant factor everywhere, which is a disguised second
`kappa` rather than a floor. At `g0 = p50/10` the factor is 0.91 and near-uniform.
Measured at `kappa = 2.37`: `g0` of p50/40, p50/10, p50/2.5 gives `true_cv` 0.0535,
0.0545, 0.0577. Anything at or below p50/10 is within 2%; the near-median value is
measurably worse. **Set it low and stop looking at it.**

**2. The continuation is unnecessary here, and later is monotonically worse.** At
`kappa = 2.37` on the c-shape: constant from iteration 0 gives `true_cv` 0.0545, ramps
over [0,250] 0.0549, [100,350] 0.0558, [250,450] 0.0580. The plan's motivating worry --
that an early lobe locks in a print direction -- is **not observed**. Caveat that limits
how far this generalizes: `seqopt` holds the geometry fixed, so `t` is the only fluid
thing; the worry was about a fluid *design*, which only `stto` has. Do not carry this
conclusion to `stto` without re-testing it there.

**3. The lobe buys layer uniformity and pays for it in raw sawtooth.** Consistent on all
three geometries, control-relative:

| geometry | `kappa` | `true_cv` | `sawtooth_raw` |
|---|---|---|---|
| c-shape | 1 | +8.5% | +64% |
| c-shape | 2.37 | -1.1% | +80% |
| c-shape | 5 | **-20.8%** | +77% |
| l-shape | 2.37 | -1.8% | +143% |
| l-shape | 5 | **-15.8%** | +127% |
| overhang bracket | 2.37 | -0.5% | +94% |
| overhang bracket | 5 | **-18.1%** | +90% |

`true_cv` improves monotonically with `kappa` and only becomes worthwhile at the narrow
end; `kappa = 1` is *worse* than no lobe at all, and Das's own half-width (2.37) is
within noise of it. Meanwhile `sawtooth_raw` roughly doubles at every `kappa > 0` while
the filtered `sawtooth` barely moves -- the filter is absorbing a corrugation that grew,
which `seqopt.py`'s own docstring calls a failure rather than a fix.

**4. That sawtooth doubling is small in absolute terms.** The ratio is what the table
shows, but the level is 2.9% -> 5.2% of a layer thickness on the c-shape. The corrugation
that motivated PR #94 and the multi-pronged plan was **21% of a layer domain-wide**
(`plans/archive/seqopt_sawtooth_multipronged.md`), so every arm here sits at a quarter of
the known-bad level or below, and the *filtered* `sawtooth` does not move at all
(0.0062 -> 0.0061). A doubling of something four times below the threshold is not a
reason to refuse the term.

The extra corrugation is nonetheless real and distributed, not a boundary artefact:
trimming 10 columns off each edge leaves the control at 0.0151 and `kappa = 5` at 0.0198
on the c-shape, and the ratio survives on all three geometries.

**Decision: the lobe is on by default, at `kappa = 2.3666`.** The angular weight is a
physical-accuracy correction, not a tuning knob -- heat does leave through
already-printed material rather than sideways -- so the question was how to carry it, not
whether. Constant, no continuation, and `g0` an order of magnitude below the run's median
`|grad t|`.

`2.3666` is Das2023's own half-width, which is the only value here with a claim to
physical meaning; everything else on the axis is a number that happened to score well.
The `seqopt` sweep above preferred 5, where 2.37 was within noise of no lobe at all --
but that finding did not survive the move to `stto`, where 2.37 bought 14.3% of `true_cv`
(Phase 3b). `stto` is the code that matters long-term, and a physically grounded setting
beats a fitted one when the two disagree by this little. If a design turns out not to
change under the lobe at this width, that is a coincidence to note rather than evidence
of a problem.

**The one open concern is budget, not level.** `sawtooth_raw` had not converged at
iteration 500 in *any* arm -- every trace is still climbing, control included -- so the
2x is a ratio between two growing curves and says nothing about where either settles. The
test that matters is a long run (~1500 iterations) checking whether the gap keeps widening
or the arms converge. Until that is done, treat `sawtooth_raw` as a watched diagnostic on
any run with `kappa > 0`.

If it does keep widening, the mechanism to look at first is the same quadrature blindness
as PR #94: the lobe reads a per-element central-difference gradient, exactly the operator
that cannot see the one-element mode, so the fix would be in how the direction is sampled
rather than in `kappa`.

Untested and worth knowing: `kappa > 5`, since `true_cv` had not turned around.

#### Phase 3b results -- `stto`, one matched pair

Everything above is `seqopt`, which holds the geometry fixed. `stto`, where density is a
design variable too, was then run as a matched pair: `configs/default.json` untouched
(800 iterations, 180x60, `Tcr` 5.0 -> 0.8 over 150-250, `hotspot_beta` 100, refresh 1),
`kappa = 0` against a constant `kappa = 2.37`, `g0 = 0.07`. The lobe is the only
difference. ~8 min per run.

| metric | `kappa = 0` | `kappa = 2.37` | change |
|---|---|---|---|
| compliance | 180.769 | 180.707 | -0.0% |
| severity re-scored at `kappa = 0` | 0.800 | 0.817 | +2.2% |
| severity re-scored at `kappa = 2.37` | 0.950 | 0.800 | **-15.8%** |
| `true_cv` | 0.0550 | 0.0471 | **-14.3%** |
| `sawtooth_raw` | 0.0072 | 0.0084 | +16.6% |
| grey fraction | 0.0065 | 0.0058 | -10.0% |

**The design changes, modestly.** Binarized density disagrees on 1.20% of elements --
boundary-pixel moves, not a topology change; the members stay where they were. The time
field moves more (RMS 0.0090 over shared solid), concentrated in the right-hand members,
whose print sequence is visibly re-timed.

**The compliance is free.** 180.707 against 180.769, and both runs sit exactly on
`Tcr = 0.8` (`true_max` 0.80001 for both), so both are constraint-limited and the lobe
cost nothing structurally.

**Each design satisfies its own measure and violates the other's, asymmetrically.** The
control reads 0.950 under the angular measure -- a 19% violation of `Tcr` -- while the
angular design reads 0.817 under the radial one, 2% over. So the radial-only design is
substantially overheating by the more physically accurate measure, while the angular
design is nearly compliant under both. That is the argument for the lobe, independent of
any uniformity gain.

**`stto` is not `seqopt` here.** `kappa = 2.37` bought 14.3% of `true_cv` in `stto`,
where the same value was within noise in `seqopt` and only `kappa = 5` helped. Do not
carry `seqopt`'s tuning across.

#### Notes for a settings-tuning agent

Read these before designing a sweep; each cost a run or an error to learn.

- **Never compare `hotspot`, `tru_max` or `f` across `kappa` arms.** `kappa` changes what
  `K_est` is measured against, so each arm reports a different quantity and is trivially
  best at its own. Re-score every finished design under one fixed measure instead --
  `eval_kappa.tmp.py` (seqopt) and `eval_stto.tmp.py` do this, reporting the *true*
  maximum severity rather than the calibrated smooth maximum, since the calibration is run
  state and not a property of the design.
- **Always budget a matched `kappa = 0` control into the sweep matrix**, and compare
  against it rather than against the best previous run.
- **Judge on `true_cv`, not `uniformity`.** The optimized metric is blind to the
  transverse sawtooth; the gap between them is what that mode is padding. Watch
  `sawtooth_raw` alongside, and read it in absolute terms -- 21% of a layer is the known
  defect level (`plans/archive/seqopt_sawtooth_multipronged.md`), and everything measured
  so far is at or below a quarter of that.
- **`g0` is not a tuning axis.** Set it about a decade below the run's own median
  `|grad t|` (logged per iteration as `grad_p50`) and leave it. See result 1 above.
- **Constant `kappa` from iteration 0 is a fine default.** No early direction lock-in
  appeared in either code, and in `seqopt` later ramps were monotonically worse. Untested:
  whether a ramp helps in `stto` specifically, where the design is fluid early -- that is
  the one place the original worry could still be real.
- **`kappa` interacts with the `Tcr` onset**, currently 150-250. Both `stto` runs were
  constraint-limited end to end, so a `kappa` ramp crossing that onset is changing the
  constraint level and the measure at the same time. Separate them or expect to be unable
  to attribute the result.
- **A scheduled `kappa` stales the hotspot calibration** and refreshes it off-cycle, as
  scheduled `rouf`/`hotspot_beta` do. Expected, but it makes the constraint trace step.
- **Cost and determinism.** `seqopt` ~4 min/run at 500 iterations, `stto` ~8 min at 800.
  Reproducibility is ~1e-11 relative on the objective (CUDA atomics), so differences above
  ~1e-6 are real; anything smaller is noise.
- **Open questions, in the order worth answering**: does `sawtooth_raw` keep diverging
  past ~1500 iterations (it had not converged at 500 in any arm, control included); does a
  `kappa` ramp help in `stto`; and is there anything above `kappa = 5`, where `true_cv`
  had not yet turned around.

## Risks

- **Self-reinforcing direction.** The lobe rewards shielding from the direction the field
  currently prints in, which may lock in an early direction. That is what the `kappa`
  continuation exists to prevent, but it is the failure mode to watch for first, and the
  Phase 2 direction plot is how it gets seen. *Not observed so far* -- constant `kappa`
  from iteration 0 matched or beat every ramp in `seqopt` and behaved in `stto` -- but
  `stto` has had only one matched pair, so this is unconfirmed rather than closed.
- **Free surfaces.** The pre-check covers uniform material only. The divisor's
  cancellation near voids and mesh boundaries is untested, and that is where the hotspot
  term is supposed to do its work.
*(Sharp layering was a third risk and is now closed -- see the layer-thickness sweep in
Pre-check results.)*
