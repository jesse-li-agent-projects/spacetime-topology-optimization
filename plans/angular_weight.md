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

The self-pair (`e1 == e2`, `d = 0`) has no direction; it takes `w_angular = 1`. The
choice is free -- it cancels between numerator and denominator -- but "no direction, no
penalty" is the statement that does not need a caveat.

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

## Risks

- **Self-reinforcing direction.** The lobe rewards shielding from the direction the field
  currently prints in, which may lock in an early direction. That is what the `kappa`
  continuation exists to prevent, but it is the failure mode to watch for first, and the
  Phase 2 direction plot is how it gets seen.
- **Free surfaces.** The pre-check covers uniform material only. The divisor's
  cancellation near voids and mesh boundaries is untested, and that is where the hotspot
  term is supposed to do its work.
*(Sharp layering was a third risk and is now closed -- see the layer-thickness sweep in
Pre-check results.)*
