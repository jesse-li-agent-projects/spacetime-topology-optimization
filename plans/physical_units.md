# Physical units

Goal: refining the mesh of one physical part leaves the optimization's behaviour the
same, with no hand-rescaling of settings. `tests/test_resolution_scaling.py` checks it:
each step below turns its `xfail`s into passes.

The rule each step follows: state a term as an area mean (or a ratio of means) over the
domain, and a length in metres. Relative roughness is the one deliberate exception; it
handles an element-scale filtering issue, so its per-element "layer" is correct.

## Steps

1. **Config lengths in metres** (PR #152, done). `RunConfig` has `width_m`/`height_m`
   and `nelx`; a geometry file states its size. The mesh-level code stays in elements.
2. **`hotspot_g0` and iso-line curvature in SI** (1/m). `g0` was per
   `timefield.unit_length`, the square root of the domain area, so padding a geometry
   with void changed what it meant. `unit_length` stays as the internal gradient unit:
   it keeps gradients of order 1, which `GRAD_EPS`/`NORMAL_EPS` are set against, at
   any resolution or part size. Settings and diagnostics convert through
   `unit_length_m`.
3. **Mean-form constraint rows.** The continuity row becomes `mean(dev**2) / tol - 1`
   (its `2 * nel` weight grows with refinement), and the start-point rows become one
   row on the base's mean print time (one row per base element now). The continuity
   row still does not converge until step 6.
4. **Tip traction** over a physical length on the right edge, from the bottom-right
   corner up, 1 mm by default, with exact integration of the nodal shape functions. A
   point load's compliance grows as `log(1/h)`.
5. **Measure MMA's `raa0`** against the per-variable gradient scale at 180x60 and
   360x120. Mean-form terms have gradients of `O(1/n)` per variable, and `raa0` is
   absolute. Change it (`raa0 / n`) only if the measurement shows it matters.
6. **`lrmin` redesign.** The continuity window is `ceil(lrmin) - 1` elements, square
   and unweighted, so it jumps at whole numbers and its physical size depends on the
   resolution. Needed, with step 3, for the continuity row to converge.

## Left as they are

- The calibrated `LogSumExp` bias: the calibration absorbs the `log(n) / beta`
  constant, so only the uncalibrated value depends on the element count.
- The pointwise maxima (hotspot severity, iso-line curvature) at re-entrant corners and
  domain boundaries converge only as fast as the sampling approaches them; element-scale
  noise has curvature `~1/h`. Both are properties of a maximum, not of a unit.
