# Constraint normalization

Closed without changes: every active constraint is already of order 1 (see
"Conclusion").

MMA's constraint handling is not scale-free. `mma_c` penalizes each constraint's
violation linearly, and the multipliers, the elastic cap and `raa0`'s damping all act
on the constraint's value as given. A constraint of the form `value / bound - 1`
reads as a relative violation whatever the bound. One of the form `value - bound`
does not: its effective weight changes with the bound, and so also through a run
where the bound is scheduled.

The stto constraints use both forms:

| constraint | form |
|---|---|
| global volume fraction | `sum / (nel * volfrac) - 1` |
| time-field continuity | `mean deviation**2 / tolerance - 1` |
| hotspot | `smooth max / Tcr - 1` |
| tool radius | `smooth max of R * concave curvature - 1` |
| gradient smoothness | `smooth max of ||H|| * length / median - 1` |
| gradient floor | `smooth max - fraction` (scheduled `fraction`) |
| print start | `mean t over the base - 1e-9` |
| stage volume, upper | `deposited / (nel * volfrac) - t_stage` |
| stage volume, lower | `-upper - 1e-5` |

Normalizing any of these changes MMA's steps, so it changes results and the golden
fixtures, and a tuned schedule (`configs/continuation.md`) may need retuning.

## Conclusion

The goal is constraints of order 1, not the `value / bound - 1` form as such.

- Gradient floor: dividing by the median makes it invariant to the scale of `|grad t|`.
  Its value is of order `fraction`, which is meant to stay of order 1 (0.5, constant,
  in `configs/default.json`). Revisit only if `fraction` is set far below that.
- Stage volume: both terms are fractions of the part's volume, so of order 1. The
  bounds are also deprecated, being incompatible with uniform layer heights.
- Print start: `t` is in [0, 1], so the value is of order 1. The bound is about 0, so a
  ratio form does not apply.
