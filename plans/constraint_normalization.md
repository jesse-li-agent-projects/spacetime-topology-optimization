# Constraint normalization

Not started: this states the problem only.

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
