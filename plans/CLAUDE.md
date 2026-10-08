This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- laplace_s03_schedule.md
    - Variant 2 (`stto_laplace`) on S 0.3 with the settled `stto` schedule: a separate
      move limit for the time-field variables, calibrated on the realized time steps.
- length_scale_options.md
    - Deferred: ways to give `stto` a minimum feature size, against the thin necks the
      hotspot constraint adds. Options and their costs only; the user wants other
      avenues tried first.
- print_order_credit.md
    - Replaces #190's front offset: the hotspot numerator gives a neighbor printed at
      the same time less credit (a tie credit under a `rouf` continuation, or a cooling
      curve), over `master`'s delay-free reference. Overnight tuning and validation.
- s03_material_behind_front.md
    - Why `stto` prints a flat time basin on the S 0.3 top bar that the hotspot misses
      (soft print-order sigmoid), and a hotspot that only credits material at least δ
      behind the front, at the part's mean gradient. Includes a plain-TO baseline.
- virtual_heat_tuning.md
    - Schedule tuning and the validation matrix for `stto_heat.py` and
      `stto_laplace.py`, whose time field solves a virtual heat equation (design in
      `archive/virtual_heat_timefield.md`). Variant 1 is set aside; ask the user often.
Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
