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
- variant2_handoff.md
    - Start here for Variant 2: where its tuning stands, the user's decisions of
      2026-10-18, the overnight batches, and pointers to the runs and transcripts.
- scale_invariant_hotspot.md
    - A hotspot severity invariant to scaling every density (`hotspot_shielding`), against
      grey bars that stay grey until forced solid; batch 14 tests it on C1. Tracks the
      user's gradient-pressure requirements (R0-R3) for any hotspot reformulation.
- length_scale_options.md
    - Deferred: ways to give `stto` a minimum feature size, against the thin necks the
      hotspot constraint adds. Options and their costs only; the user wants other
      avenues tried first.
- nucleation.md
    - Open: the hotspot pulls light, ungrounded material in as "support" under hot
      regions. Status, the pressures wanted per situation, and whether they conflict.
      Start here to resume the problem.
- virtual_heat_tuning.md
    - Schedule tuning and the validation matrix for `stto_heat.py` and
      `stto_laplace.py`, whose time field solves a virtual heat equation (design in
      `archive/virtual_heat_timefield.md`). Variant 1 is set aside; ask the user often.
Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
