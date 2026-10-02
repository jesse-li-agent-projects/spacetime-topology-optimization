This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- post_run_checks.md
    - A check on a saved `stto` design, binarized, that reports the margin of each
      condition it must meet: print start and support as hard checks, the constraint
      values and compliance as a report.
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
