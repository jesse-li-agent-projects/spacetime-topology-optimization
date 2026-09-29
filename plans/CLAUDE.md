This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- constraint_normalization.md
    - Not started, the problem only. Some constraints are `value / bound - 1` and some
      `value - bound`, so their weight in MMA differs with the bound.
- post_run_checks.md
    - Not started. A check on a saved design that reads the physical fields and
      reports the margin of each condition the design must meet (print start,
      constraint values, true hotspot max, connectivity), since a constraint can pass
      while the condition behind it slips.
Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
