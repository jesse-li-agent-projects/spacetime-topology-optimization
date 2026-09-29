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
- virtual_heat_timefield.md
    - Two new scripts beside `stto.py` where the time field is the solution of a
      virtual heat equation with the diffusivity as the design variable (Wu2025 §2.3
      with a drain; and a Laplace variant with optimizable wall data), so the time
      field has no local minima by construction. Large scope: ask the user often.
Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
