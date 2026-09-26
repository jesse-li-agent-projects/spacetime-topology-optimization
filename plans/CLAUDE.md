This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- angular_weight.md
    - Restores Das2023 Ch. 3's angular stencil weight to the hotspot term, computed
      from the local `grad t` rather than a fixed build direction. A von Mises lobe
      with a `kappa` continuation from the present isotropic behavior, and
      `Normalization.HALF_STENCIL`'s divisor generalized from a constant to a
      per-element directional reference.
- curvature_constraint.md
    - A hard MMA constraint keeping the concave curvature of `t`'s iso-lines below
      `1 / tool_radius`, so the print tool cannot collide with the part. Calibrated
      LogSumExp smooth max, with `tool_radius` scheduled from 0 as the continuation.
- curvature_constraint_handoff.md
    - The brief for that plan's Phase 5 (tune the continuation to `R = 2.5` elements):
      corrections to the plan, the sweep setup, what to report, and known traps.
Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
