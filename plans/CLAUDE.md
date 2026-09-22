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
Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
