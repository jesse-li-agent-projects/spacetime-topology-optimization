This directory contains plans for agents.

- seqopt_sawtooth_multipronged.md
    - Experiment plan for removing the gradient-padding sawtooth from `seqopt`'s
      layer-uniformity optimization, which makes the reported CV ~4x better than the
      field really is. Three prongs (harmonic void init, roughness_weight continuation,
      filtered tPhys) plus the pre-checks and dead ends already settled.
- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
