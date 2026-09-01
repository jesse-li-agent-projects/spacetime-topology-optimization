This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.

- drop_converged_cg_rows.md
    - Retire each batch row from `torch_fem.pcg` as it converges, instead of running
      every row for the slowest row's iteration count, so that batching wins at every
      mesh size and `Problem.batch_fem_solves` (and the sequential FEM path in
      `optimize.step`) can be deleted. Includes the measurements motivating it.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
