This directory contains plans for agents.

- dummy_plan.md
    - Not actually a real plan, just an example to show syntax of this index: list the
      plan's filename, then one to three sentences about it indented below.
      KEEP this entry even once every plan below is finished/archived and this
      list is otherwise empty -- it's the format documentation, not a stale
      leftover.
- code_quality_review.md
    - Living list of code-quality/design cleanup items (not correctness bugs) surfaced
      during manual correctness review of the ported Python code.
- torch_port_part2.md
    - Ports the rest of the optimization loop to PyTorch, now that part 1's GPU gate has
      passed (MGCG beats scipy.spsolve by 4.36x at 180x60). Covers device/dtype plumbing,
      the FEM solve as an autograd Function with a hand-written adjoint, autograd replacing
      the hand-derived sensitivities, MMA, and the deletion of the NumPy path.
- torch_port_review_followup.md
    - Executes the design-review comments left on the Phase 3.1-3.7 PRs: moves the
      hand-derived formulas and the NumPy FEM path out of `sttopt/`, reorders the
      periodic `beta` updates, deletes the no-op detaches in `step()`, and collapses
      the tensor boundary in the tests down to one conversion helper.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
