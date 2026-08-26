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
- torch_port.md
    - Phased plan to port sttopt to PyTorch (float64) for autodiff sensitivities and GPU.
      Gated: profile the current loop, then build and benchmark a GPU CG solver against
      scipy.spsolve, and only port the rest if the GPU solve actually wins.

Completed plans live in `plans/archive/` and are not summarized here to keep this
index short. Only open one if you specifically need the history behind a past
decision.

When a plan is executed, don't forget to update this directory accordingly: move
the completed plan and entry into `plans/archive/`.
