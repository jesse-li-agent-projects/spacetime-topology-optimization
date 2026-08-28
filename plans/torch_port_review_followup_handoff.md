# Handoff — torch_port_review_followup

Self-handoff for continuing `plans/torch_port_review_followup.md` (PR #53). Phases 1-2
are committed and done. Phase 3 is committed with a narrower scope than the plan
wrote down -- read "Phase 3: what changed" below before touching `step()` again.
Phases 4-6 are untouched.

## Status

| Phase | State |
| --- | --- |
| 1 -- Docstring and boundary fixes | Done, committed (amended once mid-flight) |
| 2 -- Delete no-op detaches, mark gradient region | Done, committed |
| 3 -- Move beta updates to the tail | Done for items 1+3; item 2 (filter-pass removal) **dropped**, not deferred -- see below |
| 4 -- Move hand-derived formulas to `tests/reference/` | Not started |
| 5 -- One tensor-boundary conversion helper | Not started |
| 6 -- Reply to review comments | Not started |

Commits so far on this branch (`worktree-torch-port-review-followup`, based on
`plan/torch-port-review-followup` @ 3632a81):
1. Phase 1 (amended once -- see below)
2. Phase 2
3. Phase 3 (items 1+3 only)

## Phase 3: what changed from the plan, and why

The plan's item 2 said: "In the head, read `state.xTilde`/`state.xPhys`/`state.tPhys`
directly instead of re-filtering. This removes one full filter pass per iteration."

This turned out not to be implementable as a pure storage-reuse trick. Worked through
with the user in-session:

- `x`/`t` stay the autograd leaves (confirmed with the user) -- so the head's
  `xTilde = filter(x)` / `xPhys = heaviside(xTilde, beta_d)` pass is required every
  iteration regardless; it's not the redundant one.
- The only real candidate for removal was the **tail's** recompute of
  `xTilde_new`/`xPhys_new`/`tPhys_new` from `x_new`/`t_new` (post-MMA), which -- once
  the beta bump moved to the tail, making `beta_d` consistent across the boundary --
  is numerically identical to what the *next* iteration's head recomputes from the
  same `x_new` anyway. Storing the head's already-computed (pre-update) `xTilde`/
  `xPhys`/`tPhys` instead (skipping the tail recompute) looked like the fix.
- That breaks `test_step_output_state_is_self_consistent`
  (`tests/test_optimize.py:257`), which pins an explicit, load-bearing invariant:
  `state.xPhys`/`state.tPhys` always equal filtering `state.x`/`state.t` at
  `state.beta_d`. `_state_from_raw`'s docstring calls this "the invariant... i.e. the
  state `step` itself would have produced" and other tests build on it. Confirmed
  empirically: dropping the tail recompute broke 7 tests (`test_e2e.py`,
  `test_robustness.py`, `test_torch_fem.py`, `test_cli.py`, `test_optimize.py`) with a
  consistent one-iteration-stale pattern.
- So the tail recompute must stay for correctness, not just fidelity. Item 2 is
  reverted; the docstring/field comments were rolled back to not claim a lag that
  doesn't exist. **This is not a "removes one filter pass" win -- that part of the
  plan was wrong**, not merely unimplemented. Don't re-attempt it without a genuinely
  different mechanism (e.g. restructuring the objective's own differentiation to go
  through `xTilde`/`tPhys` as leaves with a filter-adjoint, the way the constraints
  already do via `_grad_rows_batched`/`_filter_adjoint` -- that's a real, larger
  refactor, not a quick win, and wasn't attempted here).

What's actually done: the three periodic updates (`beta_t`, `beta_d`, hotspot
`factor`) all now bump at the tail of `step()`, next to each other, taking effect
starting the next iteration -- exactly per the Decision at the top of the plan's
Phase 3. Fast suite (412 passed, 4 skipped) is green.

**Still needed before this phase can be considered closed:**
`test_e2e_slow.py::test_thesis_4_4_reproduction` (`nloop=800`) was kicked off in the
background (job `b7veiv6j0`) to check the two boundaries this reordering actually
touches (`loop % 30`, `loop % 50`) -- **check its result before merging.** If it
fails, the bump-to-tail reordering itself (not just the dropped item 2) needs
re-examination. Its band assertions are loose (`185.0 < f0val < ceiling`,
`tru_max` within 1% of 0.8), so a genuine ordering bug and normal drift may look
similar -- read the actual numbers, don't just check pass/fail.

The plan also flagged `tests/fixtures/torch_port_designs.npz` as expected to drift
slightly from this phase and said not to regenerate unless a benchmark looks wrong --
that guidance still stands, unchanged by the item-2 revert.

## Phase 1 amendment (context for `git log`)

The first Phase 1 commit used a defensive `x0.detach()` inside `femsolve()`
(unconditionally detaching, silently). The user asked for this to be an assertion
instead ("the incoming warmstart seed, if present, should be detached") so a caller
bug shows up loudly rather than being silently papered over. Amended before Phase 2
started: `femsolve()` now asserts `x0 is None or not x0.requires_grad`; `optimize.py`'s
`U=U_new.detach()` is the (required, not redundant) thing that satisfies it.

## Next steps

1. Check job `b7veiv6j0`'s result (`tests/test_e2e_slow.py::test_thesis_4_4_reproduction`).
   If green, Phase 3 is done as scoped (items 1+3; item 2 abandoned). If not, debug
   the tail-bump reordering before moving on.
2. Phase 4 -- move hand-derived formulas out of `sttopt/` into `tests/reference/`.
   Mechanical per the plan; the plan's own verification notes (fixtures need no
   regen, `matlab_reference.py` is untouched) still hold.
3. Phase 5 -- one `Problem`/`State` conversion helper in `torch_util.py`, tests go
   torch-native.
4. Phase 6 -- reply to and resolve the four review threads that need no code change
   (text already drafted in the plan).
5. Once all phases are done, move `plans/torch_port_review_followup.md` (and this
   handoff file) to `plans/archive/` and update `plans/CLAUDE.md`'s index, per that
   directory's own convention.
