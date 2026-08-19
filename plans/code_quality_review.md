# Plan: Code quality improvements found during correctness review

> **Status (2026-08-19): open, living list.** This is not a phased implementation plan --
> it's a running list of code-quality issues (not correctness bugs) noticed while
> reviewing `sttopt/` for correctness against `resources/` and the MATLAB source (see
> `CLAUDE.local.md`, "(Transient) Current project state"). Append to it as review of more
> files continues; check items off (or move them to "Done") as they're fixed, ideally one
> per small commit per the repo's commit-size convention.

## Goal

Track cleanup/design items surfaced by manual review that are worth doing but are
independent of any single correctness fix -- so they don't get lost, and so they can be
picked off as small, separately-reviewable commits rather than bundled into whatever
correctness-fix commit happened to be in progress when they were noticed.

## Open items

### `timefield.py`

- [ ] `init_timefield`'s `variant` parameter is a bare `int` (1/2/3, `ValueError` on
  anything else) instead of an enum. Magic ints duplicated between the docstring and the
  `if/elif` dispatch (`sttopt/timefield.py:49-64`); an `IntEnum` (or plain `Enum`) would
  make call sites self-documenting and give a real type to check instead of prose.

### `optimize.py` / `conductivity.py`

- [ ] `step` (`sttopt/optimize.py:260`) inverts `fval` to recover `numer` and calls
  `hotspot_constraint` twice on refresh iterations (`sttopt/optimize.py:346-358`) --
  returning `numer`/`K_est` directly from `hotspot_constraint` would delete both the
  inversion and the second call, plus the paragraph of docstring explaining the
  workaround. A docstring paragraph justifying an awkward call pattern is itself a signal
  the interface is the wrong shape.
- [ ] `estimated_conductivity` (`sttopt/conductivity.py:119`) computes `FT_ba`/`DFT_ba`/
  `S1`/`S2` and throws them away -- only `K_est` survives. Either the computation should
  stop building values nothing uses, or (if `hotspot_constraint` genuinely needs to
  redo this work with the extra terms) the two functions should share it instead of
  `estimated_conductivity` duplicating a subset of `_conductivity_terms`'s work for
  nothing.
- [ ] `_conductivity_terms` (`sttopt/conductivity.py:75`) returns a positional 6-tuple.
  Call sites (`estimated_conductivity`, `hotspot_constraint`) have to unpack it
  positionally and remember what each slot means; a small `@dataclass` (or named tuple)
  would make call sites self-documenting and catch transposed-field bugs at write time
  instead of relying on correct unpack order.
- [ ] `Problem` (`sttopt/optimize.py:37`) has 30 fields, and `step`
  (`sttopt/optimize.py:260`) unpacks ~20 of them. Worth revisiting whether `Problem`
  should be decomposed into sub-groups (e.g. FEM assembly constants, filter/neighbor
  structures, MMA hyperparameters) that `step` can pass through instead of unpacking
  individually -- but see "Open questions" below before committing to a specific split.

### `cli.py`

- [ ] The module docstring (`sttopt/cli.py:1-36`) narrates review history in its last
  paragraph: `"Both mismatches were caught by an independent reviewer pass, not by any
  test -- tests/test_cli.py only checks that the PNG exists."` `CLAUDE.local.md`'s style
  section explicitly asks to avoid this kind of unimportant history in docs (contrast:
  history that explains a past *pitfall* to avoid repeating, which is fine to keep, and
  is arguably what the rest of this paragraph already does by explaining *why* `Obj.`/
  `Vol.` read from `prev_state` instead of `IterationRecord`). Trim to keep the
  why-it's-`prev_state`-not-`IterationRecord` reasoning and drop the meta-commentary
  about who caught it and how.
- [ ] Audit other module docstrings for the same pattern (this one was flagged by a
  previous review pass; grep for phrasing like "caught by", "reviewer", "previous
  agent" as a starting point) -- listed here as a to-check, not yet confirmed to
  recur elsewhere.

## Open questions

- Whether `Problem` decomposition is worth the churn given it's `frozen=True` and
  threaded through `build_problem`/`init_state`/`step`/tests -- needs a concrete
  sub-grouping proposal before starting, not just "it's big."
- Whether an `Enum` for `timefield` variant should also replace `Problem.tfield: int`
  (`sttopt/optimize.py:48`), or stay local to `init_timefield`'s parameter -- `tfield` is
  stored on `Problem` and round-trips through the CLI as a plain int argument, so making
  it an enum end-to-end touches more surface than just `timefield.py`.

## Done

(none yet)
