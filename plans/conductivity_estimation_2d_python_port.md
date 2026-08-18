# Plan: Port `conductivity_estimation_2d` to Python

> **Status (2026-08-19): Phase 0 in progress, paused on a MATLAB regression.** See
> "Phase 0 progress / handoff" below for exactly what's done, what's written-but-unrun,
> and what to do next. Short version: everything except actually *running* the MATLAB
> fixture harness is done. MATLAB itself stopped working mid-session in a way deeper
> than the original licensing/EPERM issue (Finding 1) — confirmed by the user
> ("MATLAB issue is running deeper than expected"). Do not resume by re-deriving
> context; read the handoff section first.

> **Note (superseded, kept for history):** MATLAB access from this sandbox was earlier
> pending a sandbox update (license checkout failing — see Finding 1). That update did
> land (see `matlab_sandbox_setup` memory) and MATLAB worked for simple calls as of
> 2026-08-18/19, but see the Status note above for a newer, deeper regression found
> while Phase 0 was running the real fixture harness.

## Goal

Port the MATLAB implementation of the Das2025 overheating-prevention space-time
topology optimization method (`conductivity_estimation_2d/conductivity_estimation_stto_main.m`
and its dependencies) to a Python package at the top level of this repo.
**Correctness is the primary constraint** — this is optimization code with several
hand-derived analytic sensitivities; a silently-wrong gradient produces a plausible-looking
but incorrect optimum, not a crash. The plan is built around getting verification
infrastructure in place *before* porting the hard math, and porting in small,
independently-testable increments rather than one big translation pass.

## Scope

**In scope** (the paper's method and its live dependencies):
- `conductivity_estimation_stto_main.m` — main script/loop
- `conductivity_est_function_st.m`, `conductivity_est_function_stt.m` — standalone
  conductivity-estimate functions (used for the original author's FD sensitivity checks)
- `mmasub.m`, `subsolv.m` — MMA optimizer (Svanberg)
- `draw_boundary.m`, `draw_combination1.m` — the plotting calls actually exercised by
  the main script

**Out of scope for now** (revisit later if wanted):
- `draw_combination2.m`, `draw_combination3.m` — alternate plotting variants not
  called from main
- `fabrication.m` — standalone post-processing animation, not called from main
- `Space_Time_TopOpt_Gravity_different_timefield.m`, `Space_Time_TopOpt_Robot.m` —
  pre-2025 baseline scripts by a different set of authors, predating the conductivity
  constraint; useful for comparison but not part of the method being ported

## Two findings that affect the plan

1. **MATLAB could not run in this sandboxed job environment as of this writing** —
   `/usr/local/bin/matlab` initializes its preferences directory but then exits with
   code 1 and no output, consistent with a blocked network call to a license server.
   You're working on a sandbox update to fix this; the plan now assumes MATLAB will be
   available and Phase 0 owns fixture generation directly (see the note at the top of
   this document). If MATLAB is still unusable when Phase 0 starts, re-run:
   `HOME=$TMPDIR MATLAB_PREFDIR=$TMPDIR/mlprefs matlab -nodisplay -nojvm -batch "disp(1+1)"`
   and fall back to generating fixtures out-of-sandbox if it still fails silently.

2. **The main script's post-loop plotting call is broken as committed, but the fix is
   unambiguous.** Line 577 calls `draw_combination(xPhys,tPhys,nStage,1.0e-1)` — there
   is no `draw_combination.m`, only `draw_combination1/2/3.m`. Comparing signatures:
   - `draw_combination1(density, timing, eps)` colors elements by the raw `timing`
     field (3 args, no stage-boundary overlay, no colorbar).
   - `draw_combination2(density, temp, timing, Ns, eps)` and `draw_combination3(XPhys,
     tPhys, T1)` both require a separate **temperature** field (`temp`/`T1`, distinct
     from the time field) as input, and draw per-stage boundary lines.

   The broken call at line 577 happens *before* `B = reshape(K_est, nely, nelx)` is
   computed (that's line 581, after `toc`/`diary off`) — at that point in the script no
   temperature field exists yet, only `xPhys`/`tPhys`. That rules out variants 2 and 3
   outright; they need an input that doesn't exist yet at that call site. Variant 1 is
   the only one whose required inputs are available, and its signature matches exactly
   once the stray `nStage` argument is dropped — almost certainly copy-paste debris from
   `draw_boundary(tPhys, nStage)` on the line directly above, which does take `nStage`.
   The corrected call is also structurally identical to the two later, working calls in
   the same script (`draw_combination1(xPhys, T, 1.0e-1)`, commented out, and
   `draw_combination1(XPhys, T1, 1.0e-1)`, the one that actually runs at the very end).

   **Resolution: treat line 577 as `draw_combination1(xPhys, tPhys, 1.0e-1)`** — a
   preview plot of the structure colored by raw print-time, before binarization and
   before the temperature-colored final plot. This is now settled, not an open question;
   Phase 9 (viz) implements only `draw_combination1`'s behavior, consistent with the
   existing Scope section above.

## Conventions to fix once, in Phase 0

These decisions are load-bearing across almost every module, so they need to be made
before any porting starts rather than discovered ad hoc:

- **Array order.** MATLAB code uses column-major linear indexing throughout —
  `e1 = (i1-1)*nely + j1` over an `(nely, nelx)` grid, `xPhys(:)`, `reshape(xmma, nely, [])`,
  the gravity matrix's `(y-1)*nely + x` column index. The Python port keeps arrays shaped
  `(nely, nelx)` and uses `order='F'` for every flatten/reshape that mirrors a MATLAB `(:)`
  or `reshape`, rather than silently switching to row-major. This gets written down once
  in a `conventions.md` and referenced from module docstrings, not re-derived per module.
  **Consequence for fixtures:** every fixture must use `nelx != nely` and an asymmetric
  field — a square or symmetric test case can pass despite a transposition bug.

- **Fixture format.** MATLAB `cell` arrays (`N_el`, `w_el`, `WE`) don't round-trip cleanly.
  Fixtures dump the element neighborhood as flat COO triplets `(e1, e2, w)` and other
  arrays as plain numeric arrays, saved as `.mat` (v7, readable directly via
  `scipy.io.loadmat`/`savemat` — no hand-rolled export format needed).

- **Tolerance policy, not bit-exactness.** The port intentionally changes some
  implementations for numerical stability / performance (e.g. `(1+exp(z))^-1` as a
  stable sigmoid, `eye(n) - L./M` as sparse ops instead of a dense `eye(10800)`, which
  is ~933 MB). State per-quantity tolerances up front: tight (`rtol=1e-10`) for
  purely algebraic quantities (filters, neighbor weights, KE), looser after a sparse
  linear solve, and *growing* tolerance with loop iteration for end-to-end comparisons
  (`subsolv`'s inner Newton line search amplifies small differences — "iteration 1
  matches to 1e-9, iteration 5 to 1e-4" is the realistic target, not exact agreement
  after 800 iterations).

## Known traps to test for explicitly (not just port silently)

- **`DFT(o)=0` on exact ties** (main script line ~366): the true derivative of the
  sigmoid mask at `Δt=0` is `rouf/4`, not 0. The self-neighbor is *always* a tie, and
  `tfield` options can produce other exact ties. An FD check on `dt1` will legitimately
  disagree with the analytic gradient at tie points — this is a known deviation in the
  original code, not a porting bug. Decide the policy explicitly: MATLAB-vs-Python
  fixtures are authoritative for *port fidelity* (reproduce the original, ties included),
  FD checks are authoritative for *math correctness* elsewhere; where they disagree at
  known tie points, that's documented, not "fixed" unilaterally.
- **The hotspot constraint's `factor` is stateful** (`if rem(loop,25)==0`, carried across
  iterations) — not a pure function of `(xPhys, tPhys)`. Unit tests must inject `factor`
  explicitly rather than recomputing it; E2E comparisons must start from the same
  initialization and run the same iteration count.
- **`WE` looks like it should equal `w_el`** (weights are a symmetric function of distance,
  and boundary truncation is symmetric) — worth an explicit unit test rather than an
  assumption; if confirmed, it simplifies the Python port (drop `WE` entirely, reuse `w_el`).
- **1-indexed DOF boundaries**: `fixeddofs = 1:2*(nely+1)`, and `F`'s point load is at
  dof `2*(nelx+1)*(nely+1)` — both need explicit off-by-one translation, tested against
  a hand-checked small case, not just "looks right."
- **`subsolv`'s `m < n` branch is the only one ever exercised** in this problem
  (`m` ≈ 70–80 constraints, `n` = `2*nelx*nely` variables). Port both branches (it's one
  function, not much extra work) but only the `m < n` branch gets tested against fixtures;
  the other is flagged as unvalidated in a comment.
- **`spdiags` semantics**: MATLAB's `spdiags` used here for pure diagonal matrices maps to
  `scipy.sparse.diags`, not a naive port of `spdiags`'s more general (and different)
  argument convention.

## Dead code not to carry over

`Kappa`, `Eel`, `K_sub`, `dsum`, `W`, `lambda1`, `lambda`, `per`, the first `Nsum`
(immediately recomputed as `Nsum3`), `fval1` and `Tru_m` (used only inside commented-out
blocks), `change = 1` (assigned, never read). Confirm each is genuinely unused during
Phase 0/1 porting and drop it — carrying it over would make the Python harder to review
than the MATLAB for no benefit.

## Target module decomposition

Proposed package name: **`sttopt`** (space-time topology optimization) — easy to
rename if you'd prefer something else.

```
sttopt/
  fem.py          # KE (plane-stress stiffness), edofMat/nodenrs connectivity,
                   # assemble_stiffness, solve_fe (free/fixed dof partition + solve)
  filters.py      # density filter (H, Hs), continuity filter (L), Heaviside
                   # projection + its derivative
  timefield.py    # the 3 initial time-field variants (tfield=1/2/3)
  gravity.py      # self-weight load matrix C construction
  compliance.py   # Cal_c_ce_whole, Cal_c_ce_for_gravity + shared time-mask ft/dfdt
  constraints.py  # global volume, time-field continuity, start-point,
                   # per-stage volume constraints (value + sensitivity each)
  conductivity.py # neighbor list construction (N_el/w_el as COO), K_est,
                   # hotspot p-norm constraint + sensitivities (df1, dt1)
  mma.py          # mmasub + subsolv (Svanberg MMA), ported close to verbatim —
                   # this is a well-specified, physics-independent algorithm
  optimize.py     # main loop orchestration (owns iteration state: beta, rou,
                   # factor, xold1/xold2, low/upp)
  viz.py          # draw_boundary, draw_combination1 (matplotlib)
  cli.py          # argparse entry point equivalent to the main script, with
                   # nelx/nely/nloop/nStage/volfrac/Theta/Tcr/tfield as flags
                   # instead of hardcoded constants
tests/
  fixtures/        # .mat files generated by the MATLAB harness (Phase 0)
  conftest.py      # fixture loader, tolerance-policy helper (assert_close)
  test_fem.py
  test_filters.py
  test_timefield.py
  test_gravity.py
  test_compliance.py
  test_constraints.py
  test_conductivity.py
  test_mma.py
  test_e2e.py       # small-grid, few-iteration full-loop comparison
```

## Testing strategy

Two complementary, language-agnostic kinds of unit test, plus one E2E test:

1. **Fixture-based (MATLAB-vs-Python) tests** — the primary defense against translation
   bugs (indexing, array order, off-by-ones). A MATLAB harness script (Phase 0 deliverable,
   run in-sandbox via `matlab -batch`) calls each MATLAB function/code-block on small,
   asymmetric, fixed-seed inputs and dumps inputs+outputs to `.mat`. Python tests load
   the same `.mat`, call the ported function, and compare per the tolerance policy above.
   This assumes the sandbox MATLAB update has landed by Phase 0 (see Finding 1); if not,
   this step falls back to an out-of-sandbox run and everything downstream of it slips
   accordingly.

2. **Finite-difference gradient checks** — pure-Python, no MATLAB dependency. Every
   function that returns an analytic sensitivity gets an FD check against its own value
   function, on the same small asymmetric grids. This validates internal consistency of
   the port independent of MATLAB availability, and is exactly what the original author's
   commented-out FD checks in the main script were already doing by hand for the hotspot
   constraint — we formalize and automate that.

3. **End-to-end test** — run the full optimization loop for a handful of iterations
   (e.g. 5) on a small grid (e.g. 8x6) in both MATLAB (via the harness) and Python,
   comparing objective/constraint trajectories with growing tolerance per iteration.
   This is the test that would actually catch "each piece is right in isolation but the
   orchestration in `optimize.py` wires them together wrong."

## Phase 0 progress / handoff (2026-08-19)

Work happened on branch `worktree-sttopt-phase0` (pushed; not merged, no PR opened yet
since nothing is verified end-to-end). Everything below is committed on that branch.

**Done and verified:**
- `sttopt/conventions.md` — array-order (column-major, `order='F'`), fixture-format
  (COO triplets for MATLAB cell arrays, plain `.mat` v7 otherwise), and tolerance-policy
  (algebraic/solved/e2e tiers) conventions, written per the plan's "Conventions to fix
  once" section above.
- `pyproject.toml` + `sttopt/__init__.py` — package scaffolding. Build backend is
  plain `setuptools` (not `hatchling` — not available offline in the sandbox venv).
  **Important sandbox finding:** `/home/jesse/v` is a pre-provisioned venv with
  numpy/scipy/matplotlib/jaxtyping/pytest already installed (already `$VIRTUAL_ENV`)
  — but its `site-packages` is read-only for the sandboxed `claude` user, so
  `pip install -e .` fails with `EROFS`. Use `PYTHONPATH=<repo root>` with
  `/home/jesse/v/bin/python3` directly instead of installing. PyPI/apt are unreachable
  from the sandbox network allowlist, so this venv is the only package source.
- `tests/conftest.py` — `load_fixture()` (`.mat` loader) and `assert_close()` /
  `e2e_rtol()` tolerance-policy helpers, implementing conventions.md. Smoke-tested
  (`tests/test_scaffolding.py`, 3/3 passing).
- **Sandbox quirk found and worked around:** running `pytest` with cwd = repo/worktree
  root crashes on collection (`PermissionError` on the `personal` symlink, which
  `claude` can't read but `jesse` can) — this happens for *any* pytest invocation from
  repo root in this sandbox, regardless of `--rootdir`/`--confcutdir`/testpaths, because
  pytest 9.1.1 always scandirs the invocation cwd. **Run pytest with cwd inside a
  subdirectory instead**, e.g. `cd tests && /home/jesse/v/bin/python3 -m pytest .`.
  Full detail in the `pytest-personal-symlink-quirk` memory.

**Written and reviewed, but NOT YET RUN (this is where a future agent picks up):**
- `tests/fixtures/generate_fixtures.m` — the MATLAB fixture-generation harness.
  Runs the real problem on a small (`nelx=7, nely=5`, asymmetric per convention) grid
  for `nloop=3` iterations, calling the actual `mmasub.m`/`subsolv.m` unmodified, with
  `Cal_c_ce_whole`/`Cal_c_ce_for_gravity` duplicated verbatim as local functions (MATLAB
  can't call another script's local functions externally — this is a faithful copy of
  lines ~605-665 of `conductivity_estimation_stto_main.m`, not a reimplementation).
  Produces 9 fixture files under `tests/fixtures/`: `fem_setup.mat`, `fem_solve.mat`,
  `filters.mat`, `gravity.mat`, `timefield.mat`, `conductivity_neighbors.mat`,
  `compliance.mat`, `constraints.mat`, `conductivity.mat`, `mma.mat`, `e2e.mat` —
  one per Phase 1-8 test module, so each `tests/test_*.py` can load its own eponymous
  fixture file (`conductivity_neighbors.mat` covers the Phase 6 neighbor-list COO
  triplets, including both `N_el`/`w_el` and `WE`, so the Trap 3 `WE == w_el` check can
  be tested directly against fixtures rather than assumed).
  The script was read through twice for correctness against the original main script
  (matched line-by-line) but **has never actually executed successfully** — MATLAB
  hung indefinitely (no stdout after 5+ minutes; `TaskOutput` confirmed the process was
  genuinely alive, not just invisible) partway through this session, and the user
  confirmed it's a deeper regression than the licensing/EPERM issue in Finding 1 (see
  the `matlab_sandbox_setup` memory's 2026-08-19 addendum). **Do not assume the script
  is bug-free just because it reads correctly** — it has zero execution evidence.
- `tests/fixtures/check_fixtures.tmp.py` — throwaway sanity-check script (per
  `*.tmp.py` convention) that loads all 9 fixture files and prints keys/shapes/NaN
  checks. Run this immediately after the harness succeeds, before trusting the fixtures.

**Next steps for whoever picks this up:**
1. Confirm MATLAB actually works again: `matlab -nodisplay -nosplash -batch "disp(1+1)"`
   should return `2` in a few seconds. If *that* hangs or is slow, the regression is
   still present — don't waste time on the harness script until this trivial call is
   fast and reliable.
2. Run the harness from the repo root:
   `matlab -nodisplay -nosplash -batch "run('tests/fixtures/generate_fixtures.m')"`
   (give it a generous timeout / run via `run_in_background` — expect it to be fast
   once MATLAB itself is healthy, since the grid is tiny, but don't assume that without
   evidence given today's hang).
3. Sanity-check the output: `cd tests/fixtures && /home/jesse/v/bin/python3 check_fixtures.tmp.py`
   (or wherever cwd needs to be to avoid the `personal`-symlink pytest issue, though
   this particular script isn't run via pytest so that quirk shouldn't apply to it).
4. If the harness errors, most likely causes given how it was built: a typo introduced
   while transcribing `Cal_c_ce_whole`/`Cal_c_ce_for_gravity` verbatim, or an off-by-one
   in the small-grid parameter choices (`rmin_cond=3` was chosen to keep conductivity
   neighborhoods non-trivial-but-not-all-to-all on the 7x5 grid — verify this still
   holds, i.e. that `coo_e1`/`coo_e2` in `conductivity_neighbors.mat` isn't just "every
   element is every other element's neighbor").
5. Once fixtures are generated and sanity-checked, commit them, mark Phase 0 complete,
   and move to Phase 1 (`fem.py`) per the phased plan below. Phases 1-7 are described as
   parallelizable once Phase 0 (and Phase 1 for anything depending on `fem.py`) lands —
   worth delegating to subagents at that point rather than doing serially, unlike Phase
   0 itself which benefited from one agent holding the whole MATLAB-source context.

## Phased implementation plan

Each phase ends with its own tests passing before the next starts. Phases 2–6 are
independent of each other's *internals* but share the fixture harness and conventions
from Phase 0, so they can proceed in parallel once Phase 0 and Phase 1 (fem — needed by
compliance) land. MMA (Phase 7) is physics-agnostic and can start any time after Phase 0.

- **Phase 0 — Infrastructure and conventions**
  - Verify MATLAB actually runs in-sandbox (see Finding 1); if not, fall back to an
    out-of-sandbox generation step before continuing.
  - Write `conventions.md` (array-order, indexing, tolerance policy) referenced by later
    modules.
  - Write the MATLAB fixture-generation harness script and run it directly (in-sandbox),
    committing `tests/fixtures/*.mat`.
  - Set up `tests/conftest.py` with `.mat` fixture loading and an `assert_close` helper
    implementing the tolerance policy.
  - Package scaffolding (`pyproject.toml`, `sttopt/__init__.py`), pytest wired into
    whatever CI/local workflow you use.

- **Phase 1 — FEM core** (`fem.py`)
  - `plane_stress_KE(nu)`, `element_dof_map(nelx, nely)`, `assemble_stiffness(...)`,
    `solve_fe(K, F, freedofs)`.
  - Fixture tests against MATLAB's `KE`, `edofMat`, and a solved `U` for a small
    fixed-density fixed-load case. No FD check needed here (linear FEM, not a
    sensitivity).

- **Phase 2 — Filters** (`filters.py`)
  - Density filter (`H`, `Hs`) construction, continuity filter `L` (as sparse, not
    dense `eye - L./M`), Heaviside projection + derivative.
  - Fixture tests against MATLAB `H`, `Hs`, `L` on a small grid; FD check of the
    projection derivative.

- **Phase 3 — Time field init & gravity load** (`timefield.py`, `gravity.py`)
  - The 3 `tfield` variants; gravity matrix `C`.
  - Fixture tests only (no sensitivities involved).

- **Phase 4 — Compliance & sensitivities** (`compliance.py`, depends on Phase 1–3)
  - `Cal_c_ce_whole` and `Cal_c_ce_for_gravity`, factoring out the shared time-mask
    `ft(tPhys, ti, lambda)` / `dfdt(...)` helper (used again in Phase 5).
  - Fixture tests for `c`, `dcx`, `dct`; FD checks of `dcx`/`dct` against `c` on a
    small grid with a handful of random density/time fields.

- **Phase 5 — Simple constraints** (`constraints.py`, depends on Phase 2–4's `ft`)
  - Global volume, time-field continuity, start-point, per-stage volume constraints.
  - Fixture + FD tests per constraint.

- **Phase 6 — Conductivity / hotspot constraint** (`conductivity.py`) — hardest phase,
  gets the most test budget.
  - Neighbor-list construction as COO triplets; verify `WE == w_el` claim (Trap 3) as
    an explicit test, then simplify accordingly.
  - `K_est` computation; cross-check against the standalone
    `conductivity_est_function_st.m`/`_stt.m` fixtures (these already exist as the
    original author's own FD-check scaffolding, so they're a natural fixture source).
  - Hotspot p-norm constraint value and `(df1, dt1)` sensitivities, with `factor`
    injected explicitly (Trap 2).
  - FD checks documented against Trap 1 (tie-point deviation is expected and recorded,
    not treated as a failure).

- **Phase 7 — MMA optimizer** (`mma.py`) — can run in parallel with Phases 1–6.
  - Port `mmasub`/`subsolv` fairly directly (well-specified numerical algorithm).
  - Fixture tests: feed MATLAB's `(xval, xmin, xmax, f0val, df0dx, fval, dfdx, ...)`
    from one real iteration of the main loop and compare `xmma` and the KKT multipliers.
    Both `m < n` and `m >= n` branches ported; only `m < n` gets a fixture test (Trap 4),
    the other flagged as unvalidated.

- **Phase 8 — Main loop orchestration** (`optimize.py`)
  - Wires Phases 1–7 together, owning the iteration-dependent state (`beta`, `rou`,
    `factor`, `xold1/xold2`, `low/upp`) that the per-phase unit tests intentionally
    inject rather than manage.
  - This is where the E2E test (small grid, few iterations, MATLAB vs Python trajectory)
    lives and must pass.

- **Phase 9 — Visualization & CLI** (`viz.py`, `cli.py`)
  - Port `draw_boundary`/`draw_combination1` to matplotlib; add the argparse entry point.
  - Lowest correctness risk (visual output, not numerical), so lightest testing —
    smoke tests that it runs without error on a small case, manual visual comparison
    against the MATLAB plots rather than pixel-exact fixtures.

## Open items

- Confirm the sandbox MATLAB update (Finding 1) has landed before Phase 0 starts;
  otherwise fixture generation needs an out-of-sandbox step first.
