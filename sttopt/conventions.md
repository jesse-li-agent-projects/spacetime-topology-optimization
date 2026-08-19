# Conventions

Decisions that are load-bearing across most of `sttopt`, fixed once here rather than
re-derived per module. Module docstrings reference this file instead of restating it.

## Array order

The MATLAB source indexes elements column-major over an `(nely, nelx)` grid:
`e1 = (i1-1)*nely + j1` (1-indexed, `i1` = x-index, `j1` = y-index), and relies on this
via `xPhys(:)`, `reshape(xmma, nely, [])`, and the gravity matrix's `(y-1)*nely + x`
column index.

The Python port keeps every field shaped `(nely, nelx)` and uses `order='F'` for every
`.flatten()`/`.reshape()` that mirrors a MATLAB `(:)` or `reshape` — never silently
switching to row-major (`order='C'`, NumPy's default). A linear element index `e` (from
`order='F'` flattening, 0-indexed) maps to grid position `(e % nely, e // nely)`.

**Consequence for fixtures**: every fixture must use `nelx != nely` and an asymmetric
field. A square or symmetric test case can pass a transposed port undetected.

## Fixture format

MATLAB `cell` arrays (`N_el`, `w_el`, `WE`) don't round-trip cleanly through
`scipy.io.loadmat`. Fixtures dump per-element neighbor lists as flat COO triplets
`(e1, e2, w)` (one row per neighbor pair, `e1`/`e2` are 1-indexed element numbers as
MATLAB produces them) rather than as cell arrays. All other arrays are plain numeric
arrays. Fixtures are saved as `.mat` v7 (`save(..., '-v7')`), readable directly via
`scipy.io.loadmat` — no hand-rolled export format.

## Tolerance policy

Not bit-exactness — the port intentionally changes some implementations for numerical
stability or performance (e.g. `(1+exp(z))^-1` as a stable sigmoid instead of the
mathematically-equivalent but overflow-prone form; `eye(n) - L./M` as sparse ops instead
of a dense `eye(10800)`, ~933 MB). Compare per quantity, not per test:

- **Tight** (`rtol=1e-10`): purely algebraic quantities with no linear solve or
  iteration involved — filter matrices (`H`, `Hs`, `L`), neighbor weights (`N_el`,
  `w_el`), the element stiffness matrix `KE`.
- **Looser**: quantities downstream of a sparse linear solve (`U`, compliance `c` and
  its sensitivities) — solver implementation differences (MATLAB `\` vs
  `scipy.sparse.linalg.spsolve`) accumulate small numerical differences.
- **Growing with iteration count**, for end-to-end trajectory comparisons only:
  `subsolv`'s inner Newton line search amplifies small per-iteration differences.
  "Iteration 1 matches to 1e-9, iteration 5 to 1e-4" is the realistic target — not
  exact agreement after hundreds of iterations. `tests/conftest.py`'s `assert_close`
  takes an iteration count and scales tolerance accordingly.

## Known deviations (not bugs)

See the plan (`plans/conductivity_estimation_2d_python_port.md`, "Known traps") for the
full list. The one that affects test expectations directly: `DFT(o) = 0` on exact
ties in the hotspot constraint's finite-difference-vs-analytic comparison is a real
property of the original code (the true derivative at `Δt=0` is `rouf/4`, not 0), not
a porting bug — FD checks are expected to disagree with the analytic gradient exactly
at tie points, and this is asserted explicitly rather than avoided.
