# Conventions

Decisions that are load-bearing across most of `sttopt`, fixed once here rather than
re-derived per module. Module docstrings reference this file instead of restating it.

## Array order

The MATLAB source indexes elements column-major over an `(nely, nelx)` grid:
`e1 = (i1-1)*nely + j1` (1-indexed, `i1` = x-index, `j1` = y-index), and relies on this
via `xPhys(:)`, `reshape(xmma, nely, [])`, and the gravity matrix's `(y-1)*nely + x`
column index.

The Python port keeps every field shaped `(nely, nelx)` but uses NumPy's native
row-major order (`order='C'`, the default — every `.flatten()`/`.reshape()` below omits
`order=` entirely) for the element enumeration, rather than mirroring MATLAB's
column-major layout: a linear element index `e` (0-indexed, C order) maps to grid
position `(e // nelx, e % nelx)`. `nelx != nely`, mixed with a genuinely asymmetric
field, is what makes a C/Fortran-order mismatch visible at all — see "Consequence for
fixtures" below — so this was safe to flip once nothing depended on bit-matching the
frozen `.mat` fixtures' *element order* any more.

Node numbering (`fem.element_dof_map`'s and `gravity.py`'s `nodenrs`) is a separate,
purely internal dof-labeling scheme, unrelated to this element-order convention — it
stays column-major, since nothing outside those two (mutually consistent) call sites
observes node numbers directly. Likewise `fem.assemble_stiffness`'s
`KE.flatten(order='F')` flattens the local 8x8 element stiffness matrix, not a grid
array, and is an unrelated internal-consistency choice paired with that function's own
`iK`/`jK` construction — not a grid-order violation to fix here.

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
stability or performance (the neighbour sigmoid and its derivative evaluated through
`exp(-|z|)`, which cannot overflow, instead of the source's `(1+exp(z))^-1` and
`FT^2*rouf*exp(z)`, which reach `0*inf = NaN` for `rouf*dt > ~709`; `eye(n) - L./M` as
sparse ops instead of a dense `eye(10800)`, ~933 MB). Compare per quantity, not per test:

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

## Mesh assumptions

Every element is a unit cell on a regular `(nely, nelx)` grid — no per-element volume
`v_e`. This isn't an oversight: Wang et al. (2019) Eq. (3)'s density-filter weighting
includes a `v_e` term, but states element volume is constant (`v_e = v0`) for their
uniform mesh, at which point `v_e` cancels out of the (normalized) filter ratio —
that's why `filters.density_filter`'s `H`/`Hs` carry no such term. A non-uniform mesh
would need `v_e` reintroduced there (and anywhere else area/volume-weighted quantities
assume uniformity, e.g. volume-fraction constraints) — a larger change than adding a
guard, so no code currently asserts uniformity; this note is the single place that
assumption is recorded.

`timefield.py`'s three variants require `nelx`/`nely` not both 1 (`nelx == nely == 1`
divides by a zero max distance in the CORNER variant); a lone-1 mesh is otherwise
well-defined but doesn't necessarily span `[0, 1]` (see that module's docstring).

## Known deviations (not bugs)

See the plan (`plans/archive/conductivity_estimation_2d_python_port.md`, "Known traps")
for the full list.

One entry there no longer applies: `DFT(o) = 0` on exact `t[a] == t[b]` ties was
originally ported verbatim from the source (`conductivity_estimation_stto_main.m`,
`if TPhys(N_ele(o))==ti`), which zeros the sigmoid's t-derivative on *any* value-tie.
That's only correct for `a == b` self-pairs, where `FT(t[a], t[a])` is constant in
`t[a]` so the true derivative is 0 — a genuine tie between two *distinct* elements has
the ordinary `rouf/4` derivative, since `FT` is smooth there. This was a bug in the
original MATLAB, not a deliberate design choice, so this port fixed it
(`_pairwise_sigmoid_terms` in `conductivity.py` now checks `a == b` by index rather
than `t[a] == t[b]` by value) rather than reproducing it — correctness of the port
takes priority over bug-for-bug fidelity to the source.

Not a measure-zero edge case, either: `timefield_edge` (linear ramp, constant down
each column) produces structural off-diagonal ties on ~5% of neighbor pairs at a
realistic 180x60 mesh, surviving density filtering — so with that timefield choice
this branch is live from iteration 0, not a rare coincidence. The old bug was also
worse than "wrong on a small set": `DFT` was ~`rouf/4` approaching a tie and exactly
`0` at one, so `dt1` was discontinuous in `t` — a hole in the gradient field that an
optimizer driving a symmetric design toward equal print times would walk straight into.
