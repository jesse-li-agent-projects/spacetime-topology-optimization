"""Space-time topology optimization with an overheating-prevention constraint.

Two optimization problems share this package. `stto.py` (CLI: `stto_cli.py`, or the
installed `sttopt` script) is a Python port of `conductivity_estimation_2d`'s MATLAB
implementation of the Das et al. (CMAME 2025) method: joint optimization of a
structural density field *and* a print-time field under a geometric conductivity/
overheating constraint. `seqopt.py` (CLI: `seqopt_cli.py`, or `sttopt-seq`) fixes the
density field -- read from a geometry `.npz` file, e.g. one built by
`geometry_builders.py` or a prior `stto` run's `final_design.npz` -- and optimizes only
the print-time field against the same overheating proxy plus a layer-uniformity
penalty; see `plans/archive/fixed_geometry_sequence_optimization.md` for why this
warranted a second problem rather than a flag on the first.

Use `stto` when the structural layout itself is still open; use `seqopt` once a layout
is fixed (by hand, or by a finished `stto` run) and only the fabrication sequence is
being tuned -- it is also the cheaper problem to iterate on, since there is no FEM
solve on its path. See `conventions.md` for array-order, fixture, and tolerance
conventions shared across this package, including how the two problems' time fields
differ.
"""
