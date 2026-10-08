# Minimum length scale: options (deferred)

**Status: deferred (user, 2026-10-08).** None of the options below is appealing enough
to implement yet; try other avenues first. This note keeps the options and the evidence
so they need not be re-derived.

## Problem

The density filter (cone, radius 4 elements) with one Heaviside projection at η = 0.5
gives no minimum feature size. With the hotspot constraint on, designs get thin necks
("hinges"): a member narrows to 1–2 elements between thicker parts.

Measured on C1 at volfrac 0.5 (180 × 60 mesh, 600 iterations, final designs of the
tuning-P follow-up runs; neck = skeleton point whose thickness is under 0.6 × the thinner
of the two sides next to it):
- 7–12 necks per hotspot run, against 3 in plain TO.
- About a quarter of them (14 of 58) sit where the print time has a local maximum, i.e.
  where two print fronts meet; only 0.7–2% of all skeleton points do. Most necks are
  not at a meeting point, so changing the time field would not remove them.
- The thinnest members of every C1 design, plain TO included, are 1–2 elements wide.

## Options

All three are continuous, and are standard in the topology optimization literature.

1. **Robust formulation** (Wang, Lazarov & Sigmund 2011, *Struct. Multidisc. Optim.*
   43:767). Also evaluate an eroded design (projection threshold above 0.5); a neck
   vanishes there and its compliance blows up, so the optimizer thickens it. The
   cheap form optimizes the eroded design's compliance and keeps the hotspot on the
   blueprint design.
   - For: the most proven; also removes the 1–2 element members.
   - Against: one more FEM solve per iteration; the hotspot and the stiffness then see
     different designs.
2. **Geometric length-scale constraints** (Zhou, Fernández, Wang & Sigmund 2015,
   *Comput. Methods Appl. Mech. Eng.* 293:266). Two scalar constraints on the filtered
   field near the projection threshold, penalizing solid (and void) features below a
   minimum size.
   - For: no extra FEM solve; the continuous counterpart of a skeleton-thickness check.
   - Against: known to need careful tuning (threshold, continuation, when to switch on).
3. **Morphological open filter** (Sigmund 2007, *Struct. Multidisc. Optim.* 33:401).
   A smooth erode-then-dilate as the density filter, so no solid feature thinner than
   its radius can exist.
   - For: the most direct.
   - Against: strongly nonlinear, converges poorly, does not preserve volume.
