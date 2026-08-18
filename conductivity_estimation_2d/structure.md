# conductivity_estimation_2d — code structure

This directory implements the method from `resources/Das2025_Overheating-Prevention-Geometric-Method-TO.pdf`
("A physics-motivated geometric method for overheating prevention in topology optimization for
additive manufacturing," Das et al., CMAME 2025).

## Entry point

**`conductivity_estimation_stto_main.m`** — the main script (a plain script, not a function, though
a commented-out function signature at the top shows it started as one:
`[xPhys, tPhys, data] = Space_Time_TopOpt_Gravity(...)`). Run it directly; parameters
(`nelx=180, nely=60, nloop=800, nStage=8, volfrac=0.5, Theta=0.1, Tcr=0.8, tfield=3`) are
hardcoded at the top rather than passed as args.

## What it does

It's a joint space-time topology optimization: simultaneously optimizes a **density field**
`x`/`xPhys` (structural layout) and a **time field** `t`/`tPhys` (per-element normalized print
time, i.e. fabrication sequence), using an MMA optimizer, with a novel **overheating/conductivity
constraint** — the paper's core contribution.

Structure of the script:

1. **Setup**: builds a continuity filter `L` and density filter `H` (both via local-neighborhood
   sparse matrices), the plane-stress stiffness matrix `KE`, and a gravity-load matrix `C` for
   self-weight during build.
2. **Time-field init**: three options (`tfield`) — distance from a corner or from an edge —
   controlling the print-sequence starting boundary.
3. **Conductivity-estimation neighborhood setup**: builds `N_el`/`w_el`, a weighted neighbor list
   per element (radius `rmin=12`) — this is the geometric proxy structure for "local thermal
   conductivity."
4. **Main loop** (up to `nloop` iterations):
   - Structural compliance of the final structure (`Cal_c_ce_whole`, nested function at the
     bottom) and of each of `nStage` intermediate (partially-built) structures under gravity
     (`Cal_c_ce_for_gravity`), with sensitivities.
   - Constraints: global volume fraction, time-field smoothness/continuity, print start point,
     per-stage volume fractions.
   - **Hotspot/overheating constraint**: estimates a local conductivity proxy `K_est` per element
     as a neighbor-weighted, time-causally-masked average of neighbor densities (only counting
     neighbors already printed, via a smooth sigmoid `FT` in print-time), aggregated into a
     p-norm constraint `factor*numer/Tcr - 1 <= 0` with hand-derived analytic sensitivities
     (`df1`, `dt1`) w.r.t. density and time fields. This is the geometric conductivity-estimation
     method from the paper, done inline.
   - Calls `mmasub` (MMA solver) to update `[x; t]`, then reapplies filters/projections to get
     `xPhys`/`tPhys`.
   - Live plots every 10 iterations (density field, time field).
5. **Post-loop**: draws final layout/time-field boundary plots and a binarized combination plot
   colored by the estimated conductivity/overheating field.

## Supporting files

- **`mmasub.m`, `subsolv.m`** — Svanberg's MMA (Method of Moving Asymptotes) optimizer; the
  generic NLP solver the main script calls each iteration.
- **`conductivity_est_function_st.m`, `conductivity_est_function_stt.m`** — standalone versions of
  the same conductivity-estimate computation, factored out for the finite-difference sensitivity
  check that's commented out in the main loop (`%% sensitivity FD check`), perturbing density
  (`_st`) vs. time field (`_stt`) respectively.
- **`draw_boundary.m`, `draw_combination1.m`, `draw_combination2.m`, `draw_combination3.m`** —
  visualization utilities: patch plots of the structure colored by print time or by estimated
  temperature/conductivity, with stage-boundary lines overlaid. `draw_combination1` is the variant
  actually called at the end of the main script; 2 and 3 add a colorbar/legend and a
  masked-temperature variant respectively but aren't called from the main script (leftover/alternate
  versions).
- **`fabrication.m`** — a separate post-processing script (not called by main) that generates an
  animation (`VideoWriter`) of the structure being "printed" element-by-element in time order,
  colored by the conductivity-derived temperature proxy.
- **`Space_Time_TopOpt_Gravity_different_timefield.m`, `Space_Time_TopOpt_Robot.m`** — these are
  the *original baseline* space-time topology optimization scripts by Weiming Wang (2020),
  predating the overheating work — same density+time co-optimization but with self-weight/gravity
  or multi-robot print-order loads and **no conductivity/overheating constraint**. They appear to
  be the reference implementation this paper's code was built on top of, kept here for comparison
  rather than being part of the paper's method.
- **`screen.txt`** — log file target for MATLAB's `diary`, overwritten each run.

## Bottom line

`conductivity_estimation_stto_main.m` is the single entry point implementing the Das2025 paper's
method; everything else is either a solver dependency (MMA), a plotting/post-processing utility, a
factored-out piece used only for the FD sensitivity check, or an unrelated older baseline script
for comparison.
