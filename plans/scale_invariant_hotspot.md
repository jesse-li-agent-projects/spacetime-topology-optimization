# A hotspot severity invariant to scaling every density

## Motivation

On C1 0.3, Variant 2's grey bars stay grey until forced to solidify, and often vanish
when they do (user, 2026-10-18). The suspected cause is the hotspot measure itself: with
`absolute` shielding, `K_e = Σ_j w F x_j^q / D` scales as `n^q` when every density is
scaled by `n`. While the bar is grey it reads as unshielded, its own density only ever
gets pushed down (`∂S_e/∂x_e = r x_e^(r-1) (1 - K_e) > 0`), and solidifying it helps only
once it is solid all at once. Removing the bar also clears the hotspot.

First attempt (batch 14): `S_e(n x) = S_e(x)` for every element, i.e.
`Σ_j x_j ∂S_e/∂x_j = 0`, so the hotspot exerts no force along "scale every density". This
removed the barrier but also every push to delete or support a violating bar; the
requirements below replace it.

## Requirements (user, 2026-10-18)

Stated as gradient pressures. `S¹_e` is the severity element `e` would have if it and its
neighborhood were binarized as they stand; "violating" means `S¹_e > Tcr`.

- **R0, iff.** No hotspot pressure on a bar if and only if its solid version passes. The
  *if* direction stops grey refusing to solidify (the current formula's barrier); the
  *only if* direction stops grey hiding a hotspot until binarization (B's failure).
  Where the two conflict, prefer violating *if* (user, 2026-10-19): a reading that is too
  loud costs optimality, one that hides costs feasibility. A weaker *if* that may be
  enough (proposed, untested): if `S¹_e ≤ Tcr`, any hotspot pressure on the bar points
  toward solid. The `absolute` hump fails even that, since below its peak it pushes grey
  toward void.
- **R1.** A violating bar's own material is pushed out: `∂S_e/∂x_e > 0`.
- **R2.** A violating bar's earlier-printed neighbors, void ones included, are pushed to
  solidify and support it: `∂S_e/∂x_j < 0`, nonzero at `x_j = 0`.
- **R3.** Void contributes nothing to the aggregate, whatever its print time (PR #98);
  possibly near-void too.

### The tension, formally

**Setting.** A bar on void support: every earlier neighbor outside the bar is void, and
the bar is uniform at density $s$. Along $s$ the bar's severity is $\sigma(s)$, with
$\sigma(1) = S^1$. Define the margin and the hiding

$$m = S^1 - T_{cr}, \qquad h(s) = S^1 - \sigma(s), \quad h(1) = 0.$$

Scaling a bar whose support is void scales its whole neighborhood, so $\sigma'(s)$ is
R1's removal pressure, and for a shielding that only sees density ratios it is the gate's
alone: $s\,\sigma'(s) = \sum_j x_j\,\partial S_e/\partial x_j$ (Euler). Treat the
aggregate as a hard max, so a reading presses iff $\sigma(s) > T_{cr}$.

**The requirements in terms of $h$.**

- R0 *if*: $m \le 0 \Rightarrow h(s) \ge m$. Any $h \ge 0$ (never reads worse than solid)
  suffices; the `absolute` hump is $h < 0$.
- R0 *only if*: $m > 0 \Rightarrow h(s) < m$ for all $s \in (0, 1]$.
- R1: $\sigma' > 0$, and integrated, $\displaystyle\int_a^1 \sigma'(s)\,ds = h(a)$.
- R3: void reads below $T_{cr}$ by some $\delta > 0$ whatever its print time, so for $S^1$
  up to 1: $\sigma(0^+) \le T_{cr} - \delta$, i.e. $h(0^+) \ge m + \delta$.
- R2 is separate: with $K_e = \sum_j w_j\,c(u_j)/D$ and $u_j = (x_j+\varepsilon)/(x_e+\varepsilon)$,

  $$\frac{\partial K_e}{\partial x_j} = \frac{w_j}{D}\,\frac{c'(u_j)}{x_e+\varepsilon},$$

  and void support has $u_j \approx 0$, so R2 $\iff c'(0) > 0$. $c = u^q$ (and `absolute`'s
  $x_j^q$) has $c'(0) = 0$ for $q > 1$.

**Proposition.** For a violating bar ($m > 0$):

1. R0 *only if* and R3 cannot both hold for a $\sigma$ continuous on $[0, 1]$: since
   $\sigma(0^+) < T_{cr} < \sigma(1)$, some $s$ has $\sigma(s) = T_{cr}$, and below it the
   bar reads as passing. Holding both needs a jump at $s = 0$, which in practice is a
   transition over $(0, \varepsilon)$: batch 14's $x/(x+\varepsilon)$ gate.
2. R1 is then starved wherever R0 holds: for any $a$ with $h(a) < m$,
   $\int_a^1 \sigma' = h(a) < m$. The total removal pressure a bar can feel while it still
   reads as violating is less than its margin; a marginal violator feels none. The rest,
   $h(0^+) - h(a) > \delta$, is spent in the transition band
   $B_m = \{s : h(s) \ge m\}$, where R0 fails.

So no formulation meets R0, R1 and R3 exactly. What's left to choose is where the band
$B_m$ sits and how wide it is; the removal pressure inside it is roughly
$(m + \delta)/|B_m|$.

| gate | $\sigma(s)$ | $h(s)$ | band $B_m$ |
|---|---|---|---|
| multiplicative $x^r$ | $s^r S^1$ | $(1 - s^r)\,S^1$ | $s \le (T_{cr}/S^1)^{1/r}$ |
| $x/(x+\varepsilon)$ | $\frac{s}{s+\varepsilon}S^1$ | $\frac{\varepsilon}{s+\varepsilon}S^1$ | $s \le \varepsilon\,T_{cr}/m$ |
| log, $\gamma \ln x$ | $S^1 + \gamma\ln s$ | $\gamma\ln(1/s)$ | $s \le e^{-m/\gamma}$ |

Every band grows to all of $(0, 1)$ as $m \to 0$ (point 2), so R0 *only if* can only be
required above a tolerance: $m > \tau$.

**Where hiding is free.** Binarization at $\eta$ sends $s < \eta$ to void, so the binarized
check never sees a bar in $B_m$ if $B_m \subset (0, \eta)$ for all $m > \tau$. That keeps
the reading honest for the check as it stands; hiding only bites if SIMP later solidifies
the bar. For the log gate this is $e^{-\tau/\gamma} \le \eta$, i.e.

$$\gamma \le \frac{\tau}{\ln(1/\eta)} \approx 0.014 \quad (\tau = 0.01,\ \eta = 0.5).$$

The price, by point 2: a violating bar denser than $\eta$ feels at most $\tau$ of removal
pressure in total, so it can only be fixed by R2's support, the time field, or compliance;
strong R1 exists only below $\eta$.

| formulation | R0 if | R0 only if | R1 | R2 | R3 |
|---|---|---|---|---|---|
| `absolute`, `x^r` gate | ✗ hump at `x^q = r/(r+q)` | ✓ for small r | below the hump | ✗ | via `x^r`, weak for small r |
| A, `x/(x+ε)` gate (batch 14) | ✓ | ✓ | ✗ | ✗ | void only, not near-void |
| B, `x/(x+ε)` gate (batch 14) | ✓ | ✗ | ✗ | ✗ | void only |
| A with `q = 1`, `x^r` gate (untested) | ✓ | band `S¹ < Tcr / x^r` | ✓ | ✓ | via `x^r` |
| A, `x/(x+ε)` gate, plus Φ below (untested) | ✓ | ✓ | via Φ | via Φ | void; near-void via `g` |

Point 2 assumes one function supplies both the reading and the pressure. Splitting them
avoids it; using a pressure that isn't the reading's gradient ("inconsistent
sensitivities") would too, but would leave MMA's model of the constraint wrong.

### What A's constraint gradient can push on (tentative)

The capped ratio sees only density ratios, so `Σ_j x_j ∂S_e/∂x_j ≈ 0`: no component along
scaling a whole neighborhood. What seems to be left:

- density *within* a stencil: the bar lighter, non-void support heavier, until the bar is
  no denser than its support and the cap reads it as fully shielded;
- print times, including those of near-void cells, since only ratios count;
- elements below ~10ε, through the gate.

This may be how batch 14's wisps formed (a bar thinned below its support reads as
shielded), but that hasn't been checked against the runs and needs more investigation.

### Candidate: an objective shaping term (untested)

Keep A with the ε gate as the constraint, an honest reading for R0 and R3, and take the
pressure from a term in the objective:

$$f = c + \lambda\,\Phi, \qquad \Phi = \sum_e \operatorname{softplus}_\kappa\!\left(S^1_e - T_{cr}\right) g(x_e),$$

with `q = 1` in `S¹` and `g` increasing, `g(0) = 0`, `g'(0)` finite (e.g. `g = x`).

- R0 from the softplus; R1 from `g'` and `∂S¹_e/∂x_e ≥ 0`; R2 from `q = 1`,
  `∂Φ/∂x_j = −softplus'·g(x_e)·w_j / (D (x_e + ε))`, finite at `x_j = 0`; R3 from
  `g(0) = 0`. It also reaches the time field through `S¹`.
- Its gradients are consistent. If `Φ ≈ 0` at the constrained optimum, adding `λΦ ≥ 0`
  should leave that optimum in place (the exact-penalty argument), so Φ mainly shapes
  the path out of infeasibility.
- Softplus, not a squared hinge (user, 2026-10-19): A's constraint puts little pressure
  on a bar's overall density, so with a squared hinge a marginal violator would feel
  almost none from either. Below the margin softplus decays as `e^{κz}/κ`, a small
  safety margin like LogSumExp's.
- Caveats: `λ` needs tuning. A neighbor that would itself be unsupported resists
  filling with force `softplus(z_j)·g'(x_j)`, hence a finite `g'(0)`; under the capped
  ratio this starts only once it carries density above ~10ε, since void on void reads
  as a ratio of 1. Since only elements with mass pull, support may grow only about a
  filter radius at a time.
- Implemented as `hotspot_shaping_weight` / `hotspot_shaping_sharpness`
  (`FieldTerms.hotspot_shaping`, `g = x`, mean over elements). Unit test, 60×20 mesh, a
  grey bar on void: the bar's leading column is pushed out (+2.1e-4) and the void
  columns before it pulled in (−1.4e-4 to −2.9e-4); on solid support the gradient is
  ~1e-30.

## Formulations (`hotspot_shielding`)

Neighbor `j` contributes `c(u)` per unit weight, `u = (x_j + ε) / (x_e + ε)`:

- **A, `capped_ratio`:** `c = min(1, u)^q`. A neighbor at least as dense as `e` counts in
  full, so a grey element reads as it would once it and its support were solid.
- **B, `ratio`:** `c = u^q`, uncapped. One dense neighbor can make up for a missing
  direction, so a grey element on partial support reads as safe and solidifying it
  reveals the hotspot. Kept as a check that the cap is what matters.

Both agree with `absolute` on binary designs to within `ε`, so the binarized gates read
the same.

**The gate.** `S_e = 1 - K_e` alone fails: void at the mesh edge has part of its stencil
missing, reads as unshielded (~0.5 at a side wall) and no design change fixes it; and
void elements would be back in the constraint with a severity set by their own print
time, which PR #98 made drop out. So the severity keeps a gate, `x_e / (x_e + ε)`: zero
on void, flat above ~10ε, so it neither breaks invariance nor pushes grey density down.
ε = 1e-3. LogSumExp only (`ratio` is unbounded below, which `p_mean` cannot take).

## Batch 14 (2026-10-18)

Config B (`lap11_C103_B_n800`, `lap11_C105_B_n800`) at 800 iterations, `field_move`
0.02, with `hotspot_shielding` A or B, on C1 0.3 (Tcr 0.6) and C1 0.5 (Tcr 0.4), and
batch 13's sweep of f, the fraction of the way to the final Tcr at iteration 0
(`variant2_handoff.md`). No new controls: the user treats this as pass/fail. Batch 13's
C1 0.3 runs and `lap11_C105_B_n800` (f = 0) stand as the `absolute` reference.

Gates as in `variant2_handoff.md`: the binarized hotspot row with `t` kept ≤ 0.01, read
with `absolute` shielding (`bin_keep_t14.tmp.py`); compliance ≤ 1.15 × plain TO; ≤ 5
interior unsupported elements; print start. Driver `lap_driver2.tmp.py` (GPU + 3 CPU × 4
threads), batch script `lap_batch14.tmp.py`, code snapshot `output/lap/snapshot_4b2cedd`.
Dashboard: `lap_dashboard3.tmp.py`.

## Results (batch 14)

Hotspot row binarized with `t` kept (absolute shielding) / compliance ÷ plain TO. ✓ =
every gate passes. Every run: 0 interior unsupported, print start passes.

| cell | f | current (batches 11, 13) | A `capped_ratio` | B `ratio` |
|---|---|---|---|---|
| C1 0.3 | 0 | −0.028 / 1.193 | **−0.009 / 1.098 ✓** | +0.463 / 1.033 |
| C1 0.3 | 1/3 | −0.011 / 1.132 ✓ | +0.042 / 2.702 | +0.587 / 7.284 |
| C1 0.3 | 2/3 | −0.008 / 1.275 | −0.012 / 2.136 | +0.561 / 2.809 |
| C1 0.3 | 1 | +0.002 / 1.085 ✓ | −0.018 / 2.304 | +0.441 / 1.378 |
| C1 0.5 | 0 | +1.490 / 1.134 | +0.201 / 1.109 | +1.495 / 1.592 |
| C1 0.5 | 1/3 | – | +0.050 / 2.028 | +1.495 / 1.516 |
| C1 0.5 | 2/3 | – | −0.006 / 1.805 | +1.343 / 1.819 |
| C1 0.5 | 1 | – | +0.078 / 5.639 | +1.428 / 1.740 |

- **B fails everywhere**, as predicted: its as-optimized row is ≤ 0.03 in 6 of 8 runs
  while the binarized row is +0.44 to +1.50 — grey on partial support reads as safe and
  the hotspot appears on binarizing.
- **A at f = 0** passes C1 0.3 and cuts C1 0.5's binarized hotspot from +1.49 to +0.20.
- **A at f > 0** wrecks compliance (1.8–5.6 × plain TO). Later on the hotspot appears to
  sit on near-void wisps: on C1 0.3 at f = 1, iterations 100–300, the top-50 severities
  have median density 0.07–0.14 and 500–900 elements exceed Tcr, most with `x < 0.1`
  (`sev_density.tmp.py`); at f = 0 no element exceeds Tcr there. One possible reason,
  unconfirmed: the `x / (x + ε)` gate counts anything above ~10ε as material and is flat
  there, so nothing deletes a wisp.
- **The f > 0 failure starts in the first iterations** (2026-10-19, `early_hist.tmp.py`,
  `sev_edge.tmp.py`, `sev_edge2.tmp.py`). On C1 0.3 at f = 1, the max severity jumps from
  0.54 to 0.93 between iterations 0 and 2, on the top edge near the print start, while
  density is still uniform at ≈ 0.3. The hotspot row of `g` then stays near +0.63
  through iteration 50 with its multiplier at 2.5e3, the volume multiplier falls to
  ~1e-6, and the objective makes no progress (1507 at it 0, 1503 at it 50; 515 for the
  current formula at the same f). Meanwhile the time field collapses: at column 80,
  `t` goes from 0.43 to 0.03 by it 20, with local ripples. At it 20, 406 of 466 violators
  are within 3 rows of the top or bottom edge. B at f = 1 shows the same edge band; A
  at f = 0 shows the same jump at it 2, but Tcr is 1.0 there; the current formula's max
  stays ≤ 0.61 at every iteration checked (0–200).
  Tentative reading: capped-ratio shielding reads a uniform grey field as its solid
  version, which is honest but holds the early, unorganized time field to the final
  Tcr. Edge elements lose the outside part of their stencil, so they read highest. At
  it 0, neighboring print times differ by ≈ 0.005 per column, well under `field_move`
  0.02, so one MMA step can reorder a neighborhood; that may be what drives the jump.
  The current formula's `x^r` gate (≈ 0.55 at x = 0.3) keeps grey below 0.6 whatever its
  shielding, which hides hotspots but also shields the early phase. Which variables
  moved at iterations 1–2 hasn't been checked yet.
- Earlier idea, superseded by the requirements above: a step gate
  `x^k / (x^k + η_g^k)` with `η_g ≈ 0.2–0.3`.

Dashboard: `plot/laplace_s03/dashboard_shielding.html` (spec `lap_spec14.tmp.json`);
table `output/lap/eval14.json` (`lap_eval14.tmp.py`).

## Batch 15 (2026-10-19)

The user's call after batch 14: don't pursue starting the hotspot at iteration 0 (f > 0)
for now. Two tests, A (`capped_ratio`) only, since A vs B can't be decided from the data
yet. Config B at 800 iterations, C1 0.3 (Tcr 0.6) and C1 0.5 (Tcr 0.4):

- **step100:** the constraint off until iteration 100 (Tcr 1.0, the severity's ceiling),
  then the final Tcr at once. A prototype of activating on the grey fraction, which is
  the eventual intent.
- **phi10, phi100:** the f = 0 Tcr schedule plus the shaping term at λ = 10 and 100.
  These bracket where a typical violator's push equals the median log-compliance
  gradient on solid elements: ~16 on C1 0.3 (A f = 1, it 200) and ~70 on C1 0.5 (A f = 0,
  it 400; `shaping_scale.tmp.py`).

Same gates and references as batch 14. Code snapshot `output/lap/snapshot_3e1591d`,
batch script `lap_batch15.tmp.py`, eval `bin_keep_t15.tmp.py` + `lap_eval15.tmp.py`,
dashboard `lap_dashboard4.tmp.py` (severity − Tcr on RdBu) with spec
`lap_spec15.tmp.json`.

## Results (batch 15)

Hotspot row binarized with `t` kept (absolute shielding) / compliance ÷ plain TO. ✓ =
every gate passes. Every run: 0 interior unsupported, print start passes.

| cell | A, f 0 (batch 14) | step100 | phi10 | phi100 |
|---|---|---|---|---|
| C1 0.3 | −0.009 / 1.098 ✓ | **−0.018 / 1.086 ✓** | −0.006 / 1.137 ✓ | −0.017 / 1.241 |
| C1 0.5 | +0.201 / 1.109 | **−0.016 / 1.059 ✓** | +1.112 / 1.063 | +0.023 / 1.069 |

- **step100 passes both cells**, C1 0.5 for the first time in any formulation. At
  activation the max severity was 0.78 (C1 0.3) and 0.61 (C1 0.5), against 0.93 at
  iteration 2 under f = 1: by iteration 100 the density had organized (compliance ~390
  vs ~1500 at the start). The row was violated by 0.30 / 0.53, the multiplier sat at
  its cap (`mma_c` 2500), and the row was back to ≤ 0.01 by iteration ~150 in both.
  One run per cell, so this may not be robust; a possible reason, untested, is that
  the final Tcr then binds through the grey phase (β_d = 4, it 120–350) while the
  topology can still change.
- **The shaping term is inconclusive.** λΦ stays ≤ ~1e-3 against log compliance ~5,
  since it only counts would-be violators. Runs that differ only by it end far apart
  (C1 0.5: +0.20, +1.11, +0.02), which looks more like path sensitivity than an
  effect of the term.
- **phi10 on C1 0.5 broke late.** Feasible at iteration 556 (max 0.397); by 560 the
  max was 0.79 on the bottom edge near the print start, with the density unchanged:
  the time field along the bottom edge developed a local dip (t 0.078, 0.075, 0.078
  across three columns), so an element printed before its left neighbors. The
  hotspot multiplier stayed ~1e-7 for several iterations while the row was violated,
  and the run never recovered (`blowup15.tmp.py`). The same fragility as the f = 1
  failure; whether the shaping term triggered it is unknown.

Dashboard: `plot/laplace_s03/dashboard_shaping.html` (spec `lap_spec15.tmp.json`);
table `output/lap/eval15.json` (`lap_eval15.tmp.py`).

### Batch 15b: the shaping term on the step schedule (2026-10-19)

The step100 schedule plus the shaping term at λ = 10 and 100 (`lap_batch15b.tmp.py`).

| cell | step100 | step100 + λ 10 | step100 + λ 100 |
|---|---|---|---|
| C1 0.3 | −0.018 / 1.086 ✓ | −0.015 / 1.078 ✓ | −0.008 / 1.094 ✓ |
| C1 0.5 | −0.016 / 1.059 ✓ | +1.495 / 1.216 | **−0.051 / 1.047 ✓** |

- Five of six step runs pass; with λ = 100, C1 0.5 has its lowest compliance yet.
- **λ = 10 on C1 0.5 failed on both schedules.** On the step schedule the row was
  violated by 0.65 at activation, escalated to ~1.4 by iteration 150 (max severity
  ~0.97, on the top and bottom edges) and never recovered, with the multiplier at its
  cap for the rest of the run. Two of two is suggestive, but these runs look
  path-sensitive: before iteration 100 the three C1 0.5 step runs have the same
  compliance to within 0.1, yet their max severity swings between 0.6 and 0.94 from
  iteration to iteration and run to run (the shaping term is ~1e-9 then, since
  Tcr = 1.0). At activation plain step100 also went up (0.61 → 0.76 at it 120) before
  recovering, so whether a run recovers may come down to the state at the step.

Dashboard: `plot/laplace_s03/dashboard_shaping_step.html` (spec `lap_spec15b.tmp.json`).

## Batch 16: robustness and generality of step 100 + λ 100 (2026-10-19)

User decisions:
- Tentatively adopt step 100 + λ 100 and test how robust it is.
- **One fixed hyperparameter set for every problem, no per-cell rescaling.** Log
  compliance makes the weights relative, so rescaling by each cell's plain-TO compliance
  (as batches 11–15 did) only reproduced `stto`'s cell-to-cell drift. The fixed set is
  `lap11_C105_B_n800`'s: uniformity 0.115, time-only hotspot 0.288, roughness 1.15e-3 →
  1.15e-4.
- No comparison against `stto` or a λ = 0 baseline: the question is whether Variant 2
  works, period. Gates unchanged.
- Cells: C1 at Tcr 0.6 only (plus the C1 0.5 Tcr 0.4 robustness cell), no x cells, no C1
  0.7, one C2 (0.3, left edge, Tcr 0.6), one S printed from the left (0.3, Tcr 0.6), and
  the L brackets at Tcr 0.6 once `stto_laplace` supports them.

Runs (`lap_batch16.tmp.py`, snapshot `output/lap/snapshot_3e1591d`):
- Robustness on C1 0.3 (Tcr 0.6) and C1 0.5 (Tcr 0.4): step 80 and 120 at λ 100; λ 30 and
  300 at step 100; and C1 0.3 step 100 + λ 100 again under the fixed weights (its batch
  15b run used ~1.5× smaller ones). The C1 0.5 reference is batch 15b's.
- Geometries, step 100 + λ 100: C1 0.5 e Tcr 0.6, C2 0.3 e Tcr 0.6, S 0.3 e Tcr 0.6.
  Plain-TO references: `val_C1_v0.5_e_plainTO`, `val_C2_v0.3_e_plainTO`,
  `val_S_v0.3_b_plainTO` (plain TO does not depend on the print base: within 0.05%).
- L corner and L middle at Tcr 0.6 (120×120, volfrac 0.5, from the clamped right edge),
  from PR #213's code (`snapshot_a8294d7`, `lap_batch16L.tmp.py`): `t` is solved on the
  L only, with wall data on every L wall but the clamped edge.

## Results (batch 16)

Binarized hotspot row with `t` kept / compliance ÷ the cell's plain TO. ✓ = every gate
passes. Every run passes start; all but C1 0.3 step 120 (1 element) have 0 interior
unsupported. Eval `lap_eval16.tmp.py` → `output/lap/eval16.json`.

**Robustness** (one knob at a time around step 100 + λ 100):

| | C1 0.3 (Tcr 0.6) | C1 0.5 (Tcr 0.4) |
|---|---|---|
| step 100, λ 100 | −0.021 / 1.135 ✓ | −0.051 / 1.047 ✓ (batch 15b) |
| step 80 | −0.002 / 1.099 ✓ | −0.029 / 1.056 ✓ |
| step 120 | **+0.663 / 1.282** | **+1.495 / 1.234** |
| λ 30 | −0.005 / 1.091 ✓ | **+1.443 / 2.302** |
| λ 300 | −0.019 / 1.150 (compliance 1.1504, just over) | −0.069 / 1.057 ✓ |

**Geometries** (step 100 + λ 100):

| cell | result |
|---|---|
| C1 0.5 e, Tcr 0.6 | **+0.600 / 1.687** |
| C2 0.3 e, Tcr 0.6 | −0.173 / 0.999 ✓ |
| S 0.3 e, Tcr 0.6 | −0.255 / 1.003 ✓ |
| L corner, Tcr 0.6 | −0.034 / 1.006 ✓ |
| L middle, Tcr 0.6 | **+0.515 / 1.218** |

- **Not robust: 9 of 14 pass.** Every failure is the same mode, seen before in batch
  15/15b's λ = 10 runs: the row is violated at the step, the max severity climbs to
  ~0.9–1.0 within ~50 iterations, mostly on the mesh's top and bottom edges, and the
  run never recovers, with the multiplier alternating between ~1e-7 and its cap
  (`mma_c` 2500). A multiplier near zero on a violated row means MMA's linear model
  expects the step to clear the violation, and the time-field step then does not.
- **Failures are not monotone in the knobs.** Step 120 failed both cells; step 80 passed
  both. Lower λ failed C1 0.5 (10, 10, 30) but not C1 0.3 (30). C1 0.5 passes at the
  stricter Tcr 0.4 but fails at 0.6. With one run per setting, this reads more like a
  run-to-run bifurcation at the step than a trend.
- **C2 and S pass with wide margins** (−0.17, −0.26): Tcr 0.6 barely binds there.
- **L corner passes;** its final time field carries sharp features in the wall data along
  the inner vertical wall, on void and outside the L. **L middle fails** with the time
  field over the whole bottom arm nearly flat near 0, a flat time basin, the failure
  `stto` also has on this cell at Tcr 0.6 (+0.607).

Dashboard: `plot/laplace_s03/dashboard_batch16.html` (`lap_dashboard5.tmp.py`, spec
`lap_spec16.tmp.json`, both in the `laplace-l-bracket` worktree; `stto` |Δt| reference
dropped).

## Edge noise in the severity, and a continuation on the angular lobe (2026-10-19)

**The user's observation:** along the mesh's top edge the hotspot severity is much rougher
than the time field (C1 0.5 Tcr 0.4 step 100 + λ 100, e.g. iteration 65).

**Cause, measured** (`edge_noise.tmp.py`): the angular lobe. At iteration 65 the top
row's severity changes by 0.073 per column on average against 0.011 in row 30, while `t`
changes by 0.005 in both; at κ = 0 the top row's roughness falls to 0.021. The lobe reads
the direction of ∇t, and on the top row that direction swings by ~21° per column (1° in
the interior): `dt/dy` there is one-sided (row 0 − row 1), row 0's `t` is half wall
data, and the wall data has small plateaus (the x-slope varies 0.001–0.006 per column
against `dt/dy` of ±0.002). Half the stencil lies outside the mesh and counts as
unshielded, so whether "behind" points in or out decides most of the shielding: e.g.
−45° gives severity 0.16, +47° gives 0.80 three columns away. Not verified: that this is
what makes MMA's model fail after the step.

**User decision:** a continuation on `hotspot_kappa` rather than changing the measure:
linear from 0 (isotropic) at iteration 0 to 2.3666 at 350, where the first two phases end.
Checked first: the row's value and its gradient (w.r.t. `xPhys` and `tPhys`) are
continuous as κ → 0, relative change ∝ κ (~1e-6 at κ = 1e-6; `kappa_limit.tmp.py`,
`kappa_grad.tmp.py`), so the ramp starts at 0. Caveat: on an organized design the
isotropic stencil reads higher (C1 0.5 at it 400: 0.62 at κ = 0 vs 0.41 at 2.37), so with
the ramp the step at iteration 100 meets a stricter effective constraint.

### Batch 17

C1 0.5 step 100 + λ 100 with the κ ramp, everything else as batch 16
(`lap_batch17.tmp.py`; GPU, ~10 min each).

| | κ 2.37 throughout | κ ramp 0 → 2.37 |
|---|---|---|
| Tcr 0.4 | −0.051 / 1.047 ✓ (batch 15b) | −0.030 / 1.054 ✓ |
| Tcr 0.6 | +0.600 / 1.687 (batch 16) | **−0.015 / 1.022 ✓** |

- **Tcr 0.6 now passes.** At the step the row is +0.048 (batch 16: +0.240), back on the
  constraint by iteration 105, and the multiplier stays small (≤ 0.02) through iteration
  350; batch 16's run escalated to ~0.97 with the multiplier at its cap.
- **Tcr 0.4:** +0.575 at the step, cleared smoothly by iteration ~150. Both runs bump at
  iteration ~400 (β_d ramp, +0.10 / +0.13) and recover.
- **Top-row roughness** (severity at the run's own κ): iteration 65 0.073 → 0.025, 200
  0.023 → 0.014, 300 0.023 → 0.016; the direction swing on the top row drops too (20.8° →
  16.7° at 65, 13.4° → 9.4° at 200).
- One run per setting, on one cell; batch 16's failures elsewhere are untested with the
  ramp.

Dashboard: `plot/laplace_s03/dashboard_kappa_ramp.html` (spec `lap_spec17.tmp.json`).

Showing the time field over the whole domain (user, 2026-10-19; now the dashboards'
default) shows every failed run in batches 15–16 with `t` in the void far above the
part's range (largest `t` 3.3–425, the part normalized to 1), every passing run ≤ 1.75.
It follows the escalation rather than leading it (C1 0.5 Tcr 0.6, batch 16: row +0.6 at
iteration 130, max `t` leaves 1 at 140), so it marks a failed run, not a cause.

## Batch 18: grey "time wires" and `chi_density_exponent` (2026-10-19)

**The user's hunch:** the near-disconnected island in the lower-right cell of the κ-ramp
run at Tcr 0.6 is what is left of a thin grey strip that shaped the time field rather
than carried load, staying thin until binarization forced it solid.

**Checked on the saved design** (`strip_role.tmp.py`): at iteration 300 the strip is 186
grey elements, mean `xPhys` 0.15, running from the diagonal member to the bottom flange.
Removing it raises compliance by 0.14% (0.09% at 350), but changes `t` on the part by up
to 0.059, about six cooling times τ: with it, the member above sits on a raised plateau
(`t` ≈ 0.836–0.840), without it `t` rises steadily from 0.776 to 0.84 across that stretch.
With `chi ∝ x^e`, a strip at `x` ≈ 0.15 conducts ~2000× better than void (floor 1e-5), so
grey material works as a cheap time wire.

**Test:** the κ-ramp run with every `chi_density_exponent` point doubled (2/2/3/5 →
4/4/6/10) or tripled (→ 6/6/9/15) (`lap_batch18.tmp.py`, `lap_batch18b.tmp.py`; GPU, ~7
min each).

| exponent | hotspot / compliance | final grey | strip / island |
|---|---|---|---|
| ×1 (batch 17) | −0.015 / 1.022 ✓ | 0.033 | strip from ~200, island |
| ×2 | −0.038 / 1.023 ✓ | 0.028 | neither; a clean triangular cell |
| ×3 | **+0.054** / 1.024 (+0.010 as optimized) | 0.039 | neither, but two thin grey members remain at the end, and small holes in the bottom-right flange |

- Doubling removes the strip and the island and passes with a larger margin. Tripling
  removes them too, but stays grey longer (iterations 300–400) and ends with grey
  members; at an exponent of 15 those cannot conduct `t`, so they are not time wires,
  which looks like a side effect on how cleanly the density resolves.
- One run each, on one cell.

Dashboard: `plot/laplace_s03/dashboard_chi2x.html` (spec `lap_spec18.tmp.json`).

## Batch 19: κ ramp + doubled exponent on batch 16's failures (2026-10-19)

Batch 18's ×2 settings (κ ramp, every `chi_density_exponent` point doubled) on the four
cells batch 16 failed, plus the C1 0.5 Tcr 0.4 reference (`lap_batch19.tmp.py`). Gates as
in batch 16 (binarized hotspot with `t` kept ≤ 0.01, compliance ≤ 1.15× plain TO, ≤ 5
interior unsupported, print start).

| cell, schedule | batch 16 (or 15b) | κ ramp + ×2 | |
|---|---|---|---|
| C1 0.5 Tcr 0.4, step 100 λ 100 | −0.051 / 1.047 ✓ (15b) | −0.000 / 1.068 | ✓ (on the limit) |
| C1 0.5 Tcr 0.4, step 100 λ 30 | +1.443 / 2.302 ✗ | −0.004 / 1.057 | ✓ |
| C1 0.3 Tcr 0.6, step 120 | +0.663 / 1.282 ✗ | −0.007 / **1.226** | ✗ compliance |
| C1 0.5 Tcr 0.4, step 120 | +1.495 / 1.234 ✗ | +1.495 / 1.330 | ✗ |
| L middle 0.5 Tcr 0.6 | +0.515 / 1.218 ✗ | +0.513 / 1.143 | ✗ hotspot |

- λ 30 now passes and the C1 0.3 step-120 hotspot passes; the step-120 run at Tcr 0.4
  still escalates (void `t` up to 152), so step 120 still looks fragile at Tcr 0.4.
- The L middle run fails with nearly the same hotspot value as batch 16, but unlike the
  C1 failures its time field stays in range (largest `t` 1.27), so the void-`t` marker
  above may not carry over to the L. Its continuous severity is only mildly above Tcr,
  over the lower arm next to the plate, where `t` is close to 0; not yet looked into.
- The reference cell's margin shrank from −0.051 to −0.000; with one run per cell it is
  unclear whether that is the settings or run-to-run spread.

**Timing.** The reference cell ran on the CPU alone at 8 threads (the laptop back in
performance mode): 31.0 min of iteration time against 50–60 min for batch 15–16's CPU
runs at 4 threads, three at once, consistent with the 3×4 vs 1×8 throughput note. The
laptop slept 21:51–22:04; the runs resumed, and `elapsed` in `iterations.jsonl` leaves
the sleep out (the driver's wall-clock minutes include it). The failing step-120 Tcr 0.4
run took ~2 s/iteration on the GPU even after the CPU run ended, against ~0.5 for batch
18's runs; the cause is not yet known (MMA's subsolver hits its inner-loop cap 5–9 times
in every batch 19 run, passing or not).

Dashboard: `plot/laplace_s03/dashboard_b19.html` (spec `lap_spec19.tmp.json`).

### The L middle collapse (batch 19, `collapse_*.tmp.py` in the laplace-l-bracket worktree)

From iteration ~410 to ~470 almost all of the part's time variation drains into the upper
arm near the re-entrant corner; the lower arm ends up printed nearly all at once (median
`t` 0.003 at 470).

- **Per element, simultaneous printing does read as unshielded**: at 470 the lower arm's
  median severity is 0.87 (Tcr 0.6), and 4177 of 5399 part elements are above Tcr. Up to
  iteration 350, though, the cooling credit's sharpness is 1, so a tie gets credit 0.5;
  with the half-stencil divisor a solid neighbourhood printed all at once then scores
  `K_est` ≈ 1, i.e. no severity. From 350 to 450 the sharpness ramps to 8 (tie credit
  0.08).
- **The row went first**: it violates from iteration 360 (multiplier at the 2500 cap
  from then on), with one worst element (`n_eff` ≈ 1) on the upper arm's top edge (row 0,
  columns 40–50). The collapse comes after.
- **The collapse is the wall data falling**: median wall data 0.32 → 0.009 from 400 to
  470, which shorts the lower arm to the plate's `t = 0`. The normalizer (raw `t`'s max
  over the part) barely moves (0.45 → 0.37), pinned at the upper arm's inner edge.
- **Why the row allows it**: its gradient is the smooth maximum's (β 100), so it sees only
  the current worst few elements. Linearized, the hotspot row favours that move at every
  snapshot checked (380–420): steepening `t` around the worst element shields it, while
  flattening elsewhere, which the gradient does not see. The actual row does not improve
  (0.27 → 0.60 over 380–425, oscillating), and once the maximum is near 1, more violating
  elements barely change it. The shaping term (a mean, so it counts the extent) also
  favoured the move at 420, for reasons not yet looked into.
- **A possible link to the initial time field**: the wall data starts as the Euclidean
  ramp from the plate, so the upper arm's inner wall starts at nearly one time (0.50–0.56
  along it), where a geodesic distance would rise from the corner to the top. At 350 `t`
  along that wall still peaks mid-arm (0.77) and falls to 0.63 at the top edge, where the
  worst element sits; batch 16's L middle run also escalated from row 0, columns 43–47.
  Untested.

## Batch 20: L brackets with the geodesic wall start (2026-10-19)

PR #213 bce86b0 starts the wall data at the walls' distance to the plate through the
domain (8-connected grid geodesic, as stto's `time_init = geodesic`); identical to the
old ramp on rectangles. lap19's L settings otherwise (`lap_batch20.tmp.py`).

| cell | earlier | geodesic start |
|---|---|---|
| L middle 0.5 Tcr 0.6 | +0.513 / 1.143 ✗ (batch 19) | +0.527 / 1.135 ✗ |
| L corner 0.5 Tcr 0.6 | −0.034 / 1.006 ✓ (batch 16) | +0.006 / 1.012 ✓ |

The start along the upper arm's inner wall now rises from the corner to the top (0.42 →
0.77, against 0.50–0.56), but L middle escalated earlier: at the Tcr step (true maximum
0.73 at iteration 100), a collapse by 175, partial recovery over 225–350, a second
escalation from 375, ending with the worst elements on the lower arm's bottom edge, a
free edge along the build direction. So the start was probably not the main cause.

Dashboard: `plot/laplace_s03/dashboard_b20.html` (spec `lap_spec20.tmp.json`).

## Is Tcr 0.4 reachable by the time field alone? (2026-10-20)

**An ideal print** (`ideal_block.tmp.py`): the whole C1 domain solid, `t` the linear ramp
from the plate, under lap19 C1 0.5 Tcr 0.4's settings. The interior stays at 0.17–0.23,
but the top and bottom rows, free edges along the build direction, sit at 0.43 (κ 2.37,
cooling sharpness 1) to 0.48 (sharpness 8). Half of their stencil is outside the domain
and counts as unshielded. Flat layers are not the best print of a bar, though (user): with
the front flat in the middle and lagging toward the edges, the gradient near an edge
tilts outward and the lobe stays in the bar (`ideal_bar.tmp.py`, a parabolic front with
the edges `a h` elements behind the axis). At the final settings, best over `a`: a bar
8 wide reaches 0.36, 16 wide 0.19, 30 wide 0.16, all under 0.4; at iteration 130's
settings 0.41, 0.24, 0.18. So edges along the build direction are not an inherent bound,
except for bars narrower than ~8 elements. The edge lag that does it is a few elements,
i.e. `t` differences of ~2 cooling times.

**Time alone on a frozen density** (`time_only.tmp.py`): lap19 C1 0.5 Tcr 0.4 step 120's
density at iteration 130 (60% grey), the run's own severity at iteration 130's settings,
time field only.

- MMA at the run's move limit (0.02, twice the cooling time 0.01) makes it worse in one
  step (true maximum 0.545 → 0.89–0.94) for both parameterizations, with no other term in
  the objective: the same jump the run made over 120–150. Smaller moves (0.002, 0.0005)
  are stable but barely move it.
- Adam on a smooth maximum sharpening from β 20 to 400, 2000 iterations, best true
  maximum: direct `t` 0.521, Variant 2 (μ, wall) 0.501, direct `t` on the 0.5-binarized
  density 0.527. None reaches 0.4; the remaining violators lie along the long top and
  bottom bands, i.e. edges along the build direction.
- Given the curved-front result, this is inconclusive about the formulation: neither
  optimizer found a time field below 0.4, but one likely exists for bars this wide.
  Variant 2 did no worse than free `t`.
- **Variant 2 with MMA, hotspot row only, 100 iterations**: at the run's move limit
  (0.02) the true maximum goes 0.545 → 0.96; at 0.005 → 0.86; at 0.002 it falls steadily
  to 0.512. So on a frozen density the time-field steps alone, at the run's trust region,
  escalate the severity: the fast-changing time field can be a cause, not only a symptom.
  In the runs, the time field's largest change per iteration after the Tcr step is
  0.026–0.033 (median over 100–200) in the failing ones and 0.002–0.008 in the passing
  ones, against a cooling time of 0.01.

## Element-scale stripes in the diffusivity field (2026-10-20)

χ<sub>μ</sub> alternates high/low every element in layers across the flow, e.g. along the
passing L corner's bottom chord next to the plate (lap20_Lc05, `mu_stripes.tmp.py`,
plot `plot/laplace_s03/stripes/`). Roughness alone does not separate passing from
failing runs; the share of the part with μ at a bound does (upper bound 0–4% passing,
24–52% failing).

- **A near-null mode.** μ enters χ unfiltered. In 1-D series flow the element-mean `t`
  differences depend only on adjacent-pair sums of the resistance 1/χ, so an alternation
  of 1/χ is invisible apart from a constant offset; nodal `t` becomes a staircase (row
  112 at iteration 400: node steps alternate 0.002/0.010, element-mean steps 0.0066–0.0070).
  In 2-D it is nearly so: removing the alternation in that window (`mu_destripe2.tmp.py`,
  0.37 decades of log χ on average) moves tPhys by ≤ 0.003 and the severity by ≤ 0.05,
  against 0.018 and 0.18 for a uniform change of the same size.
- **A weak push.** In that window the uniformity term's gradient has a small consistent
  component along the stripes over iterations 100–300 while they grow (μ amplitude 0.001
  → 0.076), and the moves follow it; it reverses by 400 and growth stops
  (`mu_stripe_drive.tmp.py`). The hotspot term barely acts there (λ ~ 1e-7), and the
  roughness term, decayed to ~1e-4, contributes nothing measurable. One window of one run.
- So with nothing restoring it, a small push accumulates into stripes that cost MMA
  steps and push μ to its bounds without changing the print. A filter on μ (as on the
  density) would remove the mode at its source.

## Batches 21–22: the time-field move limit, and a filter on μ (2026-10-20)

Batch 21 (`lap_batch21.tmp.py`): `field_move` 0.02 until the Tcr step, then 0.002.
Batch 22 (`lap_batch22.tmp.py`, branch `laplace-mu-filter` eb964ac): each base config plus
`mu_filter_rmin_m` = 2 elements, move 0.02 throughout. Binarized row with `t` kept / C ÷
plain TO / grey:

| cell | base | 21: move 0.002 after step | 22: μ filter |
|---|---|---|---|
| L middle 0.5 Tcr 0.6 | +0.527 / 1.135 ✗ | −0.052 / 1.042 / 1.8% ✓ | +0.543 / 1.158 / 14% ✗ |
| C1 0.5 Tcr 0.4 s120 | +1.495 / 1.330 ✗ | −0.009 / 1.099 / 7.7% ✓ | +0.313 / 1.113 / 4.9% ✗ (as optimized +0.010) |
| C1 0.5 Tcr 0.4 s100 | −0.000 / 1.068 ✓ | +0.429 / 1.139 / 14% ✗ (as optimized +0.014) | +0.089 / 1.107 / 9.1% ✗ (as optimized +0.011) |
| L corner 0.5 Tcr 0.6 | +0.006 / 1.012 ✓ | – | +0.528 / 1.106 / 3.7% ✗ |

- The cut move limit rescues both failing cells; L middle passes for the first time. It
  also keeps μ off its bounds (part share at the upper bound 0–2%, against 11–42% in the
  base runs). But it slows binarization: the s100 control ends 14% grey (base 4%) and
  fails once binarized, the risk noted beforehand. A move limit between 0.002 and 0.02,
  or one that relaxes as the density binarizes, is the next thing to try.
- The μ filter does what it was meant to (stripe amplitude p90 0.02–0.06 in log χ,
  against 0.2–0.5 unfiltered) but does not stop the saturation: L middle's filtered μ
  still sits at a bound over 15–20% of the part, in blocks, and fails as before. It also
  breaks both passing cells (C1 s100, L corner). So the stripes were not what decides a
  run, and removing the mode is harmful as it stands: possibly the filter cuts the
  element-scale control of χ the hotspot needs near edges (not checked).

Dashboard: `plot/laplace_s03/dashboard_b21.html` (spec `lap_spec21.tmp.json`).

## Batch 23: phase 2 held 200 iterations longer (2026-10-20)

User's hypothesis: after the Tcr step the optimizer spends its effort moving material to
meet the constraint and runs out of time to optimize the topology before the schedule
freezes it. Test: every schedule held at its iteration-350 value for 200 iterations
(1000 in all; the small roughness weight's cosine decay stretched instead,
`lap_batch23.tmp.py`).

| C1 0.5 Tcr 0.4 s100 | without the hold | with the hold |
|---|---|---|
| move 0.002 after the step (batch 21) | +0.429 / 1.139 / 14% ✗ | **−0.021 / 1.058 / 2.7% ✓** |
| move 0.02 (batch 19) | −0.000 / 1.068 ✓ | +1.385 / 1.585 / 27% ✗ |

- With the small move limit the extra time fixes it, better than batch 19's pass.
- The move-0.02 hold run says nothing about the hold: it collapsed at iteration 115–125,
  before the hold, where its schedules equal batch 19's. The only difference there is
  the roughness weight (0.4% apart, ~5e-6 absolute); density differences reach 0.02–0.05
  by iteration 50–100, and at the step one run recovers (worst severity 0.54 at 120) and
  the other escalates (0.89 at 125). So at move 0.02 the step is a coin flip, and batch
  19's pass may have been luck: single runs per cell cannot rank settings there. None of
  the four move-0.002 runs collapsed after the step.

**Time alone at the step** (`time_only.tmp.py`, the move-0.002 run's iteration-100 density
and settings): worst severity 0.638 as the run had it; MMA at 0.002 reaches 0.607 and
stalls (1000 iterations); MMA at 0.02 escalates to 0.99; Adam on Variant 2's variables
0.601. The run itself reached 0.40 by iteration 150 by moving material (density changes
twice batch 19's over 100–200, compliance 204 → 260).

## Batch 24: Tcr ramps instead of the step (2026-10-20)

User decision: move 0.002 is the working default for now; the slower schedule it needs
(phase 2 held 200 longer) is an accepted cost, to refine later. The μ filter makes the
design field more sensible, so it is retested at move 0.002 (batch 25).

C1 0.5 Tcr 0.4 from the passing step run (move 0.002 after the step, phase-2 hold),
changing only Tcr and where the move limit drops (`lap_batch24.tmp.py`):

| Tcr | binarized, t kept / C ÷ TO / grey | hotspot binding | mean \|ΔxPhys\| 100–150, 150–200 | C at 150 |
|---|---|---|---|---|
| step at 100 (lap23) | −0.021 / 1.058 / 2.7% ✓ | from 100 | 0.147, 0.134 | 275 |
| ramp A, 1 → 0.4 over 0–100 (move 0.002 from 0) | −0.017 / 1.062 / 2.6% ✓ | from ~70 | 0.075, 0.037 | 207 |
| ramp B, 1 until 100, → 0.4 by 200 | +0.070 / 1.072 / 3.8% ✗ (as optimized +0.008) | from ~160 | 0.026, 0.110 | 201 (258 at 200) |

- Ramp A meets the constraint as it tightens: at iteration 100 the worst severity is
  0.42 against the step run's 0.65, and the density moves half as much afterwards. The
  final designs end up alike (C 186–189), so the gentler approach shows in the transient,
  not in the result, at least with the phase-2 hold.
- Ramp B arrives at Tcr 0.4 with the same 0.24 excess as the step, 100 iterations later,
  and fails narrowly once binarized. One run each.

## Batch 25: the μ filter at move 0.002, with repeats (2026-10-20)

Every run at move 0.002 after the Tcr step; μ filter radius 2 elements; each config run
twice, the repeat with the roughness weight × (1 + 1e-4) (`lap_batch25.tmp.py`).
Binarized row with `t` kept / C ÷ plain TO / grey:

| cell | no filter | no filter, repeat | μ filter | μ filter, repeat |
|---|---|---|---|---|
| L middle 0.5 Tcr 0.6 | −0.052 / 1.042 / 1.8% ✓ | −0.020 / 1.033 / 1.8% ✓ | +0.055 / 1.039 / 3.0% ✗ | −0.072 / 1.036 / 1.4% ✓ |
| C1 0.5 Tcr 0.4 s120 | −0.009 / 1.099 / 7.7% ✓ | −0.023 / 1.100 / 6.8% ✓ | +0.044 / 1.183 / 10% ✗ | +0.191 / 1.171 / 12% ✗ |
| C1 0.5 Tcr 0.4 s100 + hold | −0.021 / 1.058 / 2.7% ✓ | −0.014 / 1.045 / 2.4% ✓ | +0.006 / 1.098 / 4.2% ✓ | +0.002 / 1.086 / 4.1% ✓ |
| L corner 0.5 Tcr 0.6 | −0.072 / 1.015 / 1.8% ✓ | −0.068 / 1.017 / 1.7% ✓ | −0.039 / 1.011 / 1.4% ✓ | −0.032 / 1.011 / 2.0% ✓ |

- **Move 0.002 holds up under the perturbation**: 8 of 8 without the filter pass, and
  each repeat lands near its original (at move 0.02 the same perturbation split a pass
  from a collapse).
- **The μ filter costs**: 5 of 8 pass. On C1 it ends greyer (4–12% against 2–8%) with
  worse compliance (1.086–1.183 against 1.045–1.100), and C1 s120 fails both times; on the
  L brackets it is about even (L corner compliance 1.011 against 1.016, L middle split
  1–1). So the filter makes χ<sub>μ</sub> readable but, at radius 2, makes C1 worse; why is
  not checked (one guess: the hotspot uses element-scale χ near member edges).

The user's call (2026-10-20): no μ filter for now, kept in mind because the unfiltered
χ<sub>μ</sub> looks rough. A narrower χ range (`chi_contrast` 100 and 10, batch 26) was
queued and then put on hold (`output/lap/backlog/batch26`) for batch 27.

## Batch 27: is ramp A repeatable, and a softer hotspot aggregation (2026-10-20)

All C1 0.5 Tcr 0.4 ramp A (batch 24). The replicate has the roughness weight × (1 + 1e-4);
the β runs hold `hotspot_beta` at 50 or 20 until 550, then ramp it (log) to 100 by 650,
alongside binarization. Binarized row with `t` kept / C ÷ plain TO / grey:

| run | result | as optimized |
|---|---|---|
| ramp A (batch 24) | −0.017 / 1.062 / 2.6% ✓ | +0.009 |
| ramp A, replicate | +0.052 / 1.062 / 2.3% ✗ | +0.007 |
| β 50 → 100 | +0.012 / 1.083 / 3.9% ✗ | +0.013 |
| β 20 → 100 | −0.057 / 1.059 / 2.3% ✓ | −0.049 |

- The replicate tracks the original closely (compliance, grey and row within ~1% through
  the run; first visible split at iteration 88), but it fails once binarized. So ramp A's
  pass is marginal: the binarization step decides it, not a different trajectory.
- β 20 stays much greyer until 550 (0.49 grey against 0.34) and then binarizes quickly. It
  is the only run of the four whose row settles below 0.01 well before the end (from 708),
  with margin after binarizing. One run.
- At β 100 the row's softmax already spreads over ~250–1100 elements (1/Σw²); β 20 spreads
  over ~3000 of 10800.

### What keeps ramp A grey (`grey_pressure.tmp.py`)

Per term on grey elements (0.05 < xPhys < 0.95): the gradient with respect to the density
design, constraints weighted by their logged λ. Compliance pushes density up and volume
pushes it down, and the two cancel to within a few percent, so what matters is how the
push differs between grey above 0.5 and below it. A push that is the same everywhere is
absorbed by λ_vol. The table gives that above-minus-below difference at iteration 185
(same penal and β_d in both runs); positive binarizes:

| term | ramp A | plain TO |
|---|---|---|
| compliance | −0.098 | +0.033 |
| volume | +0.202 | +0.106 |
| hotspot (λ 0.20) | −0.063 | — |
| uniformity + shaping | −0.025 | ~0 |
| total | +0.018 | +0.138 |

- The hotspot row adds material almost everywhere (on 99% of grey below 0.5 and 65% above),
  most strongly to low-density grey. That slows binarization.
- In ramp A, compliance itself pushes low-density grey up harder than high-density grey,
  which is the opposite of plain TO. Not checked why.
- Plain TO stalls at 0.31 grey while β_d is 4. It binarizes during its β_d ramp (350–450),
  which the hold moves to 550–650 in ramp A. At the point each leaves β_d = 4, the grey
  fractions are similar (0.31 against 0.34).
- At 385, ramp A's difference matches plain TO's at the same grey fraction (iteration 185),
  yet its grey moves half as fast. Not explained: possibly MMA's asymptotes, since ~25× more
  grey variables are at the floor.

### Why ramp A's grey moves slowly: MMA's step, split (`asy_replay.tmp.py`)

The density step of each grey variable is its asymptote distance s (MMA's step budget)
times the step per unit s (the gradient imbalance). Replayed with the current code:

| | ramp A at 185 | plain TO at 185 | ramp A at 385 |
|---|---|---|---|
| grey at the s ceiling (0.2) | 53% | 97% | 22% |
| median step per unit s | 0.003 | 0.008 | 0.020 |
| mean step toward a bound | +0.00026 | +0.00184 | +0.00141 |
| the same with every grey s at the ceiling | +0.00059 | +0.00184 | +0.00322 |

At 185 the imbalance accounts for about 3× and the shrunken asymptotes for about 2.3×. At
385 the asymptotes are the limit. s shrinks because whole regions of the grey reverse
direction together, in bursts: a mean of 24% of the grey per iteration in ramp A, against
1% in plain TO. The bursts barely follow swings in either multiplier.

### Line sweeps through a burst (`line_sweep.tmp.py`)

These start from replayed states and run along the step MMA took. In this replay the
burst fell at steps 206–208 rather than 209–212. Each value is the actual change at the
step divided by the linear prediction:

| step | grey reversing | hotspot row | Lagrangian | density part only: hotspot row | time part only: hotspot row |
|---|---|---|---|---|---|
| 206–208 (burst) | 53–56% | 0.24–0.72 | 0.61–0.84 | −0.86 to +0.52 | 0.95–0.96 |
| 209–211 (calm) | 7–18% | 0.80–0.89 | 0.92–0.96 | 1.30–1.57 | 0.96–0.98 |

In the burst steps, the hotspot row's response to the density part of the step doesn't
even keep the linear prediction's sign. MMA's own model still predicts 0.84–0.91 of the
linear change. The time-field part stays near-linear throughout. Not yet separated: the
density reaching the hotspot through χ ∝ xPhys^e (e ≈ 4.6 here), or acting on it directly
through the density gate and shielding.

## Batch 28: density move 0.02 over iterations 100–550 (2026-10-20)

This is ramp A with the density move limit raised from 0.01 to 0.02 over iterations
100–550, which doubles MMA's asymptote ceiling. Run twice, the replicate with the
roughness weight × (1 + 1e-4). Binarized row with `t` kept / C ÷ plain TO / grey:

| run | result | as optimized |
|---|---|---|
| ramp A / replicate (batches 24, 27) | −0.017 ✓ / +0.052 ✗, both 1.062 | +0.009 / +0.007 |
| move 0.02 | −0.027 / 1.055 / 2.1% ✓ | +0.003 |
| move 0.02, replicate | −0.024 / 1.055 / 2.2% ✓ | +0.006 |

- **It doesn't speed up phase 2.** Grey at 350 is 0.52 against 0.53, and at 550 it is
  0.33 against 0.34. This fits the split above: the step is limited by reversals and
  imbalance, not by the ceiling.
- **Both runs pass, with slightly better compliance, and they agree with each other.** They
  are less grey by 650 (0.058 against 0.077–0.097). Two runs, so this may be luck.

### What reverses the grey (`flip_terms.tmp.py`, `hot_jump.tmp.py`)

MMA moves each variable against its multiplier-weighted net gradient (checked: this
predicts the direction of 100% of moving grey elements). So a reversal is a sign flip of
the net push. The reversing elements sit at a near-exact balance: their median net push is
±0.6% of the sum of the terms. In the burst (replay steps 205–208):

- The hotspot maximum alternates every iteration between two distant spots, (32, 32) and
  (33, 167), with gaps of 0.004–0.007. The softmax weight near each spot stays at 12–14%,
  and the raw hotspot-row gradient on the reversing elements keeps its sign on 92% of them.
- The multipliers zigzag in step: λ_vol 1.482/1.465/1.496/1.475, λ_hot
  0.244/0.206/0.259/0.221. Per reversing element, the term that did most to turn it
  around is volume (λ_vol's swing) for about half and the hotspot for the other half.
  After the burst (209–212) the maximum stays on the right, and 6–18% reverse.
- `calibration_rate` (0.1) averages only the calibration offset, so it steadies the row's
  value, not its gradient or the multipliers.

## Batch 29: penal schedules (2026-10-20)

Ramp A with penal only changed (q unchanged). A: 2 → 3 over 100–200 instead of 150–350.
B: 3 → 4 over the 350–550 hold instead of 3, which makes penal a single straight ramp 2 → 5 over 150–750 (penal no longer held). Each run twice, the replicate with the
roughness weight × (1 + 1e-4). Binarized row with `t` kept / C ÷ plain TO / grey:

| run | result | grey at 350 | bulk grey at 350 | grey at 650 |
|---|---|---|---|---|
| ramp A / replicate | −0.017 ✓ / +0.052 ✗, 1.062, 2.3–2.6% | 0.53 | 0.16 | 0.08–0.10 |
| A, penal early | +0.001 / 1.053 / 7.3% ✓ | 0.39 | 0.10 | 0.085 |
| A, replicate | +0.011 / 1.062 / 4.6% ✗ | 0.38 | 0.10 | 0.072 |
| B, penal 4 in hold | −0.024 / 1.059 / 2.5% ✓ | 0.53 | 0.16 | 0.054 |
| B, replicate | −0.016 / 1.059 / 2.4% ✓ | 0.53 | 0.16 | 0.052 |

Bulk grey means grey more than 4 elements (the filter radius) from the xPhys = 0.5
contour, as a share of the domain.

- **An earlier penal ramp does speed phase 2:** grey at 350 is 0.39 against 0.53, and bulk
  grey 0.10 against 0.16. But it ends greyer (4.6–7.3%: two thin diagonal members stay
  grey) and is marginal at the gate. Compliance is higher early (239 against 213 at 185)
  and about equal at the end.
- **Penal 4 in the hold** changes little before 550 but binarizes faster in 550–650. Both
  runs pass with margin, close to each other.

## Batch 30: penal not held + density move 0.02 (2026-10-20)

Batch 29's B (penal ramps straight 2 → 5 over 150–750) with batch 28's density move
(0.02 over 100–550), run twice. Binarized row with `t` kept / C ÷ plain TO / grey:

| run | result | grey at 550 / 650 | bulk grey at 550 |
|---|---|---|---|
| ramp A | −0.017 / 1.062 / 2.6% ✓ | 0.34 / 0.10 | 0.067 |
| move 0.02 (two runs) | −0.027, −0.024 / 1.055 ✓✓ | 0.33 / 0.06 | 0.07 |
| penal not held (two runs) | −0.024, −0.016 / 1.059 ✓✓ | 0.31 / 0.05 | 0.055 |
| both | −0.030 / 1.052 / 2.1% ✓ | 0.29 / 0.03 | 0.046 |
| both, replicate | −0.046 / 1.054 / 2.3% ✓ | 0.30 / 0.04 | 0.054 |

The combination has the best margins and compliance so far, and binarizes fastest in
550–650. Phase 2 is still about as slow as before (grey at 350 is 0.51–0.52 against 0.53).
Two runs.

The user's call (2026-10-21): penal ramped straight 2 → 5 and the density move of 0.02
over 100–550 are part of the default from batch 31 on.

## Batch 31: the L cells on the current recipe (2026-10-21)

L middle and L corner (Tcr 0.6) on the full recipe: Tcr ramp 1 → 0.6 over 0–100, phase-2
hold, time-field move 0.002 throughout, and the two new defaults. Each run twice
(`lap_batch31_32.tmp.py`). Binarized row with `t` kept / C ÷ plain TO / grey:

| cell | batch 25 (step at 100, no hold), run / repeat | current recipe, run / duplicate |
|---|---|---|
| L middle | −0.052 / 1.042, −0.020 / 1.033 | −0.083 / 1.012 / 1.6%, −0.028 / 1.012 / 1.6% |
| L corner | −0.072 / 1.015, −0.068 / 1.017 | −0.037 / 1.016 / 1.5%, −0.030 / 1.017 / 1.5% |

All four pass. L middle's compliance improves markedly (1.012 against 1.033–1.042). L
corner's is unchanged, with a smaller (still clear) margin.

## Batch 32: C1 with the first 100 iterations removed (2026-10-21)

The batch-30 recipe with every schedule shifted 100 earlier (900 iterations). Tcr still
ramps over 0–100, and the density move is 0.02 from 0 to 450. Penal: 'from 0' ramps 2 → 5
over 0–600; 'held' stays at 2 for 150, then ramps 2 → 5 over 150–750; 'from 1' ramps
1 → 5 over 0–600. β_d: the full 1 → 4 ramp over 0–120, or held at 2 or at 4, each followed
by the final ramp to 32 over 450–550. Binarized row with `t` kept / C ÷ plain TO, run and
duplicate. Grey is compared at the start of the β_d ramp to 32 (450 here, 550 in batch 30):

| penal | β_d | result, run / duplicate | grey when β_d starts rising |
|---|---|---|---|
| batch 30 (not cut) | ramp | −0.030 / 1.052, −0.046 / 1.054 | 0.29, 0.30 |
| from 0 | ramp | **−0.035 / 1.043, −0.015 / 1.047** | **0.20, 0.23** |
| from 0 | hold 2 | −0.022 / 1.061, −0.001 / 1.042 | 0.31, 0.31 |
| from 0 | hold 4 | −0.031 / 1.076, −0.016 / 1.066 | 0.32, 0.33 |
| held | ramp | −0.000 / 1.052, −0.034 / 1.066 | 0.43, 0.43 |
| held | hold 2 | −0.016 / 1.069, −0.030 / 1.072 (5.2% grey) | 0.51, 0.52 |
| held | hold 4 | −0.008 / 1.070, −0.008 / 1.066 | 0.44, 0.43 |
| from 1 (one run each) | ramp / hold 2 / hold 4 | −0.020 / 1.068; **+0.035 / 1.077 ✗**; −0.000 / 1.066 | 0.40 / 0.46 / 0.43 |

- **Removing the first 100 iterations costs nothing with penal from 0 and the β_d ramp.**
  It has the best compliance so far (1.043, 1.047), both runs pass, and it is less grey at
  the matching point (0.20–0.23 against 0.29–0.30), in 900 iterations instead of 1000.
- **Holding penal at 2 for 150 iterations slows binarization** (0.43–0.52 grey at that
  point) and costs compliance. Starting penal at 1 is worse again, and is the only failure
  in the batch.
- **Holding β_d at 2 or 4 from the start** is no better than the ramp with penal from 0
  (one hold-2 duplicate reaches 1.042; the others reach 1.061–1.076).

## Batch 33: the one-phase schedule (2026-10-21)

Under capped-ratio shielding `r` no longer enters the severity (the gate is
`x / (x + eps)`), and with `tool_radius_m = 0` nothing else reads it, so the `r` and
`curvature_beta` schedules are inert and are held constant from here on.

One ramp phase, then cleanup (`lap_batch33.tmp.py`, 900 iterations). Over 0–450 every
continuation parameter ramps linearly: κ 0 → 2.37, penal and q 2 → 5, β_d 1 → 4, β_t
10 → 50; Tcr keeps its 1 → Tcr ramp over 0–100, and the χ exponent e is held at 4. From
450: β_d 4 → 32 and cooling sharpness 1 → 8 over 450–550, density move 0.02 → 0.01.
Binarized row with `t` kept / C ÷ plain TO / grey:

| run | result | grey at 200 / 450 |
|---|---|---|
| C1, batch 32 best (cut100, β_d 1 → 4 over 0–120) | −0.035 / 1.043, −0.015 / 1.047 | 0.57–0.59 / 0.20–0.23 |
| C1, one-phase / duplicate | −0.023 / 1.057, +0.009 / 1.067 | 0.64 / 0.23–0.26 |
| C1, one-phase, β_d held at 2 | −0.004 / 1.062 / 2.2% | 0.62 / 0.31 |
| C1, one-phase, q = 1 throughout | −0.006 / **1.039** / 2.9% | **0.53** / 0.23 |
| L middle, batch 31 / one-phase | −0.083 / 1.012 → **−0.238 / 1.008** / 1.5% | |
| L corner, batch 31 / one-phase | −0.037 / 1.016 → **−0.085 / 1.010** / 1.2% | |

- **The L cells improve on both margin and compliance** in 100 fewer iterations.
- **C1 loses about 0.015 in compliance** and its gate margin shrinks (the duplicate passes
  at +0.009, inside the 0.01 tolerance). The extra grey is early: at 200 the one-phase run
  is 0.64 grey against 0.57–0.59, when its β_d is 2.3 rather than 4. That matches batch
  32's β_d hold-2 runs.
- **q = 1 gives the best C1 compliance yet** and the fastest early binarization. On a
  binary design the capped ratio is 0 or 1, so q only shapes the grey; this is the first
  sign that it need not be continued. One run.

### When the result settles (batch 33 snapshots, `bin_keep_t22_at.tmp.py`)

The gate row on the binarized snapshot keeps moving after compliance and grey have
settled (C1 compliance is within 0.3% of final by 650–700). On batch 33's q = 1 run it is
+0.088 at 700 and −0.006 at 899. Swapping fields between the 700 and 899 snapshots, the
density recovers 0.049 of that and the time field 0.030. The density changes are a
handful of elements near the hotspot (mean step ~1.5e-4), pinned at the 0.01 density move
limit; the time field's steps stay below its own limit. Evaluate snapshots with a fresh
problem each: the hotspot calibration is an EMA across calls.

## Batch 34: one-phase with β_d held at 2 and q = 1 (2026-10-21)

Batch 33's one-phase schedule with β_d held at 2 until 450 and q held at 1 throughout
(`lap_batch34.tmp.py`), plus a C1 run with the density move kept at 0.02 to the end.
Binarized row with `t` kept / C ÷ plain TO / grey, and the gate on snapshots:

| run | result | 600 | 650 | 700 | 750 | 800 | 850 |
|---|---|---|---|---|---|---|---|
| C1 | −0.007 / 1.042 / 3.1% | +0.053 | +0.035 | −0.004 | −0.004 | −0.001 | −0.005 |
| C1, duplicate | −0.001 / 1.042 / 3.2% | +0.064 | +0.090 | +0.048 | +0.031 | +0.002 | −0.000 |
| C1, move 0.02 tail | +0.006 / 1.047 / 2.8% | +0.051 | +0.088 | +0.042 | +0.000 | +0.002 | +0.007 |
| L middle | −0.198 / 1.009 / 1.4% | −0.095 | −0.170 | −0.194 | −0.196 | −0.197 | −0.197 |
| L corner | −0.044 / 1.011 / 1.2% | −0.031 | −0.039 | −0.043 | −0.043 | −0.049 | −0.047 |

- **C1 compliance 1.042 on both runs**, the best and most repeatable so far; the L cells
  are unchanged (1.009, 1.011). The cost is margin: C1 passes by 0.001–0.007 binarized
  against −0.033 as optimized. q = 1 reads grey support leniently, so the continuous
  design under-protects and the binarized row lands near the limit.
- **Cutoff: 800.** Every run stays within the 0.01 tolerance from 800; at 700 the C1
  duplicate is +0.048. The L cells pass from 550.
- **The 0.02 move tail does not converge sooner** (passes from 750 against 700 and 800
  for the two 0.01 runs) and ends slightly worse.
- The host was suspended 13:44–14:04 during this batch (monotonic timers 20 min behind
  wall time); the runs resumed normally.

The user's call (2026-10-21): β_d held at 2 and q = 1 are locked in, and runs end at 800
iterations (late schedules re-anchored: time-field move 0.002 → 0.001 over 700–750,
roughness decay over 750).

## Batch 35: validation of the locked schedule (2026-10-21)

One run per cell (`lap_batch35_36.tmp.py`; cell settings from earlier configs, weights
fixed). Binarized row with `t` kept / C ÷ the cell's plain TO (`stto`'s) / grey:

| cell | result |
|---|---|
| C1 0.5 Tcr 0.6, left | −0.067 / 1.014 / 2.0% ✓ |
| C1 0.3 Tcr 0.6, left | **+0.041** / 1.115 / 3.5% ✗ (as optimized −0.007) |
| C2 0.5 Tcr 0.4, left | −0.145 / 1.008 / 1.1% ✓ |
| S 0.3 Tcr 0.6, left | −0.222 / 1.010 / 1.2% ✓ |
| S 0.3 Tcr 0.6, bottom | −0.198 / 1.012 / 1.3% ✓ |

- **C1 0.3 fails** by 0.041 once binarized; it is still improving at the end (+0.077 at
  650–700, +0.041 at 800), and its compliance is the worst of the cells. It was the hard
  cell before too (batches 12–13).
- No small enclosed holes (< 40 elements) in any final design.

**Sequential reference on C1 0.5 Tcr 0.6** (`lap_seq_ref.tmp.py`): plain TO on the same
schedule with Tcr held at 2, then the density frozen and the time field alone optimized
for 300 iterations at Tcr 0.6.

| | C | gate |
|---|---|---|
| sequential: plain TO, then time only | 176.7 | **+0.014** ✗ (as optimized +0.008) |
| simultaneous (the locked schedule) | 177.1 (+0.2%) | −0.067 ✓ |
| `stto` plain TO / `stto` at Tcr 0.6 | 174.7 / 175.4 | |

- The simultaneous and sequential designs differ on 0.8% of the domain: the same
  topology. Moving that 0.8% (and optimizing `t` with it) buys a 0.08 margin for 0.2% in
  compliance; the time field alone cannot quite meet Tcr 0.6 on the plain-TO design.
- Our plain TO is a different topology from `stto`'s (22% of the domain differs) and
  1.2% worse in compliance, so part of every C ÷ plain TO figure here is the schedule's
  own plain-TO optimum, not the hotspot.

## Batch 36: a shorter phase 1 (A) and a slower, higher β_d ramp (B) (2026-10-21)

On the locked schedule, 800 iterations each (`lap_batch35_36.tmp.py`). A: phase 1 over
0–300 (κ, penal, β_t compressed; Tcr still over 0–100), cleanup at 300–400. B: phase 1
as locked, β_d 2 → 128 over 450–750 (geometric, half batch 34's rate; cooling sharpness
still 450–550). Binarized row with `t` kept / C ÷ plain TO / grey, and the first
snapshot from which the gate stays within 0.01:

| run | C1 0.5 Tcr 0.4 | L middle | L corner |
|---|---|---|---|
| batch 34 | −0.007 / 1.042 / 3.1%, from 700 (duplicate 800) | −0.198 / 1.009 / 1.4% | −0.044 / 1.011 / 1.2% |
| A: phase 1 300 | −0.029 / 1.046 / 3.0%, from 650 | −0.144 / 1.013 / 1.3% | −0.055 / 1.013 / 1.3% |
| B: β_d → 128 over 300 | −0.038 / **1.034 / 0.6%**, from 750 | −0.177 / **1.006 / 0.1%** | −0.032 / 1.010 / 0.2% |

- **B removes the compliance blow-up**: C1's compliance peaks at 211 at the start of the
  ramp, against 386 (batch 34) and 411 (A) when β_d is 25–27. It ends nearly binary
  (0.1–0.6% grey), with the best C1 compliance so far, and the binarized row is now at or
  below the as-optimized one (−0.038 against −0.024): the continuous/binary gap is gone.
- **A works as well as batch 34** in 300 fewer iterations of phase 1 and passes from 650
  on C1. Its β_d ramp tears the design like batch 34's.
- Holes: C1 Tcr 0.4 again grows one at about (54, 169) in both A and B (and one at
  (51, 108) in B); filling any of them leaves the gate unchanged. It recurs in 6 C1 Tcr 0.4
  runs, and not in plain TO or at Tcr 0.6.

**A+B** (phase 1 over 0–300, then β_d 2 → 128 over 300–600; cooling sharpness 300–400;
800 iterations): C1 −0.036 / 1.038 / 0.7%, L middle −0.147 / 1.009 / 0.1%, L corner
−0.040 / 1.010 / 0.1%, all ✓. The gate stays within 0.01 from 600 on C1 (+0.007 at 600,
−0.031 at 625) and from 550 on both L cells, and C1's compliance is settled by 550. So
A+B could end at about 650 (a 50-iteration margin after the ramp), against 800; B alone
could end at 750. One run per cell. C1 again grows the hole at about (54, 169) (and one
at (51, 104)); filling either leaves the gate unchanged.

**Why C1 0.3 fails (batch 35), `q_gap.tmp.py`.** On grey, the capped ratio credits a
neighbour `(x_b / x_a)^q`, so q = 1 reads grey support generously, and binarizing then
drops the neighbours below 0.5. Read on the continuous final design with q = 1 / 3 / 5:
C1 0.3 −0.013 / +0.163 / +0.247 (binarized +0.041); C1 Tcr 0.4 batch 34 −0.039 / +0.033 /
+0.043 (−0.007); C1 Tcr 0.6 −0.071 / −0.060 / −0.050 (−0.067); batch 36 B (0.6% grey)
−0.034 / −0.016 / −0.004 (−0.038). Thin members put more of a hot element's stencil in
the grey edge band; ending near-binary (B's ramp) removes the dependence rather than a
different q.

## Gate tolerance in severity units (user, 2026-10-21)

From batch 37 on, a binarized design passes the hotspot gate if its worst severity is at
most Tcr + 0.025 (was: severity / Tcr − 1 ≤ 0.01). The allowance stands in for the
precision of the heuristic, in the units of the quantity that is compared.

## Batch 38: an 8 mm hotspot stencil (2026-10-21)

C1 0.5 Tcr 0.4 on the locked schedule with `rmin_cond_m` 0.008 (`lap_batch38.tmp.py`).
At 8 mm the worst severity of a 12 mm design sits in a band one stencil radius from the
plate and at inside corners, while p90–p95 barely move (`stencil_maps.tmp.py`); so Tcr
was rescaled by the p99 ratio away from the plate (1.14 → 0.46), and also run at 0.40.

| run | own gate (excess over Tcr) | C ÷ plain TO | grey | read at 12 mm, Tcr 0.4: worst |
|---|---|---|---|---|
| 12 mm, batch 37 locked | +0.017 ✓ | 1.042 | 1.1% | 0.417 |
| 8 mm, Tcr 0.46 / duplicate | −0.015 ✓ / +0.025 ✗ (by 0.0001) | 1.029 / 1.028 | 1.0% / 0.9% | 0.540 / 0.542 |
| 8 mm, Tcr 0.40 / duplicate | +0.016 ✓ / +0.034 ✗ | 1.055 / 1.058 | 2.3% / 2.0% | 0.507 / 0.491 |

- **No small holes in any of the four 8 mm runs.** The hole at about (54–55, 169–173)
  is in 10 of the 11 12 mm C1 Tcr 0.4 runs since batch 34.
- **The 8 mm designs are a different truss**: more, thinner diagonals. Read with the
  12 mm stencil, their member interiors are above Tcr (worst 0.49–0.54), because a 12 mm
  stencil around a thin member is mostly void. So the stencil radius sets a minimum
  member thickness as well as the hole behaviour; the two are confounded here.
- The plate band that dominates a 12 mm design read at 8 mm is not a problem for the
  8 mm runs themselves: their own worst elements are mid-span.

## Batch 37: k ramp, μ filter and q ramp on the locked schedule (2026-10-21)

The schedule locked after batch 36 (A+B ending at 650; `lap_batch37.tmp.py`; also the new
`configs/laplace_default.json`). Worst binarized severity − Tcr (pass ≤ +0.025) / C ÷
plain TO / grey; one run each:

| variant | C1 0.5 Tcr 0.4 | L middle | L corner | C1 0.3 Tcr 0.6 |
|---|---|---|---|---|
| locked | +0.017 / 1.042 / 1.1% | −0.032 / 1.009 / 0.1% | −0.026 / 1.010 / 0.1% | **+0.028** ✗ / 1.070 / 0.9% |
| k ramp (1 → 8 linear over phase 1) | −0.005 / 1.052 / 0.8% | −0.098 / 1.007 / 0.2% | −0.032 / 1.010 / 0.2% | |
| q ramp (1 → 3 over 300–600) | −0.016 / 1.051 / 1.0% | −0.017 / 1.009 / 0.2% | −0.028 / 1.010 / 0.1% | −0.023 / 1.084 / 0.3% |
| μ filter 1.5 | **+0.126** ✗ / 1.084 / 7.3% | −0.060 / 1.013 / 0.2% | −0.040 / 1.011 / 0.3% | |
| μ filter 2 | **+0.153** ✗ / 1.071 / 9.4% | −0.051 / 1.007 / 0.2% | −0.031 / 1.011 / 0.3% | |

- **The q ramp** brings both C1 cells inside Tcr after binarizing (C1 0.3 from a fail to
  −0.023) for about 1% (C1 0.5) to 1.4% (C1 0.3) in compliance; on the L cells it is
  neutral.
- **The k ramp** is about neutral (C1 slightly better margin, slightly worse compliance).
- **The μ filter** halves χ<sub>μ</sub> roughness on every cell (L: 0.35 → 0.16–0.21; C1:
  0.77 → 0.39–0.50) and is harmless on the L cells, but on C1 it leaves 7–9% grey and
  fails badly once binarized, as in batch 25.
- The C1 Tcr 0.4 tip hole is in every C1 run; no L-cell run has a small hole.

Plain TO on the locked schedule (Tcr held at 2, `lap38_C105_plainTO`): C 175.9 (1.007 ×
`stto`'s plain TO), no small holes; read at Tcr 0.4 its worst severity is 0.596 (12 mm)
and 0.633 (8 mm). So the hotspot costs C1 Tcr 0.4 about 3.4% in compliance at 12 mm
(181.9) and 2.2% at 8 mm with Tcr 0.46 (179.7), and the tip hole appears only with the
12 mm constraint active.
