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
  still escalates (MMA subsolver hitting its iteration cap from ~iteration 300, void `t`
  up to 152), so step 120 still looks fragile at Tcr 0.4.
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
the sleep out (the driver's wall-clock minutes include it). GPU runs that hit the
subsolver cap ran at ~2 s/iteration against ~0.5 for clean ones.

Dashboard: `plot/laplace_s03/dashboard_b19.html` (spec `lap_spec19.tmp.json`).
