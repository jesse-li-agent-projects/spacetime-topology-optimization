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
- **A at f > 0** wrecks compliance (1.8–5.6 × plain TO). The hotspot sits on near-void
  wisps: on C1 0.3 at f = 1, iterations 100–300, the top-50 severities have median
  density 0.07–0.14 and 500–900 elements exceed Tcr, most with `x < 0.1`
  (`sev_density.tmp.py`); at f = 0 no element exceeds Tcr there. The `x / (x + ε)` gate
  counts anything above ~10ε as material, and scaling a wisp's neighborhood down is the
  very direction the invariance makes force-free, so only the gate could delete it, and
  it is flat there. The gradient that remains mostly raises the wisp's earlier
  neighbors, which grows more light material.
- Next: a gate that is a step at a meaningful density, e.g. `x^k / (x^k + η_g^k)` with
  `η_g ≈ 0.2–0.3`, so a grey bar sits on its flat part and a wisp on its slope.

Dashboard: `plot/laplace_s03/dashboard_shielding.html` (spec `lap_spec14.tmp.json`);
table `output/lap/eval14.json` (`lap_eval14.tmp.py`).
