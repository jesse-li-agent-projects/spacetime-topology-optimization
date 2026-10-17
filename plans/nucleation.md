# Nucleation: material grown as "artificial support" for a hot spot

Status and handoff as of 2026-10-18. No code change is agreed yet. Ask the user before
any design decision this plan does not make. The user is open to radical changes, e.g.
a per-element objective instead of a smooth max. Reason from the least-assumptions
statement in "Requirements", not from the current form.

## Status

- **Found** 2026-10-09 (session 968d9f4e), on S 0.3 with the cooling sharpness `k` held
  at 2 or 4 (`fig4_nucleation_S03.gif`). Grey blobs grow under the flat-printed top
  bar from about 150 to 400. Each one digs its own minimum in `t`, so the blob counts as
  earlier material under the bar. At binarization the blob becomes a spur with nothing
  behind it, its severity goes to about 1, and it is never removed.
- **Masked, not solved, by the settled schedule** (`configs/default.json`). A soft `k`
  of 1 until 350 gives ties about half credit, so the bar's own elements credit each
  other and need no seed. The same family of failure shows up whenever the credit
  sharpens early, or at binarization: S vf 0.5 at Tcr 0.8 and L middle at Tcr 0.6 in the
  validation matrix (multiplier at `mma_c`, a hot spot on a region printed all at once).
  The geodesic seed (#210) fixed L middle.
- **Analysis so far.** A linearized stability condition, a corridor argument showing that
  no monotone f and g satisfy both requirements, and toys. They are summarized below.
- **Unanswered user points.** The 2026-10-09 proposals were a negative-dip g and toy v3.
  Q1–Q3 under "Open questions" are new.

## Mechanism

Severity is `s_e = f(x_e) (1 - K_e)` with `K_e = Σ_n w_en c(t_e - t_n) g(x_n) / D_e`.
Today `f = x^r` (`r` 0.5 in the grey hold, 0.05 at the end), `g = x^q` (`q` follows
`penal`), `c` is the print-order credit, and `w` is the stencil weight with its angular
lobe. The hotspot is a LogSumExp smooth max `H` with β = 100 and weights `π_e`:

```
dH/dx_n =  π_n f'(x_n)(1 - K_n)                  # own severity: pushes n out
         - Σ_e π_e f(x_e) w_en c_en g'(x_n)/D_e  # credit n gives hot e: pulls n in
```

1. **Only the max feels its own severity.** `π_n ~ exp(-β (S_max - s_n))`, which is
   about 1e-20 for an element 0.5 below the max. Material whose own severity is below the
   active level is free in the hotspot, and it grows until that severity reaches the max.
   With `f = √x` at Tcr 0.6, anything under density ~0.36 is free.
2. **Credit never pushes material out.** A neighbour of a hot element is only pulled in.
3. **Credit is one hop.** It checks that n has density and is earlier. It never checks
   that n is itself cooled, i.e. grounded through earlier material. A void element's `t`
   is a free design variable, so void can be made "earlier" anywhere.

## Requirements: the pressures we want

Hand-built scenes: `plot/nucleation_scenarios/scenarios.png` (script
`scenarios.tmp.py` beside it). Settings: f = √x, g = x³, k = 4, β = 100, base at the
bottom, void 0.01, no filter.

| # | Situation | Wanted | Today, at the cell |
|---|---|---|---|
| L1 | Thin layered member overheats | Pull material in at its edge (thicken) | Pulls in, weakly: +6.7e-7 |
| L2 | Region printed all at once (flat `t`) | Reorder `t`; no new material | Pushes its own material out (−1.5e-2); light pull on the void below |
| L3 | Member that cannot be saved (thin cantilever tip) | Remove it, or better, never form it (necking) | Pushes out (−4.5e-2) |
| P1 | Near-void under a temporal local minimum in solid | Heal `t`; keep the void empty | Pulls in: +4.7e-6, 7× L1 |
| P2 | Early grey material under a hot region | Push the grey to 0 | Pulls in: +8.4e-4, own π 1e-21 |
| ? | Solid strut from the base up to a flat bar | Real support; compliance decides (Q1) | Pulls in: +2.0e-4, the same signal as P2 |

Two further requirements come from earlier sessions:
- **R-restore (user, 2026-10-09):** "Ideally, after a perturbation (leading to slight
  non-void density), there should be gradient pressure to remove it." Neutral stability,
  i.e. a credit that is exactly zero on a band, is not enough.
- **R-grey (user, 2026-10-09):** grey regions should not dominate the hotspot early.
  Keep f low at small x, like `x^0.5`. A uniform layered grey region has severity
  about `f(ρ)(1 - g(ρ))`.

### Are they in conflict?

- **Inside the current form, yes.** Nucleation stability needs `g ≈ 0` wherever
  `f(x) < S*` (S* is the active level). "An element must count as fully hot before it
  counts as conductive." For power laws that leaks `Tcr^(q/r)` of credit. R-grey needs g
  large enough at mid density that `f(1 - g)` stays under S* minus a margin. Where `g ≈ 0`,
  uniform grey severity equals the lone-nucleus severity, so the two meet in a band
  (`plot/pres_tuning/margin_band.png`). The settled `x^2.9` already misses the 0.1
  margin: uniform grey peaks at 0.61 near ρ 0.5.
- **The bad and the good cases are not separated by density, time order or magnitude.**
  P2 and the strut, or P1 and L1, are each "earlier, light material next to a hot
  element". What differs is whether that material is grounded: would it be cool if it
  were solid? No choice of f, g, `q/r`, a lag or the smooth max sees that. The toys agree:
  with the bar held fixed, every g family grew material under it.
- **Two tensions that look real in any form:**
  - Growth from empty void needs `g'` > 0 near 0, and that conflicts with R-restore.
    The filter's grey edges let members grow from their edges even with zero
    low-density credit. The price is that the hotspot can then never start a new
    member, only thicken, reorder or remove (Q1).
  - The hotspot must act early (user, 2026-10-07: members coalesce by ~80), but whether
    grey material is grounded is not defined until the design binarizes.

## What has been tried

- **r and q schedules** (#197, #201, #204, #205). `r` 0.5 → 0.05 over 350–450 and `q`
  following `penal`. This is the settled schedule. Among the hottest 0.5% of elements
  after binarizing, the share that was grey is 92% at r = 1 and 4% at r = 0.05
  (`plans/print_order_credit.md`).
- **Grey bars, C1 0.5** (2026-10-08). Bars held near 0.35–0.42 between compliance and
  the hotspot, with MMA asymptotes collapsed. Deleting a bar made the hotspot *worse*,
  so grey sat at a local optimum of the hotspot. Damped calibration (#202) helped.
- **`k` / `kappa` continuation** (`output/kcont/eval.json`, `plot/kcont/dashboard.html`).
  Sharpening `k` early makes S 0.3 fail every time, with a blob under the top bar.
- **Toys** (scripts in `.claude/worktrees/schedule-simplify/toys.tmp/`, untracked;
  outputs `output/nucleation/`, regenerated 2026-10-18; plots `plot/nucleation/`):
  - Static (`static.py`). A half-plane of solid printed along x from the left, with a
    Gaussian `t` dip on its lower edge. Under power-law g the nucleation drive is
    positive at every void density. A lagged g (`smoothstep((x-a)/(1-a))`) gives zero
    drive below `a`: neutral, not restoring. The own-severity term never helps at a
    strict minimum.
  - Dynamic (`dynamic.py`, MMA, volume price 1). With the bar's `t` free, no g nucleates,
    because healing `t` is cheaper. With the bar's `t` fixed, every g grows material.
    The user doubted the toy is representative. In S 0.3 the bar had a flat `t` with
    nothing earlier nearby, and the minimum was the *result* of nucleation, not its
    precondition.
- **Never run:** `q` held at 6 through the grey hold with `k` 4; toy v3 (below); a
  negative-dip g.

## Candidate directions

None is agreed. On 2026-10-09 the user called the first round (close the free-grey
threshold, support-aware/recursive credit, the gradient floor, raise `q` early) "not
promising". Ask before reviving any of them, and say why the evidence above changes the
case.

- **Shape f and g** (the user's 2026-10-09 framing). Includes a g that dips below 0 at
  small x: light, early material slightly *lowers* its neighbours' credit, a "porous
  insulator", which satisfies R-restore at a small cost to R-grey. It cannot separate P2
  from the strut.
- **Grounded credit.** Weight n's credit by how cool n would be if it were solid, e.g.
  `g(x_n) γ(K_n)`, a two-hop credit. Edge growth keeps its credit and floating nuclei
  lose theirs. Cost: a second pass, and grey-phase readings of `K_n` are unreliable.
- **Change the aggregate.** Per-element constraints or multipliers, or a per-element
  objective. On its own this does not remove property 2: the pull on n is still a sum
  over hot neighbours' multipliers. A purely selfish per-element objective would never
  thicken anything (L1). It only pays off combined with a change to what credit means.
- **A real heat path.** The virtual-heat variants: `stto_heat` (set aside, see
  `plans/virtual_heat_tuning.md`) and `stto_laplace` (PR #207, open). Groundedness comes
  for free. The user's stated purpose for Variant 2 is to rule out a class of local
  minima and maxima.

## Open questions for the user

1. Should the hotspot ever justify new material that is not edge growth, such as the
   strut? Or only thickening, reordering and removal?
2. When reordering `t` and adding material both lower a severity, should `t` always win
   (it costs no compliance)?
3. Is P1 (near-void under a minimum in solid) a separate case from P2, or just P2's
   seed? With g = x³, empty void barely feels the pull. Nucleation needs grey to start.
4. Still open from 2026-10-09: build toy v3, with the bar attached to the base, a normal
   front, then a flat plateau of `t`, and test the negative-dip g on it?

## Next steps (once the user answers)

1. Turn the answers into a sign table: for each scene, the wanted sign of `dH/dx` and
   `dH/dt` at the cell. Make `scenarios.tmp.py` assert it for any candidate f, g,
   credit or aggregate. This is the cheapest test of a formulation.
2. Toy v3 for the candidates that pass, then S 0.3 with `k` held at 2. That cell
   reproduces the failure. Use matched controls on S 0.3, C1 0.3 and C1 0.5, as in
   `print_order_credit.md`.

## References

- Plans: `print_order_credit.md`, `s03_material_behind_front.md`, `length_scale_options.md`
  (necking), `virtual_heat_tuning.md`, `configs/continuation.md`. The earlier handoff
  section is in `plans/stto_tuning_handoff.md`, on the unmerged branch
  `worktree-tuning-handoff` (PR #209, closed).
- Code: `sttopt/conductivity.py` (`_conductivity_core`: g is `x[b] ** q`; credits;
  `severity`; `LogSumExp`), `sttopt/smooth_max.py` (f via `density_power`),
  `sttopt/field_terms.py`.
- Figures: `plot/pres_weekly/fig4_nucleation_S03.gif` and `fig4_candidates.png`,
  `plot/nucleation/*.png`, `plot/pres_tuning/margin_band.png`,
  `plot/nucleation_scenarios/scenarios.png`.
- Sessions: 968d9f4e (2026-10-09: discovery, derivation, toys), 827bdf37 (2026-10-08:
  grey bars, r sweep), ee3845cf (2026-10-14: presentation figures), 68373700 (this
  plan).
