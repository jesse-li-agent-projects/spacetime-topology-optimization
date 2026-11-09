# Time-field formulations on S03bT06

The three space-time problems on the same cell (`corner_loads`, 120 x 120 elements, volfrac 0.3,
print base `bottom_edge`, `Tcr` 1 -> 0.6 over 0-150), on `laplace_default.json`'s schedule. Everything
shared (`SpaceTimeConfig`) is taken from `laplace.json`; `heat.json` and `direct.json` differ from it
only where the time field forces it:

- **Both:** no hotspot row in the constraint stack (`hotspot_constraint`), as `stto_laplace` has none.
- **`heat.json`** (`stto_heat`): the drained heat equation (`drain_beta`, `poisson` time map), the
  diffusivity's own move limit (`field_move` 0.002), no wall data and no density factor in the
  diffusivity.
- **`direct.json`** (`stto`): `t` optimized directly, with log compliance (`log_compliance`) like the other two,
  the start-point constraint, the time filter, `tmove` and a roughness term. Layer uniformity alone
  rewards a sawtooth in `t`, which the harmonic parameterizations cannot form, so this is the
  `heat` default's roughness weight (the same log-compliance scale), not Laplace's 0.

Simultaneous runs, then the same three on the Laplace run's grayscale density (`--fixed-density`),
from the repo root:

```
python -m sttopt.stto_laplace_cli --config configs/time_formulations/laplace.json --tag tf_S03bT06_laplace
python -m sttopt.stto_heat_cli    --config configs/time_formulations/heat.json    --tag tf_S03bT06_heat
python -m sttopt.stto_cli         --config configs/time_formulations/direct.json  --tag tf_S03bT06_direct

python -m sttopt.stto_laplace_cli --config configs/time_formulations/laplace.json --tag tf_S03bT06_laplace_fixed --fixed-density output/tf_S03bT06_laplace
python -m sttopt.stto_heat_cli    --config configs/time_formulations/heat.json    --tag tf_S03bT06_heat_fixed    --fixed-density output/tf_S03bT06_laplace
python -m sttopt.stto_cli         --config configs/time_formulations/direct.json  --tag tf_S03bT06_direct_fixed  --fixed-density output/tf_S03bT06_laplace
```
