# Sequential against simultaneous optimization

The default cantilever (`laplace_default.json`: 180 x 60, volfrac 0.5, `Tcr` 1 -> 0.4).

- **Simultaneous:** `stto_laplace` as it is.
- **Sequential:** a compliance-only plain TO (`plain_to.json`: the default with the time-field
  terms off), then only the time field optimized on the plain TO's grayscale density, with the
  default's settings, shaping term included (`--fixed-density`).

From the repo root:

```
python -m sttopt.stto_laplace_cli --config configs/seq_vs_sim/plain_to.json --tag tex_seq_vs_sim_plain_to
python -m sttopt.stto_laplace_cli --config configs/laplace_default.json --tag tex_seq_vs_sim_sequential --fixed-density output/tex_seq_vs_sim_plain_to
python -m sttopt.stto_laplace_cli --config configs/laplace_default.json --tag tex_seq_vs_sim_simultaneous
```
