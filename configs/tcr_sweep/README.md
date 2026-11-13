# Sweep of the maximum acceptable severity on the cantilever

The default cantilever (`laplace_default.json`: 180 x 60, volfrac 0.5), with `Tcr` ramped from 1
down to each swept value `T` over iterations 0-150, and nothing else changed. The reference is
sequential optimization, as in `configs/seq_vs_sim/README.md`.

From the repo root, for each `T` in 0.45 0.50 0.55 0.60 0.70:

```
python -c "import json, sys; c = json.load(open('configs/laplace_default.json')); c['Tcr']['points'][-1][1] = float(sys.argv[1]); json.dump(c, open(f'/tmp/tcr_T{sys.argv[1]}.json', 'w'), indent=2)" $T
python -m sttopt.stto_laplace_cli --config /tmp/tcr_T$T.json --tag tcr_C105_T$T
```

and the reference:

```
python -m sttopt.stto_laplace_cli --config configs/seq_vs_sim/plain_to.json --tag tcr_C105_plainTO
python -m sttopt.stto_laplace_cli --config configs/laplace_default.json --tag tcr_C105_sequential --fixed-density output/tcr_C105_plainTO
```
