"""Throwaway sanity check for freshly-generated .mat fixtures (Phase 0 only)."""

import sys
from pathlib import Path

import numpy as np
import scipy.io

FIXTURES_DIR = Path(__file__).parent

FILES = [
    "fem_setup.mat",
    "fem_solve.mat",
    "filters.mat",
    "gravity.mat",
    "timefield.mat",
    "conductivity_neighbors.mat",
    "compliance.mat",
    "constraints.mat",
    "conductivity.mat",
    "mma.mat",
    "e2e.mat",
]

ok = True
for name in FILES:
    path = FIXTURES_DIR / name
    if not path.exists():
        print(f"MISSING: {name}")
        ok = False
        continue
    d = scipy.io.loadmat(path, squeeze_me=True)
    keys = [k for k in d if not k.startswith("__")]
    print(f"{name}: {keys}")
    for k in keys:
        v = d[k]
        if isinstance(v, np.ndarray) and v.dtype.kind in "fc":
            if np.isnan(v).any() if not hasattr(v, "toarray") else False:
                print(f"  NaN in {k}!")
                ok = False
        shape = getattr(v, "shape", None)
        print(f"  {k}: type={type(v).__name__} shape={shape}")

print("OK" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
