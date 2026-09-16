"""Write sweep configs: python3 gen_configs.tmp.py <base default.json> <out dir> <phase>."""

import copy
import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
phase = sys.argv[3]
out.mkdir(parents=True, exist_ok=True)

base.update(
    hotspot_aggregation="logsumexp_severity",
    uniformity_weight=100.0,
)
OFF = 100.0  # Tcr far above any severity: the hotspot row never binds


def cfg(**kw):
    c = copy.deepcopy(base)
    c.update(kw)
    return c


runs = {}
if phase == "p0p1":
    runs["p0_A0"] = cfg(
        Tcr=OFF, uniformity_weight=0.0, roughness_weight=0.0, hotspot_beta=25.0
    )
    runs["p0_A1"] = cfg(Tcr=OFF, hotspot_beta=25.0)
    for b in (4, 12, 25, 50):
        runs[f"p1_b{b}"] = cfg(hotspot_beta=float(b))
    runs["p1_b25_on150"] = cfg(
        hotspot_beta=25.0, Tcr={"points": [[1, 5.0], [150, 5.0], [250, 0.8]]}
    )

if phase == "p1b":
    runs["p1_b25_on150"] = cfg(
        hotspot_beta=25.0, Tcr={"points": [[1, 5.0], [150, 5.0], [250, 0.8]]}
    )
    # Does the hotspot row stay violated only because its multiplier hits mma_c?
    runs["p1_b4_c1e5"] = cfg(hotspot_beta=4.0, mma_c=1e5)
    runs["p1_b25_c1e5"] = cfg(hotspot_beta=25.0, mma_c=1e5)

if phase == "p1c":
    # A stale calibration offset lets the true max drift ~0.03 above Tcr between refreshes.
    runs["p1_b25_r1"] = cfg(hotspot_beta=25.0, hotspot_refresh_period=1)
    runs["p1_b50_r1"] = cfg(hotspot_beta=50.0, hotspot_refresh_period=1)

if phase == "p2":
    # Candidate winner: severity LSE, sharp beta, calibration refreshed every iteration.
    win = dict(hotspot_beta=50.0, hotspot_refresh_period=1)
    runs["p2_tcr079"] = cfg(**win, Tcr=0.79)  # margin for the thresholded design
    runs["p2_b100_r1"] = cfg(hotspot_beta=100.0, hotspot_refresh_period=1)
    runs["p2_rep"] = cfg(**win)  # same settings as p1_b50_r1: determinism check
    runs["p2_corner"] = cfg(**win, print_base="corner")  # not tuned on
    runs["p2_edge"] = cfg(**win, print_base="edge")  # not tuned on

if phase == "p3":
    # Delayed onset costs much less compliance than holding the hotspot row on from
    # iteration 1, and it skips the grey phase where a diluted row stalls the subsolver.
    onset = {"points": [[1, 5.0], [150, 5.0], [250, 0.8]]}
    runs["p3_on150_b50"] = cfg(hotspot_beta=50.0, Tcr=onset)
    runs["p3_on150_b50_r1"] = cfg(
        hotspot_beta=50.0, Tcr=onset, hotspot_refresh_period=1
    )
    runs["p3_on250_b50"] = cfg(
        hotspot_beta=50.0, Tcr={"points": [[1, 5.0], [250, 5.0], [350, 0.8]]}
    )

if phase == "p4":
    # Best of both: delayed onset for compliance, sharp beta + per-iteration refresh for
    # a constraint that is actually held. The earlier ramp leaves more iterations to
    # settle after the row turns on.
    runs["p4_on150_b100_r1"] = cfg(
        hotspot_beta=100.0,
        hotspot_refresh_period=1,
        Tcr={"points": [[1, 5.0], [150, 5.0], [250, 0.8]]},
    )
    runs["p4_on100_b100_r1"] = cfg(
        hotspot_beta=100.0,
        hotspot_refresh_period=1,
        Tcr={"points": [[1, 5.0], [100, 5.0], [200, 0.8]]},
    )

for name, c in runs.items():
    (out / f"{name}.json").write_text(json.dumps(c, indent=2))
    print(name)
