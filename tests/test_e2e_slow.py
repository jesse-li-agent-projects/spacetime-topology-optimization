"""Full-length reproduction of the thesis's Chapter 4.4 "different fabrication sequences"
experiment (Das2023_MScThesis, resources/), used as a regression check on the whole
stto.run() loop at production scale rather than the tiny E2E fixture's nloop=3.

800 iterations is past both continuation schedules' saturation points (rou at 240, beta
at 350 -- see stto.step's docstring), giving the constraint state time to settle
before the assertions below are checked. It is not run to full convergence: compliance
is still trending down slowly at 800 iterations (about 3% over the next 400), so the
`.f` bound below is a loose regression ceiling, not a tight optimum, per the project's
"a 1% change in performance is often negligible" standard.

tru_max is compared against the thesis's published Tmax=0.80 (p.51: "a critical
temperature value of Tcr=0.8 was used for all these designs"). t_max_thresh (Tmax
recomputed on the hard-thresholded design, matching the thesis's Eqn 4.15 definition
more literally) is intentionally not asserted here: it lags tru_max by several percent
at 800 iterations (a continuous-relaxation-vs-discrete-realization gap) and is not yet
settled -- its argmax element can still jump between competing hot spots this late in
the run, making a tight bound on it fragile.
"""

import pytest

import sttopt.stto as stto
from sttopt.run_config import RunConfig

# Matches conductivity_estimation_2d/conductivity_estimation_stto_main.m directly
# (nelx/nely/nloop/nStage/volfrac/Theta/Tcr/tfield/rmin/lrmin, and rmin_cond from the
# conductivity-filter radius set later in that script) -- not derived from
# configs/default.json, which happens to match today but isn't pinned to this
# experiment.
NELX, NELY = 180, 60
NSTAGE = 8
VOLFRAC = 0.5
THETA = 0.1
TCR = 0.8
PRINT_BASE = "opposite_corner"
NLOOP = 800
RMIN, LRMIN, RMIN_COND = 4.0, 2.0, 12.0
BETA_INIT = 1.0
# The MATLAB source's projection ramps: `beta_d` doubling every 50 iterations to 128,
# `beta_t` rising by 5 every 30 to 50.
STEP_BETA_D = {
    "points": [
        [0, 1.0],
        [50, 2.0],
        [100, 4.0],
        [150, 8.0],
        [200, 16.0],
        [250, 32.0],
        [300, 64.0],
        [350, 128.0],
    ],
    "mode": "step",
}
STEP_BETA_T = {
    "points": [
        [0, 10.0],
        [30, 15.0],
        [60, 20.0],
        [90, 25.0],
        [120, 30.0],
        [150, 35.0],
        [180, 40.0],
        [210, 45.0],
        [240, 50.0],
    ],
    "mode": "step",
}

CONFIG = RunConfig(
    nloop=NLOOP,
    nelx=NELX,
    nely=NELY,
    volfrac=VOLFRAC,
    nStage=NSTAGE,
    enable_stage_volume=True,
    Theta=THETA,
    uniformity_metric="gradient_cv",
    uniformity_weight=0.0,
    roughness_weight=0.0,
    Tcr=TCR,
    # The MATLAB source's hotspot formulation; beta is inert under p_mean.
    hotspot_normalization="neighborhood",
    hotspot_aggregation="p_mean",
    hotspot_beta=200.0,
    hotspot_refresh_period=25,
    beta_d_schedule=STEP_BETA_D,
    beta_t_schedule=STEP_BETA_T,
    print_base=PRINT_BASE,
    rmin=RMIN,
    time_filter_rmin=RMIN,
    enable_continuity=True,
    continuity_tol=1e-6,
    lrmin=LRMIN,
    rmin_cond=RMIN_COND,
    Emin=1e-9,
    Emax=1.0,
    nu=0.3,
    penal=3.0,
    eta=0.5,
    p=25.0,
    q=3.0,
    r=0.05,
    rouf=100.0,
    a0=1.0,
    mma_c=2500.0,
    move=0.01,
    tmove=0.01,
)

F_CEILING = 195.0
F_FLOOR = 185.0
TRU_MAX_TARGET = 0.8
TRU_MAX_TOL = 0.008  # 1% of TRU_MAX_TARGET


@pytest.mark.slow
def test_thesis_4_4_reproduction():
    result = stto.run(CONFIG, beta_d=BETA_INIT)
    record = result.records[-1]

    assert record.f < F_CEILING
    assert record.f > F_FLOOR
    assert abs(record.tru_max - TRU_MAX_TARGET) <= TRU_MAX_TOL
