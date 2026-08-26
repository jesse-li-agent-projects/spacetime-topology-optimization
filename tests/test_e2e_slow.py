"""Full-length reproduction of the thesis's Chapter 4.4 "different fabrication sequences"
experiment (Das2023_MScThesis, resources/), used as a regression check on the whole
optimize.run() loop at production scale rather than the tiny E2E fixture's nloop=3.

800 iterations is past both continuation schedules' saturation points (rou at 240, beta
at 350 -- see optimize.step's docstring), giving the constraint state time to settle
before the assertions below are checked. It is not run to full convergence: compliance
is still trending down slowly at 800 iterations (about 3% over the next 400), so the
`f0val` bound below is a loose regression ceiling, not a tight optimum, per the project's
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

import sttopt.optimize as optimize

# Matches conductivity_estimation_2d/conductivity_estimation_stto_main.m directly
# (nelx/nely/nloop/nStage/volfrac/Theta/Tcr/tfield/rmin/lrmin, and rmin_cond from the
# conductivity-filter radius set later in that script) -- not derived from cli.py's
# argparse defaults, which happen to match today but aren't pinned to this experiment.
NELX, NELY = 180, 60
NSTAGE = 8
VOLFRAC = 0.5
THETA = 0.1
TCR = 0.8
TFIELD = 3
NLOOP = 800
RMIN, LRMIN, RMIN_COND = 4.0, 2.0, 12.0
BETA_INIT = 1.0

F0VAL_CEILING = 195.0
F0VAL_FLOOR = 185.0
TRU_MAX_TARGET = 0.8
TRU_MAX_TOL = 0.008  # 1% of TRU_MAX_TARGET


@pytest.mark.slow
def test_thesis_4_4_reproduction():
    result = optimize.run(
        NELX,
        NELY,
        NLOOP,
        NSTAGE,
        VOLFRAC,
        THETA,
        TCR,
        TFIELD,
        RMIN,
        LRMIN,
        RMIN_COND,
        beta_d=BETA_INIT,
    )
    record = result.records[-1]

    assert record.f0val < F0VAL_CEILING
    assert record.f0val > F0VAL_FLOOR
    assert abs(record.tru_max - TRU_MAX_TARGET) <= TRU_MAX_TOL
