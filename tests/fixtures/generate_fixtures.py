"""Regenerates every `.npz` file under `tests/fixtures/` by calling the current
`sttopt` code directly -- no MATLAB involved.

These are golden/regression fixtures: frozen snapshots of what the current Python
implementation produces, not an independent cross-check against another
implementation (that role is filled by `tests/matlab_reference.py`/
`matlab_reference_loop.py`, which stay untouched by this script). Their job is to
catch an unintended future change to these functions' output, not to validate
correctness -- correctness is established by the closed-form/first-principles tests
alongside each fixture-based test, and by `test_reference_sweep.py`'s oracle sweep.

Run from anywhere (paths are resolved relative to this script's own location):
    python tests/fixtures/generate_fixtures.py
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.fem as fem
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.optimize as optimize
import sttopt.run_config as run_config
import sttopt.timefield as timefield
import tests.reference.compliance as compliance_ref
import tests.reference.conductivity as conductivity_ref
import tests.reference.fem as fem_ref

OUT = Path(__file__).parent

# Problem size -- deliberately small, asymmetric (nelx != nely), matching the
# retired MATLAB harness so existing shape assumptions in tests keep holding.
NELX, NELY = 7, 5
NSTAGE = 3
VOLFRAC = 0.5
THETA = 0.1
TCR = 0.8
TFIELD = timefield.TimeField.OPPOSITE_CORNER
NLOOP = 3
RMIN = LRMIN = 2.0
RMIN_COND = 3.0
BETA_D_INIT = 1.0
EMIN, EMAX, PENAL = 1e-9, 1.0, 3
NU = 0.3

# The rest of RunConfig's hyperparameters (eta, beta_d_max, p, q, r, rouf, a0, mma_c,
# move, tmove, batch_fem_solves) aren't varied by these fixtures, so they come from
# configs/default.json rather than being restated here.
_DEFAULT_CONFIG = run_config.RunConfig.from_dict(
    json.loads((OUT.parent.parent / "configs" / "default.json").read_text())
)
CONFIG = dataclasses.replace(
    _DEFAULT_CONFIG,
    nelx=NELX,
    nely=NELY,
    nStage=NSTAGE,
    volfrac=VOLFRAC,
    Theta=THETA,
    Tcr=TCR,
    print_base=TFIELD.name.lower(),
    rmin=RMIN,
    lrmin=LRMIN,
    rmin_cond=RMIN_COND,
    nloop=NLOOP,
)


def main():
    # -- fem_setup.npz: KE, edofMat -----------------------------------------------
    KE = fem.plane_stress_KE(nu=NU)
    edofMat = fem.element_dof_map(NELX, NELY)
    np.savez(OUT / "fem_setup.npz", KE=KE, edofMat=edofMat, nelx=NELX, nely=NELY)

    # -- fem_solve.npz: standalone FE solve at the initial (uniform) density ------
    problem = optimize.build_problem(CONFIG)
    xPhys0 = filters.heaviside_projection(
        np.full((NELY, NELX), VOLFRAC), BETA_D_INIT, problem.config.eta
    )
    c0, dcx0 = compliance_ref.whole_compliance(
        xPhys0,
        KE,
        edofMat,
        EMIN,
        EMAX,
        PENAL,
        problem.freedofs,
        problem.F,
        problem.ndof,
    )
    K0 = fem_ref.assemble_stiffness(
        KE, xPhys0, EMIN, EMAX, PENAL, edofMat, problem.ndof
    )
    U0 = fem_ref.solve_fe(K0, problem.F, problem.freedofs)
    np.savez(
        OUT / "fem_solve.npz",
        xPhys0=xPhys0,
        U0=U0,
        c0=c0,
        dcx0=dcx0,
        nelx=NELX,
        nely=NELY,
    )

    # -- filters.npz: H, Hs, L -----------------------------------------------------
    H, Hs = filters.density_filter(NELX, NELY, RMIN)
    L = filters.continuity_filter(NELX, NELY, LRMIN)
    np.savez(
        OUT / "filters.npz",
        H=H.toarray(),
        Hs=Hs,
        L=L.toarray(),
        nelx=NELX,
        nely=NELY,
    )

    # -- gravity.npz: C -------------------------------------------------------------
    C = gravity.gravity_load_matrix(NELX, NELY)
    np.savez(OUT / "gravity.npz", C=C.toarray(), nelx=NELX, nely=NELY)

    # -- timefield.npz: all 3 variants -----------------------------------------------
    tfield1 = timefield.init_timefield(NELX, NELY, timefield.TimeField.CORNER)
    tfield2 = timefield.init_timefield(NELX, NELY, timefield.TimeField.EDGE)
    tfield3 = timefield.init_timefield(NELX, NELY, timefield.TimeField.OPPOSITE_CORNER)
    np.savez(
        OUT / "timefield.npz",
        tfield1=tfield1,
        tfield2=tfield2,
        tfield3=tfield3,
        nelx=NELX,
        nely=NELY,
    )

    # -- conductivity_neighbors.npz: neighbor_weights COO triplets -------------------
    e1, e2, w = conductivity.neighbor_weights(NELX, NELY, RMIN_COND)
    np.savez(
        OUT / "conductivity_neighbors.npz",
        e1=e1,
        e2=e2,
        w=w,
        nelx=NELX,
        nely=NELY,
        rmin_cond=RMIN_COND,
    )

    # -- Main loop: run NLOOP iterations from the current (correct) init_state,
    # recomputing each module's own intermediate outputs alongside optimize.step's
    # so per-module fixtures agree with the trajectory by construction. -------------
    state = optimize.init_state(problem, BETA_D_INIT)

    xPhys_traj = [state.xPhys.copy()]
    tPhys_traj = [state.tPhys.copy()]
    dx_all = np.zeros((NELY, NELX, NLOOP))
    objf = np.zeros(NLOOP)
    vol = np.zeros(NLOOP)
    tru_max_all = np.zeros(NLOOP)
    fval_all = np.zeros((problem.m, NLOOP))
    dfdx_all = np.zeros((problem.m, problem.n, NLOOP))
    xmma_all = np.zeros((problem.n, NLOOP))
    low_all = np.zeros((problem.n, NLOOP))
    upp_all = np.zeros((problem.n, NLOOP))
    lam_all = np.zeros((problem.m, NLOOP))
    K_est_all = np.zeros((NELX * NELY, NLOOP))
    numer_all = np.zeros(NLOOP)
    factor_all = np.zeros(NLOOP)
    df1_all = np.zeros((NELX * NELY, NLOOP))
    dt1_all = np.zeros((NELX * NELY, NLOOP))
    c_whole_all = np.zeros(NLOOP)
    dcx_whole_all = np.zeros((NELY, NELX, NLOOP))
    c_grav_all = np.zeros((NLOOP, NSTAGE))
    dcx_grav_all = np.zeros((NELX * NELY, NSTAGE, NLOOP))
    dct_grav_all = np.zeros((NELX * NELY, NSTAGE, NLOOP))

    # iteration-1 mmasub snapshot (matches test_mma.py's standalone mmasub call)
    n = problem.n
    xval_1 = np.concatenate([state.x.flatten(), state.t.flatten()])
    xmin_1 = np.concatenate(
        [
            np.maximum(0.0, state.x.flatten() - problem.config.move),
            np.maximum(0.0, state.t.flatten() - problem.config.tmove),
        ]
    )
    xmax_1 = np.concatenate(
        [
            np.minimum(1.0, state.x.flatten() + problem.config.move),
            np.minimum(1.0, state.t.flatten() + problem.config.tmove),
        ]
    )
    xold1_1 = state.xold1.copy()
    xold2_1 = state.xold2.copy()
    low_1 = np.zeros(n)
    upp_1 = np.zeros(n)

    # Overwritten on the k == 0 pass below; NLOOP > 0 guarantees that pass runs.
    f0val_1 = 0.0
    df0dx_1 = np.zeros(n)
    fval_1 = np.zeros(problem.m)
    dfdx_1 = np.zeros((problem.m, n))

    for k in range(NLOOP):
        dx = filters.heaviside_projection_derivative(
            state.xTilde, state.beta_d, problem.config.eta
        )
        dx_all[:, :, k] = dx

        c_whole, dcx_whole = compliance_ref.whole_compliance(
            state.xPhys,
            KE,
            edofMat,
            EMIN,
            EMAX,
            PENAL,
            problem.freedofs,
            problem.F,
            problem.ndof,
        )
        c_whole_all[k] = c_whole
        dcx_whole_all[:, :, k] = dcx_whole

        for i, ti in enumerate(np.linspace(0, 1, NSTAGE + 1)[1:]):
            c_grav, dcx_grav, dct_grav = compliance_ref.gravity_compliance(
                state.xPhys,
                state.tPhys,
                KE,
                edofMat,
                EMIN,
                EMAX,
                PENAL,
                ti,
                problem.C,
                state.beta_t,
                problem.freedofs,
                problem.ndof,
            )
            c_grav_all[k, i] = c_grav
            dcx_grav_all[:, i, k] = dcx_grav
            dct_grav_all[:, i, k] = dct_grav

        K_est = conductivity.estimated_conductivity(
            state.xPhys, state.tPhys, e1, e2, w, problem.config.q, problem.config.rouf
        )
        hotspot = conductivity_ref.hotspot_constraint(
            state.xPhys,
            state.tPhys,
            e1,
            e2,
            w,
            dx,
            problem.H,
            problem.Hs,
            state.factor,
            problem.config.Tcr,
            problem.config.p,
            problem.config.q,
            problem.config.r,
            problem.config.rouf,
        )
        K_est_all[:, k] = K_est
        numer_all[k] = hotspot.numer
        factor_all[k] = state.factor
        df1_all[:, k] = hotspot.df1
        dt1_all[:, k] = hotspot.dt1

        state, record = optimize.step(problem, state)

        xPhys_traj.append(state.xPhys.copy())
        tPhys_traj.append(state.tPhys.copy())
        objf[k] = record.obj
        vol[k] = record.vol
        tru_max_all[k] = record.tru_max
        fval_all[:, k] = record.fval
        dfdx_all[:, :, k] = record.dfdx
        xmma_all[:, k] = record.xmma
        low_all[:, k] = record.low
        upp_all[:, k] = record.upp
        lam_all[:, k] = record.lam

        if k == 0:
            f0val_1 = record.f0val
            df0dx_1 = record.df0dx
            fval_1 = record.fval
            dfdx_1 = record.dfdx

    xPhys_traj = np.stack(xPhys_traj, axis=-1)  # (nely, nelx, nloop+1)
    tPhys_traj = np.stack(tPhys_traj, axis=-1)

    np.savez(
        OUT / "e2e.npz",
        xPhys_traj=xPhys_traj,
        tPhys_traj=tPhys_traj,
        dx_all=dx_all,
        objf=objf,
        vol=vol,
        tru_max_all=tru_max_all,
        nelx=NELX,
        nely=NELY,
        nStage=NSTAGE,
        volfrac=VOLFRAC,
        Theta=THETA,
        Tcr=TCR,
        tfield=int(TFIELD),
        nloop=NLOOP,
    )

    np.savez(
        OUT / "constraints.npz",
        fval_all=fval_all,
        dfdx_all=dfdx_all,
        m=problem.m,
        n=problem.n,
        nelx=NELX,
        nely=NELY,
        nStage=NSTAGE,
        volfrac=VOLFRAC,
        tfield=int(TFIELD),
    )

    np.savez(
        OUT / "mma.npz",
        xval_1=xval_1,
        xmin_1=xmin_1,
        xmax_1=xmax_1,
        xold1_1=xold1_1,
        xold2_1=xold2_1,
        f0val_1=f0val_1,
        df0dx_1=df0dx_1,
        fval_1=fval_1,
        dfdx_1=dfdx_1,
        low_1=low_1,
        upp_1=upp_1,
        m=problem.m,
        n=problem.n,
        xmma_all=xmma_all,
        low_all=low_all,
        upp_all=upp_all,
        lam_all=lam_all,
        nelx=NELX,
        nely=NELY,
    )

    np.savez(
        OUT / "conductivity.npz",
        K_est_all=K_est_all,
        numer_all=numer_all,
        factor_all=factor_all,
        tru_max_all=tru_max_all,
        df1_all=df1_all,
        dt1_all=dt1_all,
        nelx=NELX,
        nely=NELY,
    )

    np.savez(
        OUT / "compliance.npz",
        c_whole_all=c_whole_all,
        dcx_whole_all=dcx_whole_all,
        c_grav_all=c_grav_all,
        dcx_grav_all=dcx_grav_all,
        dct_grav_all=dct_grav_all,
        nelx=NELX,
        nely=NELY,
        nStage=NSTAGE,
    )

    print("Fixture generation complete.")


if __name__ == "__main__":
    main()
