"""Regenerates every `.npz` file under `tests/fixtures/` by calling the current
`sttopt` code directly -- no MATLAB involved.

These are golden/regression fixtures: frozen snapshots of what the current Python
implementation produces, not an independent cross-check against another
implementation (that role is filled by `tests/matlab_reference.py`/
`matlab_reference_loop.py`, which stay untouched by this script). Their job is to
catch an unintended future change to these functions' output, not to validate
correctness -- correctness is established by the closed-form/first-principles tests
alongside each fixture-based test, and by `test_reference_sweep.py`'s oracle sweep.

Run from the repo root, so `sttopt`/`tests` resolve to this checkout rather than to
whichever one the environment has installed:
    PYTHONPATH=. python tests/fixtures/generate_fixtures.py
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.fem as fem
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.mma as mma
import sttopt.stto as stto
import sttopt.run_config as run_config
import sttopt.timefield as timefield
import sttopt.torch_util as torch_util
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
ORACLE_FACTOR = 1.0

# The rest of RunConfig's hyperparameters (eta, beta_d_max, p, q, r, rouf, a0, mma_c,
# move, tmove, hotspot_*) aren't varied by these fixtures, so they come from
# configs/default.json rather than being restated here.
_DEFAULT_CONFIG = run_config.RunConfig.from_dict(
    json.loads((OUT.parent.parent / "configs" / "default.json").read_text())
)
# The radii above are in elements, of the default size.
_H = _DEFAULT_CONFIG.element_size_m
CONFIG = dataclasses.replace(
    _DEFAULT_CONFIG,
    width_m=NELX * _H,
    height_m=NELY * _H,
    nelx=NELX,
    nStage=NSTAGE,
    volfrac=VOLFRAC,
    Theta=THETA,
    Tcr=TCR,
    print_base=TFIELD.name.lower(),
    rmin_m=RMIN * _H,
    time_filter_rmin_m=RMIN * _H,
    lrmin_m=LRMIN * _H,
    rmin_cond_m=RMIN_COND * _H,
    nloop=NLOOP,
)


def N(x):
    """`torch_util.to_numpy`, short for the many conversions at the `np.savez` boundary."""
    return torch_util.to_numpy(x)


def main():
    # -- fem_setup.npz: KE, edofMat -----------------------------------------------
    KE = fem.plane_stress_KE(nu=NU)
    edofMat = fem.element_dof_map(NELX, NELY)
    np.savez(OUT / "fem_setup.npz", KE=KE, edofMat=edofMat, nelx=NELX, nely=NELY)

    # -- fem_solve.npz: standalone FE solve at the initial (uniform) density ------
    # CPU float64 regardless of the machine, so a regeneration is reproducible.
    problem = stto.build_problem(CONFIG, device="cpu", dtype=torch.float64)
    xPhys0 = filters.heaviside_projection(
        torch.full((NELY, NELX), VOLFRAC, dtype=torch.float64),
        BETA_D_INIT,
        problem.config.eta,
    )
    c0, dcx0 = compliance_ref.whole_compliance(
        xPhys0,
        problem.KE,
        problem.edofMat,
        EMIN,
        EMAX,
        PENAL,
        problem.freedofs,
        problem.F,
        problem.ndof,
    )
    K0 = fem_ref.assemble_stiffness(
        KE, N(xPhys0), EMIN, EMAX, PENAL, edofMat, problem.ndof
    )
    U0 = fem_ref.solve_fe(K0, N(problem.F), N(problem.freedofs))
    np.savez(
        OUT / "fem_solve.npz",
        xPhys0=N(xPhys0),
        U0=U0,
        c0=c0,
        dcx0=N(dcx0),
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
    # recomputing each module's own intermediate outputs alongside stto.step's
    # so per-module fixtures agree with the trajectory by construction. -------------
    state = stto.init_state(problem)
    xPhys, tPhys = stto.physical_fields(problem, state.x, state.t, state.beta_d)

    xPhys_traj = [N(xPhys)]
    tPhys_traj = [N(tPhys)]
    records = []
    dx_all = np.zeros((NELY, NELX, NLOOP))
    t_scale_all = np.zeros(NLOOP)
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
    x_1, t_1 = N(state.x).flatten(), N(state.t).flatten()
    xval_1 = np.concatenate([x_1, t_1])
    # The bounds and trust region `stto.step` hands `mmasub`.
    xmin_1 = np.zeros(n)
    xmax_1 = np.ones(n)
    move_1 = np.concatenate(
        [
            np.full(x_1.size, problem.config.move),
            np.full(t_1.size, problem.config.tmove),
        ]
    )
    trust_1 = mma.trust_region_params(
        torch.from_numpy(move_1), torch.from_numpy(xmax_1 - xmin_1)
    )
    xold1_1 = N(state.xold1)
    xold2_1 = N(state.xold2)
    low_1 = np.zeros(n)
    upp_1 = np.zeros(n)

    for k in range(NLOOP):
        xTilde = filters.apply_density_filter(state.x, problem.H, problem.Hs)
        dx = filters.heaviside_projection_derivative(
            xTilde, state.beta_d, problem.config.eta
        )
        dx_all[:, :, k] = N(dx)
        # `stto.physical_fields` divides the filtered time field by this, a constant to
        # the gradient that the MATLAB-form oracles' chain rule does not include.
        t_scale_all[k] = float(
            filters.apply_density_filter(state.t, problem.time_H, problem.time_Hs).max()
        )

        c_whole, dcx_whole = compliance_ref.whole_compliance(
            xPhys,
            problem.KE,
            problem.edofMat,
            EMIN,
            EMAX,
            PENAL,
            problem.freedofs,
            problem.F,
            problem.ndof,
        )
        c_whole_all[k] = c_whole
        dcx_whole_all[:, :, k] = N(dcx_whole)

        for i, ti in enumerate(np.linspace(0, 1, NSTAGE + 1)[1:]):
            c_grav, dcx_grav, dct_grav = compliance_ref.gravity_compliance(
                xPhys,
                tPhys,
                problem.KE,
                problem.edofMat,
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
            dcx_grav_all[:, i, k] = N(dcx_grav)
            dct_grav_all[:, i, k] = N(dct_grav)

        # The hand-derived oracle is the MATLAB-source formulation (neighborhood
        # normalization, P_MEAN at factor 1), evaluated along whatever trajectory the
        # default config produces -- not the run's own hotspot term.
        K_est = conductivity.estimated_conductivity(
            xPhys,
            tPhys,
            problem.e1,
            problem.e2,
            problem.w,
            problem.config.q,
            problem.config.rouf,
        )
        hotspot = conductivity_ref.hotspot_constraint(
            xPhys,
            tPhys,
            problem.e1,
            problem.e2,
            problem.w,
            dx,
            problem.H,
            problem.Hs,
            ORACLE_FACTOR,
            problem.config.Tcr,
            problem.config.p,
            problem.config.q,
            problem.config.r,
            problem.config.rouf,
        )
        K_est_all[:, k] = N(K_est)
        numer_all[k] = hotspot.numer
        factor_all[k] = ORACLE_FACTOR
        df1_all[:, k] = N(hotspot.df1)
        dt1_all[:, k] = N(hotspot.dt1)

        state, record = stto.step(problem, state)
        xPhys, tPhys = stto.physical_fields(problem, state.x, state.t, state.beta_d)

        xPhys_traj.append(N(xPhys))
        tPhys_traj.append(N(tPhys))
        records.append(record)

    xPhys_traj = np.stack(xPhys_traj, axis=-1)  # (nely, nelx, nloop+1)
    tPhys_traj = np.stack(tPhys_traj, axis=-1)

    def per_iteration(field):
        """`field` of every record, stacked along a trailing iteration axis."""
        return np.stack([getattr(r, field) for r in records], axis=-1)

    objf = per_iteration("obj")
    vol = per_iteration("vol")
    tru_max_all = per_iteration("tru_max")
    fval_all = per_iteration("g")  # (m, nloop)
    dfdx_all = per_iteration("dg")  # (m, n, nloop)
    xmma_all = per_iteration("xmma")
    low_all = per_iteration("low")
    upp_all = per_iteration("upp")
    lam_all = per_iteration("lam")
    m = len(records[0].g)

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
        t_scale_all=t_scale_all,
        m=m,
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
        **{f"trust_{k}": N(v) for k, v in trust_1.items()},
        xold1_1=xold1_1,
        xold2_1=xold2_1,
        f0val_1=records[0].f,
        df0dx_1=records[0].df,
        fval_1=records[0].g,
        dfdx_1=records[0].dg,
        low_1=low_1,
        upp_1=upp_1,
        m=m,
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
