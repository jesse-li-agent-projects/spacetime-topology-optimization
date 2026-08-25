"""Literal transliteration of `conductivity_estimation_stto_main.m`'s main while-loop.

The oracle for `optimize.step`/`optimize.run`'s *wiring* -- constraint row order,
per-iteration state threading, and the periodic `rou`/`beta`/`factor` schedules --
at parameter points the committed `e2e.mat` fixture (one grid, tfield=3, 3
iterations) never reaches.

Transcribed from the MATLAB main script, not from `optimize.py` and not from
`generate_fixtures.m`. It calls `sttopt.mma` for the MMA solve itself (that module is
separately fixture-validated against `mmasub.m`/`subsolv.m`), so any disagreement
with `optimize.step` isolates to orchestration rather than to the optimizer. See
`matlab_reference.py` for the per-block oracles this builds on.
"""

import numpy as np

import matlab_reference as ref

import sttopt.mma as mma


def run_reference_loop(
    nelx,
    nely,
    nloop,
    nStage,
    volfrac,
    Theta,
    Tcr,
    tfield,
    rmin,
    lrmin,
    rmin_cond,
    beta0=1.0,
    beta_max=128.0,
):
    nel = nelx * nely

    L = ref.ref_continuity_filter(nelx, nely, lrmin)
    Emax, Emin, nu, penal = 1.0, 1e-9, 0.3, 3
    KE = ref.ref_KE(nu)
    H, Hs = ref.ref_density_filter(nelx, nely, rmin)
    C = ref.ref_gravity_C(nelx, nely)
    beta, eta = beta0, 0.5

    # Deliberate divergence from the MATLAB source, which assigns `xTilde = x` and
    # `t = tPhys` unfiltered here: this oracle follows `init_state`'s corrected
    # initialization rather than transliterating the bug it fixes. See PR #26.
    x = np.full((nely, nelx), volfrac)
    xTilde = (H @ x.flatten() / Hs).reshape(nely, nelx)
    xPhys = (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    )
    t = ref.ref_timefield(nelx, nely, tfield)
    tPhys = (H @ t.flatten() / Hs).reshape(nely, nelx)

    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[2 * (nelx + 1) * (nely + 1) - 1] = -1.0
    freedofs1 = np.setdiff1d(np.arange(1, ndof + 1), np.arange(1, 2 * (nely + 1) + 1))

    xold1 = np.concatenate([x.flatten(), np.zeros(nel)])
    xold2 = xold1.copy()
    low = np.zeros(2 * nel)
    upp = np.zeros(2 * nel)
    loop = 0
    rou = 10.0
    factor = 1.0

    N_el, w_el = ref.ref_neighbors(nelx, nely, rmin_cond)
    WE = ref.ref_WE(N_el, w_el, nel)

    trace = []
    while loop < nloop:
        loop += 1
        if loop % 30 == 0 and rou < 50:
            rou = rou + 5
        if loop % 50 == 0 and beta <= beta_max:
            beta = beta * 2
        if beta > beta_max:
            beta = beta_max

        dc = np.zeros(nel)
        dt = np.zeros(nel)

        c, dcx = ref.ref_whole_compliance(
            nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs1, F
        )
        obj = c
        objf = c
        dx = (
            beta
            * (1 - np.tanh(beta * (xTilde - eta)) ** 2)
            / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
        )
        dc = dc + H @ (dcx.flatten() * dx.flatten() / Hs)

        tP = np.linspace(0, 1, nStage + 1)
        for i in range(1, nStage + 1):
            ti = tP[i]
            cg, dcxg, dctg = ref.ref_gravity_compliance(
                nelx, nely, KE, xPhys, tPhys, Emin, Emax, penal, ti, C, rou, freedofs1
            )
            obj = obj + Theta * cg
            dc = dc + Theta * (H @ (dcxg * dx.flatten() / Hs))
            dt = dt + Theta * (H @ (dctg / Hs))

        df0dx = np.concatenate([dc, dt])
        f0val = obj
        n = 2 * nel

        move = 0.01
        tmove = 0.01
        xminx = np.maximum(0.0, x.flatten() - move)
        xmaxx = np.minimum(1.0, x.flatten() + move)
        xmint = np.maximum(0.0, t.flatten() - tmove)
        xmaxt = np.minimum(1.0, t.flatten() + tmove)
        xmin = np.concatenate([xminx, xmint])
        xmax = np.concatenate([xmaxx, xmaxt])
        xval = np.concatenate([x.flatten(), t.flatten()])

        fval = [xPhys.sum() / (nelx * nely * volfrac) - 1]
        dv = np.ones(nel)
        dv = H @ (dv * dx.flatten() / Hs)
        dfdx = [np.concatenate([dv / (nelx * nely * volfrac), np.zeros(nel)])]
        vol = xPhys.sum() / (nelx * nely)

        # 1-indexed; column 0 (all rows) under the new C-order element enumeration.
        Nei = np.array([1]) if tfield == 1 else np.arange(0, nely) * nelx + 1
        kk = 2 * nel
        A = L @ tPhys.flatten()
        B = A**2 / nel
        fval.append(kk * (B.sum() - 1.0e-6))
        dft = kk * 2 * (L.T @ A)
        dft = H @ (dft / Hs) / nel
        dfdx.append(np.concatenate([np.zeros(nel), dft]))

        for ii in range(len(Nei)):
            fval.append(tPhys.flatten()[Nei[ii] - 1] - 1.0e-9)
        ss = np.zeros((len(Nei), nel))
        for ii in range(len(Nei)):
            ss[ii, Nei[ii] - 1] = 1.0
        block = (H @ (ss.T / Hs[:, None])).T
        for ii in range(len(Nei)):
            dfdx.append(np.concatenate([np.zeros(nel), block[ii]]))

        percent = 1 / nStage
        for i in range(1, nStage + 1):
            ti = tP[i]
            ft = 1 - (np.tanh(rou * ti) + np.tanh(rou * (tPhys - ti))) / (
                np.tanh(rou * ti) + np.tanh(rou * (1 - ti))
            )
            dfdt = -(rou * (np.tanh(rou * (tPhys - ti)) ** 2 - 1)) / (
                np.tanh(rou * (ti - 1)) - np.tanh(rou * ti)
            )
            xtJoint = xPhys * ft
            fval.append(xtJoint.sum() / (nelx * nely * volfrac) - i * percent)
            dfx = ft / (nelx * nely * volfrac)
            dfx = H @ (dfx.flatten() * dx.flatten() / Hs)
            dft2 = xPhys * dfdt / (nelx * nely * volfrac)
            dft2 = H @ (dft2.flatten() / Hs)
            dfdx.append(np.concatenate([dfx, dft2]))
            fval.append(-xtJoint.sum() / (nelx * nely * volfrac) + i * percent - 1.0e-5)
            dfdx.append(np.concatenate([-dfx, -dft2]))

        # hotspot: needs the factor-refresh recipe, which reads numer *before* refresh
        XPhys = xPhys.flatten()
        fv_pre, _, _, K_est, numer = ref.ref_hotspot(
            nelx, nely, xPhys, tPhys, N_el, w_el, WE, H, Hs, dx, factor, Tcr
        )
        if loop % 25 == 0:
            max_g = float(((1 - K_est) * XPhys**0.05).max())
            factor = max_g / numer
        tru_max = factor * numer
        fv, df1, dt1, _, _ = ref.ref_hotspot(
            nelx, nely, xPhys, tPhys, N_el, w_el, WE, H, Hs, dx, factor, Tcr
        )
        fval.append(fv)
        dfdx.append(np.concatenate([df1, dt1]))

        fval = np.array(fval)
        dfdx = np.array(dfdx)
        m = len(fval)

        a0 = 1.0
        a = np.zeros(m)
        c_ = np.full(m, 2500.0)
        d = np.zeros(m)
        xmma, ymma, zmma, lam, xsi, eta_, mu, zet, s_, low, upp = mma.mmasub(
            m,
            n,
            loop,
            xval,
            xmin,
            xmax,
            xold1,
            xold2,
            f0val,
            df0dx,
            fval,
            dfdx,
            low,
            upp,
            a0,
            a,
            c_,
            d,
        )

        xold2 = xold1
        xold1 = xval
        s = xmma[:nel].reshape(nely, nelx)

        xTilde = (H @ s.flatten() / Hs).reshape(nely, nelx)
        xPhys = (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / (
            np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
        )
        x = s

        t = xmma[nel:].reshape(nely, nelx)
        tPhys = (H @ t.flatten() / Hs).reshape(nely, nelx)

        trace.append(
            dict(
                loop=loop,
                objf=objf,
                f0val=f0val,
                vol=vol,
                tru_max=tru_max,
                fval=fval,
                dfdx=dfdx,
                df0dx=df0dx,
                xmma=xmma,
                low=low,
                upp=upp,
                lam=lam,
                xPhys=xPhys.copy(),
                tPhys=tPhys.copy(),
                factor=factor,
                rou=rou,
                beta=beta,
                m=m,
            )
        )
    return trace
