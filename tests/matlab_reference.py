"""Literal, loop-for-loop transliteration of `conductivity_estimation_stto_main.m`.

An independent oracle for the tests in `test_reference_sweep.py`. The committed
`.mat` fixtures pin exactly one parameter point (nelx=7, nely=5, rmin=2, rmin_cond=3,
tfield=3, nStage=3), so a transposition or fencepost bug that only shows up at some
other radius, grid shape, or `tfield` variant would pass the whole fixture suite.
This module closes that gap without needing MATLAB: it is transcribed directly from
the MATLAB main script -- NOT from `sttopt/`, and NOT from `generate_fixtures.m`
(which is itself a copy that could have drifted) -- so agreement between the two is
real evidence rather than a shared reading of the same Python.

Deliberately dumb and slow: 1-indexed arithmetic kept as in the source (with a `-1`
only at the array-access boundary), dense matrices, no vectorization. Speed does not
matter here; independence does. Do not "optimize" this file to look like `sttopt/` --
that would destroy the only property that makes it useful.
"""

import numpy as np

# ---------------------------------------------------------------- filters


def ref_continuity_filter(nelx, nely, lrmin=2):
    n = nelx * nely
    L = np.zeros((n, n))
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1
            for i2 in range(
                max(i1 - (int(np.ceil(lrmin)) - 1), 1),
                min(i1 + (int(np.ceil(lrmin)) - 1), nelx) + 1,
            ):
                for j2 in range(
                    max(j1 - (int(np.ceil(lrmin)) - 1), 1),
                    min(j1 + (int(np.ceil(lrmin)) - 1), nely) + 1,
                ):
                    e2 = (i2 - 1) * nely + j2
                    if e1 == e2:
                        continue
                    L[e1 - 1, e2 - 1] += 1
    M = L.sum(axis=1, keepdims=True)
    return np.eye(n) - L / M


def ref_density_filter(nelx, nely, rmin):
    n = nelx * nely
    H = np.zeros((n, n))
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1
            for i2 in range(
                max(i1 - (int(np.ceil(rmin)) - 1), 1),
                min(i1 + (int(np.ceil(rmin)) - 1), nelx) + 1,
            ):
                for j2 in range(
                    max(j1 - (int(np.ceil(rmin)) - 1), 1),
                    min(j1 + (int(np.ceil(rmin)) - 1), nely) + 1,
                ):
                    e2 = (i2 - 1) * nely + j2
                    H[e1 - 1, e2 - 1] += max(
                        0.0, rmin - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)
                    )
    Hs = H.sum(axis=1)
    return H, Hs


# ------------------------------------------------- conductivity neighbours


def ref_neighbors(nelx, nely, rmin):
    """Returns N_el, w_el as 1-indexed lists (index e-1 -> list for element e)."""
    N_el = [None] * (nelx * nely)
    w_el = [None] * (nelx * nely)
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1
            Nel = []
            w = []
            for i2 in range(
                max(i1 - (int(np.ceil(rmin)) - 1), 1),
                min(i1 + (int(np.ceil(rmin)) - 1), nelx) + 1,
            ):
                for j2 in range(
                    max(j1 - (int(np.ceil(rmin)) - 1), 1),
                    min(j1 + (int(np.ceil(rmin)) - 1), nely) + 1,
                ):
                    if rmin - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2) >= 0:
                        e2 = (i2 - 1) * nely + j2
                        if e2 == e1:
                            sH1 = rmin
                            wk = sH1 / rmin
                        else:
                            dist = np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)
                            sH1 = max(0.0, rmin - dist)
                            wk = sH1 / rmin
                        Nel.append(e2)
                        w.append(wk)
            N_el[e1 - 1] = Nel
            w_el[e1 - 1] = np.array(w)
    return N_el, w_el


def ref_WE(N_el, w_el, nel):
    WE = [None] * nel
    for i in range(1, nel + 1):
        E1 = N_el[i - 1]
        We = []
        for j in range(len(E1)):
            w1 = w_el[E1[j] - 1]
            n1 = N_el[E1[j] - 1]
            We.append(w1[n1.index(i)])
        WE[i - 1] = np.array(We)
    return WE


# ------------------------------------------------------------- time field


def ref_timefield(nelx, nely, tfield):
    if tfield == 1 or tfield == 3:
        ypos = np.linspace(0, nely, nely)
        xpos = np.linspace(0, nelx, nelx)
        xmesh, ymesh = np.meshgrid(xpos, ypos)
        pos = np.stack([xmesh.flatten(order="F"), ymesh.flatten(order="F")], axis=1)
        start_pos = (
            np.array([0.0, 0.0]) if tfield == 1 else np.array([0.0, float(nely)])
        )
        vec = pos - start_pos
        dis2 = (vec * vec).sum(axis=1)
        t = np.sqrt(dis2) / np.sqrt(dis2).max()
        return t.reshape(nely, nelx, order="F")
    else:
        tP = np.zeros((nely, nelx))
        t = np.linspace(0, 1, nelx)
        for i in range(1, nelx + 1):
            tP[:, i - 1] = t[i - 1]
        return tP


# ------------------------------------------------------------------- FEM


def ref_KE(nu):
    A11 = np.array(
        [[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], float
    )
    A12 = np.array(
        [[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], float
    )
    B11 = np.array(
        [[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], float
    )
    B12 = np.array(
        [[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], float
    )
    A = np.block([[A11, A12], [A12.T, A11]])
    B = np.block([[B11, B12], [B12.T, B11]])
    return 1 / (1 - nu**2) / 24 * (A + nu * B)


def ref_edofMat(nelx, nely):
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(
        1 + nely, 1 + nelx, order="F"
    )
    edofVec = (2 * nodenrs[:-1, :-1] + 1).flatten(order="F")
    off = np.array(
        [0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely + 0, 2 * nely + 1, -2, -1]
    )
    return edofVec[:, None] + off[None, :]  # 1-indexed dofs


def ref_gravity_C(nelx, nely):
    C = np.zeros(((nelx + 1) * (nely + 1), nelx * nely))
    fe = 1 / (nely * nelx)
    for x in range(1, nely + 1):
        for y in range(1, nelx + 1):
            col = (y - 1) * nely + x
            for row in [
                (y - 1) * (nely + 1) + x,
                (y - 1) * (nely + 1) + x + 1,
                y * (nely + 1) + x,
                y * (nely + 1) + x + 1,
            ]:
                C[row - 1, col - 1] += fe / 4
    return C


def _assemble(KE, dens, edofMat, ndof):
    K = np.zeros((ndof, ndof))
    for e in range(edofMat.shape[0]):
        dofs = edofMat[e] - 1
        K[np.ix_(dofs, dofs)] += KE * dens[e]
    return (K + K.T) / 2


def ref_whole_compliance(nelx, nely, KE, xPhys, Emin, Emax, penal, freedofs, F):
    edofMat = ref_edofMat(nelx, nely)
    ndof = 2 * (nely + 1) * (nelx + 1)
    dens = Emin + xPhys.flatten(order="F") ** penal * (Emax - Emin)
    K = _assemble(KE, dens, edofMat, ndof)
    U = np.zeros(ndof)
    fd = freedofs - 1
    U[fd] = np.linalg.solve(K[np.ix_(fd, fd)], F[fd])
    Ue = U[edofMat - 1]
    ce = ((Ue @ KE) * Ue).sum(axis=1).reshape(nely, nelx, order="F")
    c = float(((Emin + xPhys**penal * (Emax - Emin)) * ce).sum())
    dcx = -penal * (Emax - Emin) * xPhys ** (penal - 1) * ce
    return c, dcx


def ref_gravity_compliance(
    nelx, nely, KE, xPhys, tPhys, Emin, Emax, penal, ti, C, lamda, freedofs
):
    ft = 1 - (np.tanh(lamda * ti) + np.tanh(lamda * (tPhys - ti))) / (
        np.tanh(lamda * ti) + np.tanh(lamda * (1 - ti))
    )
    dfdt = -(lamda * (np.tanh(lamda * (tPhys - ti)) ** 2 - 1)) / (
        np.tanh(lamda * (ti - 1)) - np.tanh(lamda * ti)
    )
    xtJoint = xPhys * ft
    edofMat = ref_edofMat(nelx, nely)
    ndof = 2 * (nely + 1) * (nelx + 1)
    dens = Emin + xtJoint.flatten(order="F") ** penal * (Emax - Emin)
    K = _assemble(KE, dens, edofMat, ndof)
    f = -C @ xtJoint.flatten(order="F")
    Fm = np.zeros(((nely + 1) * (nelx + 1), 2))
    Fm[:, 1] = f
    F = Fm.T.flatten(order="F")
    U = np.zeros(ndof)
    fd = freedofs - 1
    U[fd] = np.linalg.solve(K[np.ix_(fd, fd)], F[fd])
    Ue = U[edofMat - 1]
    ce = ((Ue @ KE) * Ue).sum(axis=1).reshape(nely, nelx, order="F")
    c = float(((Emin + xtJoint**penal * (Emax - Emin)) * ce).sum())
    dcx1 = -penal * (Emax - Emin) * xtJoint ** (penal - 1) * ce * ft
    dct1 = -penal * (Emax - Emin) * xtJoint ** (penal - 1) * ce * xPhys * dfdt
    dcx2 = -(U[1::2] @ C) * ft.flatten(order="F")
    dct2 = -(U[1::2] @ C) * xPhys.flatten(order="F") * dfdt.flatten(order="F")
    dcx = 2 * dcx2 + dcx1.flatten(order="F")
    dct = 2 * dct2 + dct1.flatten(order="F")
    return c, dcx, dct


# ------------------------------------------------------- hotspot constraint


def ref_hotspot(
    nelx,
    nely,
    xPhys,
    tPhys,
    N_el,
    w_el,
    WE,
    H,
    Hs,
    dx,
    factor,
    Tcr,
    p=25,
    q=3,
    r=0.05,
    rouf=100,
):
    nel = nely * nelx
    XPhys = xPhys.flatten(order="F")
    TPhys = tPhys.flatten(order="F")
    K_est = np.zeros(nel)
    FT_el = [None] * nel
    DFT_el = [None] * nel
    for l in range(1, nel + 1):
        ti = TPhys[l - 1]
        N_ele = N_el[l - 1]
        FT = np.zeros(len(N_ele))
        DFT = np.zeros(len(N_ele))
        for o in range(len(N_ele)):
            tn = TPhys[N_ele[o] - 1]
            FT[o] = (1 + np.exp(rouf * (tn - ti))) ** (-1)
            if tn == ti:
                DFT[o] = 0.0
            else:
                DFT[o] = (
                    (1 + np.exp(rouf * (tn - ti))) ** (-2)
                    * rouf
                    * np.exp(rouf * (tn - ti))
                )
        FT_el[l - 1] = FT
        DFT_el[l - 1] = DFT

    Nsum3 = np.zeros(nel)
    for i in range(1, nel + 1):
        Nsum3[i - 1] = float((FT_el[i - 1] * w_el[i - 1]).sum())
        idx = np.array(N_el[i - 1]) - 1
        K_est[i - 1] = (
            float((XPhys[idx] ** q * w_el[i - 1] * FT_el[i - 1]).sum()) / Nsum3[i - 1]
        )

    T_val = 1 - K_est
    cond_p = (T_val * XPhys**r) ** p
    sum_cond = cond_p.sum()
    n1 = nel
    numer = (sum_cond / n1) ** (1 / p)
    fval = factor * numer / Tcr - 1

    N_sub11 = [None] * nel
    N_sub22 = [None] * nel
    for i in range(1, nel + 1):
        idx = np.array(N_el[i - 1]) - 1
        X1 = XPhys[idx]
        T_sub_i = T_val[idx] * X1**r
        N1 = Nsum3[idx]
        E1 = N_el[i - 1]
        W1 = WE[i - 1]
        W2 = w_el[i - 1]
        DFT2 = DFT_el[i - 1]
        N_sub1 = np.zeros(len(E1))
        N_sub2 = np.zeros(len(E1))
        for j in range(len(E1)):
            te = FT_el[E1[j] - 1]
            de = DFT_el[E1[j] - 1]
            n11 = N_el[E1[j] - 1]
            pos = n11.index(i)
            F2 = te[pos]
            DFT1 = de[pos]
            if E1[j] == i:
                N_sub2[j] = -X1[j] ** r * q * XPhys[i - 1] ** (q - 1) * F2 * W1[j] / N1[
                    j
                ] + (1 - K_est[i - 1]) * r * X1[j] ** (r - 1)
                N_sub1[j] = -X1[j] ** r * (
                    (X1**q * W2 * DFT2).sum() / N1[j]
                    - K_est[i - 1] * (W2 * DFT2).sum() / N1[j]
                )
            else:
                N_sub2[j] = (
                    -X1[j] ** r * q * XPhys[i - 1] ** (q - 1) * F2 * W1[j] / N1[j]
                )
                N_sub1[j] = (
                    -(
                        -XPhys[i - 1] ** q * W1[j] / N1[j]
                        + K_est[E1[j] - 1] * W1[j] / N1[j]
                    )
                    * X1[j] ** r
                    * DFT1
                )
        N_sub11[i - 1] = (T_sub_i, N_sub1)
        N_sub22[i - 1] = (T_sub_i, N_sub2)

    df1 = np.zeros(nel)
    dt1 = np.zeros(nel)
    numer1 = (sum_cond / n1) ** ((1 / p) - 1)
    denom1 = n1 * Tcr
    for i in range(nel):
        Tsub, ns1 = N_sub11[i]
        _, ns2 = N_sub22[i]
        df1[i] = (factor * numer1 / denom1) * float((Tsub ** (p - 1) * ns2).sum())
        dt1[i] = (factor * numer1 / denom1) * float((Tsub ** (p - 1) * ns1).sum())
    df1 = H @ (df1 * dx.flatten(order="F") / Hs)
    dt1 = H @ (dt1 / Hs)
    return fval, df1, dt1, K_est, numer
