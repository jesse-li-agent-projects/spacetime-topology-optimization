"""Method of Moving Asymptotes (MMA), Svanberg's primal-dual Newton subproblem solver.

Verbatim-as-reasonable port of `mmasub.m`/`subsolv.m` from Krister Svanberg's
GCMMA-MMA-code (Copyright (C) 2006-2008 Krister Svanberg, GNU GPL v3+,
<http://www.smoptit.se/>). `mmasub` builds one convex, separable MMA subproblem
(moving asymptotes + linearized constraints) around the current design point;
`subsolv` solves that subproblem to optimality via a primal-dual interior-point
Newton method with a decreasing barrier parameter `epsi`. This module is a direct
translation, not a redesign -- see conventions.md for the array-order and tolerance
conventions used to test it against the MATLAB source.
"""

import warnings

import torch
from jaxtyping import Float
from torch import Tensor


def mmasub(
    m: int,
    n: int,
    iteration: int,
    xval: Float[Tensor, " n"],
    xmin: Float[Tensor, " n"],
    xmax: Float[Tensor, " n"],
    xold1: Float[Tensor, " n"],
    xold2: Float[Tensor, " n"],
    f0val: float,
    df0dx: Float[Tensor, " n"],
    fval: Float[Tensor, " m"],
    dfdx: Float[Tensor, "m n"],
    low: Float[Tensor, " n"],
    upp: Float[Tensor, " n"],
    a0: float,
    a: Float[Tensor, " m"],
    c: Float[Tensor, " m"],
    d: Float[Tensor, " m"],
):
    """One MMA iteration: update the moving asymptotes and solve the resulting subproblem.

    Solves  minimize f_0(x) + a0*z + sum(c_i*y_i + 0.5*d_i*y_i^2)
            s.t.      f_i(x) - a_i*z - y_i <= 0,  i = 1..m
                      xmin_j <= x_j <= xmax_j,     j = 1..n
                      z >= 0, y_i >= 0

    `iteration` is the current outer-loop count (1 the first time this is called);
    `xold1`/`xold2`/`low`/`upp` carry state from previous calls (unused when
    `iteration < 2.5`, which re-initializes the asymptotes from scratch). `f0val` is
    unused by the algorithm itself (kept for signature fidelity with the MATLAB source,
    which also never reads it). Returns
    `(xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp)` -- the subproblem's
    primal optimum, its Lagrange multipliers/slacks, and the asymptotes to pass into
    the next call.
    """
    epsimin = 1e-7
    raa0 = 1e-5
    move = 0.5
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7

    device, dtype = xval.device, xval.dtype

    # Asymptotes: re-initialized on the first two iterations, then adapted based on
    # whether xval is oscillating (zzz < 0) or moving monotonically (zzz > 0) relative
    # to the last two iterates.
    if iteration < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = torch.ones(n, device=device, dtype=dtype)
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 10 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10 * (xmax - xmin)
        low = torch.maximum(low, lowmin)
        low = torch.minimum(low, lowmax)
        upp = torch.minimum(upp, uppmax)
        upp = torch.maximum(upp, uppmin)

    # Move-limited bounds alfa, beta for x within this subproblem.
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    alfa = torch.maximum(torch.maximum(zzz1, zzz2), xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    beta = torch.minimum(torch.minimum(zzz1, zzz2), xmax)

    # Separable convex approximations p0/q0 (objective) and P/Q (constraints) of the
    # reciprocal-asymptote form used by MMA, plus the constraint constant term b.
    xmami = torch.clamp(xmax - xmin, min=1e-5)
    xmamiinv = 1.0 / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = 1.0 / ux1
    xlinv = 1.0 / xl1

    p0 = torch.clamp(df0dx, min=0)
    q0 = torch.clamp(-df0dx, min=0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    P = torch.clamp(dfdx, min=0)
    Q = torch.clamp(-dfdx, min=0)
    # raa0*eeem*xmamiinv' in the source is an (m,n) matrix with every row equal to
    # raa0*xmamiinv; broadcasting over rows does the same without materializing it.
    PQ = 0.001 * (P + Q) + raa0 * xmamiinv[None, :]
    P = P + PQ
    Q = Q + PQ
    # P*spdiags(ux2,0,n,n) scales P's columns by ux2; plain broadcasting is simpler
    # and equally correct at these problem sizes (see conventions.md).
    P = P * ux2[None, :]
    Q = Q * xl2[None, :]
    b = P @ uxinv + Q @ xlinv - fval

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def subsolv(
    m: int,
    n: int,
    epsimin: float,
    low: Float[Tensor, " n"],
    upp: Float[Tensor, " n"],
    alfa: Float[Tensor, " n"],
    beta: Float[Tensor, " n"],
    p0: Float[Tensor, " n"],
    q0: Float[Tensor, " n"],
    P: Float[Tensor, "m n"],
    Q: Float[Tensor, "m n"],
    a0: float,
    a: Float[Tensor, " m"],
    b: Float[Tensor, " m"],
    c: Float[Tensor, " m"],
    d: Float[Tensor, " m"],
):
    """Solve the MMA subproblem built by `mmasub` via a primal-dual interior-point Newton method.

    Minimizes  sum(p0_j/(upp_j-x_j) + q0_j/(x_j-low_j)) + a0*z + sum(c_i*y_i + 0.5*d_i*y_i^2)
    subject to sum(P_ij/(upp_j-x_j) + Q_ij/(x_j-low_j)) - a_i*z - y_i <= b_i,
               alfa_j <= x_j <= beta_j,  y_i >= 0,  z >= 0.

    Follows a central path in the barrier parameter `epsi` (geometric decrease by 0.1
    each outer step until `epsi <= epsimin`), taking damped Newton steps on the KKT
    system at each fixed `epsi`. Returns
    `(xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma)`: the primal
    optimum, its slack `s`, and the Lagrange multipliers for the constraint types
    documented in `mmasub`.
    """
    device, dtype = low.device, low.dtype

    x = 0.5 * (alfa + beta)
    y = torch.ones(m, device=device, dtype=dtype)
    z = torch.ones((), device=device, dtype=dtype)
    lam = torch.ones(m, device=device, dtype=dtype)
    xsi = torch.clamp(1.0 / (x - alfa), min=1.0)
    eta = torch.clamp(1.0 / (beta - x), min=1.0)
    mu = torch.maximum(torch.ones(m, device=device, dtype=dtype), 0.5 * c)
    zet = torch.ones((), device=device, dtype=dtype)
    s = torch.ones(m, device=device, dtype=dtype)
    epsi = 1.0

    def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi):
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = 1.0 / ux1
        xlinv1 = 1.0 / xl1
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ uxinv1 + Q @ xlinv1
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsi
        reeta = eta * (beta - x) - epsi
        remu = mu * y - epsi
        rezet = zet * z - epsi
        res = lam * s - epsi
        # epsi (a scalar barrier target) broadcasts in place of the source's
        # epsvecn/epsvecm (epsi*ones(n)/epsi*ones(m)); no need to materialize them.
        return torch.cat(
            [rex, rey, rez[None], relam, rexsi, reeta, remu, rezet[None], res]
        )

    while epsi > epsimin:
        residu = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
        residunorm = torch.linalg.norm(residu)
        residumax = torch.abs(residu).max()

        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = 1.0 / ux1
            xlinv1 = 1.0 / xl1
            uxinv2 = 1.0 / ux2
            xlinv2 = 1.0 / xl2
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ uxinv1 + Q @ xlinv1
            # P*spdiags(uxinv2)-Q*spdiags(xlinv2): scale columns by broadcasting.
            GG = P * uxinv2[None, :] - Q * xlinv2[None, :]
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagy = d + mu / y
            diaglamyi = s / lam + 1.0 / diagy

            # Solve the bordered KKT system for the Newton step. Two equivalent
            # eliminations, chosen by whichever is the smaller dense system: the
            # source ports both `m < n` and `m >= n` branches of Svanberg's code, and
            # so does this port. Only `m < n` is fixture-tested (the real problem here
            # has m << n); the `m >= n` branch below is ported faithfully but only
            # smoke-tested against a synthetic problem, not validated against MATLAB.
            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = torch.cat([blam, delz[None]])
                # spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG': the diag
                # scales GG's columns (by n-indexed diagxinv), so it's on the right.
                Alam = torch.diag(diaglamyi) + (GG * (1.0 / diagx)[None, :]) @ GG.T
                AA = torch.cat(
                    [
                        torch.cat([Alam, a[:, None]], dim=1),
                        torch.cat([a[None, :], (-zet / z)[None, None]], dim=1),
                    ]
                )
                # The bordered (m+1, m+1) system is small (~80x80 at production) and
                # measured faster to factor on the CPU than the GPU (see
                # `plans/torch_port_part2.md`'s Phase 3.5 results) -- a defensible
                # exception to the no-round-trips rule, since it moves ~51 KB.
                solut = torch.linalg.solve(AA.cpu(), bb.cpu()).to(device)
                dlam = solut[:m]
                dz = solut[m]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                diaglamyiinv = 1.0 / diaglamyi
                dellamyi = dellam + dely / diagy
                # spdiags(diagx,0,n,n) + GG'*spdiags(diaglamyiinv,0,m,m)*GG: the diag
                # scales GG.T's columns (by m-indexed diaglamyiinv), so it's on the
                # right of GG.T here -- easy to misread as scaling GG's rows instead.
                Axx = torch.diag(diagx) + (GG.T * diaglamyiinv[None, :]) @ GG
                azz = zet / z + a @ (a / diaglamyi)
                axz = -GG.T @ (a / diaglamyi)
                bx = delx + GG.T @ (dellamyi / diaglamyi)
                bz = delz - a @ (dellamyi / diaglamyi)
                AA = torch.cat(
                    [
                        torch.cat([Axx, axz[:, None]], dim=1),
                        torch.cat([axz[None, :], azz[None, None]], dim=1),
                    ]
                )
                bb = torch.cat([-bx, (-bz)[None]])
                solut = torch.linalg.solve(AA.cpu(), bb.cpu()).to(device)
                dx = solut[:n]
                dz = solut[n]
                dlam = (
                    (GG @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi
                )

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam

            xx = torch.cat([y, z[None], lam, xsi, eta, mu, zet[None], s])
            dxx = torch.cat([dy, dz[None], dlam, dxsi, deta, dmu, dzet[None], ds])

            # Fraction-to-the-boundary step size limit (1.01x safety margin), then
            # backtracking line search on the residual norm below.
            stmxx = (-1.01 * dxx / xx).max()
            stmalfa = (-1.01 * dx / (x - alfa)).max()
            stmbeta = (1.01 * dx / (beta - x)).max()
            steg = 1.0 / max(
                stmalfa, stmbeta, stmxx, torch.ones((), device=device, dtype=dtype)
            )

            xold, yold, zold = x, y, z
            lamold, xsiold, etaold = lam, xsi, eta
            muold, zetold, sold = mu, zet, s

            itto = 0
            resinew = 2 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                residu = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
                resinew = torch.linalg.norm(residu)
                steg = steg / 2

            residunorm = resinew
            residumax = torch.abs(residu).max()
            steg = 2 * steg

        if ittt > 198:
            warnings.warn(
                f"subsolv: inner Newton loop hit the iteration cap (ittt={ittt}) "
                f"at epsi={epsi} without reaching the 0.9*epsi residual target",
                RuntimeWarning,
            )

        epsi = 0.1 * epsi

    return x, y, z, lam, xsi, eta, mu, zet, s
