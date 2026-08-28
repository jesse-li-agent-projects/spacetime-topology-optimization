"""Main-loop orchestration: wires fem/filters/timefield/gravity/compliance/constraints/
conductivity/mma together into the actual space-time topology optimization iteration.

Every module this file calls already owns its own math and its own fixture/FD tests;
this file's only job is *wiring* -- building the stacked objective/constraint arrays
MMA expects, in the exact row order the MATLAB main loop uses, and threading the
iteration-dependent state (`beta_d`, `beta_t`, `factor`, `xold1`/`xold2`, `low`/`upp`, and
the raw-vs-filtered x/t fields) from one call to the next. See `conventions.md` for
array-order conventions and `tests/matlab_reference_loop.py` (a literal transliteration
of the MATLAB source this ports) for the authoritative iteration order.

Two field pairs are threaded per design variable, and must not be conflated: `x`/`t`
are each iteration's *raw* MMA output (unfiltered, unprojected -- what next
iteration's move-limit bounds read); `xTilde`/`xPhys`/`tPhys` are the *filtered* (and,
for density, Heaviside-projected) fields the physics uses.
`xTilde` alone is carried an extra step (needed for the Heaviside-derivative chain rule
at the *start* of the next iteration, before that iteration's own update).

**The tensor boundary (`plans/torch_port_part2.md`).** `Problem` and `State` hold torch
tensors -- `Problem.device`/`.dtype` say where -- and so does every leaf module `step`
calls: `filters`/`compliance`/`constraints`/`conductivity`/`mma` all take and return
tensors, and `compliance.py`'s FEM solve runs through `torch_solve.FemSolve`'s
multigrid-CG, not a NumPy round trip. `fem.py`'s NumPy `assemble_stiffness`/`solve_fe`
stay only as `tests/reference/`'s independent oracle; nothing in `sttopt/` calls them.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.constraints as constraints
import sttopt.fem as fem
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.mma as mma
import sttopt.timefield as timefield
import sttopt.torch_fem as torch_fem
import sttopt.torch_util as torch_util


@dataclass(frozen=True)
class Problem:
    """Fixed problem setup: everything that doesn't change across iterations.

    Built once by `build_problem` and passed unchanged to every `step` call.
    """

    nelx: int
    nely: int
    nStage: int
    volfrac: float
    Theta: float
    Tcr: float
    tfield: timefield.TimeField

    device: torch.device
    dtype: torch.dtype

    KE: Float[Tensor, "8 8"]
    edofMat: Int[Tensor, "nelx*nely 8"]
    freedofs: Int[Tensor, " n_free"]
    free_mask: Bool[Tensor, " ndof"]  # True at free dofs; the matrix-free path's mask
    F: Float[Tensor, " ndof"]
    ndof: int
    H: Tensor  # sparse CSR, shape (nel, nel)
    Hs: Float[Tensor, " nel"]
    L: Tensor  # sparse CSR, shape (nel, nel)
    C: Tensor  # sparse CSR, shape ((nelx+1)*(nely+1), nel)
    e1: Int[Tensor, " npairs"]
    e2: Int[Tensor, " npairs"]
    w: Float[Tensor, " npairs"]
    Nei: Int[Tensor, " k"]

    Emin: float
    Emax: float
    penal: float
    eta: float
    beta_d_max: float
    p: float
    q: float
    r: float
    rouf: float  # hotspot/conductivity-selection sigmoid sharpness (Das 2023 zeta) -- unrelated to beta_t/beta_d
    a0: float
    mma_c: float
    move: float
    tmove: float

    m: int  # number of MMA constraint rows: vol + continuity + start-point(s) + 2*nStage + hotspot
    n: int  # number of MMA design variables: 2*nelx*nely (density half + time half)

    # Batch whole_compliance's solve with every gravity stage's into one FemSolve call
    # (plans/torch_port_part2.md Phase 3.3) rather than 1 + nStage separate calls. Part
    # 1 measured this as a 1.3-1.4x win at 90x30/180x60 and a small loss at 360x120, so
    # it is a per-Problem setting (build_problem defaults it from mesh size), not
    # unconditional.
    batch_fem_solves: bool


@dataclass(frozen=True)
class State:
    """Iteration-dependent state carried from one `step` call to the next."""

    x: Float[Tensor, "nely nelx"]  # raw density (unfiltered MMA output)
    xTilde: Float[Tensor, "nely nelx"]  # density-filtered x
    xPhys: Float[Tensor, "nely nelx"]  # density for physics purposes
    t: Float[Tensor, "nely nelx"]  # raw time field (unfiltered MMA output)
    tPhys: Float[Tensor, "nely nelx"]  # time for physics purposes
    xold1: Float[Tensor, " n"]
    xold2: Float[Tensor, " n"]
    low: Float[Tensor, " n"]
    upp: Float[Tensor, " n"]
    loop: int
    beta_t: float  # gravity/stage-mask sigmoid sharpness
    beta_d: float  # Heaviside projection sharpness
    factor: float  # hotspot-constraint rescaling constant, periodically refreshed

    # Batched FEM solution from this iteration's whole_compliance + gravity_compliance
    # solves, `(1 + nStage, ndof)`, row 0 the whole-structure solve and row `1 + i`
    # stage `i` -- carried forward to warm-start the next iteration's solves (part 1
    # measured ~25% fewer CG iterations). `None` when `Problem.batch_fem_solves` is off
    # (the sequential path doesn't produce a stacked `U` to save -- see optimize.step).
    U: Float[Tensor, "n_stage_plus_1 ndof"] | None


@dataclass(frozen=True)
class IterationRecord:
    """Per-iteration diagnostics and raw MMA outputs, for E2E/MMA/constraint-order tests."""

    obj: float  # whole-structure compliance (doesn't include intermediate structures)
    vol: float  # volume fraction (mean xPhys)
    tru_max: float  # Estimated max hotspot severity (debiased from P-mean)
    f0val: float  # objective (weighted sum of whole & per-stage compliances)
    df0dx: Float[np.ndarray, " n"]
    xmma: Float[np.ndarray, " n"]
    low: Float[np.ndarray, " n"]
    upp: Float[np.ndarray, " n"]
    lam: Float[np.ndarray, " m"]
    fval: Float[np.ndarray, " m"]
    dfdx: Float[np.ndarray, "m n"]


@dataclass(frozen=True)
class RunResult:
    state: State  # final state after nloop iterations
    # length nloop+1, index 0 is initial field
    xPhys_traj: list[Float[Tensor, "nely nelx"]]
    tPhys_traj: list[Float[Tensor, "nely nelx"]]  # length nloop+1
    # the raw (unfiltered, unprojected) counterpart of xPhys_traj/tPhys_traj -- what
    # `state.x`/`state.t` held at each loop, i.e. the input `step` would need to
    # reproduce that loop's xPhys/tPhys, not the fields themselves
    x_traj: list[Float[Tensor, "nely nelx"]]
    t_traj: list[Float[Tensor, "nely nelx"]]  # length nloop+1
    records: list[IterationRecord]  # length nloop


def build_problem(
    nelx: int,
    nely: int,
    nStage: int,
    volfrac: float,
    Theta: float,
    Tcr: float,
    tfield: timefield.TimeField,
    rmin: float,
    lrmin: float,
    rmin_cond: float,
    *,
    Emin: float = 1e-9,
    Emax: float = 1.0,
    nu: float = 0.3,
    penal: float = 3.0,
    eta: float = 0.5,
    beta_d_max: float = 128.0,
    p: float = 25.0,
    q: float = 3.0,
    r: float = 0.05,
    rouf: float = 100.0,
    a0: float = 1.0,
    mma_c: float = 2500.0,
    move: float = 0.01,
    tmove: float = 0.01,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    batch_fem_solves: bool | None = None,
) -> Problem:
    """Build the fixed FEM/filter/geometry setup once, before the loop starts.

    `rmin`/`lrmin`/`rmin_cond` are filter radii (density filter, continuity filter,
    conductivity neighborhood) -- passed explicitly rather than hardcoded so callers
    (e.g. the E2E test) can match whatever grid they're running on, since the fixture's
    radii differ from the original full-scale script's.

    :param device: device every tensor field of the returned `Problem` lives on.
    :param dtype: floating dtype every real-valued tensor field is cast to; integer
        (index/mask) fields keep their own integer/bool dtype regardless.
    :param batch_fem_solves: batch whole_compliance's and every gravity stage's solve
        into one `FemSolve` call (`plans/torch_port_part2.md` Phase 3.3). Defaults to
        on at or below the production 180x60 mesh -- part 1 measured a 1.3-1.4x win
        there and a small loss at 360x120 -- but is a plain `bool` so a caller can
        override the default in either direction.
    """
    device = torch.device(device)
    if batch_fem_solves is None:
        batch_fem_solves = nelx * nely <= 180 * 60
    # A 1x1 mesh has neither extent nor neighbours, so two of the pieces built below
    # degenerate: the CORNER/OPPOSITE_CORNER time fields normalize by a zero max
    # distance, and the continuity filter divides by a zero neighbour count. This is the
    # first point where both exist, so the check belongs here rather than in either one.
    if nelx == 1 and nely == 1:
        raise ValueError(
            f"nelx and nely cannot both be 1, got nelx={nelx}, nely={nely}"
        )

    tfield = timefield.TimeField(tfield)
    KE = fem.plane_stress_KE(nu)
    edofMat = fem.element_dof_map(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Fixed cantilever load case, stated geometrically rather than as a linear-index
    # formula so it survives a change of node numbering: unit downward point load on the
    # bottom-right node, left edge clamped in both directions.
    nodes = fem.node_grid(nelx, nely)
    F = np.zeros(ndof)
    F[2 * nodes[-1, -1] + 1] = -1.0
    left_edge = nodes[:, 0]
    fixeddofs = np.stack([2 * left_edge, 2 * left_edge + 1], axis=-1).ravel()
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    H, Hs = filters.density_filter(nelx, nely, rmin)
    L = filters.continuity_filter(nelx, nely, lrmin)
    C = gravity.gravity_load_matrix(nelx, nely)
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)

    # Print-start element(s): the whole first mesh column for tfield != CORNER, the
    # single origin element for tfield == CORNER (constraints.start_point's own
    # docstring). Element `row*nelx` is grid position `(row, 0)` per conventions.md's
    # C-order element enumeration -- i.e. column 0, every row.
    Nei = (
        np.array([0])
        if tfield == timefield.TimeField.CORNER
        else np.arange(nely) * nelx
    )

    n = 2 * nelx * nely
    # MATLAB hardcodes `m = 1 + 1 + nely + 2*nStage + 1` -- only self-consistent when
    # tfield != CORNER (Nei has nely rows); computing from len(Nei) generalizes correctly.
    m = 1 + 1 + len(Nei) + 2 * nStage + 1

    # Batch every float-valued and every int-valued raw array into one boundary crossing
    # each, rather than a `to_tensor` call per field (plans/torch_port_review_followup.md
    # Phase 5). Keys match `Problem`'s field names so they splat straight in below.
    float_fields = torch_util.to_tensors(
        {"KE": KE, "F": F, "Hs": Hs, "w": w}, device, dtype
    )
    int_fields = torch_util.to_tensors(
        {"edofMat": edofMat, "freedofs": freedofs, "e1": e1, "e2": e2, "Nei": Nei},
        device,
        torch.int64,
    )
    return Problem(
        nelx=nelx,
        nely=nely,
        nStage=nStage,
        volfrac=volfrac,
        Theta=Theta,
        Tcr=Tcr,
        tfield=tfield,
        device=device,
        dtype=dtype,
        free_mask=torch_fem.free_mask(ndof, int_fields["freedofs"], device=device),
        ndof=ndof,
        H=torch_util.csr_to_tensor(H, device, dtype),
        L=torch_util.csr_to_tensor(L, device, dtype),
        C=torch_util.csr_to_tensor(C, device, dtype),
        Emin=Emin,
        Emax=Emax,
        penal=penal,
        eta=eta,
        beta_d_max=beta_d_max,
        p=p,
        q=q,
        r=r,
        rouf=rouf,
        a0=a0,
        mma_c=mma_c,
        move=move,
        tmove=tmove,
        m=m,
        n=n,
        batch_fem_solves=batch_fem_solves,
        **float_fields,
        **int_fields,
    )


def init_state(problem: Problem, beta_d: float) -> State:
    """Initial state: `x`/`t` are the raw seed (uniform `problem.volfrac`, time field per
    `problem.tfield`); the physics fields are derived from them exactly as in `step`.

    The MATLAB source instead assigns `xTilde = x` and `t = tPhys` unfiltered here. For
    the density half that is a numeric no-op -- `x` is uniform and the filter fixes
    constants -- but no `init_timefield` variant is constant, so leaving `tPhys`
    unfiltered made iteration 1 the one iteration whose forward map disagreed with the
    `tPhys = H @ t / Hs` that `step`'s chain rule differentiates. See PR #26.
    """
    nely, nelx = problem.nely, problem.nelx
    nel = nelx * nely
    device, dtype = problem.device, problem.dtype

    def filtered(field: Tensor) -> Tensor:
        return ((problem.H @ field.flatten()) / problem.Hs).reshape(nely, nelx)

    x = torch.full((nely, nelx), problem.volfrac, device=device, dtype=dtype)
    xTilde = filtered(x)
    xPhys = filters.heaviside_projection(xTilde, beta_d, problem.eta)

    # init_timefield is a NumPy builder (plans/torch_port_part2.md Phase 3.2 item 2);
    # converted once here, at the tensor boundary.
    t = torch_util.to_tensor(
        timefield.init_timefield(nelx, nely, problem.tfield), device, dtype
    )
    tPhys = filtered(t)

    # MATLAB's xold1=xold2=[x(:); zeros(nel,1)] -- provably unread (iterations 1 and 2
    # both take mmasub's `iteration < 2.5` reinit branch), reproduced anyway for fidelity.
    xold = torch.cat([x.flatten(), torch.zeros(nel, device=device, dtype=dtype)])

    return State(
        x=x,
        xTilde=xTilde,
        xPhys=xPhys,
        t=t,
        tPhys=tPhys,
        xold1=xold,
        xold2=xold.clone(),
        low=torch.zeros(problem.n, device=device, dtype=dtype),
        upp=torch.zeros(problem.n, device=device, dtype=dtype),
        loop=0,
        beta_t=10.0,
        beta_d=beta_d,
        factor=1.0,
        U=None,
    )


def _grad_row(
    output: Float[Tensor, ""], leaves: tuple[Tensor, Tensor]
) -> Float[Tensor, " n"]:
    """Sensitivity of one scalar output w.r.t. both leaves, flattened and concatenated
    into a single MMA row -- one `torch.autograd.grad` call
    (`plans/torch_port_part2.md` Phase 3.4). `allow_unused` covers rows that depend on
    only one leaf (e.g. a density-only constraint never touches `t`): the unused
    leaf's slice is exactly zero, which is what its hand-derived predecessor returned
    too.
    """
    grads = torch.autograd.grad(output, leaves, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(leaf)).flatten()
            for g, leaf in zip(grads, leaves)
        ]
    )


def _grad_rows_batched(
    outputs: Float[Tensor, " k"],
    xTilde: Float[Tensor, "nely nelx"],
    tPhys: Float[Tensor, "nely nelx"],
    H: Tensor,
    Hs: Float[Tensor, " nel"],
) -> Float[Tensor, "k n"]:
    """Sensitivities of `k` independent scalar outputs (e.g. one per print-start
    element, or one per stage) w.r.t. both raw leaves, as `(k, n)` -- one
    `torch.autograd.grad(..., is_grads_batched=True)` call with one-hot seeds instead
    of `k` separate calls (`plans/torch_port_part2.md` Phase 3.4's Jacobian-assembly
    requirement).

    Differentiates down to the *filtered* fields (`xTilde`, `tPhys` -- density's own
    filtered field, pre-Heaviside, since `xPhys = heaviside_projection(xTilde, ...)`
    is itself an ordinary pointwise op `is_grads_batched`'s vmap handles fine) rather
    than all the way to the raw `x`/`t` leaves, then finishes the last, linear step --
    the density/continuity filter `field = H @ raw / Hs` -- by hand as a plain sparse
    matmul: `is_grads_batched`'s vmap has no batching rule for the sparse CSR matmul's
    backward (`RuntimeError: expand is unsupported for SparseCsc tensors`, confirmed
    locally), so batching must stop one step short of it. This is not a reintroduction
    of a hand-derived *physics* sensitivity -- every nonlinear term (Heaviside,
    hotspot's pairwise sigmoids, SIMP) is still autograd's -- only the filter's own
    adjoint is applied explicitly, and because `H` is symmetric by construction
    (`filters.density_filter`'s weight depends only on distance), that adjoint is `H`
    itself: exactly `torch.autograd.grad` would have produced had vmap been able to
    reach the sparse op.
    """
    k = outputs.shape[0]
    seeds = torch.eye(k, dtype=outputs.dtype, device=outputs.device)
    d_xTilde, d_tPhys = torch.autograd.grad(
        outputs,
        (xTilde, tPhys),
        grad_outputs=seeds,
        is_grads_batched=True,
        retain_graph=True,
        allow_unused=True,
    )

    def _filter_adjoint(d_field: Tensor | None) -> Tensor:
        if d_field is None:
            return torch.zeros(
                k, xTilde.numel(), dtype=xTilde.dtype, device=xTilde.device
            )
        flat = d_field.reshape(k, -1)  # (k, nel)
        return (H @ (flat / Hs).T).T  # (k, nel)

    return torch.cat([_filter_adjoint(d_xTilde), _filter_adjoint(d_tPhys)], dim=1)


def step(problem: Problem, state: State) -> tuple[State, IterationRecord]:
    """Run one optimization iteration: build the objective + every constraint's value
    in the reference's exact row order, differentiate the whole graph by autograd
    (`plans/torch_port_part2.md` Phase 3.4 -- `x`/`t` are the autograd leaves, per
    Decision 4; the filter and Heaviside projection are ordinary forward operations,
    not accompanied by a hand-derived `dx` chain-rule factor), call `mma.mmasub`, and
    unpack the result into the next state.

    Recomputing `xTilde`/`xPhys`/`tPhys` from this iteration's own `x`/`t` (rather than
    reading `state.xPhys`/`state.tPhys`, which the hand-derived predecessor did) is a
    deliberate side effect of Decision 4, not a separate change: it makes the value
    autograd differentiates and the value MMA optimizes the same expression, and it is
    the invariant `_assert_state_fields_are_consistent` (test_optimize.py) pins at
    every iteration -- `state.xPhys`/`state.tPhys` always equal what filtering
    `state.x`/`state.t` at `state.beta_d` produces.

    All three periodic state updates (`beta_t += 5` at loop%30==0, `beta_d *= 2` at
    loop%50==0, the hotspot `factor` refresh at loop%25==0) happen at the tail, next to
    each other, and take effect starting the *next* iteration's `step` call rather than
    rescaling this iteration's own `fval`/`dfdx`/`xPhys` mid-loop -- a deliberate
    simplification, not a fidelity gap. None of the three trigger against the small
    E2E fixture (`nloop=3`) -- unexercised by that fixture, not unimplemented or
    worked around.
    """
    nely, nelx, nStage = problem.nely, problem.nelx, problem.nStage
    nel = nelx * nely
    device, dtype = problem.device, problem.dtype

    loop = state.loop + 1
    beta_t = state.beta_t
    beta_d = state.beta_d

    # -- Gradient region begins: x/t become autograd leaves. --
    x = state.x.clone().requires_grad_(True)
    t = state.t.clone().requires_grad_(True)
    leaves = (x, t)
    xTilde = ((problem.H @ x.flatten()) / problem.Hs).reshape(nely, nelx)
    xPhys = filters.heaviside_projection(xTilde, beta_d, problem.eta)
    tPhys = ((problem.H @ t.flatten()) / problem.Hs).reshape(nely, nelx)

    # -- Objective: whole-structure compliance + Theta-weighted per-stage gravity compliance --
    # `Problem.batch_fem_solves` (Phase 3.3, plans/torch_port_part2.md) puts
    # whole_compliance's solve and every gravity stage's solve into one FemSolve call;
    # off, they run sequentially as before.
    stage_times = [float(ti) for ti in np.linspace(0, 1, nStage + 1)[1:]]
    if problem.batch_fem_solves:
        c_t, stage_cs, U_new = compliance.batched_whole_and_gravity_compliance(
            xPhys,
            tPhys,
            problem.KE,
            problem.edofMat,
            problem.Emin,
            problem.Emax,
            problem.penal,
            problem.freedofs,
            problem.F,
            problem.ndof,
            problem.C,
            beta_t,
            stage_times,
            x0=state.U,
        )
    else:
        c_t, _ = compliance.whole_compliance(
            xPhys,
            problem.KE,
            problem.edofMat,
            problem.Emin,
            problem.Emax,
            problem.penal,
            problem.freedofs,
            problem.F,
            problem.ndof,
        )
        stage_cs = []
        for ti in stage_times:
            cg_t, _ = compliance.gravity_compliance(
                xPhys,
                tPhys,
                problem.KE,
                problem.edofMat,
                problem.Emin,
                problem.Emax,
                problem.penal,
                ti,
                problem.C,
                beta_t,
                problem.freedofs,
                problem.ndof,
            )
            stage_cs.append(cg_t)
        U_new = None

    obj_final_only = float(
        c_t.detach()
    )  # compliance of final structure only, saved for logging
    f0val_t = c_t
    for cg_t in stage_cs:
        f0val_t = f0val_t + problem.Theta * cg_t
    f0val = float(f0val_t.detach())
    df0dx = _grad_row(f0val_t, leaves)

    # -- Move-limit bounds on this iteration's raw MMA variables --
    xflat = state.x.flatten()
    tflat = state.t.flatten()
    xminx = torch.clamp(xflat - problem.move, min=0.0)
    xmaxx = torch.clamp(xflat + problem.move, max=1.0)
    xmint = torch.clamp(tflat - problem.tmove, min=0.0)
    xmaxt = torch.clamp(tflat + problem.tmove, max=1.0)
    xmin = torch.cat([xminx, xmint])
    xmax = torch.cat([xmaxx, xmaxt])
    xval = torch.cat([xflat, tflat])

    # -- Constraints, stacked in the reference loop's exact order --
    fval_parts: list[Tensor] = []
    dfdx_parts: list[Tensor] = []

    fv_vol_t = constraints.global_volume_fraction(xPhys, problem.volfrac)
    vol_diag = float(xPhys.detach().sum() / (nelx * nely))
    fval_parts.append(fv_vol_t[None])
    dfdx_parts.append(_grad_row(fv_vol_t, leaves)[None, :])

    fv_cont_t = constraints.time_field_continuity(tPhys, problem.L)
    fval_parts.append(fv_cont_t[None])
    dfdx_parts.append(_grad_row(fv_cont_t, leaves)[None, :])

    fv_start_t = constraints.start_point(tPhys, problem.Nei)
    fval_parts.append(fv_start_t)
    dfdx_parts.append(
        _grad_rows_batched(fv_start_t, xTilde, tPhys, problem.H, problem.Hs)
    )

    stage_upper_t = torch.stack(
        [
            constraints.stage_volume_bounds(
                xPhys, tPhys, float(t_stage), problem.volfrac, beta_t
            )
            for t_stage in stage_times
        ]
    )
    stage_lower_t = -stage_upper_t - 1.0e-5
    rows_stage_upper = _grad_rows_batched(
        stage_upper_t, xTilde, tPhys, problem.H, problem.Hs
    )
    rows_stage_lower = -rows_stage_upper
    for i in range(nStage):
        fval_parts.append(torch.stack([stage_upper_t[i], stage_lower_t[i]]))
        dfdx_parts.append(torch.stack([rows_stage_upper[i], rows_stage_lower[i]]))

    # Hotspot constraint, evaluated at this iteration's (possibly stale) `factor`.
    # `factor` is refreshed every 25 iterations from this same call's `numer`/`K_est`
    # (both independent of `factor`), but the refresh only takes effect starting next
    # iteration's `step` call -- `fv`/the hotspot row below are never rescaled
    # mid-iteration.
    numer_t, K_est_t = conductivity.hotspot_value(
        xPhys,
        tPhys,
        problem.e1,
        problem.e2,
        problem.w,
        problem.p,
        problem.q,
        problem.r,
        problem.rouf,
    )
    fv_hotspot_t = state.factor * numer_t / problem.Tcr - 1
    fval_parts.append(fv_hotspot_t[None])
    dfdx_parts.append(_grad_row(fv_hotspot_t, leaves)[None, :])

    # -- Periodic state updates, all deferred to take effect starting *next*
    # iteration's step() call, never rescaling this iteration's own fval/dfdx mid-loop. --
    numer = float(numer_t.detach())
    factor = state.factor
    if loop % 25 == 0:
        max_g = float(
            torch.max((1 - K_est_t.detach()) * xPhys.detach().flatten() ** problem.r)
        )
        factor = max_g / numer
    tru_max = factor * numer

    if loop % 30 == 0 and beta_t < 50:
        beta_t += 5
    if loop % 50 == 0 and beta_d <= problem.beta_d_max:
        beta_d *= 2
    if beta_d > problem.beta_d_max:
        beta_d = problem.beta_d_max

    fval = torch.cat(fval_parts)
    dfdx = torch.cat(dfdx_parts, dim=0)

    # -- Gradient region ends: mmasub is not part of the autograd graph. df0dx/fval/
    # dfdx are the last gradient-carrying values, detached here on their way in.
    # xval/xmin/xmax never required grad -- they're built from state.x/state.t, the
    # detached fields, not the x/t leaves above.
    mma_a = torch.zeros(problem.m, device=device, dtype=dtype)
    mma_c = torch.full((problem.m,), problem.mma_c, device=device, dtype=dtype)
    mma_d = torch.zeros(problem.m, device=device, dtype=dtype)
    xmma, ymma, zmma, lam, xsi, mma_eta, mu, zet, s, low, upp = mma.mmasub(
        problem.m,
        problem.n,
        loop,
        xval,
        xmin,
        xmax,
        state.xold1,
        state.xold2,
        f0val,
        df0dx.detach(),
        fval.detach(),
        dfdx.detach(),
        state.low,
        state.upp,
        problem.a0,
        mma_a,
        mma_c,
        mma_d,
    )

    x_new = xmma[:nel].reshape(nely, nelx)
    t_new = xmma[nel:].reshape(nely, nelx)

    xTilde_new = ((problem.H @ x_new.flatten()) / problem.Hs).reshape(nely, nelx)
    xPhys_new = filters.heaviside_projection(xTilde_new, beta_d, problem.eta)
    tPhys_new = ((problem.H @ t_new.flatten()) / problem.Hs).reshape(nely, nelx)

    new_state = State(
        x=x_new,
        xTilde=xTilde_new,
        xPhys=xPhys_new,
        t=t_new,
        tPhys=tPhys_new,
        xold1=xval,
        xold2=state.xold1,
        low=low,
        upp=upp,
        loop=loop,
        beta_t=beta_t,
        beta_d=beta_d,
        factor=factor,
        # Required: femsolve() asserts its x0 argument arrives already detached (the
        # next iteration passes state.U as x0). Undetached, this leaked ~180 MB/step
        # of multigrid hierarchy across iterations -- see commit 855eb76.
        U=U_new.detach() if U_new is not None else None,
    )
    record = IterationRecord(
        obj=obj_final_only,
        vol=vol_diag,
        tru_max=tru_max,
        f0val=float(f0val),
        df0dx=torch_util.to_numpy(df0dx),
        xmma=torch_util.to_numpy(xmma),
        low=torch_util.to_numpy(low),
        upp=torch_util.to_numpy(upp),
        lam=torch_util.to_numpy(lam),
        fval=torch_util.to_numpy(fval),
        dfdx=torch_util.to_numpy(dfdx),
    )
    return new_state, record


def run(
    nelx: int,
    nely: int,
    nloop: int,
    nStage: int,
    volfrac: float,
    Theta: float,
    Tcr: float,
    tfield: timefield.TimeField,
    rmin: float,
    lrmin: float,
    rmin_cond: float,
    *,
    beta_d: float = 1.0,
    **problem_kwargs,
) -> RunResult:
    """Build the fixed setup and run `nloop` optimization iterations from the standard
    initialization (uniform density at `volfrac`, `timefield.init_timefield` for the
    print-time field). `problem_kwargs` forwards to `build_problem` (SIMP/Heaviside/
    hotspot/MMA constants) for callers that need non-default values.
    """
    problem = build_problem(
        nelx,
        nely,
        nStage,
        volfrac,
        Theta,
        Tcr,
        tfield,
        rmin,
        lrmin,
        rmin_cond,
        **problem_kwargs,
    )
    return run_from_state(problem, init_state(problem, beta_d), nloop)


def run_from_state(problem: Problem, state: State, nloop: int) -> RunResult:
    """Run `nloop` iterations from an arbitrary starting state, collecting the
    trajectory. Split out of `run` so that where the iteration starts is a caller's
    choice -- warm-starting from a previous run's final state, or entering at a
    trajectory recorded elsewhere.
    """
    xPhys_traj = [state.xPhys.clone()]
    tPhys_traj = [state.tPhys.clone()]
    x_traj = [state.x.clone()]
    t_traj = [state.t.clone()]
    records: list[IterationRecord] = []

    for _ in range(nloop):
        state, record = step(problem, state)
        xPhys_traj.append(state.xPhys.clone())
        tPhys_traj.append(state.tPhys.clone())
        x_traj.append(state.x.clone())
        t_traj.append(state.t.clone())
        records.append(record)

    return RunResult(
        state=state,
        xPhys_traj=xPhys_traj,
        tPhys_traj=tPhys_traj,
        x_traj=x_traj,
        t_traj=t_traj,
        records=records,
    )
