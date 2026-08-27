"""Tests for sttopt.torch_solve.FemSolve, the FEM solve as an autograd `Function`
(`plans/torch_port_part2.md` Phase 3.3): the adjoint, the self-adjoint warm start,
hierarchy reuse, batching, warm starting, and non-convergence propagation.
"""

import numpy as np
import pytest

import sttopt.fem as fem
from conftest import assert_close, point_load_problem

torch = pytest.importorskip("torch")

import sttopt.compliance as compliance  # noqa: E402
import sttopt.torch_fem as torch_fem  # noqa: E402
import sttopt.torch_solve as torch_solve  # noqa: E402
from benchmarks import calibrate_cg_rtol as calib  # noqa: E402
from test_torch_fem import _binary_design  # noqa: E402

EMIN, EMAX, PENAL = 1e-9, 1.0, 3
NU = 0.3


def _setup(nelx, nely):
    F_np, freedofs_np, ndof = point_load_problem(nelx, nely)
    KE = torch.tensor(fem.plane_stress_KE(NU), dtype=torch.float64)
    edofMat = torch.tensor(fem.element_dof_map(nelx, nely), dtype=torch.int64)
    mask = torch_fem.free_mask(ndof, torch.tensor(freedofs_np, dtype=torch.int64))
    F = torch.tensor(F_np, dtype=torch.float64)
    return F, ndof, KE, edofMat, mask


# --- Test 1: gradcheck ------------------------------------------------------------


def test_gradcheck_at_moderate_density():
    """`torch.autograd.gradcheck` at a small mesh, moderate density, cold start.

    Per the plan: this envelope (moderate density, stock gradcheck tolerances, the
    solver's production `rtol=1e-8`) is where gradcheck is meaningful; a near-binary
    field is not (test 2 covers that instead -- see its docstring).
    """
    nelx, nely = 4, 3
    F, ndof, KE, edofMat, mask = _setup(nelx, nely)
    rng = np.random.default_rng(0)
    nel = nelx * nely
    density = torch.tensor(
        rng.uniform(0.2, 0.8, nel), dtype=torch.float64, requires_grad=True
    )
    F = F.clone().requires_grad_(True)

    def check_fn(density, F):
        return torch_solve.femsolve(
            density, F, edofMat, KE, mask, nelx, nely, rtol=1e-8, x0=None
        )

    assert torch.autograd.gradcheck(
        check_fn, (density, F), eps=1e-6, atol=1e-5, rtol=1e-3
    )


# --- Test 2: adjoint vs the hand-derived dcx/dct, at near-binary snapshots --------

NEAR_BINARY_MESH = "90x30"  # CPU-feasible; the fixture's smaller near-binary design.


def _near_binary_snapshot():
    nelx, nely = (int(v) for v in NEAR_BINARY_MESH.split("x"))
    with np.load(calib.FIXTURES) as data:
        x = data[f"x_{NEAR_BINARY_MESH}_it0800"]
        t = data[f"t_{NEAR_BINARY_MESH}_it0800"]
    return calib.mesh_setup(nelx, nely), x, t


def _autograd_whole_compliance(xPhys, KE, edofMat, mask, nelx, nely, F):
    """`compliance.whole_compliance`'s forward, kept differentiable end to end.

    `compliance.whole_compliance` itself casts `c` to a Python `float`, which detaches
    it from autograd on purpose (its sensitivity is the hand-derived `dcx`, not an
    autograd gradient -- Phase 3.4 is what changes that). This is the independent
    autograd path test 2 compares that hand-derived formula against.
    """
    density = torch_fem.simp_density(xPhys, EMIN, EMAX, PENAL)
    U = torch_solve.femsolve(
        density.flatten(), F, edofMat, KE, mask, nelx, nely, rtol=1e-8
    )
    Ue = U[edofMat]
    ce = torch.sum((Ue @ KE) * Ue, dim=1)
    return torch.sum(density.flatten() * ce)


def _autograd_gravity_compliance(
    xPhys, tPhys, KE, edofMat, mask, nelx, nely, ti, C, beta_t, ndof
):
    """`compliance.gravity_compliance`'s forward, kept differentiable end to end."""
    t_mask = compliance.time_mask(tPhys, ti, beta_t)
    xtJoint = xPhys * t_mask
    f = -(C @ xtJoint.flatten())
    F = torch.zeros(ndof, dtype=xPhys.dtype, device=xPhys.device)
    F[1::2] = f

    density = torch_fem.simp_density(xtJoint, EMIN, EMAX, PENAL)
    U = torch_solve.femsolve(
        density.flatten(), F, edofMat, KE, mask, nelx, nely, rtol=1e-8
    )
    Ue = U[edofMat]
    ce = torch.sum((Ue @ KE) * Ue, dim=1)
    return torch.sum(density.flatten() * ce)


def test_adjoint_matches_hand_derived_whole_compliance_near_binary():
    setup, x, t = _near_binary_snapshot()
    xPhys = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    mask = setup["mask"]

    c_ref, dcx_ref = compliance.whole_compliance(
        xPhys.detach(),
        setup["KE_t"],
        setup["edofMat_t"],
        EMIN,
        EMAX,
        PENAL,
        setup["freedofs_t"],
        setup["F_t"],
        setup["ndof"],
    )
    c = _autograd_whole_compliance(
        xPhys,
        setup["KE_t"],
        setup["edofMat_t"],
        mask,
        setup["nelx"],
        setup["nely"],
        setup["F_t"],
    )
    c.backward()

    assert_close(c.detach().numpy(), c_ref, tier="solved")
    rel_active, abs_over_peak = calib.elementwise_errors(
        xPhys.grad.numpy().ravel(), dcx_ref.numpy().ravel()
    )
    assert rel_active < calib.SENSITIVITY_TOL
    assert abs_over_peak < calib.SENSITIVITY_TOL


def test_adjoint_matches_hand_derived_gravity_compliance_near_binary():
    setup, x, t = _near_binary_snapshot()
    xPhys = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    tPhys = torch.tensor(t, dtype=torch.float64, requires_grad=True)
    mask = setup["mask"]
    ti, beta_t = 0.5, calib.BETA_T

    c_ref, dcx_ref, dct_ref = compliance.gravity_compliance(
        xPhys.detach(),
        tPhys.detach(),
        setup["KE_t"],
        setup["edofMat_t"],
        EMIN,
        EMAX,
        PENAL,
        ti,
        setup["C_t"],
        beta_t,
        setup["freedofs_t"],
        setup["ndof"],
    )
    c = _autograd_gravity_compliance(
        xPhys,
        tPhys,
        setup["KE_t"],
        setup["edofMat_t"],
        mask,
        setup["nelx"],
        setup["nely"],
        ti,
        setup["C_t"],
        beta_t,
        setup["ndof"],
    )
    c.backward()

    assert_close(c.detach().numpy(), c_ref, tier="solved")
    for grad, ref, name in (
        (xPhys.grad, dcx_ref, "dcx"),
        (tPhys.grad, dct_ref, "dct"),
    ):
        rel_active, abs_over_peak = calib.elementwise_errors(
            grad.numpy().ravel(), ref.numpy().ravel()
        )
        assert rel_active < calib.SENSITIVITY_TOL, name
        assert abs_over_peak < calib.SENSITIVITY_TOL, name


# --- Test 3: the self-adjoint shortcut (lambda == 2U, zero adjoint iterations) ----


@pytest.mark.parametrize("load_depends_on_density", [False, True])
def test_self_adjoint_shortcut_gives_zero_adjoint_iterations(load_depends_on_density):
    """`lambda == 2U` (equivalently `dL/dU == 2F`) for the compliance scalar, whether
    `F` is a fixed load (the `whole_compliance` case) or itself a function of the
    differentiated density (the `gravity_compliance` case) -- the self-adjoint
    property is about `dL/dU`, not about where `F` came from.
    """
    nelx, nely = 6, 4
    F_fixed, _, KE, edofMat, mask = _setup(nelx, nely)
    if load_depends_on_density:
        rng = np.random.default_rng(2)
        weights = torch.tensor(rng.standard_normal(mask.shape[-1]), dtype=torch.float64)
        raw_density = torch.tensor(
            rng.uniform(0.1, 0.9, nelx * nely), dtype=torch.float64, requires_grad=True
        )
        F = weights * raw_density.sum()  # some differentiable function of density
        density = raw_density
    else:
        F = F_fixed
        density = torch.tensor(
            np.random.default_rng(3).uniform(0.1, 0.9, nelx * nely),
            dtype=torch.float64,
            requires_grad=True,
        )

    info: dict = {}
    U = torch_solve.femsolve(
        density, F, edofMat, KE, mask, nelx, nely, rtol=1e-12, info=info
    )
    Ue = U[edofMat]
    ce = torch.sum((Ue @ KE) * Ue, dim=1)
    L = torch.sum(density * ce)

    # lambda == 2U (equivalently dL/dU == 2F) is a statement about the *projected*
    # system FemSolve actually solves: at a fixed dof, U is pinned to 0 by
    # construction and dL/dU there is a reaction force, not the applied load, so it
    # need not match F. Restrict the comparison to the free-dof subspace, matching
    # what `FemSolve.backward` itself projects onto before solving.
    g = torch.autograd.grad(L, U, retain_graph=True)[0]
    assert_close(
        torch_fem.project(g, mask).detach().numpy(),
        torch_fem.project(2 * F, mask).detach().numpy(),
        tier="algebraic",
    )

    L.backward()
    assert info["backward_n_iter"] == 0


# --- Test 4: batched vs sequential, forward and (one-hot-seeded) backward --------


def _mg_kw(nelx, nely):
    """Force real coarsening on a mesh too small for the production default."""
    return {"max_coarse_elements": 24} if nelx * nely <= 300 else {}


def test_batched_matches_sequential_forward_and_backward():
    """The batched `(9, ndof)` path against nine sequential single solves, forward and
    backward. The backward half is what the pcg zero-`b` prerequisite fix unblocks:
    a one-hot `grad_output` (as gradcheck and a `is_grads_batched=True` Jacobian
    extraction would use) leaves every other batch member's adjoint right-hand side
    exactly zero.
    """
    nelx, nely = 18, 12
    n_batch = 9
    F, ndof, KE, edofMat, mask = _setup(nelx, nely)
    rng = np.random.default_rng(4)
    density_batch = torch.tensor(
        rng.uniform(0.05, 1.0, (n_batch, nelx * nely)),
        dtype=torch.float64,
        requires_grad=True,
    )
    F_batch = F.unsqueeze(0).expand(n_batch, -1).clone().requires_grad_(True)

    U_batch = torch_solve.femsolve(
        density_batch,
        F_batch,
        edofMat,
        KE,
        mask,
        nelx,
        nely,
        rtol=1e-11,
        **_mg_kw(nelx, nely),
    )
    assert U_batch.shape == (n_batch, ndof)

    U_seq, density_seq, F_seq = [], [], []
    for s in range(n_batch):
        d_s = density_batch[s].detach().clone().requires_grad_(True)
        f_s = F_batch[s].detach().clone().requires_grad_(True)
        U_s = torch_solve.femsolve(
            d_s, f_s, edofMat, KE, mask, nelx, nely, rtol=1e-11, **_mg_kw(nelx, nely)
        )
        assert_close(
            U_batch[s].detach().numpy(),
            U_s.detach().numpy(),
            tier="solved",
            atol=1e-6 * np.abs(U_s.detach().numpy()).max(),
        )
        U_seq.append(U_s)
        density_seq.append(d_s)
        F_seq.append(f_s)

    # One-hot backward per batch member, batched vs sequential.
    for s in range(n_batch):
        seed_batch = torch.zeros_like(U_batch)
        seed_batch[s, 0] = 1.0
        grad_density_batch, grad_F_batch = torch.autograd.grad(
            U_batch,
            (density_batch, F_batch),
            grad_outputs=seed_batch,
            retain_graph=True,
        )
        assert not torch.isnan(grad_density_batch).any()
        assert not torch.isnan(grad_F_batch).any()

        seed_single = torch.zeros_like(U_seq[s])
        seed_single[0] = 1.0
        grad_density_s, grad_F_s = torch.autograd.grad(
            U_seq[s],
            (density_seq[s], F_seq[s]),
            grad_outputs=seed_single,
            retain_graph=True,
        )
        assert_close(
            grad_density_batch[s].numpy(), grad_density_s.numpy(), tier="solved"
        )
        assert_close(grad_F_batch[s].numpy(), grad_F_s.numpy(), tier="solved")
        # Every other batch member's own gradient must be exactly zero, not nan.
        others = [i for i in range(n_batch) if i != s]
        assert torch.all(grad_density_batch[others] == 0.0)
        assert torch.all(grad_F_batch[others] == 0.0)


# --- Test 5: warm vs cold ----------------------------------------------------------


def test_warm_start_matches_cold_and_uses_fewer_iterations():
    """Warm-started from a real consecutive snapshot pair (loop 799's solution warm-
    starts loop 800's), agreeing with a cold start to `solved` tier and converging in
    fewer iterations.
    """
    nelx, nely = (int(v) for v in NEAR_BINARY_MESH.split("x"))
    setup = calib.mesh_setup(nelx, nely)
    with np.load(calib.FIXTURES) as data:
        x799 = data[f"x_{NEAR_BINARY_MESH}_it0799"]
        x800 = data[f"x_{NEAR_BINARY_MESH}_it0800"]

    density799 = torch_fem.simp_density(
        torch.tensor(x799.ravel(), dtype=torch.float64), EMIN, EMAX, PENAL
    )
    density800 = torch_fem.simp_density(
        torch.tensor(x800.ravel(), dtype=torch.float64), EMIN, EMAX, PENAL
    )
    edofMat, KE, mask, F = (
        setup["edofMat_t"],
        setup["KE_t"],
        setup["mask"],
        setup["F_t"],
    )

    info_799 = {}
    U799 = torch_solve.femsolve(
        density799, F, edofMat, KE, mask, nelx, nely, rtol=1e-11, info=info_799
    )

    info_cold = {}
    U_cold = torch_solve.femsolve(
        density800, F, edofMat, KE, mask, nelx, nely, rtol=1e-11, info=info_cold
    )
    info_warm = {}
    U_warm = torch_solve.femsolve(
        density800,
        F,
        edofMat,
        KE,
        mask,
        nelx,
        nely,
        rtol=1e-11,
        x0=U799.detach(),
        info=info_warm,
    )

    assert info_warm["forward_n_iter"] < info_cold["forward_n_iter"]
    assert_close(
        U_warm.detach().numpy(),
        U_cold.detach().numpy(),
        tier="solved",
        atol=1e-6 * np.abs(U_cold.detach().numpy()).max(),
    )


# --- Test 6: non-convergence propagates through the autograd boundary ------------


def test_forward_nonconvergence_raises():
    """A hard 0/1-contrast design and a small `max_coarse_elements`, so the solve is a
    real (imperfect) V-cycle rather than the single-level exact dense-Cholesky solve a
    mesh this small would otherwise get -- an exact preconditioner can't fail to
    converge, so it wouldn't exercise this path at all.
    """
    nelx, nely = 18, 12
    F, ndof, KE, edofMat, mask = _setup(nelx, nely)
    density = torch_fem.simp_density(
        torch.tensor(_binary_design(nelx, nely), dtype=torch.float64),
        EMIN,
        EMAX,
        PENAL,
    )
    with pytest.raises(torch_fem.CGConvergenceError):
        torch_solve.femsolve(
            density,
            F,
            edofMat,
            KE,
            mask,
            nelx,
            nely,
            rtol=1e-14,
            max_iter=2,
            max_coarse_elements=24,
        )


def test_backward_nonconvergence_raises():
    """A forward solve that converges trivially (`max_iter=0`, warm-started from the
    true `U`) followed by an adjoint solve that cannot (same `max_iter=0`, but its own
    warm start `alpha*U` is not already the answer, since `g` is not parallel to `F`)
    still raises -- the exception is not swallowed by the autograd boundary.
    """
    nelx, nely = 12, 8
    F, ndof, KE, edofMat, mask = _setup(nelx, nely)
    rng = np.random.default_rng(5)
    density = torch.tensor(rng.uniform(0.05, 1.0, nelx * nely), dtype=torch.float64)
    U_exact = torch_solve.femsolve(
        density, F, edofMat, KE, mask, nelx, nely, rtol=1e-13, max_iter=500
    )

    density.requires_grad_(True)
    U = torch_solve.femsolve(
        density, F, edofMat, KE, mask, nelx, nely, rtol=1e-8, max_iter=0, x0=U_exact
    )
    g = torch.tensor(rng.standard_normal(ndof), dtype=torch.float64)  # not ~ F
    with pytest.raises(torch_fem.CGConvergenceError):
        U.backward(g)


# --- Device: CPU and GPU agree on both the forward and the adjoint ---------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)


@requires_cuda
def test_cpu_and_gpu_agree_on_forward_and_backward():
    nelx, nely = 18, 12
    device_cpu, device_gpu = "cpu", "cuda"
    results = {}
    for device in (device_cpu, device_gpu):
        F_np, freedofs_np, ndof = point_load_problem(nelx, nely)
        t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
        KE = t(fem.plane_stress_KE(NU), torch.float64)
        edofMat = t(fem.element_dof_map(nelx, nely), torch.int64)
        mask = torch_fem.free_mask(ndof, t(freedofs_np, torch.int64), device)
        F = t(F_np, torch.float64)
        density = t(_binary_design(nelx, nely), torch.float64)
        density = torch_fem.simp_density(density, EMIN, EMAX, PENAL)
        density.requires_grad_(True)

        info: dict = {}
        U = torch_solve.femsolve(
            density,
            F,
            edofMat,
            KE,
            mask,
            nelx,
            nely,
            rtol=1e-10,
            max_coarse_elements=24,
            info=info,
        )
        L = torch.sum(U**2)
        L.backward()
        results[device] = (
            U.detach().cpu().numpy(),
            density.grad.cpu().numpy(),
            info["forward_n_iter"],
            info["backward_n_iter"],
        )

    U_cpu, grad_cpu, nf_cpu, nb_cpu = results[device_cpu]
    U_gpu, grad_gpu, nf_gpu, nb_gpu = results[device_gpu]
    assert nf_cpu == nf_gpu
    assert nb_cpu == nb_gpu
    assert_close(U_gpu, U_cpu, tier="solved", atol=1e-6 * np.abs(U_cpu).max())
    assert_close(grad_gpu, grad_cpu, tier="solved", atol=1e-6 * np.abs(grad_cpu).max())
