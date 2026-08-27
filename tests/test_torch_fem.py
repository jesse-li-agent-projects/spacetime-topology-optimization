"""Tests for sttopt.torch_fem: the matrix-free matvec/diagonal against sttopt.fem's
assembled-matrix path, and the Jacobi-PCG solver against sttopt.fem.solve_fe. All on
CPU and in float64 -- see plans/torch_port.md's Phase 1.
"""

import numpy as np
import pytest

import sttopt.fem as fem
from conftest import assert_close, point_load_problem

# torch is an optional dependency (see pyproject.toml), so skip rather than fail
# collection where it isn't installed.
torch = pytest.importorskip("torch")

import sttopt.torch_fem as torch_fem  # noqa: E402
import sttopt.torch_mg as torch_mg  # noqa: E402

# The tolerance study whose conclusions the sensitivity tests below lock in; imported
# rather than copied so the test and the study can never disagree about how the error
# is measured.
from benchmarks import calibrate_cg_rtol as calib  # noqa: E402

EMIN, EMAX, PENAL = 1e-9, 1.0, 3
NU = 0.3
MESH_SIZES = [(4, 3), (12, 8), (30, 10)]


def _setup(nelx, nely):
    """Common torch-side setup: KE, edofMat, ndof, mask, plus the numpy problem."""
    F_np, freedofs_np, ndof = point_load_problem(nelx, nely)
    KE_np = fem.plane_stress_KE(NU)
    edofMat_np = fem.element_dof_map(nelx, nely)
    KE = torch.tensor(KE_np, dtype=torch.float64)
    edofMat = torch.tensor(edofMat_np, dtype=torch.int64)
    mask = torch_fem.free_mask(ndof, torch.tensor(freedofs_np, dtype=torch.int64))
    return F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask


def _density_fields(nelx, nely, rng):
    """A handful of density fields covering uniform, random, and near-zero elements."""
    nel = nelx * nely
    uniform = np.full(nel, 0.4)
    random = rng.uniform(0.0, 1.0, nel)
    near_void = random.copy()
    near_void[: nel // 3] = 1e-6
    return {"uniform": uniform, "random": random, "near_void": near_void}


@pytest.mark.parametrize("nelx, nely", MESH_SIZES)
def test_matvec_vs_assembled(nelx, nely):
    rng = np.random.default_rng(0)
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    for name, xPhys in _density_fields(nelx, nely, rng).items():
        K = fem.assemble_stiffness(KE_np, xPhys, EMIN, EMAX, PENAL, edofMat_np, ndof)
        v = rng.standard_normal(ndof)
        expected = K @ v
        actual = torch_fem.matvec(
            torch.tensor(v, dtype=torch.float64),
            torch_fem.simp_density(
                torch.tensor(xPhys, dtype=torch.float64), EMIN, EMAX, PENAL
            ),
            edofMat,
            KE,
            ndof,
        )
        assert_close(actual.numpy(), expected, tier="algebraic")


@pytest.mark.parametrize("nelx, nely", MESH_SIZES)
def test_diagonal_vs_assembled(nelx, nely):
    rng = np.random.default_rng(1)
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    for name, xPhys in _density_fields(nelx, nely, rng).items():
        K = fem.assemble_stiffness(KE_np, xPhys, EMIN, EMAX, PENAL, edofMat_np, ndof)
        actual = torch_fem.matvec_diagonal(
            torch_fem.simp_density(
                torch.tensor(xPhys, dtype=torch.float64), EMIN, EMAX, PENAL
            ),
            edofMat,
            KE,
            ndof,
        )
        assert_close(actual.numpy(), np.asarray(K.diagonal()), tier="algebraic")


@pytest.mark.parametrize("nelx, nely", [(4, 3), (12, 8)])
def test_solve_vs_spsolve(nelx, nely):
    rng = np.random.default_rng(2)
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    F = torch.tensor(F_np, dtype=torch.float64)
    volfrac_xPhys = np.full(nelx * nely, 0.4)
    random_xPhys = rng.uniform(0.05, 1.0, nelx * nely)
    for xPhys in (volfrac_xPhys, random_xPhys):
        K = fem.assemble_stiffness(KE_np, xPhys, EMIN, EMAX, PENAL, edofMat_np, ndof)
        expected = fem.solve_fe(K, F_np, freedofs_np)
        actual, n_iter = torch_fem.solve(
            F,
            torch.tensor(xPhys, dtype=torch.float64),
            edofMat,
            KE,
            EMIN,
            EMAX,
            PENAL,
            mask,
            rtol=1e-10,
        )
        assert n_iter > 0
        assert_close(actual.numpy(), expected, tier="solved")


def test_boundary_conditions_zero_and_garbage_invariant():
    nelx, nely = 6, 4
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    xPhys = torch.full((nelx * nely,), 0.4, dtype=torch.float64)
    F = torch.tensor(F_np, dtype=torch.float64)

    U, _ = torch_fem.solve(F, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10)
    fixed = ~mask
    assert torch.all(U[fixed] == 0.0)

    F_garbage = F.clone()
    rng = np.random.default_rng(3)
    F_garbage[fixed] = torch.tensor(
        rng.standard_normal(int(fixed.sum())), dtype=torch.float64
    )
    U_garbage, _ = torch_fem.solve(
        F_garbage, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10
    )
    assert torch.all(U_garbage[fixed] == 0.0)
    assert_close(U_garbage.numpy(), U.numpy(), tier="algebraic")


def test_nonconvergence_raises():
    nelx, nely = 12, 8
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    xPhys = torch.full((nelx * nely,), 0.4, dtype=torch.float64)
    F = torch.tensor(F_np, dtype=torch.float64)
    with pytest.raises(torch_fem.CGConvergenceError) as excinfo:
        torch_fem.solve(
            F, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-14, max_iter=2
        )
    assert excinfo.value.n_iter == 2
    assert excinfo.value.rtol == 1e-14


def test_dtype_float64_end_to_end():
    nelx, nely = 6, 4
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    xPhys = torch.full((nelx * nely,), 0.4, dtype=torch.float64)
    F = torch.tensor(F_np, dtype=torch.float64)
    U, _ = torch_fem.solve(F, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10)
    assert U.dtype == torch.float64
    diag = torch_fem.jacobi_preconditioner_diag(
        torch_fem.simp_density(xPhys, EMIN, EMAX, PENAL), edofMat, KE, ndof, mask
    )
    assert diag.dtype == torch.float64
    Kv = torch_fem.matvec(
        U, torch_fem.simp_density(xPhys, EMIN, EMAX, PENAL), edofMat, KE, ndof
    )
    assert Kv.dtype == torch.float64


def test_batched_solve_matches_sequential():
    """Batching over both the vector (F) and the density field (nStage-style) must
    give the same answer as looping the sequential solve, per member.
    """
    nelx, nely = 8, 5
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    rng = np.random.default_rng(4)
    F = torch.tensor(F_np, dtype=torch.float64)
    n_stage = 4
    xPhys_batch = torch.tensor(
        rng.uniform(0.05, 1.0, (n_stage, nelx * nely)), dtype=torch.float64
    )

    sequential = []
    for s in range(n_stage):
        U_s, _ = torch_fem.solve(
            F, xPhys_batch[s], edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10
        )
        sequential.append(U_s)
    sequential = torch.stack(sequential)

    F_batch = F.unsqueeze(0).expand(n_stage, -1)
    batched, n_iter = torch_fem.solve(
        F_batch, xPhys_batch, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10
    )
    assert batched.shape == (n_stage, ndof)
    for s in range(n_stage):
        assert_close(batched[s].numpy(), sequential[s].numpy(), tier="solved")


def test_warm_start_converges_in_fewer_iterations():
    nelx, nely = 12, 8
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    xPhys = torch.full((nelx * nely,), 0.4, dtype=torch.float64)
    F = torch.tensor(F_np, dtype=torch.float64)

    U_cold, n_iter_cold = torch_fem.solve(
        F, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10
    )

    rng = np.random.default_rng(5)
    perturbation = torch.tensor(rng.standard_normal(ndof) * 1e-4, dtype=torch.float64)
    x0 = torch_fem.project(U_cold + perturbation, mask)
    U_warm, n_iter_warm = torch_fem.solve(
        F, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10, x0=x0
    )

    assert n_iter_warm < n_iter_cold
    assert_close(U_warm.numpy(), U_cold.numpy(), tier="solved")


def test_warm_start_from_exact_solution_is_returned_unchanged():
    """An already-converged warm start must short-circuit rather than take a step: at a
    zero residual the first `alpha` is a 0/0 nan, which would corrupt a correct answer.
    """
    nelx, nely = 12, 8
    F_np, freedofs_np, ndof, KE_np, edofMat_np, KE, edofMat, mask = _setup(nelx, nely)
    xPhys = torch.full((nelx * nely,), 0.4, dtype=torch.float64)
    F = torch.tensor(F_np, dtype=torch.float64)

    U, _ = torch_fem.solve(F, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10)
    U_again, n_iter = torch_fem.solve(
        F, xPhys, edofMat, KE, EMIN, EMAX, PENAL, mask, rtol=1e-10, x0=U
    )

    assert n_iter == 0
    assert torch.isfinite(U_again).all()
    assert torch.equal(U_again, U)


# --- Multigrid (sttopt.torch_mg) -------------------------------------------------
# Test meshes are deliberately small, so `max_coarse_elements` is forced down; the
# module default is sized for the production meshes and would leave these solved
# directly on level 0, exercising no V-cycle at all.
MG_KW = {"max_coarse_elements": 24}
MG_MESH_SIZES = [(12, 8), (18, 12), (24, 16)]


def _binary_design(nelx, nely):
    """Connected solid/void cantilever, exact 0.0/1.0 -- the full ~1e9 SIMP contrast.

    Chords along the top, bottom and clamped edge plus a diagonal brace, so the loaded
    corner has a real load path. A thresholded random field would instead leave solid
    islands floating on `Emin`, and the resulting system is so ill-conditioned that even
    `spsolve` cannot reach a 1e-4 relative residual -- an unsolvable problem rather than
    a hard one, and no test of the preconditioner.
    """
    rows, cols = np.mgrid[0:nely, 0:nelx]
    x = np.zeros((nely, nelx))
    x[np.abs(rows / max(nely - 1, 1) - cols / max(nelx - 1, 1)) < 0.15] = 1.0
    x[:2, :] = 1.0
    x[-2:, :] = 1.0
    x[:, :2] = 1.0
    return x.ravel()


def _mg_density_fields(nelx, nely, rng):
    return {
        "uniform": np.full(nelx * nely, 0.4),
        "random": rng.uniform(0.05, 1.0, nelx * nely),
        "binary": _binary_design(nelx, nely),
    }


def _mg_solve(nelx, nely, xPhys, **kw):
    _, _, _, _, _, KE, edofMat, mask = _setup(nelx, nely)
    F_np, _, _ = point_load_problem(nelx, nely)
    return torch_mg.solve(
        torch.tensor(F_np, dtype=torch.float64),
        torch.tensor(xPhys, dtype=torch.float64),
        edofMat,
        KE,
        EMIN,
        EMAX,
        PENAL,
        mask,
        nelx,
        nely,
        **{**MG_KW, **kw},
    )


def _hierarchy(nelx, nely, xPhys, **kw):
    _, _, _, _, _, KE, edofMat, mask = _setup(nelx, nely)
    density = torch_fem.simp_density(
        torch.tensor(xPhys, dtype=torch.float64), EMIN, EMAX, PENAL
    )
    return torch_mg.build_hierarchy(
        density, edofMat, KE, mask, nelx, nely, **{**MG_KW, **kw}
    )


@pytest.mark.parametrize("kx, ky", [(1, 1), (2, 2), (3, 3), (3, 1), (2, 3)])
def test_restriction_is_the_exact_adjoint_of_prolongation(kx, ky):
    """`<P c, f> == <c, R f>`. If this fails the V-cycle is not symmetric, and CG's
    convergence theory stops applying to it without any visible symptom.
    """
    nCx, nCy = 3, 2
    rng = np.random.default_rng(0)
    c = torch.tensor(rng.standard_normal(2 * (nCx + 1) * (nCy + 1)))
    f = torch.tensor(rng.standard_normal(2 * (nCx * kx + 1) * (nCy * ky + 1)))
    Pc = torch_mg._on_node_grid(c, nCx + 1, nCy + 1, kx, ky, torch_mg._interp_axis)
    Rf = torch_mg._on_node_grid(
        f, nCx * kx + 1, nCy * ky + 1, kx, ky, torch_mg._restrict_axis
    )
    assert Pc.shape == f.shape
    assert_close((Pc * f).sum().item(), (c * Rf).sum().item(), tier="algebraic")


@pytest.mark.parametrize("nelx, nely", [(12, 8), (18, 12)])
def test_coarse_operator_is_exact_galerkin(nelx, nely):
    """Every coarse level must equal `R A P` of the level above, to machine precision.

    This is the load-bearing claim of the element-wise coarsening: that it really is
    Galerkin, not a re-discretization that resembles it.
    """
    rng = np.random.default_rng(1)
    levels = _hierarchy(nelx, nely, _binary_design(nelx, nely), max_coarse_elements=4)
    assert len(levels) >= 3
    for fine, coarse in zip(levels, levels[1:]):
        v = torch_fem.project(
            torch.tensor(rng.standard_normal(coarse.ndof)), coarse.mask
        )
        Pv = torch_fem.project(
            torch_mg._on_node_grid(
                v,
                coarse.nelx + 1,
                coarse.nely + 1,
                fine.kx,
                fine.ky,
                torch_mg._interp_axis,
            ),
            fine.mask,
        )
        RAPv = torch_fem.project(
            torch_mg._on_node_grid(
                fine.apply_A(Pv),
                fine.nelx + 1,
                fine.nely + 1,
                fine.kx,
                fine.ky,
                torch_mg._restrict_axis,
            ),
            coarse.mask,
        )
        assert_close(coarse.apply_A(v).numpy(), RAPv.numpy(), tier="algebraic")


def test_coarsening_stops_rather_than_mis_coarsening_odd_dimensions():
    """45x15 is the case that actually arises on the production meshes: odd in both
    dimensions, so a factor of 2 is unavailable and the policy must say something
    definite rather than silently mis-coarsening.
    """
    assert torch_mg.coarsening_factors(180, 60) == (2, 2)
    assert torch_mg.coarsening_factors(45, 15) == (3, 3)
    assert torch_mg.coarsening_factors(15, 5) == (3, 1)  # x only; y cannot coarsen
    assert torch_mg.coarsening_factors(5, 5) == (1, 1)  # nothing left to do
    # Every level's dimensions are exactly the parent's divided by the factors used.
    levels = _hierarchy(24, 16, np.full(24 * 16, 0.4), max_coarse_elements=4)
    for fine, coarse in zip(levels, levels[1:]):
        assert (coarse.nelx, coarse.nely) == (
            fine.nelx // fine.kx,
            fine.nely // fine.ky,
        )
    assert torch_mg.coarsening_factors(levels[-1].nelx, levels[-1].nely) == (1, 1)


@pytest.mark.parametrize("nelx, nely", MG_MESH_SIZES)
def test_vcycle_is_symmetric_and_positive_definite(nelx, nely):
    """`<Mx, y> == <x, My>` and `<x, Mx> > 0`, across all three density fields.

    A nonsymmetric preconditioner does not announce itself -- CG just stops converging
    at the rate it should -- so this is checked directly rather than inferred from
    iteration counts.
    """
    rng = np.random.default_rng(2)
    for xPhys in _mg_density_fields(nelx, nely, rng).values():
        levels = _hierarchy(nelx, nely, xPhys)
        M = torch_mg.VCycle(levels)
        mask = levels[0].mask
        # The smoother divides by these. `Emin` keeps them positive even where xPhys is
        # exactly 0.0, at every level -- assert it rather than trusting it.
        for level in levels:
            assert torch.all(level.diag > 0.0)
        u = torch_fem.project(torch.tensor(rng.standard_normal(levels[0].ndof)), mask)
        v = torch_fem.project(torch.tensor(rng.standard_normal(levels[0].ndof)), mask)
        Mu, Mv = M(u), M(v)
        assert_close((Mu * v).sum().item(), (u * Mv).sum().item(), tier="algebraic")
        assert (u * Mu).sum().item() > 0.0
        assert torch.all(Mu[~mask] == 0.0)


@pytest.mark.parametrize("nelx, nely", MG_MESH_SIZES)
def test_mgcg_matches_spsolve(nelx, nely):
    """MGCG must reproduce `spsolve`'s answer, and satisfy the system in its own right.

    The comparison is scaled by `||U||_inf` rather than made element by element. At the
    binary field's ~1e9 contrast `cond(K)` reaches ~1e11, so `U` itself is only pinned
    down to about `cond * eps * ||U||` by float64 -- both solvers sit at that limit and
    a strict element-wise relative check would be testing the oracle's rounding, not
    this solver. The true-residual assertion is what actually pins correctness down,
    and it is independent of the oracle: CG tracks its residual by a recurrence that
    can drift from the real one, so it is recomputed here from scratch.
    """
    rng = np.random.default_rng(3)
    F_np, freedofs_np, ndof, KE_np, edofMat_np, _, _, _ = _setup(nelx, nely)
    fixed = np.setdiff1d(np.arange(ndof), freedofs_np)
    for xPhys in _mg_density_fields(nelx, nely, rng).values():
        K = fem.assemble_stiffness(KE_np, xPhys, EMIN, EMAX, PENAL, edofMat_np, ndof)
        expected = fem.solve_fe(K, F_np, freedofs_np)
        actual, n_iter = _mg_solve(nelx, nely, xPhys, rtol=1e-11)
        assert n_iter > 0

        b = F_np.copy()
        b[fixed] = 0.0
        residual = b - K @ actual.numpy()
        residual[fixed] = 0.0
        assert np.linalg.norm(residual) / np.linalg.norm(b) < 1e-10

        assert_close(
            actual.numpy(),
            expected,
            tier="solved",
            atol=1e-6 * np.abs(expected).max(),
        )


@pytest.mark.parametrize("nelx, nely", MG_MESH_SIZES)
def test_mgcg_matches_jacobi_pcg_in_far_fewer_iterations(nelx, nely):
    """Same system, same answer; the only thing that changes is the iteration count."""
    _, _, _, _, _, KE, edofMat, mask = _setup(nelx, nely)
    F_np, _, _ = point_load_problem(nelx, nely)
    F = torch.tensor(F_np, dtype=torch.float64)
    xPhys = _binary_design(nelx, nely)
    U_jac, n_jac = torch_fem.solve(
        F,
        torch.tensor(xPhys, dtype=torch.float64),
        edofMat,
        KE,
        EMIN,
        EMAX,
        PENAL,
        mask,
        rtol=1e-11,
        max_iter=50000,
    )
    U_mg, n_mg = _mg_solve(nelx, nely, xPhys, rtol=1e-11)
    assert n_mg < n_jac / 5
    # Scaled atol for the same reason as test_mgcg_matches_spsolve.
    assert_close(
        U_mg.numpy(),
        U_jac.numpy(),
        tier="solved",
        atol=1e-6 * np.abs(U_jac.numpy()).max(),
    )


def test_mgcg_batched_matches_sequential():
    """Batched over both right-hand side and density field, as the nStage solves are."""
    nelx, nely = 18, 12
    rng = np.random.default_rng(4)
    _, _, ndof, _, _, KE, edofMat, mask = _setup(nelx, nely)
    F_np, _, _ = point_load_problem(nelx, nely)
    F = torch.tensor(F_np, dtype=torch.float64)
    n_stage = 3
    xPhys = np.stack(
        [
            _binary_design(nelx, nely),
            np.full(nelx * nely, 0.4),
            rng.uniform(0.05, 1.0, nelx * nely),
        ]
    )
    sequential = [
        _mg_solve(nelx, nely, xPhys[s], rtol=1e-11)[0] for s in range(n_stage)
    ]

    batched, _ = torch_mg.solve(
        F.unsqueeze(0).expand(n_stage, -1),
        torch.tensor(xPhys, dtype=torch.float64),
        edofMat,
        KE,
        EMIN,
        EMAX,
        PENAL,
        mask,
        nelx,
        nely,
        rtol=1e-11,
        **MG_KW,
    )
    assert batched.shape == (n_stage, ndof)
    for s in range(n_stage):
        assert_close(
            batched[s].numpy(),
            sequential[s].numpy(),
            tier="solved",
            atol=1e-6 * np.abs(sequential[s].numpy()).max(),
        )


def test_mgcg_warm_start_and_boundary_conditions():
    nelx, nely = 18, 12
    _, _, ndof, _, _, _, _, mask = _setup(nelx, nely)
    xPhys = _binary_design(nelx, nely)
    U_cold, n_cold = _mg_solve(nelx, nely, xPhys, rtol=1e-11)
    assert torch.all(U_cold[~mask] == 0.0)

    rng = np.random.default_rng(5)
    x0 = torch_fem.project(
        U_cold + torch.tensor(rng.standard_normal(ndof) * 1e-6), mask
    )
    U_warm, n_warm = _mg_solve(nelx, nely, xPhys, rtol=1e-11, x0=x0)
    assert n_warm < n_cold
    assert_close(
        U_warm.numpy(),
        U_cold.numpy(),
        tier="solved",
        atol=1e-6 * np.abs(U_cold.numpy()).max(),
    )


def test_mgcg_nonconvergence_raises():
    nelx, nely = 18, 12
    with pytest.raises(torch_fem.CGConvergenceError) as excinfo:
        _mg_solve(nelx, nely, _binary_design(nelx, nely), rtol=1e-14, max_iter=2)
    assert excinfo.value.n_iter == 2


def test_mgcg_dtype_float64_end_to_end():
    nelx, nely = 12, 8
    xPhys = _binary_design(nelx, nely)
    U, _ = _mg_solve(nelx, nely, xPhys, rtol=1e-10)
    assert U.dtype == torch.float64
    levels = _hierarchy(nelx, nely, xPhys)
    for level in levels:
        assert level.diag.dtype == torch.float64
    assert levels[-1].chol.dtype == torch.float64


# --- Sensitivity accuracy: what pins the CG tolerance (the plan's test 6) ---------

CALIB_MESH = "90x30"  # the fixture's smaller near-binary design, for test runtime


def _calibration_case():
    nelx, nely = (int(v) for v in CALIB_MESH.split("x"))
    with np.load(calib.FIXTURES) as data:
        x = data[f"x_{CALIB_MESH}_it0800"]
        t = data[f"t_{CALIB_MESH}_it0800"]
    return calib.mesh_setup(nelx, nely), x, t


def test_sensitivities_from_mgcg_match_spsolve_elementwise():
    """Element-wise, not by norm: MMA reads every element of `dcx`/`dct`, so a single
    bad element is a real defect that an L2 norm would hide.
    """
    setup, x, t = _calibration_case()
    n_stage = 2
    ref = calib.sensitivities(setup, x, t, n_stage)
    with calib.mgcg_backend(setup, rtol=calib.RECOMMENDED_RTOL) as iters:
        got = calib.sensitivities(setup, x, t, n_stage)
    assert len(iters) == 1 + n_stage and min(iters) > 0

    for key in ("dcx", "dcx_g", "dct_g"):
        rel_active, abs_over_peak = calib.elementwise_errors(
            got[key].ravel(), ref[key].ravel()
        )
        assert rel_active < calib.SENSITIVITY_TOL, key
        assert abs_over_peak < calib.SENSITIVITY_TOL, key


def test_compliance_is_far_more_forgiving_than_its_sensitivities():
    """The asymmetry the tolerance policy rests on, asserted rather than assumed.

    `c` is stationary at the solution so its error is second order in the error of `U`;
    per-element `ce = Ue^T KE Ue` is not. At a tolerance loose enough to wreck `dcx`,
    `c` is still correct to more digits than the optimizer could ever need -- which is
    exactly why calibrating against `c` would pick a tolerance that quietly degrades
    MMA's search direction.
    """
    setup, x, t = _calibration_case()
    ref = calib.sensitivities(setup, x, t, 1)
    with calib.mgcg_backend(setup, rtol=1e-4):
        got = calib.sensitivities(setup, x, t, 1)

    c_err = abs(got["c"] - ref["c"]) / abs(ref["c"])
    dcx_err, _ = calib.elementwise_errors(got["dcx"], ref["dcx"])
    assert c_err < 1e-6
    assert dcx_err > 1e-3
    assert dcx_err / c_err > 1e4


def test_mgcg_sensitivity_matches_finite_difference():
    """Oracle-free check of the whole chain through the CG solve: operator, V-cycle, CG
    and the sensitivity algebra against the definition of a derivative.
    """
    nelx, nely = 12, 8
    rng = np.random.default_rng(6)
    # Mid-range densities: a near-binary field sits at the bounds of [0, 1], where a
    # central difference is not defined.
    x = rng.uniform(0.3, 0.7, (nely, nelx))
    setup = calib.mesh_setup(nelx, nely)
    elements = rng.choice(nelx * nely, 4, replace=False)
    err = calib.finite_difference_check(setup, x, calib.RECOMMENDED_RTOL, elements)
    assert err < 1e-5
