"""Tests for sttopt.torch_fem: the matrix-free matvec/diagonal against sttopt.fem's
assembled-matrix path, and the Jacobi-PCG solver against sttopt.fem.solve_fe. All on
CPU and in float64 -- see plans/torch_port.md's Phase 1.
"""

import numpy as np
import pytest
import torch

import sttopt.fem as fem
import sttopt.torch_fem as torch_fem
from conftest import assert_close, point_load_problem

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
