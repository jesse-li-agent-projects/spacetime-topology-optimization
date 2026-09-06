"""Tests for `sttopt.sensitivity.jacobian_rows` in isolation, independent of either
caller's filter chain."""

import torch

import sttopt.sensitivity as sensitivity


def test_jacobian_rows_k1_matches_autograd():
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0], requires_grad=True)
    out = (a**2).sum() + (b**3).sum()

    (da,) = torch.autograd.grad(out, (a,), retain_graph=True)
    (db,) = torch.autograd.grad(out, (b,), retain_graph=True)

    ja, jb = sensitivity.jacobian_rows(out[None], (a, b))
    torch.testing.assert_close(ja[0], da)
    torch.testing.assert_close(jb[0], db)


def test_jacobian_rows_k_gt_1_matches_per_row_autograd():
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    outputs = torch.stack([a[0] ** 2, a[1] * a[2], a.sum()])

    (ja,) = sensitivity.jacobian_rows(outputs, (a,))
    for i in range(outputs.shape[0]):
        (expected,) = torch.autograd.grad(outputs[i], (a,), retain_graph=True)
        torch.testing.assert_close(ja[i], expected)


def test_jacobian_rows_unused_leaf_is_zero():
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([3.0, 4.0], requires_grad=True)
    out = (a**2).sum()  # doesn't depend on b

    ja, jb = sensitivity.jacobian_rows(out[None], (a, b))
    assert torch.all(jb == 0)
    assert jb.shape == (1, 2)
