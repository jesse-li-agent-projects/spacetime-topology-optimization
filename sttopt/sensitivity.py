"""Batched-Jacobian assembly shared by `stto.step` and `seqopt.step`.

`stto._sensitivity_rows` finishes with a hand-applied filter adjoint that is specific
to STTO's density/continuity filter chain and stays there; this module holds only the
smaller, subtler part both problems need regardless of what sits downstream of the
autograd leaves: the `k == 1` versus `k > 1` split, and the one-hot
`is_grads_batched` trick that assembles `k` gradient rows in one call.
"""

from typing import Sequence

import torch
from jaxtyping import Float
from torch import Tensor


def jacobian_rows(
    outputs: Float[Tensor, " k"], leaves: Sequence[Tensor]
) -> tuple[Tensor, ...]:
    """Sensitivities of `k` independent scalar outputs (e.g. one per print-start
    element, or one per stage) w.r.t. each of `leaves`, as one `(k, numel)` block per
    leaf, in leaf order.

    `k == 1` differentiates with a plain (unbatched) `torch.autograd.grad`. `k > 1`
    uses `torch.autograd.grad(..., is_grads_batched=True)` with one-hot seeds instead
    of `k` separate calls -- `is_grads_batched`'s vmap has no batching rule for the
    sparse CSR matmul's backward (`RuntimeError: expand is unsupported for SparseCsc
    tensors`, confirmed locally). A `k > 1` output must therefore avoid a sparse matmul
    or a `FemSolve` inside its own graph; true of every current `k > 1` row in both
    callers, but a restriction on future ones, not a guarantee.

    `allow_unused` covers a leaf an output does not depend on: its block is exactly
    zero, matching what a hand-derived predecessor would return.

    :param outputs: `k` independent scalars sharing one autograd graph
    :param leaves: tensors to differentiate w.r.t.
    :return: one `(k, leaf.numel())` tensor per leaf, in `leaves`' order
    """
    k = outputs.shape[0]
    if k == 1:
        grads = torch.autograd.grad(
            outputs[0], tuple(leaves), retain_graph=True, allow_unused=True
        )
    else:
        seeds = torch.eye(k, dtype=outputs.dtype, device=outputs.device)
        grads = torch.autograd.grad(
            outputs,
            tuple(leaves),
            grad_outputs=seeds,
            is_grads_batched=True,
            retain_graph=True,
            allow_unused=True,
        )

    return tuple(
        (
            torch.zeros(k, leaf.numel(), dtype=leaf.dtype, device=leaf.device)
            if grad is None
            else grad.reshape(k, -1)
        )
        for leaf, grad in zip(leaves, grads)
    )
