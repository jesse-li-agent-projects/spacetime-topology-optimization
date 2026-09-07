"""Jacobian-row assembly shared by `stto.step` and `seqopt.step`.

`stto._sensitivity_rows` finishes with a hand-applied filter adjoint that is specific
to STTO's density/continuity filter chain and stays there; this module holds only the
smaller part both problems need regardless of what sits downstream of the autograd
leaves: turning `k` scalar outputs into one `(k, numel)` block per leaf.
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

    One `torch.autograd.grad` per row. The alternative -- a single
    `is_grads_batched=True` call with one-hot seeds -- is faster per call but only by a
    few ms, which is ~1% of a `step` (PR #86); it also cannot differentiate through a
    sparse matmul or `FemSolve`, because its vmap has no batching rule for either, so it
    constrained what a `k > 1` row was allowed to contain. The loop has no such
    restriction: any row that is differentiable at all can go through it.

    `allow_unused` covers a leaf an output does not depend on: its block is exactly
    zero, matching what a hand-derived predecessor would return.

    :param outputs: `k` independent scalars sharing one autograd graph
    :param leaves: tensors to differentiate w.r.t.
    :return: one `(k, leaf.numel())` tensor per leaf, in `leaves`' order
    """
    rows = [
        torch.autograd.grad(output, tuple(leaves), retain_graph=True, allow_unused=True)
        for output in outputs
    ]

    return tuple(
        torch.stack(
            [
                (
                    torch.zeros(leaf.numel(), dtype=leaf.dtype, device=leaf.device)
                    if grad is None
                    else grad.reshape(-1)
                )
                for grad in grads
            ]
        )
        for leaf, grads in zip(leaves, zip(*rows))
    )
