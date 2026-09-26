"""Calibrated smooth maxima of per-element severity fields, for constraint rows that
bound the worst element of a field rather than its average."""

import torch
from jaxtyping import Float
from torch import Tensor

from sttopt import run_config


def density_power(x: Float[Tensor, " n"], r: float) -> Float[Tensor, " n"]:
    """`x**r`, with value and gradient both `0` at `x == 0` rather than `0 * inf = nan`
    from the dead branch of a `torch.where`, for any `r > 0`."""
    solid = x > 0
    return solid * torch.where(solid, x, torch.ones_like(x)) ** r


def calibrated_logsumexp(
    values: Float[Tensor, "*batch n"], beta: float
) -> Float[Tensor, "*batch"]:
    """The smooth maximum of each row of `values`, carried onto its true maximum: the
    value is the row's max, the gradient `logsumexp(beta * values) / beta`'s.

    For a smooth maximum nested inside another, where one calibration scalar cannot
    carry every row: an uncalibrated inner maximum would overshoot by up to
    `log(n) / beta`, loosening whatever bound the outer one feeds.
    """
    smooth = torch.logsumexp(beta * values, dim=-1) / beta
    return smooth - (smooth - values.amax(dim=-1)).detach()


class CalibratedLogSumExp:
    """`log(sum(exp(beta * sev))) / beta`, carried onto the true maximum of `sev` by a
    calibration offset.

    Translation equivariant, so the severity needs no sign or clamp. Its bias is additive
    and bounded by `log(n)/beta`: sharp `beta` tracks the maximum, soft `beta` spreads
    the sensitivity past a handful of elements at the cost of a bias the calibration
    carries back onto the true maximum. `-inf` severities drop out of the sum.

    Every call measures the calibration afresh and holds it out of the gradient, so the
    value is the true maximum and the gradient is the smooth surrogate's. `calibration`
    keeps the last one, for logging.
    """

    def __init__(self, beta: "run_config.Scheduled"):
        """:param beta: sharpness, possibly scheduled."""
        self.beta = beta
        self.calibration = 0.0

    def aggregate(self, sev: Float[Tensor, " n"], loop: int) -> Float[Tensor, ""]:
        """The calibrated smooth maximum of `sev` at iteration `loop`'s sharpness,
        differentiable in `sev`."""
        beta = run_config.weight_at(self.beta, loop)
        numer = torch.logsumexp(beta * sev, dim=0) / beta
        self.calibration = float(numer.detach()) - float(sev.detach().max())
        return numer - self.calibration
