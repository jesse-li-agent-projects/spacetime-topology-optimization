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
        """:param beta: sharpness, possibly scheduled; `resolve` adopts one iteration's."""
        self.beta_schedule = beta
        self.beta = run_config.weight_at(beta, 0)
        self.calibration = 0.0

    def resolve(self, loop: int) -> None:
        """Adopt iteration `loop`'s scheduled sharpness."""
        self.beta = run_config.weight_at(self.beta_schedule, loop)

    def aggregate(self, sev: Float[Tensor, " n"]) -> Float[Tensor, ""]:
        """The calibrated smooth maximum of `sev`, differentiable in it."""
        numer = torch.logsumexp(self.beta * sev, dim=0) / self.beta
        self.calibration = float(numer.detach()) - float(sev.detach().max())
        return numer - self.calibration
