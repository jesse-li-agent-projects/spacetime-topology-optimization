"""Calibrated smooth maxima of per-element severity fields, for constraints that bound
the worst element of a field rather than its average."""

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

    Every call measures the offset afresh and holds the calibration out of the gradient.
    At `rate` 1 the calibration is that measurement, so the value is the true maximum
    and the gradient is the smooth surrogate's. A lower `rate` moves it only that share
    of the way each call (a moving average), so the function an optimizer sees changes
    slowly between iterations rather than jumping with every measurement (PR #202);
    the first call, and the first after `reset`, takes the measurement whole.
    """

    def __init__(self, beta: "run_config.Scheduled", rate: float = 1.0):
        """
        :param beta: sharpness, possibly scheduled
        :param rate: weight of each new measurement in the calibration, in (0, 1]
        """
        self.beta = beta
        self.rate = rate
        self.calibration = 0.0
        self.fresh = True  # the next measurement is taken whole

    def reset(self) -> None:
        """Forget the calibration, so the next call takes its measurement whole."""
        self.fresh = True

    def aggregate(self, sev: Float[Tensor, " n"], loop: int) -> Float[Tensor, ""]:
        """The calibrated smooth maximum of `sev` at iteration `loop`'s sharpness,
        differentiable in `sev`."""
        beta = run_config.weight_at(self.beta, loop)
        numer = torch.logsumexp(beta * sev, dim=0) / beta
        measured = float(numer.detach()) - float(sev.detach().max())
        self.calibration = blend(
            self.calibration, measured, 1.0 if self.fresh else self.rate
        )
        self.fresh = False
        return numer - self.calibration


def blend(old: float, measured: float, rate: float) -> float:
    """A calibration moved `rate` of the way to a new measurement; at `rate` 1 the
    measurement exactly, as before damping existed, not `old + (measured - old)`."""
    return measured if rate == 1 else old + rate * (measured - old)
