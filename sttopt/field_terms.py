"""The terms a space-time optimization reads from its physical `(xPhys, tPhys)` pair,
whatever design variables produce that pair: the hotspot measure, the time-field
constraints after the volume row, and the time-field objective terms after the
compliance.

A script that parametrizes `tPhys` differently (directly, or through a PDE) states its
own leading rows and its own compliance, and reads everything else here, so two scripts
cannot score the same design differently.
"""

from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

import sttopt.conductivity as conductivity
import sttopt.constraints as constraints
import sttopt.run_config as run_config
import sttopt.smooth_max as smooth_max
import sttopt.timefield as timefield
import sttopt.torch_util as torch_util
import sttopt.units as units


class ObjectiveTerms(NamedTuple):
    """An objective with the weighted time-field terms added, and their parts for
    logging."""

    value: Float[Tensor, ""]  # the objective the terms were added to
    uniformity: Float[Tensor, ""]  # layer-uniformity penalty, before its weight
    roughness: Float[Tensor, ""]  # roughness regularizer, before its weight
    uniformity_weight: float
    roughness_weight: float


@dataclass(frozen=True)
class FieldTerms:
    """Fixed setup of the shared terms. The smooth maxima hold their own calibration,
    which `constraints` refreshes in place."""

    config: run_config.SpaceTimeConfig
    e1: Int[Tensor, " npairs"]
    e2: Int[Tensor, " npairs"]
    w: Float[Tensor, " npairs"]
    # Constant `K_est` divisor for `config.hotspot_normalization`, or None where the
    # divisor is per-element -- `conductivity.constant_denominator`.
    hotspot_denom: float | None
    # Fixed geometry the angular stencil weight and the directional reference read, or
    # None where neither is used -- see `conductivity.angular_stencil`.
    hotspot_stencil: conductivity.AngularStencil | None
    # Print base elements the hotspot measure treats as infinitely dense, or None where
    # the normalization needs no print base -- `conductivity.infinite_base`.
    hotspot_base: Int[Tensor, " k"] | None
    # Smooth maximum of the hotspot severity.
    hotspot: conductivity.PMean | conductivity.LogSumExp
    # Smooth maximum of the tool-radius curvature severity, or None where
    # `config.tool_radius_m` never leaves 0 and so has no constraint.
    curvature: smooth_max.CalibratedLogSumExp | None
    # Smooth maximum of the gradient-floor severity, or None where
    # `config.min_gradient_fraction` never leaves 0 and so has no constraint.
    min_gradient: smooth_max.CalibratedLogSumExp | None
    # Likewise for the gradient-smoothness severity and `config.gradient_smoothness_m`.
    gradient_smoothness: smooth_max.CalibratedLogSumExp | None

    @classmethod
    def build(
        cls,
        config: run_config.SpaceTimeConfig,
        base: Int[np.ndarray, " k"],
        device: torch.device,
        dtype: torch.dtype,
    ) -> "FieldTerms":
        """The terms `config` enables, with tensors on `device` in `dtype`.

        :param base: the print-start elements
        :raises ValueError: if `config.tool_radius_m` is nonzero on a mesh with no
            interior element to measure curvature at
        """
        nelx, nely = config.nelx, config.nely
        # The tool-radius constraint smooth-maxes over the elements `iso_curvature`
        # measures
        if not run_config.identically_zero(config.tool_radius_m):
            measured = timefield.iso_curvature(
                torch.zeros(nely, nelx, dtype=torch.float64)
            )
            if measured.numel() == 0:
                raise ValueError(
                    f"tool_radius_m needs an interior element to measure curvature at, but iso_curvature measures none on a nelx={nelx}, nely={nely} mesh"
                )
        rmin_cond = units.in_elements(config.rmin_cond_m, config.element_size_m)
        normalization = conductivity.Normalization(config.hotspot_normalization)
        reference = conductivity.ReferenceGradient(config.hotspot_reference_gradient)
        if (
            not run_config.identically_zero(config.hotspot_front_offset_m)
            and reference == conductivity.ReferenceGradient.OWN
        ):
            raise ValueError(
                "hotspot_front_offset_m needs hotspot_reference_gradient 'part_mean' (see conductivity.ReferenceGradient)"
            )
        e1, e2, w = conductivity.neighbor_weights(nelx, nely, rmin_cond)
        hotspot_base = conductivity.infinite_base(normalization, base)
        # int32 halves the largest arrays a run keeps: the pair list.
        ints = torch_util.to_tensors({"e1": e1, "e2": e2}, device, torch.int32)

        def calibrated(
            setting: run_config.Scheduled, beta: run_config.Scheduled
        ) -> smooth_max.CalibratedLogSumExp | None:
            if run_config.identically_zero(setting):
                return None
            return smooth_max.CalibratedLogSumExp(beta)

        return cls(
            config=config,
            w=torch_util.to_tensor(w, device, dtype),
            hotspot_denom=conductivity.constant_denominator(normalization, rmin_cond),
            hotspot_stencil=conductivity.angular_stencil(
                rmin_cond, dtype, config.hotspot_kappa, device, reference
            ),
            hotspot_base=(
                None
                if hotspot_base is None
                else torch_util.to_tensor(base, device, torch.int64)
            ),
            hotspot=conductivity.make_aggregation(
                conductivity.Aggregation(config.hotspot_aggregation),
                config.p,
                config.r,
                config.hotspot_beta,
            ),
            curvature=calibrated(config.tool_radius_m, config.curvature_beta),
            min_gradient=calibrated(
                config.min_gradient_fraction, config.min_gradient_beta
            ),
            gradient_smoothness=calibrated(
                config.gradient_smoothness_m, config.gradient_smoothness_beta
            ),
            **ints,
        )

    def calibrations(self) -> dict[str, float]:
        """
        Each smooth maximum's calibration, by field name: the only state the terms
        carry from one iteration to the next.

        :return: calibration value per field name
        """
        return {
            f.name: term.calibration
            for f in fields(self)
            if hasattr(term := getattr(self, f.name), "calibration")
        }

    def restore_calibrations(self, calibrations: dict[str, float]) -> None:
        """
        Set the calibrations `calibrations()` returned, to resume a run.

        :param calibrations: calibration value per field name
        """
        for name, value in calibrations.items():
            getattr(self, name).calibration = value

    def estimated_conductivity(
        self,
        xPhys: Float[Tensor, "nely nelx"],
        tPhys: Float[Tensor, "nely nelx"],
        loop: int | None = None,
    ) -> Float[Tensor, " nel"]:
        """The `K_est` field behind the run's hotspot term.

        The one path to that field, so offline tooling cannot report a hotspot measure
        the run never optimized -- rebuilding this call by hand once plotted the print
        base as the worst hotspot in the domain (PR #98).

        :param loop: the iteration whose scheduled settings apply, or `None` for the
            values a finished run settles on.
        """
        config = self.config

        def at(setting: run_config.Scheduled) -> float:
            if loop is None:
                return run_config.final_value(setting)
            return run_config.weight_at(setting, loop)

        return conductivity.estimated_conductivity(
            xPhys,
            tPhys,
            self.e1,
            self.e2,
            self.w,
            config.q,
            at(config.rouf),
            self.hotspot_denom,
            self.hotspot_base,
            self.hotspot_stencil,
            at(config.hotspot_kappa),
            config.hotspot_g0_per_m
            * timefield.unit_length_m(tPhys, config.element_size_m),
            conductivity.ReferenceGradient(config.hotspot_reference_gradient),
            units.in_elements(at(config.hotspot_front_offset_m), config.element_size_m),
        )

    def constraints(
        self,
        xPhys: Float[Tensor, "nely nelx"],
        tPhys: Float[Tensor, "nely nelx"],
        K_est: Float[Tensor, " nel"],
        loop: int,
    ) -> dict[str, Float[Tensor, " k"]]:
        """
        The hotspot row and the enabled time-field constraints at iteration `loop`'s
        settings, keyed by the `constraints` function (or `"hotspot"`) that gives them,
        in stack order. Each smooth maximum refreshes its calibration, so its value is
        the true maximum and its gradient the smooth surrogate's.

        :param xPhys: physical densities
        :param tPhys: physical time field
        :param K_est: `estimated_conductivity` of the same fields at `loop`
        :param loop: iteration whose schedules apply
        :return: each constraint's rows, in stack order
        """
        config = self.config
        h = config.element_size_m
        Tcr = run_config.weight_at(config.Tcr, loop)
        g = {"hotspot": (self.hotspot(K_est, xPhys, loop) / Tcr - 1)[None]}
        if self.curvature is not None:
            g["tool_radius"] = constraints.tool_radius(
                xPhys,
                tPhys,
                self.curvature,
                units.in_elements(run_config.weight_at(config.tool_radius_m, loop), h),
                config.r,
                loop,
            )
        if self.min_gradient is not None:
            g["min_gradient"] = constraints.min_gradient(
                xPhys,
                tPhys,
                self.min_gradient,
                run_config.weight_at(config.min_gradient_fraction, loop),
                config.r,
                loop,
            )
        if self.gradient_smoothness is not None:
            g["gradient_smoothness"] = constraints.gradient_smoothness(
                xPhys,
                tPhys,
                self.gradient_smoothness,
                units.in_elements(
                    run_config.weight_at(config.gradient_smoothness_m, loop), h
                ),
                config.r,
                loop,
            )
        return g

    def objective(
        self,
        f: Float[Tensor, ""],
        xPhys: Float[Tensor, "nely nelx"],
        tPhys: Float[Tensor, "nely nelx"],
        loop: int,
    ) -> ObjectiveTerms:
        """`f` plus the layer-uniformity penalty and the roughness regularizer at
        iteration `loop`'s weights. Both are weighted by `xPhys`, so they measure the
        layers of the part, and their gradients move material as well as print time.

        :param f: the structural objective, e.g. the compliance terms
        :param xPhys: physical densities
        :param tPhys: physical time field
        :param loop: iteration whose weights apply
        """
        config = self.config
        uniformity = timefield.uniformity_penalty(
            tPhys, timefield.UniformityMetric(config.uniformity_metric), weights=xPhys
        )
        # The uniformity penalty alone rewards a sawtooth across the print direction
        # (`timefield._gradient_cv`).
        roughness = timefield.relative_roughness(tPhys, weights=xPhys)
        uniformity_weight = run_config.weight_at(config.uniformity_weight, loop)
        roughness_weight = run_config.weight_at(config.roughness_weight, loop)
        return ObjectiveTerms(
            value=f + uniformity_weight * uniformity + roughness_weight * roughness,
            uniformity=uniformity,
            roughness=roughness,
            uniformity_weight=uniformity_weight,
            roughness_weight=roughness_weight,
        )

    def hotspot_diagnostics(
        self,
        g_hotspot: Float[Tensor, ""],
        K_est: Float[Tensor, " nel"],
        xPhys: Float[Tensor, "nely nelx"],
    ) -> dict:
        """
        The true maximum severity and where it sits, and how many elements share the
        hotspot constraint's sensitivity -- `n_eff`, the participation ratio of
        `|d g_hotspot / d K_est|`. A large `n_eff` means the constraint acts like a
        mean over the part rather than on its maximum.

        :param g_hotspot: the hotspot constraint's value, connected to `K_est`
        :return: `true_max`, `hot_row`, `hot_col`, `n_eff`
        """
        (grad,) = torch.autograd.grad(g_hotspot, K_est, retain_graph=True)
        with torch.no_grad():
            a = torch.nan_to_num(grad.abs(), posinf=0.0)
            n_eff = float(a.sum() ** 2 / (a**2).sum()) if bool((a > 0).any()) else 0.0
            severity = conductivity.severity(K_est, xPhys, self.config.r)
            imax = int(severity.argmax())
        nelx = xPhys.shape[1]
        return dict(
            true_max=float(severity[imax]),
            hot_row=imax // nelx,
            hot_col=imax % nelx,
            n_eff=n_eff,
        )

    def admissible_tool_radius_m(
        self,
        xPhys: Float[Tensor, "nely nelx"],
        tPhys: Float[Tensor, "nely nelx"],
    ) -> float:
        """The largest tool radius that no iso-line's concave curvature (density-
        weighted, per element, as in `constraints.tool_radius`) forbids; `inf` where
        nothing is concave."""
        kappa = timefield.iso_curvature(tPhys, xPhys).flatten()
        density_r = smooth_max.density_power(xPhys[1:-1, 1:-1].flatten(), self.config.r)
        concave = -kappa / timefield.unit_length(tPhys) * density_r
        worst = float(concave.max()) if concave.numel() else 0.0
        return self.config.element_size_m / worst if worst > 0 else float("inf")
