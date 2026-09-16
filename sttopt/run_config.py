"""`RunConfig`/`SeqRunConfig`: the full hyperparameter set for a `stto`/`seqopt` run as
a single serializable object -- `Problem.config` holds the exact one a `Problem` was
built from, and every run directory carries a JSON record of exactly what produced it.

The two configs are kept flat and separate rather than sharing a base class over their
~14 overlapping fields: the two problems will not keep those fields in step (`seqopt`
has no filter radius, no FEM parameters, ...), so an inheritance hierarchy would be
fighting the two apart rather than helping. Only the JSON round-trip
(`to_dict`/`from_dict`, including the unknown-key warning) is genuinely shared, via
`_ConfigMixin`.

`nloop` is also exposed as a CLI flag on each; every other field is reachable only via
a `--config` JSON file or by constructing the dataclass directly in code. Run
bookkeeping that isn't a `build_problem` hyperparameter (`--tag`, `--device`) lives on
the CLI's `args`, not here.

Neither config has dataclass defaults: `configs/default.json`/`configs/seq_default.json`
are the single source of default settings, so a run's values are never split between
a config file and this module. That holds for a newly added field too, even though it
means run records written before the field existed no longer load -- add the field to
every config file instead.
"""

import bisect
import dataclasses
import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

import numpy as np

_T = TypeVar("_T", bound="_ConfigMixin")


class _ConfigMixin:
    """Shared JSON round-trip for `RunConfig`/`SeqRunConfig`. Not a dataclass itself --
    each subclass declares its own fields."""

    def __post_init__(self) -> None:
        # JSON has no way to name a schedule class, so a mapping in a field's place is
        # one, told apart by its keys. Done here rather than in `from_dict` so that a
        # config assembled in code from parsed JSON fragments coerces identically.
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict):
                setattr(self, f.name, schedule_from_dict(value))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls: type[_T], d: dict) -> _T:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = d.keys() - known
        if unknown:
            warnings.warn(
                f"{cls.__name__}.from_dict: dropping unrecognized key(s) "
                f"{sorted(unknown)}"
            )
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(kw_only=True)
class CosineSchedule:
    """A scalar hyperparameter decaying from `initial` to `final` along a half cosine
    over a run's first `decay_iterations` iterations, then holding `final`.

    Use it to release a regularizer once the design is in a good basin, and to ask how
    far it can be released. `decay_iterations` is independent of `nloop`, so the decay
    can finish well before the run does and leave the rest of the budget at `final`.
    It is the number of iterations still decaying: iteration 0 is exactly `initial`, and
    iteration `decay_iterations` onward is exactly `final`.

    Cosine rather than linear because it is flat at both ends -- the weight stays near
    `initial` while the design is still finding its basin, and settles onto `final`
    without a discontinuity in the objective.

    Wherever a config field accepts one of these it also accepts a bare number, which
    means a constant; `weight_at` resolves either. JSON has no way to name a class, so
    a mapping in such a field's place is one of these.
    """

    initial: float
    decay_iterations: int
    final: float

    def __post_init__(self) -> None:
        if self.decay_iterations < 1:
            raise ValueError(
                f"decay_iterations must be at least 1, got {self.decay_iterations}; a "
                f"zero-length decay is a constant at `final`, which a bare number says "
                f"more clearly"
            )

    def at(self, loop: int) -> float:
        """The value at 0-indexed iteration `loop`."""
        progress = min(max(loop / self.decay_iterations, 0.0), 1.0)
        taper = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final + (self.initial - self.final) * taper


class Interpolation(StrEnum):
    """How a `PiecewiseSchedule` fills the gaps between its points.

    LOG interpolates in log space, which suits multiplicative settings such as a
    projection or aggregation sharpness. STEP does not interpolate at all: each point's
    value holds until the next point's iteration, for a continuation that jumps.
    """

    LINEAR = "linear"
    LOG = "log"
    STEP = "step"


@dataclass(kw_only=True)
class PiecewiseSchedule:
    """A scalar hyperparameter defined by `(iteration, value)` points, held at the first
    /last value outside them and filled between them per `mode`.

    Use it for continuation that a single decay cannot express: a delayed start, a ramp
    up, a jump, or several phases tied to one timeline.
    """

    points: list[list[float]]
    mode: str = Interpolation.LINEAR

    def __post_init__(self) -> None:
        loops = [p[0] for p in self.points]
        if not loops or loops != sorted(loops):
            raise ValueError(
                f"points must be non-empty and sorted by iteration, got {self.points}"
            )
        # Raises on an unknown mode, so a typo fails at load rather than silently
        # interpolating.
        mode = Interpolation(self.mode)
        if mode is Interpolation.LOG and any(p[1] <= 0 for p in self.points):
            raise ValueError(
                f"log interpolation needs positive values, got {self.points}"
            )

    def at(self, loop: int) -> float:
        """The value at 0-indexed iteration `loop`."""
        loops = [p[0] for p in self.points]
        values = [p[1] for p in self.points]
        mode = Interpolation(self.mode)
        if mode is Interpolation.STEP:
            return float(values[max(bisect.bisect_right(loops, loop) - 1, 0)])
        if mode is Interpolation.LOG:
            return math.exp(float(np.interp(loop, loops, np.log(values))))
        return float(np.interp(loop, loops, values))


Scheduled = float | CosineSchedule | PiecewiseSchedule


def schedule_from_dict(d: dict) -> CosineSchedule | PiecewiseSchedule:
    """The schedule a JSON mapping describes, told apart by its keys."""
    return PiecewiseSchedule(**d) if "points" in d else CosineSchedule(**d)


def weight_at(setting: Scheduled, loop: int) -> float:
    """The value of a possibly-scheduled scalar at 0-indexed iteration `loop`."""
    if isinstance(setting, (CosineSchedule, PiecewiseSchedule)):
        return setting.at(loop)
    return float(setting)


def final_value(setting: Scheduled) -> float:
    """The value a possibly-scheduled scalar settles on, for tooling that reads a
    finished run rather than one iteration of it."""
    return weight_at(setting, 2**62)


@dataclass(kw_only=True)
class RunConfig(_ConfigMixin):
    """
    Full hyperparameter set for a single space-time topology optimization run,
    mirroring `stto.build_problem`'s parameters.

    :param print_base: 3D-printing base/start location, naming a
        `timefield.TimeField` member (case-insensitively) for JSON.
    :param uniformity_metric: a `timefield.UniformityMetric` member name, as in
        `SeqRunConfig`; the penalty is weighted by `xPhys`.
    :param uniformity_weight: weight of the layer-uniformity objective term; 0 disables
        it. The term is dimensionless, so this weight is on the scale of the compliance.
    :param roughness_weight: weight on `timefield.relative_roughness`, a number or a
        `CosineSchedule`, as in `SeqRunConfig` -- whose docstring explains why the
        uniformity term needs it.
    :param enable_stage_volume: whether the per-stage volume bounds
        (`constraints.stage_volume_bounds`) are in the MMA constraint stack at all.
        `nStage` still sets the `Theta`-weighted stage compliances either way.
    :param hotspot_normalization: as in `SeqRunConfig`, as are the other `hotspot_*`
        fields.
    :param time_filter_rmin: density-filter radius applied to `t`, in elements, as in
        `SeqRunConfig`; separate from `rmin` so the two fields can be smoothed
        differently. 0 leaves `t` unfiltered.
    :param enable_continuity: whether the time-field continuity constraint is in the
        MMA constraint stack at all, as in `SeqRunConfig`, as is `continuity_tol`.
    """

    # Frequently varied -- also exposed as a CLI flag in stto_cli.py.
    nloop: int

    # Config-file-only.
    nelx: int
    nely: int
    volfrac: float
    nStage: int
    enable_stage_volume: bool
    Theta: float
    uniformity_metric: str
    uniformity_weight: Scheduled
    roughness_weight: Scheduled
    Tcr: Scheduled
    hotspot_normalization: str
    hotspot_aggregation: str
    hotspot_beta: Scheduled
    print_base: str
    rmin: float
    time_filter_rmin: float
    enable_continuity: bool
    continuity_tol: float
    lrmin: float
    rmin_cond: float
    Emin: float
    Emax: float
    nu: float
    penal: Scheduled
    eta: float
    p: float  # p-mean exponent for hotspot severity aggregation (p_mean only)
    q: float  # hotspot conductivity SIMP exponent
    r: float  # density exponent for hotspot severity
    rouf: Scheduled
    a0: float
    mma_c: float
    move: float
    tmove: float

    # The density/time projection sharpnesses, scheduled like any other setting; the
    # long-standing ramps are `Interpolation.STEP` schedules in the config files.
    beta_d_schedule: Scheduled
    beta_t_schedule: Scheduled
    # Iterations between hotspot calibration refreshes. Iteration 0 is always one of
    # them, which is what calibrates the aggregate against the seed. A change in a
    # scheduled `hotspot_beta` or `rouf` also refreshes it, since both move the
    # aggregate's bias.
    hotspot_refresh_period: int


@dataclass(kw_only=True)
class SeqRunConfig(_ConfigMixin):
    """
    Full hyperparameter set for a single fixed-geometry fabrication-sequence
    (`seqopt`) run, mirroring `seqopt.build_problem`'s parameters.

    No `nelx`/`nely`: the geometry file (`--geometry`) defines the mesh, so a mismatch
    between config and geometry is unrepresentable rather than merely caught. No
    solid/void cutoff either: it would be inert on a binary geometry,
    which `geometry.load_geometry` requires by default, so `geometry.SOLID_THRESHOLD`
    is one constant rather than a setting that can disagree with itself between the
    print-start set and the time-field initialization.

    :param print_base: print-start location, naming a `timefield.TimeField` member
        (case-insensitively) for JSON.
    :param void_extension: a `timefield.VoidExtension` member name, choosing how the
        initialization fills `t` over void. Not a cosmetic choice: `geodesic` leaves a
        jump in `t` at the material interface, which on the c-shape puts the initial
        uniformity CV at 2.22 against `harmonic`'s 0.097, and a run has to spend
        iterations undoing it.
    :param enable_continuity: whether the print-time continuity constraint
        (`constraints.time_field_continuity`) is included at all; ``False`` drops it
        from the MMA constraint stack entirely, rather than relaxing it via `lrmin`.
    :param continuity_tol: `constraints.time_field_continuity`'s bound on the time
        field's mean squared deviation from its local neighborhood average -- the only
        term deciding how smooth the answer is. Inert when `enable_continuity` is False.
    :param lrmin: continuity-filter radius, in **elements**, as in `RunConfig`. Since
        the geometry file sets the mesh, rasterizing the same component at a different
        resolution changes what this radius means physically; rescale it by hand to
        match.
    :param rmin_cond: conductivity-neighborhood radius, in elements -- same caveat as
        `lrmin`.
    :param time_filter_rmin: density-filter radius applied to `t`, in elements; 0 leaves
        the time field unfiltered, which is `seqopt`'s starting design (see its module
        docstring for what a nonzero radius costs, and read `sawtooth_raw` alongside
        `sawtooth` when using one).
    :param hotspot_normalization: a `conductivity.Normalization` member name, choosing
        what `K_est` measures shielding against. Not a tuning knob: `neighborhood`
        cannot see a free surface that lies on the mesh boundary, and lets void print
        time drive the result (PR #98).
    :param hotspot_aggregation: a `conductivity.Aggregation` member name, choosing the
        smooth maximum that collapses the severity field.
    :param hotspot_beta: `logsumexp` sharpness. It sets where the sensitivity goes:
        high concentrates nearly all of it on the single hottest element, which is a
        sharp statement but makes MMA chatter as the argmax moves; low spreads it and
        blunts the term. The aggregate overshoots the true maximum by at most
        `log(sum of weights)/beta`. Inert under `p_mean`, which uses `p`.
    :param uniformity_metric: a `timefield.UniformityMetric` member name.
    :param roughness_weight: weight on `timefield.relative_roughness`, the objective's
        smoothness regularizer -- a number, or a `CosineSchedule` decaying one weight
        into a lower one. Not optional in practice: the uniformity penalty rewards a
        sawtooth across the print direction, so a run with this at 0 converges to a
        jagged field whose reported uniformity is several times better than the truth
        (PR #94). Both terms are dimensionless and divide by the same mean gradient, so
        this weight is a pure ratio and does not need rescaling with the mesh. It is not
        a light touch as a constant: on the c-shape 0.06 still leaves a wiggle 16% of a
        layer deep, and 0.18 is what flattens it, putting this term at 30-50% of the
        uniformity one. A schedule is what buys a lower end weight -- holding near 1.0
        over the first few hundred iterations and then releasing reaches the same
        smoothness ending at 0.06, and at 0.02 alongside a `time_filter_rmin` (PR #95).
        Only a strictly positive *constant* weight makes the sawtooth amplitude
        stationary. A smooth field is not a local minimum of the uniformity penalty
        alone, so under any released floor the mode creeps back with no plateau -- slowly
        enough to be irrelevant over a few hundred iterations, but a schedule's floor is
        a statement about a budget rather than about a converged field.
    :param nStage: per-stage deposition budget count; 0 disables the stage-volume
        constraints. Also the stage count the plots draw boundaries for.
    :param tmove: per-iteration trust-region half-width, in `t` units.
    :param asyclamp_min_ratio: MMA asymptote floor, as a fraction of the initial
        asymptote distance; tightens how far oscillation damping can pull in.
    :param asyclamp_max_ratio: MMA asymptote ceiling, as a multiple of the initial
        distance; usually the binding limit on step size rather than `tmove`, so see
        `mma.trust_region_params` before changing either.
    :param asyincr: factor the asymptotes relax by per iteration when a variable moves
        monotonically.
    :param asydecr: factor they tighten by when a variable oscillates.
    """

    nloop: int

    print_base: str
    void_extension: str

    enable_continuity: bool
    continuity_tol: float
    lrmin: float
    rmin_cond: float
    time_filter_rmin: float

    hotspot_weight: float
    hotspot_normalization: str
    hotspot_aggregation: str
    hotspot_beta: float
    uniformity_metric: str
    uniformity_weight: float
    roughness_weight: float | CosineSchedule

    nStage: int
    p: float  # p-mean exponent for hotspot severity aggregation (p_mean only)
    q: float  # hotspot conductivity SIMP exponent
    r: float  # density exponent for hotspot severity
    rouf: float

    a0: float
    mma_c: float
    tmove: float
    asyclamp_min_ratio: float
    asyclamp_max_ratio: float
    asyincr: float
    asydecr: float
