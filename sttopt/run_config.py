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
the CLI's `args`, not here. `configs/default.json`/`configs/seq_default.json` are the
single source of default *settings*: a field carries a dataclass default only when a
run record written before that field existed has a well-defined meaning, so that
`viz.py` can still replay an old output directory. Such a default is what the field
used to be implicitly, never what a new run should pick -- the config files carry
that.
"""

import dataclasses
import math
import warnings
from dataclasses import dataclass
from typing import TypeVar

_T = TypeVar("_T", bound="_ConfigMixin")


class _ConfigMixin:
    """Shared JSON round-trip for `RunConfig`/`SeqRunConfig`. Not a dataclass itself --
    each subclass declares its own fields."""

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
    It is the last iteration still decaying: iteration 1 is exactly `initial`, and
    iteration `decay_iterations + 1` onward is exactly `final`.

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
        """The value at 1-indexed iteration `loop`."""
        progress = min(max((loop - 1) / self.decay_iterations, 0.0), 1.0)
        taper = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final + (self.initial - self.final) * taper


def weight_at(setting: "float | CosineSchedule", loop: int) -> float:
    """The value of a possibly-scheduled scalar at 1-indexed iteration `loop`."""
    return setting.at(loop) if isinstance(setting, CosineSchedule) else float(setting)


@dataclass(kw_only=True)
class RunConfig(_ConfigMixin):
    """
    Full hyperparameter set for a single space-time topology optimization run,
    mirroring `stto.build_problem`'s parameters.

    :param print_base: 3D-printing base/start location, naming a
        `timefield.TimeField` member (case-insensitively) for JSON.
    :param Gamma: weight of the layer-thickness-uniformity objective term
        (`timefield.gradient_magnitude_std`); 0 disables it.
    :param hotspot_normalization: a `conductivity.Normalization` member name, whose
        docstring carries the variants. `configs/default.json` stays on `neighborhood`,
        unlike the `seqopt` one: this config is what the MATLAB-port fidelity fixtures
        are generated and checked against, so its hotspot term has to keep matching the
        source's rather than improving on it.
    """

    # Frequently varied -- also exposed as a CLI flag in stto_cli.py.
    nloop: int

    # Config-file-only.
    nelx: int
    nely: int
    volfrac: float
    nStage: int
    Theta: float
    Gamma: float
    Tcr: float
    hotspot_normalization: str = "neighborhood"
    hotspot_aggregation: str = "p_mean"
    hotspot_beta: float = 200.0
    hotspot_density_exponent: float | None = None
    print_base: str
    rmin: float
    lrmin: float
    rmin_cond: float
    beta_d_max: float
    Emin: float
    Emax: float
    nu: float
    penal: float
    eta: float
    p: float
    q: float
    r: float
    rouf: float
    a0: float
    mma_c: float
    move: float
    tmove: float


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
        time drive the result (PR #98). It is the default only so that run records
        written before this field existed still load and replay as they ran.
    :param hotspot_aggregation: a `conductivity.Aggregation` member name, choosing the
        smooth maximum that collapses the severity field. Defaults to the legacy
        variant for the same reason `hotspot_normalization` does.
    :param hotspot_beta: `logsumexp` sharpness. It sets where the sensitivity goes:
        high concentrates nearly all of it on the single hottest element, which is a
        sharp statement but makes MMA chatter as the argmax moves; low spreads it and
        blunts the term. The aggregate overshoots the true maximum by at most
        `log(sum of weights)/beta`. Inert under `p_mean`, which uses `p`.
    :param hotspot_density_exponent: `logsumexp`'s void-suppression exponent. Needed
        at all because void is the hottest thing in the domain -- nothing shields it,
        so an unweighted aggregate reports empty space rather than the part. Must
        exceed 1 or the weight's own gradient diverges at zero density. `null` takes
        `p_mean`'s implicit `r * p`, which is where the "must exceed 1" requirement is
        met only by coincidence of two unrelated knobs. Inert under `p_mean`, and inert
        on any binary geometry, where `x**s == x` for every `s`.
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
    hotspot_normalization: str = "neighborhood"
    hotspot_aggregation: str = "p_mean"
    hotspot_beta: float = 200.0
    hotspot_density_exponent: float | None = None
    uniformity_metric: str
    uniformity_weight: float
    roughness_weight: float | CosineSchedule

    nStage: int
    p: float
    q: float
    r: float
    rouf: float

    a0: float
    mma_c: float
    tmove: float
    asyclamp_min_ratio: float
    asyclamp_max_ratio: float
    asyincr: float
    asydecr: float

    def __post_init__(self) -> None:
        # JSON has no way to say "a CosineSchedule", so a mapping in a scheduled
        # field's place is one. Done here rather than in `from_dict` so that a config
        # assembled in code from parsed JSON fragments coerces identically.
        if isinstance(self.roughness_weight, dict):
            self.roughness_weight = CosineSchedule(**self.roughness_weight)
