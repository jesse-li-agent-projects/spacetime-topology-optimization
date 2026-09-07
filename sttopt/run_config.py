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
the CLI's `args`, not here. Neither config has default values -- `configs/default.json`
/`configs/seq_default.json` are the single source of default settings.
"""

import dataclasses
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
class RunConfig(_ConfigMixin):
    """
    Full hyperparameter set for a single space-time topology optimization run,
    mirroring `stto.build_problem`'s parameters.

    :param print_base: 3D-printing base/start location, naming a
        `timefield.TimeField` member (case-insensitively) for JSON.
    :param Gamma: weight of the layer-thickness-uniformity objective term
        (`timefield.gradient_magnitude_std`); 0 disables it.
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
    `rmin` either: `seqopt` does not filter the time field (see
    `plans/archive/fixed_geometry_sequence_optimization.md`), so there is no such
    radius to set. No solid/void cutoff either: it would be inert on a binary geometry,
    which `geometry.load_geometry` requires by default, so `geometry.SOLID_THRESHOLD`
    is one constant rather than a setting that can disagree with itself between the
    print-start set and the time-field initialization.

    :param print_base: print-start location, naming a `timefield.TimeField` member
        (case-insensitively) for JSON.
    :param enable_continuity: whether the print-time continuity constraint
        (`constraints.time_field_continuity`) is included at all; ``False`` drops it
        from the MMA constraint stack entirely, rather than relaxing it via `lrmin`.
    :param lrmin: continuity-filter radius, in **elements**, as in `RunConfig`. Since
        the geometry file sets the mesh, rasterizing the same component at a different
        resolution changes what this radius means physically; rescale it by hand to
        match.
    :param rmin_cond: conductivity-neighborhood radius, in elements -- same caveat as
        `lrmin`.
    :param uniformity_metric: a `timefield.UniformityMetric` member name.
    :param nStage: per-stage deposition budget count; 0 disables the stage-volume
        constraints. Also the stage count the plots draw boundaries for.
    """

    nloop: int

    print_base: str

    enable_continuity: bool
    lrmin: float
    rmin_cond: float

    hotspot_weight: float
    uniformity_metric: str
    uniformity_weight: float

    nStage: int
    p: float
    q: float
    r: float
    rouf: float

    a0: float
    mma_c: float
    tmove: float
