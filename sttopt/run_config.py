"""`RunConfig`: the full set of `optimize.build_problem` hyperparameters as a single
serializable object -- `Problem.config` holds the exact one a `Problem` was built
from, and every run directory carries a `config.json` record of exactly what produced
it.

`nloop` is also exposed as a `cli.py` flag; every other field is reachable only via a
`--config` JSON file or by constructing `RunConfig` directly in code. Run bookkeeping
that isn't a `build_problem` hyperparameter (`--tag`, `--device`) lives on `cli.py`'s
`args`, not here. `RunConfig` itself has no default values -- `config/default.json`
(loaded by `cli.py` when `--config` is omitted) is the single source of default
settings.
"""

import dataclasses
import warnings
from dataclasses import dataclass


@dataclass(kw_only=True)
class RunConfig:
    """
    Full hyperparameter set for a single optimization run, mirroring
    `optimize.build_problem`'s parameters.

    :param tfield: `timefield.TimeField` value, stored as a plain int for JSON.
    """

    # Frequently varied -- also exposed as a CLI flag in cli.py.
    nloop: int

    # Config-file-only.
    nelx: int
    nely: int
    volfrac: float
    nStage: int
    Theta: float
    Tcr: float
    tfield: int
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
    batch_fem_solves: bool | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = d.keys() - known
        if unknown:
            warnings.warn(
                f"RunConfig.from_dict: dropping unrecognized key(s) {sorted(unknown)}"
            )
        return cls(**{k: v for k, v in d.items() if k in known})
