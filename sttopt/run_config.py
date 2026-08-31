"""`RunConfig`: the full set of `optimize.build_problem` hyperparameters as a single
serializable object, so every run directory carries a `config.json` record of exactly
what produced it.

Only a small subset of fields (the ones varied run-to-run in practice) are exposed as
CLI flags in `cli.py`; the rest are reachable only via a `--config` JSON file or by
constructing `RunConfig` directly in code.
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

    # Frequently varied -- also exposed as CLI flags in cli.py.
    nelx: int = 180
    nely: int = 60
    nloop: int = 800
    volfrac: float = 0.5
    tag: str = "default"
    device: str | None = None

    # Config-file-only.
    nStage: int = 8
    Theta: float = 0.1
    Tcr: float = 0.8
    tfield: int = 3
    rmin: float = 4.0
    lrmin: float = 2.0
    rmin_cond: float = 12.0
    beta_d_max: float = 128.0
    Emin: float = 1e-9
    Emax: float = 1.0
    nu: float = 0.3
    penal: float = 3.0
    eta: float = 0.5
    p: float = 25.0
    q: float = 3.0
    r: float = 0.05
    rouf: float = 100.0
    a0: float = 1.0
    mma_c: float = 2500.0
    move: float = 0.01
    tmove: float = 0.01
    batch_fem_solves: bool | None = None

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
