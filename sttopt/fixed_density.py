"""Holding another run's density fixed while a run optimizes only its time field
(the CLIs' `--fixed-density`): the density is not an input to the time-field
parameterization, so this isolates how a parameterization behaves from how the density
co-evolves with it.

The raw density variable is put back after every step, and the projection sharpness is
pinned to the source run's final value, so the physical density is the source's exactly.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from jaxtyping import Float

from sttopt import run_config

Config = TypeVar("Config", bound=run_config.SpaceTimeConfig)
State = TypeVar("State")  # any driver's state: all carry the raw density as `x`


@dataclass(frozen=True)
class FixedDensity:
    """A finished run's density, to hold fixed in another run."""

    x: Float[np.ndarray, "nely nelx"]  # raw design variable
    xPhys: Float[np.ndarray, "nely nelx"]
    beta_d: float  # the projection sharpness the source's final design was built with

    @classmethod
    def load(cls, run_dir: Path) -> "FixedDensity":
        """From a run's `output/<tag>` directory: its final design and config."""
        with np.load(run_dir / "final_design.npz") as design:
            x, xPhys, loop = design["x"], design["xPhys"], int(design["loop"])
        schedule = json.loads((run_dir / "config.json").read_text())["beta_d_schedule"]
        if isinstance(schedule, dict):
            schedule = run_config.schedule_from_dict(schedule)
        return cls(x, xPhys, run_config.weight_at(schedule, loop))

    def pin(self, config: Config) -> Config:
        """`config` with the projection sharpness held at the source's."""
        return dataclasses.replace(config, beta_d_schedule=self.beta_d)

    def hold(self, state: State) -> State:
        """`state` with the source's raw density."""
        x = torch.as_tensor(self.x, device=state.x.device, dtype=state.x.dtype)
        return dataclasses.replace(state, x=x)

    def check(self, xPhys: Float[np.ndarray, "nely nelx"]) -> None:
        """Raises if the physical density a run builds from the held one is not the
        source's, as when the density filter or the domain differ between the configs.
        """
        if xPhys.shape != self.xPhys.shape:
            raise SystemExit(
                f"[error] --fixed-density: mesh shape {xPhys.shape} differs from the source's {self.xPhys.shape}"
            )
        worst = float(np.abs(xPhys - self.xPhys).max())
        if worst > 1e-8:
            raise SystemExit(
                f"[error] --fixed-density: this config's physical density differs from the source's by up to {worst:.2g}; the density filter, projection threshold and domain must match"
            )
