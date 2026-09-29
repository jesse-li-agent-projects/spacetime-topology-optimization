"""Time fields from a virtual heat equation, with the diffusivity as a design variable.

The equation has no relation to the real thermal process. Its only job is a time field
without local minima, i.e. without material that prints before everything around it:

- `heat_time_field` (Wu2025 §2.3): `div(chi grad T) - alpha T = 0`, `T = 1` on the build
  plate, zero flux elsewhere, `chi = density * chi(mu)`. The drain makes `T`
  subharmonic, so it has no interior maximum, and a decreasing map `T -> t` has no
  interior minimum over the part.
- `laplace_time_field`: `div(chi grad t) = 0`, `t = 0` on the plate and optimized
  unimodal data on the other walls, `chi = chi(mu)` over the whole domain. The maximum
  principle leaves no interior extremum over the domain.

Both solve on the nodes of the element grid and report element means, the values the
rest of the code reads. The guarantee holds on the nodes; an element mean can still be
a strict extremum where `chi` has element-scale contrast (`plans/virtual_heat_timefield.md`,
Phase 3).

Lengths are in elements, and `l_c` is `timefield.unit_length`, the square root of the
design area.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import sttopt.fem as fem
import sttopt.run_config as run_config
import sttopt.timefield as timefield
import sttopt.torch_fem as torch_fem
import sttopt.torch_solve as torch_solve

#: Floor on the density in `chi = density * chi(mu)`. The drain keeps the matrix SPD at
#: zero density; the floor keeps its conditioning and the multigrid sane.
DENSITY_FLOOR = 1e-6

#: Floor on `T` in the `neg_log` map, relative to the smallest `T` on the part. Set from
#: the part rather than as a constant, so it cannot clip part values at any drain.
NEG_LOG_FLOOR = 1e-3

# Gauss points of a unit square, 2x2, in element coordinates
_G = (0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3))
_GAUSS = [(xi, eta) for eta in _G for xi in _G]


def _shape_gradients() -> Float[np.ndarray, "4 2 4"]:
    """Gradients of the bilinear shape functions at the 2x2 Gauss points, in
    `fem.element_dof_map`'s local node order (bottom left, bottom right, top right,
    top left), x right and y up."""
    return np.array(
        [
            [
                [-(1 - eta), 1 - eta, eta, -eta],
                [-(1 - xi), -xi, xi, 1 - xi],
            ]
            for xi, eta in _GAUSS
        ]
    )


def plate_nodes(nelx: int, nely: int, print_base: str) -> Bool[np.ndarray, " nnodes"]:
    """The build plate: the nodes of the left edge (`edge`) or the bottom edge
    (`bottom_edge`).

    :raises ValueError: for any other print base
    """
    plate = np.zeros((nely + 1, nelx + 1), bool)
    base = print_base.lower()
    if base == "edge":
        plate[:, 0] = True
    elif base == "bottom_edge":
        plate[-1, :] = True
    else:
        raise ValueError(
            f"a virtual-heat time field needs a plate print base (edge or bottom_edge), got {print_base!r}"
        )
    return plate.ravel()


def wall_arc(nelx: int, nely: int, print_base: str) -> Int[np.ndarray, " n_arc"]:
    """The boundary nodes off the plate, in order along the boundary from one end of
    the plate to the other. A corner shared with the plate belongs to the plate."""
    nodes = fem.node_grid(nelx, nely)
    # The boundary loop, clockwise from the top-left corner, each node once
    loop = np.concatenate(
        [nodes[0, :], nodes[1:, -1], nodes[-1, -2::-1], nodes[-2:0:-1, 0]]
    )
    plate = plate_nodes(nelx, nely, print_base)[loop]
    last = np.flatnonzero(plate)[-1]
    rotated = np.roll(loop, -(last + 1))
    # A plate is one contiguous run of the loop, so after the rotation it is the tail
    return rotated[~plate_nodes(nelx, nely, print_base)[rotated]]


def element_means(
    u: Float[Tensor, "*batch ndof"], edof: Int[Tensor, "nel 4"]
) -> Float[Tensor, "*batch nel"]:
    """The mean of each element's four nodal values."""
    return u[..., edof].mean(dim=-1)


def diffusivity(
    mu: Float[Tensor, "*batch nel"], contrast: float
) -> Float[Tensor, "*batch nel"]:
    """`chi(mu) = contrast**(mu - 1/2)`: `chi = 1` at the initial `mu = 1/2`, and an
    MMA step in `mu` is a relative change of `chi` at any value."""
    return contrast ** (mu - 0.5)


def normalize(
    t: Float[Tensor, "nely nelx"], xPhys: Float[Tensor, "nely nelx"]
) -> Float[Tensor, "nely nelx"]:
    """`t` scaled to 1 at its maximum over the part, `xPhys >= max(xPhys) / 2`, with the
    scale treated as a constant.

    That set is never empty: on a uniform grey start it is the whole domain, and on a
    binary design it is the solid. (`max(xPhys * t)` would scale a grey start at
    `xPhys = 0.5` up to about 2.)
    """
    part = xPhys >= xPhys.max() / 2
    return t / t[part].max().detach()


@dataclass(frozen=True)
class ScalarMesh:
    """The scalar Q4 mesh a virtual-heat time field is solved on."""

    nelx: int
    nely: int
    edof: Int[Tensor, "nel 4"]
    KE: Float[Tensor, "4 4"]
    B: Float[Tensor, "4 2 4"]  # shape-function gradients at the Gauss points
    plate: Bool[Tensor, " ndof"]
    arc: Int[Tensor, " n_arc"]  # the walls off the plate, in boundary order

    @classmethod
    def build(
        cls,
        nelx: int,
        nely: int,
        print_base: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "ScalarMesh":
        def t(a, dt):
            return torch.as_tensor(a, device=device, dtype=dt)

        return cls(
            nelx=nelx,
            nely=nely,
            edof=t(fem.element_dof_map(nelx, nely, 1), torch.int64),
            KE=t(fem.diffusion_KE(), dtype),
            B=t(_shape_gradients(), dtype),
            plate=t(plate_nodes(nelx, nely, print_base), torch.bool),
            arc=t(wall_arc(nelx, nely, print_base), torch.int64),
        )

    @property
    def ndof(self) -> int:
        return (self.nelx + 1) * (self.nely + 1)

    @property
    def unit_length(self) -> float:
        """`l_c`, in elements."""
        return (self.nelx * self.nely) ** 0.5

    def solve(
        self,
        chi: Float[Tensor, " nel"],
        g: Float[Tensor, " ndof"],
        fixed: Bool[Tensor, " ndof"],
        rtol: float,
        *,
        F: Float[Tensor, " ndof"] | None = None,
        reaction: Float[Tensor, " nel"] | None = None,
        x0: Float[Tensor, " ndof"] | None = None,
    ) -> Float[Tensor, " ndof"]:
        """`-div(chi grad u) + reaction u = F` with `u = g` on `fixed`.

        :param x0: a previous solution, the warm start (its fixed dofs are ignored)
        """
        mask = ~fixed
        return torch_solve.lifted_femsolve(
            chi,
            torch.zeros_like(g) if F is None else F,
            g,
            self.edof,
            self.KE,
            mask,
            self.nelx,
            self.nely,
            reaction=reaction,
            rtol=rtol,
            x0=None if x0 is None else (x0 * mask).detach(),
        )

    def gauss_gradients(self, u: Float[Tensor, " ndof"]) -> Float[Tensor, "nel 4 2"]:
        """`grad u` at each element's 2x2 Gauss points, per element length."""
        return torch.einsum("gdi,ei->egd", self.B, u[self.edof])

    def direction_field(
        self, s: Float[Tensor, " ndof"], weights: Float[Tensor, " nel"]
    ) -> Float[Tensor, "nel 4 2"]:
        """`grad s / sqrt(|grad s|**2 + eps**2)` at the Gauss points: the unit direction
        in which `s` increases, shortened where `|grad s|` is small against the mean.

        At the Gauss points rather than the element centre, which cannot see an
        hourglass mode. `eps**2` is `timefield.NORMAL_EPS` times the squared
        `weights`-weighted mean gradient, as in `timefield.iso_curvature`.
        """
        grad = self.gauss_gradients(s)
        norm2 = (grad**2).sum(dim=-1)
        with torch.no_grad():
            w = weights[:, None].expand_as(norm2)
            mean = (w * norm2.sqrt()).sum() / w.sum()
        return grad / torch.sqrt(norm2 + timefield.NORMAL_EPS * mean**2)[..., None]

    def divergence_load(
        self, w: Float[Tensor, " nel"], X: Float[Tensor, "nel 4 2"]
    ) -> Float[Tensor, " ndof"]:
        """The load of `-div(w grad phi) = -div(w X)` with zero flux of `w (grad phi -
        X)`: `F_i = integral of w X . grad N_i`, by the 2x2 Gauss rule."""
        per_node = torch.einsum("e,egd,gdi->ei", w, X, self.B) / 4
        out = per_node.new_zeros(self.ndof)
        return out.index_add(0, self.edof.reshape(-1), per_node.reshape(-1))

    def poisson_distance(
        self,
        s: Float[Tensor, " ndof"],
        w: Float[Tensor, " nel"],
        rtol: float,
        x0: Float[Tensor, " ndof"] | None = None,
    ) -> Float[Tensor, " ndof"]:
        """The distance `phi` fitted to the direction in which `s` increases (Crane et
        al.'s step III): `div(w grad phi) = div(w X)`, `phi = 0` on the plate, zero flux
        elsewhere. A least-squares fit, with no maximum principle."""
        X = self.direction_field(s, w)
        return self.solve(
            w,
            torch.zeros_like(s),
            self.plate,
            rtol,
            F=self.divergence_load(w, X),
            x0=x0,
        )


class TimeFieldSolution(NamedTuple):
    """A virtual-heat time field and the nodal solutions behind it, which warm-start
    the next iteration's solves."""

    tPhys: Float[Tensor, "nely nelx"]
    primary: Float[Tensor, " ndof"]  # T (heat) or the harmonic field (Laplace)
    poisson: Float[Tensor, " ndof"] | None  # phi, for the `poisson` map


def heat_time_field(
    config: run_config.HeatRunConfig,
    mesh: ScalarMesh,
    xPhys: Float[Tensor, "nely nelx"],
    mu: Float[Tensor, "nely nelx"],
    previous: TimeFieldSolution | None = None,
) -> TimeFieldSolution:
    """Variant 1: the time field from the drained heat equation through the material.

    :param previous: the last iteration's solution, the warm start
    """
    shape = xPhys.shape
    density = torch.clamp(xPhys.flatten(), min=DENSITY_FLOOR)
    chi = density * diffusivity(mu.flatten(), config.chi_contrast)
    alpha = config.drain_beta / mesh.unit_length**2
    T = mesh.solve(
        chi,
        mesh.plate.to(chi.dtype),
        mesh.plate,
        config.heat_cg_rtol,
        reaction=torch.full_like(chi, alpha),
        x0=None if previous is None else previous.primary,
    )
    phi = None
    time_map = run_config.HeatTimeMap(config.time_map)
    if time_map == run_config.HeatTimeMap.ONE_MINUS:
        t = 1 - T
    elif time_map == run_config.HeatTimeMap.NEG_LOG:
        with torch.no_grad():
            part = xPhys.flatten() >= xPhys.max() / 2
            floor = NEG_LOG_FLOOR * element_means(T, mesh.edof)[part].min()
        t = -torch.log(T + floor)
    elif time_map == run_config.HeatTimeMap.POISSON:
        # T falls away from the plate, so time runs along -grad T
        phi = mesh.poisson_distance(
            -T,
            density,
            config.poisson_cg_rtol,
            x0=None if previous is None else previous.poisson,
        )
        t = phi
    else:
        raise ValueError(f"time_map must be a HeatTimeMap member, got {time_map!r}")
    t_e = element_means(t, mesh.edof).reshape(shape)
    return TimeFieldSolution(normalize(t_e, xPhys), T, phi)


def unimodal_wall_data(
    a: Float[Tensor, " n_arc"], c: Float[Tensor, " n_arc"], step: float
) -> Float[Tensor, " n_arc"]:
    """Wall data with one peak and no dip along the arc, for any `a`, `c` in `[0, 1]`.

    `a[k]` is how much the data may rise at arc node `k` walking forward from the
    plate, `c[k]` the same walking backward from the far end of the plate, each as a
    fraction of `step`. The data is the smaller of the two running sums: the minimum of
    a rising and a falling sequence, so unimodal by construction.
    """
    rising = torch.cumsum(a * step, dim=-1)
    falling = torch.flip(torch.cumsum(torch.flip(c * step, (-1,)), dim=-1), (-1,))
    return torch.minimum(rising, falling)


def wall_coefficients(
    target: Float[Tensor, " n_arc"], step: float
) -> tuple[Float[Tensor, " n_arc"], Float[Tensor, " n_arc"]]:
    """`(a, c)` for which `unimodal_wall_data` reproduces a unimodal `target` that is 0
    at the plate: `a` from its rises walking forward, `c` from its rises walking
    backward.

    :raises ValueError: if `target` is not unimodal, or rises faster than `step` per
        node
    """
    zero = target.new_zeros(1)
    d = torch.diff(torch.cat([zero, target, zero]))
    a = torch.clamp(d[:-1], min=0) / step
    c = torch.clamp(-d[1:], min=0) / step
    if not torch.allclose(unimodal_wall_data(a, c, step), target):
        raise ValueError("target wall data is not unimodal")
    if a.max() > 1 or c.max() > 1:
        raise ValueError(
            f"target wall data rises by more than one step per node ({float(max(a.max(), c.max()))} steps)"
        )
    return a, c


def wall_step(mesh: ScalarMesh) -> float:
    """The largest rise of the wall data per arc node, `2 h / l_c`: twice the rise of
    the linear ramp, so the ramp start sits inside the bounds."""
    return 2 / mesh.unit_length


def ramp_wall_coefficients(
    mesh: ScalarMesh,
) -> tuple[Float[Tensor, " n_arc"], Float[Tensor, " n_arc"]]:
    """`(a, c)` of the linear ramp away from the plate, `t = distance / l_c`: uniform
    `chi` with this wall data gives a linear `t` in the whole domain."""
    xy = np.stack(np.divmod(np.arange(mesh.ndof), mesh.nelx + 1), axis=-1)
    plate = xy[mesh.plate.cpu().numpy()]
    arc = xy[mesh.arc.cpu().numpy()]
    # Distance to the nearest plate node: to the plate line, for a straight plate
    distance = np.sqrt(((arc[:, None] - plate[None]) ** 2).sum(-1)).min(axis=1)
    target = torch.as_tensor(
        distance / mesh.unit_length, device=mesh.arc.device, dtype=mesh.KE.dtype
    )
    return wall_coefficients(target, wall_step(mesh))


def laplace_time_field(
    config: run_config.LaplaceRunConfig,
    mesh: ScalarMesh,
    xPhys: Float[Tensor, "nely nelx"],
    mu: Float[Tensor, "nely nelx"],
    a: Float[Tensor, " n_arc"],
    c: Float[Tensor, " n_arc"],
    previous: TimeFieldSolution | None = None,
) -> TimeFieldSolution:
    """Variant 2: the harmonic time field over the whole domain. `xPhys` only sets the
    normalization; the field does not see the part.

    :param previous: the last iteration's solution, the warm start
    """
    shape = xPhys.shape
    chi = diffusivity(mu.flatten(), config.chi_contrast)
    b = unimodal_wall_data(a, c, wall_step(mesh))
    g = torch.zeros(mesh.ndof, device=chi.device, dtype=chi.dtype).index_put(
        (mesh.arc,), b
    )
    fixed = mesh.plate.clone()
    fixed[mesh.arc] = True
    u = mesh.solve(
        chi,
        g,
        fixed,
        config.laplace_cg_rtol,
        x0=None if previous is None else previous.primary,
    )
    phi = None
    time_map = run_config.LaplaceTimeMap(config.time_map)
    if time_map == run_config.LaplaceTimeMap.IDENTITY:
        t = u
    elif time_map == run_config.LaplaceTimeMap.POISSON:
        # The walls then act only through the direction of grad u
        phi = mesh.poisson_distance(
            u,
            torch.ones_like(chi),
            config.poisson_cg_rtol,
            x0=None if previous is None else previous.poisson,
        )
        t = phi
    else:
        raise ValueError(f"time_map must be a LaplaceTimeMap member, got {time_map!r}")
    t_e = element_means(t, mesh.edof).reshape(shape)
    return TimeFieldSolution(normalize(t_e, xPhys), u, phi)
