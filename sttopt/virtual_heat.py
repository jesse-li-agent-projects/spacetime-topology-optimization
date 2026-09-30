"""Time fields from a virtual heat equation, with the diffusivity as a design variable.

The equation has no relation to the real thermal process. Its only job is a time field
without local minima, i.e. without material that prints before everything around it:

- `heat_time_field` (Wu2025 §2.3): `div(chi grad T) - alpha T = 0`, `T = 1` on the build
  plate, zero flux elsewhere, `chi = density * chi(mu)`. The drain makes `T`
  subharmonic, so it has no interior maximum, and a decreasing map `T -> t` has no
  interior minimum over the part.
- `laplace_time_field`: `div(chi grad t) = 0`, `t = 0` on the plate and optimized
  nonnegative data on the other walls, `chi = chi(mu)` over the whole domain. The
  maximum principle leaves no interior extremum over the domain; a dip in the wall
  data is a local minimum on the wall, and two peaks give an interior saddle between
  them.

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
    """The build plate's nodes.

    :param print_base: `edge` (the left edge) or `bottom_edge`
    :return: a mask over the nodes
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
    on_plate = plate_nodes(nelx, nely, print_base)
    last = np.flatnonzero(on_plate[loop])[-1]
    rotated = np.roll(loop, -(last + 1))
    # A plate is one contiguous run of the loop, so after the rotation it is the tail
    return rotated[~on_plate[rotated]]


def element_means(
    u: Float[Tensor, "*batch ndof"], edof: Int[Tensor, "nel 4"]
) -> Float[Tensor, "*batch nel"]:
    """The mean of each element's four nodal values."""
    return u[..., edof].mean(dim=-1)


def diffusivity(
    mu: Float[Tensor, "*batch nel"], contrast: float
) -> Float[Tensor, "*batch nel"]:
    """`chi(mu) = contrast**(mu - 1/2)`.

    `chi = 1` at the initial `mu = 1/2`, and an MMA step in `mu` is a relative change of
    `chi` at any value.
    """
    return contrast ** (mu - 0.5)


def part_mask(xPhys: Float[Tensor, "*shape"]) -> Bool[Tensor, "*shape"]:
    """The elements that count as the part: `xPhys >= max(xPhys) / 2`.

    Never empty: on a uniform grey start it is the whole domain, and on a binary design
    it is the solid.
    """
    return xPhys >= xPhys.max() / 2


def normalize(
    t: Float[Tensor, "nely nelx"], xPhys: Float[Tensor, "nely nelx"]
) -> Float[Tensor, "nely nelx"]:
    """`t` scaled to 1 at its maximum over `part_mask`, the scale treated as a constant.

    (`max(xPhys * t)` would scale a grey start at `xPhys = 0.5` up to about 2.)
    """
    return t / t[part_mask(xPhys)].max().detach()


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
        ints = dict(device=device, dtype=torch.int64)
        floats = dict(device=device, dtype=dtype)
        return cls(
            nelx=nelx,
            nely=nely,
            edof=torch.as_tensor(fem.element_dof_map(nelx, nely, 1), **ints),
            KE=torch.as_tensor(fem.diffusion_KE(), **floats),
            B=torch.as_tensor(_shape_gradients(), **floats),
            plate=torch.as_tensor(plate_nodes(nelx, nely, print_base), device=device),
            arc=torch.as_tensor(wall_arc(nelx, nely, print_base), **ints),
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

        :param chi: per-element diffusivity
        :param g: the values on `fixed`; its other dofs are ignored
        :param fixed: the Dirichlet dofs
        :param rtol: CG relative-residual tolerance
        :param F: the load, zero if `None`
        :param reaction: per-element lumped reaction (drain) coefficient
        :param x0: a previous solution, the warm start (its fixed dofs are ignored)
        :return: `u`
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

    def direction_field(self, s: Float[Tensor, " ndof"]) -> Float[Tensor, "nel 4 2"]:
        """`grad s / |grad s|` at the Gauss points: the unit direction in which `s`
        increases.

        At the Gauss points, since the element centre cannot see an hourglass mode.
        No floor relative to the mean gradient, since `T` falls exponentially (see
        `plans/virtual_heat_timefield.md`, Variant 1).
        """
        grad = self.gauss_gradients(s)
        norm2 = (grad**2).sum(dim=-1)
        # Only against 0 / 0 at an exactly flat point
        tiny = torch.finfo(grad.dtype).tiny
        return grad / torch.sqrt(norm2 + tiny)[..., None]

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
        """The distance fitted to the direction in which `s` increases.

        Crane et al.'s step III: `div(w grad phi) = div(w X)`, `phi = 0` on the plate,
        zero flux elsewhere. A least-squares fit, with no maximum principle.

        :param s: the field whose increase is the direction of time
        :param w: per-element weight of the fit
        :param rtol: CG relative-residual tolerance
        :param x0: a previous `phi`, the warm start
        :return: `phi`
        """
        X = self.direction_field(s)
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
    primary: Float[Tensor, " ndof"]  # T (heat), or the solved t (Laplace)
    poisson: Float[Tensor, " ndof"] | None  # phi, for the `poisson` map


def heat_time_field(
    config: run_config.HeatRunConfig,
    mesh: ScalarMesh,
    xPhys: Float[Tensor, "nely nelx"],
    mu: Float[Tensor, "nely nelx"],
    previous: TimeFieldSolution | None = None,
) -> TimeFieldSolution:
    """Variant 1: the time field from the drained heat equation through the material.

    :param xPhys: physical densities
    :param mu: diffusivity design field
    :param previous: the last iteration's solution, the warm start
    :return: `tPhys` and the nodal solutions
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
            part = part_mask(xPhys).flatten()
            floor = NEG_LOG_FLOOR * element_means(T, mesh.edof)[part].min()
        # Deep in void T falls below the solve error and can be negative; the clamp
        # only acts there, where T carries no resolved value to differentiate
        t = -torch.log(torch.clamp(T, min=0) + floor)
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


def _ramp(mesh: ScalarMesh) -> Float[Tensor, " n_arc"]:
    """The linear ramp away from the plate on the wall arc, `t = distance / l_c`."""
    xy = np.stack(np.divmod(np.arange(mesh.ndof), mesh.nelx + 1), axis=-1)
    plate = xy[mesh.plate.cpu().numpy()]
    arc = xy[mesh.arc.cpu().numpy()]
    # Distance to the nearest plate node: to the plate line, for a straight plate
    distance = np.sqrt(((arc[:, None] - plate[None]) ** 2).sum(-1)).min(axis=1)
    return torch.as_tensor(
        distance / mesh.unit_length, device=mesh.arc.device, dtype=mesh.KE.dtype
    )


def wall_scale(mesh: ScalarMesh) -> float:
    """The wall data at `wall = 1`: twice the ramp's largest value, so the ramp sits
    at `wall <= 1/2` and the data can rise above it anywhere."""
    return 2 * float(_ramp(mesh).max())


def ramp_wall(mesh: ScalarMesh) -> Float[Tensor, " n_arc"]:
    """The wall design of the linear ramp away from the plate: uniform `chi` with this
    wall data gives a linear `t` in the whole domain."""
    return _ramp(mesh) / wall_scale(mesh)


def wall_filter(n_arc: int, rmin: float) -> Float[np.ndarray, "n_arc n_arc"]:
    """The density filter's weights `max(0, rmin - distance)` along the wall arc.

    Past each end the arc continues into the plate, where `t = 0`, so the data is
    reflected there with its sign flipped, about the plate corner node: a ramp rising
    from the plate passes unchanged, and the filtered data of a nonnegative wall design
    stays nonnegative (each flipped term is farther away than its original).

    :param n_arc: nodes on the arc
    :param rmin: filter radius, in elements; at most 1 leaves the data unfiltered
    """
    k = np.arange(n_arc)
    weight = lambda d: np.maximum(0.0, rmin - np.abs(d))
    H = weight(k[:, None] - k[None, :])
    H[k, k] = max(rmin, 1.0)  # the node itself, also where the radius reaches no other
    # The plate corners sit at -1 and n_arc; node j reflects to -2 - j and 2 n_arc - j
    H -= weight(k[:, None] + 2 + k[None, :])
    H -= weight(2 * n_arc - k[None, :] - k[:, None])
    # By the full cone's sum, which a ramp needs to pass unchanged near the ends
    full = max(rmin, 1.0) + 2 * weight(np.arange(1, int(np.ceil(rmin)))).sum()
    return H / full


def laplace_time_field(
    config: run_config.LaplaceRunConfig,
    mesh: ScalarMesh,
    xPhys: Float[Tensor, "nely nelx"],
    mu: Float[Tensor, "nely nelx"],
    wall: Float[Tensor, " n_arc"],
    previous: TimeFieldSolution | None = None,
) -> TimeFieldSolution:
    """Variant 2: the time field of `div(chi grad t) = 0` over the whole domain.

    :param xPhys: physical densities; they only set the normalization, since the field
        does not see the part
    :param mu: diffusivity design field
    :param wall: wall design on `[0, 1]`; the wall data is `wall * wall_scale(mesh)`
    :param previous: the last iteration's solution, the warm start
    :return: `tPhys` and the nodal solutions
    """
    shape = xPhys.shape
    chi = diffusivity(mu.flatten(), config.chi_contrast)
    b = wall * wall_scale(mesh)

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
