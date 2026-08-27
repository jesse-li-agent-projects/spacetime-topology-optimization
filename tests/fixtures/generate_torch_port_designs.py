"""Generate realistic near-binary `(xPhys, tPhys)` design snapshots for the PyTorch-port
investigation (`plans/torch_port.md`, Phase 0a).

Benchmarking or accuracy-testing a linear solver on the uniform `x = volfrac` field would
be dishonest: that is the best-conditioned design the optimizer ever holds, and it holds
it only at iteration zero. The stiffness contrast that actually stresses a CG solver comes
from the Heaviside projection driving `xPhys` toward 0/1 as `beta_d` ramps, so the
snapshots here span the whole run -- early, middle, and late -- and the late ones are the
cases a go/no-go should turn on.

Two meshes are run natively (90x30 and the production 180x60); 360x120 is not run, and is
instead obtained at load time by a nearest-neighbour 2x block repeat of the 180x60 design
(`load_designs`). Nearest-neighbour, not interpolation: a near-binary design must stay
near-binary, and bilinear upscaling would manufacture intermediate densities and quietly
soften the conditioning that is the property under test. The upscaled 360x120 design
therefore has features twice as coarse in element units as a natively-converged 360x120
design would -- it preserves the hard 0/1 contrast and the void topology, which is what
drives the conditioning, not the feature scale.

Filter radii are in element units, so the 90x30 run halves them to pose the same physical
problem. `lrmin` is the exception: production `lrmin = 2.0` already resolves to the
minimum 3x3 stencil (`filters._neighbor_offsets` uses `ceil(r) - 1`), and halving it to
1.0 leaves an empty stencil, so 90x30 keeps `lrmin = 2.0`.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "torch_port_designs.npz",
        help="output .npz path",
    )
    parser.add_argument(
        "--nloop", type=int, default=800, help="iterations per run (production length)"
    )
    parser.add_argument(
        "--mesh",
        action="append",
        metavar="NELXxNELY",
        help="regenerate only this mesh, merging into an existing --out (repeatable); default is every mesh, overwriting --out",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import time

import numpy as np

import sttopt.optimize as optimize

# Matches tests/test_e2e_slow.py's reproduction of the thesis Chapter 4.4 experiment.
NSTAGE = 8
VOLFRAC = 0.5
THETA = 0.1
TCR = 0.8
TFIELD = 3
BETA_INIT = 1.0

# (nelx, nely, rmin, lrmin, rmin_cond). Radii are in element units; see module docstring
# for why 90x30 keeps lrmin at 2.0 rather than halving it.
MESHES = [
    (90, 30, 2.0, 2.0, 6.0),
    (180, 60, 4.0, 2.0, 12.0),
]

# Iterations to snapshot, as a fraction-free list of loop indices into the trajectory
# (index 0 is the initial uniform field, which is deliberately excluded -- it is the
# unrealistic case these snapshots exist to replace). beta_d doubles every 50 iterations
# from 1.0 and saturates at 128 by iteration 350, so 400+ are the hard, near-binary cases.
SNAPSHOT_LOOPS = [25, 100, 200, 400, 600, 800]


#: Meshes present in the archive natively; anything else is derived by `load_design`.
NATIVE_MESHES = tuple(f"{nelx}x{nely}" for nelx, nely, *_ in MESHES)

#: Meshes obtained by block-repeating a native one, as `derived -> (source, factor)`.
DERIVED_MESHES = {"360x120": ("180x60", 2)}


def load_design(
    mesh: str, iteration: int, path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load one `(xPhys, tPhys)` snapshot, upscaling if the mesh was not run natively.

    :param mesh: `"NELXxNELY"`, either a native mesh or a key of `DERIVED_MESHES`.
    :param iteration: snapshot loop index, one of `SNAPSHOT_LOOPS`.
    :param path: archive to read, defaulting to the one beside this script.
    :return: `(xPhys, tPhys)`, each shape `(nely, nelx)`.
    """
    path = path or Path(__file__).parent / "torch_port_designs.npz"
    source, factor = DERIVED_MESHES.get(mesh, (mesh, 1))
    if source not in NATIVE_MESHES:
        known = ", ".join((*NATIVE_MESHES, *DERIVED_MESHES))
        raise ValueError(f"unknown mesh {mesh!r}; expected one of: {known}")
    with np.load(path) as data:
        x = data[f"x_{source}_it{iteration:04d}"]
        t = data[f"t_{source}_it{iteration:04d}"]
    if factor == 1:
        return x, t
    # Nearest-neighbour block repeat, not interpolation -- see the module docstring.
    block = np.ones((factor, factor))
    return np.kron(x, block), np.kron(t, block)


def binariness(x: np.ndarray) -> float:
    """Fraction of elements within 0.01 of 0 or 1 -- a one-number summary of how far the
    Heaviside projection has driven a density field toward a hard 0/1 contrast.
    """
    return float(np.mean((x < 0.01) | (x > 0.99)))


def generate(nloop: int, meshes: list[str] | None = None) -> dict[str, np.ndarray]:
    """Run each mesh's optimization and collect its snapshots.

    :param nloop: iterations to run per mesh.
    :param meshes: `"NELXxNELY"` names to restrict generation to, or None for all.
    :return: snapshot arrays keyed `"{x,t}_{nelx}x{nely}_it{loop:04d}"`.
    """
    selected = [m for m in MESHES if meshes is None or f"{m[0]}x{m[1]}" in meshes]
    if meshes is not None and len(selected) != len(meshes):
        known = ", ".join(f"{m[0]}x{m[1]}" for m in MESHES)
        raise ValueError(f"--mesh must name one of: {known}; got {meshes}")

    out: dict[str, np.ndarray] = {}
    for nelx, nely, rmin, lrmin, rmin_cond in selected:
        t0 = time.perf_counter()
        result = optimize.run(
            nelx,
            nely,
            nloop,
            NSTAGE,
            VOLFRAC,
            THETA,
            TCR,
            TFIELD,
            rmin,
            lrmin,
            rmin_cond,
            beta_d=BETA_INIT,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"{nelx}x{nely}: {nloop} iterations in {elapsed / 60:.1f} min", flush=True
        )
        for loop in SNAPSHOT_LOOPS:
            if loop > nloop:
                continue
            x = result.xPhys_traj[loop]
            t = result.tPhys_traj[loop]
            out[f"x_{nelx}x{nely}_it{loop:04d}"] = x
            out[f"t_{nelx}x{nely}_it{loop:04d}"] = t
            print(
                f"  it{loop:04d}: binariness={binariness(x):.3f} "
                f"vol={x.mean():.3f} x in [{x.min():.2e}, {x.max():.4f}]",
                flush=True,
            )
    return out


if __name__ == "__main__":
    designs = generate(args.nloop, args.mesh)
    if args.mesh is not None and args.out.exists():
        # Regenerating one mesh leaves the others in place, so a mesh whose trajectory
        # is known-good doesn't have to be paid for again.
        with np.load(args.out) as existing:
            designs = {**dict(existing), **designs}
    np.savez_compressed(args.out, **designs)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
