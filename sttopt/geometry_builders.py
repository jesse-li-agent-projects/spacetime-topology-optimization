"""Parametric geometry builders for the Wu2025 thesis components, and a CLI to write
them (or a binarized topology-optimized layout) out as `.npz` geometry files for
`seqopt`.

Each builder rasterizes a millimetre-dimensioned polygon onto an `(nely, nelx)` grid by
point-in-polygon on element centres (not hand-written index arithmetic), so a shape
stays a readable list of vertices. All follow the repo's grid convention (row 0 is the
top; see `conventions.md`) with the build plate at the bottom, and take their physical
dimensions as keyword arguments defaulting to the thesis values, so a caller can vary
them without editing the body.
"""

import argparse
from pathlib import Path

import numpy as np
from jaxtyping import Float
from matplotlib.path import Path as MplPath


def _rasterize(
    vertices: list[tuple[float, float]],
    nelx: int,
    nely: int,
    width: float,
    height: float,
) -> Float[np.ndarray, "nely nelx"]:
    """Point-in-polygon rasterization of `vertices` (mm, origin bottom-left, y up) onto
    an `(nely, nelx)` grid spanning `width x height` mm, row 0 the top.
    """
    dx, dy = width / nelx, height / nely
    row, col = np.indices((nely, nelx))
    x = (col + 0.5) * dx
    y = height - (row + 0.5) * dy  # row 0 is the top, so it gets the largest y
    points = np.stack([x.ravel(), y.ravel()], axis=-1)
    inside = MplPath(vertices).contains_points(points)
    return inside.reshape(nely, nelx).astype(float)


def overhang_bracket(
    nelx: int, nely: int, *, width: float = 100.0, height: float = 100.0
) -> Float[np.ndarray, "nely nelx"]:
    """Wu2025 Fig. 3.4(a) / Fig. 2.13(b) "V-shaped": a 50x50 left block, a right column
    50 wide and full height, and a 45-degree cut removing the triangle below the line
    from `(50,0)` to `(100,50)`. Only `x` in `[0, 50]` sits on the build plate.
    """
    scale_x, scale_y = width / 100.0, height / 100.0
    vertices = [
        (0, 0),
        (50 * scale_x, 0),
        (100 * scale_x, 50 * scale_y),
        (100 * scale_x, 100 * scale_y),
        (50 * scale_x, 100 * scale_y),
        (50 * scale_x, 50 * scale_y),
        (0, 50 * scale_y),
    ]
    return _rasterize(vertices, nelx, nely, width, height)


def c_shape(
    nelx: int, nely: int, *, width: float = 200.0, height: float = 240.0
) -> Float[np.ndarray, "nely nelx"]:
    """Wu2025 Fig. 2.13(a): a rectangle with a slot open at the right edge, `x` in
    `[60, 200]`, `y` in `[80, 180]` (thesis dimensions).
    """
    slot_x0, slot_y0, slot_y1 = 60.0, 80.0, 180.0
    vertices = [
        (0, 0),
        (width, 0),
        (width, slot_y0),
        (slot_x0, slot_y0),
        (slot_x0, slot_y1),
        (width, slot_y1),
        (width, height),
        (0, height),
    ]
    return _rasterize(vertices, nelx, nely, width, height)


def l_shape(
    nelx: int, nely: int, *, width: float = 200.0, height: float = 200.0
) -> Float[np.ndarray, "nely nelx"]:
    """Wu2025 Fig. 2.13(c): the union of a vertical arm (`x` in `[0, 80]`, full height)
    and a horizontal base (full width, `y` in `[0, 80]`).
    """
    arm_x, base_y = 80.0, 80.0
    vertices = [
        (0, 0),
        (width, 0),
        (width, base_y),
        (arm_x, base_y),
        (arm_x, height),
        (0, height),
    ]
    return _rasterize(vertices, nelx, nely, width, height)


# name -> (builder, default width mm, default height mm). The default dimensions fix
# the natural aspect ratio `generate` derives the un-specified resolution axis from.
SHAPES = {
    "overhang_bracket": (overhang_bracket, 100.0, 100.0),
    "c_shape": (c_shape, 200.0, 240.0),
    "l_shape": (l_shape, 200.0, 200.0),
}


def _resolution_to_mesh(
    width: float, height: float, resolution: int
) -> tuple[int, int]:
    """`resolution` elements along the taller side; the other side's element count
    follows from the aspect ratio, so a caller cannot silently distort the component by
    picking mismatched `nelx`/`nely`.
    """
    if height >= width:
        nely = resolution
        nelx = max(1, round(resolution * width / height))
    else:
        nelx = resolution
        nely = max(1, round(resolution * height / width))
    return nelx, nely


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or binarize seqopt geometry .npz files."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="rasterize a named thesis shape")
    gen.add_argument("--shape", choices=sorted(SHAPES), required=True)
    gen.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="element count along the shape's taller side; the other side is derived "
        "from its aspect ratio",
    )
    gen.add_argument("--out", type=Path, required=True)

    bina = sub.add_parser(
        "binarize",
        help="threshold a topology-optimized xPhys into a printable geometry",
    )
    bina.add_argument("source", type=Path, help="an .npz file holding an xPhys key")
    bina.add_argument("--threshold", type=float, default=0.5)
    bina.add_argument("--out", type=Path, required=True)

    return parser.parse_args(argv)


def _cmd_generate(args: argparse.Namespace) -> None:
    builder, width, height = SHAPES[args.shape]
    nelx, nely = _resolution_to_mesh(width, height, args.resolution)
    xPhys = builder(nelx, nely, width=width, height=height)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        xPhys=xPhys,
        shape=args.shape,
        width=width,
        height=height,
    )
    print(
        f"Wrote {args.out}: {args.shape} at {nelx}x{nely} elements, "
        f"solid fraction {xPhys.mean():.3f}"
    )


def _cmd_binarize(args: argparse.Namespace) -> None:
    from scipy.ndimage import label

    with np.load(args.source) as data:
        if "xPhys" not in data:
            raise SystemExit(
                f"{args.source}: no 'xPhys' key found; keys present: "
                f"{sorted(data.keys())}"
            )
        xPhys = np.asarray(data["xPhys"])

    binary = (xPhys >= args.threshold).astype(float)
    changed = int(np.count_nonzero((xPhys >= 0.5) != (binary > 0.5)))
    print(
        f"Solid fraction: {xPhys.mean():.3f} -> {binary.mean():.3f}; "
        f"{changed} of {xPhys.size} elements changed side at threshold "
        f"{args.threshold:g} vs. the natural 0.5 cutoff"
    )

    # A solid element with no path to the build plate (bottom row) can't be printed --
    # worth a loud warning, since thresholding can sever a thin member.
    solid = binary.astype(bool)
    nely = solid.shape[0]
    labels, _ = label(solid)
    plate_labels = set(np.unique(labels[nely - 1, :][solid[nely - 1, :]]))
    reachable = (
        np.isin(labels, list(plate_labels)) if plate_labels else np.zeros_like(solid)
    )
    orphaned = solid & ~reachable
    if np.any(orphaned):
        print(
            f"[warning] {int(orphaned.sum())} solid elements have no path to the "
            f"build plate (bottom row) after thresholding at {args.threshold:g}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        xPhys=binary,
        threshold=args.threshold,
        source=str(args.source),
    )
    print(f"Wrote {args.out}")


def main(args: argparse.Namespace) -> None:
    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "binarize":
        _cmd_binarize(args)


if __name__ == "__main__":
    _args = _parse_args()
    main(_args)
