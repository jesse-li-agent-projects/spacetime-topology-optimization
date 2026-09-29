"""Command-line entry point for `stto_heat`, the space-time optimization with the time
field from a virtual heat equation: the counterpart of `stto_cli`, with the same
artefacts under `output/<tag>/`.

At every snapshot it also logs the local-minimum and saddle counts of the time field
over the part, the quantities the heat equation is there to control.
"""

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Space-time topology optimization with the time field from a virtual heat equation (Wu2025 §2.3)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/heat_default.json"),
        help="path to a HeatRunConfig JSON file (default: configs/heat_default.json)",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="run identifier; artefacts are saved under output/<tag>/",
    )
    parser.add_argument(
        "--tag-force",
        action="store_true",
        help="delete an existing output/<tag> directory before this run, instead of erroring out",
    )
    parser.add_argument("--nloop", type=int, help="override the config's nloop")
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=50,
        help="save the design and log the time-field counts every this many iterations (default: 50)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. 'cpu' or 'cuda:0' (default: CUDA if available)",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    import dataclasses
    import json
    import shutil
    import time

    import numpy as np
    import torch

    import sttopt
    import sttopt.checks as checks
    import sttopt.stto_heat as stto_heat
    import sttopt.torch_util as torch_util
    from sttopt.run_config import HeatRunConfig

    config = HeatRunConfig.from_dict(json.loads(args.config.read_text()))
    if args.nloop is not None:
        config = dataclasses.replace(config, nloop=args.nloop)
    assert config.nloop > 0, "Number of iterations must be positive"

    output_dir = Path("output") / args.tag
    if output_dir.exists():
        if not args.tag_force:
            raise SystemExit(
                f"[error] {output_dir} already exists. Use --tag-force to overwrite, or pick a different --tag."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    print(f"sttopt: {sttopt.__file__}")

    problem = stto_heat.build_problem(config, device=args.device)
    state = stto_heat.init_state(problem)
    base = torch_util.to_numpy(problem.Nei)
    start = time.perf_counter()

    def snapshot_counts(xPhys, tPhys) -> dict:
        """The local-minimum and saddle counts of `tPhys` over the solid of `xPhys`."""
        t = torch_util.to_numpy(tPhys)
        solid = torch_util.to_numpy(xPhys) > config.eta
        return dict(
            local_minima=int(checks.unsupported(solid, t, base).sum()),
            saddles=int(checks.saddles(t, solid, solid).sum()),
        )

    def save(path, state, xPhys, tPhys, **extra):
        np.savez_compressed(
            path,
            x=torch_util.to_numpy(state.x),
            mu=torch_util.to_numpy(state.mu),
            xPhys=torch_util.to_numpy(xPhys),
            tPhys=torch_util.to_numpy(tPhys),
            loop=state.loop,
            width_m=config.width_m,
            height_m=config.height_m,
            **extra,
        )

    with (output_dir / "iterations.jsonl").open("w") as log:
        for it in range(config.nloop):
            state, record = stto_heat.step(problem, state)
            with torch.no_grad():
                xPhys, heat = stto_heat.physical_fields(
                    problem, state.x, state.mu, state.beta_d, state.heat
                )
            diag = record.diagnostics
            print(
                f"It.: {it:4d} Obj.: {record.f:10.4f} Vol.: {float(xPhys.mean()):6.3f} "
                f"Tm.: {record.tru_max:7.3f} Unif.: {record.uniformity:8.5f} "
                f"c: {record.obj:9.3f} TmTrue: {diag['true_max']:6.3f}"
            )
            entry = dict(
                loop=it,
                elapsed=time.perf_counter() - start,
                f=record.f,
                obj=record.obj,
                vol=float(xPhys.mean()),
                tru_max=record.tru_max,
                uniformity=record.uniformity,
                roughness=record.roughness,
                roughness_weight=record.roughness_weight,
                g=record.g.tolist(),
                lam=record.lam.tolist(),
                **diag,
            )
            if it % args.snapshot_every == 0:
                entry.update(snapshot_counts(xPhys, heat.tPhys))
                save(output_dir / f"design_it{it:04d}.npz", state, xPhys, heat.tPhys)
            # Flushed per iteration so a running optimization can be followed live
            log.write(json.dumps(entry) + "\n")
            log.flush()

    design = output_dir / "final_design.npz"
    save(design, state, xPhys, heat.tPhys, f=record.f, vol=record.vol)
    report = stto_heat.check_design(problem, state.x, state.mu, state.loop)
    checks.report_path(design).write_text(json.dumps(report, indent=2))
    print(checks.summary(report))


if __name__ == "__main__":
    main(parse_args())
