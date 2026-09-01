"""Autograd vs. hand-derived sensitivities, `plans/torch_port_part2.md` Phase 3.4.

Times the hand-derived value-plus-sensitivity call (`tests/reference/`'s
`compliance`/`constraints`/`conductivity`) against the production autograd
forward-plus-backward (`sttopt`'s), for each of the six sensitivity-producing call
sites (`whole_compliance`, `gravity_compliance`, the four constraints,
`hotspot_constraint`), at 90x30 / 180x60 / 360x120, on the `it0800` near-binary
snapshots. Both sides are torch, on the same device, on the same inputs -- the plan's
"methodological trap to avoid": comparing autograd-on-GPU against the original
NumPy-on-CPU hand-derived code would measure the port and the autodiff together and
would flatter autograd. `tests/reference/`'s hand-derived formulas are the torch port
(Phase 3.2), kept as this comparison's other side after Phase 3.4's autograd
sensitivities superseded them in production (`plans/torch_port_review_followup.md`
Phase 4 moved them out of `sttopt/`); this script is the one place that comparison is
exercised end to end.

Forward and backward are timed separately (a slow backward and a slow forward have
different fixes), and peak memory is reported alongside (the `npairs`-sized hotspot
activations are the other thing that can bite, particularly at 360x120).

Standard hygiene from part 1: `torch.cuda.synchronize()` around every timed region,
warm-up discarded, median of `--repeats` timings, and the machine must be idle for the
numbers to be trustworthy (this script cannot verify that; say so when reporting).

**Fairness: both sides must report the cost of the same quantity.** The four
constraints (`tests/reference/constraints.py`) and `hotspot_constraint`
(`tests/reference/conductivity.py`) bake the density-filter/Heaviside-projection
chain rule (`H`, `Hs`, `dx`) directly into their returned sensitivity -- they hand
back a finished `dfdx` row in raw `x`/`t` space, not `d(.)/d(xPhys)`. Their
production autograd counterparts (`sttopt.constraints`/`conductivity.hotspot_value`)
deliberately do not -- Decision 4 makes the filter/projection an ordinary forward op
for the *caller* to differentiate through -- so a naive `backward()` on one of those
stops one step short, at `d(.)/d(xPhys)` or `d(.)/d(tPhys)`. Timing that against the
hand-derived function's *finished* row is not apples to apples: it omits exactly the
sparse `H @ (... / Hs)` matmul(s) the hand side spends time on.

This benchmark closes that gap by finishing the chain explicitly after every autograd
backward call, timed as part of the same forward+backward region (`_finish_density_chain`/
`_finish_time_chain` below) -- matching what `optimize.step` actually computes for
these rows (`_sensitivity_rows` differentiates to the filtered `(xTilde, tPhys)` cut
and always finishes the density/continuity filter's `H @ (.../Hs)` by hand, whether the
row is a single output or a batch). So both sides here report the cost of one thing: a
`dfdx` row in raw `x`/`t` space.

`whole_compliance`/`gravity_compliance` are the deliberate exception: `compliance.py`'s
hand-derived `dcx`/`dct` are *not* finished rows either -- they stop at `d(.)/d(xPhys)`
by design (same docstring) -- so both sides of those two cells already stop at the same
place with no fix needed; adding a chain-finish there would make *that* comparison
unfair in the other direction.

**360x120's `npairs` is 16x 180x60's, not 4x.** `conductivity.neighbor_weights` scales
both element count and the conductivity filter radius `rmin_cond` by 2x between these
two meshes, so `npairs` (elements times a neighbourhood-area term) scales by ~4x*4x,
not the Risks section's assumed 4x: measured `68744592` vs `4204240`, i.e. **16.35x**.
Each `npairs`-sized float64 intermediate is therefore ~550 MB, not ~34 MB, and
`hotspot_constraint`'s forward alone (hand-derived *or* naive autograd -- this is not
an autograd-vs-hand difference) needs several of them live at once, which does not fit
this benchmark's 8 GB card regardless of implementation. `bench_hotspot` below adds a
third `"compiled"` mode (`torch.compile` on `hotspot_value`, Phase 3.4's escape hatch
1) precisely because it is what resolves this -- fusion means the elementwise pair
algebra never materializes most of those intermediates at all, cutting hotspot's own
peak memory at 360x120 from OOM to ~100 MB over baseline in ad hoc testing. Any cell
that still runs out of memory prints `OOM` in its row instead of aborting the run.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=["90x30", "180x60", "360x120"],
        help="meshes to benchmark",
    )
    parser.add_argument(
        "--iteration", type=int, default=800, help="near-binary snapshot iteration"
    )
    parser.add_argument(
        "--device", default=None, help="torch device (default: cuda if available)"
    )
    parser.add_argument(
        "--repeats", type=int, default=7, help="timed repeats per cell (median)"
    )
    parser.add_argument("--warmup", type=int, default=2, help="discarded warm-up reps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

import gc
import time

import numpy as np
import torch

import sttopt.compliance as compliance
import sttopt.conductivity as conductivity
import sttopt.constraints as constraints
import sttopt.filters as filters
import sttopt.gravity as gravity
import sttopt.torch_fem as torch_fem
import sttopt.torch_util as torch_util
import tests.reference.compliance as compliance_ref
import tests.reference.conductivity as conductivity_ref
import tests.reference.constraints as constraints_ref
from benchmarks.calibrate_cg_rtol import BETA_T, EMAX, EMIN, PENAL, RECOMMENDED_RTOL
from tests.fixtures.generate_torch_port_designs import load_design

RMIN_BY_MESH = {"90x30": 2.0, "180x60": 4.0, "360x120": 8.0}
LRMIN = 2.0
RMIN_COND_BY_MESH = {"90x30": 6.0, "180x60": 12.0, "360x120": 24.0}
Q, R, P, ROUF = 3.0, 0.05, 25.0, 100.0
VOLFRAC = 0.5


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _cleanup(device):
    """Drop autograd graphs and reclaim CUDA memory between cells. Autograd graphs
    hold Python reference cycles (a `Function`'s saved tensors can reach back to the
    graph that produced them), so a bare `del` of a closure's locals is not enough to
    free them promptly -- `gc.collect()` first, then `torch.cuda.empty_cache()` to
    return the freed blocks to the driver rather than leaving them idle in the caching
    allocator. Without this, per-cell peak-memory numbers would still be correct (they
    reset per repeat) but the *cumulative* footprint across cells/meshes in one process
    creeps up, which is what starved the 360x120 hand-derived hotspot cell of the ~500
    MiB it needed on an 8 GB card in the pre-fix run.
    """
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _time_and_peak(fn, device, repeats, warmup):
    """Median wall time (s) and peak CUDA memory (bytes, 0 on CPU) of `fn()`, called for
    its side effect of building an autograd graph or not -- `fn` returns nothing
    interesting, only its timing matters.
    """
    is_cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        fn()
    times = []
    peak = 0
    for _ in range(repeats):
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
        if is_cuda:
            peak = max(peak, torch.cuda.max_memory_allocated())
    return float(np.median(times)), peak


def _setup(mesh: str, device: str, iteration: int):
    nelx, nely = (int(v) for v in mesh.split("x"))
    x, t = load_design(mesh, iteration)
    dtype = torch.float64
    import sttopt.fem as fem

    KE = torch_util.to_tensor(fem.plane_stress_KE(0.3), device, dtype)
    edofMat = torch_util.to_tensor(fem.element_dof_map(nelx, nely), device, torch.int64)
    ndof = 2 * (nelx + 1) * (nely + 1)
    nodes = fem.node_grid(nelx, nely)
    F_np = np.zeros(ndof)
    F_np[2 * nodes[-1, -1] + 1] = -1.0
    left_edge = nodes[:, 0]
    fixeddofs = np.stack([2 * left_edge, 2 * left_edge + 1], axis=-1).ravel()
    freedofs_np = np.setdiff1d(np.arange(ndof), fixeddofs)
    freedofs = torch_util.to_tensor(freedofs_np, device, torch.int64)
    F = torch_util.to_tensor(F_np, device, dtype)
    C = torch_util.csr_to_tensor(gravity.gravity_load_matrix(nelx, nely), device, dtype)
    H, Hs = filters.density_filter(nelx, nely, RMIN_BY_MESH[mesh])
    H_t = torch_util.csr_to_tensor(H, device, dtype)
    Hs_t = torch_util.to_tensor(Hs, device, dtype)
    L_t = torch_util.csr_to_tensor(
        filters.continuity_filter(nelx, nely, LRMIN), device, dtype
    )
    e1, e2, w = conductivity.neighbor_weights(nelx, nely, RMIN_COND_BY_MESH[mesh])
    e1_t = torch_util.to_tensor(e1, device, torch.int64)
    e2_t = torch_util.to_tensor(e2, device, torch.int64)
    w_t = torch_util.to_tensor(w, device, dtype)
    Nei = torch_util.to_tensor(np.arange(nely) * nelx, device, torch.int64)

    xPhys = torch_util.to_tensor(x, device, dtype)
    tPhys = torch_util.to_tensor(t, device, dtype)
    dx = filters.heaviside_projection_derivative(xPhys, 128.0, 0.5)

    return dict(
        nelx=nelx,
        nely=nely,
        ndof=ndof,
        KE=KE,
        edofMat=edofMat,
        freedofs=freedofs,
        F=F,
        C=C,
        H=H_t,
        Hs=Hs_t,
        L=L_t,
        e1=e1_t,
        e2=e2_t,
        w=w_t,
        Nei=Nei,
        xPhys=xPhys,
        tPhys=tPhys,
        dx=dx,
    )


def _leaf(t):
    return t.clone().detach().requires_grad_(True)


def _finish_density_chain(d_xPhys, dx, H, Hs):
    """`H @ (d_xPhys * dx / Hs)` -- the density-filter/Heaviside-projection backward
    the hand-derived constraints/hotspot bake into their own returned row (see this
    module's docstring's "Fairness" note). Applied to an autograd `d(.)/d(xPhys)` to
    make it a finished `dfdx` row, comparable to the hand side's.
    """
    return H @ (d_xPhys.flatten() * dx.flatten() / Hs)


def _finish_time_chain(d_tPhys, H, Hs):
    """`H @ (d_tPhys / Hs)` -- same as `_finish_density_chain` but for the time field,
    which is only density-filtered, never Heaviside-projected (no `dx` factor).
    """
    return H @ (d_tPhys.flatten() / Hs)


class Cell:
    """One (site, mode) timing: forward-only and forward+backward, plus peak memory."""

    def __init__(self, site, mode):
        self.site, self.mode = site, mode
        self.fwd_ms = self.fwdbwd_ms = self.peak_mb = 0.0
        self.oom = False

    def row(self):
        if self.oom:
            return f"{self.site:<22}{self.mode:<12}{'OOM':>10}{'':>12}{'':>10}"
        return (
            f"{self.site:<22}{self.mode:<12}{self.fwd_ms:>10.2f}"
            f"{self.fwdbwd_ms:>12.2f}{self.peak_mb:>10.1f}"
        )


def _time_and_peak_or_oom(fn, device, repeats, warmup):
    """`_time_and_peak`, but a `torch.OutOfMemoryError` is caught and reported as an
    OOM rather than aborting the whole benchmark run -- some (site, mode, mesh)
    combinations are expected to genuinely not fit (see this module's docstring's
    memory note for 360x120's hotspot cells). Returns `(ms, peak_bytes, oom)`.
    """
    try:
        ms, peak = _time_and_peak(fn, device, repeats, warmup)
        return ms, peak, False
    except torch.OutOfMemoryError:
        _cleanup(device)
        return 0.0, 0, True


def bench_whole_compliance(s, device, repeats, warmup, rows):
    for mode in ("hand", "autograd"):
        xPhys = _leaf(s["xPhys"])
        cell = Cell("whole_compliance", mode)

        def fwd():
            if mode == "hand":
                compliance_ref.whole_compliance(
                    xPhys.detach(),
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    s["freedofs"],
                    s["F"],
                    s["ndof"],
                )
            else:
                compliance.whole_compliance(
                    xPhys.detach().requires_grad_(True),
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    s["freedofs"],
                    s["F"],
                    s["ndof"],
                )

        cell.fwd_ms, _ = _time_and_peak(fwd, device, repeats, warmup)

        def fwdbwd():
            if mode == "hand":
                compliance_ref.whole_compliance(
                    xPhys.detach(),
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    s["freedofs"],
                    s["F"],
                    s["ndof"],
                )
            else:
                leaf = xPhys.detach().requires_grad_(True)
                c, _U = compliance.whole_compliance(
                    leaf,
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    s["freedofs"],
                    s["F"],
                    s["ndof"],
                )
                c.backward()

        cell.fwdbwd_ms, peak = _time_and_peak(fwdbwd, device, repeats, warmup)
        cell.fwd_ms *= 1e3
        cell.fwdbwd_ms *= 1e3
        cell.peak_mb = peak / 2**20
        rows.append(cell)
        _cleanup(device)


def bench_gravity_compliance(s, device, repeats, warmup, rows):
    ti = 0.5
    for mode in ("hand", "autograd"):
        cell = Cell("gravity_compliance", mode)

        def fwd():
            xPhys = s["xPhys"].detach()
            tPhys = s["tPhys"].detach()
            if mode == "hand":
                compliance_ref.gravity_compliance(
                    xPhys,
                    tPhys,
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    ti,
                    s["C"],
                    BETA_T,
                    s["freedofs"],
                    s["ndof"],
                )
            else:
                compliance.gravity_compliance(
                    xPhys.requires_grad_(True),
                    tPhys.requires_grad_(True),
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    ti,
                    s["C"],
                    BETA_T,
                    s["freedofs"],
                    s["ndof"],
                )

        cell.fwd_ms, _ = _time_and_peak(fwd, device, repeats, warmup)

        def fwdbwd():
            xPhys = s["xPhys"].detach()
            tPhys = s["tPhys"].detach()
            if mode == "hand":
                compliance_ref.gravity_compliance(
                    xPhys,
                    tPhys,
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    ti,
                    s["C"],
                    BETA_T,
                    s["freedofs"],
                    s["ndof"],
                )
            else:
                xl, tl = xPhys.requires_grad_(True), tPhys.requires_grad_(True)
                cg, _U = compliance.gravity_compliance(
                    xl,
                    tl,
                    s["KE"],
                    s["edofMat"],
                    EMIN,
                    EMAX,
                    PENAL,
                    ti,
                    s["C"],
                    BETA_T,
                    s["freedofs"],
                    s["ndof"],
                )
                cg.backward()

        cell.fwdbwd_ms, peak = _time_and_peak(fwdbwd, device, repeats, warmup)
        cell.fwd_ms *= 1e3
        cell.fwdbwd_ms *= 1e3
        cell.peak_mb = peak / 2**20
        rows.append(cell)
        _cleanup(device)


def bench_constraints(s, device, repeats, warmup, rows):
    specs = [
        (
            "global_volume_fraction",
            lambda xPhys: constraints_ref.global_volume_fraction(
                xPhys.detach(), s["dx"], s["H"], s["Hs"], VOLFRAC
            ),
            lambda xPhys: constraints.global_volume_fraction(xPhys, VOLFRAC),
        ),
        (
            "time_field_continuity",
            lambda tPhys: constraints_ref.time_field_continuity(
                tPhys.detach(), s["L"], s["H"], s["Hs"]
            ),
            lambda tPhys: constraints.time_field_continuity(tPhys, s["L"]),
        ),
        (
            "start_point",
            lambda tPhys: constraints_ref.start_point(
                tPhys.detach(), s["Nei"], s["H"], s["Hs"]
            ),
            lambda tPhys: constraints.start_point(tPhys, s["Nei"]),
        ),
        (
            "stage_volume_bounds",
            lambda xPhys, tPhys: constraints_ref.stage_volume_bounds(
                xPhys.detach(),
                tPhys.detach(),
                s["dx"],
                s["H"],
                s["Hs"],
                0.5,
                VOLFRAC,
                50.0,
            ),
            lambda xPhys, tPhys: constraints.stage_volume_bounds(
                xPhys, tPhys, 0.5, VOLFRAC, 50.0
            ),
        ),
    ]

    def field_for(name):
        return "xPhys" if name in ("global_volume_fraction",) else "tPhys"

    for name, hand_fn, autograd_fn in specs:
        for mode in ("hand", "autograd"):
            cell = Cell(name, mode)

            def fwd(name=name, hand_fn=hand_fn, autograd_fn=autograd_fn, mode=mode):
                if name == "stage_volume_bounds":
                    if mode == "hand":
                        hand_fn(s["xPhys"], s["tPhys"])
                    else:
                        autograd_fn(
                            s["xPhys"].detach().requires_grad_(True),
                            s["tPhys"].detach().requires_grad_(True),
                        )
                else:
                    field = s[field_for(name)]
                    if mode == "hand":
                        hand_fn(field)
                    else:
                        autograd_fn(field.detach().requires_grad_(True))

            cell.fwd_ms, _ = _time_and_peak(fwd, device, repeats, warmup)

            def fwdbwd(name=name, hand_fn=hand_fn, autograd_fn=autograd_fn, mode=mode):
                if name == "stage_volume_bounds":
                    if mode == "hand":
                        hand_fn(s["xPhys"], s["tPhys"])
                    else:
                        xl = s["xPhys"].detach().requires_grad_(True)
                        tl = s["tPhys"].detach().requires_grad_(True)
                        out = autograd_fn(xl, tl)
                        d_xPhys, d_tPhys = torch.autograd.grad(out, (xl, tl))
                        _finish_density_chain(d_xPhys, s["dx"], s["H"], s["Hs"])
                        _finish_time_chain(d_tPhys, s["H"], s["Hs"])
                else:
                    field = s[field_for(name)]
                    if mode == "hand":
                        hand_fn(field)
                    else:
                        leaf = field.detach().requires_grad_(True)
                        out = autograd_fn(leaf)
                        (d_field,) = torch.autograd.grad(
                            out, (leaf,), grad_outputs=torch.ones_like(out)
                        )
                        if field_for(name) == "xPhys":
                            _finish_density_chain(d_field, s["dx"], s["H"], s["Hs"])
                        else:
                            _finish_time_chain(d_field, s["H"], s["Hs"])

            cell.fwdbwd_ms, peak = _time_and_peak(fwdbwd, device, repeats, warmup)
            cell.fwd_ms *= 1e3
            cell.fwdbwd_ms *= 1e3
            cell.peak_mb = peak / 2**20
            rows.append(cell)
            _cleanup(device)


def bench_hotspot(s, device, repeats, warmup, rows):
    # "compiled" is escape hatch 1 from plans/torch_port_part2.md Phase 3.4
    # ("torch.compile the forward... try this before writing anything by hand"),
    # included here (not just "hand"/"autograd") because it is the thing that
    # actually resolves the 360x120 OOM below -- see this module's docstring.
    compiled_hotspot_value = torch.compile(conductivity.hotspot_value)
    for mode in ("hand", "autograd", "compiled"):
        cell = Cell("hotspot_constraint", mode)
        hotspot_fn = (
            compiled_hotspot_value
            if mode == "compiled"
            else (conductivity.hotspot_value)
        )

        def fwd(mode=mode, hotspot_fn=hotspot_fn):
            xPhys, tPhys = s["xPhys"].detach(), s["tPhys"].detach()
            if mode == "hand":
                conductivity_ref.hotspot_constraint(
                    xPhys,
                    tPhys,
                    s["e1"],
                    s["e2"],
                    s["w"],
                    s["dx"],
                    s["H"],
                    s["Hs"],
                    1.0,
                    0.8,
                    P,
                    Q,
                    R,
                    ROUF,
                )
            else:
                hotspot_fn(
                    xPhys.requires_grad_(True),
                    tPhys.requires_grad_(True),
                    s["e1"],
                    s["e2"],
                    s["w"],
                    P,
                    Q,
                    R,
                    ROUF,
                )

        cell.fwd_ms, _, cell.oom = _time_and_peak_or_oom(fwd, device, repeats, warmup)

        def fwdbwd(mode=mode, hotspot_fn=hotspot_fn):
            xPhys, tPhys = s["xPhys"].detach(), s["tPhys"].detach()
            if mode == "hand":
                conductivity_ref.hotspot_constraint(
                    xPhys,
                    tPhys,
                    s["e1"],
                    s["e2"],
                    s["w"],
                    s["dx"],
                    s["H"],
                    s["Hs"],
                    1.0,
                    0.8,
                    P,
                    Q,
                    R,
                    ROUF,
                )
            else:
                xl, tl = xPhys.requires_grad_(True), tPhys.requires_grad_(True)
                numer, _K_est = hotspot_fn(
                    xl, tl, s["e1"], s["e2"], s["w"], P, Q, R, ROUF
                )
                fval = 1.0 * numer / 0.8 - 1
                d_xPhys, d_tPhys = torch.autograd.grad(fval, (xl, tl))
                _finish_density_chain(d_xPhys, s["dx"], s["H"], s["Hs"])
                _finish_time_chain(d_tPhys, s["H"], s["Hs"])

        if not cell.oom:
            cell.fwdbwd_ms, peak, cell.oom = _time_and_peak_or_oom(
                fwdbwd, device, repeats, warmup
            )
            cell.peak_mb = peak / 2**20
        cell.fwd_ms *= 1e3
        cell.fwdbwd_ms *= 1e3
        rows.append(cell)
        _cleanup(device)


def main():
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"repeats={args.repeats} (median), warmup={args.warmup}, "
        f"snapshot it{args.iteration:04d}, CG rtol={RECOMMENDED_RTOL:.0e}"
    )
    print(
        "REMINDER: these numbers are only trustworthy if the machine was otherwise "
        "idle for this run (plans/torch_port_part2.md's 'Standard hygiene' note)."
    )

    header = (
        f"{'site':<22}{'mode':<12}{'fwd(ms)':>10}{'fwd+bwd(ms)':>12}{'peak(MB)':>10}"
    )
    for mesh in args.meshes:
        print(f"\n=== {mesh} ===")
        print(header)
        print("-" * len(header))
        s = _setup(mesh, device, args.iteration)
        rows = []
        bench_whole_compliance(s, device, args.repeats, args.warmup, rows)
        bench_gravity_compliance(s, device, args.repeats, args.warmup, rows)
        bench_constraints(s, device, args.repeats, args.warmup, rows)
        bench_hotspot(s, device, args.repeats, args.warmup, rows)
        for cell in rows:
            print(cell.row(), flush=True)


if __name__ == "__main__":
    main()
