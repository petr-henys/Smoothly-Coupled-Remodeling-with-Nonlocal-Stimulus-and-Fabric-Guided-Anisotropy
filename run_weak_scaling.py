"""MPI weak-scaling benchmark for representative FEM solves.

This benchmark is intentionally lightweight and self-contained: it constructs a
unit-cube mesh whose global size grows with the number of MPI ranks to keep the
work per rank approximately constant, solves a linear PDE, and records timings.

Typical usage (run these as separate jobs with different `-n`):
    mpirun -n 1 python run_weak_scaling.py --cells-per-rank 20000 --repeats 3
    mpirun -n 2 python run_weak_scaling.py --cells-per-rank 20000 --repeats 3
    mpirun -n 4 python run_weak_scaling.py --cells-per-rank 20000 --repeats 3

Outputs (rank 0):
    results/weak_scaling/weak_scaling.csv
    results/weak_scaling/weak_scaling_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import ufl
import dolfinx
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)
from mpi4py import MPI
from petsc4py import PETSc


@dataclass(frozen=True)
class BenchmarkResult:
    timestamp_utc: str
    problem: str
    pdeg: int
    ksp_type: str
    pc_type: str
    ranks: int
    nx: int
    ny: int
    nz: int
    cells_global: int
    dofs_global: int
    cells_per_rank_target: int
    cells_per_rank_actual: float
    dofs_per_rank_actual: float
    t_mesh_s: float
    t_problem_setup_s: float
    t_solve_s: float
    ksp_iters: int
    ksp_reason: int


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_conda_toolchain_on_path() -> None:
    """Ensure FFCx JIT finds a C compiler when running from a local conda prefix.

    DOLFINx/FFCx JIT-compiles UFL forms via a subprocess call to a compiler
    (typically `gcc`). When users execute with an explicit interpreter path
    (e.g. `./.conda/bin/python ...`) without activating the environment, the
    corresponding `bin/` directory might not be on PATH. We therefore prepend
    `sys.executable`'s directory if it contains a compiler.
    """
    conda_bin = Path(sys.executable).resolve().parent
    gcc = conda_bin / "gcc"
    if not gcc.exists():
        return

    path = os.environ.get("PATH", "")
    conda_bin_str = str(conda_bin)
    if conda_bin_str not in path.split(os.pathsep):
        os.environ["PATH"] = conda_bin_str + os.pathsep + path


def _mkdir_mpi(path: Path, comm: MPI.Comm) -> None:
    if comm.rank == 0:
        path.mkdir(parents=True, exist_ok=True)
    comm.Barrier()


def _timed_stage(comm: MPI.Comm, fn: Callable[[], Any]) -> tuple[Any, float]:
    comm.Barrier()
    t0 = time.perf_counter()
    out = fn()
    comm.Barrier()
    t1 = time.perf_counter()
    t_local = t1 - t0
    t_max = float(comm.allreduce(t_local, op=MPI.MAX))
    return out, t_max


def _compute_weak_scaling_resolution(size: int, cells_per_rank: int) -> tuple[int, int, int]:
    if size <= 0:
        raise ValueError(f"MPI size must be positive, got {size}.")
    if cells_per_rank <= 0:
        raise ValueError(f"cells_per_rank must be positive, got {cells_per_rank}.")

    target_cells_total = int(cells_per_rank) * int(size)
    n = int(math.ceil(target_cells_total ** (1.0 / 3.0)))
    n = max(1, n)
    return n, n, n


def _build_mesh(comm: MPI.Comm, nx: int, ny: int, nz: int) -> mesh.Mesh:
    return mesh.create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [int(nx), int(ny), int(nz)],
        cell_type=mesh.CellType.hexahedron,
        ghost_mode=mesh.GhostMode.shared_facet,
    )


def _boundary_facets(domain: mesh.Mesh) -> np.ndarray:
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    return facets.astype(np.int32, copy=False)


def _solve_poisson(domain: mesh.Mesh, *, pdeg: int, ksp_type: str, pc_type: str) -> tuple[int, int, float, float]:
    V = fem.functionspace(domain, ("Lagrange", int(pdeg)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dx = ufl.Measure("dx", domain=domain)
    a_form = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
    f = fem.Constant(domain, default_scalar_type(1.0))
    L_form = fem.form(f * v * dx)

    facets = _boundary_facets(domain)
    fdim = domain.topology.dim - 1
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

    A = create_matrix(a_form)
    b = create_vector(V)
    x = fem.Function(V)

    def _setup() -> PETSc.KSP:
        assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(A)
        ksp.setType(str(ksp_type))
        pc = ksp.getPC()
        pc.setType(str(pc_type))
        ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
        ksp.setFromOptions()
        ksp.setUp()
        return ksp

    ksp, t_setup = _timed_stage(domain.comm, _setup)

    def _assemble_and_solve() -> tuple[int, int]:
        with b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(b, L_form)
        apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        ksp.solve(b, x.x.petsc_vec)
        x.x.scatter_forward()
        return int(ksp.getIterationNumber()), int(ksp.getConvergedReason())

    (iters, reason), t_solve = _timed_stage(domain.comm, _assemble_and_solve)

    ksp.destroy()
    A.destroy()
    b.destroy()

    return iters, reason, t_setup, t_solve


def _solve_elasticity(domain: mesh.Mesh, *, pdeg: int, ksp_type: str, pc_type: str) -> tuple[int, int, float, float]:
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", int(pdeg), (gdim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Small-strain linear elasticity (SPD with Dirichlet plane).
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim)

    dx = ufl.Measure("dx", domain=domain)
    a_form = fem.form(ufl.inner(sigma(u), eps(v)) * dx)

    body = fem.Constant(domain, default_scalar_type((1.0, 0.0, 0.0)))
    L_form = fem.form(ufl.inner(body, v) * dx)

    # Clamp x=0 plane to remove rigid motion.
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    facets = facets.astype(np.int32, copy=False)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(np.zeros(gdim, dtype=default_scalar_type), dofs, V)

    A = create_matrix(a_form)
    b = create_vector(V)
    x = fem.Function(V)

    def _setup() -> PETSc.KSP:
        assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(A)
        ksp.setType(str(ksp_type))
        pc = ksp.getPC()
        pc.setType(str(pc_type))
        ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
        ksp.setFromOptions()
        ksp.setUp()
        return ksp

    ksp, t_setup = _timed_stage(domain.comm, _setup)

    def _assemble_and_solve() -> tuple[int, int]:
        with b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(b, L_form)
        apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])
        b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        ksp.solve(b, x.x.petsc_vec)
        x.x.scatter_forward()
        return int(ksp.getIterationNumber()), int(ksp.getConvergedReason())

    (iters, reason), t_solve = _timed_stage(domain.comm, _assemble_and_solve)

    ksp.destroy()
    A.destroy()
    b.destroy()

    return iters, reason, t_setup, t_solve


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--problem",
        choices=("poisson", "elasticity"),
        default="poisson",
        help="Which linear solve to benchmark.",
    )
    p.add_argument("--pdeg", type=int, default=1, help="Polynomial degree (Lagrange).")
    p.add_argument("--cells-per-rank", type=int, default=20000, help="Target cells per MPI rank.")
    p.add_argument("--nx", type=int, default=0, help="Override mesh nx (0 = auto from weak scaling).")
    p.add_argument("--ny", type=int, default=0, help="Override mesh ny (0 = auto from weak scaling).")
    p.add_argument("--nz", type=int, default=0, help="Override mesh nz (0 = auto from weak scaling).")
    p.add_argument("--repeats", type=int, default=3, help="Repeat count; reports median solve time.")
    p.add_argument("--ksp-type", type=str, default="cg", help="PETSc KSP type.")
    p.add_argument("--pc-type", type=str, default="gamg", help="PETSc PC type.")
    p.add_argument("--outdir", type=str, default="results/weak_scaling", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_conda_toolchain_on_path()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if args.nx > 0 and args.ny > 0 and args.nz > 0:
        nx, ny, nz = int(args.nx), int(args.ny), int(args.nz)
    else:
        nx, ny, nz = _compute_weak_scaling_resolution(size, int(args.cells_per_rank))

    outdir = Path(args.outdir)
    _mkdir_mpi(outdir, comm)

    domain, t_setup = _timed_stage(comm, lambda: _build_mesh(comm, nx, ny, nz))

    # Global mesh and DOF counts.
    tdim = domain.topology.dim
    cells_global = int(domain.topology.index_map(tdim).size_global)

    if args.problem == "poisson":
        V = fem.functionspace(domain, ("Lagrange", int(args.pdeg)))
    else:
        gdim = domain.geometry.dim
        V = fem.functionspace(domain, ("Lagrange", int(args.pdeg), (gdim,)))
    dofs_global = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)

    # Repeat solves and report the median wall time (max over ranks).
    setup_times: list[float] = []
    solve_times: list[float] = []
    iters_last = 0
    reason_last = 0

    for _ in range(int(max(1, args.repeats))):
        if args.problem == "poisson":
            iters, reason, t_setup_prob, t_solve = _solve_poisson(
                domain,
                pdeg=int(args.pdeg),
                ksp_type=str(args.ksp_type),
                pc_type=str(args.pc_type),
            )
        else:
            iters, reason, t_setup_prob, t_solve = _solve_elasticity(
                domain,
                pdeg=int(args.pdeg),
                ksp_type=str(args.ksp_type),
                pc_type=str(args.pc_type),
            )
        setup_times.append(float(t_setup_prob))
        solve_times.append(float(t_solve))
        iters_last = int(iters)
        reason_last = int(reason)

    t_setup_med = float(np.median(np.asarray(setup_times, dtype=float)))
    t_solve_med = float(np.median(np.asarray(solve_times, dtype=float)))

    cells_per_rank_actual = float(cells_global) / float(size)
    dofs_per_rank_actual = float(dofs_global) / float(size)

    result = BenchmarkResult(
        timestamp_utc=_utc_now_str(),
        problem=str(args.problem),
        pdeg=int(args.pdeg),
        ksp_type=str(args.ksp_type),
        pc_type=str(args.pc_type),
        ranks=int(size),
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        cells_global=int(cells_global),
        dofs_global=int(dofs_global),
        cells_per_rank_target=int(args.cells_per_rank),
        cells_per_rank_actual=cells_per_rank_actual,
        dofs_per_rank_actual=dofs_per_rank_actual,
        t_mesh_s=float(t_setup),
        t_problem_setup_s=float(t_setup_med),
        t_solve_s=float(t_solve_med),
        ksp_iters=int(iters_last),
        ksp_reason=int(reason_last),
    )

    if rank == 0:
        csv_path = outdir / "weak_scaling.csv"
        _write_csv_row(csv_path, asdict(result))

        meta_path = outdir / "weak_scaling_meta.json"
        meta = {
            "created_utc": _utc_now_str(),
            "dolfinx_version": getattr(dolfinx, "__version__", None),
            "petsc_version": PETSc.Sys.getVersion(),
            "python": {
                "executable": sys.executable,
                "version": sys.version,
            },
            "note": "CSV rows are appended; filter by (problem,pdeg,ksp_type,pc_type) as needed.",
            "petsc_options_hint": (
                "You can override PETSc options via environment variables or CLI, e.g. "
                "`-ksp_view -ksp_converged_reason -log_view` passed through mpirun."
            ),
        }
        # Avoid clobbering metadata from previous runs: keep the first file unless absent.
        if not meta_path.exists():
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)

        print(
            f"[weak_scaling] problem={result.problem} ranks={result.ranks} "
            f"cells={result.cells_global} dofs={result.dofs_global} "
            f"t_solve={result.t_solve_s:.3f}s (median over {len(solve_times)} repeats)"
        )


if __name__ == "__main__":
    main()
