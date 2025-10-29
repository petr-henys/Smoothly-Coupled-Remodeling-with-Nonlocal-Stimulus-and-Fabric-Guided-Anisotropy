import numpy as np
import pytest


@pytest.mark.mpi
@pytest.mark.unit
def test_qoi_dirichlet_energy_scalar_richardson():
    """Use Dirichlet energy of a smooth scalar field as QoI.

    Q(h) = 1/2 ∫ |∇f_h|^2 dx converges to a limit with mesh refinement.
    We apply Richardson/GCI on Q(h) across uniform refinements and assert
    sensible p > 0, decreasing GCIs, and ratio consistency.
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    import ufl
    from analysis.utils import (
        mpi_scalar_integral,
        compute_richardson_triplets_qoi,
    )

    comm = MPI.COMM_WORLD
    # Uniform-ish ratio to fit solver assumptions (r ≈ 1.5)
    N_list = [8, 12, 18, 27]  # 4 levels → 2 triplets

    def make_field(N):
        domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        Q = fem.functionspace(domain, P1)
        f = fem.Function(Q, name="f")
        # Quadratic manufactured solution (nontrivial gradient)
        f.interpolate(
            lambda x: (
                x[0] * x[0]
                - 0.5 * x[1] * x[1]
                + 0.75 * x[2] * x[2]
                + 0.3 * x[0] * x[1]
                + 0.2 * x[1] * x[2]
                + 0.15 * x[0] * x[2]
            )
        )
        f.x.scatter_forward()
        return domain, f

    domains, fields = zip(*(make_field(N) for N in N_list))
    h_values = 1.0 / np.asarray(N_list, dtype=float)

    # QoI: Dirichlet energy 0.5 * ∫ |∇f_h|^2
    Q_vals = []
    for dom, f in zip(domains, fields):
        e_density = 0.5 * ufl.inner(ufl.grad(f), ufl.grad(f))
        Q_vals.append(mpi_scalar_integral(e_density, dom))

    rows = compute_richardson_triplets_qoi(h_values.tolist(), Q_vals)
    assert len(rows) == len(N_list) - 2

    for i, row in enumerate(rows):
        assert np.isfinite(row["p"]) and row["p"] > 0.5 and row["p"] < 3.5
        # Finer level should be closer to Q_ext → smaller GCI32
        assert row["GCI32_percent"] <= row["GCI21_percent"] + 1e-8
        # Consistency: GCI32/GCI21 ≈ (h3/h2)^p = (1/r32)^p
        ratio = row["GCI32"] / row["GCI21"] if row["GCI21"] > 0 else np.nan
        expected = (1.0 / row["r32"]) ** row["p"] if row["r32"] > 0 else np.nan
        if np.isfinite(ratio) and np.isfinite(expected):
            assert abs(ratio - expected) / max(1e-12, expected) < 0.25


@pytest.mark.mpi
@pytest.mark.unit
def test_qoi_dirichlet_energy_vector_richardson():
    """Use Dirichlet energy of a smooth vector field as QoI.

    Q(h) = 1/2 ∫ |∇u_h|^2 dx with u quadratic in each component.
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    import ufl
    from analysis.utils import (
        mpi_scalar_integral,
        compute_richardson_triplets_qoi,
    )

    comm = MPI.COMM_WORLD
    N_list = [8, 12, 18, 27]

    def make_field(N):
        domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
        V = fem.functionspace(
            domain, basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
        )
        u = fem.Function(V, name="u")
        u.interpolate(lambda x: np.vstack([x[0] ** 2, 0.5 * x[1] ** 2, 0.25 * x[2] ** 2]))
        u.x.scatter_forward()
        return domain, u

    domains, fields = zip(*(make_field(N) for N in N_list))
    h_values = 1.0 / np.asarray(N_list, dtype=float)

    Q_vals = []
    for dom, u in zip(domains, fields):
        e_density = 0.5 * ufl.inner(ufl.grad(u), ufl.grad(u))
        Q_vals.append(mpi_scalar_integral(e_density, dom))

    rows = compute_richardson_triplets_qoi(h_values.tolist(), Q_vals)
    assert len(rows) == len(N_list) - 2

    for row in rows:
        assert np.isfinite(row["p"]) and row["p"] > 0.5 and row["p"] < 3.5
        assert row["GCI32_percent"] <= row["GCI21_percent"] + 1e-8
        ratio = row["GCI32"] / row["GCI21"] if row["GCI21"] > 0 else np.nan
        expected = (1.0 / row["r32"]) ** row["p"] if row["r32"] > 0 else np.nan
        if np.isfinite(ratio) and np.isfinite(expected):
            assert abs(ratio - expected) / max(1e-12, expected) < 0.25


@pytest.mark.mpi
@pytest.mark.unit
def test_qoi_gradient_norm_scalar_richardson():
    """Use L2 norm of gradient as QoI (not variance).

    Q(h) = ∫ |∇f_h|^2 dx behaves similarly to energy and is meaningful for GCI.
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    import ufl
    from analysis.utils import (
        mpi_scalar_integral,
        compute_richardson_triplets_qoi,
    )

    comm = MPI.COMM_WORLD
    N_list = [8, 12, 18, 27]

    def make_field(N):
        domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        Q = fem.functionspace(domain, P1)
        f = fem.Function(Q, name="f")
        f.interpolate(lambda x: np.sin(1.3 * np.pi * x[0]) * np.cos(0.7 * np.pi * x[1]) + 0.2 * x[2])
        f.x.scatter_forward()
        return domain, f

    domains, fields = zip(*(make_field(N) for N in N_list))
    h_values = 1.0 / np.asarray(N_list, dtype=float)

    Q_vals = []
    for dom, f in zip(domains, fields):
        Q_vals.append(mpi_scalar_integral(ufl.inner(ufl.grad(f), ufl.grad(f)), dom))

    rows = compute_richardson_triplets_qoi(h_values.tolist(), Q_vals)
    assert len(rows) == len(N_list) - 2

    for row in rows:
        assert np.isfinite(row["p"]) and row["p"] > 0.5
        assert row["GCI32_percent"] <= row["GCI21_percent"] + 1e-8
        ratio = row["GCI32"] / row["GCI21"] if row["GCI21"] > 0 else np.nan
        expected = (1.0 / row["r32"]) ** row["p"] if row["r32"] > 0 else np.nan
        if np.isfinite(ratio) and np.isfinite(expected):
            assert abs(ratio - expected) / max(1e-12, expected) < 0.35

