import numpy as np
import pytest
import basix
import ufl
from types import SimpleNamespace
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type

from simulation.homogenizator import KUBCHomogenizer, SUBCHomogenizer

def _make_config(domain, E, nu):
    """Create minimal config for homogenization tests (MPa stress units)."""
    return SimpleNamespace(
        domain=domain,
        smooth_eps=1e-8,
        dx=ufl.Measure("dx", domain=domain),
        ds=ufl.Measure("ds", domain=domain),
        rho_min=1e-9,  # kg/m³
        E0=E,  # MPa
        E0_c=E,  # MPa (constitutive)
        n_power=1.0,
        n_power_c=1.0,
        nu=nu,
        nu_c=nu,
        ksp_type="cg",
        pc_type="gamg",
        ksp_rtol=1e-12,
        ksp_atol=1e-14,
        ksp_max_it=500,
        nitsche_alpha=200.0,
        nitsche_theta=1.0,
        E_ref=E,
        verbose=False,
    )

def _make_fields(domain):
    gdim = domain.geometry.dim
    if gdim != 3:
        pytest.skip("Tests defined for 3D only")
    # Density field (scalar)
    V_rho = fem.functionspace(domain, ("P", 1))
    rho = fem.Function(V_rho)
    rho.x.array[:] = 1.0
    return rho

def _make_fields_custom(domain, rho_value: float):
    """Construct constant rho field with provided values."""
    gdim = domain.geometry.dim
    # rho
    V_rho = fem.functionspace(domain, ("P", 1))
    rho = fem.Function(V_rho)
    rho.x.array[:] = rho_value
    return rho

def _theoretical_C(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] = lam + 2 * mu
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C

def _build_mesh():
    return mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)

def _assert_C_symmetric(C, *, atol=1e-10):
    assert C.shape == (6, 6)
    assert np.allclose(C, C.T, atol=atol)

def _run_homogenizer(h_cls, rho, cfg, *, degree=1, symmetric_atol=1e-10, **run_kwargs):
    out = h_cls(rho, cfg, degree=degree).run(**run_kwargs)
    _assert_C_symmetric(out["C_voigt"], atol=symmetric_atol)
    return out

def _assert_rank_consistency(C_local, *, atol=1e-10, rtol=1e-10):
    comm = MPI.COMM_WORLD
    C_root = comm.bcast(C_local if comm.rank == 0 else None, root=0)
    assert np.allclose(C_local, C_root, atol=atol, rtol=rtol)

# MPI-marked isotropy test
@pytest.mark.mpi
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("method", ["KUBC", "SUBC"])
def test_homogenizer_isotropic(n, method):
    """Test isotropic material homogenization for KUBC and SUBC methods."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    E, nu = (10.0, 0.25) if method == "KUBC" else (15.0, 0.3)
    cfg = _make_config(domain, E, nu)
    rho = _make_fields(domain)
    
    HomogenizerClass = KUBCHomogenizer if method == "KUBC" else SUBCHomogenizer
    kwargs = {"eps_mag": 1e-4} if method == "KUBC" else {"sigma_mag": 1.0}
    
    out = _run_homogenizer(HomogenizerClass, rho, cfg, **kwargs)
    C_num = out["C_voigt"]
    C_th = _theoretical_C(E, nu)
    assert np.allclose(C_num, C_th, rtol=5e-4, atol=5e-6)


@pytest.mark.mpi
@pytest.mark.parametrize("n", [3, 4])
def test_kubc_rank_consistency(n):
    """Each MPI rank must return identical C tensor."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    cfg = _make_config(domain, 12.3, 0.27)
    rho = _make_fields(domain)
    C_local = _run_homogenizer(KUBCHomogenizer, rho, cfg, eps_mag=5e-4)["C_voigt"]
    _assert_rank_consistency(C_local)

@pytest.mark.mpi
@pytest.mark.parametrize("n", [3, 4])
def test_subc_rank_consistency(n):
    """Each MPI rank must return identical C tensor."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    cfg = _make_config(domain, 9.7, 0.22)
    rho = _make_fields(domain)
    C_local = _run_homogenizer(SUBCHomogenizer, rho, cfg, sigma_mag=1.0)["C_voigt"]
    _assert_rank_consistency(C_local)

@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
@pytest.mark.parametrize("rho_lo_val", [0.2, 0.1])
def test_density_scaling_kubc(unit_cube, rho_lo_val):
    domain = unit_cube
    E, nu = 9.0, 0.28
    cfg_hi = _make_config(domain, E, nu)
    cfg_lo = cfg_hi

    rho_hi = _make_fields(domain)  # rho=1.0
    # rho low
    V_rho = rho_hi.function_space
    rho_lo = fem.Function(V_rho)
    rho_lo.x.array[:] = rho_lo_val

    C_hi = _run_homogenizer(KUBCHomogenizer, rho_hi, cfg_hi, eps_mag=1e-4)["C_voigt"]
    C_lo = _run_homogenizer(KUBCHomogenizer, rho_lo, cfg_lo, eps_mag=1e-4)["C_voigt"]

    ratio = C_lo[0, 0] / C_hi[0, 0]
    # Expect near linear scaling with rho when n_power=1
    assert np.isclose(ratio, rho_lo_val, rtol=6e-2, atol=1e-2)
    # Shear scales similarly
    assert np.isclose(C_lo[3, 3] / C_hi[3, 3], rho_lo_val, rtol=6e-2, atol=1e-2)


@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
@pytest.mark.parametrize("rho_lo_val", [0.1, 0.2])
def test_density_scaling_subc(unit_cube, rho_lo_val):
    domain = unit_cube
    E, nu = 11.0, 0.24
    cfg = _make_config(domain, E, nu)
    rho_hi = _make_fields(domain)  # rho=1.0
    V_rho = rho_hi.function_space
    rho_lo = fem.Function(V_rho)
    rho_lo.x.array[:] = rho_lo_val

    C_hi = _run_homogenizer(SUBCHomogenizer, rho_hi, cfg, sigma_mag=1.0)["C_voigt"]
    C_lo = _run_homogenizer(SUBCHomogenizer, rho_lo, cfg, sigma_mag=1.0)["C_voigt"]
    ratio = C_lo[0, 0] / C_hi[0, 0]
    assert np.isclose(ratio, rho_lo_val, rtol=6e-2, atol=1e-2)
    assert np.isclose(C_lo[3, 3] / C_hi[3, 3], rho_lo_val, rtol=6e-2, atol=1e-2)

# ---------------- Analytic solution for uniform media ------------------------

def _voigt_from_tensor(C4: np.ndarray) -> np.ndarray:
    """Map 4th-order tensor C_ijkl to 6x6 Voigt consistent with solver mapping."""
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    Cv = np.zeros((6, 6), dtype=float)
    for I, (i, j) in enumerate(idx):
        for J, (k, l) in enumerate(idx):
            if i == j:
                row_val_kl = C4[i, j, k, l]
                row_val_lk = C4[i, j, l, k]
            else:
                row_val_kl = 0.5 * (C4[i, j, k, l] + C4[j, i, k, l])
                row_val_lk = 0.5 * (C4[i, j, l, k] + C4[j, i, l, k])
            Cv[I, J] = row_val_kl if k == l else 0.5 * (row_val_kl + row_val_lk)
    return Cv


def _analytical_C_voigt_uniform(E0: float, nu: float, rho: float, n: float) -> np.ndarray:
    """Closed-form C_voigt for uniform fields used by the solver."""
    gdim = 3
    I = np.eye(gdim)
    # Smooth density clamp (rho >> rho_min in tests)
    rho_eff = rho
    E_eff = E0 * (rho_eff ** n)
    lam = E_eff * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E_eff / (2 * (1 + nu))

    # Isotropic elasticity tensor
    C_iso = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C_iso[i, j, k, l] = lam * I[i, j] * I[k, l] + mu * (
                        I[i, k] * I[j, l] + I[i, l] * I[j, k]
                    )

    return _voigt_from_tensor(C_iso)


@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
def test_kubc_uniform_analytic_matches(unit_cube):
    """Uniform medium should match the closed-form C (within FE/Nitsche tolerance)."""
    domain = unit_cube
    E, nu, n = 14.0, 0.27, 1.0
    cfg = _make_config(domain, E, nu)
    rho = _make_fields_custom(domain, rho_value=1.0)

    out = _run_homogenizer(KUBCHomogenizer, rho, cfg, symmetric_atol=1e-12, eps_mag=1e-4)
    C_num = out["C_voigt"]
    C_ref = _analytical_C_voigt_uniform(E, nu, rho=1.0, n=n)

    assert C_num.shape == (6, 6)
    assert np.allclose(C_num, C_num.T, atol=1e-12)
    assert np.allclose(C_num, C_ref, rtol=1e-3, atol=2e-5)


@pytest.mark.mpi
def test_subc_uniform_analytic_matches():
    domain = _build_mesh()
    E, nu, n = 9.0, 0.23, 1.0
    cfg = _make_config(domain, E, nu)
    rho = _make_fields_custom(domain, rho_value=1.0)

    out = _run_homogenizer(SUBCHomogenizer, rho, cfg, symmetric_atol=1e-12, sigma_mag=1.0)
    C_num = out["C_voigt"]
    C_ref = _analytical_C_voigt_uniform(E, nu, rho=1.0, n=n)

    assert C_num.shape == (6, 6)
    assert np.allclose(C_num, C_num.T, atol=1e-12)
    assert np.allclose(C_num, C_ref, rtol=2e-3, atol=5e-5)
