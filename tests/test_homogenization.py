import numpy as np
import pytest
import basix
import ufl
from types import SimpleNamespace
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from petsc4py import PETSc

from simulation.homogenizator import KUBCHomogenizer, SUBCHomogenizer

def _make_config(domain, E, nu, xi=0.0):
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
        xi_aniso=xi,
        xi_aniso_c=xi,
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
    # A_dir (second-order tensor, identity)
    element_A = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim, gdim))  # Fixed
    V_A = fem.functionspace(domain, element_A)
    A_dir = fem.Function(V_A)
    def _id(x):
        n = x.shape[1]
        data = np.zeros((gdim * gdim, n), dtype=default_scalar_type)
        for i in range(gdim):
            data[i * gdim + i] = 1.0
        return data
    A_dir.interpolate(_id)
    return rho, A_dir

def _make_fields_custom(domain, rho_value: float, A_const: np.ndarray):
    """Construct constant rho and A_dir fields with provided values."""
    gdim = domain.geometry.dim
    # rho
    V_rho = fem.functionspace(domain, ("P", 1))
    rho = fem.Function(V_rho)
    rho.x.array[:] = rho_value
    # A_dir with constant tensor A_const
    element_A = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim, gdim))
    V_A = fem.functionspace(domain, element_A)
    A_dir = fem.Function(V_A)
    A_const = np.asarray(A_const, dtype=float)
    assert A_const.shape == (gdim, gdim)
    def _const(x):
        n = x.shape[1]
        data = np.zeros((gdim * gdim, n), dtype=default_scalar_type)
        for i in range(gdim):
            for j in range(gdim):
                data[i * gdim + j] = A_const[i, j]
        return data
    A_dir.interpolate(_const)
    return rho, A_dir

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

def _run_homogenizer(h_cls, rho, A_dir, cfg, *, degree=1, symmetric_atol=1e-10, **run_kwargs):
    out = h_cls(rho, A_dir, cfg, degree=degree).run(**run_kwargs)
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
    rho, A_dir = _make_fields(domain)
    
    HomogenizerClass = KUBCHomogenizer if method == "KUBC" else SUBCHomogenizer
    kwargs = {"eps_mag": 1e-4} if method == "KUBC" else {"sigma_mag": 1.0}
    
    out = _run_homogenizer(HomogenizerClass, rho, A_dir, cfg, **kwargs)
    C_num = out["C_voigt"]
    C_th = _theoretical_C(E, nu)
    assert np.allclose(C_num, C_th, rtol=5e-4, atol=5e-6)


@pytest.mark.mpi
@pytest.mark.parametrize("n", [3, 4])
def test_kubc_rank_consistency(n):
    """Each MPI rank must return identical C tensor."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    cfg = _make_config(domain, 12.3, 0.27)
    rho, A_dir = _make_fields(domain)
    C_local = _run_homogenizer(KUBCHomogenizer, rho, A_dir, cfg, eps_mag=5e-4)["C_voigt"]
    _assert_rank_consistency(C_local)

@pytest.mark.mpi
@pytest.mark.parametrize("n", [3, 4])
def test_subc_rank_consistency(n):
    """Each MPI rank must return identical C tensor."""
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    cfg = _make_config(domain, 9.7, 0.22)
    rho, A_dir = _make_fields(domain)
    C_local = _run_homogenizer(SUBCHomogenizer, rho, A_dir, cfg, sigma_mag=1.0)["C_voigt"]
    _assert_rank_consistency(C_local)

# ---------------- Additional tests: anisotropy and contrast ------------------

@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
@pytest.mark.parametrize("xi", [0.4, 0.8, 1.2])
def test_kubc_anisotropy_axial_stiffening(unit_cube, xi):
    domain = unit_cube
    E, nu = 10.0, 0.25
    cfg = _make_config(domain, E, nu, xi=xi)
    # A_dir aligned with e1 (principal direction x)
    A = np.diag([1.0, 0.0, 0.0])
    rho, A_dir = _make_fields_custom(domain, rho_value=1.0, A_const=A)
    C_th_iso = _theoretical_C(E, nu)

    C = _run_homogenizer(KUBCHomogenizer, rho, A_dir, cfg, eps_mag=1e-4)["C_voigt"]

    # Expect C11 increased by ~ xi*E; other entries ~ isotropic
    assert C.shape == (6, 6)
    assert np.allclose(C, C.T, atol=1e-10)
    assert np.isclose(C[0, 0], C_th_iso[0, 0] + xi * E, rtol=2e-2, atol=5e-4)
    # Unaffected components within tolerance
    for i in (1, 2):
        assert np.isclose(C[i, i], C_th_iso[i, i], rtol=5e-3, atol=5e-6)
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        assert np.isclose(C[i, j], C_th_iso[i, j], rtol=5e-3, atol=5e-6)
    for k in (3, 4, 5):
        assert np.isclose(C[k, k], C_th_iso[k, k], rtol=5e-3, atol=5e-6)


@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
@pytest.mark.parametrize("xi", [0.3, 0.6, 1.0])
def test_subc_anisotropy_axial_stiffening(unit_cube, xi):
    domain = unit_cube
    E, nu = 12.0, 0.22
    cfg = _make_config(domain, E, nu, xi=xi)
    A = np.diag([1.0, 0.0, 0.0])
    rho, A_dir = _make_fields_custom(domain, rho_value=1.0, A_const=A)
    C_th_iso = _theoretical_C(E, nu)

    C = _run_homogenizer(SUBCHomogenizer, rho, A_dir, cfg, sigma_mag=1.0)["C_voigt"]

    assert C.shape == (6, 6)
    assert np.allclose(C, C.T, atol=1e-10)
    assert np.isclose(C[0, 0], C_th_iso[0, 0] + xi * E, rtol=3e-2, atol=1e-3)
    for i in (1, 2):
        assert np.isclose(C[i, i], C_th_iso[i, i], rtol=1e-2, atol=1e-5)
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        assert np.isclose(C[i, j], C_th_iso[i, j], rtol=1e-2, atol=1e-5)
    for k in (3, 4, 5):
        assert np.isclose(C[k, k], C_th_iso[k, k], rtol=1e-2, atol=1e-5)


@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
@pytest.mark.parametrize("rho_lo_val", [0.2, 0.1])
def test_density_scaling_kubc(unit_cube, rho_lo_val):
    domain = unit_cube
    E, nu = 9.0, 0.28
    cfg_hi = _make_config(domain, E, nu, xi=0.0)
    cfg_lo = cfg_hi  # same config (anisotropy off)

    rho_hi, A_dir = _make_fields(domain)  # rho=1.0
    # rho low
    V_rho = rho_hi.function_space
    rho_lo = fem.Function(V_rho)
    rho_lo.x.array[:] = rho_lo_val

    C_hi = _run_homogenizer(KUBCHomogenizer, rho_hi, A_dir, cfg_hi, eps_mag=1e-4)["C_voigt"]
    C_lo = _run_homogenizer(KUBCHomogenizer, rho_lo, A_dir, cfg_lo, eps_mag=1e-4)["C_voigt"]

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
    cfg = _make_config(domain, E, nu, xi=0.0)
    rho_hi, A_dir = _make_fields(domain)  # rho=1.0
    V_rho = rho_hi.function_space
    rho_lo = fem.Function(V_rho)
    rho_lo.x.array[:] = rho_lo_val

    C_hi = _run_homogenizer(SUBCHomogenizer, rho_hi, A_dir, cfg, sigma_mag=1.0)["C_voigt"]
    C_lo = _run_homogenizer(SUBCHomogenizer, rho_lo, A_dir, cfg, sigma_mag=1.0)["C_voigt"]
    ratio = C_lo[0, 0] / C_hi[0, 0]
    assert np.isclose(ratio, rho_lo_val, rtol=6e-2, atol=1e-2)
    assert np.isclose(C_lo[3, 3] / C_hi[3, 3], rho_lo_val, rtol=6e-2, atol=1e-2)


@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
@pytest.mark.parametrize("theta_deg", [30.0, 45.0])
def test_kubc_anisotropy_rotation_alignment(unit_cube, theta_deg):
    domain = unit_cube
    E, nu, xi = 10.0, 0.25, 0.7
    cfg = _make_config(domain, E, nu, xi=xi)
    # Rotate projector diag(1,0,0) by theta about z
    theta = np.deg2rad(theta_deg)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                   [np.sin(theta),  np.cos(theta), 0.0],
                   [0.0,           0.0,           1.0]], dtype=float)
    A0 = np.diag([1.0, 0.0, 0.0])
    A = Rz @ A0 @ Rz.T
    rho, A_dir = _make_fields_custom(domain, rho_value=1.0, A_const=A)

    out = _run_homogenizer(KUBCHomogenizer, rho, A_dir, cfg, eps_mag=1e-4)
    R_fab = out["R_fabric"]
    E_pr = out["E_principal"]
    # First principal direction should align with rotated x-axis (up to sign)
    v_model = R_fab[:, 0]
    v_true = Rz[:, 0]
    align = abs(np.dot(v_model, v_true))
    assert align > 0.98
    # Anisotropy present: E1 > E2 ~ E3 (within loose tolerance)
    assert E_pr[0] > E_pr[1] * 1.05
    assert E_pr[0] > E_pr[2] * 1.05


@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
def test_anisotropy_contrast_effects_kubc(unit_cube):
    domain = unit_cube
    E, nu = 10.0, 0.25
    A = np.diag([1.0, 0.0, 0.0])
    rho, A_dir = _make_fields_custom(domain, rho_value=1.0, A_const=A)
    out_low = _run_homogenizer(KUBCHomogenizer, rho, A_dir, _make_config(domain, E, nu, xi=0.05), eps_mag=2e-4)
    out_high = _run_homogenizer(KUBCHomogenizer, rho, A_dir, _make_config(domain, E, nu, xi=1.5), eps_mag=2e-4)
    # Principal modulus along fabric grows with xi
    assert out_high["E_principal"][0] > out_low["E_principal"][0] * 1.4
    # Anisotropy ratio grows
    assert out_high["E_ratio"] > out_low["E_ratio"] * 1.2


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


def _analytical_C_voigt_uniform(E0: float, nu: float, xi: float, rho: float, n: float, A_dir: np.ndarray, smooth_eps: float) -> np.ndarray:
    """Closed-form C_voigt for uniform fields used by the solver.

    Uses the same Ahat construction as the code: Ahat = (A^T A + eps I) / tr(A^T A + eps I).
    """
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

    # Ahat consistent with solver
    A = np.asarray(A_dir, dtype=float)
    M = (A.T @ A) + smooth_eps * I
    Ahat = M / np.trace(M)

    # Anisotropic rank-one 4th-order addition: xi * E_eff * Ahat_ij * Ahat_kl
    C_aniso = np.einsum("ij,kl->ijkl", Ahat, Ahat, optimize=True) * (xi * E_eff)

    C4 = C_iso + C_aniso
    return _voigt_from_tensor(C4)


@pytest.mark.mpi
@pytest.mark.parametrize("unit_cube", [3, 4], indirect=True)
def test_kubc_uniform_analytic_matches(unit_cube):
    """Uniform medium should match the closed-form C (within FE/Nitsche tolerance)."""
    domain = unit_cube
    E, nu, xi, n = 14.0, 0.27, 0.9, 1.0
    cfg = _make_config(domain, E, nu, xi=xi)
    # Rotated anisotropy to exercise off-diagonals in Voigt
    theta = np.deg2rad(30.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                   [np.sin(theta),  np.cos(theta), 0.0],
                   [0.0,           0.0,           1.0]], dtype=float)
    A0 = np.diag([1.0, 0.0, 0.0])
    A = Rz @ A0 @ Rz.T
    rho, A_dir = _make_fields_custom(domain, rho_value=1.0, A_const=A)

    out = _run_homogenizer(KUBCHomogenizer, rho, A_dir, cfg, symmetric_atol=1e-12, eps_mag=1e-4)
    C_num = out["C_voigt"]
    C_ref = _analytical_C_voigt_uniform(E, nu, xi, rho=1.0, n=n, A_dir=A, smooth_eps=cfg.smooth_eps)

    assert C_num.shape == (6, 6)
    assert np.allclose(C_num, C_num.T, atol=1e-12)
    assert np.allclose(C_num, C_ref, rtol=1e-3, atol=2e-5)


@pytest.mark.mpi
def test_subc_uniform_analytic_matches():
    domain = _build_mesh()
    E, nu, xi, n = 9.0, 0.23, 0.5, 1.0
    cfg = _make_config(domain, E, nu, xi=xi)
    A = np.diag([1.0, 0.0, 0.0])
    rho, A_dir = _make_fields_custom(domain, rho_value=1.0, A_const=A)

    out = _run_homogenizer(SUBCHomogenizer, rho, A_dir, cfg, symmetric_atol=1e-12, sigma_mag=1.0)
    C_num = out["C_voigt"]
    C_ref = _analytical_C_voigt_uniform(E, nu, xi, rho=1.0, n=n, A_dir=A, smooth_eps=cfg.smooth_eps)

    assert C_num.shape == (6, 6)
    assert np.allclose(C_num, C_num.T, atol=1e-12)
    assert np.allclose(C_num, C_ref, rtol=2e-3, atol=5e-5)
