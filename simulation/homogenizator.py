from __future__ import annotations

"""Homogenization: KUBC (Dirichlet/Nitsche) and SUBC (traction-controlled) elasticity solvers."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import basix
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, default_scalar_type

from simulation.config import Config
from simulation.subsolvers import smooth_max, unittrace_psd_from_any
from simulation.utils import build_nullspace
from simulation.logger import get_logger


Scalar = PETSc.ScalarType

__all__ = ["KUBCHomogenizer", "SUBCHomogenizer"]


@dataclass
class _Elasticity:
    """Constitutive helper: anisotropic elasticity with density and fabric modulation."""
    V: fem.FunctionSpace
    rho: fem.Function
    A_dir: fem.Function
    cfg: Config

    def __post_init__(self):
        self.gdim = self.V.mesh.geometry.dim
        self.smooth_eps = float(self.cfg.smooth_eps)

        self.dx = self.cfg.dx
        self.ds = self.cfg.ds

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        self.a_base = ufl.inner(self.sigma(self.u, self.rho), self.eps(self.v)) * self.dx
        self.L_zero = (
            ufl.inner(
                fem.Constant(self.V.mesh, default_scalar_type((0.0,) * self.gdim)),
                self.v,
            )
            * self.ds
        )

    def eps(self, u):
        """Symmetric gradient ε(u)."""
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho):
        """Cauchy stress: smoothed density, anisotropic fabric reinforcement."""
        rho_eff = smooth_max(rho, self.cfg.rho_min, self.smooth_eps)
        E = self.cfg.E0_c * (rho_eff ** self.cfg.n_power_c)

        eps_ten = self.eps(u)
        I = ufl.Identity(self.gdim)
        nu = self.cfg.nu_c

        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        Asym = 0.5 * (self.A_dir + ufl.transpose(self.A_dir))
        Ahat = unittrace_psd_from_any(Asym, self.gdim, self.smooth_eps)

        # Anisotropic projected contribution
        sigma_aniso = (
            self.cfg.xi_aniso_c * E
        ) * ufl.inner(Ahat, eps_ten) * Ahat

        return 2 * mu * eps_ten + lmbda * ufl.tr(eps_ten) * I + sigma_aniso

    def sigma_const_strain(self, Eps_np: np.ndarray):
        """Stress field for constant macroscopic strain tensor Ē."""
        rho_eff = smooth_max(self.rho, self.cfg.rho_min, self.smooth_eps)
        E = self.cfg.E0_c * (rho_eff ** self.cfg.n_power_c)

        I = ufl.Identity(self.gdim)
        nu = self.cfg.nu_c
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        Asym = 0.5 * (self.A_dir + ufl.transpose(self.A_dir))
        Ahat = unittrace_psd_from_any(Asym, self.gdim, self.smooth_eps)

        Eps = ufl.as_tensor(np.asarray(Eps_np, dtype=float).tolist())
        sigma_aniso = (self.cfg.xi_aniso_c * E) * ufl.inner(Ahat, Eps) * Ahat
        return 2 * mu * Eps + lmbda * ufl.tr(Eps) * I + sigma_aniso


class _HomogCommon:
    """Shared helpers for Voigt notation, averaging, SPD inverse, KSP setup."""
    _VOIGT_IDX = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    _SHEAR_PAIRS = ((1, 2), (0, 2), (0, 1))

    def __init__(
        self,
        rho: fem.Function,
        A_dir: fem.Function,
        cfg: Config,
        degree: int = 1,
    ):
        self.cfg = cfg
        self.domain = rho.function_space.mesh
        assert self.domain is A_dir.function_space.mesh and self.domain is cfg.domain

        self.gdim = self.domain.geometry.dim
        if self.gdim != 3:
            raise NotImplementedError("Only 3D homogenization is supported.")

        Pk_vec = basix.ufl.element(
            "Lagrange",
            self.domain.basix_cell(),
            degree,
            shape=(self.gdim,),
        )
        self.V = fem.functionspace(self.domain, Pk_vec)

        self.rho = rho
        self.A_dir = A_dir
        self.dx = cfg.dx
        self.ds = cfg.ds

        self.mech = _Elasticity(self.V, rho, A_dir, cfg)
        self._u = fem.Function(self.V, name="u")
        self._nullspace = build_nullspace(self.V)
        # Logger honoring cfg.verbose
        self._logger = get_logger(self.domain.comm, verbose=bool(getattr(self.cfg, "verbose", True)), name=self.__class__.__name__)

        one = fem.Constant(self.domain, Scalar(1.0))
        vol_local = fem.assemble_scalar(fem.form(one * self.dx))
        self.vol = self.domain.comm.allreduce(vol_local, op=MPI.SUM)
        self._inv_vol = 1.0 / max(self.vol, 1e-30)

    def _log(self, msg: str):
        # INFO-level message; warning/error messages are handled inline
        self._logger.info(msg)

    def _average_stress(self, u: fem.Function) -> np.ndarray:
        sigma = self.mech.sigma(u, self.rho)
        S = np.zeros((self.gdim, self.gdim), dtype=float)
        for i in range(self.gdim):
            for j in range(self.gdim):
                val_loc = fem.assemble_scalar(fem.form(sigma[i, j] * self.dx))
                S[i, j] = (
                    self.domain.comm.allreduce(val_loc, op=MPI.SUM)
                    * self._inv_vol
                )
        return S

    def _average_engineering_strain(self, u: fem.Function) -> np.ndarray:
        eps = ufl.sym(ufl.grad(u))
        vals = np.zeros(6, dtype=float)
        for i in range(3):
            val_loc = fem.assemble_scalar(fem.form(eps[i, i] * self.dx))
            vals[i] = (
                self.domain.comm.allreduce(val_loc, op=MPI.SUM)
                * self._inv_vol
            )
        for k, (i, j) in enumerate(self._SHEAR_PAIRS, start=3):
            val_loc = fem.assemble_scalar(
                fem.form((2.0 * eps[i, j]) * self.dx)
            )
            vals[k] = (
                self.domain.comm.allreduce(val_loc, op=MPI.SUM)
                * self._inv_vol
            )
        return vals

    @staticmethod
    def _stress_tensor_to_voigt(sig: np.ndarray) -> np.ndarray:
        return np.array(
            [sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1]],
            dtype=float,
        )

    @staticmethod
    def _voigt_to_tensor(Cv: np.ndarray) -> np.ndarray:
        C = np.zeros((3, 3, 3, 3), dtype=float)
        idx = _HomogCommon._VOIGT_IDX
        for I, (i, j) in enumerate(idx):
            row_pairs = {(i, j)} | ({(j, i)} if i != j else set())
            for J, (k, l) in enumerate(idx):
                col_pairs = {(k, l)} | ({(l, k)} if k != l else set())
                val = float(Cv[I, J])
                for p, q in row_pairs:
                    for r, s in col_pairs:
                        C[p, q, r, s] = val
        return C

    @staticmethod
    def _tensor_to_voigt(C: np.ndarray) -> np.ndarray:
        Cv = np.zeros((6, 6), dtype=float)
        idx = _HomogCommon._VOIGT_IDX
        for I, (i, j) in enumerate(idx):
            for J, (k, l) in enumerate(idx):
                if i == j:
                    row_val_kl = C[i, j, k, l]
                    row_val_lk = C[i, j, l, k]
                else:
                    row_val_kl = 0.5 * (C[i, j, k, l] + C[j, i, k, l])
                    row_val_lk = 0.5 * (C[i, j, l, k] + C[j, i, l, k])
                Cv[I, J] = (
                    row_val_kl
                    if k == l
                    else 0.5 * (row_val_kl + row_val_lk)
                )
        return Cv

    @staticmethod
    def _rotate_C_to_basis(Cv: np.ndarray, R: np.ndarray) -> np.ndarray:
        C = _HomogCommon._voigt_to_tensor(Cv)
        Rt = R.T
        Cp = np.einsum("pi,qj,rk,sl,ijkl->pqrs", Rt, Rt, Rt, Rt, C, optimize=True)
        return _HomogCommon._tensor_to_voigt(Cp)

    @staticmethod
    def _spd_inverse(
        M: np.ndarray,
        rel_tol: float = 1e-10,
        abs_tol: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Ms = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(Ms)
        tol = max(
            abs_tol,
            rel_tol * float(np.max(np.abs(w)) if w.size else 0.0),
        )
        w_clip = np.maximum(w, tol)
        M_spd = V @ np.diag(w_clip) @ V.T
        Minv = V @ np.diag(1.0 / w_clip) @ V.T
        return Minv, M_spd

    def _average_fabric_and_rotation(self) -> Tuple[np.ndarray, np.ndarray]:
        gdim = self.gdim
        Abar = np.zeros((gdim, gdim), dtype=float)
        for i in range(gdim):
            for j in range(gdim):
                val_local = fem.assemble_scalar(
                    fem.form(self.A_dir[i, j] * self.dx)
                )
                Abar[i, j] = (
                    self.domain.comm.allreduce(val_local, op=MPI.SUM)
                    * self._inv_vol
                )
        Abar = 0.5 * (Abar + Abar.T)
        vals, vecs = np.linalg.eigh(Abar)
        order = np.argsort(vals)[::-1]
        R = vecs[:, order]
        return Abar, R

    @staticmethod
    def _extract_principal_props(
        S_f: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        E = 1.0 / np.array([S_f[0, 0], S_f[1, 1], S_f[2, 2]], dtype=float)
        G = 1.0 / np.array([S_f[3, 3], S_f[4, 4], S_f[5, 5]], dtype=float)
        nu = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                if i != j:
                    nu[i, j] = -S_f[i, j] / S_f[i, i]
        E_ratio = float(np.max(E) / np.min(E))
        return E, G, nu, E_ratio

    def _finalize_output(self, C: np.ndarray, S_in: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Common post-processing from stiffness matrix to output dictionary.

        - Symmetrise C, compute compliance S (SPD inverse)
        - Scale to dimensional units using `sigma_c`
        - Rotate to fabric principal basis and extract principal properties
        """
        C = 0.5 * (C + C.T)
        if S_in is None:
            S, _ = self._spd_inverse(C)
        else:
            S = S_in

        sigma_c = float(getattr(self.cfg, "sigma_c", 1.0))
        C_dim = C * sigma_c
        S_dim = S / max(sigma_c, 1e-300)

        Abar, R = self._average_fabric_and_rotation()
        C_f = self._rotate_C_to_basis(C, R)
        S_f, _ = self._spd_inverse(C_f)

        E, G, nu, E_ratio = self._extract_principal_props(S_f)

        return {
            "C_voigt": C,
            "S_voigt": S,
            "C_voigt_dim": C_dim,
            "S_voigt_dim": S_dim,
            "C_voigt_fabric": C_f,
            "E_principal": E,
            "nu_principal": nu,
            "G_principal": G,
            "E_ratio": E_ratio,
            "R_fabric": R,
            "Abar": Abar,
        }

    def _create_ksp(self, prefix: str = "homog") -> PETSc.KSP:
        ksp = PETSc.KSP().create(self.domain.comm)
        ksp.setOptionsPrefix(prefix + "_")
        opts = PETSc.Options()
        opts[f"{prefix}_ksp_type"] = getattr(self.cfg, "ksp_type", "minres")
        opts[f"{prefix}_pc_type"] = getattr(self.cfg, "pc_type", "gamg")
        opts[f"{prefix}_ksp_rtol"] = getattr(self.cfg, "ksp_rtol", 1e-7)
        opts[f"{prefix}_ksp_atol"] = getattr(self.cfg, "ksp_atol", 1e-8)
        opts[f"{prefix}_ksp_max_it"] = getattr(self.cfg, "ksp_max_it", 200)
        ksp.setFromOptions()
        ksp.getPC().setReusePreconditioner(True)
        ksp.setInitialGuessNonzero(True)
        return ksp

    def _assemble_matrix(
        self,
        a_form: fem.Form,
        bcs: Optional[List[fem.DirichletBC]] = None,
    ) -> PETSc.Mat:
        A = fem.petsc.create_matrix(a_form)
        fem.petsc.assemble_matrix(A, a_form, bcs=bcs or [])
        A.assemble()
        return A

    def _assemble_vector(
        self,
        L_form: fem.Form,
        a_form: fem.Form,
        bcs: Optional[List[fem.DirichletBC]] = None,
    ) -> PETSc.Vec:
        b = fem.petsc.create_vector(self.V)
        fem.petsc.assemble_vector(b, L_form)
        fem.petsc.apply_lifting(b, [a_form], bcs=[bcs or []])
        b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs or [])
        b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        return b


class KUBCHomogenizer(_HomogCommon):
    """Kinematic Uniform BC: affine displacement via Nitsche method, extract C from ⟨σ⟩."""
    def __init__(
        self,
        rho: fem.Function,
        A_dir: fem.Function,
        cfg: Config,
        degree: int = 1,
    ):
        super().__init__(rho, A_dir, cfg, degree)

        self._facet_normal = ufl.FacetNormal(self.domain)
        self._cell_diam = ufl.CellDiameter(self.domain)

        self._uD = fem.Function(self.V, name="uD")
        self._Eps_tmp = np.zeros((3, 3), dtype=float)

        def _affine(x):
            return (self._Eps_tmp @ x).astype(default_scalar_type)

        self._affine = _affine

    def _forms_for_case(self, Eps: np.ndarray) -> Tuple[fem.Form, fem.Form]:
        a = self.mech.a_base
        L = self.mech.L_zero

        self._Eps_tmp[:, :] = Eps
        self._uD.interpolate(self._affine)

        n = self._facet_normal
        h = self._cell_diam

        alpha = float(getattr(self.cfg, "nitsche_alpha", 100.0))
        theta = float(getattr(self.cfg, "nitsche_theta", 1.0))
        # Local penalty scaling: gamma/h ~ alpha*(2*mu+lambda)/h (robust across heterogeneity)
        rho_eff = smooth_max(self.rho, self.cfg.rho_min, self.cfg.smooth_eps)
        E_dim = self.cfg.E0_c * (rho_eff ** self.cfg.n_power_c)
        nu = self.cfg.nu_c
        lmbda = E_dim * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E_dim / (2 * (1 + nu))
        Kpen = 2 * mu + lmbda
        gamma_over_h = (alpha * Kpen) / h

        u, v = self.mech.u, self.mech.v
        sigma_u = self.mech.sigma(u, self.rho)
        sigma_v = self.mech.sigma(v, self.rho)

        a_bnd = (
            - ufl.dot(ufl.dot(sigma_u, n), v) * self.ds
            - theta * ufl.dot(ufl.dot(sigma_v, n), u) * self.ds
            + gamma_over_h * ufl.dot(u, v) * self.ds
        )
        L_bnd = (
            - theta * ufl.dot(ufl.dot(sigma_v, n), self._uD) * self.ds
            + gamma_over_h * ufl.dot(self._uD, v) * self.ds
        )

        return fem.form(a + a_bnd), fem.form(L + L_bnd)

    @staticmethod
    def _canonical_strains(
        mag: float,
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        Z = np.zeros((3, 3), dtype=float)
        out: List[Tuple[str, np.ndarray, np.ndarray]] = []

        # Normal strains
        for i, name in enumerate(("e11", "e22", "e33")):
            E = Z.copy()
            E[i, i] = mag
            eng = np.zeros(6)
            eng[i] = mag
            out.append((name, E, eng))

        # Engineering shear strains
        for k, (name, (i, j)) in enumerate(
            zip(("g23", "g13", "g12"), _HomogCommon._SHEAR_PAIRS)
        ):
            E = Z.copy()
            E[i, j] = 0.5 * mag
            E[j, i] = 0.5 * mag
            eng = np.zeros(6)
            eng[3 + k] = mag
            out.append((name, E, eng))

        return out

    def run(self, eps_mag: float = 1e-3) -> Dict[str, np.ndarray]:
        self._log(f"[KUBC] starting homogenization (eps_mag = {eps_mag:g})")

        cases = self._canonical_strains(eps_mag)
        C = np.zeros((6, 6), dtype=float)

        for j, (name, Eps, eng) in enumerate(cases):
            self._log(f"[KUBC]   solving case {name}")

            a_form, L_form = self._forms_for_case(Eps)
            A = self._assemble_matrix(a_form)
            A.setNearNullSpace(self._nullspace)

            b = self._assemble_vector(L_form, a_form)

            ksp = self._create_ksp(prefix=f"kubc_{name}")
            ksp.setOperators(A)
            ksp.setUp()
            ksp.solve(b, self._u.x.petsc_vec)
            self._u.x.scatter_forward()

            reason = ksp.getConvergedReason()
            if reason < 0:
                self._logger.warning(f"KUBC case {name} failed to converge (reason {reason})")

            sig_bar = self._average_stress(self._u)
            svec = self._stress_tensor_to_voigt(sig_bar)
            sval = float(eng[np.nonzero(eng)][0])
            C[:, j] = svec / sval

            ksp.destroy()
            A.destroy()
            b.destroy()

        out = self._finalize_output(C)

        E = out["E_principal"]
        self._log(
            f"[KUBC] done → E_principal = {E[0]:.4g}, {E[1]:.4g}, {E[2]:.4g}"
        )

        return out


class SUBCHomogenizer(_HomogCommon):
    """Static Uniform BC: traction-controlled tests, extract S from ⟨ε⟩."""
    def __init__(
        self,
        rho: fem.Function,
        A_dir: fem.Function,
        cfg: Config,
        degree: int = 1,
    ):
        super().__init__(rho, A_dir, cfg, degree)
        self._A_bulk: Optional[PETSc.Mat] = None
        self._ksp: Optional[PETSc.KSP] = None
        self._a_form: Optional[fem.Form] = None

    def _ensure_bulk_system(self):
        if self._A_bulk is not None:
            return

        self._a_form = fem.form(self.mech.a_base)
        A = self._assemble_matrix(self._a_form)
        A.setNearNullSpace(self._nullspace)
        A.setNullSpace(self._nullspace)
        self._A_bulk = A

        ksp = self._create_ksp(prefix="subc")
        ksp.setOperators(self._A_bulk)
        ksp.setUp()
        self._ksp = ksp

    def _sigma_basis(self, mag: float) -> List[Tuple[str, np.ndarray]]:
        Z = np.zeros((3, 3), dtype=float)
        out: List[Tuple[str, np.ndarray]] = []

        for i, name in enumerate(("s11", "s22", "s33")):
            S = Z.copy()
            S[i, i] = mag
            out.append((name, S))

        for name, (i, j) in zip(("s23", "s13", "s12"), self._SHEAR_PAIRS):
            S = Z.copy()
            S[i, j] = mag
            S[j, i] = mag
            out.append((name, S))

        return out

    def _assemble_rhs_for_sigma(self, Sig_np: np.ndarray) -> fem.Form:
        n = ufl.FacetNormal(self.domain)
        Sig = ufl.as_tensor(Sig_np.tolist())
        t = ufl.dot(Sig, n)
        v = self.mech.v
        L_bnd = ufl.dot(t, v) * self.ds
        return fem.form(self.mech.L_zero + L_bnd)

    def run(self, sigma_mag: float = 1.0) -> Dict[str, np.ndarray]:
        self._log(f"[SUBC] starting homogenization (sigma_mag = {sigma_mag:g})")

        self._ensure_bulk_system()

        SIGMA = self._sigma_basis(sigma_mag)
        Ebar = np.zeros((6, 6), dtype=float)

        for j, (name, Sig) in enumerate(SIGMA):
            self._log(f"[SUBC]   solving case {name}")

            L_form = self._assemble_rhs_for_sigma(Sig)
            b = self._assemble_vector(L_form, self._a_form)

            # Project RHS to be consistent with singular operator
            self._nullspace.remove(b)

            self._ksp.solve(b, self._u.x.petsc_vec)
            self._u.x.scatter_forward()

            reason = self._ksp.getConvergedReason()
            if reason < 0:
                self._logger.warning(f"SUBC case {name} failed to converge (reason {reason})")

            Ebar[:, j] = self._average_engineering_strain(self._u)
            b.destroy()

        S = 0.5 * (Ebar + Ebar.T) / sigma_mag
        C, _ = self._spd_inverse(S)

        out = self._finalize_output(C, S_in=S)

        E = out["E_principal"]
        self._log(
            f"[SUBC] done → E_principal = {E[0]:.4g}, {E[1]:.4g}, {E[2]:.4g}"
        )

        return out
