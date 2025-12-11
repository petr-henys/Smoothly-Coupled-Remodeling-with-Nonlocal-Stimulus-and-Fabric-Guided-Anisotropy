"""MPI-parallel femur load interpolation with precomputed traction caching.

Provides:
- HipLoadSpec, MuscleLoadSpec: Load specifications
- LoadingCase: Configuration for one loading scenario (hip + muscles)
- Loader: Precomputes and caches traction fields for all loading cases

MPI pattern: Rank 0 owns pyvista geometry, computes loads; other ranks gather
coordinates to rank 0, receive scattered tractions back.

Two traction fields:
- traction: Hip + muscle loads on proximal surface (load_tag)
- traction_cut: Equilibrating reaction on distal cut (cut_tag)

Units: mm, N, MPa
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import dolfinx
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import basix.ufl
import ufl
from mpi4py import MPI

from simulation.femur_css import FemurCSS, load_json_points
from simulation.paths import FemurPaths
from simulation.femur_loads import HIPJointLoad, MuscleLoad, vector_from_angles


# =============================================================================
# Load Specifications (Dataclasses)
# =============================================================================

@dataclass
class HipLoadSpec:
    """Hip joint load: magnitude [N], angles [deg], Gaussian spread [deg]."""
    magnitude: float
    alpha_sag: float       # Sagittal angle (+ anterior)
    alpha_front: float     # Frontal angle (+ lateral)
    sigma_deg: float       # Contact patch size
    flip: bool             # Flip direction (compression into head)


@dataclass
class MuscleLoadSpec:
    """Muscle load: name, magnitude [N], angles [deg], Gaussian spread [mm]."""
    name: str              # Muscle ID (glmed, glmin, glmax, psoas, vastus_*)
    magnitude: float       # Force magnitude [N]
    alpha_sag: float       # Sagittal angle (+ anterior)
    alpha_front: float     # Frontal angle (+ lateral)
    sigma: float           # Attachment area size [mm]
    flip: bool             # Flip direction (contraction pulls)


@dataclass 
class LoadingCase:
    """A loading case = hip + muscles combination for a specific gait phase."""
    name: str
    weight: float
    hip: HipLoadSpec = field(default=None)
    muscles: List[MuscleLoadSpec] = field(default_factory=list)
    
    def set_hip(self, magnitude: float, alpha_sag: float, alpha_front: float, 
                sigma_deg: float, flip: bool) -> LoadingCase:
        """Set the hip joint load. Returns self for chaining."""
        self.hip = HipLoadSpec(
            magnitude=magnitude, 
            alpha_sag=alpha_sag, 
            alpha_front=alpha_front,
            sigma_deg=sigma_deg, 
            flip=flip
        )
        return self
    
    def add_muscle(
        self, 
        name: str, 
        magnitude: float, 
        alpha_sag: float, 
        alpha_front: float,
        sigma: float, 
        flip: bool
    ) -> LoadingCase:
        """Add a muscle load. Returns self for chaining."""
        self.muscles.append(MuscleLoadSpec(
            name=name, 
            magnitude=magnitude, 
            alpha_sag=alpha_sag, 
            alpha_front=alpha_front,
            sigma=sigma, 
            flip=flip
        ))
        return self


# Muscle name to JSON path mapping
MUSCLE_PATHS = {
    "glmed": FemurPaths.GL_MED_JSON,
    "glmin": FemurPaths.GL_MIN_JSON,
    "glmax": FemurPaths.GL_MAX_JSON,
    "psoas": FemurPaths.PSOAS_JSON,
    "vastus_lateralis": FemurPaths.VASTUS_LATERALIS_JSON,
    "vastus_medialis": FemurPaths.VASTUS_MEDIALIS_JSON,
    "vastus_intermedius": FemurPaths.VASTUS_INTERMEDIUS_JSON,
}


@dataclass
class CachedTraction:
    """Cached traction arrays for a single loading case."""
    name: str
    weight: float
    traction: np.ndarray      # Flat array for proximal traction
    traction_cut: np.ndarray  # Flat array for cut equilibrium traction


class Loader:
    """MPI-parallel loader with precomputed traction caching.
    
    All loading cases computed ONCE in precompute_loading_cases().
    Subsequent set_loading_case() just copies cached arrays.
    """
    
    def __init__(self, mesh: dolfinx.mesh.Mesh, facet_tags: dmesh.MeshTags, 
                 load_tag: int, cut_tag: int):
        """
        Initialize loader with mesh and facet tags.
        
        Args:
            mesh: DOLFINx mesh
            facet_tags: MeshTags for boundary facets
            load_tag: Facet tag for proximal load surface (hip + muscles)
            cut_tag: Facet tag for distal cut surface (equilibrium reaction)
        """
        self.mesh = mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.facet_tags = facet_tags
        self.load_tag = load_tag
        self.cut_tag = cut_tag
        self.gdim = mesh.geometry.dim
        
        # Vector function space for traction (P1, 3D)
        P1_vec = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(self.gdim,))
        self.V = fem.functionspace(mesh, P1_vec)
        
        # Traction fields (updated by set_loading_case)
        self.traction = fem.Function(self.V, name="Traction")
        self.traction_cut = fem.Function(self.V, name="TractionCut")
        
        # Geometry objects (rank 0 only)
        self._hip_loader: HIPJointLoad = None
        self._muscle_loaders: dict[str, MuscleLoad] = {}
        self._femur_mesh = None
        self._css: FemurCSS = None
        
        # Cut surface geometry (all ranks)
        self._cut_centroid: np.ndarray = None
        self._cut_area: float = 0.0
        
        # Cached forms for cut equilibrium
        self._I0_form: fem.Form = None
        self._J_forms: List[fem.Form] = None
        
        # Cached DOFs on cut surface
        self._cut_dofs: np.ndarray = None
        
        # Cached coordinates and MPI patterns for interpolation
        self._n_owned: int = 0
        self._x_owned: np.ndarray = None
        self._mpi_counts: List[int] = None
        self._mpi_displs: List[int] = None
        
        # Precomputed traction cache: case_name -> CachedTraction
        self._cache: Dict[str, CachedTraction] = {}
        
        # Equilibrium verification forms (initialized in _init_equilibrium_forms)
        self._force_forms_load: List[fem.Form] = None
        self._moment_forms_load: List[fem.Form] = None
        self._force_forms_cut: List[fem.Form] = None
        self._moment_forms_cut: List[fem.Form] = None
        
        # Initialize geometry
        if self.rank == 0:
            self._init_geometry_rank0()
        
        self._init_cut_geometry()
        self._init_interpolation_cache()
        self._init_equilibrium_forms()
        
        self.comm.Barrier()
    
    def _init_geometry_rank0(self) -> None:
        """Load femur geometry and create hip loader on rank 0."""
        import pyvista as pv
        
        self._femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
        le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
        
        self._css = FemurCSS(self._femur_mesh, head_line, le_me_line, side="left")
        self._hip_loader = HIPJointLoad(self._femur_mesh, self._css, use_cell_data=False)
    
    def _init_cut_geometry(self) -> None:
        """Compute cut surface geometry (centroid, area) and cache equilibrium forms."""
        ds_cut = ufl.Measure(
            "ds", domain=self.mesh, 
            subdomain_data=self.facet_tags, 
            subdomain_id=self.cut_tag
        )
        area_form = fem.form(fem.Constant(self.mesh, 1.0) * ds_cut)
        local_area = fem.assemble_scalar(area_form)
        self._cut_area = self.comm.allreduce(local_area, op=MPI.SUM)
        
        # Compute centroid exactly using surface integrals: x_c = ∫ x dS / ∫ dS
        x = ufl.SpatialCoordinate(self.mesh)
        centroid = np.zeros(3, dtype=np.float64)
        for i in range(self.gdim):
            local_val = fem.assemble_scalar(fem.form(x[i] * ds_cut))
            centroid[i] = self.comm.allreduce(local_val, op=MPI.SUM)
        
        if self._cut_area > 1e-12:
            self._cut_centroid = centroid / self._cut_area
        else:
            self._cut_centroid = np.zeros(3, dtype=np.float64)
        
        self._cache_equilibrium_forms(ds_cut)
        self._cache_cut_dofs()
    
    def _cache_equilibrium_forms(self, ds_cut: ufl.Measure) -> None:
        """Pre-compile UFL forms for cut equilibrium."""
        x = ufl.SpatialCoordinate(self.mesh)
        x_c = fem.Constant(self.mesh, self._cut_centroid.astype(np.float64))
        r = x - x_c
        
        self._I0_form = fem.form(ufl.inner(r, r) * ds_cut)
        
        self._J_forms = []
        for i in range(self.gdim):
            for j in range(i, self.gdim):
                self._J_forms.append(fem.form(r[i] * r[j] * ds_cut))
    
    def _cache_cut_dofs(self) -> None:
        """Cache owned DOF indices on cut surface."""
        V = self.traction_cut.function_space
        fdim = self.mesh.topology.dim - 1
        cut_facets = self.facet_tags.find(self.cut_tag)
        
        all_dofs = fem.locate_dofs_topological(V, fdim, cut_facets)
        n_owned = V.dofmap.index_map.size_local
        self._cut_dofs = np.array([d for d in all_dofs if d < n_owned], dtype=np.int32)
    
    def _init_interpolation_cache(self) -> None:
        """Cache DOF coordinates and MPI communication patterns."""
        V = self.traction.function_space
        imap = V.dofmap.index_map
        self._n_owned = imap.size_local
        
        all_coords = V.tabulate_dof_coordinates()
        self._x_owned = np.ascontiguousarray(all_coords[:self._n_owned])
        
        all_n = self.comm.allgather(self._n_owned)
        self._mpi_counts = all_n
        self._mpi_displs = [sum(all_n[:i]) for i in range(len(all_n))]
    
    def _init_equilibrium_forms(self) -> None:
        """Pre-compile forms for verifying force and moment equilibrium."""
        x = ufl.SpatialCoordinate(self.mesh)
        x_c = fem.Constant(self.mesh, self._cut_centroid.astype(np.float64))
        r = x - x_c  # Position relative to cut centroid
        
        ds_load = ufl.Measure(
            "ds", domain=self.mesh, 
            subdomain_data=self.facet_tags, 
            subdomain_id=self.load_tag
        )
        ds_cut = ufl.Measure(
            "ds", domain=self.mesh, 
            subdomain_data=self.facet_tags, 
            subdomain_id=self.cut_tag
        )
        
        # Force forms: ∫ t_i dS for i=0,1,2
        self._force_forms_load = [
            fem.form(self.traction[i] * ds_load) for i in range(self.gdim)
        ]
        self._force_forms_cut = [
            fem.form(self.traction_cut[i] * ds_cut) for i in range(self.gdim)
        ]
        
        # Moment forms: ∫ (r × t)_i dS for i=0,1,2
        # (r × t)_0 = r_1*t_2 - r_2*t_1
        # (r × t)_1 = r_2*t_0 - r_0*t_2
        # (r × t)_2 = r_0*t_1 - r_1*t_0
        t_load = self.traction
        t_cut = self.traction_cut
        
        self._moment_forms_load = [
            fem.form((r[1]*t_load[2] - r[2]*t_load[1]) * ds_load),
            fem.form((r[2]*t_load[0] - r[0]*t_load[2]) * ds_load),
            fem.form((r[0]*t_load[1] - r[1]*t_load[0]) * ds_load),
        ]
        self._moment_forms_cut = [
            fem.form((r[1]*t_cut[2] - r[2]*t_cut[1]) * ds_cut),
            fem.form((r[2]*t_cut[0] - r[0]*t_cut[2]) * ds_cut),
            fem.form((r[0]*t_cut[1] - r[1]*t_cut[0]) * ds_cut),
        ]
    
    def _verify_equilibrium(self, case_name: str, rtol: float = 1e-2) -> dict:
        """
        Verify force and moment equilibrium for current traction fields.
        
        Computes:
            F_total = ∫_load t dS + ∫_cut t_cut dS  (should be ≈ 0)
            M_total = ∫_load r×t dS + ∫_cut r×t_cut dS  (should be ≈ 0)
        
        Args:
            case_name: Name of loading case (for logging)
            rtol: Relative tolerance for equilibrium check
            
        Returns:
            dict with F_load, F_cut, F_total, M_load, M_cut, M_total, is_balanced
        """
        # Integrate forces on load surface
        F_load = np.zeros(3, dtype=np.float64)
        for i in range(3):
            local_val = fem.assemble_scalar(self._force_forms_load[i])
            F_load[i] = self.comm.allreduce(local_val, op=MPI.SUM)
        
        # Integrate forces on cut surface
        F_cut = np.zeros(3, dtype=np.float64)
        for i in range(3):
            local_val = fem.assemble_scalar(self._force_forms_cut[i])
            F_cut[i] = self.comm.allreduce(local_val, op=MPI.SUM)
        
        # Integrate moments on load surface
        M_load = np.zeros(3, dtype=np.float64)
        for i in range(3):
            local_val = fem.assemble_scalar(self._moment_forms_load[i])
            M_load[i] = self.comm.allreduce(local_val, op=MPI.SUM)
        
        # Integrate moments on cut surface
        M_cut = np.zeros(3, dtype=np.float64)
        for i in range(3):
            local_val = fem.assemble_scalar(self._moment_forms_cut[i])
            M_cut[i] = self.comm.allreduce(local_val, op=MPI.SUM)
        
        F_total = F_load + F_cut
        M_total = M_load + M_cut
        
        # Check equilibrium
        F_ref = max(np.linalg.norm(F_load), np.linalg.norm(F_cut), 1.0)
        M_ref = max(np.linalg.norm(M_load), np.linalg.norm(M_cut), 1.0)
        
        F_err = np.linalg.norm(F_total) / F_ref
        M_err = np.linalg.norm(M_total) / M_ref
        
        is_balanced = (F_err < rtol) and (M_err < rtol)
        
        # Only print warning if equilibrium is NOT satisfied
        if not is_balanced and self.rank == 0:
            print(f"  [✗] Equilibrium check FAILED for '{case_name}':")
            print(f"      F_load = [{F_load[0]:+.2f}, {F_load[1]:+.2f}, {F_load[2]:+.2f}] N")
            print(f"      F_cut  = [{F_cut[0]:+.2f}, {F_cut[1]:+.2f}, {F_cut[2]:+.2f}] N")
            print(f"      F_total= [{F_total[0]:+.2e}, {F_total[1]:+.2e}, {F_total[2]:+.2e}] N (err={F_err:.2e})")
            print(f"      M_load = [{M_load[0]:+.2f}, {M_load[1]:+.2f}, {M_load[2]:+.2f}] N·mm")
            print(f"      M_cut  = [{M_cut[0]:+.2f}, {M_cut[1]:+.2f}, {M_cut[2]:+.2f}] N·mm")
            print(f"      M_total= [{M_total[0]:+.2e}, {M_total[1]:+.2e}, {M_total[2]:+.2e}] N·mm (err={M_err:.2e})")
            print(f"      ⚠ WARNING: Equilibrium not satisfied (rtol={rtol})")
        
        return {
            "F_load": F_load,
            "F_cut": F_cut, 
            "F_total": F_total,
            "M_load": M_load,
            "M_cut": M_cut,
            "M_total": M_total,
            "F_err": F_err,
            "M_err": M_err,
            "is_balanced": is_balanced,
        }
    
    def precompute_loading_cases(self, cases: List[LoadingCase]) -> None:
        """Precompute and cache traction fields for all loading cases."""
        for case in cases:
            self._precompute_single_case(case)
        
        self.comm.Barrier()
    
    def _precompute_single_case(self, case: LoadingCase) -> None:
        """Compute and cache traction arrays for a single loading case."""
        # Reset working arrays
        self.traction.x.array[:] = 0.0
        self.traction_cut.x.array[:] = 0.0
        
        # Apply hip load
        if case.hip is not None:
            self._apply_hip_internal(case.hip)
        
        # Apply muscle loads
        for muscle in case.muscles:
            self._apply_muscle_internal(muscle)
        
        self.traction.x.scatter_forward()
        
        # Integrate ACTUAL forces and moments from the FEM traction field
        # This accounts for interpolation errors between pyvista and FEM mesh
        F_actual, M_actual = self._integrate_load_surface_forces()
        
        # Compute cut equilibrium using actual integrated forces
        self._compute_cut_equilibrium_internal(F_actual, M_actual)
        self.traction_cut.x.scatter_forward()
        
        # Verify force and moment equilibrium
        self._verify_equilibrium(case.name)
        
        # Cache the computed arrays
        self._cache[case.name] = CachedTraction(
            name=case.name,
            weight=case.weight,
            traction=self.traction.x.array.copy(),
            traction_cut=self.traction_cut.x.array.copy(),
        )
    
    def _integrate_load_surface_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate actual force and moment from traction field on load surface.
        
        Returns:
            (F, M): Force and moment vectors from ∫t dS and ∫r×t dS
        """
        F = np.zeros(3, dtype=np.float64)
        for i in range(3):
            local_val = fem.assemble_scalar(self._force_forms_load[i])
            F[i] = self.comm.allreduce(local_val, op=MPI.SUM)
        
        M = np.zeros(3, dtype=np.float64)
        for i in range(3):
            local_val = fem.assemble_scalar(self._moment_forms_load[i])
            M[i] = self.comm.allreduce(local_val, op=MPI.SUM)
        
        return F, M
    
    def set_loading_case(self, case_name: str) -> None:
        """
        Set traction fields from precomputed cache.
        
        This is O(n) array copy - no interpolation, no MPI communication.
        
        Args:
            case_name: Name of precomputed loading case
        
        Raises:
            KeyError: If case_name was not precomputed
        """
        cached = self._cache[case_name]
        self.traction.x.array[:] = cached.traction
        self.traction_cut.x.array[:] = cached.traction_cut
        self.traction.x.scatter_forward()
        self.traction_cut.x.scatter_forward()
    
    def get_cached_weight(self, case_name: str) -> float:
        """Get the weight of a cached loading case."""
        return self._cache[case_name].weight
    
    def get_cached_names(self) -> List[str]:
        """Get list of precomputed loading case names."""
        return list(self._cache.keys())
    
    def _apply_hip_internal(self, spec: HipLoadSpec) -> tuple[np.ndarray, np.ndarray]:
        """Apply hip load and return (force, moment) contribution."""
        F_applied = np.zeros(3, dtype=np.float64)
        r_app = np.zeros(3, dtype=np.float64)
        
        if self.rank == 0:
            v = vector_from_angles(spec.magnitude, spec.alpha_sag, spec.alpha_front)
            self._hip_loader.apply_gaussian_load(
                force_vector_css=v, 
                sigma_deg=spec.sigma_deg, 
                flip=spec.flip
            )
            F_applied[:] = -v if spec.flip else v
            r_app[:] = self._hip_loader.head_center_world
        
        self.comm.Bcast(F_applied, root=0)
        self.comm.Bcast(r_app, root=0)
        
        M_applied = np.cross(r_app - self._cut_centroid, F_applied)
        
        self._interpolate_and_add(self._hip_loader)
        
        return F_applied, M_applied
    
    def _apply_muscle_internal(self, spec: MuscleLoadSpec) -> tuple[np.ndarray, np.ndarray]:
        """Apply muscle load and return (force, moment) contribution."""
        F_applied = np.zeros(3, dtype=np.float64)
        r_app = np.zeros(3, dtype=np.float64)
        loader = None
        
        if self.rank == 0:
            loader = self._get_muscle_loader(spec.name)
            v = vector_from_angles(spec.magnitude, spec.alpha_sag, spec.alpha_front)
            loader.apply_gaussian_load(
                force_vector_css=v, 
                sigma=spec.sigma, 
                flip=spec.flip
            )
            F_applied[:] = -v if spec.flip else v
            r_app[:] = loader._curve.mean(axis=0)
        
        self.comm.Bcast(F_applied, root=0)
        self.comm.Bcast(r_app, root=0)
        
        M_applied = np.cross(r_app - self._cut_centroid, F_applied)
        
        self._interpolate_and_add(loader)
        
        return F_applied, M_applied
    
    def _get_muscle_loader(self, name: str) -> MuscleLoad:
        """Get or create muscle loader by name (rank 0 only)."""
        if name not in self._muscle_loaders:
            if name not in MUSCLE_PATHS:
                raise ValueError(f"Unknown muscle: {name}. Available: {list(MUSCLE_PATHS.keys())}")
            
            loader = MuscleLoad(self._femur_mesh, self._css, use_cell_data=False)
            loader.set_attachment_points(load_json_points(MUSCLE_PATHS[name]))
            self._muscle_loaders[name] = loader
        
        return self._muscle_loaders[name]
    
    def _interpolate_and_add(self, loader_obj) -> None:
        """Interpolate load from loader_obj and add to traction field."""
        bs = self.V.dofmap.index_map_bs
        counts = self._mpi_counts
        displs = self._mpi_displs
        
        if self.rank == 0:
            total_n = sum(counts)
            recv_coords = np.empty((total_n, 3), dtype=np.float64)
            
            self.comm.Gatherv(
                self._x_owned,
                [recv_coords, [n * 3 for n in counts], [d * 3 for d in displs], MPI.DOUBLE],
                root=0
            )
            
            computed_values = loader_obj(recv_coords)
            
            local_values = np.empty((self._n_owned, 3), dtype=np.float64)
            self.comm.Scatterv(
                [computed_values, [n * 3 for n in counts], [d * 3 for d in displs], MPI.DOUBLE],
                local_values,
                root=0
            )
        else:
            self.comm.Gatherv(self._x_owned, None, root=0)
            local_values = np.empty((self._n_owned, 3), dtype=np.float64)
            self.comm.Scatterv(None, local_values, root=0)
        
        self.traction.x.array[:self._n_owned * bs] += local_values.flatten()
    
    def _compute_cut_equilibrium_internal(self, F_total: np.ndarray, M_total: np.ndarray) -> None:
        """Compute equilibrating traction on the cut surface."""
        if self._cut_area < 1e-12:
            return
        
        F_norm = np.linalg.norm(F_total)
        M_norm = np.linalg.norm(M_total)
        if F_norm < 1e-15 and M_norm < 1e-15:
            return
        
        t_F = -F_total / self._cut_area
        
        # Assemble I0 (collective)
        local_I0 = fem.assemble_scalar(self._I0_form)
        I0 = self.comm.allreduce(local_I0, op=MPI.SUM)
        
        # Assemble J tensor (collective)
        J = np.zeros((3, 3), dtype=np.float64)
        form_idx = 0
        for i in range(3):
            for j in range(i, 3):
                local_val = fem.assemble_scalar(self._J_forms[form_idx])
                global_val = self.comm.allreduce(local_val, op=MPI.SUM)
                J[i, j] = global_val
                J[j, i] = global_val
                form_idx += 1
        
        if I0 < 1e-12:
            C = np.zeros(3, dtype=np.float64)
        else:
            K = I0 * np.eye(3) - J
            try:
                C = -np.linalg.solve(K, M_total)
            except np.linalg.LinAlgError:
                C = -np.linalg.pinv(K) @ M_total
        
        if len(self._cut_dofs) == 0:
            return
        
        bs = self.V.dofmap.index_map_bs
        x_cut = self._x_owned[self._cut_dofs]
        r_cut = x_cut - self._cut_centroid
        t_M = np.cross(C, r_cut)
        t_total = t_F + t_M
        
        traction_array = self.traction_cut.x.array
        for i, dof in enumerate(self._cut_dofs):
            traction_array[dof * bs:(dof + 1) * bs] = t_total[i]
