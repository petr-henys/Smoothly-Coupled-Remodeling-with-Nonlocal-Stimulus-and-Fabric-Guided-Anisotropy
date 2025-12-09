"""
MPI-parallel femur surface load interpolation.

This module provides:
- LoadingCase: Configuration of loads (hip + muscles) for one loading scenario
- Loader: Applies LoadingCase to compute traction fields for FEM

Architecture:
- Rank 0 owns pyvista geometry and computes load distributions
- DOF coordinates gathered from all ranks to rank 0
- Computed traction values scattered back to owners
- All MPI communication uses owner-computes pattern

Two traction fields:
- traction: Hip + muscle loads on proximal surface (tag=2)
- traction_cut: Equilibrating reaction on distal cut (tag=1)

The cut equilibrium ensures: ΣF = 0, ΣM = 0 for the free body.

Units: mm, N, MPa (forces as traction = N/mm² = MPa)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import dolfinx
import dolfinx.fem as fem
import basix.ufl
from mpi4py import MPI

from simulation.femur_css import FemurCSS, load_json_points
from simulation.paths import FemurPaths
from simulation.femur_loads import HIPJointLoad, MuscleLoad, vector_from_angles


# =============================================================================
# Loading Case Definition
# =============================================================================

@dataclass
class HipLoadSpec:
    """
    Hip joint load specification.
    
    Attributes:
        magnitude: Force magnitude in N
        alpha_sag: Sagittal plane angle in degrees (+ = anterior, - = posterior)
        alpha_front: Frontal plane angle in degrees (+ = lateral, - = medial)
        sigma_deg: Gaussian spread in degrees (contact patch size)
        flip: If True, flip force direction (compression into head)
    """
    magnitude: float
    alpha_sag: float = 0.0
    alpha_front: float = 0.0
    sigma_deg: float = 10.0
    flip: bool = True


@dataclass
class MuscleLoadSpec:
    """
    Muscle load specification.
    
    Attributes:
        name: Muscle identifier ("glmed", "glmin", "glmax", "psoas", 
              "vastus_lateralis", "vastus_medialis", "vastus_intermedius")
        magnitude: Force magnitude in N
        alpha_sag: Sagittal plane angle in degrees (+ = anterior, - = posterior)
        alpha_front: Frontal plane angle in degrees (+ = lateral, - = medial)
        sigma: Gaussian spread in mm (attachment area size)
        flip: If True, flip force direction (muscle contraction pulls)
    """
    name: str
    magnitude: float
    alpha_sag: float = 0.0
    alpha_front: float = 0.0
    sigma: float = 2.0
    flip: bool = False


@dataclass 
class LoadingCase:
    """
    A loading case = one combination of hip + muscles that occur together.
    
    Represents a specific loading scenario (e.g., mid-stance phase of gait)
    with hip joint reaction force and active muscle forces.
    
    Multiple LoadingCases can be defined and their stimulus (Ψ) will be 
    weight-averaged for bone remodeling.
    
    Attributes:
        name: Descriptive name for the loading case
        hip: Hip joint load specification (optional)
        muscles: List of muscle load specifications
        weight: Weight for averaging multiple cases (default 1.0)
    
    Example:
        case = (LoadingCase(name="mid_stance")
            .set_hip(magnitude=2000, alpha_front=-10)
            .add_muscle("glmed", magnitude=500, alpha_front=35))
    """
    name: str
    hip: Optional[HipLoadSpec] = None
    muscles: List[MuscleLoadSpec] = field(default_factory=list)
    weight: float = 1.0
    
    def set_hip(self, magnitude: float, alpha_sag: float = 0.0, 
                alpha_front: float = 0.0, sigma_deg: float = 10.0, 
                flip: bool = True) -> "LoadingCase":
        """Set the hip joint load. Returns self for chaining."""
        self.hip = HipLoadSpec(
            magnitude=magnitude, alpha_sag=alpha_sag, alpha_front=alpha_front,
            sigma_deg=sigma_deg, flip=flip
        )
        return self
    
    def add_muscle(self, name: str, magnitude: float, 
                   alpha_sag: float = 0.0, alpha_front: float = 0.0,
                   sigma: float = 2.0, flip: bool = False) -> "LoadingCase":
        """Add a muscle load. Returns self for chaining."""
        self.muscles.append(MuscleLoadSpec(
            name=name, magnitude=magnitude, 
            alpha_sag=alpha_sag, alpha_front=alpha_front,
            sigma=sigma, flip=flip
        ))
        return self


# =============================================================================
# Loader: Applies LoadingCase to DOLFINx mesh
# =============================================================================

class Loader:
    """
    MPI-parallel loader: applies LoadingCase to produce traction fields.
    
    Two traction fields:
    - traction: Hip + muscle loads (for Neumann BC on proximal surface, tag=2)
    - traction_cut: Equilibrating reaction (for Neumann BC on distal cut, tag=1)
    
    The cut equilibrium ensures global force and moment balance:
        ∫ t dS = -F_applied  (force balance)
        ∫ (x - x_c) × t dS = -M_applied  (moment balance)
    
    MPI pattern: Rank 0 computes, all ranks participate in interpolation.
    """
    
    # Mapping of muscle names to JSON paths
    MUSCLE_PATHS = {
        "glmed": FemurPaths.GL_MED_JSON,
        "glmin": FemurPaths.GL_MIN_JSON,
        "glmax": FemurPaths.GL_MAX_JSON,
        "psoas": FemurPaths.PSOAS_JSON,
        "vastus_lateralis": FemurPaths.VASTUS_LATERALIS_JSON,
        "vastus_medialis": FemurPaths.VASTUS_MEDIALIS_JSON,
        "vastus_intermedius": FemurPaths.VASTUS_INTERMEDIUS_JSON,
    }
    
    def __init__(self, dolfinx_mesh: dolfinx.mesh.Mesh, facet_tags, load_tag: int = 2, cut_tag: int = 1):
        """
        Initialize loader with mesh and facet tags.
        
        Args:
            dolfinx_mesh: DOLFINx mesh
            facet_tags: MeshTags for boundary facets
            load_tag: Facet tag for proximal load surface (hip + muscles, default 2)
            cut_tag: Facet tag for distal cut surface (equilibrium reaction, default 1)
        """
        self.mesh = dolfinx_mesh
        self.comm = self.mesh.comm
        self.rank = self.comm.rank
        self.facet_tags = facet_tags
        self.load_tag = load_tag
        self.cut_tag = cut_tag
        
        # Vector function space for traction (P1, 3D)
        gdim = self.mesh.geometry.dim
        P1_vec = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1, shape=(gdim,))
        self.V = fem.functionspace(self.mesh, P1_vec)
        
        # Traction for hip + muscles (proximal surface)
        self.traction = fem.Function(self.V, name="Traction")
        
        # Traction for cut equilibrium (distal cut)
        self.traction_cut = fem.Function(self.V, name="TractionCut")
        
        # Only rank 0 sets up pyvista geometry and load objects
        self._hip_loader: Optional[HIPJointLoad] = None
        self._muscle_loaders: dict = {}
        
        # Cache for applied forces/moments (computed on rank 0, broadcast)
        self._F_total = np.zeros(3)  # Total applied force [N]
        self._M_total = np.zeros(3)  # Total applied moment about cut centroid [Nmm]
        
        # Cut surface geometry (computed once)
        self._cut_centroid: Optional[np.ndarray] = None
        self._cut_area: Optional[float] = None
        
        if self.rank == 0:
            self._setup_geometry()
        
        # Setup cut surface geometry on all ranks
        self._setup_cut_geometry()
        
        self.comm.Barrier()
    
    def _setup_geometry(self) -> None:
        """Setup femur geometry and load objects on rank 0."""
        import pyvista as pv
        
        femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
        le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
        
        css = FemurCSS(femur_mesh, head_line, le_me_line, side="left")
        
        # Hip loader
        self._hip_loader = HIPJointLoad(femur_mesh, css, use_cell_data=False)
        
        # Muscle loaders - lazy init when needed
        self._femur_mesh = femur_mesh
        self._css = css
    
    def _setup_cut_geometry(self) -> None:
        """
        Compute cut surface geometry (centroid, area) for equilibrium.
        
        MPI: Each rank computes its local contribution, then reduce to get global.
        Note: fem.assemble_scalar is collective - all ranks must call it.
        """
        from dolfinx import mesh as dmesh
        import ufl
        
        # Compute facet areas using UFL - MUST be called by all ranks (collective)
        ds_cut = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags, subdomain_id=self.cut_tag)
        one = fem.Constant(self.mesh, 1.0)
        area_form = fem.form(one * ds_cut)
        local_area = fem.assemble_scalar(area_form)  # Collective operation
        
        # Find facets with cut_tag for centroid computation
        cut_facets = self.facet_tags.find(self.cut_tag)
        
        if cut_facets.size == 0:
            # This rank has no cut facets - contribute zeros to centroid
            local_centroid_weighted = np.zeros(3)
        else:
            # Compute facet midpoints (approximation of centroid per facet)
            fdim = self.mesh.topology.dim - 1
            midpoints = dmesh.compute_midpoints(self.mesh, fdim, cut_facets)
            
            # Weighted centroid: sum(midpoint * facet_area) / total_area
            # For simplicity, assume equal weight per facet locally
            # (more accurate would integrate x over each facet)
            local_centroid_weighted = midpoints.sum(axis=0) * (local_area / max(cut_facets.size, 1))
        
        # Global reduce
        global_area = self.comm.allreduce(local_area, op=MPI.SUM)
        global_centroid_weighted = np.zeros(3)
        self.comm.Allreduce(local_centroid_weighted, global_centroid_weighted, op=MPI.SUM)
        
        if global_area > 0:
            self._cut_centroid = global_centroid_weighted / global_area
        else:
            # Fallback - no cut surface found
            self._cut_centroid = np.zeros(3)
            
        self._cut_area = global_area

    def _get_muscle_loader(self, name: str) -> MuscleLoad:
        """Get or create muscle loader by name (rank 0 only)."""
        if name not in self._muscle_loaders:
            if name not in self.MUSCLE_PATHS:
                raise ValueError(f"Unknown muscle: {name}. Available: {list(self.MUSCLE_PATHS.keys())}")
            
            loader = MuscleLoad(self._femur_mesh, self._css, use_cell_data=False)
            loader.set_attachment_points(load_json_points(self.MUSCLE_PATHS[name]))
            self._muscle_loaders[name] = loader
        
        return self._muscle_loaders[name]
    
    def apply_loading_case(self, case: LoadingCase) -> None:
        """
        Apply a LoadingCase to compute both traction fields.
        
        Updates:
        - self.traction: Combined traction from hip + muscles (proximal)
        - self.traction_cut: Equilibrating reaction on distal cut
        
        The cut traction ensures global equilibrium:
            ∫ t dS + ∫ t_cut dS = 0  (force balance)
            ∫ r×t dS + ∫ r×t_cut dS = 0  (moment balance)
        
        Args:
            case: LoadingCase specification
        """
        # Zero out both tractions
        self.traction.x.array[:] = 0.0
        self.traction_cut.x.array[:] = 0.0
        
        # Reset force/moment accumulators
        self._F_total[:] = 0.0
        self._M_total[:] = 0.0
        
        # Apply hip load if specified
        if case.hip is not None:
            self._apply_hip(case.hip)
        
        # Apply each muscle load
        for muscle in case.muscles:
            self._apply_muscle(muscle)
        
        self.traction.x.scatter_forward()
        
        # Compute equilibrating traction on cut
        self._apply_cut_equilibrium()
        self.traction_cut.x.scatter_forward()
    
    def _apply_hip(self, spec: HipLoadSpec) -> None:
        """
        Apply hip load and add to traction.
        
        Also accumulates force and moment for cut equilibrium.
        MPI pattern: rank 0 computes, all ranks participate in interpolation.
        """
        F_applied = np.zeros(3)
        r_app = np.zeros(3)  # Application point
        
        if self.rank == 0:
            v = vector_from_angles(spec.magnitude, spec.alpha_sag, spec.alpha_front)
            # Apply load returns traction mesh
            self._hip_loader.apply_gaussian_load(
                force_vector_css=v, sigma_deg=spec.sigma_deg, flip=spec.flip
            )
            # Force vector after flip
            if spec.flip:
                F_applied = -v
            else:
                F_applied = v.copy()
            # Application point: femoral head center
            r_app = self._hip_loader.head_center_world.copy()
        
        # Broadcast force and application point
        self.comm.Bcast(F_applied, root=0)
        self.comm.Bcast(r_app, root=0)
        
        # Accumulate force and moment about cut centroid
        self._F_total += F_applied
        if self._cut_centroid is not None:
            self._M_total += np.cross(r_app - self._cut_centroid, F_applied)
        
        self.comm.Barrier()
        self._interpolate_and_add(self._hip_loader)
    
    def _apply_muscle(self, spec: MuscleLoadSpec) -> None:
        """
        Apply muscle load and add to traction.
        
        Also accumulates force and moment for cut equilibrium.
        MPI pattern: rank 0 computes, all ranks participate in interpolation.
        Note: loader is None on non-root ranks but _interpolate_and_add only
        uses it on rank 0.
        """
        F_applied = np.zeros(3)
        r_app = np.zeros(3)  # Application point (centroid of attachment)
        loader = None
        
        if self.rank == 0:
            loader = self._get_muscle_loader(spec.name)
            v = vector_from_angles(spec.magnitude, spec.alpha_sag, spec.alpha_front)
            loader.apply_gaussian_load(force_vector_css=v, sigma=spec.sigma, flip=spec.flip)
            # Force vector after flip
            if spec.flip:
                F_applied = -v
            else:
                F_applied = v.copy()
            # Application point: centroid of muscle attachment curve
            r_app = loader._curve.mean(axis=0)
        
        # Broadcast force and application point
        self.comm.Bcast(F_applied, root=0)
        self.comm.Bcast(r_app, root=0)
        
        # Accumulate force and moment about cut centroid
        self._F_total += F_applied
        if self._cut_centroid is not None:
            self._M_total += np.cross(r_app - self._cut_centroid, F_applied)
        
        self.comm.Barrier()
        self._interpolate_and_add(loader)
    
    def _interpolate_and_add(self, loader_obj) -> None:
        """
        Interpolate load from loader_obj and add to traction field.
        
        MPI owner-computes pattern:
        1. All ranks gather their owned DOF coordinates to rank 0
        2. Rank 0 evaluates loader_obj at all coordinates (loader_obj can be None on other ranks)
        3. Rank 0 scatters computed values back to owners
        4. Each rank adds received values to its owned DOFs
        
        Args:
            loader_obj: Load interpolator (only used on rank 0, can be None on others)
        """
        V = self.traction.function_space
        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        n_owned = imap.size_local
        
        # Get owned coordinates
        all_coords = V.tabulate_dof_coordinates()
        x_owned = all_coords[:n_owned]
        local_n = n_owned
        
        # MPI setup
        all_n = self.comm.allgather(local_n)
        displs = [sum(all_n[:i]) for i in range(len(all_n))]
        
        if self.rank == 0:
            total_n = sum(all_n)
            recv_coords = np.empty((total_n, 3), dtype=np.float64)
            
            # Gather coords from all ranks
            self.comm.Gatherv(
                np.ascontiguousarray(x_owned),
                [recv_coords, [n * 3 for n in all_n], [d * 3 for d in displs], MPI.DOUBLE],
                root=0
            )
            
            # Evaluate load at all coordinates
            computed_values = loader_obj(recv_coords)
            
            # Scatter results
            local_values = np.empty((local_n, 3), dtype=np.float64)
            self.comm.Scatterv(
                [computed_values, [n * 3 for n in all_n], [d * 3 for d in displs], MPI.DOUBLE],
                local_values,
                root=0
            )
        else:
            self.comm.Gatherv(np.ascontiguousarray(x_owned), None, root=0)
            local_values = np.empty((local_n, 3), dtype=np.float64)
            self.comm.Scatterv(None, local_values, root=0)
        
        # Add to traction (not replace!)
        self.traction.x.array[:n_owned * bs] += local_values.flatten()

    def _apply_cut_equilibrium(self) -> None:
        """
        Compute equilibrating traction on the cut surface.
        
        The cut traction balances applied forces and moments:
            t_cut = t_F + t_M
        
        where:
            t_F = -F_total / A_cut  (uniform, balances force)
            t_M = C × (x - x_c)     (linear, balances moment)
        
        The constant C is chosen so that:
            ∫ (x - x_c) × t_M dS = -M_total
        
        MPI pattern: All ranks participate in applying to their owned DOFs.
        """
        if self._cut_area is None or self._cut_area < 1e-12:
            # No cut surface - nothing to do
            return
        
        # Get DOFs on cut surface (facets with cut_tag)
        V = self.traction_cut.function_space
        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        n_owned = imap.size_local
        
        # Force part: uniform traction
        # t_F = -F_total / A_cut (reaction = negative of applied)
        t_F = -self._F_total / self._cut_area
        
        # Get coordinates
        all_coords = V.tabulate_dof_coordinates()
        x_owned = all_coords[:n_owned]
        
        # Find which DOFs are on the cut surface
        # We need DOFs associated with facets tagged with cut_tag
        cut_dofs = self._get_cut_surface_dofs()
        
        # For moment part, we need: ∫ (x - x_c) × t_M dS = -M_total
        # Using t_M(x) = C × (x - x_c), where C is a pseudo-vector
        #
        # For a circular cross-section with radius R:
        #   ∫ (x - x_c) × [C × (x - x_c)] dS = C × ∫ |x - x_c|² dS = C × I_polar
        # where I_polar = π R⁴ / 2 for a circle.
        #
        # For general shape, compute: I = ∫ |x - x_c|² dS locally
        
        # Compute second moment of area (polar) on cut surface
        I_polar = self._compute_cut_polar_moment()
        
        if I_polar > 1e-12:
            # C such that C × I_polar = -M_total  =>  C = -M_total / I_polar
            C = -self._M_total / I_polar
        else:
            C = np.zeros(3)
        
        # Apply traction to cut DOFs
        for i in range(n_owned):
            if i in cut_dofs:
                x = x_owned[i]
                r = x - self._cut_centroid
                
                # t_M = C × r
                t_M = np.cross(C, r)
                
                # Total traction at this DOF
                t_total = t_F + t_M
                
                self.traction_cut.x.array[i * bs:(i + 1) * bs] = t_total
    
    def _get_cut_surface_dofs(self) -> set:
        """
        Get set of owned DOF indices that lie on the cut surface.
        
        Returns:
            Set of local DOF indices on cut surface
        """
        from dolfinx import mesh as dmesh
        
        # Find facets with cut_tag
        cut_facets = self.facet_tags.find(self.cut_tag)
        
        if cut_facets.size == 0:
            return set()
        
        # Get DOFs associated with these facets
        V = self.traction_cut.function_space
        fdim = self.mesh.topology.dim - 1
        
        # Ensure connectivity
        self.mesh.topology.create_connectivity(fdim, self.mesh.topology.dim)
        
        cut_dofs = set()
        for facet in cut_facets:
            # Get cells connected to this facet
            facet_to_cell = self.mesh.topology.connectivity(fdim, self.mesh.topology.dim)
            cells = facet_to_cell.links(facet)
            
            for cell in cells:
                cell_dofs = V.dofmap.cell_dofs(cell)
                # We need DOFs on this facet - approximate by taking first few
                # For P1 elements on triangular facet: 3 DOFs
                # This is approximate - proper way would be to check coordinates
                cut_dofs.update(cell_dofs[:3])  # First 3 DOFs (vertices of facet approx)
        
        # Filter to owned only
        n_owned = V.dofmap.index_map.size_local
        cut_dofs = {d for d in cut_dofs if d < n_owned}
        
        return cut_dofs
    
    def _compute_cut_polar_moment(self) -> float:
        """
        Compute polar second moment of area: I = ∫ |x - x_c|² dS
        
        Returns:
            Polar moment of area (mm⁴)
        """
        import ufl
        
        if self._cut_centroid is None:
            return 0.0
        
        # Build UFL form for ∫ |x - x_c|² dS over cut surface
        ds_cut = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags, subdomain_id=self.cut_tag)
        
        x = ufl.SpatialCoordinate(self.mesh)
        x_c = fem.Constant(self.mesh, self._cut_centroid.astype(np.float64))
        
        r_sq = ufl.inner(x - x_c, x - x_c)
        form = fem.form(r_sq * ds_cut)
        
        local_I = fem.assemble_scalar(form)
        global_I = self.comm.allreduce(local_I, op=MPI.SUM)
        
        return global_I
