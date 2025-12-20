"""MPI-parallel femur load mapping with traction caching.

Rank 0 owns the PyVista geometry and evaluates surface tractions at gathered
DOF coordinates; all ranks receive scattered values and assemble a DOLFINx
traction field. Loading cases are precomputed once and then replayed cheaply.

Units: geometry in mm; input forces in N; traction output in MPa (N/mm²).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import basix.ufl
from mpi4py import MPI

from simulation.femur_css import FemurCSS, load_json_points
from simulation.paths import FemurPaths
from simulation.femur_loads import HIPJointLoad, MuscleLoad, vector_from_angles
from simulation.utils import assign, get_owned_size


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
    """One gait phase: hip load + zero or more muscle loads."""
    name: str
    day_cycles: float             # Number of loading cycles per day
    hip: HipLoadSpec | None       # Hip joint load (None if no hip load)
    muscles: List[MuscleLoadSpec] # Muscle loads (empty list if none)


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
    """Cached owned-DOF traction values for one loading case."""
    name: str
    day_cycles: float
    traction: np.ndarray  # Flat array for proximal traction


class Loader:
    """MPI-parallel traction loader with precomputed case cache.

    Builds tractions on the proximal surface; the cut surface is clamped via
    Dirichlet BCs in mechanics.
    """
    
    def __init__(self, mesh: dmesh.Mesh, facet_tags: dmesh.MeshTags, load_tag: int):
        """Create loader for `mesh` and proximal surface tag `load_tag`."""
        self.mesh = mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.facet_tags = facet_tags
        self.load_tag = load_tag
        self.gdim = mesh.geometry.dim
        
        # Vector function space for traction (P1, 3D)
        P1_vec = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(self.gdim,))
        self.V = fem.functionspace(mesh, P1_vec)
        
        # Traction field (updated by set_loading_case)
        self.traction = fem.Function(self.V, name="Traction")
        
        # Geometry objects (rank 0 only)
        self._hip_loader: HIPJointLoad = None
        self._muscle_loaders: Dict[str, MuscleLoad] = {}
        self._femur_mesh = None
        self._css: FemurCSS = None
        
        # Cached coordinates and MPI patterns for interpolation
        self._n_owned: int = 0
        self._x_owned: np.ndarray = None
        self._mpi_counts: List[int] = None
        self._mpi_displs: List[int] = None
        
        # Precomputed traction cache: case_name -> CachedTraction
        self._cache: Dict[str, CachedTraction] = {}
        
        # Initialize geometry (rank 0 only)
        if self.rank == 0:
            self._init_geometry_rank0()
        
        self._init_interpolation_cache()
        self.comm.Barrier()
    
    def _init_geometry_rank0(self) -> None:
        """Load femur geometry and initialize loaders (rank 0 only)."""
        import pyvista as pv
        
        self._femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
        le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
        
        self._css = FemurCSS(self._femur_mesh, head_line, le_me_line, side="left")
        self._hip_loader = HIPJointLoad(self._femur_mesh, self._css, use_cell_data=False)
    
    def _init_interpolation_cache(self) -> None:
        """Cache owned DOF coordinates and Gatherv/Scatterv layout.

        For efficiency and robustness, we evaluate tractions only at DOFs that live on
        facets marked with `self.load_tag` (typically the proximal surface).

        DOLFINx has historically differed in whether `tabulate_dof_coordinates()` returns
        coordinates per *block* DOF (one row per node) or per *scalar* DOF (duplicated
        per component). We detect the layout and cache one coordinate per *block* DOF.
        """
        V = self.traction.function_space
        imap = V.dofmap.index_map
        bs = int(V.dofmap.index_map_bs)
        n_blocks_total = int(imap.size_local + imap.num_ghosts)

        # --- 1) Find DOFs on the loaded facets and map them to unique block indices
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        facets = self.facet_tags.find(self.load_tag)

        if facets.size == 0:
            dofs = np.empty(0, dtype=np.int32)
        else:
            dofs = fem.locate_dofs_topological(V, fdim, facets)

        if dofs.size == 0:
            block_ids = np.empty(0, dtype=np.int32)
        else:
            # If DOF indices range beyond the number of blocks, they are scalar indices.
            if bs > 1 and int(dofs.max()) >= n_blocks_total:
                block_ids = np.unique(dofs // bs).astype(np.int32, copy=False)
            else:
                block_ids = np.unique(dofs).astype(np.int32, copy=False)

        # Keep only owned (non-ghost) blocks; we will populate ghosts via scatter.
        self._owned_blocks = np.ascontiguousarray(block_ids[block_ids < imap.size_local], dtype=np.int32)
        self._n_owned = int(self._owned_blocks.size)

        # --- 2) Cache coordinates for owned blocks
        coords = V.tabulate_dof_coordinates()
        n_coords = int(coords.shape[0])

        if n_coords == n_blocks_total:
            # One coordinate per block DOF
            block_coords = coords
        elif bs > 1 and n_coords == n_blocks_total * bs:
            # One coordinate per scalar DOF (duplicated per component); take first in each block
            block_coords = coords[::bs]
        else:
            # Fallback: try a conservative stride interpretation, else assume block layout
            if bs > 1 and n_coords % n_blocks_total == 0 and (n_coords // n_blocks_total) == bs:
                block_coords = coords[::bs]
            else:
                block_coords = coords[:n_blocks_total]

        self._x_owned = np.ascontiguousarray(block_coords[self._owned_blocks])

        # --- 3) MPI gather/scatter layout in "number of points" (NOT multiplied by gdim here)
        all_n = self.comm.allgather(self._n_owned)
        self._mpi_counts = all_n
        self._mpi_displs = [sum(all_n[:i]) for i in range(len(all_n))]

    def precompute_loading_cases(self, cases: List[LoadingCase]) -> None:
        """Precompute and cache tractions for all `cases` (collective)."""
        for case in cases:
            self._precompute_single_case(case)
        self.comm.Barrier()
    
    def _precompute_single_case(self, case: LoadingCase) -> None:
        """Compute and cache traction array for a single loading case."""
        assign(self.traction, 0.0, scatter=False)
        
        if case.hip is not None:
            self._apply_hip_internal(case.hip)
        
        for muscle in case.muscles:
            self._apply_muscle_internal(muscle)
        
        self.traction.x.scatter_forward()
        
        n_owned = get_owned_size(self.traction)
        self._cache[case.name] = CachedTraction(
            name=case.name,
            day_cycles=case.day_cycles,
            traction=self.traction.x.array[:n_owned].copy(),
        )
    
    def set_loading_case(self, case_name: str) -> None:
        """Load cached traction into `self.traction` (owned DOFs + scatter)."""
        cached = self._cache[case_name]
        assign(self.traction, cached.traction, scatter=True)
    
    def get_cached_day_cycles(self, case_name: str) -> float:
        """Get the day_cycles of a cached loading case."""
        return self._cache[case_name].day_cycles
    
    def get_cached_names(self) -> List[str]:
        """Get list of precomputed loading case names."""
        return list(self._cache.keys())
    
    def _apply_hip_internal(self, spec: HipLoadSpec) -> None:
        """Apply hip load to traction field."""
        if self.rank == 0:
            v = vector_from_angles(spec.magnitude, spec.alpha_sag, spec.alpha_front)
            self._hip_loader.apply_gaussian_load(
                force_vector_css=v,
                sigma_deg=spec.sigma_deg,
                flip=spec.flip,
            )
        self._interpolate_and_add(self._hip_loader)
    
    def _apply_muscle_internal(self, spec: MuscleLoadSpec) -> None:
        """Apply muscle load to traction field."""
        loader = None
        if self.rank == 0:
            loader = self._get_muscle_loader(spec.name)
            v = vector_from_angles(spec.magnitude, spec.alpha_sag, spec.alpha_front)
            loader.apply_gaussian_load(
                force_vector_css=v,
                sigma=spec.sigma,
                flip=spec.flip,
            )
        self._interpolate_and_add(loader)
    
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
        """Gather coords → rank-0 evaluate → scatter → add into traction field.

        We evaluate only at DOFs that live on facets marked with `self.load_tag`.
        """
        counts = self._mpi_counts
        displs = self._mpi_displs

        gdim = int(self.gdim)
        V = self.traction.function_space
        bs = int(V.dofmap.index_map_bs)
        if bs != gdim:
            raise RuntimeError(f"Traction space block size (bs={bs}) must match gdim={gdim}.")

        # --- Gather coordinates, evaluate on rank 0, scatter back values
        if self.rank == 0:
            total_n = int(sum(counts))
            recv_coords = np.empty((total_n, gdim), dtype=np.float64)

            self.comm.Gatherv(
                self._x_owned,
                [recv_coords, [n * gdim for n in counts], [d * gdim for d in displs], MPI.DOUBLE],
                root=0,
            )

            computed_values = loader_obj(recv_coords) if loader_obj is not None else np.zeros((total_n, gdim), dtype=np.float64)

            computed_values = np.asarray(computed_values, dtype=np.float64)
            if computed_values.shape != (total_n, gdim):
                raise ValueError(
                    f"Loader returned array of shape {computed_values.shape}, expected {(total_n, gdim)}."
                )
            if not np.isfinite(computed_values).all():
                bad = np.where(~np.isfinite(computed_values))
                raise ValueError(f"Loader produced non-finite traction values at indices {bad}.")

            local_values = np.empty((self._n_owned, gdim), dtype=np.float64)
            self.comm.Scatterv(
                [computed_values, [n * gdim for n in counts], [d * gdim for d in displs], MPI.DOUBLE],
                local_values,
                root=0,
            )
        else:
            self.comm.Gatherv(self._x_owned, None, root=0)
            local_values = np.empty((self._n_owned, gdim), dtype=np.float64)
            self.comm.Scatterv(None, local_values, root=0)

        # --- Add into the traction field (owned scalar DOFs only); ghosts handled by scatter_forward
        if self._n_owned > 0:
            blocks = self._owned_blocks
            # scalar dof indices for these blocks (flattened)
            dof_idx = (blocks[:, None] * bs + np.arange(bs, dtype=np.int32)[None, :]).ravel()
            self.traction.x.array[dof_idx] += local_values[:, :bs].ravel()

