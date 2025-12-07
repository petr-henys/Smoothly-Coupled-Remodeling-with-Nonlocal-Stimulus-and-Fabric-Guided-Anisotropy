"""MPI-parallel femur surface load interpolation."""

import numpy as np
import dolfinx
import dolfinx.fem as fem
import basix.ufl
from mpi4py import MPI

from simulation.femur_css import FemurCSS, load_json_points
from simulation.paths import FemurPaths
from simulation.femur_loads import HIPJointLoad, MuscleLoad, vector_from_angles


class Loader:
    """Parallel loader: rank 0 computes loads, scatter to all."""
    
    def __init__(self, dolfinx_mesh: dolfinx.mesh.Mesh):
        self.mesh = dolfinx_mesh
        self.comm = self.mesh.comm
        self.rank = self.comm.rank
        
        # Create vector function space for traction (3D vectors)
        gdim = self.mesh.geometry.dim
        P1_vec = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1, shape=(gdim,))
        self.V = fem.functionspace(self.mesh, P1_vec)
        
        # Create traction functions
        self.hip_fun = fem.Function(self.V, name="Hip Joint Load")
        self.glmed_fun = fem.Function(self.V, name="GL med Load")
        self.glmin_fun = fem.Function(self.V, name="GL min Load")
        self.glmax_fun = fem.Function(self.V, name="GL max Load")
        self.psoas_fun = fem.Function(self.V, name="Psoas Load")

        self.vastus_lateralis_fun = fem.Function(self.V, name="Vastus Lateralis Load")
        self.vastus_medialis_fun = fem.Function(self.V, name="Vastus Medialis Load")
        self.vastus_intermedius_fun = fem.Function(self.V, name="Vastus Intermedius Load")
        
        # Only rank 0 sets up pyvista and load objects
        self.hip = None
        self.gl_med = None
        self.gl_min = None
        self.gl_max = None
        self.psoas = None

        self.vastus_lateralis = None
        self.vastus_medialis = None
        self.vastus_intermedius = None
        
        if self.rank == 0:
            import pyvista as pv
            femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
            head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
            le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
            
            css = FemurCSS(femur_mesh, head_line, le_me_line, side="left")
            
            self.hip = HIPJointLoad(femur_mesh, css, use_cell_data=False)

            self.gl_med = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.gl_med.set_attachment_points(load_json_points(FemurPaths.GL_MED_JSON))

            self.gl_min = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.gl_min.set_attachment_points(load_json_points(FemurPaths.GL_MIN_JSON))

            self.gl_max = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.gl_max.set_attachment_points(load_json_points(FemurPaths.GL_MAX_JSON))

            self.psoas = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.psoas.set_attachment_points(load_json_points(FemurPaths.PSOAS_JSON))

            self.vastus_lateralis = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.vastus_lateralis.set_attachment_points(load_json_points(FemurPaths.VASTUS_LATERALIS_JSON))

            self.vastus_medialis = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.vastus_medialis.set_attachment_points(load_json_points(FemurPaths.VASTUS_MEDIALIS_JSON))

            self.vastus_intermedius = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.vastus_intermedius.set_attachment_points(load_json_points(FemurPaths.VASTUS_INTERMEDIUS_JSON))
        
        self.comm.Barrier()
    
    def _interpolate_mpi(self, loader_obj, target_fun: fem.Function) -> None:
        """Owner-computes pattern: gather coords, eval on rank 0, scatter back."""
        # DOF info
        V = target_fun.function_space
        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        n_owned = imap.size_local
        
        # Owned coordinates only
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
            
            # Evaluate load
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
        
        # Assign and sync
        target_fun.x.array[:n_owned * bs] = local_values.flatten()
        target_fun.x.scatter_forward()
    
    def hip_force(self, magnitude: float, alpha_sag: float, alpha_front: float, 
                  sigma_deg: float = 10.0, flip: bool = True) -> fem.Function:
        """Apply hip joint load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.hip.apply_gaussian_load(force_vector_css=v, 
                                         sigma_deg=sigma_deg, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.hip, self.hip_fun)
        
        return self.hip_fun

    def glmed_force(self, magnitude: float, alpha_sag: float, alpha_front: float, 
                    sigma: float = 2.0, flip: bool = False) -> fem.Function:
        """Apply gluteus medius load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.gl_med.apply_gaussian_load(force_vector_css=v, 
                                            sigma=sigma, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.gl_med, self.glmed_fun)
        
        return self.glmed_fun
    
    def glmin_force(self, magnitude: float, alpha_sag: float, alpha_front: float, 
                    sigma: float = 2.0, flip: bool = False) -> fem.Function:
        """Apply gluteus minimus load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.gl_min.apply_gaussian_load(force_vector_css=v, 
                                            sigma=sigma, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.gl_min, self.glmin_fun)
        
        return self.glmin_fun
    
    def glmax_force(self, magnitude: float, alpha_sag: float, alpha_front: float, 
                    sigma: float = 2.0, flip: bool = False) -> fem.Function:
        """Apply gluteus maximus load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.gl_max.apply_gaussian_load(force_vector_css=v, 
                                            sigma=sigma, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.gl_max, self.glmax_fun)
        
        return self.glmax_fun
    
    def psoas_force(self, magnitude: float, alpha_sag: float, alpha_front: float, sigma: float = 2.0, flip: bool = False) -> fem.Function:
        """Apply psoas load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.psoas.apply_gaussian_load(force_vector_css=v, 
                                           sigma=sigma, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.psoas, self.psoas_fun)
        
        return self.psoas_fun
    
    def vastus_lateralis_force(self, magnitude: float, alpha_sag: float, alpha_front: float, sigma: float = 2.0, flip: bool = False) -> fem.Function:   
        """Apply vastus lateralis load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.vastus_lateralis.apply_gaussian_load(force_vector_css=v, 
                                           sigma=sigma, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.vastus_lateralis, self.vastus_lateralis_fun)
        
        return self.vastus_lateralis_fun
    
    def vastus_medialis_force(self, magnitude: float, alpha_sag: float, alpha_front: float, sigma: float = 2.0, flip: bool = False) -> fem.Function:   
        """Apply vastus medialis load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.vastus_medialis.apply_gaussian_load(force_vector_css=v, 
                                           sigma=sigma, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.vastus_medialis, self.vastus_medialis_fun)
        
        return self.vastus_medialis_fun
    
    def vastus_intermedius_force(self, magnitude: float, alpha_sag: float, alpha_front: float, sigma: float = 2.0, flip: bool = False) -> fem.Function:   
        """Apply vastus intermedius load. Returns interpolated traction field."""
        # Rank 0 computes the load distribution
        if self.rank == 0:
            v = vector_from_angles(magnitude=magnitude, alpha_sag=alpha_sag, 
                                   alpha_front=alpha_front)
            self.vastus_intermedius.apply_gaussian_load(force_vector_css=v, 
                                           sigma=sigma, flip=flip)
        
        self.comm.Barrier()
        
        # All ranks interpolate using MPI communication
        self._interpolate_mpi(self.vastus_intermedius, self.vastus_intermedius_fun)
        
        return self.vastus_intermedius_fun


    def collect_loads(self) -> fem.Function:
        """Collect all applied loads into a single traction field."""
        total_fun = fem.Function(self.V, name="Total Applied Load")
        
        # Sum the individual load functions
        total_fun.x.array[:] = (self.hip_fun.x.array + 
                                self.glmed_fun.x.array + 
                                self.glmin_fun.x.array + 
                                self.glmax_fun.x.array +
                                self.psoas_fun.x.array + self.vastus_lateralis_fun.x.array +
                                self.vastus_medialis_fun.x.array + self.vastus_intermedius_fun.x.array)
        total_fun.x.scatter_forward()
        
        return total_fun


if __name__ == "__main__":
    from dolfinx import plot
    import pyvista as pv
    from simulation.febio_parser import FEBio2Dolfinx
    
    comm = MPI.COMM_WORLD
    
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    
    loader = Loader(domain)
    hip_fun = loader.hip_force(magnitude=2000.0, alpha_sag=0.0,
                               alpha_front=0.0, sigma_deg=10.0, flip=True)
    glmed_fun = loader.glmed_force(magnitude=500.0, alpha_sag=0.0, 
                                   alpha_front=0.0, sigma=2.0, flip=False)
    
    # Only rank 0 saves VTK output
    if comm.rank == 0:
        topology, cells, geometry = plot.vtk_mesh(loader.V)
        grid = pv.UnstructuredGrid(topology, cells, geometry)
        grid["Hip Joint Load"] = hip_fun.x.array.reshape((-1, 3))
        grid["GL med Load"] = glmed_fun.x.array.reshape((-1, 3))
        grid.save("hip_load.vtu")
        print("Saved hip_load.vtu")