"""MPI-parallel loader for femur surface loads.

Handles pyvista mesh reading and load computation on rank 0,
then broadcasts/scatters results to all MPI ranks for DOLFINx interpolation.
"""

import numpy as np
import dolfinx
import dolfinx.fem as fem
import basix.ufl
from mpi4py import MPI

from simulation.femur_css import FemurCSS, load_json_points
from simulation.paths import FemurPaths
from simulation.femur_loads import HIPJointLoad, MuscleLoad, vector_from_angles


class Loader:
    """MPI-parallel loader for hip and gluteus medius loads.
    
    PyVista mesh reading and load objects are only created on rank 0.
    Load interpolation uses MPI gather/scatter for parallel evaluation.
    """
    
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
        
        # Only rank 0 sets up pyvista and load objects
        self.hip = None
        self.gl_med = None
        
        if self.rank == 0:
            import pyvista as pv
            femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
            head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
            le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
            
            css = FemurCSS(femur_mesh, head_line, le_me_line, side="right")
            
            self.hip = HIPJointLoad(femur_mesh, css, use_cell_data=False)
            self.gl_med = MuscleLoad(femur_mesh, css, use_cell_data=False)
            self.gl_med.set_attachment_points(load_json_points(FemurPaths.GL_MED_JSON))
        
        self.comm.Barrier()
    
    def _interpolate_mpi(self, loader_obj, target_fun: fem.Function) -> None:
        """Interpolate load values safely using 'Owner Computes' pattern.
        
        Only OWNED coordinates are sent to rank 0 for evaluation.
        Ghosts are updated via Dolfinx scatter_forward mechanism.
        """
        # 1. Získání informací o DOFs
        V = target_fun.function_space
        imap = V.dofmap.index_map
        bs = V.dofmap.index_map_bs
        
        # size_local udává počet vlastněných entit (uzlů) v tomto ranku
        n_owned = imap.size_local
        
        # Získání souřadnic všech lokálních uzlů (Owned + Ghosts)
        # Dolfinx standard: prvních 'n_owned' řádků jsou owned uzly.
        all_coords = V.tabulate_dof_coordinates()
        
        # Vyřízneme JEN souřadnice, které tento rank vlastní
        x_owned = all_coords[:n_owned]
        
        # Počet vlastněných uzlů pro MPI komunikaci
        local_n = n_owned
        
        # Gather počtů od všech ranků
        all_n = self.comm.allgather(local_n)
        
        # Displacements pro Gatherv/Scatterv (kam se data uloží v bufferu)
        displs = [sum(all_n[:i]) for i in range(len(all_n))]
        
        if self.rank == 0:
            total_n = sum(all_n)
            
            # Buffer pro příchozí souřadnice (pouze owned od všech)
            recv_coords = np.empty((total_n, 3), dtype=np.float64)
            
            # 1. GATHER: Všichni pošlou své owned souřadnice
            self.comm.Gatherv(
                np.ascontiguousarray(x_owned),
                [recv_coords, [n * 3 for n in all_n], [d * 3 for d in displs], MPI.DOUBLE],
                root=0
            )
            
            # 2. EVALUATE: Rank 0 vyhodnotí zátěž na všech bodech
            # loader_obj je např. HIPJointLoad, vrací (N, 3)
            computed_values = loader_obj(recv_coords) 
            
            # 3. SCATTER: Rozeslání výsledků zpět vlastníkům
            local_values = np.empty((local_n, 3), dtype=np.float64)
            self.comm.Scatterv(
                [computed_values, [n * 3 for n in all_n], [d * 3 for d in displs], MPI.DOUBLE],
                local_values,
                root=0
            )
        else:
            # Worker pošle své owned souřadnice
            self.comm.Gatherv(np.ascontiguousarray(x_owned), None, root=0)
            
            # Worker připraví buffer a přijme výsledky
            local_values = np.empty((local_n, 3), dtype=np.float64)
            self.comm.Scatterv(None, local_values, root=0)
        
        # 4. ASSIGN & SYNC
        # Zápis do pole funkce. Protože jsme pracovali jen s n_owned,
        # zapisujeme přesně do 'owned' části pole (začátek).
        # target_fun.x.array je ploché pole [x0, y0, z0, x1...] nebo blokové,
        # flatten() to srovná správně pro P1 Vector space.
        target_fun.x.array[:n_owned * bs] = local_values.flatten()
        
        # KLÍČOVÝ KROK: Propagace hodnot na ghost uzly sousedních procesů.
        # Tím se vyhladí přechody na partition boundary.
        target_fun.x.scatter_forward()
    
    def hip_force(self, magnitude: float, alpha_sag: float, alpha_front: float, 
                  sigma_deg: float = 10.0, flip: bool = True) -> fem.Function:
        """Apply hip joint load and interpolate to DOLFINx function.
        
        Args:
            magnitude: Force magnitude in N.
            alpha_sag: Sagittal plane angle in degrees.
            alpha_front: Frontal plane angle in degrees.
            sigma_deg: Gaussian spread in degrees (default: 10.0).
            flip: Flip force direction (default: True).
        
        Returns:
            DOLFINx function with interpolated traction field.
        """
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
        """Apply gluteus medius load and interpolate to DOLFINx function.
        
        Args:
            magnitude: Force magnitude in N.
            alpha_sag: Sagittal plane angle in degrees.
            alpha_front: Frontal plane angle in degrees.
            sigma: Gaussian spread in mm (default: 2.0).
            flip: Flip force direction (default: False).
        
        Returns:
            DOLFINx function with interpolated traction field.
        """
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