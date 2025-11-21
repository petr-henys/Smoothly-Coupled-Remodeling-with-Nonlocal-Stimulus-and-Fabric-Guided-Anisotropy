from typing import Optional
from petsc4py import PETSc
import dolfinx.fem.petsc
import ufl

class L2Projector:
    """Reusable L2 projector: builds mass matrix once, solves M*u = RHS."""

    _A: PETSc.Mat  # The mass matrix
    _b: PETSc.Vec  # The rhs vector
    _lhs: dolfinx.fem.Form  # The compiled form for the mass matrix
    _ksp: PETSc.KSP  # The PETSc solver
    _x: dolfinx.fem.Function  # The solution vector
    _dx: ufl.Measure  # Integration measure
    _dS: ufl.Measure  # Integration measure on interface

    def __init__(
        self,
        space: dolfinx.fem.FunctionSpace,
        petsc_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        form_compiler_options: Optional[dict] = None,
        metadata: Optional[dict] = None,
        alpha: float = 1.0,

    ):
        petsc_options = {} if petsc_options is None else petsc_options
        jit_options = {} if jit_options is None else jit_options
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options

        # Assemble projection matrix once
        self.u = ufl.TrialFunction(space)
        self.v = ufl.TestFunction(space)
        self._dx = ufl.Measure("dx", domain=space.mesh, metadata=metadata)
        self._dS = ufl.Measure("dS", domain=space.mesh, metadata=metadata)


        if space.ufl_element().is_cellwise_constant():
            a = ufl.inner(self.u, self.v) * self._dx \
                + alpha * ufl.inner(ufl.jump(self.u), ufl.jump(self.v)) * self._dS
        else:
            a = (ufl.inner(self.u, self.v) + alpha * ufl.inner(ufl.grad(self.u), ufl.grad(self.v))) * self._dx

        self._lhs = dolfinx.fem.form(a, jit_options=jit_options, form_compiler_options=form_compiler_options)
        self._A = dolfinx.fem.petsc.assemble_matrix(self._lhs)
        self._A.assemble()

        # Create vectors to store right hand side and the solution
        self._x = dolfinx.fem.Function(space)
        self._b = dolfinx.fem.Function(space)

        # Create Krylov Subspace solver
        self._ksp = PETSc.KSP().create(space.mesh.comm)
        self._ksp.setOperators(self._A)

        # Set PETSc options
        prefix = f"projector_{id(self)}"
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for k, v in petsc_options.items():
            opts[k] = v
        opts.prefixPop()
        self._ksp.setFromOptions()
        for opt in opts.getAll().keys():
            del opts[opt]

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(prefix)
        self._A.setFromOptions()
        self._b.x.petsc_vec.setOptionsPrefix(prefix)
        self._b.x.petsc_vec.setFromOptions()

    def reassemble_lhs(self):
        dolfinx.fem.petsc.assemble_matrix(self._A, self._lhs)
        self._A.assemble()

    def assemble_rhs(self, h: ufl.core.expr.Expr):
        """
        Assemble the right hand side of the problem
        """
        rhs = ufl.inner(h, self.v) * self._dx
        rhs_compiled = dolfinx.fem.form(rhs)
        self._b.x.array[:] = 0.0
        dolfinx.fem.petsc.assemble_vector(self._b.x.petsc_vec, rhs_compiled)
        self._b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self._b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def project(self, expr, result: Optional[dolfinx.fem.Function] = None) -> dolfinx.fem.Function:
        """Project UFL expression to function space via mass matrix solve."""
        if result is None:
            result = self._x
        
        self.assemble_rhs(expr)
        self._ksp.solve(self._b.x.petsc_vec, result.x.petsc_vec)
        result.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        return result

    def __del__(self):
        if hasattr(self, "_A"):
            self._A.destroy()
        if hasattr(self, "_ksp"):
            self._ksp.destroy()