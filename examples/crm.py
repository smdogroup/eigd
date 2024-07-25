import logging
import os
import time

from eigd import (
    IRAM,
    BasicLanczos,
    SpLuOperator,
    add_eig_total_derivative,
)
from mpi4py import MPI
import numpy as np
import scipy

import tacs
from tacs import TACS, constitutive, elements, pyTACS


class CRM:

    def __init__(
        self,
        comm=MPI.COMM_WORLD,
        N=10,
        m=None,
        omega0=10.0,
        solver_type="BasicLanczos",
        tol=1e-14,
        rtol=1e-10,
        eig_atol=1e-5,
        adjoint_method="shift-invert",
        adjoint_options={},
        cost=1,
    ):
        self.N = N
        self.m = m
        self.omega0 = omega0
        self.solver_type = solver_type
        self.rtol = rtol
        self.tol = tol
        self.eig_atol = eig_atol
        self.adjoint_method = adjoint_method
        self.adjoint_options = adjoint_options
        self.cost = cost

        self.assembler = self._create_crm_model(comm)

        # Output for visualization
        flag = (
            TACS.OUTPUT_CONNECTIVITY
            | TACS.OUTPUT_NODES
            | TACS.OUTPUT_DISPLACEMENTS
            | TACS.OUTPUT_STRAINS
            | TACS.OUTPUT_STRESSES
            | TACS.OUTPUT_EXTRAS
            | TACS.OUTPUT_LOADS
        )
        self.f5 = TACS.ToFH5(self.assembler, TACS.BEAM_OR_SHELL_ELEMENT, flag)

        return

    def _create_crm_model(self, comm):

        # Instantiate FEAAssembler
        structOptions = {
            "printtiming": True,
        }

        root = os.path.split(os.path.dirname(tacs.__file__))[0]
        bdfFile = os.path.join(root, "examples", "crm", "CRM_box_2nd.bdf")
        FEA = pyTACS(bdfFile, options=structOptions, comm=comm)

        # Material properties
        rho = 2780.0  # density kg/m^3
        E = 73.1e9  # Young's modulus (Pa)
        nu = 0.33  # Poisson's ratio
        ys = 324.0e6  # yield stress

        # Setup (isotropic) property and constitutive objects
        self.prop = constitutive.MaterialProperties(rho=rho, E=E, nu=nu, ys=ys)

        # Callback function used to setup TACS element objects and DVs
        def elemCallBack(
            dvNum, compID, compDescript, elemDescripts, globalDVs, **kwargs
        ):
            # Shell thickness
            t = 0.01  # m
            tMin = 0.002  # m
            tMax = 0.05  # m

            # Set one thickness dv for every component
            con = constitutive.IsoShellConstitutive(self.prop, t=t, tNum=dvNum)

            # Define reference axis for local shell stresses
            if "SKIN" in compDescript:  # USKIN + LSKIN
                sweep = 35.0 / 180.0 * np.pi
                refAxis = np.array([np.sin(sweep), np.cos(sweep), 0])
            else:  # RIBS + SPARS + ENGINE_MOUNT
                refAxis = np.array([0.0, 0.0, 1.0])

            # For each element type in this component,
            # pass back the appropriate tacs element object
            elemList = []
            transform = elements.ShellRefAxisTransform(refAxis)
            for elemDescript in elemDescripts:
                if elemDescript in ["CQUAD4", "CQUADR"]:
                    elem = elements.Quad4Shell(transform, con)
                elif elemDescript in ["CTRIA3", "CTRIAR"]:
                    elem = elements.Tri3Shell(transform, con)
                else:
                    print("Uh oh, '%s' not recognized" % (elemDescript))
                elemList.append(elem)

            # Add scale for thickness dv
            scale = [100.0]
            return elemList, scale

        # Set up elements and TACS assembler
        FEA.initialize(elemCallBack)

        return FEA.assembler

    def _create_matrices(self):

        self.K = self.assembler.createMat()
        self.M = self.assembler.createMat()

        self.assembler.assembleMatType(TACS.STIFFNESS_MATRIX, self.K)
        self.assembler.assembleMatType(TACS.MASS_MATRIX, self.M)

        # Convert to scipy format
        K0, _ = self.K.getMat()
        M0, _ = self.M.getMat()

        K0 = K0.tocsr()
        M0 = M0.tocsr()

        self.ndof = K0.shape[0]
        self.dof = self._create_reduced_indices(K0)

        self.Kr = self._delete_rows_and_columns(K0, self.dof)
        self.Mr = self._delete_rows_and_columns(M0, self.dof)

        return

    def _create_reduced_indices(self, A):
        A = A.tocsr()
        A.eliminate_zeros()

        dof = []
        for k in range(A.shape[0]):
            if (
                A.indptr[k + 1] - A.indptr[k] == 1
                and A.indices[A.indptr[k]] == k
                and np.isclose(A.data[A.indptr[k]], 1.0)
            ):
                # This is a constrained DOF
                pass
            else:
                # Store the free DOF index
                dof.append(k)

        return dof

    def _delete_rows_and_columns(self, A, dof):
        iptr = [0]
        cols = []
        data = []

        indices = -np.ones(A.shape[0])
        indices[dof] = np.arange(len(dof))

        for i in dof:
            for jp in range(A.indptr[i], A.indptr[i + 1]):
                j = A.indices[jp]

                if indices[j] >= 0:
                    cols.append(indices[j])
                    data.append(A.data[jp])

            iptr.append(len(cols))

        return scipy.sparse.csr_matrix((data, cols, iptr), shape=(len(dof), len(dof)))

    def write_eigenvectors(self):
        for i in range(self.N):
            u = self.Q[:, i]
            self.write_output(u, filename="results/crm/output_%d.f5" % i)
        return

    def write_output(self, u, filename="output.f5"):
        u0 = self.assembler.createVec()
        u0_array = u0.getArray()
        u0_array[self.dof] = u
        self.assembler.setVariables(u0)
        self.f5.writeToFile(filename)
        return

    def get_design_vars(self):
        x = self.assembler.createDesignVec()
        self.assembler.getDesignVars(x)
        x_array = x.getArray()
        return np.array(x_array)

    def set_design_vars(self, x0):
        x = self.assembler.createDesignVec()
        x_array = x.getArray()
        x_array[:] = x0
        self.assembler.setDesignVars(x)
        return

    def initialize(self):
        self.profile = {}
        self.profile["solver_type"] = self.solver_type
        self.profile["adjoint_method"] = self.adjoint_method
        self.profile["adjoint_options"] = self.adjoint_options
        self.profile["N"] = self.N

        self._create_matrices()

        sigma = self.omega0**2
        mat = self.Kr - sigma * self.Mr
        mat = mat.tocsc()
        self.factor = SpLuOperator(mat)

        logging.info("Solve eigenvalue problem")
        t1 = time.time()

        for i in range(self.cost):
            if self.solver_type == "IRAM":
                if self.m is None:
                    self.m = max(2 * self.N + 1, 60)
                self.eig_solver = IRAM(N=self.N, m=self.m, eig_atol=self.eig_atol)
                self.lam, self.Q = self.eig_solver.solve(
                    self.Kr, self.Mr, self.factor, sigma
                )
            else:
                if self.m is None:
                    self.m = max(3 * self.N + 1, 60)
                self.eig_solver = BasicLanczos(
                    N=self.N, m=self.m, eig_atol=self.eig_atol, tol=self.tol
                )
                self.lam, self.Q = self.eig_solver.solve(
                    self.Kr, self.Mr, self.factor, sigma
                )

        t2 = time.time()
        t = (t2 - t1) / self.cost
        self.profile["eigenvalue solve time"] = t
        logging.info("Eigenvalue solve time: %5.2f s" % t)
        logging.info("lam = %s" % self.lam)
        self.profile["m"] = self.m
        self.profile["eig_solver.m"] = str(self.eig_solver.m)
        print("eig_solver.m = %d" % self.eig_solver.m)

        # self.write_eigenvectors()
        # exit()

        return

    def initialize_adjoint(self):
        self.Qb = np.zeros(self.Q.shape)
        self.lamb = np.zeros(self.lam.shape)

        return

    def get_compliance(self):
        """Get the compliance"""

        f = np.zeros(self.ndof)
        f[1::6] = 1.0
        fr = f[self.dof]

        # Compute the compliance
        compliance = 0.0
        for i in range(self.N):
            val = self.Q[:, i].dot(fr)
            compliance += (val * val) / self.lam[i]

        return compliance

    def add_compliance_derivative(self, compb=1.0):

        f = np.zeros(self.ndof)
        f[1::6] = 1.0
        fr = f[self.dof]

        for i in range(self.N):
            val = self.Q[:, i].dot(fr)
            self.Qb[:, i] += 2.0 * compb * val * fr / self.lam[i]
            self.lamb[i] -= compb * (val * val) / self.lam[i] ** 2

        return

    def finalize_adjoint(self):

        class Callback:
            def __init__(self):
                self.res_list = []

            def __call__(self, rk=None):
                self.res_list.append(rk)

        callback = Callback()

        self.profile["adjoint solution method"] = self.adjoint_method
        self.factor.count = 0

        # Solve the adjoint problem
        t0 = time.time()
        for i in range(self.cost):
            if i != 0:
                callback.res_list = []
            psi, corr_data = self.eig_solver.solve_adjoint(
                self.Qb,
                rtol=self.rtol,
                method=self.adjoint_method,
                callback=callback,
                **self.adjoint_options,
            )
        t1 = time.time()
        t = (t1 - t0) / self.cost

        # Hang on to the adjoint variables
        self.psi = psi

        self.profile["adjoint preconditioner count"] = self.factor.count
        self.profile["adjoint solution time"] = t
        self.profile["adjoint residuals"] = np.array(callback.res_list).tolist()
        self.profile["adjoint correction data"] = corr_data
        logging.info("Adjoint solve time: %8.2f s" % t)

        # Set up the info to compute the total derivative
        dfdx = self.assembler.createDesignVec()
        dfdx_array = dfdx.getArray()
        w = self.assembler.createVec()
        v = self.assembler.createVec()
        w_array = w.getArray()
        v_array = v.getArray()

        grad = np.zeros(dfdx_array.shape)

        def dAdx(wr, vr):
            w_array[self.dof] = wr
            v_array[self.dof] = vr
            dfdx.zeroEntries()
            self.assembler.addMatDVSensInnerProduct(
                1.0, TACS.STIFFNESS_MATRIX, w, v, dfdx
            )
            return np.array(dfdx.getArray())

        def dBdx(wr, vr):
            w_array[self.dof] = wr
            v_array[self.dof] = vr
            dfdx.zeroEntries()
            self.assembler.addMatDVSensInnerProduct(1.0, TACS.MASS_MATRIX, w, v, dfdx)
            return np.array(dfdx.getArray())

        # Compute the total derivative
        self.grad = add_eig_total_derivative(
            self.lam,
            self.Q,
            self.lamb,
            self.Qb,
            psi,
            dAdx,
            dBdx,
            grad,
            adj_corr_data=corr_data,
        )

        t2 = time.time()
        self.profile["total derivative time"] = t2 - t1
        logging.info("Total derivative time: %5.2f s" % (t2 - t1))

        return


if __name__ == "__main__":
    # enable logging printout
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    crm = CRM(solver_type="BasicLanczos", adjoint_method="dl", m=30, N=10, cost=1)

    dh = 1e-6
    x0 = crm.get_design_vars()

    crm.initialize()
    c0 = crm.get_compliance()
    crm.initialize_adjoint()
    crm.add_compliance_derivative()
    crm.finalize_adjoint()

    pert = np.random.uniform(size=x0.shape)

    ans = pert.dot(crm.grad)
    x1 = x0 + dh * pert

    crm.set_design_vars(x1)
    crm.initialize()
    c1 = crm.get_compliance()

    fd = (c1 - c0) / dh
    print("Ans = ", ans)
    print("FD  = ", fd)
    print("Rel err = ", (ans - fd) / fd)
