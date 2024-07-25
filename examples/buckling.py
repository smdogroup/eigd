import time

from fe_utils import populate_Be_and_Te
import matplotlib.pylab as plt
import matplotlib.tri as tri
from node_filter import NodeFilter
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse import linalg

from eigd import IRAM, BasicLanczos, SpLuOperator, eval_adjoint_residual_norm


class TopologyAnalysis:

    def __init__(
        self,
        fltr,
        conn,
        X,
        bcs,
        forces={},
        E=1.0,
        nu=0.3,
        ptype_K="simp",
        ptype_M="simp",
        ptype_G="simp",
        rho0_K=1e-6,
        rho0_M=1e-9,
        rho0_G=1e-9,
        p=3.0,
        q=5.0,
        density=1.0,
        sigma=3.0,
        N=10,
        m=None,
        solver_type="IRAM",
        tol=0.0,
        rtol=1e-10,
        eig_atol=1e-5,
        adjoint_method="shift-invert",
        adjoint_options={},
        cost=1,
        deriv_type="tensor",
    ):
        self.ptype_K = ptype_K.lower()
        self.ptype_M = ptype_M.lower()
        self.ptype_G = ptype_G.lower()

        self.rho0_K = rho0_K
        self.rho0_M = rho0_M
        self.rho0_G = rho0_G

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.p = p
        self.q = q
        self.density = density
        self.sigma = sigma  # Shift value
        self.N = N  # Number of modes
        self.m = m
        self.solver_type = solver_type
        self.tol = tol
        self.rtol = rtol
        self.eig_atol = eig_atol
        self.adjoint_method = adjoint_method
        self.adjoint_options = adjoint_options
        self.cost = cost
        self.deriv_type = deriv_type

        self.bcs = bcs
        self.forces = forces

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = 2 * self.nnodes

        # Set the initial design variable values
        self.x = 0.5 * np.ones(self.fltr.num_design_vars)
        self.xb = np.zeros(self.x.shape)

        self.Q = None
        self.lam = None

        # Compute the constitutive matrix
        self.E = E
        self.nu = nu
        self.C0 = E * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )
        self.C0 *= 1.0 / (1.0 - nu**2)

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)
        self.f = self._compute_forces(self.nvars, forces)

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 8), dtype=int)
        self.var[:, ::2] = 2 * self.conn
        self.var[:, 1::2] = 2 * self.conn + 1

        self.dfds = None
        self.pp = None

        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.var[index, :]:
                for jj in self.var[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        self.i = np.array(i, dtype=int)
        self.j = np.array(j, dtype=int)

        self._init_profile()
        return

    def _compute_reduced_variables(self, nvars, bcs):
        """
        Compute the reduced set of variables
        """
        reduced = list(range(nvars))

        # For each node that is in the boundary condition dictionary
        for node in bcs:
            uv_list = bcs[node]

            # For each index in the boundary conditions (corresponding to
            # either a constraint on u and/or constraint on v
            for index in uv_list:
                var = 2 * node + index
                reduced.remove(var)

        return reduced

    def _compute_forces(self, nvars, forces):
        """
        Unpack the dictionary containing the forces
        """
        f = np.zeros(nvars)

        for node in forces:
            f[2 * node] += forces[node][0]
            f[2 * node + 1] += forces[node][1]

        return f

    def get_stiffness_matrix(self, rhoE):
        """
        Assemble the stiffness matrix
        """

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ke = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]
            Ke += detJ[:, np.newaxis, np.newaxis] * Be.transpose(0, 2, 1) @ C @ Be

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def get_stiffness_matrix_deriv(self, rhoE, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """

        dfdrhoE = np.zeros(self.nelems)

        # The element-wise variables
        ue = np.zeros((self.nelems, 8) + u.shape[1:])
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])

        ue[:, ::2, ...] = u[2 * self.conn, ...]
        ue[:, 1::2, ...] = u[2 * self.conn + 1, ...]

        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]

            if psi.ndim == 1 and u.ndim == 1:
                se = np.einsum("nij,nj -> ni", Be, psie)
                te = np.einsum("nij,nj -> ni", Be, ue)
                dfdrhoE += detJ * np.einsum("ij,nj,ni -> n", self.C0, se, te)
            elif psi.ndim == 2 and u.ndim == 2:
                se = Be @ psie
                te = Be @ ue
                dfdrhoE += detJ * np.einsum("ij,njk,nik -> n", self.C0, se, te)

        if self.ptype_K == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1.0)
        else:  # ramp
            dfdrhoE[:] *= (1.0 + self.q) / (1.0 + self.q * (1.0 - rhoE)) ** 2

        dKdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dKdrho, self.conn[:, i], dfdrhoE)
        dKdrho *= 0.25

        return dKdrho

    def get_stress_stiffness_matrix(self, rhoE, u):
        """
        Assemble the stess stiffness matrix
        """

        # Get the element-wise solution variables
        ue = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        Ge = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            # Compute the stresses in each element
            s = np.einsum("nij,njk,nk -> ni", C, Be, ue)

            G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
            Ge[:, 0::2, 0::2] += G0e
            Ge[:, 1::2, 1::2] += G0e

        G = sparse.coo_matrix((Ge.flatten(), (self.i, self.j)))
        G = G.tocsr()

        return G

    def intital_Be_and_Te(self):
        # Compute gauss points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # Compute Be and Te, detJ
        Be = np.zeros((self.nelems, 3, 8, 4))
        Te = np.zeros((self.nelems, 3, 4, 4, 4))
        detJ = np.zeros((self.nelems, 4))

        for j in range(2):
            for i in range(2):
                xi, eta = gauss_pts[i], gauss_pts[j]
                index = 2 * j + i
                Bei = Be[:, :, :, index]
                Tei = Te[:, :, :, :, index]

                detJ[:, index] = populate_Be_and_Te(
                    self.nelems, xi, eta, xe, ye, Bei, Tei
                )

        return Be, Te, detJ

    def intital_stress_stiffness_matrix_deriv(self, rhoE, Te, detJ, psi, phi):

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        self.C = C.reshape((self.nelems, 3, 3))

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])
        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        phie = np.zeros((self.nelems, 8) + phi.shape[1:])
        phie[:, ::2, ...] = phi[2 * self.conn, ...]
        phie[:, 1::2, ...] = phi[2 * self.conn + 1, ...]

        pp0 = psie[:, ::2] @ phie[:, ::2].transpose(0, 2, 1)
        pp1 = psie[:, 1::2] @ phie[:, 1::2].transpose(0, 2, 1)

        se = np.einsum("nijlm,njl -> nim", Te, (pp0 + pp1))
        dfds = detJ[:, np.newaxis, :] * se

        return dfds

    def get_stress_stiffness_matrix_uderiv_tensor(self, dfds, Be):

        Cdfds = self.C @ dfds
        dfdue = np.einsum("nijm,nim -> nj", Be, Cdfds)

        dfdu = np.zeros(2 * self.nnodes)
        np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
        np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])

        return dfdu

    def get_stress_stiffness_matrix_xderiv_tensor(self, rhoE, u, dfds, Be):

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        dfds = np.einsum("nim, ij -> njm", dfds, self.C0)
        dfdrhoE = np.einsum("njm,njkm,nk -> n", dfds, Be, ue)

        if self.ptype_G == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1)
        else:  # ramp
            dfdrhoE[:] *= (2.0 + self.q) / (1.0 + (self.q + 1.0) * (1.0 - rhoE)) ** 2

        dfdrho = np.zeros(self.nnodes)
        np.add.at(dfdrho, self.conn, dfdrhoE[:, np.newaxis])
        dfdrho *= 0.25

        return dfdrho

    def get_stress_stiffness_matrix_uderiv(self, rhoE, psi, phi):
        """
        Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

        Note "solver" returns the solution of the system of equations

        K * sol = rhs

        Given the right-hand-side rhs. ie. sol = solver(rhs)
        """

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])
        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        phie = np.zeros((self.nelems, 8) + phi.shape[1:])
        phie[:, ::2, ...] = phi[2 * self.conn, ...]
        phie[:, 1::2, ...] = phi[2 * self.conn + 1, ...]

        dfdue = np.zeros((self.nelems, 8))

        # Compute the element stress stiffness matrix
        Be = np.zeros((self.nelems, 3, 8))
        Te = np.zeros((self.nelems, 3, 4, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        if psi.ndim == 2 and phi.ndim == 2 and self.pp is None:
            pp0 = psie[:, ::2] @ phie[:, ::2].transpose(0, 2, 1)
            pp1 = psie[:, 1::2] @ phie[:, 1::2].transpose(0, 2, 1)
            self.pp = pp0 + pp1

        for xi, eta in [(xi, eta) for xi in gauss_pts for eta in gauss_pts]:
            detJ = populate_Be_and_Te(self.nelems, xi, eta, xe, ye, Be, Te)

            if psi.ndim == 1 and phi.ndim == 1:
                se0 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, ::2], phie[:, ::2])
                se1 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, 1::2], phie[:, 1::2])
                se = se0 + se1

            elif psi.ndim == 2 and phi.ndim == 2:
                se = np.einsum("nijl,njl -> ni", Te, self.pp)

            # Add contributions to the derivative w.r.t. u
            dfds = detJ[:, np.newaxis] * se
            BeC = np.matmul(Be.transpose(0, 2, 1), C)
            dfdue += np.einsum("njk,nk -> nj", BeC, dfds)

        dfdu = np.zeros(2 * self.nnodes)
        np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
        np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])

        return dfdu

    def get_stress_stiffness_matrix_xderiv(self, rhoE, u, psi, phi):
        """
        Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

        Note "solver" returns the solution of the system of equations

        K * sol = rhs

        Given the right-hand-side rhs. ie. sol = solver(rhs)
        """

        dfdrhoE = np.zeros(self.nelems)

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])
        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        phie = np.zeros((self.nelems, 8) + phi.shape[1:])
        phie[:, ::2, ...] = phi[2 * self.conn, ...]
        phie[:, 1::2, ...] = phi[2 * self.conn + 1, ...]

        # Compute the element stress stiffness matrix
        Be = np.zeros((self.nelems, 3, 8))
        Te = np.zeros((self.nelems, 3, 4, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        if psi.ndim == 2 and phi.ndim == 2 and self.pp is None:
            pp0 = psie[:, ::2] @ phie[:, ::2].transpose(0, 2, 1)
            pp1 = psie[:, 1::2] @ phie[:, 1::2].transpose(0, 2, 1)
            self.pp = pp0 + pp1

        for xi, eta in [(xi, eta) for xi in gauss_pts for eta in gauss_pts]:
            detJ = populate_Be_and_Te(self.nelems, xi, eta, xe, ye, Be, Te)

            if psi.ndim == 1 and phi.ndim == 1:
                se0 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, ::2], phie[:, ::2])
                se1 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, 1::2], phie[:, 1::2])
                se = se0 + se1

            elif psi.ndim == 2 and phi.ndim == 2:
                se = np.einsum("nijl,njl -> ni", Te, self.pp)

            dfds = detJ[:, np.newaxis] * se @ self.C0
            dfdrhoE += np.einsum("nj,njk,nk -> n", dfds, Be, ue)

        if self.ptype_G == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1)
        else:  # ramp
            dfdrhoE[:] *= (2.0 + self.q) / (1.0 + (self.q + 1.0) * (1.0 - rhoE)) ** 2

        dfdrho = np.zeros(self.nnodes)
        np.add.at(dfdrho, self.conn, dfdrhoE[:, np.newaxis])
        dfdrho *= 0.25

        return dfdrho

    def eval_area(self):
        return np.sum(self.detJ.reshape(-1) * np.tile(self.rhoE, 4))

    def eval_area_gradient(self):
        dfdrhoE = np.sum(self.detJ, axis=1)

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.apply_gradient(dfdrho, self.x)

    def reduce_vector(self, forces):
        """
        Eliminate essential boundary conditions from the vector
        """
        return forces[self.reduced]

    def reduce_matrix(self, matrix):
        """
        Eliminate essential boundary conditions from the matrix
        """
        temp = matrix[self.reduced, :]
        return temp[:, self.reduced]

    def full_vector(self, vec):
        """
        Transform from a reduced vector without dirichlet BCs to the full vector
        """
        temp = np.zeros((self.nvars,) + vec.shape[1:], dtype=vec.dtype)
        temp[self.reduced, ...] = vec[:, ...]
        return temp

    def full_matrix(self, mat):
        """
        Transform from a reduced matrix without dirichlet BCs to the full matrix
        """
        temp = np.zeros((self.nvars, self.nvars), dtype=mat.dtype)
        for i in range(len(self.reduced)):
            for j in range(len(self.reduced)):
                temp[self.reduced[i], self.reduced[j]] = mat[i, j]
        return temp

    def _init_profile(self):
        self.profile = {}
        self.profile["nnodes"] = self.nnodes
        self.profile["nelems"] = self.nelems
        self.profile["solver_type"] = self.solver_type
        self.profile["adjoint_method"] = self.adjoint_method
        self.profile["adjoint_options"] = self.adjoint_options
        self.profile["N"] = self.N
        self.profile["E"] = self.E
        self.profile["nu"] = self.nu
        self.profile["density"] = self.density
        self.profile["p"] = self.p
        self.profile["eig_atol"] = self.eig_atol
        self.profile["ftype"] = self.fltr.ftype
        self.profile["r0"] = self.fltr.r0

        return

    def solve_eigenvalue_problem(self, rhoE, store=False):
        """
        Compute the smallest buckling load factor BLF
        """

        t0 = time.time()

        K = self.get_stiffness_matrix(rhoE)
        self.Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        self.Kfact = linalg.factorized(self.Kr.tocsc())
        ur = self.Kfact(fr)
        self.u = self.full_vector(ur)

        # Find the gemoetric stiffness matrix
        G = self.get_stress_stiffness_matrix(rhoE, self.u)
        self.Gr = self.reduce_matrix(G)

        t1 = time.time()
        self.profile["matrix assembly time"] += t1 - t0

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        for i in range(self.cost):
            if self.N >= self.nvars:
                mu, self.Q = eigh(self.Gr.todense(), self.Kr.todense())
                mu, self.Qr = mu[: self.N], self.Qr[:, : self.N]
            else:
                self.profile["sigma"] = self.sigma if i == 0 else None

                # Compute the shifted operator
                mat = self.Kr + self.sigma * self.Gr
                mat = mat.tocsc()
                self.factor = SpLuOperator(mat)
                self.factor.count = 0

                if self.solver_type == "IRAM":
                    if self.m is None:
                        self.m = max(2 * self.N + 1, 60)
                    self.eig_solver = IRAM(
                        N=self.N, m=self.m, eig_atol=self.eig_atol, mode="buckling"
                    )
                    mu, self.Qr = self.eig_solver.solve(
                        self.Gr, self.Kr, self.factor, self.sigma
                    )
                else:
                    if self.m is None:
                        self.m = max(3 * self.N + 1, 60)
                    self.eig_solver = BasicLanczos(
                        N=self.N,
                        m=self.m,
                        eig_atol=self.eig_atol,
                        tol=self.tol,
                        mode="buckling",
                    )
                    mu, self.Qr = self.eig_solver.solve(
                        self.Gr,
                        self.Kr,
                        self.factor,
                        self.sigma,
                    )

                    if store:
                        self.profile["eig_res"] = self.eig_solver.eig_res.tolist()

                self.profile["solve preconditioner count"] += (
                    self.factor.count if i == 0 else 0
                )

        t2 = time.time()
        t = (t2 - t1) / self.cost

        self.profile["eigenvalue solve time"] += t
        self.profile["m"] = self.m
        self.profile["eig_solver.m"] = str(self.eig_solver.m)
        self.BLF = mu[: self.N]

        # project the eigenvectors back to the full space
        Q = np.zeros((self.nvars, self.N), dtype=self.rhoE.dtype)
        Q[self.reduced, ...] = self.Qr[:, ...]

        return mu, Q

    def compliance(self):
        return self.f.dot(self.u)

    def compliance_derivative(self):
        dfdrho = -1.0 * self.get_stiffness_matrix_deriv(self.rhoE, self.u, self.u)
        return self.fltr.apply_gradient(dfdrho, self.x)

    def eval_ks_buckling(self, ks_rho=160.0):
        mu = 1 / self.BLF
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        ks_min = c + np.log(np.sum(eta)) / ks_rho
        return ks_min

    def eval_ks_buckling_derivative(self, ks_rho=160.0):
        t0 = time.time()
        mu = 1 / self.BLF
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        eta = eta / np.sum(eta)

        dfdrho = np.zeros(self.nnodes)
        if self.deriv_type == "vector":
            for i in range(self.N):
                dKdx = self.get_stiffness_matrix_deriv(
                    self.rhoE, self.Q[:, i], self.Q[:, i]
                )
                dGdx = self.get_stress_stiffness_matrix_xderiv(
                    self.rhoE, self.u, self.Q[:, i], self.Q[:, i]
                )

                dGdu = self.get_stress_stiffness_matrix_uderiv(
                    self.rhoE, self.Q[:, i], self.Q[:, i]
                )
                dGdur = self.reduce_vector(dGdu)
                adjr = -self.Kfact(dGdur)
                adj = self.full_vector(adjr)

                dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)

                dfdrho -= eta[i] * (dGdx + mu[i] * dKdx)

        elif self.deriv_type == "tensor":
            eta_Q = (eta[:, np.newaxis] * self.Q.T).T
            eta_mu_Q = (eta[:, np.newaxis] * mu[:, np.newaxis] * self.Q.T).T

            dKdx = self.get_stiffness_matrix_deriv(self.rhoE, eta_mu_Q, self.Q)

            dfds = self.intital_stress_stiffness_matrix_deriv(
                self.rhoE, self.Te, self.detJ, eta_Q, self.Q
            )
            dGdu = self.get_stress_stiffness_matrix_uderiv_tensor(dfds, self.Be)
            dGdur = self.reduce_vector(dGdu)
            adjr = -self.Kfact(dGdur)
            adj = self.full_vector(adjr)

            dGdx = self.get_stress_stiffness_matrix_xderiv_tensor(
                self.rhoE, self.u, dfds, self.Be
            )
            dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)

            dfdrho -= dGdx + dKdx

        t1 = time.time()
        self.profile["total derivative time"] += t1 - t0

        return self.fltr.apply_gradient(dfdrho, self.x)

    def get_eigenvector_aggregate(self, rho, node, mode="tanh"):
        if mode == "exp":
            eta = np.exp(-rho * (self.lam - np.min(self.lam)))
        else:
            lam_a = 0.0
            lam_b = 50.0

            a = np.tanh(rho * (self.lam - lam_a))
            b = np.tanh(rho * (self.lam - lam_b))
            eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        # print(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * np.dot(self.Q[node, i], self.Q[node, i])

        return h

    def add_eigenvector_aggregate_derivative(self, hb, rho, node, mode="tanh"):
        if mode == "exp":
            eta = np.exp(-rho * (self.lam - np.min(self.lam)))
        else:
            lam_a = 0.0
            lam_b = 50.0

            a = np.tanh(rho * (self.lam - lam_a))
            b = np.tanh(rho * (self.lam - lam_b))
            eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * np.dot(self.Q[node, i], self.Q[node, i])

        Qb = np.zeros(self.Q.shape)
        for i in range(self.N):
            Qb[node, i] += 2.0 * hb * eta[i] * self.Q[node, i]
            self.Qrb[:, i] += Qb[self.reduced, i]

            if mode == "exp":
                self.lamb[i] -= (
                    hb * rho * eta[i] * (np.dot(self.Q[node, i], self.Q[node, i]) - h)
                )
            else:
                self.lamb[i] -= (
                    hb
                    * rho
                    * eta[i]
                    * (a[i] + b[i])
                    * (np.dot(self.Q[node, i], self.Q[node, i]) - h)
                )

        return

    def KSmax(self, q, ks_rho):
        c = np.max(q)
        eta = np.exp(ks_rho * (q - c))
        ks_max = c + np.log(np.sum(eta)) / ks_rho
        return ks_max

    def eigenvector_aggregate_magnitude(self, rho, node):
        # Tanh aggregate
        lam_a = 0.0
        lam_b = 1000.0
        a = np.tanh(rho * (self.lam - lam_a))
        b = np.tanh(rho * (self.lam - lam_b))
        eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * self.Q[node, i] ** 2

        return h, eta, a, b

    def get_eigenvector_aggregate_max(self, rho, node):
        h, _, _, _ = self.eigenvector_aggregate_magnitude(rho, node)
        h = self.KSmax(h, rho)
        return h

    def add_eigenvector_aggregate_max_derivative(self, hb, rho, node):
        h_mag, eta, a, b = self.eigenvector_aggregate_magnitude(rho, node)

        eta_h = np.exp(rho * (h_mag - np.max(h_mag)))
        eta_h = eta_h / np.sum(eta_h)

        h = np.dot(eta_h, h_mag)

        def D(q):

            nn = len(q)
            eta_Dq = np.zeros(nn)

            for i in range(nn):
                eta_Dq[i] = eta_h[i] * q[i]
            return eta_Dq

        Qb = np.zeros(self.Q.shape)
        for i in range(self.N):
            Qb[node, i] += 2.0 * hb * eta[i] * D(self.Q[node, i])
            self.Qrb[:, i] += Qb[self.reduced, i]
            self.lamb[i] -= (
                hb
                * rho
                * eta[i]
                * (a[i] + b[i])
                * (self.Q[node, i].T @ D(self.Q[node, i]) - h)
            )

        return

    def initialize(self, store=False):
        self.profile["total derivative time"] = 0.0
        self.profile["adjoint solution time"] = 0.0
        self.profile["matrix assembly time"] = 0.0
        self.profile["eigenvalue solve time"] = 0.0
        self.profile["solve preconditioner count"] = 0
        self.profile["adjoint preconditioner count"] = 0

        # Apply the filter
        self.rho = self.fltr.apply(self.x)

        # Average the density to get the element-wise density
        self.rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
        )

        self.Be, self.Te, self.detJ = self.intital_Be_and_Te()

        # Solve the eigenvalue problem
        self.lam, self.Q = self.solve_eigenvalue_problem(self.rhoE, store)

        if store:
            self.profile["eigenvalues"] = self.BLF.tolist()

        return

    def initialize_adjoint(self):
        self.xb = np.zeros(self.x.shape)
        self.rhob = np.zeros(self.nnodes)

        self.lamb = np.zeros(self.lam.shape)
        self.Qrb = np.zeros(self.Qr.shape)

        return

    def check_adjoint_residual(self, A, B, lam, Q, Qb, psi, b_ortho=False):
        res, orth = eval_adjoint_residual_norm(A, B, lam, Q, Qb, psi, b_ortho=b_ortho)
        for i in range(Q.shape[1]):
            ratio = orth[i] / np.linalg.norm(Q[:, i])
            self.profile["adjoint norm[%2d]" % i] = res[i]
            self.profile["adjoint ortho[%2d]" % i] = ratio
            self.profile["adjoint lam[%2d]" % i] = lam[i]

        return res

    def add_check_adjoint_residual(self):
        return self.check_adjoint_residual(
            self.Gr, self.Kr, self.lam, self.Qr, self.Qrb, self.psir, b_ortho=False
        )

    def finalize_adjoint(self):

        class Callback:
            def __init__(self):
                self.res_list = []

            def __call__(self, rk=None):
                self.res_list.append(rk)

        callback = Callback()

        self.profile["adjoint solution method"] = self.adjoint_method
        self.factor.count = 0

        t0 = time.time()
        for i in range(self.cost):
            if i != 0:
                callback.res_list = []
            psir, corr_data = self.eig_solver.solve_adjoint(
                self.Qrb,
                rtol=self.rtol,
                method=self.adjoint_method,
                callback=callback,
                **self.adjoint_options,
            )
        t1 = time.time()
        t = (t1 - t0) / self.cost

        self.psir = psir

        self.profile["adjoint preconditioner count"] += self.factor.count
        self.profile["adjoint solution time"] += t
        self.profile["adjoint residuals"] = np.array(callback.res_list).tolist()
        self.profile["adjoint iterations"] = len(callback.res_list)
        self.profile["adjoint correction data"] = corr_data

        def dAdu(wr, vr):
            w = self.full_vector(wr)
            v = self.full_vector(vr)
            if w.ndim == 1 and v.ndim == 1:
                return self.get_stress_stiffness_matrix_uderiv(self.rhoE, w, v)

            elif w.ndim == 2 and v.ndim == 2:
                if self.dfds is None:
                    self.dfds = self.intital_stress_stiffness_matrix_deriv(
                        self.rhoE, self.Te, self.detJ, w, v
                    )
                return self.get_stress_stiffness_matrix_uderiv_tensor(
                    self.dfds, self.Be
                )

        dBdu = None

        # Compute the derivative of the function wrt the fundamental path
        # Compute the adjoint for K * adj = d ( psi^{T} * G(u, x) * phi ) / du
        dfdu0 = np.zeros(2 * self.nnodes)
        dfdu0 = self.eig_solver.add_total_derivative(
            self.lamb,
            self.Qrb,
            self.psir,
            dAdu,
            dBdu,
            dfdu0,
            adj_corr_data=corr_data,
            deriv_type=self.deriv_type,
        )

        # Create functions for computing dA/dx and dB/dx
        def dAdx(wr, vr):
            w = self.full_vector(wr)
            v = self.full_vector(vr)

            if w.ndim == 1 and v.ndim == 1:
                return self.get_stress_stiffness_matrix_xderiv(self.rhoE, self.u, w, v)

            elif w.ndim == 2 and v.ndim == 2:
                if self.dfds is None:
                    self.dfds = self.intital_stress_stiffness_matrix_deriv(
                        self.rhoE, self.Te, self.detJ, w, v
                    )
                return self.get_stress_stiffness_matrix_xderiv_tensor(
                    self.rhoE, self.u, self.dfds, self.Be
                )

        def dBdx(wr, vr):
            w = self.full_vector(wr)
            v = self.full_vector(vr)
            return self.get_stiffness_matrix_deriv(self.rhoE, w, v)

        self.rhob = self.eig_solver.add_total_derivative(
            self.lamb,
            self.Qrb,
            self.psir,
            dAdx,
            dBdx,
            self.rhob,
            adj_corr_data=corr_data,
            deriv_type=self.deriv_type,
        )

        # Solve the adjoint for the fundamental path
        dfdur = self.reduce_vector(dfdu0)
        psir = -self.Kfact(dfdur)
        psi = self.full_vector(psir)

        self.rhob += self.get_stiffness_matrix_deriv(self.rhoE, psi, self.u)

        self.xb += self.fltr.apply_gradient(self.rhob, self.x)

        t2 = time.time()
        self.profile["total derivative time"] += t2 - t1

        return

    def test_eigenvector_aggregate_derivatives(
        self, rho=100, dh_cd=1e-4, dh_cs=1e-20, node=None, pert=None, mode="tanh"
    ):

        hb = 1.0
        if node is None:
            node = (8 + 1) * 16 + 16

        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        self.initialize_adjoint()
        self.add_eigenvector_aggregate_derivative(hb, rho, node, mode=mode)
        self.finalize_adjoint()

        if pert is None:
            pert = np.random.uniform(size=self.x.shape)

        data = {}
        data["ans"] = np.dot(pert, self.xb)
        data.update(self.profile)

        if self.solver_type == "BasicLanczos":
            # Perturb the design variables for complex-step
            self.x = np.array(x0).astype(complex)
            self.x.imag += dh_cs * pert
            self.initialize()
            h1 = self.get_eigenvector_aggregate(rho, node, mode=mode)

            data["dh_cs"] = dh_cs
            data["cs"] = h1.imag / dh_cs
            data["cs_err"] = np.fabs((data["ans"] - data["cs"]) / data["cs"])

        self.x = x0 - dh_cd * pert
        self.initialize()
        h3 = self.get_eigenvector_aggregate(rho, node, mode=mode)

        self.x = x0 + dh_cd * pert
        self.initialize()
        h4 = self.get_eigenvector_aggregate(rho, node, mode=mode)

        data["dh_cd"] = dh_cd
        data["cd"] = (h4 - h3) / (2 * dh_cd)
        data["cd_err"] = np.fabs((data["ans"] - data["cd"]) / data["cd"])

        # Reset the design variables
        self.x = x0

        if self.solver_type == "BasicLanczos":
            print(
                "%25s  %25s  %25s  %25s  %25s"
                % ("Answer", "CS", "CD", "CS Rel Error", "CD Rel Error")
            )
            print(
                "%25.15e  %25.15e  %25.15e  %25.15e  %25.15e"
                % (data["ans"], data["cs"], data["cd"], data["cs_err"], data["cd_err"])
            )
        else:
            print("%25s  %25s  %25s" % ("Answer", "CD", "CD Rel Error"))
            print(
                "%25.15e  %25.15e  %25.15e" % (data["ans"], data["cd"], data["cd_err"])
            )

        return data

    def test_ks_buckling_derivatives(self, dh_fd=1e-4, ks_rho=30, pert=None):
        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        t0 = time.time()
        dks = self.eval_ks_buckling_derivative(ks_rho)
        t1 = time.time()

        pert = np.random.uniform(size=x0.shape)

        ans = np.dot(pert, dks)

        self.x = x0 + dh_fd * pert
        self.initialize()
        c1 = self.eval_ks_buckling(ks_rho)

        self.x = x0 - dh_fd * pert
        self.initialize()
        c2 = self.eval_ks_buckling(ks_rho)

        cd = (c1 - c2) / (2 * dh_fd)

        print("\nTotal derivative for ks-buckling:", self.deriv_type + " type")
        print("Ans:                 ", ans)
        print("CD:                  ", cd)
        print("Rel err:             ", (ans - cd) / cd)
        print("Time for derivative: ", t1 - t0, "s")

        return

    def test_compliance_derivatives(self, dh_fd=1e-4, pert=None):
        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        dks = self.compliance_derivative()

        pert = np.random.uniform(size=x0.shape)

        ans = np.dot(pert, dks)

        self.x = x0 + dh_fd * pert
        self.initialize()
        c1 = self.compliance()

        self.x = x0 - dh_fd * pert
        self.initialize()
        c2 = self.compliance()

        cd = (c1 - c2) / (2 * dh_fd)

        print("\nTotal derivative for true compliance")
        print("Ans:                 ", ans)
        print("CD:                  ", cd)
        print("Rel err:             ", (ans - cd) / cd)

        return

    def test_eigenvector_aggregate_max_derivatives(
        self, dh_fd=1e-4, rho_agg=100, pert=None, node=None
    ):

        hb = 1.0
        if node is None:
            node = []
            for i in range(self.nnodes):
                node.append(i)

        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        self.initialize_adjoint()
        self.add_eigenvector_aggregate_max_derivative(hb, rho_agg, node)
        self.finalize_adjoint()

        pert = np.random.uniform(size=x0.shape)

        ans = np.dot(pert, self.xb)

        self.x = x0 + dh_fd * pert
        self.initialize()
        h1 = self.get_eigenvector_aggregate_max(rho_agg, node)

        self.x = x0 - dh_fd * pert
        self.initialize()
        h2 = self.get_eigenvector_aggregate_max(rho_agg, node)
        cd = (h1 - h2) / (2 * dh_fd)

        print("\nTotal derivative for aggregate-max")
        print("Ans = ", ans)
        print("CD  = ", cd)
        print("Rel err = ", (ans - cd) / cd, "\n")

        return

    def get_pts_and_tris(self, eta=None):
        pts = np.zeros((self.nnodes, 3))

        if eta is not None:
            u = self.Q.dot(eta)
            pts[:, 0] = self.X[:, 0] + 10 * u[::2]
            pts[:, 1] = self.X[:, 1] + 10 * u[1::2]

        # Create the triangles
        tris = np.zeros((2 * self.nelems, 3), dtype=int)
        tris[: self.nelems, 0] = self.conn[:, 0]
        tris[: self.nelems, 1] = self.conn[:, 1]
        tris[: self.nelems, 2] = self.conn[:, 2]

        tris[self.nelems :, 0] = self.conn[:, 0]
        tris[self.nelems :, 1] = self.conn[:, 2]
        tris[self.nelems :, 2] = self.conn[:, 3]

        return pts, tris, self.rho

    def plot(self, field, u=None, scale=1.0, ax=None, **kwargs):
        """
        Create a plot
        """

        # Create the triangles
        triangles = np.zeros((2 * self.nelems, 3), dtype=int)
        triangles[: self.nelems, 0] = self.conn[:, 0]
        triangles[: self.nelems, 1] = self.conn[:, 1]
        triangles[: self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems :, 0] = self.conn[:, 0]
        triangles[self.nelems :, 1] = self.conn[:, 2]
        triangles[self.nelems :, 2] = self.conn[:, 3]

        # Create the triangulation object
        if u is None:
            x = self.X[:, 0]
            y = self.X[:, 1]
        else:
            x = self.X[:, 0] + scale * u[0::2]
            y = self.X[:, 1] + scale * u[1::2]
        tri_obj = tri.Triangulation(x, y, triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, field, **kwargs)

        return

    def plot_design(self, path=None, index=None):
        fig, ax = plt.subplots()
        self.plot(self.rho, ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")

        # plot the bcs
        for i, v in self.bcs.items():
            ax.scatter(self.X[i, 0], self.X[i, 1], color="k")

        for i, v in self.forces.items():
            ax.quiver(self.X[i, 0], self.X[i, 1], v[0], v[1], color="r", scale=1e-3)

        if index is not None:
            for i in index:
                ax.scatter(
                    self.X[i, 0], self.X[i, 1], color="orange", s=5, clip_on=False
                )

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=150)

        plt.close(fig)

        return

    def plot_topology(self, ax):
        # Set the number of levels to use.
        levels = np.linspace(0.0, 1.0, 26)

        # Make sure that there are no ticks on either axis (these affect the bounding-box
        # and make extra white-space at the corners). Finally, turn off the axis so its
        # not visible.
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("off")

        self.plot(self.rho, ax=ax, levels=levels, cmap="viridis", extend="max")

        return

    def plot_mode(self, k, ax):
        if k < self.N and k >= 0 and self.Q is not None:
            # Set the number of levels to use.
            levels = np.linspace(0.0, 1.0, 26)

            # Make sure that there are no ticks on either axis (these affect the bounding-box
            # and make extra white-space at the corners). Finally, turn off the axis so its
            # not visible.
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.axis("off")

            value = np.fabs(np.max(self.Q[:, k])) + np.fabs(np.min(self.Q[:, k]))
            scale = 0.5 / value

            self.plot(
                self.rho,
                ax=ax,
                u=self.Q[:, k],
                scale=scale,
                levels=levels,
                cmap="viridis",
                extend="max",
            )

        return

    def plot_residuals(self, path=None):
        fig, ax = plt.subplots()
        ax.plot(self.profile["adjoint residuals"], marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=300)

        plt.close(fig)
        return fig, ax


def domain_compressed_column(nx=64, ny=128, Lx=1.0, Ly=2.0, shear_force=False):
    """
    ________
    |      |
    |      |
    |      | ny
    |      |
    |______|
       nx
    """
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)

    nelems = nx * ny
    nnodes = (nx + 1) * (ny + 1)
    nodes = np.arange(nnodes, dtype=int).reshape(nx + 1, ny + 1)
    conn = np.zeros((nelems, 4), dtype=int)
    X = np.zeros((nnodes, 2))

    for j in range(ny + 1):
        for i in range(nx + 1):
            X[nodes[i, j], 0] = x[i]
            X[nodes[i, j], 1] = y[j]

    for j in range(ny):
        for i in range(nx):
            conn[i + nx * j, 0] = nodes[i, j]
            conn[i + nx * j, 1] = nodes[i + 1, j]
            conn[i + nx * j, 2] = nodes[i + 1, j + 1]
            conn[i + nx * j, 3] = nodes[i, j + 1]

    # Create the symmetry in the problem
    dvmap = np.zeros((nx + 1, ny + 1), dtype=int)
    index = 0

    # 2-way reflection left to right
    for i in range(nx // 2 + 1):
        for j in range(ny + 1):
            if dvmap[i, j] >= 0:
                dvmap[i, j] = index
                dvmap[nx - i, j] = index
                index += 1

    num_design_vars = index
    dvmap = dvmap.flatten()

    # apply boundary conditions at the bottom nodes
    bcs = {}
    for i in range(nx + 1):
        bcs[nodes[i, 0]] = [0, 1]

    # apply a force at the top middle
    # force is independent of the mesh size,
    P = 1e-3
    forces = {}

    if shear_force is True:
        # apply a shear force at the top middle
        for i in range(nx + 1):
            forces[nodes[i, ny]] = [P / (nx + 1), 0]

    else:
        # apply a vertical force at the top middle
        offset = int(np.ceil(nx / 30))
        for i in range(offset):
            forces[nodes[nx // 2 - i - 1, ny]] = [0, -P / (2 * offset + 1)]
            forces[nodes[nx // 2 + i + 1, ny]] = [0, -P / (2 * offset + 1)]
        forces[nodes[nx // 2, ny]] = [0, -P / (2 * offset + 1)]

    return conn, X, dvmap, num_design_vars, bcs, forces


def make_model(
    nx=64, ny=128, Lx=1.0, Ly=2.0, rfact=4.0, N=10, shear_force=False, **kwargs
):
    """

    Parameters
    ----------
    ny : int
        Number of nodes in the y-direction
    rfact : real
        Filter radius as a function of the element side length
    N : int
        Number of eigenvalues and eigenvectors to compute
    """

    conn, X, dvmap, num_design_vars, bcs, forces = domain_compressed_column(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, shear_force=shear_force
    )

    fltr = NodeFilter(
        conn,
        X,
        r0=rfact * (Lx / nx),
        dvmap=dvmap,
        num_design_vars=num_design_vars,
        projection=kwargs.get("projection"),
        beta=kwargs.get("b0"),
    )

    # delete the projection and beta from the kwargs
    if "projection" in kwargs:
        del kwargs["projection"]
    if "b0" in kwargs:
        del kwargs["b0"]

    topo = TopologyAnalysis(fltr, conn, X, bcs=bcs, forces=forces, N=N, **kwargs)

    return topo


if __name__ == "__main__":
    np.random.seed(0)

    import sys

    solver_type = "BasicLanczos"
    if "IRAM" in sys.argv:
        solver_type = "IRAM"

    sigma = 3.0

    if "dl" in sys.argv:
        solver_type = "BasicLanczos"
        method = "dl"
        sigma = 6.0
        adjoint_options = {"lanczos_guess": False}
    elif "pcpg" in sys.argv:
        method = "pcpg"
        adjoint_options = {"lanczos_guess": True}
    elif "pgmres" in sys.argv:
        method = "pgmres"
        adjoint_options = {"lanczos_guess": True}
    elif "laa" in sys.argv:
        method = "laa"
        adjoint_options = {}
    else:
        method = "sibk"
        adjoint_options = {
            "lanczos_guess": True,
            "update_guess": False,
            "bs_target": 1,
        }

    print("method = ", method)
    print("adjoint_options = ", adjoint_options)
    print("solver_type = ", solver_type)

    topo = make_model(
        nx=64,
        rfact=4.0,
        N=10,
        sigma=sigma,
        solver_type=solver_type,
        adjoint_method=method,
        adjoint_options=adjoint_options,
        shear_force=False,
        deriv_type="tensor",
    )

    # Check the eigenvector aggregate derivatives
    data = topo.test_eigenvector_aggregate_derivatives(mode="tanh", rho=100.0)

    if "adjoint residuals" in data and len(data["adjoint residuals"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(data["adjoint residuals"], marker="o", markersize=4)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")

        plt.show()
