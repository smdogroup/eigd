import time

from eigd import (
    IRAM,
    BasicLanczos,
    SpLuOperator,
    eval_adjoint_residual_norm,
)
from fe_utils import compute_detJ, populate_thermal_Be, populate_thermal_He
import matplotlib.pylab as plt
import matplotlib.tri as tri
from node_filter import NodeFilter
import numpy as np
from scipy import sparse
from scipy.linalg import eigh


class ThermalTopologyAnalysis:

    def __init__(
        self,
        fltr,
        conn,
        X,
        node_sets={},
        element_sets={},
        kappa=1.0,
        density=1.0,
        heat_capacity=1.0,
        rho0=1e-6,
        p=3,
        beta=0.1,
        sigma=-0.1,
        N=10,
        m=None,
        Ntarget=None,
        solver_type="IRAM",
        tol=1e-14,
        rtol=1e-10,
        eig_atol=1e-5,
        adjoint_method="shift-invert",
        adjoint_options={},
        cost=1,
        deriv_type="tensor",
    ):

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.kappa = kappa
        self.density = density
        self.heat_capacity = heat_capacity
        self.rho0 = rho0
        self.p = p
        self.beta = beta
        self.sigma = sigma
        self.N = N
        self.m = m
        self.Ntarget = Ntarget
        self.node_sets = node_sets
        self.element_sets = element_sets
        self.solver_type = solver_type
        self.tol = tol
        self.rtol = rtol
        self.eig_atol = eig_atol
        self.adjoint_method = adjoint_method
        self.adjoint_options = adjoint_options
        self.cost = cost
        self.deriv_type = deriv_type

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = self.nnodes

        # Set the initial design variable values
        self.x = 0.95 * np.ones(self.fltr.num_design_vars)

        self.Q = None
        self.lam = None

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 4), dtype=int)
        self.var[:, :] = self.conn

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

        # Initialize the mean vectors for extracting/applying thermal loads
        self._init_mean_coefficients()
        self._init_profile()

        return

    def get_stiffness_matrix(self, rhoE):
        """
        Assemble the stiffness matrix
        """

        # Compute the element stiffnesses
        kappa = self.kappa * ((1 - self.beta) * rhoE**self.p + self.beta)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 4 x 4 element stiffness matrix
        Ke = np.zeros((self.nelems, 4, 4), dtype=rhoE.dtype)
        Be = np.zeros((self.nelems, 2, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                detJ = populate_thermal_Be(self.nelems, xi, eta, xe, ye, Be)

                # This is a fancy (and fast) way to compute the element matrices
                Ke += np.einsum("n,nij,nik -> njk", kappa * detJ, Be, Be)

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def get_stiffness_matrix_deriv(self, rhoE, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """

        # Derivative w.r.t. kappa
        dfdk = np.zeros(self.nelems)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = np.zeros((self.nelems, 4))
        psie = np.zeros((self.nelems, 4))

        ue = u[self.conn, ...]
        psie = psi[self.conn, ...]

        Be = np.zeros((self.nelems, 2, 4))
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        if u.ndim == 1 and psi.ndim == 1:
            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = populate_thermal_Be(self.nelems, xi, eta, xe, ye, Be)
                    dfdk += np.einsum("n,nim,nil,nm,nl -> n", detJ, Be, Be, psie, ue)
        elif u.ndim == 2 and psi.ndim == 2:
            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = populate_thermal_Be(self.nelems, xi, eta, xe, ye, Be)
                    dfdk += np.einsum("n,nim,nil,nmk,nlk -> n", detJ, Be, Be, psie, ue)

        # Compute the derivative w.r.t. rhoE
        dfdrhoE = (
            (1.0 - self.beta) * self.kappa * dfdk * self.p * rhoE ** (self.p - 1.0)
        )

        return dfdrhoE

    def get_mass_matrix(self, rhoE):
        """
        Assemble the mass matrix
        """

        # Compute the element density
        c = self.heat_capacity * self.density * ((1.0 - self.beta) * rhoE + self.beta)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element mass matrices
        Me = np.zeros((self.nelems, 4, 4), dtype=rhoE.dtype)

        # Set the interpolation matrices for all of the elements
        He = np.zeros((self.nelems, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                detJ = populate_thermal_He(self.nelems, xi, eta, xe, ye, He)

                # This is a fancy (and fast) way to compute the element matrices
                Me += np.einsum("n,ni,nj -> nij", c * detJ, He, He)

        M = sparse.coo_matrix((Me.flatten(), (self.i, self.j)))
        M = M.tocsr()

        return M

    def get_mass_matrix_deriv(self, rhoE, u, v):
        """
        Compute the derivative of the mass matrix
        """

        # Derivative with respect to element density
        dfdrhoE = np.zeros(self.nelems)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = u[self.conn, ...]
        ve = v[self.conn, ...]

        # The interpolation matrix for each element
        He = np.zeros((self.nelems, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        if u.ndim == 1 and v.ndim == 1:
            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = populate_thermal_He(self.nelems, xi, eta, xe, ye, He)
                    dfdrhoE += np.einsum("n,ni,nj,ni,nj -> n", detJ, He, He, ue, ve)
        elif u.ndim == 2 and v.ndim == 2:
            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = populate_thermal_He(self.nelems, xi, eta, xe, ye, He)
                    dfdrhoE += np.einsum("n,ni,nj,nik,njk -> n", detJ, He, He, ue, ve)

        dfdrhoE[:] *= (1.0 - self.beta) * self.heat_capacity * self.density

        return dfdrhoE

    def _init_profile(self):
        self.profile = {}
        self.profile["nnodes"] = self.nnodes
        self.profile["nelems"] = self.nelems
        self.profile["solver_type"] = self.solver_type
        self.profile["adjoint_method"] = self.adjoint_method
        self.profile["adjoint_options"] = self.adjoint_options
        self.profile["N"] = self.N
        self.profile["Ntarget"] = self.Ntarget
        self.profile["kappa"] = self.kappa
        self.profile["density"] = self.density
        self.profile["heat_capacity"] = self.heat_capacity
        self.profile["rho0"] = self.rho0
        self.profile["p"] = self.p
        self.profile["eig_atol"] = self.eig_atol
        self.profile["ftype"] = self.fltr.ftype
        self.profile["r0"] = self.fltr.r0

        return

    def solve_eigenvalue_problem(self, rhoE, store=False):
        """
        Compute the smallest natural frequencies
        """

        t0 = time.time()

        K = self.get_stiffness_matrix(rhoE)
        M = self.get_mass_matrix(rhoE)

        t1 = time.time()
        self.profile["matrix assembly time"] = t1 - t0

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        for i in range(self.cost):
            if self.N >= self.nvars:
                lam, Q = eigh(K.todense(), M.todense())
            else:
                # Compute the shifted operator
                mat = K - self.sigma * M
                mat = mat.tocsc()
                self.factor = SpLuOperator(mat)
                self.profile["sigma"] = self.sigma if i == 0 else None

                self.K = K
                self.M = M

                self.factor.count = 0
                if self.solver_type == "IRAM":
                    if self.m is None:
                        self.m = max(2 * self.N + 1, 60)
                    self.eig_solver = IRAM(N=self.N, m=self.m, eig_atol=self.eig_atol)
                    lam, Q = self.eig_solver.solve(
                        self.K, self.M, self.factor, self.sigma
                    )
                else:
                    if self.m is None:
                        self.m = max(3 * self.N + 1, 60)
                    if self.Ntarget is not None:
                        self.eig_solver = BasicLanczos(
                            Ntarget=self.Ntarget,
                            m=self.m,
                            eig_atol=self.eig_atol,
                            tol=self.tol,
                        )
                    else:
                        self.eig_solver = BasicLanczos(
                            N=self.N,
                            m=self.m,
                            eig_atol=self.eig_atol,
                            tol=self.tol,
                        )
                    lam, Q = self.eig_solver.solve(
                        self.K, self.M, self.factor, self.sigma
                    )

                    if store:
                        self.profile["eig_res"] = self.eig_solver.eig_res.tolist()

                self.profile["solve preconditioner count"] = (
                    self.factor.count if i == 0 else None
                )

        t2 = time.time()
        t = (t2 - t1) / self.cost
        self.profile["eigenvalue solve time"] = t
        self.profile["m"] = self.m
        self.profile["eig_solver.m"] = str(self.eig_solver.m)

        # Reset the number of eigenvalues/eigenvectors - this could change if Ntarget is set
        self.N = len(lam)

        return lam, Q

    def initialize(self, store=False):
        # Apply the fltr
        self.rho = self.fltr.apply(self.x)

        # Average the density to get the element-wise density
        self.rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
        )

        # Solve the eigenvalue problem
        self.lam, self.Q = self.solve_eigenvalue_problem(self.rhoE, store)

        if store:
            self.profile["eigenvalues"] = self.lam.tolist()

        return

    def initialize_adjoint(self):
        self.xb = np.zeros(self.x.shape)
        self.rhoEb = np.zeros(self.rhoE.shape)
        self.lamb = np.zeros(self.lam.shape)
        self.Qb = np.zeros(self.Q.shape)

        return

    def check_adjoint_residual(self, A, B, lam, Q, Qb, psi, b_ortho=False):
        res, orth = eval_adjoint_residual_norm(A, B, lam, Q, Qb, psi, b_ortho=b_ortho)
        for i in range(Q.shape[1]):
            ratio = orth[i] / np.linalg.norm(Q[:, i])
            self.profile["adjoint norm[%2d]" % i] = res[i]
            self.profile["adjoint ortho[%2d]" % i] = ratio
            self.profile["adjoint lam[%2d]" % i] = lam[i]

        return res

    def add_check_adjoint_residual(self, b_ortho=False):
        return self.check_adjoint_residual(
            self.K, self.M, self.lam, self.Q, self.Qb, self.psi, b_ortho=b_ortho
        )

    def eval_area(self):
        area = 0.0
        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]
        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                detJ = compute_detJ(self.nelems, xi, eta, xe, ye)
                area += np.sum(detJ * self.rhoE)
        return area

    def eval_area_gradient(self):
        dfdrhoE = np.zeros(self.nelems)

        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                detJ = compute_detJ(self.nelems, xi, eta, xe, ye)

                dfdrhoE[:] += detJ

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.apply_gradient(dfdrho, self.x)

    def _init_mean_coefficients(self):
        self.mean_vecs = {}

        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for name in self.element_sets:
            v = np.zeros(self.nnodes)

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = compute_detJ(self.nelems, xi, eta, xe, ye)

                    for e in self.element_sets[name]:
                        v[self.var[e, :]] += detJ[e]

            v = v / np.sum(v)
            self.mean_vecs[name] = v

        return

    def get_thermal_compliance(self, vec):
        compliance = 0.0
        for i in range(1, self.N):
            val = self.Q[:, i].dot(vec)
            compliance += (val * val) / self.lam[i]

        return compliance

    def add_thermal_compliance_derivative(self, compb, vec):
        for i in range(1, self.N):
            val = self.Q[:, i].dot(vec)
            self.Qb[:, i] += 2.0 * compb * val * vec / self.lam[i]
            self.lamb[i] -= compb * (val * val) / self.lam[i] ** 2

        return

    def get_eigenvector_aggregate(self, rho, node):
        # Exponential aggregate
        # eta = np.exp(-rho * (self.lam[1:] - np.min(self.lam[1:])))

        # Tanh aggregate
        lam_a = 0.0
        lam_b = 40.0
        a = np.tanh(rho * (self.lam[1:] - lam_a))
        b = np.tanh(rho * (self.lam[1:] - lam_b))
        eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(1, self.N):
            h += eta[i - 1] * self.Q[node, i] ** 2

        return h

    def add_eigenvector_aggregate_derivative(self, hb, rho, node):
        # Exponential aggregate
        # eta = np.exp(-rho * (self.lam[1:] - np.min(self.lam[1:])))

        # Tanh aggregate
        lam_a = 0.0
        lam_b = 40.0
        a = np.tanh(rho * (self.lam[1:] - lam_a))
        b = np.tanh(rho * (self.lam[1:] - lam_b))
        eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(1, self.N):
            h += eta[i - 1] * self.Q[node, i] ** 2

        for i in range(1, self.N):
            self.Qb[node, i] += 2.0 * hb * eta[i - 1] * self.Q[node, i]

            # self.lamb[i] -= hb * rho * eta[i - 1] * (self.Q[node, i] ** 2 - h)
            self.lamb[i] -= (
                hb
                * rho
                * eta[i - 1]
                * (a[i - 1] + b[i - 1])
                * (self.Q[node, i] ** 2 - h)
            )

        return

    def get_mean_coefficients(self):
        coef = {}
        for name in self.element_sets:
            coef[name] = self.Q.T @ self.mean_vecs[name]

        return coef

    def add_mean_derivatives(self, coefb):
        for name in self.element_sets:
            self.Qb += np.outer(self.mean_vecs[name], coefb[name])

        return

    def finalize_adjoint(self):

        class Callback:
            def __init__(self):
                self.res_list = []

            def __call__(self, rk=None):
                self.res_list.append(rk)

        callback = Callback()

        # self.profile["adjoint solution method"] = self.adjoint_method
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

        dAdx = lambda w, v: self.get_stiffness_matrix_deriv(self.rhoE, w, v)
        dBdx = lambda w, v: self.get_mass_matrix_deriv(self.rhoE, w, v)

        # Compute the total derivative
        self.rhoEb = self.eig_solver.add_total_derivative(
            self.lamb,
            self.Qb,
            psi,
            dAdx,
            dBdx,
            self.rhoEb,
            adj_corr_data=corr_data,
            deriv_type=self.deriv_type,
        )

        rhob = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(rhob, self.conn[:, i], self.rhoEb)
        rhob *= 0.25

        self.xb += self.fltr.apply_gradient(rhob, self.x)

        t2 = time.time()
        t = (t2 - t1) / self.cost
        self.profile["total derivative time"] = t

        return

    def test_eigenvector_aggregate_derivatives(
        self, rho=10.0, node=0, dh_cs=1e-6, dh_fd=1e-6, dh_cd=1e-4, pert=None
    ):
        # Compute the elastic properties
        self.initialize(store=True)
        h = self.get_eigenvector_aggregate(rho, node)

        # Copy the design variables
        x0 = np.array(self.x)

        # Create the derivatives
        self.initialize_adjoint()

        hb = 1.0
        self.add_eigenvector_aggregate_derivative(hb, rho, node)
        self.finalize_adjoint()

        # Set a random perturbation to the design variables
        if pert is None:
            pert = np.random.uniform(size=self.x.shape)

        # The exact derivative
        data = {}
        data["ans"] = np.dot(pert, self.xb)
        data.update(self.profile)

        if self.solver_type == "BasicLanczos":
            # Perturb the design variables for complex-step
            self.x = np.array(x0).astype(complex)
            self.x.imag += dh_cs * pert
            self.initialize()
            h1 = self.get_eigenvector_aggregate(rho, node)

            data["dh_cs"] = dh_cs
            data["cs"] = h1.imag / dh_cs
            data["cs_err"] = np.fabs((data["ans"] - data["cs"]) / data["cs"])

        # Perturb the design variables for finite-difference
        self.x = x0 + dh_fd * pert
        self.initialize()
        h2 = self.get_eigenvector_aggregate(rho, node)

        # Compute the finite-difference and relative error
        data["dh_fd"] = dh_fd
        data["fd"] = (h2 - h) / (dh_fd)
        data["fd_err"] = np.fabs((data["ans"] - data["fd"]) / data["fd"])

        self.x = x0 - dh_cd * pert
        self.initialize()
        h3 = self.get_eigenvector_aggregate(rho, node)

        self.x = x0 + dh_cd * pert
        self.initialize()
        h4 = self.get_eigenvector_aggregate(rho, node)

        data["dh_cd"] = dh_cd
        data["cd"] = (h4 - h3) / (2 * dh_cd)
        data["cd_err"] = np.fabs((data["ans"] - data["cd"]) / data["cd"])

        # Reset the design variables
        self.x = x0

        if self.solver_type == "BasicLanczos":
            print(
                "%25s  %25s  %25s  %25s  %25s"
                % ("Answer", "CS", "FD", "CS Rel Error", "FD Rel Error")
            )
            print(
                "%25.15e  %25.15e  %25.15e  %25.15e  %25.15e"
                % (data["ans"], data["cs"], data["fd"], data["cs_err"], data["fd_err"])
            )
        else:
            print("%25s  %25s  %25s" % ("Answer", "FD", "FD Rel Error"))
            print(
                "%25.15e  %25.15e  %25.15e" % (data["ans"], data["fd"], data["fd_err"])
            )

        return data

    def test_mean_derivatives(
        self, coefb=None, dh_cs=1e-6, dh_fd=1e-6, dh_cd=1e-4, pert=None
    ):
        # Compute the elastic properties
        self.initialize(store=True)
        h = self.get_mean_coefficients()

        # Create the derivatives of a random function
        if coefb is None:
            coefb = {}
            for name in h:
                coefb[name] = np.random.uniform(size=h[name].shape)

        # Set a random perturbation to the design variables
        if pert is None:
            pert = np.random.uniform(size=self.x.shape)

        # Copy the design variables
        x0 = np.array(self.x)

        self.initialize_adjoint()
        self.add_mean_derivatives(coefb)
        self.finalize_adjoint()

        # The exact derivative
        data = {}
        data["ans"] = np.dot(pert, self.xb)
        data.update(self.profile)

        if self.solver_type == "BasicLanczos":
            # Perturb the design variables for complex-step
            self.x = np.array(x0).astype(complex)
            self.x.imag += dh_cs * pert
            self.initialize()
            h1 = self.get_mean_coefficients()

            data["dh_cs"] = dh_cs
            data["cs"] = 0.0
            for name in h1:
                data["cs"] += np.sum(coefb[name] * h1[name].imag / dh_cs)
            data["cs_err"] = np.fabs((data["ans"] - data["cs"]) / data["cs"])

        # Perturb the design variables for finite-difference
        self.x = x0 + dh_fd * pert
        self.initialize()
        h2 = self.get_mean_coefficients()

        # Compute the finite-difference and relative error
        data["dh_fd"] = dh_fd
        data["fd"] = 0.0
        for name in h2:
            data["fd"] += np.sum(coefb[name] * (h2[name] - h[name]) / dh_fd)
        data["fd_err"] = np.fabs((data["ans"] - data["fd"]) / data["fd"])

        self.x = x0 - dh_cd * pert
        self.initialize()
        h3 = self.get_mean_coefficients()

        self.x = x0 + dh_cd * pert
        self.initialize()
        h4 = self.get_mean_coefficients()

        data["dh_cd"] = dh_cd
        data["cd"] = 0.0
        for name in h3:
            data["cd"] += np.sum(coefb[name] * (h4[name] - h3[name]) / (2 * dh_cd))
        data["cd_err"] = np.fabs((data["ans"] - data["cd"]) / data["cd"])

        # Reset the design variables
        self.x = x0

        if self.solver_type == "BasicLanczos":
            print(
                "%25s  %25s  %25s  %25s  %25s"
                % ("Answer", "CS", "FD", "CS Rel Error", "FD Rel Error")
            )
            print(
                "%25.15e  %25.15e  %25.15e  %25.15e  %25.15e"
                % (data["ans"], data["cs"], data["fd"], data["cs_err"], data["fd_err"])
            )
        else:
            print("%25s  %25s  %25s" % ("Answer", "FD", "FD Rel Error"))
            print(
                "%25.15e  %25.15e  %25.15e" % (data["ans"], data["fd"], data["fd_err"])
            )

        return data

    def test_compliance_derivatives(
        self, vec=None, dh_cs=1e-6, dh_fd=1e-6, dh_cd=1e-4, pert=None
    ):
        if vec is None:
            vec = np.random.uniform(size=self.nnodes)

        # Compute the elastic properties
        self.initialize(store=True)
        h = self.get_thermal_compliance(vec)

        # Copy the design variables
        x0 = np.array(self.x)

        # Create the derivatives
        self.initialize_adjoint()

        self.add_thermal_compliance_derivative(1.0, vec)
        self.finalize_adjoint()
        self.add_check_adjoint_residual(b_ortho=True)

        # Set a random perturbation to the design variables
        if pert is None:
            pert = np.random.uniform(size=self.x.shape)

        # The exact derivative
        data = {}
        data["ans"] = np.dot(pert, self.xb)
        data.update(self.profile)

        if self.solver_type == "BasicLanczos":
            # Perturb the design variables for complex-step
            self.x = np.array(x0).astype(complex)
            self.x.imag += dh_cs * pert
            self.initialize()
            h1 = self.get_thermal_compliance(vec)

            data["dh_cs"] = dh_cs
            data["cs"] = h1.imag / dh_cs
            data["cs_err"] = np.fabs((data["ans"] - data["cs"]) / data["cs"])

        # Perturb the design variables for finite-difference
        self.x = x0 + dh_fd * pert
        self.initialize()
        h2 = self.get_thermal_compliance(vec)

        # Compute the finite-difference and relative error
        data["dh_fd"] = dh_fd
        data["fd"] = (h2 - h) / dh_fd
        data["fd_err"] = np.fabs((data["ans"] - data["fd"]) / data["fd"])

        self.x = x0 - dh_cd * pert
        self.initialize()
        h3 = self.get_thermal_compliance(vec)

        self.x = x0 + dh_cd * pert
        self.initialize()
        h4 = self.get_thermal_compliance(vec)

        # Compute the finite-difference and relative error
        data["dh_cd"] = dh_cd
        data["cd"] = (h4 - h3) / (2 * dh_cd)
        data["cd_err"] = np.fabs((data["ans"] - data["cd"]) / data["cd"])

        # Reset the design variables
        self.x = x0

        if self.solver_type == "BasicLanczos":
            print(
                "%25s  %25s  %25s  %25s  %25s %25s %25s"
                % (
                    "Answer",
                    "CS",
                    "FD",
                    "CD",
                    "CS Rel Error",
                    "FD Rel Error",
                    "CD Rel Error",
                )
            )
            print(
                "%25.15e  %25.15e  %25.15e  %25.15e  %25.15e  %25.15e  %25.15e"
                % (
                    data["ans"],
                    data["cs"],
                    data["fd"],
                    data["cd"],
                    data["cs_err"],
                    data["fd_err"],
                    data["cd_err"],
                )
            )
        else:
            print("%25s  %25s  %25s" % ("Answer", "FD", "FD Rel Error"))
            print(
                "%25.15e  %25.15e  %25.15e" % (data["ans"], data["fd"], data["fd_err"])
            )

        return data

    def plot(self, field, ax=None, **kwargs):
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
        x = self.X[:, 0]
        y = self.X[:, 1]
        tri_obj = tri.Triangulation(x, y, triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        c = ax.tricontourf(tri_obj, field, **kwargs)

        # reduce the number of points in the contour plot
        for c in c.collections:
            c.set_rasterized(True)

        return

    def plot_design(self, set1=None, set2=None, path=None):
        fig, ax = plt.subplots()
        self.plot(self.rho, ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")
        if set1 is not None:
            for loc in set1:
                for e in self.element_sets[loc]:
                    xe = self.X[self.conn[e, :], 0]
                    ye = self.X[self.conn[e, :], 1]
                    ax.fill(xe, ye, edgecolor="none", facecolor="red", alpha=0.25)

        if set2 is not None:
            for loc in set2:
                for e in self.element_sets[loc]:
                    xe = self.X[self.conn[e, :], 0]
                    ye = self.X[self.conn[e, :], 1]
                    ax.fill(xe, ye, edgecolor="none", facecolor="blue", alpha=0.25)

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=300)

        plt.close(fig)

        return

    def plot_mode(self, k, ax):
        if k < self.N and k >= 0 and self.Q is not None:
            # Set the number of levels to use.
            levels = np.linspace(np.min(self.Q[:, k]), np.max(self.Q[:, k]), 26)

            # Make sure that there are no ticks on either axis (these affect the bounding-box
            # and make extra white-space at the corners). Finally, turn off the axis so its
            # not visible.
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.axis("off")

            self.plot(
                self.Q[:, k],
                ax=ax,
                levels=levels,
                cmap="viridis",
                extend="max",
            )

        return

    def plot_modes(self):
        nx = int(np.ceil(np.sqrt(self.N)))
        ny = int(np.ceil(self.N / nx))
        fig, ax = plt.subplots(nx, ny)

        for i in range(self.N):
            self.plot_mode(i, ax[i // ny, i % ny])

        plt.close(fig)
        return

    def plot_residuals(self, path=None):
        fig, ax = plt.subplots()
        ax.plot(self.profile["adjoint residuals"], marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        # ax.legend("upper right")

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=300)

        plt.close(fig)
        return fig, ax


def make_model(nx=128, ny=128, Lx=1.0, Ly=1.0, rfact=4.0, **kwargs):
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)

    # Set the filter matrix
    r0 = rfact * (Ly / ny)

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

    eset = []
    for j in range(ny // 2, 3 * ny // 4):
        for i in range(nx // 2, 3 * nx // 4):
            eset.append(i + nx * j)
    element_sets = {"center": eset}

    fltr = NodeFilter(conn, X, r0=r0)
    topo = ThermalTopologyAnalysis(fltr, conn, X, element_sets=element_sets, **kwargs)

    return topo


def make_opt_model(nx=256, Lx=1.0, rfact=4.0, epsilon=0.0, element_sets=None, **kwargs):
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Lx + epsilon, nx + 1)

    # Set the fltr matrix
    r0 = rfact * (Lx / nx)

    nelems = nx * nx
    nnodes = (nx + 1) * (nx + 1)
    nodes = np.arange(nnodes, dtype=int).reshape(nx + 1, nx + 1)
    conn = np.zeros((nelems, 4), dtype=int)
    X = np.zeros((nnodes, 2))

    for j in range(nx + 1):
        for i in range(nx + 1):
            X[nodes[i, j], 0] = x[i]
            X[nodes[i, j], 1] = y[j]

    for j in range(nx):
        for i in range(nx):
            conn[i + nx * j, 0] = nodes[i, j]
            conn[i + nx * j, 1] = nodes[i + 1, j]
            conn[i + nx * j, 2] = nodes[i + 1, j + 1]
            conn[i + nx * j, 3] = nodes[i, j + 1]

    if element_sets is None:
        element_sets = {}

    if "center" in element_sets:
        for j in range(3 * nx // 8, 5 * nx // 8):
            for i in range(3 * nx // 8, 5 * nx // 8):
                element_sets["center"].append(i + nx * j)

    for k in range(4):
        key = "corner%d" % (k)
        if key in element_sets:
            istart = nx // 8 + (5 * nx // 8) * (k % 2)
            iend = istart + nx // 8
            jstart = nx // 8 + (5 * nx // 8) * (k // 2)
            jend = jstart + nx // 8

            for j in range(jstart, jend):
                for i in range(istart, iend):
                    element_sets[key].append(i + nx * j)

    for k in range(4):
        key = "edge%d" % (k)
        if key in element_sets:
            # only consider the four edges i=0, i=nx, j=0, j=nx
            offset = int(np.ceil(nx // 32))
            if k == 0:
                for i in range(nx):
                    for j in range(offset):
                        element_sets[key].append(i + nx * j)
            elif k == 1:
                for i in range(nx):
                    for j in range(nx - offset, nx):
                        element_sets[key].append(i + nx * j)
            elif k == 2:
                for j in range(offset, nx - offset):
                    for i in range(offset):
                        element_sets[key].append(i + nx * j)
            elif k == 3:
                for j in range(offset, nx - offset):
                    for i in range(nx - offset, nx):
                        element_sets[key].append(i + nx * j)

    # Create the dv map
    dvmap = -np.ones((nx + 1, nx + 1), dtype=int)

    index = 0
    for i in range(nx // 2, nx + 1):
        for j in range(nx // 2, i + 1):
            dvmap[i, j] = index
            dvmap[j, i] = index

            dvmap[nx - i, j] = index
            dvmap[j, nx - i] = index

            dvmap[i, nx - j] = index
            dvmap[nx - j, i] = index

            dvmap[nx - i, nx - j] = index
            dvmap[nx - j, nx - i] = index

            index += 1

    num_design_vars = index
    dvmap = dvmap.flatten()

    fltr = NodeFilter(
        conn,
        X,
        r0=r0,
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

    topo = ThermalTopologyAnalysis(fltr, conn, X, element_sets=element_sets, **kwargs)

    return topo


if __name__ == "__main__":
    import sys

    element_sets = {"center": []}

    if "dl" in sys.argv:
        method = "dl"
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

    solver_type = "BasicLanczos"
    if "IRAM" in sys.argv:
        solver_type = "IRAM"

    print("method = ", method)
    print("adjoint_options = ", adjoint_options)
    print("solver_type = ", solver_type)

    for epsilon in [0.1, 1e-6, 1e-8]:
        # Create the topology optimization problem
        topo = make_opt_model(
            nx=128,
            rfact=4.0,
            N=20,
            m=90,
            p=3,
            epsilon=epsilon,
            solver_type=solver_type,
            adjoint_method=method,
            adjoint_options=adjoint_options,
            element_sets=element_sets,
            eig_atol=1e-5,
            rtol=1e-12,
            deriv_type="tensor",
        )

        # Test the compliance derivatives
        data = topo.test_compliance_derivatives(dh_cs=1e-20)
