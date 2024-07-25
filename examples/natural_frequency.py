import time

from fe_utils import populate_Be_and_He
import matplotlib.pylab as plt
import matplotlib.tri as tri
from node_filter import NodeFilter
import numpy as np
from scipy import sparse
from scipy.linalg import eigh

from eigd import IRAM, BasicLanczos, SpLuOperator, eval_adjoint_residual_norm


class TopologyAnalysis:

    def __init__(
        self,
        fltr,
        conn,
        X,
        node_sets={},
        element_sets={},
        E=1.0,
        nu=0.3,
        ptype_K="simp",
        ptype_M="simp",
        rho0_K=1e-6,
        rho0_M=1e-9,
        p=3.0,
        q=5.0,
        density=1.0,
        sigma=-10.0,
        N=10,
        m=None,
        solver_type="IRAM",
        tol=1e-14,
        rtol=1e-10,
        eig_atol=1e-5,
        adjoint_method="sibk",
        adjoint_options={},
        cost=1,
        deriv_type="tensor",
    ):
        self.ptype_K = ptype_K.lower()
        self.ptype_M = ptype_M.lower()

        self.rho0_K = rho0_K
        self.rho0_M = rho0_M

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.p = p
        self.q = q
        self.density = density
        self.sigma = sigma  # Shift value
        self.N = N  # Number of modes
        self.m = m
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
        self.nvars = 2 * self.nnodes

        # Set the initial design variable values
        self.x = 0.95 * np.ones(self.fltr.num_design_vars)

        self.Q = None
        self.lam = None

        # Compute the constitutive matrix
        self.E = E
        self.nu = nu
        self.C0 = E * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )
        self.C0 *= 1.0 / (1.0 - nu**2)

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 8), dtype=int)
        self.var[:, ::2] = 2 * self.conn
        self.var[:, 1::2] = 2 * self.conn + 1

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

    def intital_Be_and_He(self):
        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # Compute Be and Te, detJ
        Be = np.zeros((self.nelems, 3, 8, 4))
        He = np.zeros((self.nelems, 2, 8, 4))
        detJ = np.zeros((self.nelems, 4))

        for j in range(2):
            for i in range(2):
                xi, eta = gauss_pts[i], gauss_pts[j]
                index = 2 * i + j
                Bei = Be[:, :, :, index]
                Hei = He[:, :, :, index]
                detJ[:, index] = populate_Be_and_He(
                    self.nelems, xi, eta, xe, ye, Bei, Hei
                )

        return Be, He, detJ

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

        for j in range(2):
            for i in range(2):
                detJ = self.detJ[:, 2 * i + j]
                Be = self.Be[:, :, :, 2 * i + j]

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

        if psi.ndim == 1 and u.ndim == 1:
            for j in range(2):
                for i in range(2):
                    Be = self.Be[:, :, :, 2 * i + j]
                    detJ = self.detJ[:, 2 * i + j]

                    se = np.einsum("nij,nj -> ni", Be, psie)
                    te = np.einsum("nij,nj -> ni", Be, ue)
                    dfdrhoE += np.einsum("n,ij,nj,ni -> n", detJ, self.C0, se, te)
        elif psi.ndim == 2 and u.ndim == 2:
            for j in range(2):
                for i in range(2):
                    Be = self.Be[:, :, :, 2 * i + j]
                    detJ = self.detJ[:, 2 * i + j]

                    se = Be @ psie
                    te = Be @ ue
                    dfdrhoE += detJ * np.einsum("ij,njk,nik -> n", self.C0, se, te)

        if self.ptype_K == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1.0)
        else:  # ramp
            dfdrhoE[:] *= (1.0 + self.q) / (1.0 + self.q * (1.0 - rhoE)) ** 2

        return dfdrhoE

    def get_mass_matrix(self, rhoE):
        """
        Assemble the mass matrix
        """

        # Compute the element density
        if self.ptype_M == "msimp":
            nonlin = self.simp_c1 * rhoE**6.0 + self.simp_c2 * rhoE**7.0
            cond = (rhoE > 0.1).astype(int)
            density = self.density * (rhoE * cond + nonlin * (1 - cond))
        elif self.ptype_M == "ramp":
            density = self.density * (
                (self.q + 1.0) * rhoE / (1 + self.q * rhoE) + self.rho0_M
            )
        else:  # linear
            density = self.density * rhoE

        # Assemble all of the the 8 x 8 element mass matrices
        Me = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for j in range(2):
            for i in range(2):
                detJ = self.detJ[:, 2 * i + j]
                He = self.He[:, :, :, 2 * i + j]

                # This is a fancy (and fast) way to compute the element matrices
                Me += np.einsum("n,nij,nil -> njl", density * detJ, He, He)

        M = sparse.coo_matrix((Me.flatten(), (self.i, self.j)))
        M = M.tocsr()

        return M

    def get_mass_matrix_deriv(self, rhoE, u, v):
        """
        Compute the derivative of the mass matrix
        """

        # Derivative with respect to element density
        dfdrhoE = np.zeros(self.nelems)

        # The element-wise variables
        ue = np.zeros((self.nelems, 8) + u.shape[1:])
        ve = np.zeros((self.nelems, 8) + v.shape[1:])

        ue[:, ::2, ...] = u[2 * self.conn, ...]
        ue[:, 1::2, ...] = u[2 * self.conn + 1, ...]

        ve[:, ::2, ...] = v[2 * self.conn, ...]
        ve[:, 1::2, ...] = v[2 * self.conn + 1, ...]

        if u.ndim == 1 and v.ndim == 1:
            for j in range(2):
                for i in range(2):
                    He = self.He[:, :, :, 2 * i + j]
                    detJ = self.detJ[:, 2 * i + j]

                    eu = np.einsum("nij,nj -> ni", He, ue)
                    ev = np.einsum("nij,nj -> ni", He, ve)
                    dfdrhoE += np.einsum("n,ni,ni -> n", detJ, eu, ev)
        elif u.ndim == 2 and v.ndim == 2:
            for j in range(2):
                for i in range(2):
                    He = self.He[:, :, :, 2 * i + j]
                    detJ = self.detJ[:, 2 * i + j]

                    eu = He @ ue
                    ev = He @ ve
                    dfdrhoE += detJ * np.einsum("nik,nik -> n", eu, ev)

        if self.ptype_M == "msimp":
            dnonlin = 6.0 * self.simp_c1 * rhoE**5.0 + 7.0 * self.simp_c2 * rhoE**6.0
            cond = (rhoE > 0.1).astype(int)
            dfdrhoE[:] *= self.density * (cond + dnonlin * (1 - cond))
        elif self.ptype_M == "ramp":
            dfdrhoE[:] *= self.density * (1.0 + self.q) / (1.0 + self.q * rhoE) ** 2
        else:  # linear
            dfdrhoE[:] *= self.density

        return dfdrhoE

    def eval_area(self):
        return np.sum(self.detJ.reshape(-1) * np.tile(self.rhoE, 4))

    def eval_area_gradient(self):
        dfdrhoE = np.sum(self.detJ, axis=1)

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.apply_gradient(dfdrho, self.x)

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
                    N = self.N + 3
                    if self.m is None:
                        self.m = max(2 * N + 1, 60)
                    self.eig_solver = IRAM(N=N, m=self.m, eig_atol=self.eig_atol)
                    lam, Q = self.eig_solver.solve(
                        self.K, self.M, self.factor, self.sigma
                    )
                else:
                    N = self.N + 3
                    if self.m is None:
                        self.m = max(3 * N + 1, 60)
                    self.eig_solver = BasicLanczos(
                        N=N, m=self.m, eig_atol=self.eig_atol, tol=self.tol
                    )
                    lam, Q = self.eig_solver.solve(
                        self.K,
                        self.M,
                        self.factor,
                        self.sigma,
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

        # Discard the rigid modes
        lam0 = lam[3:]
        Q0 = Q[:, 3:]

        # Ensure that the eigenvectors have consistent signs
        if self.Q is not None:
            for i in range(self.N):
                if np.dot(Q0[:, i], self.Q[:, i]) < 0.0:
                    Q0[:, i] *= -1.0

        return lam0, Q0

    def initialize(self, store=False):
        # Apply the filter
        self.rho = self.fltr.apply(self.x)

        # Average the density to get the element-wise density
        self.rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
        )

        self.Be, self.He, self.detJ = self.intital_Be_and_He()

        # Solve the eigenvalue problem
        self.lam, self.Q = self.solve_eigenvalue_problem(self.rhoE, store)

        self.profile["natural frequencies"] = np.sqrt(self.lam).tolist()

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

    def finalize_adjoint(self):

        class Callback:
            def __init__(self):
                self.res_list = []

            def __call__(self, rk=None):
                self.res_list.append(rk)

        callback = Callback()

        # Create functions for computing dA/dx and dB/dx
        dAdx = lambda w, v: self.get_stiffness_matrix_deriv(self.rhoE, w, v)
        dBdx = lambda w, v: self.get_mass_matrix_deriv(self.rhoE, w, v)

        Q0b = np.zeros((self.nvars, 3 + self.N))
        Q0b[:, 3:] = self.Qb

        self.profile["adjoint solution method"] = self.adjoint_method
        self.factor.count = 0

        t0 = time.time()
        for i in range(self.cost):
            if i != 0:
                callback.res_list = []
            psi0, data = self.eig_solver.solve_adjoint(
                Q0b,
                rtol=self.rtol,
                method=self.adjoint_method,
                callback=callback,
                **self.adjoint_options,
            )
        t1 = time.time()
        t = (t1 - t0) / self.cost

        self.psi = psi0[:, 3:]

        self.profile["adjoint preconditioner count"] = self.factor.count
        self.profile["adjoint solution time"] = t
        self.profile["adjoint residuals"] = np.array(callback.res_list).tolist()
        self.profile["adjoint correction data"] = data

        # Modify the correction data to discard the lowest
        # 3 modes (associated with the rigid bodies)
        data0 = {}
        for i in data:
            if i >= 3:
                items = []
                for j, xi, eta in data[i]:
                    if j >= 3:
                        items.append((j, xi, eta))
                if len(items) > 0:
                    data0[i] = items

        lamb0 = np.zeros(3 + len(self.lamb))
        lamb0[3:] = self.lamb
        self.rhoEb = self.eig_solver.add_total_derivative(
            lamb0,
            Q0b,
            psi0,
            dAdx,
            dBdx,
            self.rhoEb,
            adj_corr_data=data0,
            deriv_type=self.deriv_type,
        )

        rhob = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(rhob, self.conn[:, i], self.rhoEb)
        rhob *= 0.25

        self.xb += self.fltr.apply_gradient(rhob, self.x)

        t2 = time.time()
        self.profile["total derivative time"] = t2 - t1

        return

    def get_frequencies(self):
        return np.sqrt(self.lam)

    def add_frequency_derivatives(self, omegab):
        # Compute the derivatives of the natural frequencies
        for i, lam in enumerate(self.lam):
            self.lamb[i] += 0.5 * omegab[i] / np.sqrt(lam)

        return

    def get_point_coefficients(self, name):
        x0 = None
        xcoef = None
        if name in self.node_sets:
            nodes = self.node_sets[name]
            weight = 1.0 / len(nodes)

            # Compute the location
            x0 = np.zeros(3)
            x0[0] = weight * np.sum(self.X[nodes, 0])
            x0[1] = weight * np.sum(self.X[nodes, 1])

            if self.Q is not None:
                xcoef = np.zeros((3, self.N), dtype=self.rhoE.dtype)
                for i in range(self.N):
                    xcoef[0, i] = weight * np.sum(self.Q[2 * nodes, i])
                    xcoef[1, i] = weight * np.sum(self.Q[2 * nodes + 1, i])
        else:
            raise ValueError("Unrecognized point name")

        return x0, xcoef

    def add_point_derivative(self, name, x0b, xcoefb):
        if name in self.node_sets:
            if xcoefb is not None:
                nodes = self.node_sets[name]
                weight = 1.0 / len(nodes)

                for i in range(self.N):
                    self.Qb[2 * nodes, i] += weight * xcoefb[0, i]
                    self.Qb[2 * nodes + 1, i] += weight * xcoefb[1, i]

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

    def plot_design(self, path=None, node_sets=False):
        fig, ax = plt.subplots()
        self.plot(self.rho, ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")

        if node_sets:
            for name in self.node_sets:
                for e in self.element_sets[name]:
                    xe = self.X[self.conn[e], 0]
                    ye = self.X[self.conn[e], 1]
                    ax.fill(xe, ye, "b", alpha=0.25)

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


class MinFreqOpt:
    def __init__(self, topo, ks_param=1.0, fixed_mass=1.0):

        self.topo = topo
        self.ks_param = ks_param
        self.fixed_mass = fixed_mass

        # Store the info to evaluate the KS function
        self.ks_min = 0.0
        self.node_sets = self.topo.node_sets

        self.coef = {}
        self.coefb = {}
        self.omega = None
        self.omegab = None

    def initialize(self, store=False):
        self.topo.initialize(store)

        # Get the natural frequencies
        self.omega = self.topo.get_frequencies()

        # Get the coefficients
        self.coef = {}
        for name in self.node_sets:
            _, self.coef[name] = self.topo.get_point_coefficients(name)

        self.ks_min, self.omegab, self.coefb = self._eval_min_frequency(
            self.omega, self.coef, ks_param=self.ks_param, fixed_mass=self.fixed_mass
        )

    def initialize_adjoint(self):
        self.topo.initialize_adjoint()

    def finalize_adjoint(self):
        self.topo.add_frequency_derivatives(self.omegab)
        for name in self.node_sets:
            self.topo.add_point_derivative(name, None, self.coefb[name])

        self.topo.finalize_adjoint()

    def get_area_gradient(self):
        return self.area_gradient

    def get_min_frequency(self):
        return self.ks_min

    def _eval_min_frequency(self, omega, xcoef, ks_param=30.0, fixed_mass=1.0):
        N = len(omega)

        omegab = np.zeros(omega.shape, dtype=self.topo.rhoE.dtype)
        xcoefb = {}

        ks0vals = {}
        eigs = {}

        min_val = np.min(omega)
        for name in xcoef:
            c0 = xcoef[name]
            M0 = np.eye(N) + fixed_mass * np.dot(c0.T, c0)
            K0 = np.diag(omega**2)

            # Compute the natural frequencies of the combined structure + point masses
            lam0, Q0 = eigh(K0, M0)

            # Evaluate the frequencies
            omega0 = np.sqrt(lam0)

            eigs[name] = (omega0, Q0)

            # Find the approximate minimum frequency using the KS function
            min_omega0 = np.min(omega0)
            ks0_exp = np.exp(-ks_param * (omega0 - min_omega0))

            # Compute the minimum omega0
            ks0vals[name] = min_omega0 - np.log(np.sum(ks0_exp)) / ks_param

            if ks0vals[name] < min_val:
                min_val = ks0vals[name]

        eta0 = {}
        exp0_sum = 0.0
        for name in ks0vals:
            eta0[name] = np.exp(-ks_param * (ks0vals[name] - min_val))
            exp0_sum += eta0[name]

        ks = min_val - np.log(exp0_sum) / ks_param

        for name in ks0vals:
            eta0[name] /= exp0_sum

        for name in xcoef:
            c0 = xcoef[name]
            omega0, Q0 = eigs[name]

            # Find the approximate minimum frequency using the KS function
            min_omega0 = np.min(omega0)
            ks0_exp = np.exp(-ks_param * (omega0 - min_omega0))
            ks0_eta = ks0_exp / np.sum(ks0_exp)

            xcoefb[name] = np.zeros(c0.shape, dtype=self.topo.rhoE.dtype)

            omega0b = np.zeros(N)
            for i in range(N):
                omega0b[i] = 0.5 * ks0_eta[i] * eta0[name] / omega0[i]

            # Add the contributions to the derivative wrt omega
            omegab += 2.0 * omega * np.diag(Q0 @ (np.diag(omega0b) @ Q0.T))

            # Add derivative contributions from the mass matrix term
            for i in range(N):
                s = 2.0 * omega0b[i] * fixed_mass * omega0[i] ** 2
                xcoefb[name][:] -= s * np.outer(np.dot(c0, Q0[:, i]), Q0[:, i])

        return ks, omegab, xcoefb

    def test_ks_func(self, dh_cs=1e-6, dh_fd=1e-6, pert=None):
        # Initialize the problem
        self.initialize(store=True)
        ks1 = self.get_min_frequency()

        # Copy the design variables
        x0 = np.array(self.topo.x)

        # Create the derivatives
        self.initialize_adjoint()
        self.finalize_adjoint()
        self.topo.add_check_adjoint_residual(b_ortho=True)

        # Set a random perturbation to the design variables
        if pert is None:
            pert = np.random.uniform(size=x0.shape)

        # The exact derivative
        data = {}
        data["ans"] = np.dot(pert, self.topo.xb)
        data.update(self.topo.profile)

        # Perturb the design variables for finite-difference
        self.topo.x = x0 + dh_fd * pert
        self.initialize()
        ks2 = self.get_min_frequency()

        # Compute the finite-difference and relative error
        data["dh_fd"] = dh_fd
        data["fd"] = (ks2 - ks1) / dh_fd
        data["fd_err"] = np.fabs((data["ans"] - data["fd"]) / data["fd"])

        # Reset the design variables
        self.topo.x = x0

        print("%25s  %25s  %25s" % ("Answer", "FD", "FD Rel Error"))
        print("%25.15e  %25.15e  %25.15e" % (data["ans"], data["fd"], data["fd_err"]))

        return data


def make_model(
    nx=128, ny=64, Lx=1.0, Ly=1.0, rfact=4.0, N=10, Mx=3, My=3, ns=2, **kwargs
):
    """
    Make a symmetric model for optimization

    Parameters
    ----------
    ny : int
        Number of nodes in the y-direction
    rfact : real
        Filter radius as a function of the element side length
    N : int
        Number of eigenvalues and eigenvectors to compute
    Mx : int
        Number of masses in the x-direction
    Ny : int
        Number of masses in the y-direction
    ns : int
        Number of adjacent nodes to include in the node set
    """

    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)

    # Set the fltr matrix
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

    # Create the symmetry in the problem
    dvmap = np.zeros((nx + 1, ny + 1), dtype=int)

    node_sets = {}
    element_sets = {}

    # Make sure that ns is at least rfact
    ns = max(int(ns * ny // 32), int(rfact // 2))

    sx = nx // (Mx - 1)
    sy = ny // (My - 1)

    for i in range(Mx):
        for j in range(My):
            if True:
                name = "node[%d,%d]" % (i, j)
                node_set = []
                element_set = []

                # make sure that the node set is symmetric
                if i < Mx // 2:
                    imin = max(0, sx * i - ns + 1)
                    imax = min(nx, sx * i + ns + 1)
                else:
                    imin_temp = max(0, sx * (Mx - i - 1) - ns + 1)
                    imax_temp = min(nx, sx * (Mx - i - 1) + ns + 1)
                    imin = max(0, nx - imax_temp)
                    imax = min(nx, nx - imin_temp)

                if j < My // 2:
                    jmin = max(0, sy * j - ns)
                    jmax = min(ny, sy * j + ns)
                else:
                    jmin_temp = max(0, sy * (My - j - 1) - ns)
                    jmax_temp = min(ny, sy * (My - j - 1) + ns)
                    jmin = max(0, ny - jmax_temp)
                    jmax = min(ny, ny - jmin_temp)

            for ii in range(imin, imax):
                for jj in range(jmin, jmax):
                    node_set.append(nodes[ii, jj])
                    element_set.append(ii + nx * jj)
                    dvmap[ii, jj] = -1

            node_sets[name] = np.array(node_set, dtype=int)
            element_sets[name] = np.array(element_set, dtype=int)

    index = 0
    for i in range(nx // 2 + 1):
        for j in range(ny // 2 + 1):
            if dvmap[i, j] >= 0:
                dvmap[i, j] = index
                dvmap[nx - i, j] = index
                dvmap[i, ny - j] = index
                dvmap[nx - i, ny - j] = index
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

    topo = TopologyAnalysis(
        fltr, conn, X, N=N, node_sets=node_sets, element_sets=element_sets, **kwargs
    )

    return topo


def make_opt_model(ny=96, rfact=4.0, N=10, Mx=3, My=3, ns=2, **kwargs):
    nx = 4 * ny
    Lx = 4.0
    Ly = 1.0

    topo = make_model(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, rfact=rfact, N=N, Mx=Mx, My=My, ns=ns, **kwargs
    )

    return topo


if __name__ == "__main__":
    np.random.seed(0)

    import sys

    solver_type = "IRAM"
    if "BasicLanczos" in sys.argv:
        solver_type = "BasicLanczos"

    if "dl" in sys.argv:
        solver_type = "BasicLanczos"
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

    print("method = ", method)
    print("adjoint_options = ", adjoint_options)
    print("solver_type = ", solver_type)

    N = 20
    nx = 128
    ny = 64
    topo = make_model(
        nx=nx,
        ny=ny,
        Lx=2.0,
        Ly=1.0,
        N=N,
        solver_type=solver_type,
        adjoint_method=method,
        adjoint_options=adjoint_options,
    )
    opt = MinFreqOpt(topo)

    data = opt.test_ks_func()

    if "adjoint residuals" in data and len(data["adjoint residuals"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(data["adjoint residuals"], marker="o", markersize=4)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")

        plt.show()
