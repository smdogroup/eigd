import warnings

# Import the modified version of ARPACK with the Lanczos extraction
from .arpack import eigsh_mod
import numpy as np
from scipy.sparse import linalg
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._interface import LinearOperator, aslinearoperator


class SpLuOperator(LinearOperator):
    def __init__(self, mat):
        self.lu = splu(mat)
        self.shape = mat.shape
        self.dtype = mat.dtype
        self.count = 0

    def _matvec(self, x):
        if x.ndim == 2:
            self.count += x.shape[1]
        else:
            self.count += 1
        return self.lu.solve(x.astype(self.dtype))


def _project(U, V, X):
    """Perform an oblique projection X <- X - U @ (V.T @ X)"""
    t = V.T @ X
    X[:] -= U @ t
    return X


def add_eig_total_derivative(
    lam,
    Phi,
    lamb,
    Phib,
    psi,
    dAdx,
    dBdx,
    dfdx,
    adj_corr_data={},
    mode="normal",
    deriv_type="vector",
):
    """
    Compute the total derivative given the eigenvector adjoint solution.

    This function works for the regular adjoint or the adjoint corrected in
    the presence of repeated eigenvalues.

    Parameters
    ----------
    lam : ndarray
        Array of eigenvalues of length N.
    Phi : ndarray
        An n by N matrix of the eigenvectors.
    lamb : ndarray
        Array of length N of the derivatives of the function with respect to the
        eigenvalues.
    Phib : ndarray
        A n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    psi : ndarray
        The n by N matrix of the adjoint variables.
    dAdx : function
        Function that takes two arguments w and v of dimension n and returns

        dAdx(w, v) = w.T @ dA/dx @ v.
    dBdx : function
        Function that takes two arguments w and v of dimension n and returns

        dBdx(w, v) = w.T @ dB/dx @ v.
    dfdx : ndarray
        The array that the total derivative will be added to.
    """
    n = Phi.shape[0]
    N = Phi.shape[1]

    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")
    if len(lam) != N:
        raise ValueError(f"Eigenvalues must be of length {N}")
    if psi.shape != (n, N):
        raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
    if Phi.shape != (n, N):
        raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
    if Phib.shape != (n, N):
        raise ValueError(f"Right-hand-side must have the shape ({n},{N})")

    if deriv_type == "vector":
        if mode == "normal":
            for i in range(N):
                # Add the derivative contributions from the stiffness matrix
                if dAdx is not None:
                    w = lamb[i] * Phi[:, i] + psi[:, i]
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            w += xi * Phi[:, j]

                    dfdx += dAdx(w, Phi[:, i])

                if dBdx is not None:
                    # Compute beta
                    beta = 0.5 * Phi[:, i].dot(Phib[:, i])

                    # Add the derivative contributions from the mass matrix
                    w = (beta + lam[i] * lamb[i]) * Phi[:, i] + lam[i] * psi[:, i]
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            w += eta * Phi[:, j]

                    dfdx -= dBdx(w, Phi[:, i])
        elif mode == "buckling":
            for i in range(N):
                if dAdx is not None:
                    # Add the derivative contributions from the mass matrix
                    w = lam[i] * (lamb[i] * Phi[:, i] + psi[:, i])
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            w += eta * Phi[:, j]

                    dfdx += dAdx(w, Phi[:, i])

                if dBdx is not None:
                    # Compute beta
                    beta = 0.5 * Phi[:, i].dot(Phib[:, i])

                    w = (lamb[i] - beta) * Phi[:, i] + psi[:, i]
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            w += xi * Phi[:, j]

                    dfdx += dBdx(w, Phi[:, i])
    elif deriv_type == "tensor":
        W = np.zeros((n, N))
        if mode == "normal":
            if dAdx is not None:
                for i in range(N):
                    W[:, i] = lamb[i] * Phi[:, i] + psi[:, i]
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            W[:, i] += xi * Phi[:, j]

                dfdx += dAdx(W, Phi)

            if dBdx is not None:
                for i in range(N):
                    # Compute beta
                    beta = 0.5 * Phi[:, i].dot(Phib[:, i])

                    # Add the derivative contributions from the mass matrix
                    W[:, i] = (beta + lam[i] * lamb[i]) * Phi[:, i] + lam[i] * psi[:, i]
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            W[:, i] += eta * Phi[:, j]

                dfdx -= dBdx(W, Phi)
        elif mode == "buckling":
            if dAdx is not None:
                for i in range(N):
                    # Add the derivative contributions from the mass matrix
                    W[:, i] = lam[i] * (lamb[i] * Phi[:, i] + psi[:, i])
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            W[:, i] += eta * Phi[:, j]

                dfdx += dAdx(W, Phi)

            if dBdx is not None:
                for i in range(N):
                    # Compute beta
                    beta = 0.5 * Phi[:, i].dot(Phib[:, i])

                    W[:, i] = (lamb[i] - beta) * Phi[:, i] + psi[:, i]
                    if i in adj_corr_data:
                        for j, xi, eta in adj_corr_data[i]:
                            W[:, i] += xi * Phi[:, j]

                dfdx += dBdx(W, Phi)

    return dfdx


def eval_adjoint_residual_norm(A, B, lam, Phi, Phib, psi, mode="normal", b_ortho=False):
    """
    Compute the norm of the residuals of the adjoint equations.

    This computes the norm of the residual

    res[i] = || A @ psi[:, i] - lam[i] * B @ psi[:, i] - b ||_{2}

    for i = 1 ... N where the right-hand-side is given by

    b = - (I - B @ Phi[:, i] @ Phi[:, i].T) @ Phib[:, i].

    In addition, the adjoint solution must satisfy the orthogonality property

    ortho[i] = Phi[:, i].T @ B @ psi[:, i] = 0.

    Parameters
    ----------
    A : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    B : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    lam : ndarray
        Array of eigenvalues of length N.
    Phi : ndarray
        An n by N matrix of the eigenvectors.
    Phib : ndarray
        A n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    psi : ndarray
        The n by N matrix of the adjoint variables.
    b_ortho : bool
        Compute the residuals for the case when the equations are modified such that
        the solution is B orthogonal to all eigenvectors. This equation will satisfy

        Phi.T @ B @ psi = 0.
    mode : str
        If mode == "normal" then for a natural frequency problem, A would be the
        stiffness matrix and B is the mass matrix.

        If mode == "buckling" then for a linearized buckling problem, A would be the
        geometric/stress stiffness matrix and B would be the stiffness matrix.

    Returns
    -------
    psi : ndarray
        The residual norms for i = 1 ... N.
    ortho : ndarray
        The orthogonality condition for i = 1 ... N.
    """

    n = A.shape[1]
    N = Phi.shape[1]

    if len(lam) != N:
        raise ValueError(f"Eigenvalues must be of length {N}")
    if A.shape != (n, n):
        raise ValueError(f"A must have dimensions ({n},{n})")
    if B.shape != (n, n):
        raise ValueError(f"B must have dimensions ({n},{n})")
    if psi.shape != (n, N):
        raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
    if Phi.shape != (n, N):
        raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
    if Phib.shape != (n, N):
        raise ValueError(f"Right-hand-side must have the shape ({n},{N})")
    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")

    if b_ortho:
        BPhi = B @ Phi

    res = np.zeros(N)
    ortho = np.zeros(N)
    for i in range(N):
        w = B @ Phi[:, i]
        b = -(Phib[:, i] - w * Phi[:, i].dot(Phib[:, i]))
        if mode == "normal":
            r = (A @ psi[:, i] - (B @ psi[:, i]) * lam[i]) - b
        else:
            r = (B @ psi[:, i] + (A @ psi[:, i]) * lam[i]) - b

        if b_ortho:
            r = _project(BPhi, Phi, r)
            ortho[i] = np.max(np.abs(BPhi.T @ psi[:, i]))
        else:
            ortho[i] = np.abs(w.dot(psi[:, i]))

        res[i] = np.linalg.norm(r)

    return res, ortho


def _is_close(a, b, atol=1e-5):
    if np.fabs(a - b) < atol:
        return True
    return False


def are_eigenvalues_repeated(lam, atol=1e-5):
    """
    Check if any of the eigenvalues are close to being repeated.

    This relies on the eigenvalues being sorted in ascending order.

    Parameters
    lam : ndarray
        Array of eigenvalues sorted in ascending order.
    """

    N = len(lam)
    for i in range(N - 1):
        if _is_close(lam[i], lam[i + 1], atol=atol):
            return True

    return False


def generate_adjoint_correction(
    lam, Phi, psi, G=None, Phib=None, eig_atol=1e-5, mode="normal"
):
    """
    Add the adjoint "correction". This correction is applied along the computed
    eigenvector directions. For the case of repeated eigenvalues, this correction
    is designed to work with functions that exhibit an underlying differentiability.

    For this correction to work properly, the adjoint solution must be computed
    so that the eigenvector adjoint variables satisfy

    Phi.T @ B @ psi = 0.

    The code works by computing the N by N matrix

    G = -Phi.T @ Phib

    Parameters
    ----------
    G : ndarray
        An N by N matrix.
    A : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    B : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    lam : ndarray
        Array of eigenvalues of length N.
    Phi : ndarray
        An n by N matrix of the eigenvectors.
    Phib : ndarray
        A n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    psi : ndarray
        The n by N matrix of the adjoint variables.

    Other Parameters
    ----------------
    eig_atol : real
        Tolerance for detecting whether two eigenvalues are numerically repeated.
    """

    N = len(lam)
    n = Phi.shape[0]

    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")
    if G is None:
        if Phi.shape != (n, N):
            raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
        if Phib.shape != (n, N):
            raise ValueError(f"Right-hand-side must have the shape ({n},{N})")
        if psi.shape != (n, N):
            raise ValueError(f"Eigenvector adjoint must have the shape ({n},{N})")
    else:
        if G.shape != (N, N):
            raise ValueError(f"G must have dimensions ({N},{N})")
        if Phi.shape != (n, N):
            raise ValueError(f"Phi must have dimensions ({n},{N})")

    if G is None:
        G = -Phi.T @ Phib

    if mode == "normal":
        G0 = G
    elif mode == "buckling":
        G0 = np.diag(lam) @ G

    data = {}
    for i in range(N):
        for j in range(i):
            if _is_close(lam[i], lam[j], atol=eig_atol):
                xi = 0.5 * (G0[j, i] - G0[i, j]) / (lam[j] - lam[i])
                eta = 0.5 * (lam[i] * G0[j, i] - lam[j] * G0[i, j]) / (lam[j] - lam[i])

                if i not in data:
                    data[i] = []
                if j not in data:
                    data[j] = []

                data[i].append((j, xi, eta))
                data[j].append((i, xi, eta))
            else:
                xi = G0[j, i] / (lam[j] - lam[i])
                psi[:, i] += xi * Phi[:, j]

                xi = G0[i, j] / (lam[i] - lam[j])
                psi[:, j] += xi * Phi[:, i]

    return data


def laa(
    Phib,
    B,
    factor,
    sigma,
    lam,
    V,
    Y,
    theta,
    indices,
    D0=None,
    b_ortho=False,
    mode="normal",
):
    """
    Compute the Lanczos adjoint approximation of the eigenvector adjoint equations.

    Assume that the eigenvectors are extracted from a Lanczos subspace that uses
    a shift-invert operator. This extraction process takes the form

    Phi = V @ Y[:, indices].

    Here V is a B-orthonormal Lanczos subspace, Y are the eigenvectors from the
    reduced eigenvalue problem T @ Y[:, i] = theta[i] * Y[:, i], so Y is orthonormal.
    The indices array is selected so that the transformed eigenvalues are in the
    desired order. The symmetric Hessenberg matrix T is not an input since just
    its eigenvalues and eigenvectors are required. The matrix T is

    T = V^{T} @ B @ factor(B @ V)

    where the shift-invert operator computes the action of the operator.
    The eigenvalues of the original problem A @ Phi[:, i] = B @ Phi[:, i] * lam[i] are

    lam[i] = 1.0 / theta[i] + sigma

    for i = 1 ... N.

    The Lanczos adjoint approximation satisfies the following Galerkin property

    V.T @ (A @ psi[:, i] - lam[i] * B @ psi[:, i] - b[:, i]) = 0
    phi[:, i].dot(B @ psi[:, i]) = 0

    where b[:, i] = Phib[: i] - B @ Phi[:, i] * Phi[:, i].dot(Phib[:, i]).

    Parameters
    ----------
    Phib : ndarray
        A n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    B : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    factor : ndarray, sparse matrix or LinearOperator
        A factored matrix that computes the action ``(A - sigma * B)^{-1} @ x``.
    sigma : real
        The scalar shift value used for the factored matrix.
    lam : ndarray
        Array of eigenvalues of length N.
    V : ndarray
        The m-dimensional B-orthogonal Lanczos space.
    Y : ndarray
        The square m-dimensional matrix of eigenvectors for the reduced problem.
    theta : ndarray
        Array of the shifted m eigenvalues of the matrix T.
    indices : ndarray
        Array of indices that sorts the Ritz values.
    D0 : ndarray
        Matrix of dimensions m by N. Specifying this argument overrides the internal
        computation of the matrix D.
    b_ortho : bool
        Make the approximate solution B orthogonal to all converged eigenvectors.

    Returns
    -------
    psi : ndarray
        The n by N Lanczos adjoint approximation.
    """

    n = B.shape[1]
    m = len(theta)
    N = Phib.shape[1]

    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")
    if len(lam) != N:
        raise ValueError(f"Eigenvalues must be of length {N}")
    if Phib.shape != (n, N):
        raise ValueError(f"Right-hand-side must have the shape ({n},{N})")
    if B.shape != (n, n):
        raise ValueError(f"B must have dimensions ({n},{n})")
    if factor.shape != (n, n):
        raise ValueError(f"Factorized operator must have dimensions ({n},{n})")
    if len(indices) != m:
        raise ValueError(f"Length of indices array must be (m = {m})")
    if V.shape != (n, m):
        raise ValueError(f"Dimension of the Lanczos subspace must be ({n},{m})")
    if D0 is not None and D0.shape != (m, N):
        raise ValueError(f"D0 must have dimensions of ({m},{N})")

    if D0 is not None:
        for i in range(N):
            for j in range(i):
                D[indices[i], j] = D0[i, j]
                D[indices[j], i] = D0[j, i]

        for i in range(N, m):
            for j in range(N):
                D[indices[i], j] = D0[i, j]
    elif b_ortho:
        Yb = V.T @ Phib
        D = np.zeros((m, N))

        for i in range(N, m):
            for j in range(N):
                scale = 1.0 / (theta[indices[j]] - theta[indices[i]])
                D[indices[i], j] = scale * Y[:, indices[i]].dot(Yb[:, j])
    else:
        Yb = V.T @ Phib
        D = np.zeros((m, N))

        for i in range(m):
            for j in range(N):
                if i != indices[j]:
                    D[i, j] = Y[:, i].dot(Yb[:, j]) / (theta[indices[j]] - theta[i])

    if mode == "normal":
        psi = -factor(B @ V @ (Y @ (D / (lam - sigma))))
    elif mode == "buckling":
        psi = -factor(B @ V @ (Y @ (sigma * (D / (lam - sigma)))))

    return psi


def dl(
    Phib,
    B,
    factor,
    sigma,
    lam,
    Phi,
    indices,
    V,
    T,
    Y,
    theta,
    eig_atol=1e-5,
    mode="normal",
):
    """
    Compute the solution of the adjoint equations using the AD algorithm.

    Parameters
    ----------
    Phib : ndarray
        A n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    B : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    factor : ndarray, sparse matrix or LinearOperator
        A factored matrix that computes the action ``(A - sigma * B)^{-1} @ x``.
    lam : ndarray
        Array of the N eigenvalues.
    Phi : ndarray
        An n by N matrix of the eigenvectors.
    indices : ndarray
        Array of indices that sorts the Ritz values.
    V : ndarray
        The m-dimensional B-orthogonal Lanczos space.
    T : ndarray
        The reduced m by m matrix Lanczos matrix.
    Y : ndarray
        The square m-dimensional matrix of eigenvectors for the reduced problem.
    theta : ndarray
        Array of the shifted m eigenvalues of the matrix T.
    eig_atol : real
        Tolerance for detecting whether two eigenvalues are equal.

    Returns
    -------
    psi : ndarray
        The n by N adjoint variables.
    data : dict
        Dictionary containing information about the adjoint modification.
    """

    n = B.shape[1]
    m = len(theta)
    N = Phib.shape[1]

    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")
    if len(lam) != N:
        raise ValueError(f"Eigenvalues must be of length {N}")
    if Phib.shape != (n, N):
        raise ValueError(f"Right-hand-side must have the shape ({n},{N})")
    if B.shape != (n, n):
        raise ValueError(f"B must have dimensions ({n},{n})")
    if factor.shape != (n, n):
        raise ValueError(f"Factorized operator must have dimensions ({n},{n})")
    if len(indices) != m:
        raise ValueError(f"Length of indices array must be (m = {m})")
    if V.shape != (n, m):
        raise ValueError(f"Dimension of the Lanczos subspace must be ({n},{m})")

    # Check if we have repeated eigenvalues
    repeated = are_eigenvalues_repeated(lam, atol=eig_atol)

    # These values are defined only if the eigenvalues are repeated
    data = {}

    # Values needed for the case with repeated eigenvalues
    G = None
    BPhi = None

    if repeated:
        # Find the G matrix
        BPhi = B @ Phi
        G = -Phi.T @ Phib

        # Compute the modified residual
        R = Phib + BPhi @ G

        # Compute the derivatives of V and Y
        Vb = R @ Y[:, indices[:N]].T
        Yb = V.T @ R
    else:
        Vb = Phib @ Y[:, indices[:N]].T
        Yb = V.T @ Phib

    D = np.zeros((m, m))
    for i in range(m):
        for j in range(N):
            ii, jj = indices[i], indices[j]
            if ii == jj:
                pass
            elif i < N and j < N and _is_close(lam[i], lam[j], atol=eig_atol):
                pass
            else:
                D[ii, jj] = Y[:, ii].dot(Yb[:, j]) / (theta[jj] - theta[ii])

    # Compute the derivatives of the T matrix
    Tb = Y @ (D @ Y.T)

    # Start the reverse mode through the Lanczos process
    t = B @ factor(B @ V[:, m - 1])

    for j in range(m):
        Vb[:, j] += Tb[j, m - 1] * t
    sb = B @ (V[:, :m] @ Tb[:, m - 1])

    u = factor(sb)
    Vb[:, m - 1] += B @ u

    # Compute Vb, over-writing Vb
    for i in range(m - 2, -1, -1):
        # Reconstruct tB = B @ factor(B @ V[:, i])
        if i == 0:
            t = B @ V[:, i : i + 2] @ T[i : i + 2, i]
        else:
            t = B @ V[:, i - 1 : i + 2] @ T[i - 1 : i + 2, i]

        c0 = V[:, i + 1].dot(Vb[:, i + 1]) - T[i + 1, i] * Tb[i + 1, i]
        sb = (Vb[:, i + 1] - c0 * (B @ V[:, i + 1])) / T[i + 1, i]

        Vb[:, i - 1] -= T[i - 1, i] * sb
        Vb[:, i] -= T[i, i] * sb

        hb = V[:, : i + 1].T @ sb - Tb[: i + 1, i]

        for j in range(i + 1):
            Vb[:, j] -= hb[j] * t
        sb -= B @ (V[:, : i + 1] @ hb)

        # Store the previous value of w
        Vb[:, i + 1] = u

        # Set the new value for Vb
        u = factor(sb)
        Vb[:, i] += B @ u

    # Store the last vector
    Vb[:, 0] = u

    # Compute the adjoint variables
    if mode == "normal":
        psi = -Vb @ (Y[:, indices[:N]] / (lam - sigma))
    elif mode == "buckling":
        psi = -Vb @ (sigma * Y[:, indices[:N]] / (lam - sigma))

    if repeated:
        # Ensure that Phi.T @ B @ psi = 0
        psi = _project(Phi, BPhi, psi)

        # Compute the solution contribution along the eigenvectors
        data = generate_adjoint_correction(
            lam,
            Phi,
            psi,
            G=G,
            eig_atol=eig_atol,
            mode=mode,
        )

    return psi, data


def pcpg(
    Phib,
    A,
    B,
    lam,
    Phi,
    mode="normal",
    psi=None,
    sigma=None,
    factor=None,
    rtol=1e-10,
    atol=1e-30,
    eig_atol=1e-5,
    maxiter=100,
    reset=25,
    callback=None,
):
    """
    Use the PCPG algorithm to solve the eigenvalue adjoint problem.

    Parameters
    ----------
    Phib : ndarray
        An n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    A : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    B : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    lam : ndarray
        Array of eigenvalues of length N.
    Phi : ndarray
        An n by N matrix of the eigenvectors.

    Returns
    -------
    psi : ndarray
        The n by N matrix of the adjoint variables.
    data : dict
        Dictionary containing information about the adjoint modification.

    Other Parameters
    ----------------
    psi : ndarray
        Initial guess for the solution.
    sigma : real
        The scalar shift value used for the factored matrix.
    factor : LinearOperator or function
        Computes the action of a preconditioner ``y = factor(x)``.
    rtol : real
        Relative tolerance.
    atol : real
        Absolute tolerance.
    eig_atol : real
        Tolerance for detecting whether two eigenvalues are equal.
    maxiter : int
        Maximum number of iterations per eigenvalue.
    reset : int
        Reset iteration count.
    callback : function
        User-supplied function passed to gmres.

    References
    ----------
    Alvin, K. F., "Efficient Computation of Eigenvector Sensitivities
    for Structural Dynamics", AIAA Journal, 1997, Vol. 35, No 11
    """

    n = A.shape[1]
    N = Phib.shape[1]

    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")
    if A.shape != (n, n):
        raise ValueError(f"A must have dimensions ({n},{n})")
    if B.shape != (n, n):
        raise ValueError(f"B must have dimensions ({n},{n})")
    if psi is not None and psi.shape != (n, N):
        raise ValueError(f"Initial guess must have the shape ({n},{N})")
    if Phi.shape != (n, N):
        raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
    if Phib.shape != (n, N):
        raise ValueError(f"Right-hand-side must have the shape ({n},{N})")

    if factor is None:
        if sigma is None:
            sigma = 0.9 * lam[0]
        if mode == "normal":
            P = A - sigma * B
        elif mode == "buckling":
            P = B + sigma * A
        factor = SpLuOperator(P.tocsc())

    if psi is not None:
        _psi = psi
    else:
        _psi = np.zeros((n, N), dtype=Phib.dtype)

    # Compute the maximum norm of any of the columns of Phib
    rnorm0 = np.sqrt(np.max(np.sum(Phib**2, axis=0)))

    BPhi = B @ Phi

    G = np.zeros((N, N))

    _info = []
    for i in range(N):
        if mode == "normal":
            R = -Phib[:, i] - (A @ _psi[:, i] - lam[i] * (B @ _psi[:, i]))
        elif mode == "buckling":
            R = -Phib[:, i] - (B @ _psi[:, i] + lam[i] * (A @ _psi[:, i]))
        G[:, i] = Phi.T @ R
        R -= BPhi @ G[:, i]

        Z = np.zeros(n)
        P0 = np.zeros(n)
        zTr_prev = 1.0

        converge = False
        for k in range(maxiter):
            res = np.linalg.norm(R)
            if callback is not None:
                callback(res)

            if res < rtol * rnorm0 or res < atol:
                converge = True
                break

            # Compute the action of the operator
            # (I - Phi @ (B @ Phi).T) @ (A - sigma B)^{-1) @ (I - B @ Phi @ Phi.T) @ r
            Z[:] = R
            Z = _project(Phi, BPhi, factor(_project(BPhi, Phi, Z)))

            if k % reset == 0:  # Reset
                zTr = Z.dot(R)
                P = np.copy(Z)
                zTr_prev = zTr
            else:
                zTr = Z.dot(R)
                beta = zTr / zTr_prev
                zTr_prev = zTr
                P = Z + beta * P0

            # Update the solution
            tA = A @ P
            tB = B @ P
            if mode == "normal":
                alpha = zTr / (tA.dot(P) - lam[i] * tB.dot(P))
            elif mode == "buckling":
                alpha = zTr / (tB.dot(P) + lam[i] * tA.dot(P))

            # Update the solution
            _psi[:, i] += alpha * P

            # Update the residual
            if mode == "normal":
                R[:] = R[:] - alpha * (tA - lam[i] * tB)
            elif mode == "buckling":
                R[:] = R[:] - alpha * (tB + lam[i] * tA)

            # Store the previous steppgmres
            P0[:] = P

        _info.append(converge)

    # Compute the solution contribution along the eigenvectors
    data = generate_adjoint_correction(
        lam, Phi, _psi, G=G, eig_atol=eig_atol, mode=mode
    )

    return _psi, data, _info


def pgmres(
    Phib,
    A,
    B,
    lam,
    Phi,
    mode="normal",
    psi=None,
    sigma=None,
    factor=None,
    rtol=1e-10,
    atol=1e-30,
    eig_atol=1e-5,
    maxiter=50,
    callback=None,
):
    """
    Use right-preconditioned PGMRES to solve the eigenvector adjoint system.

    After each system is solved, the solution for all subsequent adjoints (j = i + 1, ... N-1)
    are updated based on the data used to solve the most recent system. This
    requires the storage of the vectors ``Z[:, i] = M^{-1} @ W[:, i]``.

    Parameters
    ----------
    Phib : ndarray
        An n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    A : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    B : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    lam : ndarray
        Array of eigenvalues of length N.
    Phi : ndarray
        An n by N matrix of the eigenvectors.

    Returns
    -------
    psi : ndarray
        The n by N matrix of the adjoint variables.
    data : dict
        Dictionary containing information about the adjoint modification.

    Other Parameters
    ----------------
    psi : ndarray
        Initial guess for the solution.
    sigma : real
        The scalar shift value used for the factored matrix.
    factor : LinearOperator or function
        Computes the action of a preconditioner ``y = factor(x)``.
    rtol : real
        Relative tolerance.
    atol : real
        Absolute tolerance.
    restart : int
        Parameter for gmres
    maxiter : int
        Parameter for gmres
    callback : function
        User-supplied function passed to gmres.
    """

    n = A.shape[1]
    N = Phib.shape[1]

    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")
    if len(lam) != N:
        raise ValueError(f"Eigenvalues must be of length {N}")
    if A.shape != (n, n):
        raise ValueError(f"A must have dimensions ({n},{n})")
    if B.shape != (n, n):
        raise ValueError(f"B must have dimensions ({n},{n})")
    if psi is not None and psi.shape != (n, N):
        raise ValueError(f"Initial guess must have the shape ({n},{N})")
    if Phi.shape != (n, N):
        raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
    if Phib.shape != (n, N):
        raise ValueError(f"Right-hand-side must have the shape ({n},{N})")

    if factor is None:
        if sigma is None:
            sigma = 0.9 * lam[0]
        if mode == "normal":
            P = A - sigma * B
        elif mode == "buckling":
            P = B + sigma * A
        factor = SpLuOperator(P.tocsc())

    oper = lambda x: factor(_project(BPhi, Phi, x.copy()))
    inner_product = lambda x, y: x.dot(y)
    norm = lambda x: np.sqrt(inner_product(x, x))

    if psi is not None:
        _psi = psi
    else:
        _psi = np.zeros((n, N), dtype=Phib.dtype)

    # Compute the maximum norm of any of the columns of Phib
    rnorm0 = np.sqrt(np.max(np.sum(Phib**2, axis=0)))

    BPhi = B @ Phi

    G = np.zeros((N, N))
    W = np.zeros((n, maxiter + 1))
    Z = np.zeros((n, maxiter))
    H = np.zeros((maxiter + 1, maxiter))

    _info = []

    for i in range(N):
        if mode == "normal":
            R = -Phib[:, i] - (A @ _psi[:, i] - lam[i] * (B @ _psi[:, i]))
        elif mode == "buckling":
            R = -Phib[:, i] - (B @ _psi[:, i] + lam[i] * (A @ _psi[:, i]))
        G[:, i] = Phi.T @ R
        R -= BPhi @ G[:, i]

        beta = norm(R)
        if callback is not None:
            callback(beta)

        if beta < rtol * rnorm0 or beta < atol:
            _info.append(0)
            continue

        W[:, 0] = R / beta

        for j in range(maxiter):
            Z[:, j] = oper(W[:, j])
            tA = A @ Z[:, j]
            tB = B @ Z[:, j]

            if mode == "normal":
                W[:, j + 1] = _project(BPhi, Phi, tA - lam[i] * tB)
            elif mode == "buckling":
                W[:, j + 1] = _project(BPhi, Phi, tB + lam[i] * tA)

            for k in range(j + 1):
                H[k, j] = inner_product(W[:, j + 1], W[:, k])
                W[:, j + 1] -= H[k, j] * W[:, k]

            H[j + 1, j] = norm(W[:, j + 1])
            W[:, j + 1] = W[:, j + 1] / H[j + 1, j]

            rhs = np.zeros(j + 2)
            rhs[0] = beta
            y, _, _, _ = np.linalg.lstsq(H[: j + 2, : j + 1], rhs, rcond=None)
            res = np.linalg.norm(H[: j + 2, : j + 1].dot(y) - rhs)

            if callback is not None:
                callback(res)

            if res < rtol * rnorm0 or res < atol:
                _psi[:, i] += Z[:, : j + 1] @ y
                _info.append(j)
                break
            elif j == maxiter - 1:
                _psi[:, i] += Z[:, : j + 1] @ y
                _info.append(-1)

    # Compute the solution contribution along the eigenvectors Phi
    data = generate_adjoint_correction(
        lam, Phi, _psi, G=G, eig_atol=eig_atol, mode=mode
    )

    return _psi, data, _info


def _solve_lstsq(alpha, H, r):
    """Helper function for the shift-invert method"""
    I = np.eye(H.shape[0], H.shape[1])
    H0 = I - alpha * H
    y, _, _, _ = np.linalg.lstsq(H0, r, rcond=None)
    res = np.linalg.norm(H0 @ y - r)
    return y, res


def sibk(
    Phib,
    A,
    B,
    lam,
    Phi,
    mode="normal",
    psi=None,
    sigma=None,
    factor=None,
    rtol=1e-10,
    atol=1e-30,
    eig_atol=1e-5,
    maxiter=50,
    bs_target=1,
    update_guess=False,
    callback=None,
    nrestart=2,
):
    """
    This function approximately solves the eigenvector adjoint equations using a shift
    and invert block Krylov subspace method.

    The governing equations for the adjoint variables psi[:, i], i = 1 ... N are

    A @ psi[:, i] - lam[i] * B @ psi[:, i] = b[:, i]

    where b[:, i] = - (Phib[:, i] - B @ Phi[:, i] Phi[:, i].dot(Phib[:, i])). The adjoint
    variables must also satisfy the constraint Phi[:, i].dot(B @ psi[:, i]) = 0.

    Here (lam[i], Phi[:, i]) are the eigenvalue/vector pairs of the eigenvalue problem
    A @ Phi[:, i] - lam[i] * B @ Phi[:, i] = 0. The vectors Phib[:, i] are the adjoint
    right-hand-sides.

    The factor argument provides the action of a factorized preconditioner of the form

    factor(x) = (A - sigma * B)^{-1} @ x.

    The method uses a block Krylov method with a block size target ``bs_target``.

    Parameters
    ----------
    Phib : ndarray
        An n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
        to compute a function f(Phi), then Phib = df/d(Phi).
    A : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    B : ndarray, sparse matrix or LinearOperator
        An n by n square matrix operator.
    lam : ndarray
        Array of the N eigenvalues.
    Phi : ndarray
        An n by N matrix of the eigenvectors.

    Returns
    -------
    psi : ndarray
        The n by N matrix of the adjoint variables.
    data : dict
        Dictionary containing information about the adjoint modification.

    Other Parameters
    ----------------
    psi : ndarray
        Initial guess for the solution. Often taken as the Lanczos adjoint approximation.
    factor : ndarray, sparse matrix or LinearOperator
        A factored matrix that computes the action ``(A - sigma * B)^{-1} @ x``.
    sigma : real
        The scalar shift value used for the factored matrix.
    rtol : real
        Relative tolerance to drop residuals.
    atol : real
        Absolute tolerance to drop residuals.
    eig_atol : real
        Tolerance for detecting whether two eigenvalues are equal.
    maxiter : int
        Maximum number of iterations per eigenvalue.
    bs_target : int
        Target block size for the block shift-invert method.
    update_guess : bool
        Apply the updates to subsequent adjoint variables.
    callback : function
        User-supplied function that is called at each iteration.
    nrestart : int
        Number of restarts allowed per eigenvector.
    """

    n = A.shape[1]
    N = Phib.shape[1]

    if mode not in ("normal", "buckling"):
        raise ValueError(f"Unknown mode {mode!r}")
    if len(lam) != N:
        raise ValueError(f"Eigenvalues must be of length {N}")
    if A.shape != (n, n):
        raise ValueError(f"A must have dimensions ({n},{n})")
    if B.shape != (n, n):
        raise ValueError(f"B must have dimensions ({n},{n})")
    if psi is not None and psi.shape != (n, N):
        raise ValueError(f"Initial guess must have the shape ({n},{N})")
    if Phi.shape != (n, N):
        raise ValueError(f"Eigenvectors must have the shape ({n},{N})")
    if Phib.shape != (n, N):
        raise ValueError(f"Right-hand-side must have the shape ({n},{N})")

    inner_product = lambda x, y: x.dot(y)
    norm = lambda x: np.sqrt(inner_product(x, x))

    if factor is None:
        if sigma is None:
            sigma = 0.9 * lam[0]
        if mode == "normal":
            P = A - sigma * B
        elif mode == "buckling":
            P = B + sigma * A
        factor = SpLuOperator(P.tocsc())

    # Compute the maximum norm of any of the columns of Phib
    rnorm0 = np.sqrt(np.max(np.sum(Phib**2, axis=0)))

    # Compute the product B @ Phi
    BPhi = B @ Phi

    # Storage for the Arnoldi subspace
    W = np.zeros((n, maxiter + bs_target))
    Z = np.zeros((n, maxiter))

    # Store space for the G matrix
    G = -Phi.T @ Phib

    # Set the initial guess
    if psi is not None:
        _psi = psi
    else:
        _psi = np.zeros((n, N), dtype=Phib.dtype)

    # Compute the initial residual
    if mode == "normal":
        R = -Phib - (A @ _psi - (B @ _psi) * lam)
    elif mode == "buckling":
        R = -Phib - (B @ _psi + (A @ _psi) * lam)
    R = _project(BPhi, Phi, R)

    _info = []
    i = 0
    restart = 0
    while i < N:
        r = np.zeros((maxiter + bs_target, bs_target))

        bs = 0
        while i + bs < N and bs < bs_target:
            # Re-compute the current residual - the remaining residuals are estimates
            k = i + bs
            if update_guess:
                _psi[:, k] = _project(Phi, BPhi, _psi[:, k])
                if mode == "normal":
                    W[:, bs] = -Phib[:, k] - (
                        A @ _psi[:, k] - lam[k] * (B @ _psi[:, k])
                    )
                elif mode == "buckling":
                    W[:, bs] = -Phib[:, k] - (
                        B @ _psi[:, k] + lam[k] * (A @ _psi[:, k])
                    )
                W[:, bs] = _project(BPhi, Phi, W[:, bs])
            else:
                W[:, bs] = R[:, k]

            beta0 = norm(W[:, bs])
            if callback is not None:
                callback(beta0)

            if beta0 < rtol * rnorm0 or beta0 < atol:
                _info.append(0)
                break

            # Orthogonalize against existing W vectors
            for j in range(bs):
                r[j, bs] = inner_product(W[:, bs], W[:, j])
                W[:, bs] -= r[j, bs] * W[:, j]

            W[:, bs] = _project(BPhi, Phi, W[:, bs])
            r[bs, bs] = norm(W[:, bs])
            W[:, bs] /= r[bs, bs]

            bs += 1

        # No new vectors found
        if bs == 0:
            i += 1
            continue

        H = np.zeros((maxiter + bs, maxiter))
        y = np.zeros((maxiter, bs))

        for j in range(bs, maxiter + bs):
            kp = j - bs
            Z[:, kp] = factor(W[:, kp])
            if mode == "normal":
                W[:, j] = _project(BPhi, Phi, B @ Z[:, kp])
            elif mode == "buckling":
                W[:, j] = _project(BPhi, Phi, A @ Z[:, kp])

            for k in range(j - 1, -1, -1):
                H[k, kp] = inner_product(W[:, j], W[:, k])
                W[:, j] -= H[k, kp] * W[:, k]
            W[:, j] = _project(BPhi, Phi, W[:, j])

            H[j, kp] = norm(W[:, j])
            W[:, j] /= H[j, kp]

            res = 0.0
            H0 = H[: j + 1, : j + 1 - bs]
            for k in range(bs):
                if mode == "normal":
                    alpha = lam[i + k] - sigma
                elif mode == "buckling":
                    alpha = -(lam[i + k] - sigma)
                y[: kp + 1, k], res0 = _solve_lstsq(alpha, H0, r[: j + 1, k])
                res = max((res, res0))

            if callback is not None:
                callback(res)

            if res < rtol * rnorm0 or res < atol:
                _info.append(j)
                _psi[:, i : i + bs] += Z[:, :j] @ y[:j, :]

                if update_guess:
                    # Update the residuals for the remaining adjoints
                    if i + bs < N:
                        # Perform the product with the remaining residuals
                        r0 = W[:, : j + 1].T @ R[:, i + bs :]

                        # Store temp data from the solution of the least squares problems
                        y0 = np.zeros((j + 1 - bs, N - (i + bs)))
                        t0 = np.zeros((j + 1, N - (i + bs)))

                        for k in range(i + bs, N):
                            if mode == "normal":
                                alpha = lam[k] - sigma
                            elif mode == "buckling":
                                alpha = -(lam[k] - sigma)
                            yk, res = _solve_lstsq(alpha, H0, r0[:, k - (i + bs)])
                            y0[:, k - (i + bs)] = yk

                            t0[:, k - (i + bs)] = -alpha * H0 @ yk
                            t0[:-bs, k - (i + bs)] += yk

                        # Update the adjoint variables at once
                        _psi[:, i + bs :] += Z[:, : j + 1 - bs] @ y0

                        # Update all the residuals at once for k = i + bs ... N
                        # R[:, k] -= W[:, :j] @ yk - alpha * W[:, :j+1] @ H @ yk
                        R[:, i + bs :] -= W[:, : j + 1] @ t0

                # Increment the index
                i += bs
                restart = 0

                break
            elif j == maxiter + bs - 1:
                _psi[:, i : i + bs] += Z[:, :j] @ y[:j, :]

                # Terminate if the restarts haven't worked...
                if restart >= nrestart:
                    restart = 0
                    i += bs
                    break

                restart += 1

    # Compute the solution contribution along the eigenvectors Phi
    data = generate_adjoint_correction(
        lam, Phi, _psi, G=G, eig_atol=eig_atol, mode=mode
    )

    return _psi, data, _info


class BasicLanczos:
    """
    Basic implementation of the Lanczos method.

    The primary purpose of this method is to support complex-step verification.

    Parameters
    ----------
    N : int
        Number of eigenvalues and eigenvectors to compute.
    m : int
        Size of the Lanczos subspace.
    tol : real
        Solution tolerance.
    dtype : np.dtype
        Data type to use. Real or complex supported.
    Ntarget : int
        Target number of eigenvalues and eigenvectors to compute. Adjust N so that
        lam[N - 1] and lam[N] are distinct.
    eig_atol : real
        Absolute tolerance for checking if two eigenvalues are numerically repeated.
    mode : str
        If mode == "normal" then for a natural frequency problem, A would be the
        stiffness matrix and B is the mass matrix.

        If mode == "buckling" then for a linearized buckling problem, A would be the
        geometric/stress stiffness matrix and B would be the stiffness matrix.
    """

    def __init__(
        self,
        N=10,
        m=60,
        tol=1e-14,
        Ntarget=None,
        eig_atol=1e-5,
        mode="normal",
        ortho_type="full",
    ):
        self.N = N
        self.m_max = m
        self.tol = tol
        self.Ntarget = Ntarget
        self.eig_atol = eig_atol
        self.mode = mode
        self.ortho_type = ortho_type

        if self.Ntarget is not None and not isinstance(self.Ntarget, int):
            raise ValueError("Ntarget must be an integer or None")
        if ortho_type not in ("full", "selective"):
            raise ValueError(f"Unknown ortho_type {ortho_type!r}")
        if mode not in ("normal", "buckling"):
            raise ValueError(f"Unknown mode {mode!r}")

        return

    def _eigh(self, T):
        """
        Solve the symmetric eigenvalue problem. If the matrix T is complex, treat the
        complex components as forward derivatives
        """

        if np.issubdtype(T.dtype, np.complexfloating):
            lam, Q = np.linalg.eigh(T.real)

            w = np.zeros(T.shape[0], dtype=T.dtype)
            w.real = lam
            w.imag = np.diag(Q.T @ T.imag @ Q)

            v = np.zeros((T.shape[0], T.shape[0]), dtype=T.dtype)
            v.real = Q
            D = Q.T @ T.imag @ Q
            for i in range(T.shape[0]):
                for j in range(T.shape[0]):
                    if i == j or lam[i] == lam[j]:
                        D[i, j] = 0.0
                    else:
                        D[i, j] = D[i, j] / (lam[j] - lam[i])

            v.imag = Q @ D

            return w, v
        else:
            return np.linalg.eigh(T)

    def _solve_reduced_problem(self, alpha, beta, sigma, m, dtype=float):
        """Solve the reduced eigenvalue subproblem"""

        # Now solve the eigenvalue problem
        T = np.zeros((m, m), dtype=dtype)
        for i in range(m):
            T[i, i] = alpha[i]
        for i in range(m - 1):
            T[i, i + 1] = beta[i]
            T[i + 1, i] = beta[i]

        # Compute the eigenvectors of the reduced system
        theta, Y = self._eigh(T)

        # Perform the inverse of the shift and invert to recover the true eigenvalues and
        # sort them so that they are in the desired order
        if self.mode == "normal":
            lam = 1.0 / theta + sigma
            indices = np.argsort(lam)
        elif self.mode == "buckling":
            lam = sigma * theta / (theta - 1.0)
            indices = np.argsort(-1.0 / lam)

        return theta, Y, T, lam, indices

    def _eigenvalues_converged(self, beta, Y, m, N, tol):
        """Check if the smallest N eigenvalues have convergence to the specified tolerance."""
        count = 0
        for j in range(m):
            err = np.abs(beta[m - 1] * Y[m - 1, j])
            if err < tol:
                count += 1
            else:
                break

        return count >= N

    def solve(self, A, B, factor, sigma):
        """
        Solve the eigenvalue problem using a shift and invert Lanczos method with full
        B-orthogonalization. The full-orthogonalization makes this equivalent to Arnoldi,
        but only the tridiagonal coefficients are retained.

        Parameters
        ----------
        A : ndarray, sparse matrix or LinearOperator
            An n by n square matrix operator.
        B : ndarray, sparse matrix or LinearOperator
            An n by n square matrix operator.
        factor : ndarray, sparse matrix or LinearOperator
            if mode == "normal":
                The shift-invert transformation gives the problem

                (A - sigma * B)^{-1} @ B @ phi = 1.0/(lam - sigma) @ phi

                `factor` comptues the action ``(A - sigma * B)^{-1} @ x``.
            elif mode == "buckling":
                The shift-invert transformation gives the problem

                (B + sigma * A)^{-1} @ B @ phi = lam/(lam - sigma) @ phi

                `factor` computes the action ``(B + sigma * A)^{-1} @ x
        sigma : real
            The scalar shift value used for the factored matrix.
        """

        n = A.shape[1]
        dtype = A.dtype

        if A.shape != (n, n):
            raise ValueError(f"A must have dimensions ({n},{n})")
        if B.shape != (n, n):
            raise ValueError(f"B must have dimensions ({n},{n})")
        if factor.shape != (n, n):
            raise ValueError(f"Factorized operator must have dimensions ({n},{n})")

        self.factor = aslinearoperator(factor)
        self.B = aslinearoperator(B)
        self.sigma = sigma

        # This is not used in this function, but is required for derivatives
        self.A = aslinearoperator(A)

        # Form the operator
        oper = lambda x: self.factor(self.B @ x)

        # Set the inner product and norm
        inner_product = lambda x, y: y.dot(self.B @ x)
        norm = lambda x: np.sqrt(inner_product(x, x))

        # Lanczos coefficients
        self.alpha = np.zeros(self.m_max, dtype=dtype)
        self.beta = np.zeros(self.m_max, dtype=dtype)

        # Lanczos subspace
        self.V = np.zeros((n, self.m_max + 1), dtype=dtype)

        # Generate an initial random vector
        rng = np.random.default_rng(12345)
        self.V[:, 0] = rng.uniform(size=n, low=-1.0, high=1.0)

        b0 = norm(self.V[:, 0])
        self.V[:, 0] = self.V[:, 0] / b0

        self.m = self.m_max
        if self.ortho_type == "full":
            for i in range(1, self.m_max + 1):
                # Compute V[i] = factor(B @ V[:, i - 1])
                self.V[:, i] = oper(self.V[:, i - 1])
                if i > 1:
                    self.V[:, i] -= self.beta[i - 2] * self.V[:, i - 2]

                # Perform full orthogonalization using modified Gram Schmidt
                for j in range(i - 1, -1, -1):
                    h = inner_product(self.V[:, j], self.V[:, i])
                    self.V[:, i] -= h * self.V[:, j]

                    if j == i - 1:
                        self.alpha[i - 1] = h

                # Normalize the vector
                self.beta[i - 1] = norm(self.V[:, i])
                self.V[:, i] = self.V[:, i] / self.beta[i - 1]

                if i >= 2:
                    theta, Y, T, lam, indices = self._solve_reduced_problem(
                        self.alpha, self.beta, self.sigma, i, dtype=dtype
                    )

                    Y0 = Y[:, indices]
                    N = self.N
                    if self.Ntarget is not None:
                        N = self.Ntarget
                    if self._eigenvalues_converged(self.beta, Y0, i, N, self.tol):
                        self.m = i
                        break

        elif self.ortho_type == "selective":
            S = None
            for i in range(1, self.m_max + 1):
                # Compute V[i] = factor(B @ V[:, i - 1])
                self.V[:, i] = oper(self.V[:, i - 1])
                if i > 1:
                    self.V[:, i] -= self.beta[i - 2] * self.V[:, i - 2]

                # Perform orthogonalization against the two previous vectors
                # self.V[:, i-1] and self.V[:, i-2]
                for j in range(i - 1, max(-1, i - 3), -1):
                    h = inner_product(self.V[:, j], self.V[:, i])
                    self.V[:, i] -= h * self.V[:, j]

                    if j == i - 1:
                        self.alpha[i - 1] = h

                # Orthogonalize against the existing selected set of Ritz vectors
                if S is not None:
                    for j in range(S.shape[1]):
                        h = inner_product(S[:, j], self.V[:, i])
                        self.V[:, i] -= h * S[:, j]

                # Normalize the vector
                self.beta[i - 1] = norm(self.V[:, i])
                self.V[:, i] = self.V[:, i] / self.beta[i - 1]

                # Update the Ritz values that are nearly converged
                if i >= 2:
                    theta, Y, T, lam, indices = self._solve_reduced_problem(
                        self.alpha, self.beta, self.sigma, i, dtype=dtype
                    )

                    # Sort the eigenvectors by index
                    Y0 = Y[:, indices]

                    N = self.N
                    if self.Ntarget is not None:
                        N = self.Ntarget
                    if self._eigenvalues_converged(self.beta, Y0, i, N, self.tol):
                        self.m = i
                        break

                    # Find the indices of the almost converged Ritz values
                    conv = []
                    err_tol = np.sqrt(self.tol)
                    for j in range(i):
                        err = np.abs(self.beta[i - 1] * Y0[i - 1, j])
                        if err < err_tol:
                            conv.append(j)

                    # Compute the Ritz vectors for the selected set of converged eigenvalues
                    S = self.V[:, :i] @ Y0[:, conv]

        # Solve the reduced eigenvalue problem
        self.theta, self.Y, self.T, self.lam, self.indices = (
            self._solve_reduced_problem(
                self.alpha, self.beta, self.sigma, self.m, dtype=dtype
            )
        )

        # Apply an adaptive approach to select N so that lam[N] and lam[N + 1] are distinct
        if self.Ntarget is not None:
            self.N = self.Ntarget
            while self.N < self.m:
                if _is_close(
                    self.lam[self.indices[self.N - 1]].real,
                    self.lam[self.indices[self.N]].real,
                    self.eig_atol,
                ):
                    self.N += 1
                else:
                    break
        else:
            if _is_close(
                self.lam[self.indices[self.N - 1]].real,
                self.lam[self.indices[self.N]].real,
                self.eig_atol,
            ):
                warnings.warn(
                    f"BasicLanczos: Ritz values {self.N} and {self.N+1} are numerically repeated."
                )

        self.lam0 = self.lam[self.indices[: self.N]]
        self.Y0 = self.Y[:, self.indices[: self.N]]

        # Check whether we've met the tolerance or not
        self.fail = False
        self.eig_res = np.zeros(self.N)
        for j in range(self.N):
            self.eig_res[j] = np.abs(self.beta[-1] * self.Y0[-1, j])
            if self.eig_res[j] > self.tol:
                self.fail = True

        # Now, compute the eigenvectors
        self.Phi = self.V[:, : self.m].dot(self.Y0)

        return self.lam0, self.Phi

    def solve_adjoint(
        self,
        Phib,
        method="sibk",
        psi=None,
        rtol=1e-10,
        atol=1e-30,
        lanczos_guess=True,
        **kwargs,
    ):
        """
        Solve the adjoint equations for the eigenvector problem.

        This method first computes the Lanczos adjoint approximation then solves the
        adjoint system using the block eigenvector adjoint linear solver. If nrestart = 0,
        then only the Lanczos adjoint approximation is returned.

        Parameters
        ----------
        Phib : ndarray
            An n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
            to compute a function f(Phi), then Phib = df/d(Phi).

        Returns
        -------
        psi : ndarray
            The n by N adjoint variables.
        data : dict
            Dictionary containing information about the adjoint modification.

        Other Parameters
        ----------------
        method : str
            The type of method to use either "pcpg", "pgmres", "sibk", "dl" or "laa"
        psi : ndarray
            An n by N matrix of the initial guess for the adjoint variables.
        rtol : real
            Relative tolerance for the adjoint equations.
        atol : real
            Absolute tolerance for the adjoint equations.
        eig_atol : real
            Tolerance for detecting whether two eigenvalues are equal.
        lanczos_guess : bool
            Use an initial guess for psi.
        """
        n = self.A.shape[1]

        if method not in ("pcpg", "pgmres", "sibk", "laa", "dl"):
            raise ValueError(f"Unknown method {method!r}")
        if psi is not None and psi.shape != (n, self.N):
            raise ValueError(f"Initial guess must have the shape ({n},{self.N})")

        if method == "dl":
            lanczos_guess = False

        data = {}  # The adjoint correction data
        if lanczos_guess or method == "laa":
            psi = laa(
                Phib,
                self.B,
                self.factor,
                self.sigma,
                self.lam0,
                self.V[:, : self.m],
                self.Y,
                self.theta,
                self.indices,
                b_ortho=True,
                mode=self.mode,
            )
        else:
            psi = np.zeros((n, self.N))

        if method == "pcpg":
            psi, data, info = pcpg(
                Phib,
                self.A,
                self.B,
                self.lam0,
                self.Phi,
                mode=self.mode,
                psi=psi,
                factor=self.factor,
                rtol=rtol,
                atol=atol,
                eig_atol=self.eig_atol,
                **kwargs,
            )
        elif method == "pgmres":
            psi, data, info = pgmres(
                Phib,
                self.A,
                self.B,
                self.lam0,
                self.Phi,
                mode=self.mode,
                psi=psi,
                factor=self.factor,
                rtol=rtol,
                atol=atol,
                eig_atol=self.eig_atol,
                **kwargs,
            )
        elif method == "sibk":
            psi, data, info = sibk(
                Phib,
                self.A,
                self.B,
                mode=self.mode,
                factor=self.factor,
                sigma=self.sigma,
                lam=self.lam0,
                Phi=self.Phi,
                psi=psi,
                rtol=rtol,
                atol=atol,
                eig_atol=self.eig_atol,
                **kwargs,
            )
        elif method == "laa":
            data = generate_adjoint_correction(
                self.lam0,
                self.Phi,
                psi,
                Phib=Phib,
                eig_atol=self.eig_atol,
                mode=self.mode,
            )
        elif method == "dl":
            psi, data = dl(
                Phib,
                self.B,
                self.factor,
                self.sigma,
                self.lam0,
                self.Phi,
                self.indices,
                self.V[:, : self.m],
                self.T,
                self.Y,
                self.theta,
                self.eig_atol,
                mode=self.mode,
            )

        return psi, data

    def eval_adjoint_residual_norm(self, Phib, psi, b_ortho=False):
        """
        Evaluate the adjoint residual norm and orthogonality

        Parameters
        ----------
        Phib : ndarray
            An n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
            to compute a function f(Phi), then Phib = df/d(Phi).
        psi : ndarray
            The n by N adjoint variables.

        Returns
        -------
        res : ndarray
            An N dimensional array of the residuals
        ortho : ndarray
            An N dimensional array containing the value Phi[:, i].dot(B @ psi[:, i])
        """

        return eval_adjoint_residual_norm(
            self.A,
            self.B,
            self.lam0,
            self.Phi,
            Phib,
            psi,
            mode=self.mode,
            b_ortho=b_ortho,
        )

    def add_total_derivative(
        self, lamb, Phib, psi, dAdx, dBdx, dfdx, adj_corr_data={}, deriv_type="vector"
    ):
        """
        Compute the total derivative

        Parameters
        ----------
        lamb : ndarray
            Array of length N of the derivatives of the function with respect to the
            eigenvalues.
        Phib : ndarray
            A n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
            to compute a function f(Phi), then Phib = df/d(Phi).
        psi : ndarray
            The n by N matrix of the adjoint variables.
        dAdx : function
            Function that takes two arguments w and v of dimension n and returns

            dAdx(w, v) = w.T @ dA/dx @ v.
        dBdx : function
            Function that takes two arguments w and v of dimension n and returns

            dBdx(w, v) = w.T @ dB/dx @ v.
        dfdx : ndarray
            The array that the total derivative will be added to.
        """

        return add_eig_total_derivative(
            self.lam0,
            self.Phi,
            lamb,
            Phib,
            psi,
            dAdx,
            dBdx,
            dfdx,
            adj_corr_data=adj_corr_data,
            mode=self.mode,
            deriv_type=deriv_type,
        )


class IRAM:
    """
    Wrapper for ARPACK (an Implicitly-Restarted Arnoldi Method) that also extracts the
    Lanczos subspace and the symmetric Hessenberg matrix to compute the Lanczos approximation
    of the eigenvector adjoint equations.

    This uses the wrapper from scipy with an extra layer that extracts T and V.

    Parameters
    ----------
    N : int
        Number of eigenvalues and eigenvectors to compute.
    m : int
        Size of the Lanczos subspace.
    eig_atol : real
        Tolerance for detecting whether two eigenvalues are numerically repeated.
    tol : real
        Solution tolerance passed to ARPACK.
    """

    def __init__(self, N=10, m=None, eig_atol=1e-5, tol=0.0, mode="normal"):
        self.N = N
        if m is None:
            self.m = np.max((20, 2 * N + 1))
        else:
            self.m = np.max((20, 2 * N + 1, m))
        self.tol = tol
        self.eig_atol = eig_atol
        self.mode = mode

        if mode not in ("normal", "buckling"):
            raise ValueError(f"Unknown mode {mode!r}")

        return

    def solve(self, A, B, factor, sigma):
        """
        Solve the eigenvalue problem using a shift and invert Lanczos method with
        implicit restarting using ARPACK.

        Parameters
        ----------
        A : ndarray, sparse matrix or LinearOperator
            An n by n square matrix operator.
        B : ndarray, sparse matrix or LinearOperator
            An n by n square matrix operator.
        factor : ndarray, sparse matrix or LinearOperator
            A factored matrix that computes the action ``(A - sigma * B)^{-1} @ x``.
        sigma : real
            The scalar shift value used for the factored matrix.
        """

        n = A.shape[1]
        if A.shape != (n, n):
            raise ValueError(f"A must have dimensions ({n},{n})")
        if B.shape != (n, n):
            raise ValueError(f"B must have dimensions ({n},{n})")
        if factor.shape != (n, n):
            raise ValueError(f"Factorized operator must have dimensions ({n},{n})")

        self.factor = aslinearoperator(factor)
        self.B = aslinearoperator(B)
        self.sigma = sigma
        self.A = aslinearoperator(A)

        which = "LM"
        if self.mode == "normal":
            A0 = self.A
        elif self.mode == "buckling":
            A0 = self.B

        self.lam, self.Phi, self.T, self.V = eigsh_mod(
            A0,
            M=self.B,
            OPinv=self.factor,
            k=self.N,
            sigma=self.sigma,
            which=which,
            mode=self.mode,
            tol=self.tol,
            ncv=self.m,
        )

        # Perform an extra extraction step here.. This could be avoided if we could
        # extract Ym directly from ARPACK.
        self.theta, self.Y = np.linalg.eigh(self.T)

        if self.mode == "normal":
            eigs = 1.0 / self.theta + sigma
            self.indices = np.argsort(eigs)
        elif self.mode == "buckling":
            eigs = sigma * self.theta / (self.theta - 1.0)
            self.indices = np.argsort(-1.0 / eigs)

        if _is_close(
            eigs[self.indices[self.N - 1]].real,
            eigs[self.indices[self.N]].real,
            self.eig_atol,
        ):
            warnings.warn(
                f"IRAM: Ritz values {self.N} and {self.N+1} are numerically repeated."
            )

        # Use a modal assurance to test whether to flip the sign on Ym.
        # Here mac will be +/- 1
        for i in range(self.N):
            q = self.V @ self.Y[:, self.indices[i]]
            denom = np.linalg.norm(q) * np.linalg.norm(self.Phi[:, i])
            mac = self.Phi[:, i].dot(q) / denom

            if mac < 0.0:
                self.Y[:, self.indices[i]] *= -1.0

        return self.lam, self.Phi

    def solve_adjoint(
        self,
        Phib,
        method="sibk",
        psi=None,
        rtol=1e-10,
        atol=1e-30,
        lanczos_guess=True,
        **kwargs,
    ):
        """
        Solve the adjoint equations for the eigenvector problem.

        This method first computes the Lanczos adjoint approximation then solves the
        adjoint system using the block eigenvector adjoint linear solver. If nrestart = 0,
        then only the Lanczos adjoint approximation is returned.

        Parameters
        ----------
        Phib : ndarray
            An n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
            to compute a function f(Phi), then Phib = df/d(Phi).

        Returns
        -------
        psi : ndarray
            The n by N adjoint variables.
        data : dict
            Dictionary containing information about the adjoint modification.

        Other Parameters
        ----------------
        psi : ndarray
            An n by N matrix of the initial guess for the adjoint variables. If not specified,
            the Lanczos adjoint approximation is computed.
        rtol : real
            Relative tolerance to drop residuals.
        atol : real
            Absolute tolerance to drop residuals.
        lanczos_guess : bool
            Use an initial guess for psi.
        """
        n = self.A.shape[1]

        if method not in ("pcpg", "pgmres", "sibk", "laa", "dl"):
            raise ValueError(f"Unknown method {method!r}")
        if psi is not None and psi.shape != (n, self.N):
            raise ValueError(f"Initial guess must have the shape ({n},{self.N})")

        data = {}  # Adjoint correction data

        if method == "dl":
            warnings.warn(
                f'Adjoint method "{method}" is not recommended for the ARPACK IRAM eigenvalue sovler.'
            )
            lanczos_guess = False

        if lanczos_guess or method == "laa":
            psi = laa(
                Phib,
                self.B,
                self.factor,
                self.sigma,
                self.lam,
                self.V,
                self.Y,
                self.theta,
                self.indices,
                b_ortho=True,
                mode=self.mode,
            )
        else:
            psi = np.zeros((n, self.N))

        if method == "pcpg":
            psi, data, info = pcpg(
                Phib,
                self.A,
                self.B,
                self.lam,
                self.Phi,
                mode=self.mode,
                psi=psi,
                factor=self.factor,
                rtol=rtol,
                atol=atol,
                eig_atol=self.eig_atol,
                **kwargs,
            )
        elif method == "pgmres":
            psi, data, info = pgmres(
                Phib,
                self.A,
                self.B,
                self.lam,
                self.Phi,
                mode=self.mode,
                psi=psi,
                factor=self.factor,
                rtol=rtol,
                atol=atol,
                eig_atol=self.eig_atol,
                **kwargs,
            )
        elif method == "sibk":
            psi, data, info = sibk(
                Phib,
                self.A,
                self.B,
                mode=self.mode,
                factor=self.factor,
                sigma=self.sigma,
                lam=self.lam,
                Phi=self.Phi,
                psi=psi,
                rtol=rtol,
                atol=atol,
                eig_atol=self.eig_atol,
                **kwargs,
            )
        elif method == "laa":
            data = generate_adjoint_correction(
                self.lam,
                self.Phi,
                psi,
                Phib=Phib,
                eig_atol=self.eig_atol,
                mode=self.mode,
            )
        elif method == "dl":
            psi, data = dl(
                Phib,
                self.B,
                self.factor,
                self.sigma,
                self.lam,
                self.Phi,
                self.indices,
                self.V,
                self.T,
                self.Y,
                self.theta,
                self.eig_atol,
                mode=self.mode,
            )

        return psi, data

    def eval_adjoint_residual_norm(self, Phib, psi, b_ortho=False):
        """
        Evaluate the adjoint residual norm and orthogonality

        Parameters
        ----------
        Phib : ndarray
            An n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
            to compute a function f(Phi), then Phib = df/d(Phi).
        psi : ndarray
            The n by N adjoint variables.

        Returns
        -------
        res : ndarray
            An N dimensional array of the residuals
        ortho : ndarray
            An N dimensional array containing the value Phi[:, i].dot(B @ psi[:, i])
        """

        return eval_adjoint_residual_norm(
            self.A,
            self.B,
            self.lam,
            self.Phi,
            Phib,
            psi,
            mode=self.mode,
            b_ortho=b_ortho,
        )

    def add_total_derivative(
        self, lamb, Phib, psi, dAdx, dBdx, dfdx, adj_corr_data={}, deriv_type="vector"
    ):
        """
        Compute the total derivative

        Parameters
        ----------
        lamb : ndarray
            Array of length N of the derivatives of the function with respect to the
            eigenvalues.
        Phib : ndarray
            A n by N matrix of the adjoint right-hand-sides. If the eigenvectors are used
            to compute a function f(Phi), then Phib = df/d(Phi).
        psi : ndarray
            The n by N matrix of the adjoint variables.
        dAdx : function
            Function that takes two arguments w and v of dimension n and returns

            dAdx(w, v) = w.T @ dA/dx @ v.
        dBdx : function
            Function that takes two arguments w and v of dimension n and returns

            dBdx(w, v) = w.T @ dB/dx @ v.
        dfdx : ndarray
            The array that the total derivative will be added to.
        """

        return add_eig_total_derivative(
            self.lam,
            self.Phi,
            lamb,
            Phib,
            psi,
            dAdx,
            dBdx,
            dfdx,
            adj_corr_data=adj_corr_data,
            mode=self.mode,
            deriv_type=deriv_type,
        )
