import warnings
import numpy as np
import scipy
from scipy.sparse.linalg._eigen.arpack.arpack import (
    _SymmetricArpackParams,
    _aslinearoperator_with_dtype,
    get_OPinv_matvec,
    get_inv_matvec,
    ArpackError,
)
from scipy.sparse import issparse
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.linalg import eig, eigh
from scipy._lib._threadsafety import ReentrancyLock


# ARPACK is not threadsafe or reentrant (SAVE variables), so we need a
# lock and a re-entering check.
_ARPACK_LOCK = ReentrancyLock(
    "Nested calls to eigs/eighs not allowed: " "ARPACK is not re-entrant"
)


class _SymmetricArpackParamsAndData(_SymmetricArpackParams):
    def __init__(
        self,
        n,
        k,
        tp,
        matvec,
        mode=1,
        M_matvec=None,
        Minv_matvec=None,
        sigma=None,
        ncv=None,
        v0=None,
        maxiter=None,
        which="LM",
        tol=0,
    ):
        _SymmetricArpackParams.__init__(
            self,
            n,
            k,
            tp,
            matvec,
            mode=mode,
            M_matvec=M_matvec,
            Minv_matvec=Minv_matvec,
            sigma=sigma,
            ncv=ncv,
            v0=v0,
            maxiter=maxiter,
            which=which,
            tol=tol,
        )

    def extract(self, return_eigenvectors):
        rvec = return_eigenvectors
        ierr = 0
        howmny = "A"  # return all eigenvectors
        sselect = np.zeros(self.ncv, "int")  # unused
        h = self.workl[0 : 2 * self.ncv].copy()
        v = self.v.copy()
        v = v.reshape((-1, self.ncv))

        d, z, ierr = self._arpack_extract(
            rvec,
            howmny,
            sselect,
            self.sigma,
            self.bmat,
            self.which,
            self.k,
            self.tol,
            self.resid,
            self.v,
            self.iparam[0:7],
            self.ipntr,
            self.workd[0 : 2 * self.n],
            self.workl,
            ierr,
        )
        if ierr != 0:
            raise ArpackError(ierr, infodict=self.extract_infodict)
        k_ok = self.iparam[4]
        d = d[:k_ok]
        z = z[:, :k_ok]

        Tm = np.zeros((self.ncv, self.ncv))
        for i in range(self.ncv - 1):
            Tm[i, i + 1] = h[i + 1]
            Tm[i + 1, i] = h[i + 1]

        for i in range(self.ncv):
            Tm[i, i] = h[self.ncv + i]

        if return_eigenvectors:
            return d, z, Tm, v
        else:
            return d


def eigsh_mod(
    A,
    k=6,
    M=None,
    sigma=None,
    which="LM",
    v0=None,
    ncv=None,
    maxiter=None,
    tol=0,
    return_eigenvectors=True,
    Minv=None,
    OPinv=None,
    mode="normal",
):
    """
    Find k eigenvalues and eigenvectors of the real symmetric square matrix
    or complex Hermitian matrix A.

    Solves ``A @ x[i] = w[i] * x[i]``, the standard eigenvalue problem for
    w[i] eigenvalues with corresponding eigenvectors x[i].

    If M is specified, solves ``A @ x[i] = w[i] * M @ x[i]``, the
    generalized eigenvalue problem for w[i] eigenvalues
    with corresponding eigenvectors x[i].

    Note that there is no specialized routine for the case when A is a complex
    Hermitian matrix. In this case, ``eigsh()`` will call ``eigs()`` and return the
    real parts of the eigenvalues thus obtained.

    Parameters
    ----------
    A : ndarray, sparse matrix or LinearOperator
        A square operator representing the operation ``A @ x``, where ``A`` is
        real symmetric or complex Hermitian. For buckling mode (see below)
        ``A`` must additionally be positive-definite.
    k : int, optional
        The number of eigenvalues and eigenvectors desired.
        `k` must be smaller than N. It is not possible to compute all
        eigenvectors of a matrix.

    Returns
    -------
    w : array
        Array of k eigenvalues.
    v : array
        An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
        the eigenvector corresponding to the eigenvalue ``w[i]``.

    Other Parameters
    ----------------
    M : An N x N matrix, array, sparse matrix, or linear operator representing
        the operation ``M @ x`` for the generalized eigenvalue problem

            A @ x = w * M @ x.

        M must represent a real symmetric matrix if A is real, and must
        represent a complex Hermitian matrix if A is complex. For best
        results, the data type of M should be the same as that of A.
        Additionally:

            If sigma is None, M is symmetric positive definite.

            If sigma is specified, M is symmetric positive semi-definite.

            In buckling mode, M is symmetric indefinite.

        If sigma is None, eigsh requires an operator to compute the solution
        of the linear equation ``M @ x = b``. This is done internally via a
        (sparse) LU decomposition for an explicit matrix M, or via an
        iterative solver for a general linear operator.  Alternatively,
        the user can supply the matrix or operator Minv, which gives
        ``x = Minv @ b = M^-1 @ b``.
    sigma : real
        Find eigenvalues near sigma using shift-invert mode.  This requires
        an operator to compute the solution of the linear system
        ``[A - sigma * M] x = b``, where M is the identity matrix if
        unspecified.  This is computed internally via a (sparse) LU
        decomposition for explicit matrices A & M, or via an iterative
        solver if either A or M is a general linear operator.
        Alternatively, the user can supply the matrix or operator OPinv,
        which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
        Note that when sigma is specified, the keyword 'which' refers to
        the shifted eigenvalues ``w'[i]`` where:

            if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.

            if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.

            if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.

        (see further discussion in 'mode' below)
    v0 : ndarray, optional
        Starting vector for iteration.
        Default: random
    ncv : int, optional
        The number of Lanczos vectors generated ncv must be greater than k and
        smaller than n; it is recommended that ``ncv > 2*k``.
        Default: ``min(n, max(2*k + 1, 20))``
    which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
        If A is a complex Hermitian matrix, 'BE' is invalid.
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : Largest (in magnitude) eigenvalues.

            'SM' : Smallest (in magnitude) eigenvalues.

            'LA' : Largest (algebraic) eigenvalues.

            'SA' : Smallest (algebraic) eigenvalues.

            'BE' : Half (k/2) from each end of the spectrum.

        When k is odd, return one more (k/2+1) from the high end.
        When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
        (see discussion in 'sigma', above).  ARPACK is generally better
        at finding large values than small values.  If small eigenvalues are
        desired, consider using shift-invert mode for better performance.
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        Default: ``n*10``
    tol : float
        Relative accuracy for eigenvalues (stopping criterion).
        The default value of 0 implies machine precision.
    Minv : N x N matrix, array, sparse matrix, or LinearOperator
        See notes in M, above.
    OPinv : N x N matrix, array, sparse matrix, or LinearOperator
        See notes in sigma, above.
    return_eigenvectors : bool
        Return eigenvectors (True) in addition to eigenvalues.
        This value determines the order in which eigenvalues are sorted.
        The sort order is also dependent on the `which` variable.

            For which = 'LM' or 'SA':
                If `return_eigenvectors` is True, eigenvalues are sorted by
                algebraic value.

                If `return_eigenvectors` is False, eigenvalues are sorted by
                absolute value.

            For which = 'BE' or 'LA':
                eigenvalues are always sorted by algebraic value.

            For which = 'SM':
                If `return_eigenvectors` is True, eigenvalues are sorted by
                algebraic value.

                If `return_eigenvectors` is False, eigenvalues are sorted by
                decreasing absolute value.

    mode : string ['normal' | 'buckling' | 'cayley']
        Specify strategy to use for shift-invert mode.  This argument applies
        only for real-valued A and sigma != None.  For shift-invert mode,
        ARPACK internally solves the eigenvalue problem
        ``OP @ x'[i] = w'[i] * B @ x'[i]``
        and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
        into the desired eigenvectors and eigenvalues of the problem
        ``A @ x[i] = w[i] * M @ x[i]``.
        The modes are as follows:

            'normal' :
                OP = [A - sigma * M]^-1 @ M,
                B = M,
                w'[i] = 1 / (w[i] - sigma)

            'buckling' :
                OP = [A - sigma * M]^-1 @ A,
                B = A,
                w'[i] = w[i] / (w[i] - sigma)

            'cayley' :
                OP = [A - sigma * M]^-1 @ [A + sigma * M],
                B = M,
                w'[i] = (w[i] + sigma) / (w[i] - sigma)

        The choice of mode will affect which eigenvalues are selected by
        the keyword 'which', and can also impact the stability of
        convergence (see [2] for a discussion).

    Raises
    ------
    ArpackNoConvergence
        When the requested convergence is not obtained.

        The currently converged eigenvalues and eigenvectors can be found
        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
        object.

    See Also
    --------
    eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
    svds : singular value decomposition for a matrix A

    Notes
    -----
    This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
    functions which use the Implicitly Restarted Lanczos Method to
    find the eigenvalues and eigenvectors [2]_.

    References
    ----------
    .. [1] ARPACK Software, https://github.com/opencollab/arpack-ng
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import eigsh
    >>> identity = np.eye(13)
    >>> eigenvalues, eigenvectors = eigsh(identity, k=6)
    >>> eigenvalues
    array([1., 1., 1., 1., 1., 1.])
    >>> eigenvectors.shape
    (13, 6)

    """

    n = A.shape[0]

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k >= n:
        warnings.warn(
            "k >= N for N * N square matrix. "
            "Attempting to use scipy.linalg.eigh instead.",
            RuntimeWarning,
            stacklevel=2,
        )

        if issparse(A):
            raise TypeError(
                "Cannot use scipy.linalg.eigh for sparse A with "
                "k >= N. Use scipy.linalg.eigh(A.toarray()) or"
                " reduce k."
            )
        if isinstance(A, LinearOperator):
            raise TypeError(
                "Cannot use scipy.linalg.eigh for LinearOperator " "A with k >= N."
            )
        if isinstance(M, LinearOperator):
            raise TypeError(
                "Cannot use scipy.linalg.eigh for LinearOperator " "M with k >= N."
            )

        return eigh(A, b=M, eigvals_only=not return_eigenvectors)

    if sigma is None:
        A = _aslinearoperator_with_dtype(A)
        matvec = A.matvec

        if OPinv is not None:
            raise ValueError("OPinv should not be specified " "with sigma = None.")
        if M is None:
            # standard eigenvalue problem
            mode = 1
            M_matvec = None
            Minv_matvec = None
            if Minv is not None:
                raise ValueError("Minv should not be " "specified with M = None.")
        else:
            # general eigenvalue problem
            mode = 2
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = _aslinearoperator_with_dtype(Minv)
                Minv_matvec = Minv.matvec
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    else:
        # sigma is not None: shift-invert mode
        if Minv is not None:
            raise ValueError("Minv should not be specified when sigma is")

        # normal mode
        if mode == "normal":
            mode = 3
            matvec = None
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma, hermitian=True, tol=tol)
            else:
                OPinv = _aslinearoperator_with_dtype(OPinv)
                Minv_matvec = OPinv.matvec
            if M is None:
                M_matvec = None
            else:
                M = _aslinearoperator_with_dtype(M)
                M_matvec = M.matvec

        # buckling mode
        elif mode == "buckling":
            mode = 4
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma, hermitian=True, tol=tol)
            else:
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            matvec = _aslinearoperator_with_dtype(A).matvec
            M_matvec = None

        # cayley-transform mode
        elif mode == "cayley":
            mode = 5
            matvec = _aslinearoperator_with_dtype(A).matvec
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma, hermitian=True, tol=tol)
            else:
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            if M is None:
                M_matvec = None
            else:
                M_matvec = _aslinearoperator_with_dtype(M).matvec

        # unrecognized mode
        else:
            raise ValueError("unrecognized mode '%s'" % mode)

    params = _SymmetricArpackParamsAndData(
        n,
        k,
        A.dtype.char,
        matvec,
        mode,
        M_matvec,
        Minv_matvec,
        sigma,
        ncv,
        v0,
        maxiter,
        which,
        tol,
    )

    with _ARPACK_LOCK:
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)
