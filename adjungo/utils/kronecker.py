"""Block Kronecker product utilities for GLM systems."""

import numpy as np
from numpy.typing import NDArray


def kronecker_eye(A: NDArray, n: int) -> NDArray:
    """
    Compute A ⊗ I_n (Kronecker product with identity).

    Useful for forming block-diagonal systems in fully implicit solvers.

    Args:
        A: Matrix (s, s)
        n: Size of identity matrix

    Returns:
        A ⊗ I_n of shape (s*n, s*n)
    """
    s = A.shape[0]
    result = np.zeros((s * n, s * n))

    for i in range(s):
        for j in range(s):
            result[i*n:(i+1)*n, j*n:(j+1)*n] = A[i, j] * np.eye(n)

    return result


def eye_kronecker(n: int, B: NDArray) -> NDArray:
    """
    Compute I_n ⊗ B (identity Kronecker B).

    Args:
        n: Size of identity matrix
        B: Matrix (m, m)

    Returns:
        I_n ⊗ B of shape (n*m, n*m)
    """
    m = B.shape[0]
    result = np.zeros((n * m, n * m))

    for i in range(n):
        result[i*m:(i+1)*m, i*m:(i+1)*m] = B

    return result


def block_matvec(A: NDArray, x: NDArray, s: int, n: int) -> NDArray:
    """
    Compute (A ⊗ I_n) @ x efficiently without forming full matrix.

    Args:
        A: Block structure matrix (s, s)
        x: Vector of length s*n (or matrix (s, n))
        s: Number of blocks
        n: Block size

    Returns:
        Result of (A ⊗ I_n) @ x
    """
    x_reshaped = x.reshape(s, n)
    result = np.zeros((s, n))

    for i in range(s):
        for j in range(s):
            result[i] += A[i, j] * x_reshaped[j]

    return result.ravel()


def block_solve(A: NDArray, b: NDArray, s: int, n: int) -> NDArray:
    """
    Solve (A ⊗ I_n) x = b efficiently.

    This is useful when A is small and structured (e.g., triangular).

    Args:
        A: Block structure matrix (s, s)
        b: Right-hand side of length s*n
        s: Number of blocks
        n: Block size

    Returns:
        Solution x of length s*n
    """
    b_reshaped = b.reshape(s, n)
    x = np.zeros((s, n))

    # If A is lower triangular, use forward substitution
    if np.allclose(A, np.tril(A)):
        for i in range(s):
            rhs = b_reshaped[i].copy()
            for j in range(i):
                rhs -= A[i, j] * x[j]
            if not np.isclose(A[i, i], 0):
                x[i] = rhs / A[i, i]
            else:
                x[i] = rhs

    # If A is upper triangular, use backward substitution
    elif np.allclose(A, np.triu(A)):
        for i in range(s - 1, -1, -1):
            rhs = b_reshaped[i].copy()
            for j in range(i + 1, s):
                rhs -= A[i, j] * x[j]
            if not np.isclose(A[i, i], 0):
                x[i] = rhs / A[i, i]
            else:
                x[i] = rhs

    # General case: solve full system
    else:
        A_full = kronecker_eye(A, n)
        x = np.linalg.solve(A_full, b)
        return x

    return x.ravel()
