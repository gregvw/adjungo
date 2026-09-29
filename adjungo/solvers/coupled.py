"""Applying a coupled stage factorization, in both orientations.

A tableau with a dense ``A`` has one ``(s·n) × (s·n)`` operator per step
rather than one per stage. Three sweeps solve with that single operator -- the
tangent forward, the first-order adjoint and the second-order adjoint -- and
two of them want its transpose.

The pair lives in its own module, depending on nothing in this package, so
that :mod:`adjungo.solvers.base` can use it without importing the concrete
solver that builds the factorization. One implementation of each orientation
means the sweeps cannot drift apart in how they read it; that orientation is
checked directly, as an inner-product identity, in
``tests/test_fully_implicit.py``.
"""

from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

__all__ = ["solve_coupled", "solve_coupled_transposed"]


def solve_coupled(factorization: Any, rhs: NDArray) -> NDArray:
    """Apply the coupled stage operator's inverse to a per-stage right side.

    Used by the forward (tangent) sensitivity, which solves the *same*
    linear system the forward Newton iteration converged on, with a
    different right-hand side.

    Args:
        factorization: ``lu_factor`` result for the forward coupled Jacobian.
        rhs: Right-hand side with shape ``(s, n)``.

    Returns:
        Solution with shape ``(s, n)``.
    """
    s, n = rhs.shape
    solution = scipy.linalg.lu_solve(factorization, rhs.ravel())
    return np.asarray(solution, dtype=float).reshape(s, n)


def solve_coupled_transposed(factorization: Any, rhs: NDArray) -> NDArray:
    """Apply the transposed coupled stage operator to a per-stage right side.

    Shared by the first-order adjoint and the second-order (sensitivity)
    adjoint, which solve the *same* linear system with different right-hand
    sides. One implementation means the two cannot drift apart in how they
    interpret the factorization's orientation.

    Args:
        factorization: ``lu_factor`` result for the forward coupled Jacobian.
        rhs: Right-hand side with shape ``(s, n)``.

    Returns:
        Solution with shape ``(s, n)``.
    """
    s, n = rhs.shape
    solution = scipy.linalg.lu_solve(factorization, rhs.ravel(), trans=1)
    return np.asarray(solution, dtype=float).reshape(s, n)
