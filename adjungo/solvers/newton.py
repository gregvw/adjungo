"""Newton solver mixin for nonlinear stage equations."""

from typing import Callable, Any
import numpy as np
import scipy.linalg
from numpy.typing import NDArray


class NewtonMixin:
    """Mixin providing Newton iteration for nonlinear stage equations."""

    def newton_solve(
        self,
        residual_fn: Callable[[NDArray], NDArray],
        jacobian_fn: Callable[[NDArray], NDArray],
        z0: NDArray,
        tol: float = 1e-10,
        max_iter: int = 10,
    ) -> tuple[NDArray, Any]:
        """
        Newton's method for solving nonlinear system.

        Args:
            residual_fn: Function computing residual r(z)
            jacobian_fn: Function computing Jacobian J(z)
            z0: Initial guess
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            z: Solution
            lu: Final Jacobian factorization (for reuse in adjoint)
        """
        z = z0.copy()
        lu = None

        for iteration in range(max_iter):
            r = residual_fn(z)
            if np.linalg.norm(r) < tol:
                break

            J = jacobian_fn(z)
            lu = scipy.linalg.lu_factor(J)
            dz = scipy.linalg.lu_solve(lu, -r)
            z += dz

        # Return final factorization for reuse
        if lu is None:
            J = jacobian_fn(z)
            lu = scipy.linalg.lu_factor(J)

        return z, lu
