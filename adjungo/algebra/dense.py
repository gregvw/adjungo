"""Dense linear algebra backend using NumPy/SciPy."""

from typing import cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray


class DenseBackend:
    """NumPy/SciPy implementation of linear algebra operations."""

    def solve(self, A: NDArray, b: NDArray) -> NDArray:
        """Solve linear system Ax = b using direct solve."""
        return np.linalg.solve(A, b)

    def lu_factor(self, A: NDArray) -> tuple[NDArray, NDArray]:
        """
        Compute LU factorization using scipy.

        Returns:
            (lu, piv) tuple from scipy.linalg.lu_factor
        """
        return cast("tuple[NDArray, NDArray]", scipy.linalg.lu_factor(A))

    def lu_solve(
        self,
        factorization: tuple[NDArray, NDArray],
        b: NDArray,
        trans: int = 0,
    ) -> NDArray:
        """
        Solve using precomputed LU factorization.

        Args:
            factorization: (lu, piv) from lu_factor
            b: Right-hand side
            trans: 0 for Ax=b, 1 for A^T x=b

        Returns:
            Solution x
        """
        return cast("NDArray", scipy.linalg.lu_solve(factorization, b, trans=trans))

    def norm(self, x: NDArray) -> float:
        """Compute L2 norm."""
        return float(np.linalg.norm(x))
