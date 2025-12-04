"""Linear algebra backend protocol."""

from typing import Protocol, Any, Tuple
from numpy.typing import NDArray


class LinearAlgebraBackend(Protocol):
    """
    Protocol for linear algebra operations.
    Allows swapping between dense, sparse, matrix-free implementations.
    """

    def solve(self, A: NDArray, b: NDArray) -> NDArray:
        """
        Solve linear system Ax = b.

        Args:
            A: System matrix
            b: Right-hand side

        Returns:
            Solution x
        """
        ...

    def lu_factor(self, A: NDArray) -> Any:
        """
        Compute LU factorization of A.

        Args:
            A: Matrix to factor

        Returns:
            Factorization object (implementation-specific)
        """
        ...

    def lu_solve(
        self, factorization: Any, b: NDArray, trans: int = 0
    ) -> NDArray:
        """
        Solve using precomputed LU factorization.

        Args:
            factorization: Precomputed factorization
            b: Right-hand side
            trans: 0 for Ax=b, 1 for A^T x=b

        Returns:
            Solution x
        """
        ...

    def norm(self, x: NDArray) -> float:
        """
        Compute vector norm.

        Args:
            x: Vector

        Returns:
            Norm value
        """
        ...
