"""Matrix-free operator wrappers."""

from typing import Callable
import numpy as np
from numpy.typing import NDArray


class LinearOperator:
    """
    Matrix-free linear operator wrapper.
    Useful for large-scale problems with structured matrices.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        matvec: Callable[[NDArray], NDArray],
        rmatvec: Callable[[NDArray], NDArray] | None = None,
    ):
        """
        Initialize linear operator.

        Args:
            shape: (m, n) dimensions
            matvec: Function computing A @ x
            rmatvec: Function computing A.T @ x (optional)
        """
        self.shape = shape
        self._matvec = matvec
        self._rmatvec = rmatvec

    def matvec(self, x: NDArray) -> NDArray:
        """Compute A @ x."""
        return self._matvec(x)

    def rmatvec(self, x: NDArray) -> NDArray:
        """Compute A.T @ x."""
        if self._rmatvec is None:
            raise NotImplementedError("Transpose operation not provided")
        return self._rmatvec(x)

    def __matmul__(self, x: NDArray) -> NDArray:
        """Support A @ x syntax."""
        return self.matvec(x)
