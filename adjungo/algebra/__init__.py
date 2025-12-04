"""Linear algebra backend abstractions."""

from adjungo.algebra.protocols import LinearAlgebraBackend
from adjungo.algebra.dense import DenseBackend

__all__ = [
    "LinearAlgebraBackend",
    "DenseBackend",
]
