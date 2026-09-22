"""Linear algebra backend abstractions."""

from adjungo.algebra.dense import DenseBackend
from adjungo.algebra.protocols import LinearAlgebraBackend

__all__ = [
    "DenseBackend",
    "LinearAlgebraBackend",
]
