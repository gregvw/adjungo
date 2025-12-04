"""General Linear Method specification."""

from dataclasses import dataclass
from functools import cached_property
from enum import Enum, auto
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class StageType(Enum):
    """Classification of stage matrix structure."""
    EXPLICIT = auto()   # A strictly lower triangular
    DIRK = auto()       # A lower triangular, varying diagonal
    SDIRK = auto()      # A lower triangular, constant diagonal γ
    IMPLICIT = auto()   # A dense


class PropType(Enum):
    """Classification of propagation matrix structure."""
    IDENTITY = auto()   # V = I (RK)
    SHIFT = auto()      # V is shift matrix (Adams/BDF)
    TRIANGULAR = auto() # V lower triangular
    DENSE = auto()      # V general


@dataclass(frozen=True)
class GLMethod:
    """General Linear Method tableau."""

    A: NDArray  # (s, s) - internal stage coefficients
    U: NDArray  # (s, r) - history → stages
    B: NDArray  # (r, s) - stages → output
    V: NDArray  # (r, r) - history → output
    c: NDArray  # (s,)   - abscissae

    @cached_property
    def s(self) -> int:
        """Number of internal stages."""
        return self.A.shape[0]

    @cached_property
    def r(self) -> int:
        """Number of external stages."""
        return self.V.shape[0]

    @cached_property
    def stage_type(self) -> StageType:
        """Classify the stage matrix structure."""
        return _classify_stage_structure(self.A)

    @cached_property
    def prop_type(self) -> PropType:
        """Classify the propagation matrix structure."""
        return _classify_prop_structure(self.V)

    @cached_property
    def sdirk_gamma(self) -> Optional[float]:
        """Return γ if SDIRK, otherwise None."""
        if self.stage_type == StageType.SDIRK:
            return float(self.A[0, 0])
        return None

    @cached_property
    def explicit_stage_indices(self) -> list[int]:
        """Stages i where a_{ii} = 0 (can be evaluated explicitly)."""
        return [i for i in range(self.s) if np.isclose(self.A[i, i], 0)]


def _classify_stage_structure(A: NDArray) -> StageType:
    """Classify stage matrix structure."""
    s = A.shape[0]

    # Check strictly lower triangular
    if np.allclose(A, np.tril(A, -1)):
        return StageType.EXPLICIT

    # Check lower triangular
    if np.allclose(A, np.tril(A)):
        diag = np.diag(A)
        nonzero_diag = diag[~np.isclose(diag, 0)]

        if len(nonzero_diag) == 0:
            return StageType.EXPLICIT

        # SDIRK: ALL diagonal entries must be equal and nonzero
        # (not just the nonzero ones being equal!)
        if len(nonzero_diag) == s and np.allclose(diag, diag[0]):
            return StageType.SDIRK
        return StageType.DIRK

    return StageType.IMPLICIT


def _classify_prop_structure(V: NDArray) -> PropType:
    """Classify propagation matrix structure."""
    r = V.shape[0]

    # Check identity
    if np.allclose(V, np.eye(r)):
        return PropType.IDENTITY

    # Check shift matrix
    if r > 1 and np.allclose(V, np.eye(r, k=1)):
        return PropType.SHIFT

    # Check lower triangular
    if np.allclose(V, np.tril(V)):
        return PropType.TRIANGULAR

    return PropType.DENSE
