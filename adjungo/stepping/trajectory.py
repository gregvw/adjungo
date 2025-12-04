"""Trajectory storage for optimization."""

from dataclasses import dataclass
from numpy.typing import NDArray
from adjungo.solvers.base import StepCache


@dataclass
class Trajectory:
    """Full trajectory storage for optimization."""

    Y: NDArray          # (N+1, r, n) external stages at each step
    Z: NDArray          # (N, s, n) internal stages at each step
    caches: list[StepCache]  # Per-step cached data

    @property
    def N(self) -> int:
        """Number of time steps."""
        return len(self.caches)

    @property
    def n(self) -> int:
        """State dimension."""
        return self.Y.shape[2]

    @property
    def r(self) -> int:
        """Number of external stages."""
        return self.Y.shape[1]

    @property
    def s(self) -> int:
        """Number of internal stages."""
        return self.Z.shape[1]
