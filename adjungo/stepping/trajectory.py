"""Trajectory storage for optimization."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from adjungo.core.plan import DiscretizationPlan
from adjungo.solvers.base import StepCache


def stage_view(packed: NDArray, plan: DiscretizationPlan) -> NDArray:
    """View a packed stage array as ``(N, s, ...)``.

    Only defined when every step has the same stage count. This is a
    compatibility accessor for callers that predate C-18, not a second storage
    mode: the returned array is a view of ``packed`` and shares its memory.

    Raises:
        ValueError: when the plan's steps differ in ``s``. There is no
            rectangular shape to return, and returning a padded one would be a
            silent sentinel (C-7).
    """
    s = plan.uniform_stage_count
    if s is None:
        counts = sorted({m.s for m in plan.methods})
        raise ValueError(
            f"This plan's steps have differing stage counts {counts}, so its "
            f"stage storage has no (N, s, ...) shape. Use "
            f"plan.stages(array, step) for the step's block (NUMERICS.md "
            f"C-18.5)."
        )
    return packed.reshape(plan.N, s, *packed.shape[1:])


@dataclass
class Trajectory:
    """Full trajectory storage for optimization.

    ``Z`` is packed as ``(sum_n s_n, n)`` with the plan's per-step offsets
    (C-18.5). Use ``plan.stages(traj.Z, step)`` for one step's block, or
    :attr:`Z_rect` when the plan has a single stage count throughout.
    """

    Y: NDArray          # (N+1, r, n) external stages at each step
    Z: NDArray          # (sum_n s_n, n) packed internal stages
    caches: list[StepCache]  # Per-step cached data
    plan: DiscretizationPlan

    @property
    def N(self) -> int:
        """Number of time steps."""
        return len(self.caches)

    @property
    def n(self) -> int:
        """State dimension."""
        return int(self.Y.shape[2])

    @property
    def r(self) -> int:
        """Number of external stages."""
        return int(self.Y.shape[1])

    @property
    def s(self) -> int:
        """Internal stage count, when every step shares one.

        Raises:
            ValueError: when the steps differ, since no single ``s`` exists.
        """
        s = self.plan.uniform_stage_count
        if s is None:
            raise ValueError(
                "This trajectory's plan has steps with differing stage "
                "counts, so `s` is not a property of the trajectory. Use "
                "plan.s(step) (NUMERICS.md C-18.5)."
            )
        return s

    @property
    def Z_rect(self) -> NDArray:
        """``Z`` as ``(N, s, n)``; requires one stage count throughout."""
        return stage_view(self.Z, self.plan)

    def stages(self, step: int) -> NDArray:
        """The ``(s_n, n)`` internal-stage block of ``step``, as a view."""
        return self.plan.stages(self.Z, step)

    def copy(self) -> "Trajectory":
        """Copy the stage and external storage, sharing the caches.

        The plan is shared rather than copied: it is immutable, and C-18.2
        requires every consumer of this trajectory to differentiate the
        discretization it records.
        """
        return Trajectory(
            Y=self.Y.copy(),
            Z=self.Z.copy(),
            caches=self.caches,
            plan=self.plan,
        )


def packed_like(plan: DiscretizationPlan, trailing: int) -> NDArray:
    """A zeroed packed stage array with ``trailing`` components per stage."""
    return np.zeros((plan.total_stages, trailing))
