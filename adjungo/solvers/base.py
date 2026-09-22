"""Base stage solver interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

if TYPE_CHECKING:
    from adjungo.core.method import GLMethod
    from adjungo.core.problem import Problem


@dataclass
class StepCache:
    """Cached data from forward solve, reused in adjoint/sensitivity.

    ``stage_factorizations[i]`` holds the LU factorization of the stage
    matrix ``I - h A[i,i] F_i`` evaluated at the **converged** stage value
    ``Z[i]``, or ``None`` when stage ``i`` is explicit (``A[i,i] == 0``). The
    adjoint and the tangent both reuse it, the adjoint via
    ``lu_solve(..., trans=1)``.

    Storing one factorization per stage rather than a single object per step
    keeps the operator each stage used attributable. Entries are **never**
    shared between stages: precedent R-9 records a defect in which one
    stage's factorization was published for every stage of the step, giving
    the adjoint the transpose of the wrong matrix.

    ``coupled_factorization`` is the exception that proves the rule, and it
    is a different object with a different meaning. A tableau with a dense
    ``A`` cannot be solved stage by stage at all: all ``s`` stages form one
    ``(s·n) × (s·n)`` system whose Jacobian has blocks
    ``δ_ij I - h A[i,j] F_j``. There is genuinely one matrix for the step,
    and its transpose is exactly the operator of the adjoint stage system,
    so the single factorization is reused by both. It is set only by
    :class:`~adjungo.solvers.implicit.ImplicitStageSolver`, and
    ``stage_factorizations`` is left ``None`` in that case: no per-stage
    matrix exists to name.
    """

    Z: NDArray                          # (s, n) stage values
    F: list[NDArray]                    # s Jacobians, each (n, n)
    G: list[NDArray]                    # s control Jacobians, each (n, ν)
    stage_factorizations: list[Any] | None = None
    stage_matrix: NDArray | None = None
    coupled_factorization: Any = None


class StageSolver(ABC):
    """Solves the stage equations for one time step.

    ``y_scale`` is the characteristic state magnitude used by the C-5.1
    stage-convergence test. It is **captured at construction** and never
    re-read from mutable state, so the threshold a stage is certified against
    is the same one the route was configured with.
    """

    y_scale: float = 1.0

    @abstractmethod
    def solve_stages(
        self,
        y_history: NDArray,      # (r, n) external stages
        u_stages: NDArray,       # (s, ν) controls at stages
        t_n: float,
        h: float,
        problem: "Problem",
        method: "GLMethod",
        step: int | None = None,
    ) -> tuple[NDArray, StepCache]:
        """
        Solve stage equations for one time step.

        Args:
            y_history: External stages from previous step (r, n)
            u_stages: Control values at each stage (s, ν)
            t_n: Time at start of step
            h: Step size
            problem: Problem specification
            method: GLM tableau
            step: Index of this time step, used only to locate a stage
                failure in the message required by NUMERICS.md C-5.3

        Returns:
            Z: Internal stage values (s, n)
            cache: Stored data for adjoint/sensitivity
        """
        ...

    @abstractmethod
    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,  # (r, n) external adjoints
        cache: StepCache,
        method: "GLMethod",
        h: float,
    ) -> NDArray:
        """
        Solve adjoint stage equations: A^T μ = B^T λ.

        Args:
            lambda_ext: External stage adjoints (r, n)
            cache: Cached data from forward solve
            method: GLM tableau
            h: Step size

        Returns:
            μ: Stage adjoints (s, n)
        """
        ...
