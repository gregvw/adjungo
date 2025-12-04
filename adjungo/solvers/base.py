"""Base stage solver interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from numpy.typing import NDArray


@dataclass
class StepCache:
    """Cached data from forward solve, reused in adjoint/sensitivity."""

    Z: NDArray                          # (s, n) stage values
    F: list[NDArray]                    # s Jacobians, each (n, n)
    G: list[NDArray]                    # s control Jacobians, each (n, ν)
    factorization: Optional[Any] = None # LU of stage matrix (reusable)
    stage_matrix: Optional[NDArray] = None


class StageSolver(ABC):
    """Solves the stage equations for one time step."""

    @abstractmethod
    def solve_stages(
        self,
        y_history: NDArray,      # (r, n) external stages
        u_stages: NDArray,       # (s, ν) controls at stages
        t_n: float,
        h: float,
        problem: "Problem",
        method: "GLMethod",
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
    ) -> NDArray:
        """
        Solve adjoint stage equations: A^T μ = B^T λ.

        Args:
            lambda_ext: External stage adjoints (r, n)
            cache: Cached data from forward solve
            method: GLM tableau

        Returns:
            μ: Stage adjoints (s, n)
        """
        ...
