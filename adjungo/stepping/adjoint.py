"""Backward adjoint propagation."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from adjungo.stepping.trajectory import Trajectory
from adjungo.solvers.base import StageSolver
from adjungo.core.objective import Objective
from adjungo.core.method import GLMethod


@dataclass
class AdjointTrajectory:
    """Adjoint variables for all steps."""

    Lambda: NDArray     # (N+1, r, n) external stage adjoints
    Mu: NDArray         # (N, s, n) internal stage adjoints
    WeightedAdj: NDArray  # (N, s, n) Λ_k^n = Σ_j a_{jk} μ_j + Σ_j b_{jk} λ_j


def adjoint_solve(
    trajectory: Trajectory,
    objective: Objective,
    method: GLMethod,
    stage_solver: StageSolver,
    h: float,
) -> AdjointTrajectory:
    """
    Algorithm 2: Backward adjoint solve from glm_opt.tex Section 7.

    For n = N, ..., 1:
        1. Solve A_n^T μ^n = B_n^T λ^n for stage adjoints
        2. Update λ^{n-1} = U^T μ^n + V^T λ^n + ∂J/∂y^[n-1]
        3. Compute Λ_k^n for all k

    Args:
        trajectory: Forward solution trajectory
        objective: Objective function
        method: GLM tableau
        stage_solver: Stage equation solver
        h: Step size

    Returns:
        Adjoint trajectory with Lambda, Mu, and weighted adjoints
    """
    N = trajectory.N
    s, r, n = method.s, method.r, trajectory.n

    Lambda = np.zeros((N + 1, r, n))
    Mu = np.zeros((N, s, n))
    WeightedAdj = np.zeros((N, s, n))

    # Terminal condition
    Lambda[N] = objective.dJ_dy_terminal(trajectory.Y[N])

    for step in range(N - 1, -1, -1):
        cache = trajectory.caches[step]

        # Solve A^T μ = B^T λ (reuses factorization from forward!)
        Mu[step] = stage_solver.solve_adjoint_stages(
            Lambda[step + 1], cache, method, h
        )

        # Compute weighted adjoint: Λ_k = Σ_j a_{jk} μ_j + Σ_j b_{jk} λ_j
        for k in range(s):
            WeightedAdj[step, k] = (
                method.A[:, k] @ Mu[step] +  # Σ_j a_{jk} μ_j
                method.B[:, k] @ Lambda[step + 1]  # Σ_j b_{jk} λ_j
            )

        # Update: λ^{n-1} = U^T μ^n + V^T λ^n + ∂J/∂y^[n-1]
        Lambda[step] = (
            method.U.T @ Mu[step] +
            method.V.T @ Lambda[step + 1] +
            objective.dJ_dy(trajectory.Y[step], step)
        )

    return AdjointTrajectory(Lambda=Lambda, Mu=Mu, WeightedAdj=WeightedAdj)
