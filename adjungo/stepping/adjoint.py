"""Backward adjoint propagation."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from adjungo.core.objective import Objective
from adjungo.solvers.base import StageSolver, step_solvers
from adjungo.stepping.trajectory import Trajectory, packed_like


@dataclass
class AdjointTrajectory:
    """Adjoint variables for all steps.

    ``Mu`` and ``WeightedAdj`` are packed as ``(sum_n s_n, n)`` with the
    trajectory plan's offsets (C-18.5).
    """

    Lambda: NDArray       # (N+1, r, n) external stage adjoints
    Mu: NDArray           # (sum_n s_n, n) internal stage adjoints
    WeightedAdj: NDArray  # (sum_n s_n, n) Lambda_k^n


def adjoint_solve(
    trajectory: Trajectory,
    objective: Objective,
    stage_solver: StageSolver | Sequence[StageSolver],
) -> AdjointTrajectory:
    """
    Algorithm 2: Backward adjoint solve from glm_opt.tex Section 7.

    For n = N-1, ..., 0:
        1. Solve A_n^T mu^n = B_n^T lambda^n for stage adjoints
        2. Update lambda^n = U^T mu^n + V^T lambda^{n+1} + dJ/dy^[n]
        3. Compute Lambda_k^n for all k

    The plan is read from ``trajectory``, by step index, and is never
    recomputed or re-selected: per C-18.2 this sweep must differentiate the
    forward objective that was actually executed.

    Args:
        trajectory: Forward solution trajectory, carrying its plan
        objective: Objective function
        stage_solver: Stage equation solver, or one per step

    Returns:
        Adjoint trajectory with Lambda, packed Mu, and packed weighted adjoints
    """
    plan = trajectory.plan
    N = trajectory.N
    n = trajectory.n
    r = trajectory.r

    Lambda = np.zeros((N + 1, r, n))
    Mu = packed_like(plan, n)
    WeightedAdj = packed_like(plan, n)
    solvers = step_solvers(stage_solver, N)

    # Terminal condition
    Lambda[N] = objective.dJ_dy_terminal(trajectory.Y[N])

    for step in range(N - 1, -1, -1):
        method = plan.method_at(step)
        h = plan.step_size(step)
        cache = trajectory.caches[step]
        Mu_step = plan.stages(Mu, step)
        W_step = plan.stages(WeightedAdj, step)

        # Solve A^T mu = B^T lambda (reuses factorization from forward!)
        Mu_step[...] = solvers[step].solve_adjoint_stages(
            Lambda[step + 1], cache, method, h
        )

        # Weighted adjoint: Lambda_k = sum_j a_{jk} mu_j + sum_j b_{jk} lambda_j
        for k in range(method.s):
            W_step[k] = (
                method.A[:, k] @ Mu_step +          # sum_j a_{jk} mu_j
                method.B[:, k] @ Lambda[step + 1]   # sum_j b_{jk} lambda_j
            )

        # Update: lambda^n = U^T mu^n + V^T lambda^{n+1} + dJ/dy^[n]
        Lambda[step] = (
            method.U.T @ Mu_step +
            method.V.T @ Lambda[step + 1] +
            objective.dJ_dy(trajectory.Y[step], step)
        )

    return AdjointTrajectory(Lambda=Lambda, Mu=Mu, WeightedAdj=WeightedAdj)
