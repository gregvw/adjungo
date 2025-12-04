"""Gradient assembly."""

import numpy as np
from numpy.typing import NDArray

from adjungo.stepping.trajectory import Trajectory
from adjungo.stepping.adjoint import AdjointTrajectory
from adjungo.core.objective import Objective
from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem


def assemble_gradient(
    trajectory: Trajectory,
    adjoint: AdjointTrajectory,
    u: NDArray,
    objective: Objective,
    method: GLMethod,
    problem: Problem,
    h: float,
) -> NDArray:
    """
    From glm_opt.tex equation for ∇_{u_k^n} Ĵ:

    ∇_{u_k^n} Ĵ = ∂J/∂u_k^n - h (G_k^n)^T Λ_k^n

    where Λ_k^n = Σ_i a_{ik} μ_i^n + Σ_i b_{ik} λ_i^n is the weighted adjoint.

    Args:
        trajectory: Forward solution trajectory
        adjoint: Adjoint trajectory
        u: Control array (N, s, ν)
        objective: Objective function
        method: GLM tableau
        problem: Problem specification
        h: Step size

    Returns:
        Gradient array (N, s, ν)
    """
    N, s, nu = u.shape
    grad = np.zeros_like(u)

    for step in range(N):
        cache = trajectory.caches[step]

        for k in range(s):
            # ∂J/∂u contribution (if objective depends on u directly)
            grad[step, k] = objective.dJ_du(u[step, k], step, k)

            # Constraint contribution: -h G_k^T Λ_k
            G_k = cache.G[k]  # (n, ν)
            Lambda_k = adjoint.WeightedAdj[step, k]  # (n,)

            grad[step, k] -= h * G_k.T @ Lambda_k

    return grad
