"""Hessian-vector product assembly."""

import numpy as np
from numpy.typing import NDArray

from adjungo.stepping.trajectory import Trajectory
from adjungo.stepping.adjoint import AdjointTrajectory
from adjungo.stepping.sensitivity import (
    SensitivityTrajectory,
    AdjointSensitivityTrajectory,
)
from adjungo.core.objective import Objective
from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem


def assemble_hessian_vector_product(
    trajectory: Trajectory,
    adjoint: AdjointTrajectory,
    sensitivity: SensitivityTrajectory,
    adj_sensitivity: AdjointSensitivityTrajectory,
    u: NDArray,
    delta_u: NDArray,
    objective: Objective,
    method: GLMethod,
    problem: Problem,
    h: float,
) -> NDArray:
    """
    From glm_opt.tex:

    [∇²Ĵ]δu = J_{uu}δu + H_{uΛ}δΛ + H_{uZ}δZ + H_{uu}^{constr}δu

    At stage (n, k):
        (H_{uΛ}δΛ)_k^n = -h (G_k^n)^T δΛ_k^n
        (H_{uZ}δZ)_k^n = -h F_{yu}^{n,k}[Λ_k^n]^T δZ_k^n
        (H_{uu}^{constr}δu)_k^n = -h F_{uu}^{n,k}[Λ_k^n] δu_k^n

    Args:
        trajectory: Forward solution trajectory
        adjoint: Adjoint trajectory
        sensitivity: State sensitivity trajectory
        adj_sensitivity: Adjoint sensitivity trajectory
        u: Control array (N, s, ν)
        delta_u: Control perturbation (N, s, ν)
        objective: Objective function
        method: GLM tableau
        problem: Problem specification
        h: Step size

    Returns:
        Hessian-vector product (N, s, ν)
    """
    N, s, nu = u.shape
    hvp = np.zeros_like(u)

    for step in range(N):
        cache = trajectory.caches[step]

        for k in range(s):
            # J_{uu} δu (from objective)
            hvp[step, k] = objective.d2J_du2(u[step, k], step, k) @ delta_u[step, k]

            # -h G_k^T δΛ_k
            delta_Lambda_k = adj_sensitivity.delta_WeightedAdj[step, k]
            hvp[step, k] -= h * cache.G[k].T @ delta_Lambda_k

            # Second-order terms (if available)
            # -h F_{yu}[Λ_k]^T δZ_k
            # -h F_{uu}[Λ_k] δu_k
            # (Placeholder - requires second derivatives)

    return hvp
