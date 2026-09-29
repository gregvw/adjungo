"""Gradient assembly."""

import numpy as np
from numpy.typing import NDArray

from adjungo.core.objective import Objective
from adjungo.core.problem import Problem
from adjungo.stepping.adjoint import AdjointTrajectory
from adjungo.stepping.trajectory import Trajectory


def assemble_gradient(
    trajectory: Trajectory,
    adjoint: AdjointTrajectory,
    u: NDArray,
    objective: Objective,
    problem: Problem,
) -> NDArray:
    """
    From glm_opt.tex equation for grad_{u_k^n} J:

    grad_{u_k^n} J = dJ/du_k^n - h_n (G_k^n)^T Lambda_k^n

    where Lambda_k^n = sum_i a_{ik} mu_i^n + sum_i b_{ik} lambda_i^n is the
    weighted adjoint.

    ``h_n`` is the step's own size, read from the trajectory's plan by step
    index (C-18.4). The plan is never re-selected here; see C-18.2.

    Args:
        trajectory: Forward solution trajectory, carrying its plan
        adjoint: Adjoint trajectory
        u: Stage controls, packed ``(sum_n s_n, nu)`` or, when the plan has
            one stage count throughout, rectangular ``(N, s, nu)``
        objective: Objective function
        problem: Problem specification

    Returns:
        The gradient, in the same shape ``u`` was supplied in. A gradient is
        a covector on the control space, so returning it in a different
        layout than the point it was taken at would make ``u - alpha * g``
        silently broadcast instead of step.
    """
    plan = trajectory.plan
    caller_shape = np.shape(u)
    u = plan.pack(u, "control")
    grad = np.zeros_like(u)

    for step in range(plan.N):
        cache = trajectory.caches[step]
        h = plan.step_size(step)
        u_step = plan.stages(u, step)
        grad_step = plan.stages(grad, step)
        W_step = plan.stages(adjoint.WeightedAdj, step)

        for k in range(plan.s(step)):
            # dJ/du contribution (if objective depends on u directly)
            grad_step[k] = objective.dJ_du(u_step[k], step, k)

            # Constraint contribution: +h_n G_k^T Lambda_k
            G_k = cache.G[k]  # (n, nu)
            grad_step[k] += h * G_k.T @ W_step[k]

    return grad.reshape(caller_shape)
