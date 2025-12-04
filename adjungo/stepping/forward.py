"""Forward state propagation."""

from typing import Callable, Union
import numpy as np
from numpy.typing import NDArray

from adjungo.stepping.trajectory import Trajectory
from adjungo.solvers.base import StageSolver
from adjungo.core.problem import Problem
from adjungo.core.method import GLMethod


def forward_solve(
    y0: NDArray,
    u: Union[NDArray, Callable],  # (N, s, ν) or callable
    t_span: tuple[float, float],
    N: int,
    problem: Problem,
    method: GLMethod,
    stage_solver: StageSolver,
) -> Trajectory:
    """
    Algorithm 1: Forward state solve from glm_opt.tex Section 7.

    For n = 1, ..., N:
        1. Solve A_n Z^n = U y^[n-1] for internal stages
        2. Update y^[n] = V y^[n-1] + B_n Z^n
        3. Store F_k^n, G_k^n for all stages

    Args:
        y0: Initial state
        u: Control array (N, s, ν) or callable u(t, step, stage)
        t_span: Time interval (t0, tf)
        N: Number of time steps
        problem: Problem specification
        method: GLM tableau
        stage_solver: Stage equation solver

    Returns:
        Trajectory containing Y, Z, and cached data
    """
    h = (t_span[1] - t_span[0]) / N
    s, r, n = method.s, method.r, problem.state_dim

    # Initialize external stages
    Y = np.zeros((N + 1, r, n))
    Y[0] = _initialize_external_stages(y0, method, r, n)

    Z = np.zeros((N, s, n))
    caches = []

    for step in range(N):
        t_n = t_span[0] + step * h
        u_stages = _get_stage_controls(u, step, method, h, t_n)

        # Solve stage equations
        Z[step], cache = stage_solver.solve_stages(
            Y[step], u_stages, t_n, h, problem, method
        )
        caches.append(cache)

        # Propagate external stages: y^[n] = V y^[n-1] + B Z^n
        f_stages = np.array(
            [
                problem.f(Z[step, k], u_stages[k], t_n + method.c[k] * h)
                for k in range(s)
            ]
        )
        Y[step + 1] = method.V @ Y[step] + h * (method.B @ f_stages)

    return Trajectory(Y=Y, Z=Z, caches=caches)


def _initialize_external_stages(
    y0: NDArray, method: GLMethod, r: int, n: int
) -> NDArray:
    """Initialize external stages from initial condition."""
    y_ext = np.zeros((r, n))
    y_ext[0] = y0
    # For r > 1, would need starting procedure
    return y_ext


def _get_stage_controls(
    u: Union[NDArray, Callable],
    step: int,
    method: GLMethod,
    h: float,
    t_n: float,
) -> NDArray:
    """Extract or compute control values at stages."""
    if callable(u):
        # Compute controls at stage times
        u_stages = np.array(
            [u(t_n + method.c[k] * h, step, k) for k in range(method.s)]
        )
    else:
        # Extract from array
        u_stages = u[step]
    return u_stages
