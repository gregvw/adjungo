"""Forward state propagation."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from adjungo.core.affine import require_immutable_coefficients
from adjungo.core.plan import DiscretizationPlan
from adjungo.core.problem import Problem
from adjungo.solvers.base import StageSolver, step_solvers
from adjungo.stepping.trajectory import Trajectory, packed_like


def forward_solve(
    y0: NDArray,
    u: NDArray | Callable,  # packed (sum_n s_n, nu) or callable
    plan: DiscretizationPlan,
    problem: Problem,
    stage_solver: StageSolver | Sequence[StageSolver],
) -> Trajectory:
    """
    Algorithm 1: Forward state solve from glm_opt.tex Section 7.

    For n = 0, ..., N-1, with step n spanning ``[t_n, t_{n+1}]`` and carrying
    ``h_n`` and its own method (NUMERICS.md C-18.4):

        1. Solve the stage equations for the internal stages Z^n
        2. Update y^[n+1] = V^(n) y^[n] + h_n (B^(n) f_stages)
        3. Store F_k^n, G_k^n for all stages

    Args:
        y0: Initial state
        u: Packed stage controls ``(sum_n s_n, nu)``, or a callable
            ``u(t, step, stage)``
        plan: The discretization plan. Held fixed for this solve; C-2's
            exactness claim is stated relative to it.
        problem: Problem specification
        stage_solver: Stage equation solver, or one per step

    Returns:
        Trajectory containing Y, packed Z, and cached data
    """
    # The boundary the C-15.7 invariant is actually about. Every step below
    # stores what problem.F and problem.G returned in the step cache, and
    # AffineDynamics returns its buffers by identity, so from here on the tape
    # aliases them. forward_solve is exported, and composing it with
    # adjoint_solve and assemble_gradient bypasses GLMOptimizer entirely --
    # which returned the full 0.0770625 displacement in silence. Checking at
    # the point of retention covers every composition rather than every
    # caller.
    require_immutable_coefficients(problem)

    N = plan.N
    n = problem.state_dim
    r = plan.method_at(0).r

    Y = np.zeros((N + 1, r, n))
    Y[0] = _initialize_external_stages(y0, r, n)

    Z = packed_like(plan, n)
    solvers = step_solvers(stage_solver, N)
    caches = []

    for step in range(N):
        method = plan.method_at(step)
        t_n = plan.t(step)
        h = plan.step_size(step)
        Z_step = plan.stages(Z, step)
        u_stages = _get_stage_controls(u, step, plan)

        stages, cache = solvers[step].solve_stages(
            Y[step], u_stages, t_n, h, problem, method, step=step
        )
        Z_step[...] = stages
        caches.append(cache)

        # Propagate external stages: y^[n+1] = V y^[n] + h_n (B f_stages)
        stage_times = plan.stage_times(step)
        f_stages = np.array(
            [
                problem.f(Z_step[k], u_stages[k], stage_times[k])
                for k in range(method.s)
            ]
        )
        Y[step + 1] = method.V @ Y[step] + h * (method.B @ f_stages)

    return Trajectory(Y=Y, Z=Z, caches=caches, plan=plan)


def _initialize_external_stages(y0: NDArray, r: int, n: int) -> NDArray:
    """Initialize external stages from initial condition."""
    y_ext = np.zeros((r, n))
    y_ext[0] = y0
    # For r > 1, would need a starting procedure (C-18.3, C-Q4)
    return y_ext


def _get_stage_controls(
    u: NDArray | Callable,
    step: int,
    plan: DiscretizationPlan,
) -> NDArray:
    """Extract or compute control values at this step's stage times."""
    if callable(u):
        stage_times = plan.stage_times(step)
        return np.array(
            [
                u(stage_times[k], step, k)
                for k in range(plan.s(step))
            ]
        )
    return plan.stages(plan.pack(u, "control"), step)
