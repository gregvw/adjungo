"""Hessian-vector product assembly."""

import numpy as np
from numpy.typing import NDArray

from adjungo.core.affine import affine_dynamics_verified
from adjungo.core.method import GLMethod
from adjungo.core.objective import Objective
from adjungo.core.problem import Problem, ProblemStructure
from adjungo.stepping.adjoint import AdjointTrajectory
from adjungo.stepping.sensitivity import (
    AdjointSensitivityTrajectory,
    SensitivityTrajectory,
)
from adjungo.stepping.trajectory import Trajectory


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
    t0: float = 0.0,
    structure: "ProblemStructure | None" = None,
) -> NDArray:
    """
    Assemble the Hessian-vector product by differentiating the gradient.

    The reduced gradient (see
    :func:`adjungo.optimization.gradient.assemble_gradient`) is

        g_k^n = ∂J/∂u_k^n + h (G_k^n)^T Λ_k^n

    Differentiating in the direction ``δu`` gives every term below, and every
    one of them carries a **plus** sign:

        [∇²Ĵ δu]_k^n = J_uu δu_k^n
                     + h (F_yu[Λ_k^n])^T δZ_k^n
                     + h  F_uu[Λ_k^n]   δu_k^n
                     + h (G_k^n)^T δΛ_k^n

    The first three come from ``h (δG_k)^T Λ_k``, since ``G = ∂f/∂u`` and

        [(δG_k)^T Λ_k]_b = Σ_ℓ Λ_ℓ ( Σ_a ∂²f_ℓ/∂u_b∂y_a δZ_a
                                    + Σ_c ∂²f_ℓ/∂u_b∂u_c δu_c ).

    Two consequences are load-bearing (NUMERICS.md C-14.1):

    * The sign is ``+h`` throughout, matching ``assemble_gradient``. The
      opposite sign would make ``⟨v, H v⟩`` change sign on the dominant term
      while leaving the operator symmetric, so symmetry cannot detect it.
    * ``F_yu_action`` and ``F_uu_action`` must be contracted against the
      weighted adjoint ``Λ_k``, not against ``δZ`` or ``δu``. Their ``v``
      argument sums over the equation index ℓ, which has length ``n``; for
      ``n ≠ ν`` contracting ``F_uu_action`` against ``δu`` is not even
      shape-conformable.

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
        t0: Initial time

    Returns:
        Hessian-vector product (N, s, ν)

    Raises:
        NotImplementedError: If the problem lacks ``F_yu_action`` or
            ``F_uu_action``. Skipping the curvature terms would return a
            Gauss-Newton-like approximation while claiming an exact Hessian.
    """
    # Both conditions are required, and the second is the load-bearing one.
    #
    # ``structure`` is a public constructor argument of GLMOptimizer, so
    # ``jointly_affine=True`` can be *declared* by any caller for any problem.
    # Unlike the constant-Jacobian declaration of M6, that claim has no cheap
    # exact check: the residual test inside the affine stage solve establishes
    # that ``f`` is affine in the *state*, and says nothing about curvature in
    # the control. A problem with ``f = M y + b(u)`` would pass every stage
    # check and still have a nonzero ``f_uu``, so believing the declaration
    # alone would return a Gauss-Newton approximation from a function that
    # promises an exact Hessian. That is the shape of precedent R-9, and it is
    # what clause C-16.5 forbids.
    #
    # ``affine_dynamics_verified`` is not a declaration. It checks that the
    # problem is an AffineDynamics whose guaranteeing methods are still the
    # ones that class defines, by object identity. A declaration that fails it
    # simply does not open the skip: the terms are computed, the callbacks are
    # required, and the caller gets more work rather than a wrong answer.
    skip_dynamics_curvature = (
        structure is not None
        and structure.jointly_affine
        and affine_dynamics_verified(problem)
    )
    if not skip_dynamics_curvature:
        for name in ("F_yu_action", "F_uu_action"):
            if getattr(problem, name, None) is None:
                raise NotImplementedError(
                    f"exact Hessian-vector products require problem.{name}(); "
                    f"{type(problem).__name__} does not provide it"
                )

    N, s, _nu = u.shape
    hvp = np.zeros_like(u)

    for step in range(N):
        cache = trajectory.caches[step]
        Lambda_k = adjoint.WeightedAdj[step]  # Weighted adjoints
        t_n = t0 + step * h

        for k in range(s):
            z_k, u_k = trajectory.Z[step, k], u[step, k]
            t_k = t_n + method.c[k] * h

            # J_uu δu (from objective)
            hvp[step, k] = (
                objective.d2J_du2(u_k, step, k) @ delta_u[step, k]
            )

            # +h G_k^T δΛ_k (adjoint sensitivity contribution)
            delta_Lambda_k = adj_sensitivity.delta_WeightedAdj[step, k]
            hvp[step, k] += h * cache.G[k].T @ delta_Lambda_k

            if skip_dynamics_curvature:
                continue

            # +h (F_yu[Λ_k])^T δZ_k
            F_yu_Lambda = np.asarray(
                problem.F_yu_action(z_k, u_k, t_k, Lambda_k[k])
            )
            hvp[step, k] += (
                h * F_yu_Lambda.T @ sensitivity.delta_Z[step, k]
            )

            # +h F_uu[Λ_k] δu_k
            F_uu_Lambda = np.asarray(
                problem.F_uu_action(z_k, u_k, t_k, Lambda_k[k])
            )
            hvp[step, k] += h * F_uu_Lambda @ delta_u[step, k]

    return hvp
