"""State and adjoint sensitivity equations."""

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from adjungo.core.method import GLMethod
from adjungo.core.objective import Objective
from adjungo.core.problem import Problem
from adjungo.solvers.base import StageSolver
from adjungo.solvers.implicit import (
    solve_coupled,
    solve_coupled_transposed,
)
from adjungo.stepping.adjoint import AdjointTrajectory
from adjungo.stepping.trajectory import Trajectory


@dataclass
class SensitivityTrajectory:
    """State sensitivity trajectory."""

    delta_Y: NDArray  # (N+1, r, n) external stage sensitivities
    delta_Z: NDArray  # (N, s, n) internal stage sensitivities


@dataclass
class AdjointSensitivityTrajectory:
    """Adjoint sensitivity trajectory."""

    delta_Lambda: NDArray      # (N+1, r, n) external adjoint sensitivities
    delta_Mu: NDArray          # (N, s, n) internal adjoint sensitivities
    delta_WeightedAdj: NDArray # (N, s, n) weighted adjoint sensitivities


def forward_sensitivity(
    trajectory: Trajectory,
    delta_u: NDArray,
    method: GLMethod,
    stage_solver: StageSolver,
    problem: Problem,
    h: float,
) -> SensitivityTrajectory:
    """
    Algorithm 3: Forward state sensitivity from glm_opt.tex Section 7.

    Linearize the forward problem:
        Z^n = U y^{n-1} + h A f(Z^n, u^n)
        y^n = V y^{n-1} + h B f(Z^n, u^n)

    Taking differential (tangent plane):
        δZ^n = U δy^{n-1} + h A [F δZ^n + G δu^n]
        δy^n = V δy^{n-1} + h B [F δZ^n + G δu^n]

    Rearranging:
        (I - h A ⊗ F) δZ^n = U δy^{n-1} + h (A ⊗ G) δu^n
        δy^n = V δy^{n-1} + h B (F δZ^n + G δu^n)

    Key: Same system structure as forward solve, just different RHS!
          For explicit methods: forward substitution
          For implicit methods: reuse cached factorization

    Args:
        trajectory: Forward solution trajectory
        delta_u: Control perturbation (N, s, ν)
        method: GLM tableau
        stage_solver: Stage equation solver
        problem: Problem specification
        h: Step size

    Returns:
        State sensitivity trajectory
    """
    N = trajectory.N
    s, r, n = method.s, method.r, trajectory.n
    A = method.A

    delta_Y = np.zeros((N + 1, r, n))
    delta_Z = np.zeros((N, s, n))

    # Zero initial condition for sensitivity
    delta_Y[0] = 0

    for step in range(N):
        cache = trajectory.caches[step]

        # The tangent system is the forward stage system linearised:
        #   δZ_i = U[i] δy + h Σ_j A[i,j] ( F_j δZ_j + G_j δu_j )
        # Its operator has blocks δ_ij I - h A[i,j] F_j, which is exactly the
        # forward Newton Jacobian, so the converged factorization is reused
        # rather than rebuilt. Rebuilding would be a second opportunity to
        # evaluate F at the wrong point (C-5.4).
        if cache.coupled_factorization is not None:
            # Dense A: the sum over j runs over every stage, so there is no
            # order in which the stages become available one at a time.
            rhs_coupled = np.empty((s, n))
            for i in range(s):
                acc = np.asarray(method.U[i] @ delta_Y[step], dtype=float)
                for j in range(s):
                    if A[i, j] != 0.0:
                        acc = acc + h * A[i, j] * (
                            cache.G[j] @ delta_u[step, j]
                        )
                rhs_coupled[i] = acc
            delta_Z[step] = solve_coupled(
                cache.coupled_factorization, rhs_coupled
            )
        else:
            for i in range(s):
                # RHS: U δy^{n-1} + h Σ_{j<i} a_{ij} [F_j δZ_j + G_j δu_j]
                rhs = method.U[i] @ delta_Y[step]

                # Add coupling from previous stages
                for j in range(i):
                    rhs += h * A[i, j] * (cache.F[j] @ delta_Z[step, j] +
                                          cache.G[j] @ delta_u[step, j])

                # For explicit stages (a_{ii} = 0): δZ_i = rhs
                if np.isclose(A[i, i], 0):
                    delta_Z[step, i] = rhs
                else:
                    # Implicit stage. Differentiating
                    #   Z_i - h a_ii f(Z_i, u_i, t_i)
                    #       = U[i] y + h Σ_{j<i} A[i,j] f_j
                    # gives
                    #   (I - h a_ii F_i) δZ_i = rhs + h a_ii G_i δu_i
                    # The operator is exactly the forward stage matrix, so
                    # the factorization taken at the converged stage value is
                    # reused rather than rebuilt.
                    gamma = A[i, i]
                    rhs_implicit = rhs + h * gamma * (
                        cache.G[i] @ delta_u[step, i]
                    )
                    lu = (
                        cache.stage_factorizations[i]
                        if cache.stage_factorizations is not None
                        else None
                    )
                    if lu is None:
                        delta_Z[step, i] = np.linalg.solve(
                            np.eye(n) - h * gamma * cache.F[i], rhs_implicit
                        )
                    else:
                        delta_Z[step, i] = scipy.linalg.lu_solve(
                            lu, rhs_implicit
                        )

        # Propagate sensitivity: δy^n = V δy^{n-1} + h B Σ_i [F_i δZ_i + G_i δu_i]
        delta_Y[step + 1] = method.V @ delta_Y[step]

        for i in range(s):
            # Add contribution from each stage
            f_sens = cache.F[i] @ delta_Z[step, i] + cache.G[i] @ delta_u[step, i]
            delta_Y[step + 1] += h * method.B[:, i:i+1] @ f_sens[np.newaxis, :]

    return SensitivityTrajectory(delta_Y=delta_Y, delta_Z=delta_Z)


def adjoint_sensitivity(
    trajectory: Trajectory,
    adjoint: AdjointTrajectory,
    sensitivity: SensitivityTrajectory,
    u: NDArray,
    delta_u: NDArray,
    method: GLMethod,
    stage_solver: StageSolver,
    problem: Problem,
    h: float,
    t0: float = 0.0,
    objective: Objective | None = None,
) -> AdjointSensitivityTrajectory:
    """
    Algorithm 4: Backward adjoint sensitivity from glm_opt.tex Section 7.

    Solve:
        A_n^T δμ^n = B_n^T δλ^n + Γ^n
        δλ^{n-1} = U^T δμ^n + V^T δλ^n + J_{yy} δy^[n-1]

    where Γ^n contains second-derivative terms:
        Γ_k^n = h[F_{yy}^{n,k}[Λ_k^n] δZ_k^n + F_{yu}^{n,k}[Λ_k^n] δu_k^n]

    Derivation (NUMERICS.md C-2, C-9.4). The first-order adjoint stage
    relation is ``μ_i = h F_i^T Λ_i``. Differentiating it in the direction
    ``δu`` gives

        δμ_i = h F_i^T δΛ_i + h (δF_i)^T Λ_i

    and the second term expands, component by component, as

        [(δF_i)^T Λ_i]_b = Σ_ℓ Λ_ℓ ( Σ_a ∂²f_ℓ/∂y_b∂y_a δZ_a
                                    + Σ_c ∂²f_ℓ/∂y_b∂u_c δu_c )

    so that ``Γ_i`` contracts the problem's second-derivative callbacks
    against the **weighted adjoint** ``Λ_i`` and then applies the resulting
    matrix to ``δZ_i`` and ``δu_i``. Contracting against ``δZ`` or ``δu``
    instead sums over the wrong tensor index: ``∂²f_ℓ/∂y_a∂y_b`` is symmetric
    in ``(a, b)`` but carries no symmetry in ``ℓ``, and for ``n ≠ ν`` the
    mixed term is not even shape-conformable.

    The terminal condition is ``δλ^[N] = J_yy^terminal(y^[N]) δy^[N]``, not
    zero. Zero is correct only for an affine terminal cost.

    Key insight: This is a LINEAR problem (same structure as adjoint solve)!
    - For explicit methods: backward substitution
    - For implicit methods: reuse transposed factorization from adjoint
    - Only difference: enhanced RHS with second derivatives

    Args:
        trajectory: Forward solution trajectory
        adjoint: Adjoint trajectory
        sensitivity: State sensitivity trajectory
        u: Control array (N, s, ν)
        delta_u: Control perturbation (N, s, ν)
        method: GLMethod tableau
        stage_solver: Stage equation solver
        problem: Problem specification
        h: Step size
        t0: Initial time (default 0.0)
        objective: Objective function, supplying the terminal and running
            state Hessians. Required.

    Returns:
        Adjoint sensitivity trajectory

    Raises:
        ValueError: If ``objective`` is omitted. There is no defensible
            default: dropping the objective Hessian silently returns the
            second-order adjoint of a different problem (NUMERICS.md C-7).
        NotImplementedError: If the problem or objective lacks a required
            second-derivative callback.
    """
    if objective is None:
        raise ValueError(
            "adjoint_sensitivity requires an objective: the terminal "
            "condition δλ^[N] = J_yy^terminal δy^[N] and the running term "
            "J_yy δy^[n] both come from it. Passing None would silently "
            "compute the Hessian of a different objective."
        )

    for name in ("F_yy_action", "F_yu_action"):
        if getattr(problem, name, None) is None:
            raise NotImplementedError(
                f"second-order adjoint requires problem.{name}(); "
                f"{type(problem).__name__} does not provide it"
            )
    for name in ("d2J_dy2", "d2J_dy2_terminal"):
        if getattr(objective, name, None) is None:
            raise NotImplementedError(
                f"second-order adjoint requires objective.{name}(); "
                f"{type(objective).__name__} does not provide it"
            )

    N = trajectory.N
    s, r, n = method.s, method.r, trajectory.n
    A = method.A
    B = method.B

    delta_Lambda = np.zeros((N + 1, r, n))
    delta_Mu = np.zeros((N, s, n))
    delta_WeightedAdj = np.zeros((N, s, n))

    # Terminal condition: δλ^[N] = J_yy^terminal(y^[N]) δy^[N].
    # Zero is correct only for an affine terminal cost.
    J_yy_T = np.asarray(objective.d2J_dy2_terminal(trajectory.Y[N]))
    for row in range(r):
        delta_Lambda[N, row] = J_yy_T @ sensitivity.delta_Y[N, row]

    # Backward sweep (same direction as adjoint solve)
    for step in range(N - 1, -1, -1):
        cache = trajectory.caches[step]
        Lambda_k = adjoint.WeightedAdj[step]  # Weighted adjoints Λ_k

        # Second-derivative forcing:
        #   Γ_k = h [ F_yy[Λ_k] δZ_k + F_yu[Λ_k] δu_k ]
        # The callbacks contract their ``v`` argument over the equation index
        # ℓ, so ``v`` must be the weighted adjoint Λ_k.
        Gamma = np.zeros((s, n))
        t_n = t0 + step * h
        for k in range(s):
            t_k = t_n + method.c[k] * h
            z_k, u_k = trajectory.Z[step, k], u[step, k]

            F_yy_Lam = np.asarray(
                problem.F_yy_action(z_k, u_k, t_k, Lambda_k[k])
            )
            F_yu_Lam = np.asarray(
                problem.F_yu_action(z_k, u_k, t_k, Lambda_k[k])
            )
            Gamma[k] = h * (
                F_yy_Lam @ sensitivity.delta_Z[step, k]
                + F_yu_Lam @ delta_u[step, k]
            )

        delta_lambda_ext = delta_Lambda[step + 1]

        # Differentiating μ_i = h F_i^T Λ_i gives
        #   δμ_i = h F_i^T ( Σ_j A[j,i] δμ_j + Σ_l B[l,i] δλ_l ) + Γ_i
        # where Γ_i collects the terms from differentiating F_i itself.
        # F_i (not F_j) multiplies the entire weighted sum; see
        # ExplicitStageSolver.solve_adjoint_stages for the derivation.
        #
        # This is the same linear system the *first-order* adjoint solves,
        # with a different right-hand side, so it is solved the same way.
        if cache.coupled_factorization is not None:
            # Dense A: the sum over j runs over every stage, so no ordering
            # makes the stages available one at a time. The operator is the
            # transpose of the forward coupled Jacobian. Every A-weighted
            # coupling lives in that operator, so the right-hand side carries
            # only the external-adjoint term and Γ; adding an A term here as
            # well -- the natural slip when adapting the triangular branch
            # below -- would count the coupling twice.
            rhs_coupled = np.empty((s, cache.Z.shape[1]))
            for i in range(s):
                rhs_coupled[i] = (
                    h * cache.F[i].T @ (B[:, i] @ delta_lambda_ext)
                    + Gamma[i]
                )
            delta_Mu[step] = solve_coupled_transposed(
                cache.coupled_factorization, rhs_coupled
            )
        else:
            # A is (strictly) lower triangular, so the same system is block
            # triangular and backward substitution solves it in s small
            # blocks instead of one (s*n) solve.
            for i in range(s - 1, -1, -1):
                weighted = B[:, i] @ delta_lambda_ext
                for j in range(i + 1, s):
                    weighted = weighted + A[j, i] * delta_Mu[step, j]
                rhs = h * cache.F[i].T @ weighted + Gamma[i]

                # Implicit stages carry the same transposed stage solve as
                # the first-order adjoint: (I - h a_ii F_i^T) δμ_i = rhs.
                lu = (
                    cache.stage_factorizations[i]
                    if cache.stage_factorizations is not None
                    else None
                )
                if lu is None:
                    delta_Mu[step, i] = rhs
                else:
                    delta_Mu[step, i] = scipy.linalg.lu_solve(
                        lu, rhs, trans=1
                    )

        # Compute weighted adjoint sensitivities for Hessian assembly
        # δΛ_k = Σ_j a_{jk} δμ_j + Σ_j b_{jk} δλ_j
        for k in range(s):
            delta_WeightedAdj[step, k] = (
                A[:, k] @ delta_Mu[step] +  # Σ_j a_{jk} δμ_j
                B[:, k] @ delta_Lambda[step + 1]  # Σ_j b_{jk} δλ_j
            )

        # Propagate external stages backward:
        #   δλ^[n] = U^T δμ^n + V^T δλ^[n+1] + J_yy(y^[n], n) δy^[n]
        # The running term mirrors the ∂J/∂y^[n] term in adjoint_solve; it is
        # present for n = 0 .. N-1, while node N is the terminal condition.
        delta_Lambda[step] = method.U.T @ delta_Mu[step]
        delta_Lambda[step] += method.V.T @ delta_lambda_ext

        J_yy = np.asarray(objective.d2J_dy2(trajectory.Y[step], step))
        for row in range(r):
            delta_Lambda[step, row] += J_yy @ sensitivity.delta_Y[step, row]

    return AdjointSensitivityTrajectory(
        delta_Lambda=delta_Lambda,
        delta_Mu=delta_Mu,
        delta_WeightedAdj=delta_WeightedAdj,
    )
