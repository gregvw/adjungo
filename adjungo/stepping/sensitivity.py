"""State and adjoint sensitivity equations."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from adjungo.stepping.trajectory import Trajectory
from adjungo.stepping.adjoint import AdjointTrajectory
from adjungo.solvers.base import StageSolver
from adjungo.core.problem import Problem
from adjungo.core.method import GLMethod


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

        # Solve sensitivity system stage-by-stage
        # For explicit: forward substitution
        # For implicit: would reuse factorization

        for i in range(s):
            # Build RHS: U δy^{n-1} + h Σ_{j<i} a_{ij} [F_j δZ_j + G_j δu_j]
            rhs = method.U[i] @ delta_Y[step]

            # Add coupling from previous stages
            for j in range(i):
                # a_{ij} * [F_j^T δZ_j + G_j δu_j]
                rhs += h * A[i, j] * (cache.F[j] @ delta_Z[step, j] +
                                       cache.G[j] @ delta_u[step, j])

            # For explicit stages (a_{ii} = 0): δZ_i = rhs + h a_{ii} G_i δu_i
            if np.isclose(A[i, i], 0):
                delta_Z[step, i] = rhs
            else:
                # Implicit stage: (I - h a_{ii} F_i) δZ_i = rhs + h a_{ii} G_i δu_i
                # For now, use explicit approximation (needs proper implicit solver)
                # TODO: Reuse cached factorization for implicit methods
                gamma = A[i, i]
                I_minus_hgamma_F = np.eye(n) - h * gamma * cache.F[i]
                rhs_implicit = rhs + h * gamma * (cache.G[i] @ delta_u[step, i])
                delta_Z[step, i] = np.linalg.solve(I_minus_hgamma_F, rhs_implicit)

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
    objective=None,
) -> AdjointSensitivityTrajectory:
    """
    Algorithm 4: Backward adjoint sensitivity from glm_opt.tex Section 7.

    Solve:
        A_n^T δμ^n = B_n^T δλ^n + Γ^n
        δλ^{n-1} = U^T δμ^n + V^T δλ^n + J_{yy} δy^[n-1]

    where Γ^n contains second-derivative terms:
        Γ_k^n = h[F_{yy}^{n,k}[Λ_k^n] δZ_k^n + F_{yu}^{n,k}[Λ_k^n] δu_k^n]

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
        objective: Objective function (for J_{yy} term)

    Returns:
        Adjoint sensitivity trajectory
    """
    N = trajectory.N
    s, r, n = method.s, method.r, trajectory.n
    A = method.A
    B = method.B

    delta_Lambda = np.zeros((N + 1, r, n))
    delta_Mu = np.zeros((N, s, n))
    delta_WeightedAdj = np.zeros((N, s, n))

    # Terminal condition for adjoint sensitivity
    # δλ[N] should capture how the terminal objective Hessian couples with state sensitivity
    # For terminal cost J(y_final), we have: δλ[N] depends on d²J/dy² δy_final
    # BUT in the discrete adjoints paper, this is handled through the gradient assembly
    # So the terminal condition here should indeed be zero for the Lagrangian formulation
    delta_Lambda[N] = 0

    # TODO: Verify if we need terminal Hessian contribution here or in gradient assembly

    # Backward sweep (same direction as adjoint solve)
    for step in range(N - 1, -1, -1):
        cache = trajectory.caches[step]
        Lambda_k = adjoint.WeightedAdj[step]  # Weighted adjoints Λ_k

        # Compute second-derivative forcing terms Γ_k
        # Γ_k = h [F_{yy}[Λ_k] δZ_k + F_{yu}[Λ_k] δu_k]
        Gamma = np.zeros((s, n))

        for k in range(s):
            # Check if problem has second derivatives
            if hasattr(problem, 'F_yy_action') and hasattr(problem, 'F_yu_action'):
                # Compute time at this step
                t_n = t0 + step * h

                # Compute Hessian-vector products
                # Γ_k = h * Λ_kᵀ [F_{yy} δZ_k + F_{yu} δu_k]
                # Using bilinearity of Hessian forms

                # F_{yy} δZ_k term
                F_yy_dZ = problem.F_yy_action(
                    trajectory.Z[step, k],
                    u[step, k],
                    t_n + method.c[k] * h,
                    sensitivity.delta_Z[step, k]
                )

                # F_{yu} δu_k term
                F_yu_du = problem.F_yu_action(
                    trajectory.Z[step, k],
                    u[step, k],
                    t_n + method.c[k] * h,
                    delta_u[step, k]
                )

                # Contract with weighted adjoint
                Gamma[k] = h * Lambda_k[k].T @ (F_yy_dZ + F_yu_du)

        # Solve adjoint sensitivity system: A^T δμ = B^T δλ + Γ
        # Same structure as adjoint solve, just enhanced RHS

        delta_lambda_ext = delta_Lambda[step + 1]

        # Backward substitution for explicit methods
        # For implicit methods, would reuse transposed factorization
        for i in range(s - 1, -1, -1):
            # Terminal contribution: h B[i]^T F[i]^T δλ
            delta_Mu[step, i] = h * cache.F[i].T @ (B[:, i] @ delta_lambda_ext)

            # Add second-derivative forcing
            delta_Mu[step, i] += Gamma[i]

            # Coupling from later stages: h Σ_{j>i} a_{ji} F_j^T δμ_j
            for j in range(i + 1, s):
                delta_Mu[step, i] += h * A[j, i] * cache.F[j].T @ delta_Mu[step, j]

            # For implicit stages (a_{ii} ≠ 0), would solve:
            # (I - h a_{ii} F_i^T) δμ_i = rhs
            # Using cached transpose factorization
            if not np.isclose(A[i, i], 0):
                # For now, just note that implicit case needs factorization
                # In production, would reuse cache.factorization with trans=True
                pass

        # Compute weighted adjoint sensitivities for Hessian assembly
        # δΛ_k = Σ_j a_{jk} δμ_j + Σ_j b_{jk} δλ_j
        for k in range(s):
            delta_WeightedAdj[step, k] = (
                A[:, k] @ delta_Mu[step] +  # Σ_j a_{jk} δμ_j
                B[:, k] @ delta_Lambda[step + 1]  # Σ_j b_{jk} δλ_j
            )

        # Propagate external stages backward
        # δλ^{n-1} = U^T δμ + V^T δλ + J_{yy} δy^{n-1}
        delta_Lambda[step] = method.U.T @ delta_Mu[step]
        delta_Lambda[step] += method.V.T @ delta_lambda_ext

        # Add objective Hessian term J_{yy} δy (running cost contribution)
        # For now omitted - would need objective.d2J_dy2(y[step], step)

    return AdjointSensitivityTrajectory(
        delta_Lambda=delta_Lambda,
        delta_Mu=delta_Mu,
        delta_WeightedAdj=delta_WeightedAdj,
    )
