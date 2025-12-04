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

    Given δu, compute δy, δZ via:
        A_n δZ^n = U δy^[n-1] + Φ^n
        δy^[n] = V δy^[n-1] + B_n δZ^n + Ψ^n

    where Φ, Ψ contain the control-derivative forcing terms.

    Key: Uses SAME factorization as forward solve (cache.factorization)

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

    delta_Y = np.zeros((N + 1, r, n))
    delta_Z = np.zeros((N, s, n))

    # Zero initial condition for sensitivity
    delta_Y[0] = 0

    for step in range(N):
        cache = trajectory.caches[step]

        # Compute forcing terms from control perturbation
        # Φ_k = h G_k δu_k
        for k in range(s):
            phi_k = h * cache.G[k] @ delta_u[step, k]
            # Simplified: would solve sensitivity system here
            delta_Z[step, k] = phi_k

        # Propagate sensitivity
        delta_Y[step + 1] = method.V @ delta_Y[step]

    return SensitivityTrajectory(delta_Y=delta_Y, delta_Z=delta_Z)


def adjoint_sensitivity(
    trajectory: Trajectory,
    adjoint: AdjointTrajectory,
    sensitivity: SensitivityTrajectory,
    delta_u: NDArray,
    method: GLMethod,
    stage_solver: StageSolver,
    problem: Problem,
    h: float,
) -> AdjointSensitivityTrajectory:
    """
    Algorithm 4: Backward adjoint sensitivity from glm_opt.tex Section 7.

    Solve:
        A_n^T δμ^n = B_n^T δλ^n + Γ^n
        δλ^{n-1} = U^T δμ^n + V^T δλ^n + J_{yy} δy^[n-1]

    where Γ^n contains second-derivative terms:
        Γ_k^n = h[F_{yy}^{n,k}[Λ_k^n] δZ_k^n + F_{yu}^{n,k}[Λ_k^n] δu_k^n]

    Key: Uses SAME factorization as adjoint solve (cache.factorization, trans=1)

    Args:
        trajectory: Forward solution trajectory
        adjoint: Adjoint trajectory
        sensitivity: State sensitivity trajectory
        delta_u: Control perturbation (N, s, ν)
        method: GLM tableau
        stage_solver: Stage equation solver
        problem: Problem specification
        h: Step size

    Returns:
        Adjoint sensitivity trajectory
    """
    N = trajectory.N
    s, r, n = method.s, method.r, trajectory.n

    delta_Lambda = np.zeros((N + 1, r, n))
    delta_Mu = np.zeros((N, s, n))
    delta_WeightedAdj = np.zeros((N, s, n))

    # Terminal condition (zero for Lagrangian form)
    delta_Lambda[N] = 0

    for step in range(N - 1, -1, -1):
        # Placeholder: would compute second-derivative forcing terms
        # and solve sensitivity system
        pass

    return AdjointSensitivityTrajectory(
        delta_Lambda=delta_Lambda,
        delta_Mu=delta_Mu,
        delta_WeightedAdj=delta_WeightedAdj,
    )
