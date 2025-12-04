"""DIRK stage solver."""

import numpy as np
from numpy.typing import NDArray

from adjungo.solvers.base import StageSolver, StepCache
from adjungo.core.problem import Problem
from adjungo.core.method import GLMethod


class DIRKStageSolver(StageSolver):
    """DIRK stage solver (each stage has different diagonal element)."""

    def __init__(self) -> None:
        self._f_cached: list[NDArray] = []

    def solve_stages(
        self,
        y_history: NDArray,
        u_stages: NDArray,
        t_n: float,
        h: float,
        problem: Problem,
        method: GLMethod,
    ) -> tuple[NDArray, StepCache]:
        """Solve DIRK stages."""
        s, n = method.s, problem.state_dim

        Z = np.zeros((s, n))
        F_list: list[NDArray] = []
        G_list: list[NDArray] = []
        self._f_cached = [np.zeros(n) for _ in range(s)]

        for i in range(s):
            t_stage = t_n + method.c[i] * h

            # Build RHS
            rhs = method.U[i] @ y_history
            for j in range(i):
                rhs += h * method.A[i, j] * self._f_cached[j]

            if np.isclose(method.A[i, i], 0):
                # Explicit stage
                Z[i] = rhs
            else:
                # Implicit stage: Z_i = rhs + h * a_ii * f(Z_i, u_i)
                # For linear problems: Z_i = rhs + h * a_ii * (F * Z_i + G * u_i)
                # Rearranging: (I - h * a_ii * F) * Z_i = rhs + h * a_ii * G * u_i

                # Get Jacobians (for LTI, these are constant)
                F_matrix = problem.F(rhs, u_stages[i], t_stage)
                G_matrix = problem.G(rhs, u_stages[i], t_stage)

                # Build implicit system
                gamma = method.A[i, i]
                I_minus_gamma_hF = np.eye(n) - h * gamma * F_matrix
                rhs_implicit = rhs + h * gamma * (G_matrix @ u_stages[i])

                # Solve linear system
                Z[i] = np.linalg.solve(I_minus_gamma_hF, rhs_implicit)

            self._f_cached[i] = problem.f(Z[i], u_stages[i], t_stage)
            F_list.append(problem.F(Z[i], u_stages[i], t_stage))
            G_list.append(problem.G(Z[i], u_stages[i], t_stage))

        return Z, StepCache(Z=Z, F=F_list, G=G_list)

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: GLMethod,
        h: float,
    ) -> NDArray:
        """Solve adjoint stages for DIRK."""
        s = method.s
        n = cache.Z.shape[1]
        A, B = method.A, method.B

        mu = np.zeros((s, n))

        for i in range(s - 1, -1, -1):
            mu[i] = h * cache.F[i].T @ (B[:, i] @ lambda_ext)
            for j in range(i + 1, s):
                mu[i] += h * A[j, i] * cache.F[j].T @ mu[j]

        return mu
