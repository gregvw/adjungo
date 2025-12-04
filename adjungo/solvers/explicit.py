"""Explicit stage solver."""

import numpy as np
from numpy.typing import NDArray

from adjungo.solvers.base import StageSolver, StepCache
from adjungo.core.problem import Problem
from adjungo.core.method import GLMethod


class ExplicitStageSolver(StageSolver):
    """Forward substitution for strictly lower triangular A."""

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
        """Solve explicit stages via forward substitution."""
        s, n = method.s, problem.state_dim
        A, U, c = method.A, method.U, method.c

        Z = np.zeros((s, n))
        F_list: list[NDArray] = []
        G_list: list[NDArray] = []
        self._f_cached = [np.zeros(n) for _ in range(s)]

        for i in range(s):
            # Z_i = Σ_j U[i,j] y_j + h Σ_{j<i} A[i,j] f_j
            Z[i] = U[i] @ y_history
            for j in range(i):
                Z[i] += h * A[i, j] * self._f_cached[j]

            # Evaluate and cache
            t_stage = t_n + c[i] * h
            self._f_cached[i] = problem.f(Z[i], u_stages[i], t_stage)
            F_list.append(problem.F(Z[i], u_stages[i], t_stage))
            G_list.append(problem.G(Z[i], u_stages[i], t_stage))

        return Z, StepCache(Z=Z, F=F_list, G=G_list)

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: GLMethod,
    ) -> NDArray:
        """Backward substitution for A^T (upper triangular)."""
        s = method.s
        n = cache.Z.shape[1]
        A, B = method.A, method.B

        mu = np.zeros((s, n))

        # Backward substitution: μ_i = h Σ_{j>i} a_{ji} F_i^T μ_j + b_i F_i^T λ
        for i in range(s - 1, -1, -1):
            mu[i] = B[:, i].T @ lambda_ext
            for j in range(i + 1, s):
                mu[i] += A[j, i] * cache.F[i].T @ mu[j]

        return mu
