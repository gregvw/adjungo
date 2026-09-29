"""Explicit stage solver."""

import numpy as np
from numpy.typing import NDArray

from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem
from adjungo.solvers.base import GLMStageSolver, StepCache


class ExplicitStageSolver(GLMStageSolver):
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
        step: int | None = None,
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
        h: float,
    ) -> NDArray:
        """Backward substitution for A^T (upper triangular).

        Differentiating the stage equations

            Z_i = Σ_k U[i,k] y^[n]_k + h Σ_j A[i,j] f(Z_j, u_j, t_j)

        with respect to ``Z_i`` gives the adjoint stage relation

            μ_i = h F_i^T ( Σ_j A[j,i] μ_j + Σ_l B[l,i] λ_l ) = h F_i^T Λ_i

        The stage Jacobian is evaluated at stage ``i`` and factors out of the
        *entire* weighted sum, because ``Z_i`` enters every equation only
        through ``f(Z_i, ...)``.  Applying ``F_j^T`` to the coupling term
        instead is wrong whenever ``s > 1`` and the Jacobian varies between
        stages; it is invisible for ``s = 1`` (empty coupling sum) and for a
        constant Jacobian (``F_i == F_j``).  See NUMERICS.md C-14.1 and the
        monolithic cross-check in ``tests/test_oracle_gradient.py``.
        """
        s = method.s
        n = cache.Z.shape[1]
        A, B = method.A, method.B

        mu = np.zeros((s, n))

        for i in range(s - 1, -1, -1):
            weighted = B[:, i] @ lambda_ext
            for j in range(i + 1, s):
                weighted = weighted + A[j, i] * mu[j]
            mu[i] = h * cache.F[i].T @ weighted

        return mu
