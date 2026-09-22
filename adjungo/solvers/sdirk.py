"""SDIRK stage solver with factorization reuse."""

from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem
from adjungo.solvers.base import StageSolver, StepCache


class SDIRKStageSolver(StageSolver):
    """
    Exploits constant diagonal γ: factor (I - hγF) once per step.
    For linear problems with constant F, factor once for all steps.
    """

    def __init__(self, reuse_across_steps: bool = False) -> None:
        self._global_factorization: Any | None = None
        self._reuse_across_steps = reuse_across_steps
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
        """Solve SDIRK stages with factorization reuse."""
        gamma = method.sdirk_gamma
        if gamma is None:
            raise ValueError("Method is not SDIRK")

        s, n = method.s, problem.state_dim

        Z = np.zeros((s, n))
        F_list: list[NDArray] = []
        G_list: list[NDArray] = []
        self._f_cached = [np.zeros(n) for _ in range(s)]

        factorization: Any | None = None

        for i in range(s):
            t_stage = t_n + method.c[i] * h

            # Build RHS: U[i] @ y_history + h Σ_{j<i} A[i,j] f_j
            rhs = method.U[i] @ y_history
            for j in range(i):
                rhs += h * method.A[i, j] * self._f_cached[j]

            if np.isclose(method.A[i, i], 0):
                # Explicit stage
                Z[i] = rhs
            else:
                # Implicit stage: solve (I - hγF)Z = rhs
                if factorization is None:
                    # First implicit stage: compute and factor
                    F_i = problem.F(
                        Z[i - 1] if i > 0 else y_history[0],
                        u_stages[i],
                        t_stage,
                    )
                    stage_matrix = np.eye(n) - h * gamma * F_i
                    factorization = scipy.linalg.lu_factor(stage_matrix)

                Z[i] = scipy.linalg.lu_solve(factorization, rhs)

            self._f_cached[i] = problem.f(Z[i], u_stages[i], t_stage)
            F_list.append(problem.F(Z[i], u_stages[i], t_stage))
            G_list.append(problem.G(Z[i], u_stages[i], t_stage))

        return Z, StepCache(
            Z=Z, F=F_list, G=G_list, factorization=factorization
        )

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: GLMethod,
        h: float,
    ) -> NDArray:
        """
        Solve adjoint stages using cached factorization.

        Differentiating the stage equations gives

            (I - h γ F_i^T) μ_i = h F_i^T ( Σ_{j>i} A[j,i] μ_j
                                            + Σ_l B[l,i] λ_l )

        ``F_i`` (not ``F_j``) multiplies the entire weighted sum, because
        ``Z_i`` enters every stage equation only through ``f(Z_i, ...)``.

        Key: ``scipy.linalg.lu_solve`` with ``trans=1`` solves ``A^T x = b``,
        so the forward factorization of ``I - h γ F`` is reused directly.
        """
        s = method.s
        n = cache.Z.shape[1]
        A, B = method.A, method.B

        mu = np.zeros((s, n))

        for i in range(s - 1, -1, -1):
            weighted = B[:, i] @ lambda_ext
            for j in range(i + 1, s):
                weighted = weighted + A[j, i] * mu[j]
            rhs = h * cache.F[i].T @ weighted

            if cache.factorization is not None and not np.isclose(
                method.A[i, i], 0
            ):
                # Use cached factorization with transpose
                mu[i] = scipy.linalg.lu_solve(
                    cache.factorization, rhs, trans=1
                )
            else:
                mu[i] = rhs

        return mu
