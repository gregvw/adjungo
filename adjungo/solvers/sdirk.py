"""SDIRK stage solver with factorization reuse."""

import numpy as np
import scipy.linalg
from typing import Optional, Any
from numpy.typing import NDArray

from adjungo.solvers.base import StageSolver, StepCache
from adjungo.core.problem import Problem
from adjungo.core.method import GLMethod


class SDIRKStageSolver(StageSolver):
    """
    Exploits constant diagonal γ: factor (I - hγF) once per step.
    For linear problems with constant F, factor once for all steps.
    """

    def __init__(self, reuse_across_steps: bool = False) -> None:
        self._global_factorization: Optional[Any] = None
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

        factorization: Optional[Any] = None

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
        Key: scipy.linalg.lu_solve with trans=1 solves A^T x = b.
        """
        s = method.s
        n = cache.Z.shape[1]
        A, B = method.A, method.B

        mu = np.zeros((s, n))

        # Solve (I - hγF)^T μ = h * (F_i^T B λ + coupling terms)
        for i in range(s - 1, -1, -1):
            rhs = h * cache.F[i].T @ (B[:, i] @ lambda_ext)
            for j in range(i + 1, s):
                rhs += h * A[j, i] * cache.F[j].T @ mu[j]

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
