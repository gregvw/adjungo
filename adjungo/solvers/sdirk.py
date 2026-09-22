"""SDIRK stage solver with factorization reuse."""

from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem
from adjungo.solvers.base import StageSolver, StepCache
from adjungo.solvers.newton import NewtonMixin, stage_context


class SDIRKStageSolver(NewtonMixin, StageSolver):
    """
    SDIRK stage solver: a constant diagonal coefficient ``γ`` means every
    implicit stage has a matrix of the same *form*, ``I - h γ F``.

    **Factorization reuse is not performed.** Each stage publishes the LU
    factorization taken at its own converged iterate. That is the matrix the
    stage equation actually used and therefore the matrix whose transpose the
    adjoint must use (NUMERICS.md C-5.4).

    Reuse is only exact when ``F`` is genuinely constant, and the solver
    cannot establish that. An earlier version of this class probed ``F`` at
    ``(y_history[0], u_i, t_i)`` and, on agreement, published one shared
    factorization object for every stage. The probe varied only ``u`` and
    ``t``: it never varied the state, so a Jacobian depending on ``y`` alone
    passed it. The adjoint then solved with ``(I - h γ F_0)^T`` at every
    stage instead of ``(I - h γ F_i)^T``. For ``f = y^2 + u`` at ``N = 4``
    this produced a relative gradient error of 4.24e-05 against the
    monolithic reference, seven orders above the certified tolerance, while
    every test in the suite still passed because the C-14 population's
    Jacobian happens to depend on ``u``. See precedent R-9.

    The substitution also bought nothing: Newton has already factored each
    stage by the time it converges, so sharing the object saved no work and
    only risked the adjoint matrix. Skipping the factorization *work* requires
    knowing constancy in advance, from a declared
    :class:`~adjungo.core.problem.ProblemStructure`, not from a runtime probe
    at points the stages do not visit. That is milestone M6.
    """

    def __init__(
        self, reuse_across_steps: bool = False, y_scale: float = 1.0
    ) -> None:
        self._reuse_across_steps = reuse_across_steps
        self.y_scale = y_scale

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
        """Solve the SDIRK stage equations.

        Stage ``i`` satisfies

            Z_i - h γ f(Z_i, u_i, t_i) = U[i] y^[n] + h Σ_{j<i} A[i,j] f_j

        solved by Newton. This previously solved ``(I - h γ F) Z_i = rhs``
        with ``F`` evaluated at the *previous* stage value ``Z[i-1]``. That
        form is the correct stage equation only when ``f(y, u, t) = F y``
        exactly: it silently drops ``h γ f(0, u, t)``, so any control term,
        any inhomogeneity, and any nonlinearity were simply omitted from the
        stage solve. The observed consequence was first-order accuracy from a
        method whose whole purpose is higher order (defect B3).
        """
        gamma = method.sdirk_gamma
        if gamma is None:
            raise ValueError("Method is not SDIRK")

        s, n = method.s, problem.state_dim
        eye = np.eye(n)

        Z = np.zeros((s, n))
        F_list: list[NDArray] = []
        G_list: list[NDArray] = []
        factorizations: list[Any] = []
        f_cached = [np.zeros(n) for _ in range(s)]

        for i in range(s):
            t_stage = t_n + method.c[i] * h

            rhs = method.U[i] @ y_history
            for j in range(i):
                rhs = rhs + h * method.A[i, j] * f_cached[j]

            if np.isclose(method.A[i, i], 0):
                Z[i] = rhs
                factorizations.append(None)
            else:
                # Every loop-varying quantity is bound as a default argument.
                # Late binding here would silently solve the wrong stage
                # equation if these closures ever outlived the iteration.
                def residual(
                    z: NDArray,
                    _u: NDArray = u_stages[i],
                    _t: float = t_stage,
                    _rhs: NDArray = rhs,
                ) -> NDArray:
                    return np.asarray(
                        z - h * gamma * problem.f(z, _u, _t) - _rhs,
                        dtype=float,
                    )

                def jac(
                    z: NDArray,
                    _u: NDArray = u_stages[i],
                    _t: float = t_stage,
                ) -> NDArray:
                    return np.asarray(
                        eye - h * gamma * problem.F(z, _u, _t), dtype=float
                    )

                Z[i], lu = self.newton_solve(
                    residual,
                    jac,
                    rhs,
                    y_scale=self.y_scale,
                    context=stage_context("SDIRK", i, step, t_stage),
                )
                factorizations.append(lu)

            f_cached[i] = problem.f(Z[i], u_stages[i], t_stage)
            F_list.append(problem.F(Z[i], u_stages[i], t_stage))
            G_list.append(problem.G(Z[i], u_stages[i], t_stage))

        return Z, StepCache(
            Z=Z, F=F_list, G=G_list, stage_factorizations=factorizations
        )

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: GLMethod,
        h: float,
    ) -> NDArray:
        """
        Solve adjoint stages using the cached factorization.

        Differentiating the stage equations gives

            (I - h γ F_i^T) μ_i = h F_i^T ( Σ_{j>i} A[j,i] μ_j
                                            + Σ_l B[l,i] λ_l )

        ``F_i`` (not ``F_j``) multiplies the entire weighted sum, because
        ``Z_i`` enters every stage equation only through ``f(Z_i, ...)``.

        Key: ``scipy.linalg.lu_solve`` with ``trans=1`` solves ``A^T x = b``,
        so the forward factorization of ``I - h γ F_i`` — taken at the
        converged stage value — is reused directly.
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

            lu = (
                cache.stage_factorizations[i]
                if cache.stage_factorizations is not None
                else None
            )
            if lu is None:
                mu[i] = rhs
            else:
                mu[i] = scipy.linalg.lu_solve(lu, rhs, trans=1)

        return mu
