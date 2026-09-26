"""DIRK stage solver."""

import functools
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem
from adjungo.solvers.base import StageSolver, StepCache
from adjungo.solvers.factorization import FactorizationStore
from adjungo.solvers.newton import StageDispatchMixin, stage_context


class DIRKStageSolver(StageDispatchMixin, StageSolver):
    """DIRK stage solver (each stage has a different diagonal element).

    Stages are solved one at a time in order, because ``A`` is lower
    triangular: stage ``i`` depends only on stages ``j < i``.
    """

    def __init__(
        self,
        y_scale: float = 1.0,
        reuse_across_steps: bool = False,
        reuse_across_calls: bool = False,
        needs_newton: bool = True,
    ) -> None:
        self.y_scale = y_scale
        #: Whether stage equations are solved by iteration. Set from
        #: :attr:`~adjungo.core.requirements.SolverRequirements.needs_newton`,
        #: which was computed and unconsumed before milestone M7. ``False``
        #: routes each stage through one exact linear solve.
        self.needs_newton = needs_newton
        #: A Jacobian depending on time alone (C-17.6): the matrix at a stage
        #: time recurs on every call on the same mesh but differs between
        #: stage times, so the key carries the stage time and reuse spans
        #: calls, never stages or steps. A constant Jacobian already reuses
        #: across all three and keeps the coarser key.
        self.key_by_stage_time = reuse_across_calls and not reuse_across_steps
        #: A DIRK tableau has distinct diagonal entries, so each implicit
        #: stage presents a different matrix ``I - h A[i,i] F`` even when
        #: ``F`` is constant. Keying on ``A[i,i]`` therefore gives one
        #: factorization per distinct diagonal coefficient for the entire
        #: solve, not one per step. Stages sharing a coefficient share an
        #: entry, which is sound for the same reason: the store compares the
        #: matrices before reusing.
        self.factorizations = FactorizationStore(
            reuse_enabled=reuse_across_steps or reuse_across_calls,
            across_calls=self.key_by_stage_time,
        )

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
        """Solve the DIRK stage equations.

        Stage ``i`` satisfies

            Z_i - h A[i,i] f(Z_i, u_i, t_i) = U[i] y^[n] + h Σ_{j<i} A[i,j] f_j

        which is solved by Newton for a general nonlinear ``f``.

        This previously linearized: it assumed ``f(z, u, t) = F z + G u`` and
        solved ``(I - h a_ii F) Z_i = rhs + h a_ii G u_i`` with ``F`` and
        ``G`` frozen at ``rhs``. That is exact only for a linear time-
        invariant problem, and for any other problem it silently integrated a
        different equation. It is also the reason the nonlinear DIRK test was
        skipped rather than failing.
        """
        s, n = method.s, problem.state_dim

        Z = np.zeros((s, n))
        F_list: list[NDArray] = []
        G_list: list[NDArray] = []
        factorizations: list[Any] = []
        f_cached = [np.zeros(n) for _ in range(s)]

        for i in range(s):
            t_stage = t_n + method.c[i] * h
            a_ii = method.A[i, i]

            rhs = method.U[i] @ y_history
            for j in range(i):
                rhs = rhs + h * method.A[i, j] * f_cached[j]

            if np.isclose(a_ii, 0):
                Z[i] = rhs
                factorizations.append(None)
            else:
                Z[i], lu = self._solve_implicit_stage(
                    rhs, u_stages[i], t_stage, h, a_ii, n, problem, i, step
                )
                factorizations.append(lu)

            f_cached[i] = problem.f(Z[i], u_stages[i], t_stage)
            F_list.append(problem.F(Z[i], u_stages[i], t_stage))
            G_list.append(problem.G(Z[i], u_stages[i], t_stage))

        return Z, StepCache(
            Z=Z, F=F_list, G=G_list, stage_factorizations=factorizations
        )

    def _solve_implicit_stage(
        self,
        rhs: NDArray,
        u_i: NDArray,
        t_stage: float,
        h: float,
        a_ii: float,
        n: int,
        problem: Problem,
        stage: int,
        step: int | None,
    ) -> tuple[NDArray, Any]:
        """Newton-solve ``z - h a_ii f(z, u_i, t) = rhs``."""
        eye = np.eye(n)

        def residual(z: NDArray) -> NDArray:
            return np.asarray(
                z - h * a_ii * problem.f(z, u_i, t_stage) - rhs, dtype=float
            )

        def jacobian(z: NDArray) -> NDArray:
            return np.asarray(
                eye - h * a_ii * problem.F(z, u_i, t_stage), dtype=float
            )

        return self.solve_stage_equation(
            residual,
            jacobian,
            rhs,
            y_scale=self.y_scale,
            context=stage_context("DIRK", stage, step, t_stage),
            factor=functools.partial(
                self.factorizations.factor,
                (a_ii, h, t_stage) if self.key_by_stage_time else (a_ii, h),
            ),
        )

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: GLMethod,
        h: float,
    ) -> NDArray:
        """Solve the adjoint stage equations for DIRK.

        Differentiating the stage equations gives

            (I - h A[i,i] F_i^T) μ_i = h F_i^T ( Σ_{j>i} A[j,i] μ_j
                                                 + Σ_l B[l,i] λ_l )

        Two points, each previously wrong:

        * ``F_i`` (not ``F_j``) multiplies the whole weighted sum, because
          ``Z_i`` enters every stage equation only through ``f(Z_i, ...)``.
        * The transposed stage solve on the left must actually be applied.
          Omitting it adjoints an explicit method while the forward solve ran
          implicit, which is the defect recorded as B2.

        ``lu_solve(..., trans=1)`` reuses the forward factorization of
        ``I - h A[i,i] F_i``, which the stage solve computed at the
        converged stage value.
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
