"""Fully implicit stage solver: one coupled ``(s·n) × (s·n)`` system per step.

A tableau whose stage matrix ``A`` is dense -- Gauss, Radau, Lobatto -- cannot
be advanced stage by stage. Stage ``i`` depends on every other stage, including
those with larger index, so there is no ordering in which the stages become
available one at a time. The ``s`` stage equations must be solved together.

History, recorded as NUMERICS.md precedent R-4: this class and its DIRK
counterpart previously returned zero arrays as placeholders. A Gauss-2 forward
solve therefore returned the initial condition unchanged, with no exception and
no warning, and the caller had no way to distinguish that from a correct result.
Under C-7 a silent sentinel is a contract violation rather than an incomplete
feature, so the bodies were deleted and construction was made to raise. This
module now implements the solve.
"""

import functools

import numpy as np
from numpy.typing import NDArray

from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem
from adjungo.solvers.base import GLMStageSolver, StepCache
from adjungo.solvers.coupled import (
    solve_coupled_transposed,
)
from adjungo.solvers.factorization import FactorizationStore
from adjungo.solvers.newton import StageDispatchMixin, StageSolveError


def coupled_stage_context(step: int | None, t_n: float, h: float) -> str:
    """Build the C-5.3 failure locator for a coupled step solve."""
    where = "step ?" if step is None else f"step {step}"
    return (
        f"fully implicit {where}, coupled stage system, "
        f"t=[{t_n:.6g}, {t_n + h:.6g}]"
    )


class ImplicitStageSolver(StageDispatchMixin, GLMStageSolver):
    """Solve all stages of a dense-``A`` tableau simultaneously by Newton.

    **Forward.** The stage equations are

        ``R_i(Z) = Z_i - Σ_k U[i,k] y^[n]_k - h Σ_j A[i,j] f(Z_j, u_j, t_j)``

    with Jacobian blocks

        ``∂R_i/∂Z_j = δ_ij I - h A[i,j] F_j``.

    Note which stage the Jacobian belongs to: block ``(i, j)`` carries
    ``F_j``, the Jacobian at the stage being *differentiated with respect
    to*, not at the stage whose equation is being written. This is the same
    distinction that produced defect B0 in the explicit adjoint (precedent
    R-5), appearing here transposed.

    **Adjoint.** Differentiating gives

        ``μ_p = h F_p^T ( Σ_i A[i,p] μ_i + Σ_l B[l,p] λ_l )``

    where the sum over ``i`` runs over **all** stages, not only those above
    or below ``p``, because ``A`` is dense. Collecting the ``μ`` terms, block
    ``(p, i)`` of the resulting operator is ``δ_pi I - h A[i,p] F_p^T``,
    which is exactly ``(∂R_i/∂Z_p)^T``. The adjoint operator is therefore the
    transpose of the forward Jacobian, and
    ``scipy.linalg.lu_solve(..., trans=1)`` applies it with no
    refactorisation.

    That identity is not a convenience: it is what makes the adjoint the
    derivative of the map the forward solve actually realised. The
    factorization is taken at the converged iterate (C-5.4), so the matrix
    differentiated and the matrix solved are the same object.

    **Why the explicit and DIRK solvers are not routed through this one.**
    Mathematically they could be: a (strictly) lower-triangular ``A`` makes
    the coupled system block triangular, and their backward substitution is
    that solve done in ``s`` small blocks rather than one ``(s·n)``
    factorisation. They are kept separate because the substitution is
    asymptotically cheaper -- ``s`` factorisations of size ``n`` against one
    of size ``s·n`` -- and because those routes are certified as written.
    Rewriting them to route through a dense coupled solve would re-open their
    certification for no numerical gain. See NUMERICS.md C-6.1.
    """

    def __init__(
        self,
        y_scale: float = 1.0,
        reuse_across_steps: bool = False,
        needs_newton: bool = True,
        *,
        reuse_across_calls: bool = False,
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
        #: The coupled Newton Jacobian is ``I - h (A kron I) blockdiag(F_j)``
        #: of size ``s*n``. With a constant ``F`` it is the same matrix at
        #: every step, so the single ``O((s*n)^3)`` factorization is taken
        #: once for the whole solve. This is the largest saving of the three
        #: families, and it is also the one where an unverified reuse would
        #: do the most damage: the adjoint, the tangent and the second-order
        #: adjoint all solve with these same factors.
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
        """Solve the coupled stage system for one step.

        The unknown handed to Newton is the flattened ``(s·n)`` vector of all
        stage values. Reusing
        :meth:`~adjungo.solvers.newton.StageDispatchMixin.solve_stage_equation`
        rather than writing a second iteration here means this route
        inherits the C-5.1 scaled convergence test, the C-5.4
        factor-at-the-converged-iterate guarantee, and the C-7
        raise-on-failure behaviour, instead of restating them and risking
        divergence between the two. It also inherits the affine route: a
        coupled system whose ``f`` is affine in the state is solved by one
        ``(s*n)`` linear solve, which is where avoiding Newton actually
        pays, since assembling the coupled Jacobian costs ``s^2`` blocks.
        """
        s, n = method.s, problem.state_dim
        A, U = method.A, method.U
        t_stages = t_n + method.c * h
        eye_sn = np.eye(s * n)

        # Constant part of the residual: the external-stage contribution.
        base = np.asarray(U @ y_history, dtype=float).reshape(s, n)

        def residual(z_flat: NDArray) -> NDArray:
            Z = z_flat.reshape(s, n)
            f_all = np.stack(
                [
                    np.asarray(problem.f(Z[j], u_stages[j], t_stages[j]))
                    for j in range(s)
                ]
            )
            return np.asarray(
                (Z - base - h * (A @ f_all)).ravel(), dtype=float
            )

        def jacobian(z_flat: NDArray) -> NDArray:
            Z = z_flat.reshape(s, n)
            F_all = [
                np.asarray(problem.F(Z[j], u_stages[j], t_stages[j]))
                for j in range(s)
            ]
            J = np.array(eye_sn)
            for i in range(s):
                for j in range(s):
                    if A[i, j] != 0.0:
                        J[i * n:(i + 1) * n, j * n:(j + 1) * n] -= (
                            h * A[i, j] * F_all[j]
                        )
            return J

        try:
            z_flat, lu = self.solve_stage_equation(
                residual,
                jacobian,
                base.ravel(),
                y_scale=self.y_scale,
                context=coupled_stage_context(step, t_n, h),
                factor=functools.partial(
                    self.factorizations.factor,
                    ("coupled", h, t_n)
                    if self.key_by_stage_time
                    else ("coupled", h),
                ),
            )
        except StageSolveError as exc:
            raise self._locate_failing_stage(exc, s, n) from None

        Z = z_flat.reshape(s, n)

        F_list = [
            np.asarray(problem.F(Z[i], u_stages[i], t_stages[i]))
            for i in range(s)
        ]
        G_list = [
            np.asarray(problem.G(Z[i], u_stages[i], t_stages[i]))
            for i in range(s)
        ]

        return Z, StepCache(
            Z=Z,
            F=F_list,
            G=G_list,
            stage_factorizations=None,
            coupled_factorization=lu,
        )

    @staticmethod
    def _locate_failing_stage(
        exc: StageSolveError, s: int, n: int
    ) -> StageSolveError:
        """Re-raise a coupled failure naming the worst stage block.

        C-5.3 requires the stage index in the failure message. A coupled
        solve has no single failing stage, so the clause is satisfied by
        naming the block carrying the largest residual component -- the
        stage a reader should look at first. When the residual vector is
        unavailable the original error is returned unchanged rather than
        fabricating an index.
        """
        r = exc.residual_vector
        if r is None or r.size != s * n:
            return exc

        blocks = np.max(np.abs(np.asarray(r).reshape(s, n)), axis=1)
        worst = int(np.argmax(blocks))
        return StageSolveError(
            exc.iterations,
            exc.residual,
            exc.tol,
            f"{exc.context}, worst block stage {worst} "
            f"(||r_{worst}||_inf = {blocks[worst]:.6e})",
            r,
        )

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: GLMethod,
        h: float,
    ) -> NDArray:
        """Solve the transposed coupled system for the stage adjoints.

        The right-hand side carries only the external-adjoint term,

            ``rhs_p = h F_p^T Σ_l B[l,p] λ_l``,

        because every ``A``-weighted coupling between stage adjoints lives in
        the operator rather than in the right-hand side. Placing an ``A``
        term in both -- the natural transcription error when adapting the
        triangular backward substitution the explicit and DIRK solvers use --
        would count the coupling twice.
        """
        s = method.s
        n = cache.Z.shape[1]
        B = method.B

        if cache.coupled_factorization is None:
            raise ValueError(
                "Fully implicit adjoint requires the coupled factorization "
                "produced by the forward solve, but the cache does not carry "
                "one. This indicates the cache came from a different solver."
            )

        rhs = np.empty((s, n))
        for p in range(s):
            rhs[p] = h * cache.F[p].T @ (B[:, p] @ lambda_ext)

        return solve_coupled_transposed(cache.coupled_factorization, rhs)
