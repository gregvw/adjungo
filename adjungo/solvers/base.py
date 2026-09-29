"""Base stage solver interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from adjungo.solvers.coupled import solve_coupled, solve_coupled_transposed

if TYPE_CHECKING:
    from adjungo.core.method import GLMethod
    from adjungo.core.plan import StepMethod
    from adjungo.core.problem import Problem




@dataclass
class StepCache:
    """Cached data from forward solve, reused in adjoint/sensitivity.

    ``stage_factorizations[i]`` holds the LU factorization of the stage
    matrix ``I - h A[i,i] F_i`` evaluated at the **converged** stage value
    ``Z[i]``, or ``None`` when stage ``i`` is explicit (``A[i,i] == 0``). The
    adjoint and the tangent both reuse it, the adjoint via
    ``lu_solve(..., trans=1)``.

    Storing one factorization per stage rather than a single object per step
    keeps the operator each stage used attributable. Entries are **never**
    shared between stages: precedent R-9 records a defect in which one
    stage's factorization was published for every stage of the step, giving
    the adjoint the transpose of the wrong matrix.

    ``coupled_factorization`` is the exception that proves the rule, and it
    is a different object with a different meaning. A tableau with a dense
    ``A`` cannot be solved stage by stage at all: all ``s`` stages form one
    ``(s·n) × (s·n)`` system whose Jacobian has blocks
    ``δ_ij I - h A[i,j] F_j``. There is genuinely one matrix for the step,
    and its transpose is exactly the operator of the adjoint stage system,
    so the single factorization is reused by both. It is set only by
    :class:`~adjungo.solvers.implicit.ImplicitStageSolver`, and
    ``stage_factorizations`` is left ``None`` in that case: no per-stage
    matrix exists to name.
    """

    Z: NDArray                          # (s, n) stage values
    F: list[NDArray]                    # s Jacobians, each (n, n)
    G: list[NDArray]                    # s control Jacobians, each (n, ν)
    stage_factorizations: list[Any] | None = None
    stage_matrix: NDArray | None = None
    coupled_factorization: Any = None


# ``M`` is the method kind a solver handles: a stage solver is written
# against one coefficient layout -- ``ExplicitStageSolver`` reads
# ``method.A``, which a partitioned method does not have -- so the binding
# belongs in the type rather than in a runtime check no dispatch can reach.
class StageSolver[M: StepMethod](ABC):
    """Solves the stage equations for one time step.

    ``y_scale`` is the characteristic state magnitude used by the C-5.1
    stage-convergence test. It is **captured at construction** and never
    re-read from mutable state, so the threshold a stage is certified against
    is the same one the route was configured with.
    """

    y_scale: float = 1.0

    @abstractmethod
    def solve_stages(
        self,
        y_history: NDArray,      # (r, n) external stages
        u_stages: NDArray,       # (s, ν) controls at stages
        t_n: float,
        h: float,
        problem: "Problem",
        method: M,
        step: int | None = None,
    ) -> tuple[NDArray, StepCache]:
        """
        Solve stage equations for one time step.

        Args:
            y_history: External stages from previous step (r, n)
            u_stages: Control values at each stage (s, ν)
            t_n: Time at start of step
            h: Step size
            problem: Problem specification
            method: GLM tableau
            step: Index of this time step, used only to locate a stage
                failure in the message required by NUMERICS.md C-5.3

        Returns:
            Z: Internal stage values (s, n)
            cache: Stored data for adjoint/sensitivity
        """
        ...

    @abstractmethod
    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,  # (r, n) external adjoints
        cache: StepCache,
        method: M,
        h: float,
    ) -> NDArray:
        """
        Solve adjoint stage equations: A^T μ = B^T λ.

        Args:
            lambda_ext: External stage adjoints (r, n)
            cache: Cached data from forward solve
            method: GLM tableau
            h: Step size

        Returns:
            μ: Stage adjoints (s, n)
        """
        ...

    @abstractmethod
    def weighted_adjoint(
        self,
        mu: NDArray,          # (s, n) stage adjoints
        lambda_ext: NDArray,  # (r, n) external adjoints
        method: M,
    ) -> NDArray:
        """``Λ_k = Σ_j a_jk μ_j + Σ_j b_jk λ_j``, shape ``(s, n)``.

        See :class:`GLMStageSolver` for the single-``A`` implementation.
        """
        ...

    @abstractmethod
    def solve_tangent_stages(
        self,
        delta_y_history: NDArray,  # (r, n) external stage sensitivities
        du_stages: NDArray,        # (s, ν) control perturbations
        cache: StepCache,
        method: M,
        h: float,
        step: int,
    ) -> NDArray:
        """Solve ``(I − h A ⊗ F) δZ = U δy + h (A ⊗ G) δu`` for ``δZ``.

        See :class:`GLMStageSolver` for the single-``A`` implementation.
        """
        ...

    @abstractmethod
    def solve_adjoint_sensitivity_stages(
        self,
        delta_lambda_ext: NDArray,  # (r, n) external adjoint sensitivities
        gamma: NDArray,             # (s, n) second-derivative forcing
        cache: StepCache,
        method: M,
        h: float,
    ) -> NDArray:
        """Solve ``A^T δμ = B^T δλ + Γ`` for the second-order stage adjoints.

        See :class:`GLMStageSolver` for the single-``A`` implementation.
        """
        ...


class GLMStageSolver(StageSolver["GLMethod"]):
    """A stage solver for a tableau that has a single ``A``.

    Holds the operations below, which read ``method.A``. They were inline in
    the stepping sweeps before C-18; a step implementation owns every
    operation that reads its own stage coupling (C-18.1), not only the two
    solves above, or the sweeps would have to branch on method type in four
    places instead of one.

    A partitioned method has no ``A`` at all -- ``A^q`` and ``A^p`` act on
    different halves of the state -- so it is not a specialisation of this
    class and does not inherit these. The split is in the type rather than in
    a runtime check because the dispatch in
    :mod:`adjungo.solvers.factory` makes the wrong pairing unreachable, and
    an unreachable check is one no test can exercise.
    """

    def weighted_adjoint(
        self,
        mu: NDArray,          # (s, n) stage adjoints
        lambda_ext: NDArray,  # (r, n) external adjoints
        method: "GLMethod",
    ) -> NDArray:
        """``Λ_k = Σ_j a_jk μ_j + Σ_j b_jk λ_j``, shape ``(s, n)``.

        This is the quantity the gradient contracts ``G_k`` against, so an
        error here is an error in every derivative the step produces. It is
        used by both the first- and second-order backward sweeps, which is
        why it is one method rather than two copies.
        """
        s = method.s
        A, B = method.A, method.B
        return np.array(
            [A[:, k] @ mu + B[:, k] @ lambda_ext for k in range(s)]
        )

    def solve_tangent_stages(
        self,
        delta_y_history: NDArray,  # (r, n) external stage sensitivities
        du_stages: NDArray,        # (s, ν) control perturbations
        cache: StepCache,
        method: "GLMethod",
        h: float,
        step: int,
    ) -> NDArray:
        """Solve ``(I − h A ⊗ F) δZ = U δy + h (A ⊗ G) δu`` for ``δZ``.

        The operator is exactly the forward Newton Jacobian, so the
        factorization taken at the converged stage value is reused rather
        than rebuilt. Rebuilding it would be a second opportunity to evaluate
        ``F`` at the wrong point (C-5.4).
        """
        s = method.s
        n = cache.Z.shape[1]
        A = method.A
        dZ = np.zeros((s, n))

        if cache.coupled_factorization is not None:
            # Dense A: the sum over j runs over every stage, so there is no
            # order in which the stages become available one at a time.
            rhs_coupled = np.empty((s, n))
            for i in range(s):
                acc = np.asarray(method.U[i] @ delta_y_history, dtype=float)
                for j in range(s):
                    if A[i, j] != 0.0:
                        acc = acc + h * A[i, j] * (cache.G[j] @ du_stages[j])
                rhs_coupled[i] = acc
            dZ[...] = solve_coupled(cache.coupled_factorization, rhs_coupled)
            return dZ

        for i in range(s):
            # RHS: U δy^{n-1} + h Σ_{j<i} a_ij [F_j δZ_j + G_j δu_j]
            rhs = method.U[i] @ delta_y_history
            for j in range(i):
                rhs = rhs + h * A[i, j] * (
                    cache.F[j] @ dZ[j] + cache.G[j] @ du_stages[j]
                )

            if A[i, i] == 0.0:  # exact, as in the forward solve (C-8.3)
                dZ[i] = rhs
                continue

            # Implicit stage. Differentiating
            #   Z_i - h a_ii f(Z_i, u_i, t_i) = U[i] y + h Σ_{j<i} A[i,j] f_j
            # gives (I - h a_ii F_i) δZ_i = rhs + h a_ii G_i δu_i, whose
            # operator is the forward stage matrix.
            gamma = A[i, i]
            rhs_implicit = rhs + h * gamma * (cache.G[i] @ du_stages[i])
            lu = (
                cache.stage_factorizations[i]
                if cache.stage_factorizations is not None
                else None
            )
            if lu is None:
                # Unreachable given the forward solvers: an implicit stage
                # always records the factorization it used, and an explicit
                # one (a_ii = 0) took the branch above.
                #
                # This used to rebuild I - h a_ii F_i and call
                # np.linalg.solve. That produced the right answer, and was
                # still wrong to do. It is an LU factorization that no
                # FactorizationStore counts, so it would make the C-15
                # certified count understate the real cost, and it would do
                # so precisely when the invariant this module depends on had
                # already broken -- the quietest possible moment. Refusing
                # states the invariant.
                raise RuntimeError(
                    f"no factorization recorded for implicit stage {i} of "
                    f"step {step} (A[{i},{i}] = {gamma}). The forward solve "
                    "must publish the matrix it used, because the tangent "
                    "solves with that same matrix (NUMERICS.md C-5.4)."
                )
            dZ[i] = scipy.linalg.lu_solve(lu, rhs_implicit)

        return dZ

    def solve_adjoint_sensitivity_stages(
        self,
        delta_lambda_ext: NDArray,  # (r, n) external adjoint sensitivities
        gamma: NDArray,             # (s, n) second-derivative forcing
        cache: StepCache,
        method: "GLMethod",
        h: float,
    ) -> NDArray:
        """Solve ``A^T δμ = B^T δλ + Γ`` for the second-order stage adjoints.

        The same linear system the first-order adjoint solves, with a
        different right-hand side, so it is solved the same way.
        """
        s = method.s
        n = cache.Z.shape[1]
        A, B = method.A, method.B
        dmu = np.zeros((s, n))

        if cache.coupled_factorization is not None:
            # Dense A: the operator is the transpose of the forward coupled
            # Jacobian. Every A-weighted coupling lives in that operator, so
            # the right-hand side carries only the external-adjoint term and
            # Γ; adding an A term here as well -- the natural slip when
            # adapting the triangular branch below -- would count the
            # coupling twice.
            rhs_coupled = np.empty((s, n))
            for i in range(s):
                rhs_coupled[i] = (
                    h * cache.F[i].T @ (B[:, i] @ delta_lambda_ext)
                    + gamma[i]
                )
            dmu[...] = solve_coupled_transposed(
                cache.coupled_factorization, rhs_coupled
            )
            return dmu

        # A is (strictly) lower triangular, so the same system is block
        # triangular and backward substitution solves it in s small blocks
        # instead of one (s*n) solve.
        for i in range(s - 1, -1, -1):
            weighted = B[:, i] @ delta_lambda_ext
            for j in range(i + 1, s):
                weighted = weighted + A[j, i] * dmu[j]
            rhs = h * cache.F[i].T @ weighted + gamma[i]

            # Implicit stages carry the same transposed stage solve as the
            # first-order adjoint: (I - h a_ii F_i^T) δμ_i = rhs.
            lu = (
                cache.stage_factorizations[i]
                if cache.stage_factorizations is not None
                else None
            )
            dmu[i] = rhs if lu is None else scipy.linalg.lu_solve(
                lu, rhs, trans=1
            )

        return dmu


def step_solvers(
    stage_solver: "StageSolver[Any] | Sequence[StageSolver[Any]]", n_steps: int
) -> tuple["StageSolver[Any]", ...]:
    """Normalize a solver argument to one solver per step (C-18.1).

    A plan may execute a different method at each step, and the route that
    solves a step's stage equations is a property of that step's method, so
    the stepping code indexes solvers by step. A single solver is broadcast,
    which is what every uniform plan supplies.

    Broadcasting is *not* the same as accepting a length mismatch: a sequence
    of the wrong length is refused rather than cycled or truncated, because
    either would silently execute a step with the route belonging to a
    different method.
    """
    if isinstance(stage_solver, StageSolver):
        return (stage_solver,) * n_steps
    solvers = tuple(stage_solver)
    if len(solvers) != n_steps:
        raise ValueError(
            f"expected one stage solver per step ({n_steps}), got "
            f"{len(solvers)} (NUMERICS.md C-18.1)."
        )
    for step, solver in enumerate(solvers):
        if not isinstance(solver, StageSolver):
            raise TypeError(
                f"stage_solver[{step}] must be a StageSolver, got "
                f"{type(solver).__name__}."
            )
    return solvers
