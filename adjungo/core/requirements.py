"""Solver requirements deduction."""

from dataclasses import dataclass

from adjungo.core.method import GLMethod, StageType
from adjungo.core.problem import Linearity, ProblemStructure


@dataclass
class SolverRequirements:
    """What computational primitives are needed per time step."""

    # Stage solve requirements
    needs_newton: bool
    newton_system_size: int  # n for DIRK, ns for fully implicit

    # Factorization strategy
    #: Number of *distinct* stage matrices in one step. SDIRK has one, because
    #: every implicit stage shares the diagonal coefficient; DIRK has one per
    #: distinct nonzero diagonal entry; the fully implicit route has one, of
    #: size ``s*n``. Without reuse the solve costs at least this many
    #: factorizations per step, and more when Newton takes several iterations.
    factorizations_per_step: int
    factorization_size: int
    #: Whether stages within a step present the same matrix. This is true only
    #: when the Jacobian is genuinely constant. It must never be inferred from
    #: the Jacobian merely being control-independent: a Jacobian depending on
    #: the state alone is control-independent and still differs between
    #: stages, which is precisely the case precedent R-9 records as producing
    #: a 4.24e-05 relative gradient error with the whole suite passing.
    can_reuse_across_stages: bool
    #: Whether the same matrices recur at every step. Equivalent to
    #: ``can_reuse_across_stages`` here, since the mesh is uniform (C-3) and
    #: so ``h`` does not vary, but kept separate because a reimplementation
    #: supporting variable steps would have to distinguish them.
    can_reuse_across_steps: bool

    # What to store for optimization
    store_jacobians: bool
    store_stage_values: bool
    trajectory_vectors_per_step: int  # ns + nr for (Z, y)

    def factorizations_for_solve(self, steps: int) -> int | None:
        """Predicted LU factorizations for a whole forward solve.

        Returns ``None`` when the count is not predictable. Without reuse,
        every Newton iteration factors, and the iteration count depends on the
        problem and the initial guess; asserting a number there would be
        asserting a property of Newton, not of this dispatch.

        With reuse the count *is* predictable and is certified: it is the
        number of distinct stage matrices, independent of ``steps``. Clause
        C-15 requires this to be checked, because a reuse path that quietly
        stops reusing produces bit-identical answers and is therefore
        invisible to every accuracy test in the suite.

        The adjoint, tangent and second-order adjoint sweeps add nothing to
        this count. They solve with the transpose or with the same operator,
        via ``lu_solve``, on factors the forward solve already took.
        """
        if not self.can_reuse_across_steps:
            return None
        return self.factorizations_per_step


def deduce_requirements(
    method: GLMethod,
    problem: ProblemStructure,
    state_dim: int,
) -> SolverRequirements:
    """Dispatch logic from linalg_requirements.tex."""

    is_explicit = method.stage_type == StageType.EXPLICIT
    is_linear = problem.linearity in (
        Linearity.LINEAR,
        Linearity.BILINEAR,
        Linearity.QUASILINEAR,
    )

    if is_explicit:
        return SolverRequirements(
            needs_newton=False,
            newton_system_size=0,
            factorizations_per_step=0,
            factorization_size=0,
            can_reuse_across_stages=False,
            can_reuse_across_steps=False,
            store_jacobians=True,  # Still need F, G for adjoints
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r,
        )

    needs_newton = not is_linear
    n = state_dim

    if method.stage_type == StageType.SDIRK:
        # SDIRK's defining property is a single diagonal coefficient, so all
        # implicit stages present the matrix I - h g F: one distinct matrix
        # per step regardless of what the Jacobian does.
        #
        # Reuse of that matrix *across* stages and steps needs more: F must
        # not vary. This previously read
        #
        #     jacobian_constant or not jacobian_control_dependent
        #
        # which is false. A Jacobian depending on the state alone satisfies
        # the second disjunct and still differs at every stage. That is the
        # R-9 defect, stated as a deduction rule instead of as a probe. It
        # was inert only because nothing consumed the flag; M6 consumes it.
        return SolverRequirements(
            needs_newton=needs_newton,
            newton_system_size=n,
            factorizations_per_step=1,
            factorization_size=n,
            can_reuse_across_stages=problem.jacobian_constant,
            can_reuse_across_steps=problem.jacobian_constant,
            store_jacobians=True,
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r,
        )

    if method.stage_type == StageType.DIRK:
        # One distinct matrix per distinct nonzero diagonal entry. Explicit
        # stages (a_ii = 0) need no factorization, and two stages sharing a
        # coefficient share a matrix when F is constant.
        distinct_diagonals = {
            float(method.A[i, i])
            for i in range(method.s)
            if method.A[i, i] != 0.0
        }
        return SolverRequirements(
            needs_newton=needs_newton,
            newton_system_size=n,
            factorizations_per_step=len(distinct_diagonals),
            factorization_size=n,
            can_reuse_across_stages=False,
            can_reuse_across_steps=problem.jacobian_constant,
            store_jacobians=True,
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r,
        )

    # Fully implicit
    return SolverRequirements(
        needs_newton=needs_newton,
        newton_system_size=n * method.s,
        factorizations_per_step=1,
        factorization_size=n * method.s,
        can_reuse_across_stages=problem.jacobian_constant,
        can_reuse_across_steps=problem.jacobian_constant,
        store_jacobians=True,
        store_stage_values=True,
        trajectory_vectors_per_step=method.s + method.r,
    )
