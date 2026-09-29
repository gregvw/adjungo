"""Solver requirements deduction."""

from dataclasses import dataclass

from adjungo.core.method import GLMethod, StageType
from adjungo.core.plan import StepMethod
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

    #: Whether the matrix at a given stage time recurs on every later call on
    #: the same mesh. True when ``F`` depends on neither the state nor the
    #: control -- ``state_affine`` with ``jacobian_control_dependent=False``,
    #: so ``F = M(t)`` -- because the stage matrix ``I - h a_ii M(t_i)`` is
    #: then a function of the stage time alone, whatever the control. It says
    #: nothing about reuse *within* a solve: distinct stage times still give
    #: distinct matrices (C-17.2). Implied by a constant Jacobian. The store
    #: verifies every hit exactly (C-15.2), so a false statement here is
    #: refused, not reused.
    can_reuse_across_calls: bool = False

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

    def factorizations_for_repeated_solve(self) -> int | None:
        """Predicted LU factorizations for a later solve on the same mesh.

        Zero when every stage matrix of the later solve already occurred in an
        earlier one: always so for a constant Jacobian, and so for a Jacobian
        depending on time alone, whose matrices recur at the same stage times
        on every call (C-17.6). ``None`` otherwise, for the reason
        :meth:`factorizations_for_solve` gives.

        Like that count, this one is certified because it is invisible to
        every accuracy test: a store that stopped reusing across calls would
        return bit-identical derivatives at a higher cost.
        """
        if self.can_reuse_across_steps or self.can_reuse_across_calls:
            return 0
        return None


def deduce_requirements(
    method: StepMethod,
    problem: ProblemStructure,
    state_dim: int,
) -> SolverRequirements:
    """Dispatch logic following ``docs/linalg_requirements.tex``.

    That document is a design document, and its reuse rules were refuted by
    the certification work behind C-15: it derived factorization reuse from
    the tableau alone. Its status preamble records the correction. What is
    followed here is its classification of stage structure, not its
    conclusions about reuse.
    """

    is_explicit = method.stage_type == StageType.EXPLICIT

    if method.stage_type == StageType.PARTITIONED:
        # A partitioned method that reaches here was accepted by
        # PartitionedMethod's constructor, which refuses any pair whose
        # dependency graph has a cycle (C-6.2). So the stage system resolves
        # by substitution: no Newton iteration, no matrix, nothing to factor.
        #
        # This is *not* the same statement as StageType.EXPLICIT, which is a
        # property of one triangular A. Symplectic Euler's A^p has a nonzero
        # diagonal and Verlet's A^q is not triangular; neither is explicit as
        # an array. Routing them through the branch below would be right by
        # coincidence, and would stop being right for the first partitioned
        # method that does need a solve.
        return SolverRequirements(
            needs_newton=False,
            newton_system_size=0,
            factorizations_per_step=0,
            factorization_size=0,
            can_reuse_across_stages=False,
            can_reuse_across_steps=False,
            store_jacobians=True,
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r,
        )

    # Every remaining branch reads the single `A` of an ordinary tableau,
    # which is what the early return above leaves behind.
    assert isinstance(method, GLMethod)

    # A stage equation is linear in its unknown exactly when f is affine in
    # the state. `state_affine` says that directly. The enum membership test
    # below is the legacy route, kept because a caller may still construct a
    # ProblemStructure by declaring only `linearity`: all three of these
    # members describe an f that is affine in y (LINEAR: f = M(t)y + b(u,t);
    # BILINEAR: f = (H + uV)y; QUASILINEAR: F depends on u alone). It is a
    # weaker and more fragile statement of the same fact, since it requires
    # reading three enum descriptions as "affine in y", which is why
    # `state_affine` now exists.
    is_linear = problem.state_affine or problem.linearity in (
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

    # F independent of the state and the control means F = M(t): each stage
    # matrix is a function of its stage time alone, and recurs exactly at that
    # time on every call on the same mesh (C-17.6). A constant Jacobian is the
    # special case that also recurs across stages and steps.
    jacobian_depends_on_time_only = (
        problem.state_affine and not problem.jacobian_control_dependent
    )
    can_reuse_across_calls = (
        problem.jacobian_constant or jacobian_depends_on_time_only
    )

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
            can_reuse_across_calls=can_reuse_across_calls,
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
            can_reuse_across_calls=can_reuse_across_calls,
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
        can_reuse_across_calls=can_reuse_across_calls,
    )
