"""Solver factory and dispatch logic."""

from adjungo.core.method import GLMethod, StageType
from adjungo.core.problem import ProblemStructure
from adjungo.core.requirements import SolverRequirements
from adjungo.solvers.base import StageSolver
from adjungo.solvers.dirk import DIRKStageSolver
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.solvers.implicit import ImplicitStageSolver
from adjungo.solvers.sdirk import SDIRKStageSolver


def create_stage_solver(
    method: GLMethod,
    requirements: SolverRequirements,
    problem_structure: ProblemStructure,
    y_scale: float = 1.0,
) -> StageSolver:
    """
    Decision tree following ``docs/linalg_requirements.tex`` Section 6, whose
    status preamble records the four rules of that document which were
    subsequently refuted. The three-axis separation below is that document's
    own Section 1, which survived; its single ordered linearity test, which
    merged the axes, did not.

    Three independent axes decide the route, and merging them is how a wrong
    rule gets written:

    * the **tableau** fixes the shape of the solve -- none, sequential, or one
      coupled ``(s*n)`` system -- and is known statically from ``GLMethod``;
    * the **vector field** decides whether that solve is linear, through
      ``requirements.needs_newton``;
    * the **linear algebra** decides how the assembled matrix is represented
      and factored. Eligibility for a specialised factorization is a property
      of the assembled matrix, never of the tableau. ``K = I - h A (x) F`` is
      not symmetric for a general ``F``, so no tableau classification can
      establish that a Cholesky factorization applies.

    Args:
        method: GLM tableau
        requirements: Deduced solver requirements
        problem_structure: Problem structure information
        y_scale: Characteristic state magnitude for the C-5.1 stage
            convergence test, captured by the solver at construction

    Returns:
        Appropriate stage solver
    """

    if method.stage_type == StageType.EXPLICIT:
        return ExplicitStageSolver()

    if method.stage_type == StageType.SDIRK:
        return SDIRKStageSolver(
            reuse_across_steps=requirements.can_reuse_across_steps,
            reuse_across_calls=requirements.can_reuse_across_calls,
            y_scale=y_scale,
            needs_newton=requirements.needs_newton,
        )

    if method.stage_type == StageType.DIRK:
        return DIRKStageSolver(
            y_scale=y_scale,
            reuse_across_steps=requirements.can_reuse_across_steps,
            reuse_across_calls=requirements.can_reuse_across_calls,
            needs_newton=requirements.needs_newton,
        )

    # Fully implicit (dense A): one coupled (s*n) Newton solve per step.
    return ImplicitStageSolver(
        y_scale=y_scale,
        reuse_across_steps=requirements.can_reuse_across_steps,
        reuse_across_calls=requirements.can_reuse_across_calls,
        needs_newton=requirements.needs_newton,
    )
