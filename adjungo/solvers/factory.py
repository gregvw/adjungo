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
    Decision tree from linalg_requirements.tex Section 6.

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
            y_scale=y_scale,
        )

    if method.stage_type == StageType.DIRK:
        return DIRKStageSolver(y_scale=y_scale)

    # Fully implicit (dense A): one coupled (s*n) Newton solve per step.
    return ImplicitStageSolver(y_scale=y_scale)
