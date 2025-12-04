"""Solver factory and dispatch logic."""

from adjungo.solvers.base import StageSolver
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.solvers.sdirk import SDIRKStageSolver
from adjungo.solvers.dirk import DIRKStageSolver
from adjungo.solvers.implicit import ImplicitStageSolver
from adjungo.core.method import GLMethod, StageType
from adjungo.core.requirements import SolverRequirements
from adjungo.core.problem import ProblemStructure


def create_stage_solver(
    method: GLMethod,
    requirements: SolverRequirements,
    problem_structure: ProblemStructure,
) -> StageSolver:
    """
    Decision tree from linalg_requirements.tex Section 6.

    Args:
        method: GLM tableau
        requirements: Deduced solver requirements
        problem_structure: Problem structure information

    Returns:
        Appropriate stage solver
    """

    if method.stage_type == StageType.EXPLICIT:
        return ExplicitStageSolver()

    if method.stage_type == StageType.SDIRK:
        return SDIRKStageSolver(
            reuse_across_steps=requirements.can_reuse_across_steps
        )

    if method.stage_type == StageType.DIRK:
        return DIRKStageSolver()

    # Fully implicit
    return ImplicitStageSolver()
