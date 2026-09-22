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

    # Fully implicit. ImplicitStageSolver.__init__ raises (NUMERICS.md C-6.2);
    # constructing it here rather than returning a placeholder keeps the refusal
    # on the direct-construction path too, not only via GLMOptimizer.
    return ImplicitStageSolver()
