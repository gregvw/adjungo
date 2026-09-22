"""Core abstractions for GLM optimization."""

from adjungo.core.method import GLMethod, PropType, StageType
from adjungo.core.problem import Problem, ProblemStructure
from adjungo.core.requirements import SolverRequirements, deduce_requirements

__all__ = [
    "GLMethod",
    "Problem",
    "ProblemStructure",
    "PropType",
    "SolverRequirements",
    "StageType",
    "deduce_requirements",
]
