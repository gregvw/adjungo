"""Core abstractions for GLM optimization."""

from adjungo.core.method import GLMethod, StageType, PropType
from adjungo.core.problem import Problem, ProblemStructure
from adjungo.core.requirements import SolverRequirements, deduce_requirements

__all__ = [
    "GLMethod",
    "StageType",
    "PropType",
    "Problem",
    "ProblemStructure",
    "SolverRequirements",
    "deduce_requirements",
]
