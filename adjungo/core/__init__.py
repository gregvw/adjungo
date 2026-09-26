"""Core abstractions for GLM optimization."""

from adjungo.core.method import (
    GLMethod,
    PropType,
    StageType,
    TableauDeclarationError,
)
from adjungo.core.problem import Problem, ProblemStructure
from adjungo.core.requirements import SolverRequirements, deduce_requirements

__all__ = [
    "GLMethod",
    "Problem",
    "ProblemStructure",
    "PropType",
    "SolverRequirements",
    "StageType",
    "TableauDeclarationError",
    "deduce_requirements",
]
