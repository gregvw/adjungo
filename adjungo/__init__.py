"""
Adjungo: General Linear Method optimization library for optimal control problems.

This library provides a framework for solving optimal control problems using
General Linear Methods (GLMs) with support for:
- Forward state solving
- Backward adjoint computation
- Sensitivity analysis
- Gradient and Hessian-vector product assembly
"""

__version__ = "0.1.0"

from adjungo.core.method import GLMethod, StageType, PropType
from adjungo.core.problem import Problem, ProblemStructure
from adjungo.optimization.interface import GLMOptimizer

__all__ = [
    "GLMethod",
    "StageType",
    "PropType",
    "Problem",
    "ProblemStructure",
    "GLMOptimizer",
]
