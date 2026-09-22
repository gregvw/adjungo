"""Optimization interface for external optimizers."""

from adjungo.optimization.gradient import assemble_gradient
from adjungo.optimization.hessian import assemble_hessian_vector_product
from adjungo.optimization.interface import GLMOptimizer
from adjungo.optimization.parametrization import (
    AffineControlParametrization,
    NodalControl,
    PiecewiseConstantControl,
)

__all__ = [
    "AffineControlParametrization",
    "GLMOptimizer",
    "NodalControl",
    "PiecewiseConstantControl",
    "assemble_gradient",
    "assemble_hessian_vector_product",
]
