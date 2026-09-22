"""Optimization interface for external optimizers."""

from adjungo.optimization.gradient import assemble_gradient
from adjungo.optimization.hessian import assemble_hessian_vector_product
from adjungo.optimization.interface import GLMOptimizer

__all__ = [
    "GLMOptimizer",
    "assemble_gradient",
    "assemble_hessian_vector_product",
]
