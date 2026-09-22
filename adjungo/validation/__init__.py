"""Independent verification references (C-14.1 oracle hierarchy).

This subpackage exists to provide a *truth source* for the discrete
derivatives produced by :mod:`adjungo.stepping` and
:mod:`adjungo.optimization`.  It deliberately shares no code with those
modules: it re-derives the discrete system from the GLM tableau alone.

Nothing in :mod:`adjungo.validation` is on a production code path.  It is
imported by tests and by diagnostic scripts only.
"""

from adjungo.validation.reference import (
    ReferenceSolution,
    reference_gradient,
    reference_hessian,
    reference_solve,
)

__all__ = [
    "ReferenceSolution",
    "reference_gradient",
    "reference_hessian",
    "reference_solve",
]
