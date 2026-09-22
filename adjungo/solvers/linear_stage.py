"""Direct solution of stage equations that are affine in their unknown.

When ``f`` is affine in the state, the stage residual

    R_i(z) = z - h a_ii f(z, u_i, t_i) - rhs

is an affine function of ``z`` whose Jacobian ``K = I - h a_ii F`` does not
depend on ``z`` at all. One Newton step from any starting point therefore lands
exactly on the solution, and the iteration, the convergence loop and the
C-5.3 stage-failure mode all disappear.

What this route does and does not save
--------------------------------------

It does **not** reduce the factorization count. After milestone M6 a
declared-constant Jacobian already costs one factorization for an entire solve
(``NUMERICS.md`` C-15), and this route uses the same store. What it removes per
stage is one Jacobian evaluation, one call into the store, and the iteration
machinery. The saving is real but modest for DIRK and SDIRK, where ``K`` is
``n x n``; it is larger for a fully coupled tableau, where assembling ``K``
costs ``s^2`` Jacobian blocks of size ``n x n``.

The honest reason to take this route is not speed. It is that the stage values
are obtained by an exact solve rather than by an iteration stopped at a
tolerance, so they are *closer* to the quantity C-2 defines, and that a problem
whose structure was misdescribed is refused loudly instead of being iterated
quietly.

Why the published factorization is the right one
------------------------------------------------

Clause C-16.4 records this argument; clause C-5.4 requires the adjoint to use
the transpose of the matrix the
forward stage equation actually used, factored at the converged stage value.
This function factors ``K`` at the *starting* point ``z0``, not at the solution
``z``. That is correct here, and provably so rather than by assumption: for an
affine ``f`` the Jacobian ``F`` is independent of the state, so ``K`` evaluated
at ``z0`` and at ``z`` are the same matrix, element for element. There is no
approximation to bound.

That argument depends entirely on ``f`` actually being affine, so this module
does not take it on faith.

The check
---------

After computing ``z``, the residual ``R(z)`` is evaluated and tested against the
same scaled C-5.1 threshold Newton would have used. For an affine ``R`` it is
zero up to the backward error of the linear solve. For a non-affine ``R`` it is
not, and the call raises.

This is a check, not a probe. It evaluates the residual at the point the
computation actually uses and generalizes nothing; contrast precedent R-9, where
a Jacobian was sampled at one point and the result asserted of stages that were
never sampled. The cost is one extra evaluation of ``f`` per stage, which is
``O(n^2)`` for a dense problem and guards an ``O(n^3)`` solve.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

__all__ = ["NonAffineStageEquation", "linear_stage_solve"]


class NonAffineStageEquation(RuntimeError):
    """A stage equation routed as affine did not solve in one step.

    The affine route is selected only for a problem that is affine *by
    construction* -- an unmodified :class:`~adjungo.core.affine.AffineDynamics`,
    whose ``f`` is computed from coefficient arrays it owns and cannot replace.
    Reaching this exception therefore means the construction guarantee has been
    defeated, for instance by a subclass that intercepts attribute access, and
    the stage values returned would not satisfy the stage equation.

    This is a hard guard with no override. Returning the unconverged value would
    change which discrete map the gradient differentiates, which is the failure
    mode C-2 exists to exclude.
    """

    def __init__(
        self, context: str, residual: float, tol: float
    ) -> None:
        super().__init__(
            f"{context}: the stage equation was routed as affine in its "
            f"unknown, so one linear solve should have satisfied it exactly, "
            f"but the residual after that solve is {residual:.3e} against a "
            f"threshold of {tol:.3e}. The dynamics are not affine in the "
            f"state, or the coefficients changed during the solve."
        )
        self.residual = residual
        self.tol = tol


def linear_stage_solve(
    residual_fn: Callable[[NDArray], NDArray],
    jacobian_fn: Callable[[NDArray], NDArray],
    z0: NDArray,
    y_scale: float = 1.0,
    context: str = "stage solve",
    factor: Callable[[NDArray], Any] | None = None,
) -> tuple[NDArray, Any]:
    """Solve an affine stage equation exactly, in one linear solve.

    Args:
        residual_fn: The stage residual ``R(z)``, affine in ``z``.
        jacobian_fn: ``dR/dz``. For an affine ``R`` this is constant; the
            argument is accepted so that the signature matches
            :meth:`~adjungo.solvers.newton.NewtonMixin.newton_solve` and the
            two routes are interchangeable at the call site.
        z0: Any starting point. The result does not depend on it except through
            rounding, because one Newton step on an affine residual is exact.
        y_scale: Characteristic state magnitude for the C-5.1 threshold used by
            the verification below.
        context: Text prefixed to a failure message, naming the caller.
        factor: How to factor the Jacobian. Defaults to
            ``scipy.linalg.lu_factor``. Callers owning a
            :class:`~adjungo.solvers.factorization.FactorizationStore` pass a
            bound method so a constant Jacobian is factored once per solve.

    Returns:
        ``(z, lu)``: the stage value and the LU factorization of ``dR/dz``,
        suitable for the adjoint via ``lu_solve(..., trans=1)``.

    Raises:
        NonAffineStageEquation: If the residual at ``z`` does not meet the
            C-5.1 threshold, which for an affine residual it must.
    """
    from adjungo.solvers.newton import stage_solve_tolerance

    lu_factor = scipy.linalg.lu_factor if factor is None else factor

    z0 = np.asarray(z0, dtype=float)
    lu = lu_factor(np.asarray(jacobian_fn(z0), dtype=float))
    z = z0 + scipy.linalg.lu_solve(lu, -np.asarray(residual_fn(z0)))

    r = np.asarray(residual_fn(z))
    residual = float(np.max(np.abs(r)))
    tol = stage_solve_tolerance(z, y_scale)
    if residual > tol:
        raise NonAffineStageEquation(context, residual, tol)

    return z, lu
