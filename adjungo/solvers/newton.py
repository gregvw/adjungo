"""Newton solver for nonlinear stage equations."""

from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from adjungo.solvers.linear_stage import linear_stage_solve

#: Relative term of the C-5.1 stage-solve convergence test.
NEWTON_RTOL = 1e-13

#: Coefficient of the absolute term: ``atol = NEWTON_ATOL_FACTOR * y_scale``.
NEWTON_ATOL_FACTOR = 1e-13

#: Nominal stage-solve tolerance at unit scale, i.e.
#: ``stage_solve_tolerance(z, 1.0)`` for ``||z||_inf <= 1``.
#:
#: This is the **accuracy basis for every implicit method** and the quantity
#: clause C-3.4 requires the implicit certified tolerance to be a stated
#: multiple of. The derivation:
#:
#: A converged stage value satisfies ``||R(Z*)||_inf <= tol`` rather than
#: ``R(Z*) = 0``.  Writing ``Z_exact`` for the true root,
#:
#:     ``||Z* - Z_exact|| <= ||R_Z^{-1}|| * tol``,
#:
#: so the forward trajectory carries an error of order ``kappa * tol``, where
#: ``kappa`` bounds the stage-Jacobian inverse over the step.  The discrete
#: gradient is evaluated *at* ``Z*``; it is the exact derivative of the map
#: the code actually realizes, but the independent reference converges to its
#: own ``Z*`` and the two differ at this same order.  Agreement between
#: package and reference is therefore floored near ``kappa * tol``, **not** at
#: machine epsilon.
#:
#: Tests import this constant instead of hardcoding a number, so loosening
#: stage convergence cannot silently loosen a correctness claim.
STAGE_NEWTON_TOL = NEWTON_RTOL + NEWTON_ATOL_FACTOR


def stage_context(family: str, stage: int, step: int | None, t: float) -> str:
    """Build the failure locator required by NUMERICS.md C-5.3.

    The clause requires the exception to name the step index and the
    stage index. The stage time is included as well: it is what a user
    reading a physical model actually recognises, and it disambiguates
    when a caller drives the stepper directly without a step counter.
    """
    where = "step ?" if step is None else f"step {step}"
    return f"{family} {where}, stage {stage}, t={t:.6g}"


def stage_solve_tolerance(z: NDArray, y_scale: float) -> float:
    """Return the C-5.1 convergence threshold for a stage iterate.

    Clause C-5.1 requires a **scaled** test,

        ``||r(Z)||_inf <= rtol * ||Z||_inf + atol``,   ``atol = 1e-13 * y_scale``

    and prohibits an unscaled absolute test on raw magnitudes, which is
    simultaneously too strict for large states and too loose for small ones.
    A state of magnitude ``1e6`` cannot reach an absolute residual of
    ``1e-12`` at all in double precision; a state of magnitude ``1e-6`` would
    be accepted at a relative accuracy of only ``1e-6``.

    ``y_scale`` is the characteristic state magnitude, captured once at
    construction (default ``max(||y0||_inf, 1)``) rather than re-read per
    step, so the threshold a stage is certified against cannot drift as the
    trajectory evolves.
    """
    return NEWTON_RTOL * float(np.max(np.abs(z))) + NEWTON_ATOL_FACTOR * y_scale


class StageSolveError(RuntimeError):
    """A nonlinear stage equation did not converge.

    Raised rather than returning the last iterate. Clause C-7 forbids silent
    sentinels, and an unconverged stage value is exactly that: every
    downstream derivative would be the derivative of a different discrete
    map (C-5.4).
    """

    def __init__(
        self,
        iterations: int,
        residual: float,
        tol: float,
        context: str,
        residual_vector: NDArray | None = None,
    ) -> None:
        super().__init__(
            f"{context}: Newton failed to converge in {iterations} "
            f"iterations; final ||r||_inf = {residual:.6e}, tolerance "
            f"{tol:.6e}"
        )
        self.iterations = iterations
        self.residual = residual
        self.tol = tol
        #: The C-5.3 locator, kept verbatim so a caller can rebuild a more
        #: specific message without parsing the formatted text of this one.
        self.context = context
        #: Final residual, kept so that a caller solving a *coupled* system
        #: can report which stage block failed. C-5.3 requires the stage
        #: index, and a coupled solve has no single failing stage until the
        #: residual is examined per block.
        self.residual_vector = residual_vector


class NewtonMixin:
    """Mixin providing Newton iteration for nonlinear stage equations.

    ``newton_entries`` counts calls to :meth:`newton_solve` on this instance.
    It exists because the affine route added in milestone M7 is invisible to
    every accuracy test: a direct linear solve and a converged Newton solve
    produce the same stage values to solver tolerance, so no assertion on a
    gradient, a Hessian or an order rate can tell which one ran. The dispatch is
    therefore certified by counting, exactly as factorization reuse is (C-15.1,
    C-16.3).
    """

    #: Calls to :meth:`newton_solve` since the last :meth:`reset_newton_count`.
    newton_entries: int = 0

    def reset_newton_count(self) -> None:
        self.newton_entries = 0

    def newton_solve(
        self,
        residual_fn: Callable[[NDArray], NDArray],
        jacobian_fn: Callable[[NDArray], NDArray],
        z0: NDArray,
        y_scale: float = 1.0,
        max_iter: int = 50,
        context: str = "stage solve",
        factor: Callable[[NDArray], Any] | None = None,
    ) -> tuple[NDArray, Any]:
        """
        Newton's method for a nonlinear stage equation.

        Convergence uses the **scaled** criterion of clause C-5.1,
        ``||r||_inf <= rtol * ||z||_inf + atol``; see
        :func:`stage_solve_tolerance`. An unscaled absolute test is
        prohibited by that clause.

        Two properties are load-bearing for the adjoint (NUMERICS.md C-5.4):

        * The returned factorization is computed **at the converged iterate**.
          Factoring at the previous iterate and returning that is the defect
          recorded as B7: the adjoint would then apply the transpose of a
          matrix evaluated at the wrong point, and the resulting gradient
          error would shrink with the Newton tolerance rather than vanish,
          which is easy to misread as discretization error.
        * Non-convergence raises. Returning the last iterate would silently
          change which discrete map the gradient differentiates.

        Args:
            residual_fn: Function computing residual ``r(z)``
            jacobian_fn: Function computing Jacobian ``dr/dz``
            z0: Initial guess
            y_scale: Characteristic state magnitude setting the absolute term
                of the C-5.1 threshold
            max_iter: Maximum iterations
            context: Text prefixed to a failure message, naming the caller
            factor: How to factor a Jacobian. Defaults to
                ``scipy.linalg.lu_factor``. A caller that owns a
                :class:`~adjungo.solvers.factorization.FactorizationStore`
                passes a bound method so that a constant Jacobian is factored
                once rather than once per stage per step. Substituting this
                cannot change the matrix that is solved with: the store
                returns a stored factorization only after verifying that its
                matrix is element-for-element identical to this one.

        Returns:
            ``(z, lu)``: the converged iterate and the LU factorization of
            ``dr/dz`` evaluated at it, suitable for reuse by the adjoint via
            ``lu_solve(..., trans=1)``.

        Raises:
            StageSolveError: If the C-5.1 threshold is not reached within
                ``max_iter`` iterations.
        """
        lu_factor = scipy.linalg.lu_factor if factor is None else factor

        self.newton_entries += 1

        z = np.array(z0, dtype=float)
        residual = np.inf
        tol = stage_solve_tolerance(z, y_scale)
        r = np.zeros_like(z)

        for iteration in range(max_iter + 1):
            r = residual_fn(z)
            residual = float(np.max(np.abs(r)))
            tol = stage_solve_tolerance(z, y_scale)
            if residual <= tol:
                # Factor at the converged iterate, not at a previous one.
                return z, lu_factor(jacobian_fn(z))
            if iteration == max_iter:
                break
            lu = lu_factor(jacobian_fn(z))
            z = z + scipy.linalg.lu_solve(lu, -r)

        raise StageSolveError(max_iter, residual, tol, context, r)


class StageDispatchMixin(NewtonMixin):
    """Chooses between the Newton route and the exact affine route.

    Both routes take the same arguments and return the same pair, so the choice
    is made in one place rather than repeated in each solver. Repeating it would
    put the same three-line branch in three files, which is how two of them stay
    correct and one does not.

    The choice is driven by ``needs_newton``, which
    :func:`~adjungo.core.requirements.deduce_requirements` computes and which,
    before milestone M7, nothing consumed: every implicit method entered Newton
    regardless of whether its stage equations were linear.

    The default is ``True``. A solver constructed without an opinion iterates,
    which is correct for every problem and merely slower than necessary for
    some. The error direction is toward doing more work, never toward asserting
    a structure the problem does not have.
    """

    #: Whether stage equations must be solved by iteration. ``False`` selects
    #: :func:`~adjungo.solvers.linear_stage.linear_stage_solve`, which is exact
    #: for a residual that is affine in its unknown and which verifies that
    #: affineness by testing the residual at the value it returns.
    needs_newton: bool = True

    def solve_stage_equation(
        self,
        residual_fn: Callable[[NDArray], NDArray],
        jacobian_fn: Callable[[NDArray], NDArray],
        z0: NDArray,
        y_scale: float = 1.0,
        context: str = "stage solve",
        factor: Callable[[NDArray], Any] | None = None,
    ) -> tuple[NDArray, Any]:
        """Solve one stage equation by whichever route the structure allows."""
        if self.needs_newton:
            return self.newton_solve(
                residual_fn,
                jacobian_fn,
                z0,
                y_scale=y_scale,
                context=context,
                factor=factor,
            )
        return linear_stage_solve(
            residual_fn,
            jacobian_fn,
            z0,
            y_scale=y_scale,
            context=context,
            factor=factor,
        )
