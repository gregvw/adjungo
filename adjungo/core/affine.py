"""Affine dynamics ``y' = M y + C u + b``, as a representation rather than a claim.

Why this is a class and not a flag
----------------------------------

Milestone M6 accepted a *declaration* that the Jacobian is constant because it
could **verify** that declaration: comparing two assembled stage matrices
element for element costs ``O(n^2)`` and guards an ``O(n^3)`` factorization, so
the check is asymptotically cheaper than the work it protects. See
``NUMERICS.md`` C-15. The governing clause for this module is C-16, and
C-16.2 is the ruling this file implements.

No comparably cheap check exists for the statement "the dynamics have zero
curvature". Evaluating ``f`` at some sample points and observing that it looks
affine is a **probe**: it asks what ``f`` does at points and then generalizes to
points the computation never visits. Generalizing a probe is exactly the defect
recorded as precedent R-9, where a Jacobian probe that varied ``u`` and ``t``
but never ``y`` produced a 4.24e-05 relative gradient error with the whole test
suite passing.

So the structural facts here are not declared and then checked. They are
**constructive**. This class owns ``M``, ``C`` and ``b`` and computes ``f``,
``F``, ``G`` and all three contracted-Hessian actions from them. A caller cannot
instantiate it with non-affine dynamics, because there is nothing to instantiate
it with except the three coefficient arrays. Affineness is a property of this
class, provable by reading it, rather than an assertion by the caller that
nothing can check.

What this class licenses
------------------------

1. ``f`` is affine in ``y``, so every implicit stage equation is affine in its
   unknown and is solved exactly by one linear solve. Newton is not entered.
2. ``F = M`` is constant, so the M6 factorization store reuses a single
   factorization for an entire solve.
3. ``f_yy``, ``f_yu`` and ``f_uu`` all vanish identically, so the
   dynamics-curvature contributions to the second-order adjoint and to the
   Hessian-vector product are structurally zero.

Point 3 is the one that requires care. It licenses skipping the *dynamics*
curvature only. **Objective curvature is unaffected** and is still assembled in
full: an affine plant under a quadratic cost has a perfectly nonzero Hessian.

The trap this class exists to avoid
-----------------------------------

``Linearity.LINEAR`` is documented as "F independent of y, u". That gives
``f(y, u, t) = M(t) y + b(u, t)``: affine in the **state**, with ``b`` free to
be any function of the control whatsoever. It does **not** give ``f_uu = 0``.

The counterexample is already in the test suite.
``tests/problems.py::ConstantJacobianQuadraticControl`` is
``f = A y + B (u * u)``. Its ``F`` is the constant ``A``, so it satisfies
``LINEAR`` by that member's own definition, and its ``F_uu_action`` returns
``2 diag(B^T v)``, which is not zero. A dispatch rule of the form "``LINEAR``
implies skip curvature" would return a Gauss-Newton approximation from a
function whose docstring promises an exact Hessian.

That is why this module distinguishes two facts that the single enum conflates:

``state_affine``
    ``f(y, u, t) = M(u, t) y + b(u, t)``. Stage equations are linear in their
    unknowns for fixed controls. Says nothing about curvature in ``u``.

``jointly_affine``
    ``f(y, u, t) = M(t) y + C(t) u + b(t)``. All three dynamics-curvature blocks
    vanish. This is equivalent to zero curvature, not merely sufficient for it:
    a function whose second derivatives vanish identically is affine.

``ConstantJacobianQuadraticControl`` is ``state_affine`` and not
``jointly_affine``, which is precisely the distinction that keeps it from
losing its ``F_uu`` term.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["AffineDynamics", "affine_dynamics_verified"]


class AffineDynamics:
    """``y' = M y + C u + b`` with constant coefficients.

    Args:
        M: State matrix, shape ``(n, n)``.
        C: Control matrix, shape ``(n, nu)``.
        b: Constant forcing, shape ``(n,)``. Defaults to zero.

    The coefficient arrays are **copied at construction** and exposed through
    read-only properties backed by non-writeable buffers. Neither rebinding
    ``problem.M`` nor writing into ``problem.M[0, 0]`` is possible. This matters
    because the route selected for this problem asserts that ``F`` is the same
    matrix at every stage of every step; a coefficient that could change between
    steps would make the M6 factorization store reuse a factorization of a
    matrix that no longer exists.
    """

    def __init__(
        self, M: NDArray, C: NDArray, b: NDArray | None = None
    ) -> None:
        M_arr = np.array(M, dtype=float, copy=True)
        C_arr = np.array(C, dtype=float, copy=True)

        if M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError(
                f"M must be square (n, n); got shape {M_arr.shape}"
            )
        n = M_arr.shape[0]
        if C_arr.ndim != 2 or C_arr.shape[0] != n:
            raise ValueError(
                f"C must have shape (n, nu) with n = {n}; "
                f"got shape {C_arr.shape}"
            )

        b_arr = (
            np.zeros(n)
            if b is None
            else np.array(b, dtype=float, copy=True).reshape(-1)
        )
        if b_arr.shape != (n,):
            raise ValueError(
                f"b must have shape ({n},); got shape {b_arr.shape}"
            )

        for arr in (M_arr, C_arr, b_arr):
            arr.flags.writeable = False

        self._M = M_arr
        self._C = C_arr
        self._b = b_arr
        self._nu = int(C_arr.shape[1])

    @property
    def M(self) -> NDArray:
        """State matrix, read-only."""
        return self._M

    @property
    def C(self) -> NDArray:
        """Control matrix, read-only."""
        return self._C

    @property
    def b(self) -> NDArray:
        """Constant forcing, read-only."""
        return self._b

    @property
    def state_dim(self) -> int:
        return int(self._M.shape[0])

    @property
    def control_dim(self) -> int:
        return int(self._nu)

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._M @ y + self._C @ u + self._b)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._M

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._C

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(n, n)``.

        Defined rather than omitted so that the route which skips this term can
        be compared against the route which calls it. The two must agree
        exactly, because skipping is the removal of an addition of zero, not an
        approximation of one.
        """
        n = self.state_dim
        return np.zeros((n, n))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(n, nu)``."""
        return np.zeros((self.state_dim, self._nu))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(nu, nu)``."""
        return np.zeros((self._nu, self._nu))


#: The methods whose bodies carry the affine guarantee. If a subclass replaces
#: any of them, the guarantee is gone and the structural route must not be
#: taken. ``state_dim`` and ``control_dim`` are excluded deliberately: they are
#: dimensions, and a wrong one raises a shape error immediately rather than
#: producing a plausible wrong number.
_GUARANTEED_METHODS = (
    "f",
    "F",
    "G",
    "F_yy_action",
    "F_yu_action",
    "F_uu_action",
)


def affine_dynamics_verified(problem: Any) -> bool:
    """Whether ``problem`` is affine *by construction*, checked exactly.

    Being an instance of :class:`AffineDynamics` is not by itself sufficient. A
    subclass can override ``f`` with anything at all, and then the coefficient
    arrays describe nothing. This checks that every method carrying the
    guarantee is still the one :class:`AffineDynamics` defines, by object
    identity on the function found through the MRO.

    The check is exact and has no failure mode toward acceptance: an overridden
    method is a different function object, so the answer is ``False`` and the
    general route is taken. Error direction is toward doing more work, never
    toward claiming a structure the problem does not have. This mirrors the M6
    rule that the declaration selects the route and an exact comparison
    establishes the fact (``NUMERICS.md`` C-15.2, C-16.2).
    """
    if not isinstance(problem, AffineDynamics):
        return False
    cls = type(problem)
    return all(
        getattr(cls, name, None) is getattr(AffineDynamics, name)
        for name in _GUARANTEED_METHODS
    )
