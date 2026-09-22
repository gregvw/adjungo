"""Problem specification protocols."""

from enum import Enum, auto
from typing import Protocol

from numpy.typing import NDArray


class Problem(Protocol):
    """User provides callbacks; solver never forms full Jacobians unless needed."""

    @property
    def state_dim(self) -> int:
        """State dimension n."""
        ...

    @property
    def control_dim(self) -> int:
        """Control dimension ν."""
        ...

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """RHS evaluation: ẏ = f(y, u, t)."""
        ...

    # First derivatives (required for implicit methods and optimization)
    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """State Jacobian: ∂f/∂y, shape (n, n)."""
        ...

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """Control Jacobian: ∂f/∂u, shape (n, ν)."""
        ...

    # Second derivatives (optional - for second-order optimization)
    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Contracted Hessian: Σ_ℓ v_ℓ ∂²f_ℓ/∂y∂y, shape (n, n)."""
        ...

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Contracted Hessian: Σ_ℓ v_ℓ ∂²f_ℓ/∂y∂u, shape (n, ν)."""
        ...

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Contracted Hessian: Σ_ℓ v_ℓ ∂²f_ℓ/∂u∂u, shape (ν, ν)."""
        ...


class Linearity(Enum):
    """Problem linearity classification.

    **These members classify the state Jacobian only.** None of them
    establishes that the dynamics have zero curvature in the control, so none
    may be used to skip a contracted-Hessian term. ``LINEAR`` permits
    ``f = M y + b(u, t)`` with ``b`` arbitrary in ``u``. Use
    :attr:`ProblemStructure.jointly_affine` for that question.
    """
    LINEAR = auto()       # F independent of y, u
    BILINEAR = auto()     # F = H + uV
    QUASILINEAR = auto()  # F depends on u only
    SEMILINEAR = auto()   # F constant + nonlinear part
    NONLINEAR = auto()    # General


class ProblemStructure:
    """Deduced from problem or specified by user.

    Three facts here are *independent axes*, and merging them is how a wrong
    dispatch rule gets written:

    ``state_affine``
        ``f(y, u, t) = M(u, t) y + b(u, t)``. Every implicit stage equation is
        linear in its unknown, so a single linear solve is exact and Newton
        need not be entered. Says nothing about curvature in ``u``.

    ``jointly_affine``
        ``f(y, u, t) = M(t) y + C(t) u + b(t)``. All three dynamics-curvature
        blocks ``f_yy``, ``f_yu``, ``f_uu`` vanish identically. Equivalent to
        zero curvature, since a function with identically vanishing second
        derivatives is affine. Implies ``state_affine``.

    ``jacobian_constant``
        ``F`` is the same matrix everywhere, which is what licenses reuse of a
        factorization across stages and steps (C-15). Implied by
        ``jointly_affine`` with constant coefficients, but not by
        ``state_affine``: ``M(u, t)`` may vary.

    ``linearity`` is retained for continuity and **must never gate a curvature
    skip**. ``Linearity.LINEAR`` means "F independent of y, u", which permits
    ``f = M y + b(u, t)`` with ``b`` arbitrary in ``u``; see
    :mod:`adjungo.core.affine` and the in-tree counterexample
    ``tests/problems.py::ConstantJacobianQuadraticControl``, which is ``LINEAR``
    with a nonzero ``F_uu``.
    """

    def __init__(
        self,
        linearity: Linearity,
        jacobian_constant: bool,
        jacobian_control_dependent: bool,
        has_second_derivatives: bool,
        state_affine: bool = False,
        jointly_affine: bool = False,
    ):
        if jointly_affine and not state_affine:
            raise ValueError(
                "jointly_affine implies state_affine: f = M(t)y + C(t)u + b(t) "
                "is affine in the state. Constructing a structure that claims "
                "zero curvature while denying state affineness would enable "
                "the curvature skip and disable the direct stage solve, which "
                "is not a combination any problem exhibits."
            )
        self.linearity = linearity
        self.jacobian_constant = jacobian_constant
        self.jacobian_control_dependent = jacobian_control_dependent
        self.has_second_derivatives = has_second_derivatives
        self.state_affine = state_affine
        self.jointly_affine = jointly_affine
