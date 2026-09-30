"""Problem specification protocols."""

from enum import Enum, auto
from typing import Protocol

from numpy.typing import NDArray


class SeparableProblem(Protocol):
    """Optional split right-hand side for the partitioned route (C-8.4).

    A partitioned method never needs the whole of ``f`` at once. It needs
    ``f^q`` from the momentum and ``f^p`` from the position, control and
    time, and inside C-8.4's separable domain those arguments are exactly
    what each half reads. Declaring them separately lets the solver evaluate each block
    from its own arguments instead of assembling a whole state vector that
    the method has not finished computing.

    Declaring both is optional and strictly additive: a problem that
    declares neither is solved through ``f`` alone, under the totality
    assumption C-8.4 states for that path. Declaring exactly one is an
    authoring error and is refused. ``f`` stays required either way, and
    both halves are checked against it, exactly, at every completed stage.

    What this settles is one requirement of C-8.4's domain -- that neither
    half needs the state half the method has not computed. The rest of the
    domain is unaffected: the Jacobian block structure is still checked at
    runtime, and Hamiltonian structure is still a caller hypothesis.

    Why it exists: three separate valid Hamiltonians were refused by
    schemes that invented a value for the half no stage had written --
    ``NaN`` (an ``f`` built as a matrix product returns ``NaN`` for the
    whole row), a fixed finite constant (``log q`` needs ``q > 0``), and
    the step's incoming state (a domain that moves with ``t`` invalidates
    it at the stage time). Separability constrains what ``f`` *reads*, not
    where it is *defined*, and validity is joint in ``(y, u, t)``, so no
    value the solver holds is safe to invent with in general. See C-8.4.
    """

    def f_q(self, p: NDArray, u: NDArray, t: float) -> NDArray:
        """``f^q = ∂H/∂p``, from the momentum half alone."""
        ...

    def f_p(self, q: NDArray, u: NDArray, t: float) -> NDArray:
        """``f^p = -∂H/∂q``, from the position half, control and time."""
        ...


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

    Nor does any member establish that ``F`` is constant in time: ``LINEAR``
    permits ``M = M(t)``, so none may be read as
    :attr:`ProblemStructure.jacobian_constant` (C-15.1).
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
    with a nonzero ``F_uu``. For the same reason it **must never imply
    ``jacobian_constant``**: ``LINEAR`` also permits ``M = M(t)``, as
    ``tests/problems.py::LinearTimeVarying`` does.
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
