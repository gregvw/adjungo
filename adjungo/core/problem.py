"""Problem specification protocols."""

from typing import Protocol, Optional
from enum import Enum, auto
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
    """Problem linearity classification."""
    LINEAR = auto()       # F independent of y, u
    BILINEAR = auto()     # F = H + uV
    QUASILINEAR = auto()  # F depends on u only
    SEMILINEAR = auto()   # F constant + nonlinear part
    NONLINEAR = auto()    # General


class ProblemStructure:
    """Deduced from problem or specified by user."""

    def __init__(
        self,
        linearity: Linearity,
        jacobian_constant: bool,
        jacobian_control_dependent: bool,
        has_second_derivatives: bool,
    ):
        self.linearity = linearity
        self.jacobian_constant = jacobian_constant
        self.jacobian_control_dependent = jacobian_control_dependent
        self.has_second_derivatives = has_second_derivatives
