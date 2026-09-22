"""Objective function protocols."""

from typing import TYPE_CHECKING, Protocol

from numpy.typing import NDArray

if TYPE_CHECKING:
    from adjungo.stepping.trajectory import Trajectory


class Objective(Protocol):
    """Objective function for optimal control problem."""

    def evaluate(self, trajectory: "Trajectory", u: NDArray) -> float:
        """
        Evaluate the objective J(y, u).

        Args:
            trajectory: Solution trajectory containing Y and Z
            u: Control array (N, s, ν)

        Returns:
            Objective value
        """
        ...

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        """
        Terminal cost derivative: ∂J/∂y(T).

        Args:
            y_final: Final state (r, n)

        Returns:
            Gradient w.r.t. final state (r, n)
        """
        ...

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        """
        Running cost derivative w.r.t. state: ∂J/∂y^[n].

        Args:
            y: External state (r, n)
            step: Time step index

        Returns:
            Gradient (r, n)
        """
        ...

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        """
        Running cost derivative w.r.t. control: ∂J/∂u_k^n.

        Args:
            u_stage: Control at stage k of step n (ν,)
            step: Time step index
            stage: Stage index

        Returns:
            Gradient (ν,)
        """
        ...

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        """
        Second derivative w.r.t. control: ∂²J/∂u_k^n∂u_k^n.

        Args:
            u_stage: Control at stage k of step n (ν,)
            step: Time step index
            stage: Stage index

        Returns:
            Hessian (ν, ν)
        """
        ...

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        """
        Running cost Hessian w.r.t. state: ∂²J/∂y^[n]∂y^[n].

        This is the second derivative of the *same* running term whose first
        derivative :meth:`dJ_dy` returns, evaluated at node ``step`` for
        ``step`` in ``0 .. N-1``. Node ``N`` is covered by
        :meth:`d2J_dy2_terminal` instead; see NUMERICS.md C-9.2.

        Required for :meth:`adjungo.optimization.interface.GLMOptimizer.\
hessian_vector_product`. Omitting it does not make the Hessian approximate;
        it makes the Hessian wrong, because the second-order adjoint is then
        forced by the wrong right-hand side.

        Args:
            y: External state (r, n)
            step: Time step index

        Returns:
            Hessian (n, n) acting on one external stage row
        """
        ...

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        """
        Terminal cost Hessian: ∂²J/∂y(T)².

        Supplies the terminal condition of the second-order adjoint,
        ``δλ^[N] = d2J_dy2_terminal(y^[N]) δy^[N]``. A zero terminal condition
        is correct only when the terminal cost is affine.

        Args:
            y_final: Final state (r, n)

        Returns:
            Hessian (n, n) acting on one external stage row
        """
        ...
