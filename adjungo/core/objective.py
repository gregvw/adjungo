"""Objective function protocols."""

from typing import Protocol
from numpy.typing import NDArray


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
