"""Objective function protocols."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from numpy.typing import NDArray

if TYPE_CHECKING:
    from adjungo.stepping.trajectory import Trajectory


class Objective(Protocol):
    """Objective function for optimal control problem.

    ``evaluate`` is the only member specified on a layout. Every derivative
    callback below is per-step or per-stage and is handed its indices, so a
    plan whose steps carry different stage counts (C-18.5) reaches the
    gradient and the Hessian-vector product without any of them changing.

    Such a plan has no rectangular ``(N, s, ν)`` control array for
    ``evaluate``, and handing it the packed ``(Σ_n s_n, ν)`` array instead
    would not fail: an objective would index stage ``(n, k)`` out of whatever
    step happens to lie at packed row ``n`` and return a number. Supporting
    it is therefore opt-in -- see :class:`PackedObjective` -- rather than
    assumed from the fact that an implementation exists.
    """

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


@runtime_checkable
class PackedObjective(Protocol):
    """An :class:`Objective` that also evaluates on the packed layout.

    Implement this to make an objective usable on a plan whose steps carry
    different stage counts (C-18.5). Such a plan already reaches ``gradient``
    and ``hessian_vector_product``, whose callbacks are per-stage; what it
    lacks without this method is the scalar value, and therefore a line
    search and the SciPy adapters.

    It is a separate method rather than a flag because the arithmetic really
    is different. An objective with per-stage quadrature weights, such as
    ``einsum("k,nkv->", w, u**2)``, has no single ``w`` to apply once a plan
    mixes stage counts; it must read each step's own weights. Writing that
    is the declaration.

    Both methods must return the same number on any plan where both apply,
    and :func:`tests.test_discretization_plan` checks that they do on the
    degenerate uniform plan.
    """

    def evaluate_packed(self, trajectory: "Trajectory", u: NDArray) -> float:
        """
        Evaluate ``J(y, u)`` with stage controls in the packed layout.

        Args:
            trajectory: Solution trajectory. ``trajectory.plan`` carries the
                per-step offsets; ``trajectory.plan.stages(u, n)`` is step
                ``n``'s ``(s_n, ν)`` block.
            u: Packed stage controls, shape ``(Σ_n s_n, ν)``

        Returns:
            Objective value
        """
        ...
