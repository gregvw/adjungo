"""Shared verification problems and objectives (C-14 test population).

Clause C-14 requires that the certification population jointly exercise every
structural feature whose absence can hide a derivative defect.  The dominant
historical example is finding B0: an incorrect stage-Jacobian index in the
adjoint stage solve was invisible for every problem in the old suite, because
each one had either ``s = 1`` (empty stage-coupling sum) or a constant
Jacobian (``F_i == F_j``).

:class:`CoupledNonlinear` and :class:`FullCostObjective` therefore satisfy all
of the following *at the same time*:

* ``s > 1`` is exercised by the methods the tests pair them with;
* the state Jacobian ``F`` varies with **both** state and control;
* ``n_x = 3`` and ``nu = 2``, so ``n_x != nu`` and neither is 1 (a shape
  transpose or a contraction over the wrong index cannot stay hidden);
* ``F_yy``, ``F_yu`` and ``F_uu`` are all nonzero;
* the objective has a terminal cost, a running state cost, **and** a running
  control cost, with pairwise-distinct diagonal weights;
* the tests using them set ``t_span[0] != 0`` and non-uniform controls, and
  the right-hand side depends explicitly on ``t``.

:class:`ScalarAnchor` and :class:`AnchorObjective` implement the closed-form
anchor of clause C-14.2: for ``y_1 = y_0 + h u`` and ``J = y_1^2 / 2`` the
exact discrete derivatives are ``dJ/du = h y_1`` and ``d2J/du2 v = h^2 v``,
with no truncation error and no reference implementation involved.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AnchorObjective",
    "CoupledNonlinear",
    "FullCostObjective",
    "LinearTimeVarying",
    "ScalarAnchor",
    "make_controls",
]


def make_controls(
    N: int, s: int, nu: int, seed: int = 0, scale: float = 0.5
) -> NDArray:
    """Non-uniform pseudo-random controls of shape ``(N, s, nu)``.

    Controls deliberately differ across steps, stages, and components. A
    control that is constant across stages cannot expose a defect in the
    per-stage control Jacobian.
    """
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((N, s, nu))


class ScalarAnchor:
    """``y' = u`` in one dimension: the closed-form anchor of C-14.2."""

    state_dim = 1
    control_dim = 1

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([u[0]])

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.zeros((1, 1))

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.ones((1, 1))

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((1, 1))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((1, 1))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((1, 1))


class AnchorObjective:
    """``J = y^[N]^2 / 2``; terminal cost only."""

    def evaluate(self, trajectory, u: NDArray) -> float:
        return 0.5 * float(trajectory.Y[-1, 0, 0] ** 2)

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        return np.array(y_final, dtype=float)

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros_like(y)

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros_like(u_stage)

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros((1, 1))

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((1, 1))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return np.ones((1, 1))


class LinearTimeVarying:
    """``y' = A(t) y + B u``: Jacobian varies with ``t`` but not with ``y``.

    Useful as a contrast case. Because ``F`` depends on the stage abscissa it
    differs between stages, so it still exposes a stage-index defect, but all
    second derivatives vanish.
    """

    state_dim = 2
    control_dim = 2

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.F(y, u, t) @ y + self.G(y, u, t) @ u

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[-0.3, 1.0 + 0.5 * t], [-0.8 * t, -0.2]])

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[1.0, 0.0], [0.25, 1.0]])

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((2, 2))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((2, 2))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((2, 2))


class CoupledNonlinear:
    """The C-14 workhorse: ``n_x = 3``, ``nu = 2``, all second derivatives live.

    .. math::

        f_0 &= y_1 y_2 + u_0 \\sin(y_0) + 0.3\\,t \\\\
        f_1 &= -\\sin(y_0) - 0.2 y_1 + u_1 y_2 \\\\
        f_2 &= 0.5 y_0 u_0 - 0.1 y_2 + u_1^2

    Every first and second derivative below is written out by hand from these
    formulas; none is obtained by differentiating package code. The class is
    therefore usable as an input to both the package and the independent
    reference without creating a shared-error channel.
    """

    state_dim = 3
    control_dim = 2

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array(
            [
                y[1] * y[2] + u[0] * np.sin(y[0]) + 0.3 * t,
                -np.sin(y[0]) - 0.2 * y[1] + u[1] * y[2],
                0.5 * y[0] * u[0] - 0.1 * y[2] + u[1] ** 2,
            ]
        )

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array(
            [
                [u[0] * np.cos(y[0]), y[2], y[1]],
                [-np.cos(y[0]), -0.2, u[1]],
                [0.5 * u[0], 0.0, -0.1],
            ]
        )

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array(
            [
                [np.sin(y[0]), 0.0],
                [0.0, y[2]],
                [0.5 * y[0], 2.0 * u[1]],
            ]
        )

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2 f_l / dy dy``."""
        d2f0 = np.array(
            [
                [-u[0] * np.sin(y[0]), 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        d2f1 = np.zeros((3, 3))
        d2f1[0, 0] = np.sin(y[0])
        # f_2 is affine in y, so its state Hessian vanishes.
        return v[0] * d2f0 + v[1] * d2f1

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2 f_l / dy du``, shape ``(n_x, nu)``."""
        d2f0 = np.zeros((3, 2))
        d2f0[0, 0] = np.cos(y[0])
        d2f1 = np.zeros((3, 2))
        d2f1[2, 1] = 1.0
        d2f2 = np.zeros((3, 2))
        d2f2[0, 0] = 0.5
        return v[0] * d2f0 + v[1] * d2f1 + v[2] * d2f2

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2 f_l / du du``, shape ``(nu, nu)``."""
        d2f2 = np.zeros((2, 2))
        d2f2[1, 1] = 2.0
        return v[2] * d2f2


class FullCostObjective:
    """Terminal, running-state, and running-control costs with distinct weights.

    .. math::

        J = \\tfrac12 (y^{[N]} - y_t)^T Q_T (y^{[N]} - y_t)
          + \\tfrac12 \\sum_{n=0}^{N-1} (y^{[n]})^T Q\\, y^{[n]}
          + \\tfrac12 \\sum_{n,k} (u^n_k)^T R\\, u^n_k

    The running state sum covers steps ``0 .. N-1`` and the terminal term
    covers step ``N``. That split is not a modelling preference; it is what
    :func:`adjungo.stepping.adjoint.adjoint_solve` differentiates, and the
    objective must state the same function the adjoint assumes.

    All weights are distinct so that a transposed or mis-indexed weight matrix
    changes the answer.
    """

    def __init__(
        self,
        nx: int = 3,
        nu: int = 2,
        q_terminal: NDArray | None = None,
        q_running: NDArray | None = None,
        r_control: NDArray | None = None,
        y_target: NDArray | None = None,
    ) -> None:
        self.nx = nx
        self.nu = nu
        self.Q_T = (
            np.diag(1.0 + np.arange(nx, dtype=float))
            if q_terminal is None
            else np.asarray(q_terminal, dtype=float)
        )
        self.Q = (
            np.diag(0.05 * (3.0 - np.arange(nx, dtype=float)))
            if q_running is None
            else np.asarray(q_running, dtype=float)
        )
        self.R = (
            np.diag(0.07 + 0.11 * np.arange(nu, dtype=float))
            if r_control is None
            else np.asarray(r_control, dtype=float)
        )
        self.y_target = (
            np.linspace(0.2, -0.3, nx)
            if y_target is None
            else np.asarray(y_target, dtype=float)
        )

    def evaluate(self, trajectory, u: NDArray) -> float:
        Y = trajectory.Y
        N = Y.shape[0] - 1
        d = Y[N, 0] - self.y_target
        total = 0.5 * float(d @ self.Q_T @ d)
        for n in range(N):
            total += 0.5 * float(Y[n, 0] @ self.Q @ Y[n, 0])
        for n in range(u.shape[0]):
            for k in range(u.shape[1]):
                total += 0.5 * float(u[n, k] @ self.R @ u[n, k])
        return total

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        out = np.zeros_like(np.asarray(y_final, dtype=float))
        out[0] = self.Q_T @ (np.asarray(y_final)[0] - self.y_target)
        return out

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        out = np.zeros_like(np.asarray(y, dtype=float))
        out[0] = self.Q @ np.asarray(y)[0]
        return out

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self.R @ np.asarray(u_stage, dtype=float)

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self.R.copy()

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return self.Q.copy()

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return self.Q_T.copy()
