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
    "BilinearStateAffine",
    "ConstantJacobianQuadraticControl",
    "CoupledNonlinear",
    "FullCostObjective",
    "LinearTimeVarying",
    "ScalarAnchor",
    "StateDependentJacobian",
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


class ConstantJacobianQuadraticControl:
    """``y' = A y + B (u * u)``: ``F = A`` exactly, with real control curvature.

    The M6 factorization-reuse fixture. It is deliberately the *hardest*
    problem that still has a constant state Jacobian, and understanding why
    there is no harder one is the point.

    A constant ``F`` means ``f`` is affine in ``y``. That makes every implicit
    stage equation affine in its unknown, so Newton converges in one step and
    the stage matrix ``I - h a_ii A`` is the same at every stage of every step.
    Reuse is therefore exact, and it is exact for a *mathematical* reason, not
    a tolerance. There is no such thing as a constant-Jacobian problem that
    nonetheless needs several Newton iterations.

    What a constant ``F`` does **not** force is a trivial second-order path.
    Here ``f`` is quadratic in ``u``, so ``F_uu`` is nonzero and the
    Hessian-vector product exercises real curvature while ``F_yy`` and
    ``F_yu`` vanish. The control Jacobian ``G = 2 B diag(u)`` varies with
    ``u``, which keeps the tangent and second-order adjoint sweeps from
    degenerating into constants.

    ``A`` is non-normal and non-symmetric so that a transpose confusion cannot
    hide, and ``B`` is rectangular with ``n = 3``, ``nu = 2``.
    """

    state_dim = 3
    control_dim = 2

    A = np.array(
        [
            [-0.70, 1.30, 0.00],
            [-0.40, -0.25, 0.90],
            [0.15, -0.60, -1.10],
        ]
    )
    B = np.array([[1.0, 0.0], [0.30, 1.0], [-0.20, 0.45]])

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.A @ y + self.B @ (u * u)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.A

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return 2.0 * self.B * u[np.newaxis, :]

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((self.state_dim, self.state_dim))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((self.state_dim, self.control_dim))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``Σ_l v_l d²f_l/du²`` = ``2 diag(Bᵀ v)``, shape ``(ν, ν)``."""
        return 2.0 * np.diag(self.B.T @ v)


class StateDependentJacobian:
    """``y' = A y + c * y*y + B u``: ``F`` varies with the state, not with ``u``.

    The R-9 regression fixture. ``F = A + 2 c diag(y)`` depends on ``y`` alone,
    so it is *control-independent* and *time-independent* while still differing
    at every stage. A caller declaring ``jacobian_constant=True`` for it is
    making a false statement, and the historical failure mode is that nothing
    notices: the forward solve is unaffected, the method keeps its design
    order, and only the adjoint -- which reuses the factorization -- is wrong.
    """

    state_dim = 2
    control_dim = 2
    c = 0.6

    A = np.array([[-0.5, 0.9], [-0.7, -0.3]])
    B = np.array([[1.0, 0.2], [0.0, 1.0]])

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.A @ y + self.c * y * y + self.B @ u

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.A + 2.0 * self.c * np.diag(y)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.B

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return 2.0 * self.c * np.diag(v)

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((self.state_dim, self.control_dim))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((self.control_dim, self.control_dim))


class BilinearStateAffine:
    """``y' = (A + u_0 V) y + C u``: affine in the state, curved in ``(y, u)``.

    This fixture exists to reach the *second-order adjoint's* curvature skip,
    which the other affine fixtures cannot.

    The distinguishing property is the pattern of nonzero curvature blocks:

    ===============  =======  ====================================
    Block            Value    Reached by
    ===============  =======  ====================================
    ``F_yy``         zero     ``adjoint_sensitivity``
    ``F_yu``         nonzero  ``adjoint_sensitivity``
    ``F_uu``         zero     ``assemble_hessian_vector_product``
    ===============  =======  ====================================

    ``ConstantJacobianQuadraticControl`` has the opposite pattern -- nonzero
    ``F_uu`` and zero ``F_yu`` -- so a defect that wrongly skips the dynamics
    curvature in ``adjoint_sensitivity`` alone changes nothing for it, and a
    test built on it reports a pass. That gap was found by defect injection
    (M7-6) and is why both fixtures are needed.

    ``f`` is affine in ``y`` for fixed ``u``, so a stage equation is linear in
    its unknown and the direct affine solve applies and verifies. ``F`` depends
    on ``u``, so the Jacobian is *not* constant and no factorization may be
    reused across stages; that keeps this fixture from also depending on the
    M6 route.
    """

    state_dim = 3
    control_dim = 2

    A = np.array(
        [
            [-0.55, 0.90, 0.20],
            [-0.30, -0.40, 0.75],
            [0.10, -0.50, -0.95],
        ]
    )
    V = np.array(
        [
            [0.25, -0.40, 0.10],
            [0.60, 0.15, -0.30],
            [-0.20, 0.35, 0.45],
        ]
    )
    C = np.array([[1.0, 0.0], [0.30, 1.0], [-0.20, 0.45]])

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return (self.A + u[0] * self.V) @ y + self.C @ u

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.A + u[0] * self.V

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        out = self.C.copy()
        out[:, 0] = out[:, 0] + self.V @ y
        return out

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((self.state_dim, self.state_dim))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``[j, k] = sum_l v_l d2f_l/(dy_j du_k)``.

        Only ``k = 0`` contributes, and ``d2f_l/(dy_j du_0) = V[l, j]``, so
        column 0 is ``V^T v`` and column 1 is zero.
        """
        out = np.zeros((self.state_dim, self.control_dim))
        out[:, 0] = self.V.T @ v
        return out

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((self.control_dim, self.control_dim))
