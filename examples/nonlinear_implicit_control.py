"""Controlled Van der Pol oscillator solved with a fully implicit method.

Run directly::

    .venv/bin/python examples/nonlinear_implicit_control.py

This example is executed by ``tests/test_examples.py``.

Why this example exists
-----------------------

``minimum_energy_oscillator.py`` uses ``rk4`` on dynamics that are affine in
``(y, u)``. That combination exercises neither of the two things the implicit
routes are for: there are no stage equations to solve, and every second
derivative of ``f`` is identically zero.

Here the dynamics are genuinely nonlinear and the method is ``gauss2``, the
fully implicit family certified by milestone M3. Two consequences follow, and
they are what this example is for:

1. **The stages are coupled.** ``gauss2`` has a dense tableau ``A``, so its two
   stages cannot be solved one after another. Each step is one Newton solve on
   a coupled system of size ``s * n = 2 * 2 = 4``, not two solves of size 2.
2. **The Jacobian depends on the state.** ``df/dy`` contains ``-2 mu x v``, so
   it differs from stage to stage and from step to step. No factorization may
   be shared between stages: precedent R-9 in ``NUMERICS.md`` records a
   gradient error of 4.24e-05 produced by exactly that substitution, passing
   every test in the suite at the time because the population's Jacobian
   happened to depend only on ``u``. Reuse here is correctly refused, because
   the structure that permits it is not declared and is not true.

Physical problem
----------------

The Van der Pol oscillator with an applied force ``u(t)``. With state
``y = (x, v)``:

.. math::

    \\dot{x} &= v \\\\
    \\dot{v} &= \\mu (1 - x^2) v - x + u

The ``mu (1 - x^2) v`` term is negative damping for ``|x| < 1`` and positive
damping for ``|x| > 1``, which is what drives the system onto its limit cycle.
The task is to bring it from a point on that limit cycle to rest at the origin
by ``T``, spending as little control energy as possible:

.. math::

    J = \\tfrac{1}{2} (y(T) - y_\\text{target})^T Q_T (y(T) - y_\\text{target})
      + \\tfrac{1}{2} \\sum_{n,k} R\\, (u^n_k)^2

Scales
------

The equation is in its standard dimensionless form, so ``x``, ``v``, ``t`` and
``mu`` are pure numbers.

===================== ================= ==========================
Quantity              Value             Basis
===================== ================= ==========================
``mu``                1.0               nonlinear, not yet stiff
limit-cycle period    ~6.66             standard result for mu = 1
horizon ``T``         6.0               ~0.9 of one limit cycle
step ``h``            0.15              T/N with N = 40
===================== ================= ==========================

``mu = 1`` is deliberate. The relaxation limit ``mu >> 1`` is stiff, and
stiffness is what would motivate an implicit method on grounds of stability;
at ``mu = 1`` the method is instead being exercised on grounds of *coupling*,
which is the structural property this example is about. The mesh resolves the
limit cycle at roughly 44 steps per period.

What this example is not
------------------------

As in the linear example, the gradient is the exact derivative of the
**discrete** objective on this fixed mesh (C-2), not an approximation to the
derivative of the continuous problem (C-4). Those are validated separately.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from adjungo import GLMOptimizer
from adjungo.methods.runge_kutta import gauss2

MU = 1.0

T_FINAL = 6.0
N_STEPS = 40
Y0 = np.array([2.0, 0.0])  # on the limit cycle, at its rightmost excursion
Y_TARGET = np.array([0.0, 0.0])  # the unstable equilibrium

TERMINAL_WEIGHT = 50.0
CONTROL_WEIGHT = 0.01


class VanDerPolOscillator:
    """``y = (x, v)``; ``u`` is an applied force.

    Every callback below is exact. The second derivatives are not optional:
    per C-7 adjungo refuses rather than dropping them, because dropping them
    silently yields a Gauss-Newton operator while the public method still
    promises the exact Hessian. Unlike the affine example, here they are
    genuinely non-zero.
    """

    state_dim = 2
    control_dim = 1

    def __init__(self, mu: float = MU) -> None:
        if mu < 0.0:
            raise ValueError(f"mu must be non-negative, got {mu}")
        self.mu = mu

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        x, v = y[0], y[1]
        return np.array([v, self.mu * (1.0 - x * x) * v - x + u[0]])

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``df/dy``. Depends on the **state**, which is the whole point."""
        x, v = y[0], y[1]
        return np.array(
            [
                [0.0, 1.0],
                [-2.0 * self.mu * x * v - 1.0, self.mu * (1.0 - x * x)],
            ]
        )

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``df/du``."""
        return np.array([[0.0], [1.0]])

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, w: NDArray
    ) -> NDArray:
        """``sum_l w_l d2f_l/dy dy``.

        Only ``f_1`` has second derivatives in ``y``:
        ``d2f_1/dx2 = -2 mu v`` and ``d2f_1/dx dv = -2 mu x``, with
        ``d2f_1/dv2 = 0``. ``w`` is the multiplier contracted over the
        equation index.
        """
        x, v = y[0], y[1]
        return w[1] * np.array(
            [
                [-2.0 * self.mu * v, -2.0 * self.mu * x],
                [-2.0 * self.mu * x, 0.0],
            ]
        )

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, w: NDArray
    ) -> NDArray:
        """Zero: ``u`` enters additively, so there is no ``y u`` cross term."""
        return np.zeros((2, 1))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, w: NDArray
    ) -> NDArray:
        """Zero: ``f`` is affine in ``u``."""
        return np.zeros((1, 1))


class MinimumEnergyObjective:
    """Terminal state tracking plus quadratic control energy."""

    def __init__(
        self,
        y_target: NDArray = Y_TARGET,
        terminal_weight: float = TERMINAL_WEIGHT,
        control_weight: float = CONTROL_WEIGHT,
    ) -> None:
        self.y_target = np.asarray(y_target, dtype=float)
        self.terminal_weight = terminal_weight
        self.control_weight = control_weight

    def evaluate(self, trajectory, u: NDArray) -> float:
        miss = trajectory.Y[-1][0] - self.y_target
        return float(
            0.5 * self.terminal_weight * np.dot(miss, miss)
            + 0.5 * self.control_weight * np.sum(u**2)
        )

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        return self.terminal_weight * (y_final - self.y_target)

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros(2)

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self.control_weight * u_stage

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self.control_weight * np.eye(1)

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((2, 2))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return self.terminal_weight * np.eye(2)


def build_optimizer(n_steps: int = N_STEPS) -> GLMOptimizer:
    """Assemble the optimizer for the Van der Pol rendezvous problem."""
    return GLMOptimizer(
        problem=VanDerPolOscillator(),
        objective=MinimumEnergyObjective(),
        method=gauss2(),
        t_span=(0.0, T_FINAL),
        N=n_steps,
        y0=Y0,
    )


def solve(n_steps: int = N_STEPS):
    """Solve with ``trust-ncg`` and the exact Hessian-vector product."""
    optimizer = build_optimizer(n_steps)
    fun, jac = optimizer.scipy_interface()
    hessp = optimizer.scipy_hessp()
    u0 = np.zeros(n_steps * gauss2().s * 1)
    result = minimize(
        fun, u0, jac=jac, hessp=hessp, method="trust-ncg",
        options={"maxiter": 200},
    )
    return optimizer, result, result.x.reshape(n_steps, gauss2().s, 1)


def main() -> None:
    method = gauss2()
    optimizer, result, u_optimal = solve()

    print("Controlled Van der Pol oscillator, fully implicit (gauss2)")
    print("=" * 62)
    print(f"  stages s                 : {method.s}")
    print(f"  state dimension n        : {VanDerPolOscillator.state_dim}")
    print(
        "  coupled Newton system    : "
        f"{method.s * VanDerPolOscillator.state_dim} "
        "unknowns per step (s*n, dense A)"
    )
    print(f"  steps N                  : {N_STEPS}")
    print()

    zero = np.zeros_like(u_optimal)
    uncontrolled = optimizer.trajectory(zero).Y[-1][0]
    controlled = optimizer.trajectory(u_optimal).Y[-1][0]

    print(f"  converged                : {result.success}")
    print(f"  iterations               : {result.nit}")
    print(f"  J before optimisation    : {optimizer.objective_value(zero):.6f}")
    print(f"  J after  optimisation    : {result.fun:.6f}")
    print(f"  |grad J|_inf at optimum  : {np.max(np.abs(result.jac)):.3e}")
    print()
    print(
        "  final state, uncontrolled: "
        f"{np.array2string(uncontrolled, precision=4)}"
    )
    print(
        "  final state, controlled  : "
        f"{np.array2string(controlled, precision=4)}"
    )
    print(f"  target                   : {np.array2string(Y_TARGET, precision=4)}")
    print()
    print("Each step solved one coupled 4-unknown Newton system. The Jacobian")
    print("depends on the state, so no factorization is shared between stages;")
    print("see precedent R-9 in NUMERICS.md for what happens when one is.")


if __name__ == "__main__":
    main()
