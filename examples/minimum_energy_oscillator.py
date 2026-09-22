"""Minimum-energy rendezvous for a damped mass-spring oscillator.

Run directly::

    .venv/bin/python examples/minimum_energy_oscillator.py

This example is executed by ``tests/test_examples.py``, so it cannot rot
silently: if the public API changes, the test fails rather than the README
quietly becoming wrong.

Physical problem
----------------

A mass ``m`` on a spring of stiffness ``k`` with linear damping ``c`` is driven
by a force ``u(t)``. With state ``y = (x, v)`` for position and velocity,

.. math::

    \\dot{x} &= v \\\\
    \\dot{v} &= \\bigl(u - k x - c v\\bigr) / m

The task is to bring the mass from rest at ``x = 1 m`` to rest at the origin
at ``T = 4 s`` while spending as little control energy as possible:

.. math::

    J = \\tfrac{1}{2} (y(T) - y_\\text{target})^T Q_T (y(T) - y_\\text{target})
      + \\tfrac{1}{2} \\sum_{n,k} R\\, (u^n_k)^2

Units and scales
----------------

All quantities are SI. The characteristic scales are

===================== ================= ==========================
Quantity              Value             Basis
===================== ================= ==========================
mass ``m``            1.0 kg            chosen
stiffness ``k``       4.0 N/m           gives ω₀ = 2 rad/s
damping ``c``         0.4 N·s/m         ζ = c/(2√(km)) = 0.1, underdamped
natural period        π s               2π/ω₀
horizon ``T``         4 s               ≈ 1.27 natural periods
step ``h``            0.05 s            T/N with N = 80; h·ω₀ = 0.1
===================== ================= ==========================

``h ω₀ = 0.1`` keeps roughly 60 steps per natural period, so the mesh resolves
the oscillation and the reported optimum is a property of the dynamics rather
than of the discretisation. The damping ratio ``ζ = 0.1`` is well inside the
underdamped regime, so the uncontrolled system oscillates and the control has
something to do.

What this example is not
------------------------

The gradient adjungo returns is the **exact derivative of the discrete
objective at this fixed mesh**, not an approximation of the derivative of the
continuous problem. Those are different quantities and are validated
separately; see NUMERICS.md clauses C-2 and C-4. Refining ``N`` changes the
discrete problem being solved, and therefore changes the exact gradient.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from adjungo import GLMOptimizer
from adjungo.methods.runge_kutta import rk4

MASS = 1.0  # kg
STIFFNESS = 4.0  # N/m
DAMPING = 0.4  # N.s/m

T_FINAL = 4.0  # s
N_STEPS = 80
Y0 = np.array([1.0, 0.0])  # (x [m], v [m/s]) -- released from rest at x = 1 m
Y_TARGET = np.array([0.0, 0.0])  # rest at the origin

TERMINAL_WEIGHT = 100.0  # penalty on missing the target state
CONTROL_WEIGHT = 0.01  # penalty on control energy


class DampedOscillator:
    """``y = (x, v)``; ``u`` is an applied force in newtons."""

    state_dim = 2
    control_dim = 1

    def __init__(
        self,
        mass: float = MASS,
        stiffness: float = STIFFNESS,
        damping: float = DAMPING,
    ) -> None:
        if mass <= 0.0:
            raise ValueError(f"mass must be positive, got {mass}")
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

    @property
    def natural_frequency(self) -> float:
        """``omega_0 = sqrt(k/m)`` in rad/s."""
        return float(np.sqrt(self.stiffness / self.mass))

    @property
    def damping_ratio(self) -> float:
        """``zeta = c / (2 sqrt(k m))``; underdamped when ``< 1``."""
        return float(
            self.damping / (2.0 * np.sqrt(self.stiffness * self.mass))
        )

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        x, v = y[0], y[1]
        return np.array(
            [v, (u[0] - self.stiffness * x - self.damping * v) / self.mass]
        )

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``df/dy``. Constant here, but still a function of the arguments."""
        return np.array(
            [
                [0.0, 1.0],
                [-self.stiffness / self.mass, -self.damping / self.mass],
            ]
        )

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``df/du``."""
        return np.array([[0.0], [1.0 / self.mass]])

    # ------------------------------------------------------------------
    # Second derivatives, required by the exact Hessian-vector product.
    #
    # These are all exactly zero here, and that is a mathematical fact
    # about this problem rather than a convenient simplification: ``f`` is
    # affine in ``(y, u)``, so every second partial derivative vanishes
    # identically.
    #
    # They must still be written. adjungo cannot infer that ``f`` is affine
    # from the callbacks it is given, and NUMERICS.md C-7 forbids guessing:
    # a missing second-derivative callback raises rather than being treated
    # as zero, because for a nonlinear problem that substitution silently
    # returns a Gauss-Newton operator while the public method still promises
    # the exact Hessian.
    #
    # ``v`` is the adjoint multiplier and is contracted over the *equation*
    # index, so the returned matrices have the shapes below regardless of
    # ``v``.
    # ------------------------------------------------------------------

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2f_l/dy dy``; zero because ``f`` is affine in ``y``."""
        return np.zeros((2, 2))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2f_l/dy du``; zero because ``f`` has no ``y u`` term."""
        return np.zeros((2, 1))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2f_l/du du``; zero because ``f`` is affine in ``u``."""
        return np.zeros((1, 1))


class MinimumEnergyObjective:
    """Terminal state tracking plus quadratic control energy.

    Every derivative the certified path needs is supplied. ``d2J_dy2`` and
    ``d2J_dy2_terminal`` are **not optional** for
    :meth:`~adjungo.optimization.interface.GLMOptimizer.hessian_vector_product`:
    omitting them does not make the Hessian approximate, it makes it wrong,
    because the second-order adjoint is then driven by the wrong right-hand
    side. adjungo refuses rather than substituting a Gauss-Newton operator.
    """

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
        return np.zeros(2)  # no running state cost

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self.control_weight * u_stage

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self.control_weight * np.eye(1)

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((2, 2))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return self.terminal_weight * np.eye(2)


def build_optimizer(n_steps: int = N_STEPS) -> GLMOptimizer:
    """Assemble the optimizer for the rendezvous problem."""
    return GLMOptimizer(
        problem=DampedOscillator(),
        objective=MinimumEnergyObjective(),
        method=rk4(),
        t_span=(0.0, T_FINAL),
        N=n_steps,
        y0=Y0,
    )


def solve(n_steps: int = N_STEPS, use_hessian: bool = True):
    """Solve the problem and return ``(optimizer, result, u_optimal)``.

    ``use_hessian`` selects ``trust-ncg`` with the exact Hessian-vector
    product over ``L-BFGS-B`` with the gradient alone.
    """
    optimizer = build_optimizer(n_steps)
    fun, jac = optimizer.scipy_interface()
    u0 = np.zeros(n_steps * rk4().s * 1)

    if use_hessian:
        result = minimize(
            fun,
            u0,
            jac=jac,
            hessp=optimizer.scipy_hessp(),
            method="trust-ncg",
            options={"gtol": 1e-10, "maxiter": 200},
        )
    else:
        result = minimize(
            fun,
            u0,
            jac=jac,
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 500},
        )

    u_optimal = result.x.reshape(n_steps, rk4().s, 1)
    return optimizer, result, u_optimal


def main() -> None:
    problem = DampedOscillator()
    h = T_FINAL / N_STEPS

    print("Minimum-energy rendezvous for a damped mass-spring oscillator")
    print("=" * 62)
    print(f"  natural frequency  omega_0 = {problem.natural_frequency:.4f} rad/s")
    print(f"  damping ratio      zeta    = {problem.damping_ratio:.4f} (underdamped)")
    print(f"  natural period             = {2*np.pi/problem.natural_frequency:.4f} s")
    print(f"  horizon T                  = {T_FINAL} s")
    print(f"  step h                     = {h:.4f} s   (h*omega_0 = {h*problem.natural_frequency:.3f})")
    print()

    optimizer, result, u_optimal = solve()
    y_final = optimizer.trajectory(u_optimal).Y[-1][0]

    print(f"  converged                  : {result.success}")
    print(f"  iterations                 : {result.nit}")
    print(f"  objective J                : {result.fun:.10e}")
    print(f"  |grad J|_inf               : {np.max(np.abs(result.jac)):.3e}")
    print()
    print(f"  final position x(T)        : {y_final[0]: .6e} m")
    print(f"  final velocity v(T)        : {y_final[1]: .6e} m/s")
    print(f"  peak control               : {np.max(np.abs(u_optimal)):.6f} N")
    print()

    uncontrolled = optimizer.trajectory(np.zeros_like(u_optimal)).Y[-1][0]
    print(f"  uncontrolled x(T)          : {uncontrolled[0]: .6e} m")
    print(f"  uncontrolled v(T)          : {uncontrolled[1]: .6e} m/s")
    print()
    print("The gradient driving this solve is the exact derivative of the")
    print("discrete objective at this mesh, not a finite-difference or")
    print("continuous-adjoint approximation (NUMERICS.md C-2).")


if __name__ == "__main__":
    main()
