"""Pendulum swing-up: a nonlinear problem that can be checked against a closed form.

Run directly::

    .venv/bin/python -m examples.pendulum_swing_up

It imports a sibling (``examples.symbolic``), so running it by path puts
``examples/`` on ``sys.path`` rather than the repository root and the import
fails. ``-m`` resolves the package from the root, which is also how the tests
reach it.

The problem
-----------

A torque-driven pendulum, state ``y = (θ, ω)`` and control ``u`` the applied
torque per unit moment of inertia:

.. math::

    \\dot θ = ω, \\qquad \\dot ω = -ω_0^2 \\sin θ - c\\, ω + u.

It starts hanging at rest, ``θ = 0``, and is asked to finish inverted,
``θ = π``, ``ω = 0``, paying a quadratic terminal miss and the C-9.3 stage
quadrature of the control energy. The derivative callbacks are differentiated
from ``f`` by sympy through :mod:`examples.symbolic`, not written by hand.

Why this problem earns its place
--------------------------------

The other nonlinear example here, ``nonlinear_implicit_control.py``, already
demonstrates coupled stages and Newton on a Van der Pol oscillator. What it
cannot do is say what the right answer is: a Van der Pol trajectory has no
closed form, so it is checked only against the independently assembled
discrete reference, the first tier of the C-14.1 hierarchy.

The pendulum admits the second tier for a *nonlinear* vector field, which no
example here previously had. ``rocket_ascent.py`` has a closed form only under
a constant burn, and ``double_integrator.py``'s is linear.

**A closed-form nonlinear trajectory.** With no torque and no drag the
pendulum librates, and the motion from rest at amplitude ``θ₀`` is exactly

.. math::

    θ(t) = 2\\arcsin\\!\\big(k\\,\\operatorname{sn}(K(m) - ω_0 t,\\; m)\\big),
    \\qquad ω(t) = -2 ω_0 k \\operatorname{cn}(K(m) - ω_0 t,\\; m),

with ``k = sin(θ₀/2)`` and ``m = k²``; ``sn`` and ``cn`` are Jacobi elliptic
functions and ``K`` the complete elliptic integral of the first kind. This is
a genuine closed form for a genuinely nonlinear motion: at the amplitude used
here the period ``4K(m)/ω₀`` exceeds the small-angle ``2π/ω₀`` by a third, so
a solve that quietly linearised the sine would miss it by far more than its
discretisation error. :func:`libration` supplies it, and the forward solve
converges to it at the method's order.

**A conserved first integral.** The undriven, undamped pendulum conserves

.. math::  E = \\tfrac12 ω^2 + ω_0^2 (1 - \\cos θ),

which the closed form satisfies identically, since
``sn² + cn² = 1`` turns ``E`` into ``2ω_0^2k^2``. This is an oracle of a
different kind from a pointwise comparison: it constrains every point of every
undriven trajectory rather than one endpoint.

It also distinguishes the methods, and the example measures that rather than
asserting it. ``rk4`` does not conserve ``E``; its error grows in proportion
to the elapsed time. ``gauss2`` is Gauss--Legendre collocation and therefore
symplectic (Hairer, Lubich & Wanner, *Geometric Numerical Integration*,
Thm. VI.4.2), so its energy error stays bounded however long the integration
runs. Measured the same way for both -- the largest energy error the run ever
attains -- a thirty-two-fold increase in the integration time multiplies that
error by about 27 for ``rk4`` and by 1.003 for ``gauss2``.

Both facts describe the *uncontrolled, undamped* pendulum. They validate the
vector field and the stepping, not the optimum; the controlled problem below
carries drag and torque, and conserves nothing. Keeping the validation
configuration distinct from the solved one is deliberate.

What is not claimed
-------------------

The swing-up optimum has no closed form and none is asserted. Its derivatives
are checked against the independent discrete reference, and the solve is
reported, not certified. Nothing here imposes a torque bound: the control is
limited only by the energy weight, so this is not a bounded-torque swing-up.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import ellipj, ellipk

from adjungo.methods.runge_kutta import gauss2
from adjungo.optimization.interface import GLMOptimizer
from examples.symbolic import SymbolicDynamics

__all__ = [
    "NATURAL_FREQUENCY",
    "SwingUpCost",
    "build_optimizer",
    "energy",
    "libration",
    "libration_period",
    "pendulum_dynamics",
    "solve",
    "stage_times",
]

#: ``ω₀ = sqrt(g / L)``. Fixes the time scale; no other constant depends on it.
NATURAL_FREQUENCY = 1.3

#: Linear viscous damping in the *controlled* problem. The validation
#: configuration sets this to zero, which is what makes ``E`` conserved.
DRAG = 0.25

T_FINAL = 6.0
N_STEPS = 20
Y0 = np.array([0.0, 0.0])
TARGET = np.array([np.pi, 0.0])
TERMINAL_WEIGHT = np.diag([10.0, 1.0])
ENERGY_WEIGHT = 0.05

#: Release amplitude for the closed-form checks, in radians. Chosen well
#: outside the small-angle regime -- see :func:`libration_period`.
VALIDATION_AMPLITUDE = 2.0


def pendulum_dynamics(drag: float = DRAG) -> SymbolicDynamics:
    """``f = (ω, -ω₀² sin θ - c ω + u)``, with its derivatives differentiated.

    Only ``f`` is written down. ``F``, ``G`` and the three contracted second
    derivatives are produced by :class:`examples.symbolic.SymbolicDynamics`;
    ``tests/test_examples.py`` checks all five against derivatives taken by
    hand, which for this field are short enough to write out independently.
    """
    theta, omega, torque, time = sp.symbols("theta omega tau t")
    f = [
        omega,
        -(NATURAL_FREQUENCY**2) * sp.sin(theta) - drag * omega + torque,
    ]
    return SymbolicDynamics(f, (theta, omega), (torque,), time)


# ---------------------------------------------------------------------------
# The closed form, and the invariant
# ---------------------------------------------------------------------------


def libration(t: NDArray | float, amplitude: float = VALIDATION_AMPLITUDE):
    """Exact ``(θ, ω)`` of the undriven, undamped pendulum released from rest.

    Released from rest at ``amplitude``, the pendulum librates with
    ``θ(t) = 2 arcsin(k sn(K - ω₀t, m))``, ``k = sin(amplitude/2)``, ``m = k²``.
    Differentiating and using ``dn = sqrt(1 - m sn²)`` collapses the velocity
    to ``ω(t) = -2 ω₀ k cn(K - ω₀t, m)``, with no square root left to cancel.

    At ``t = 0`` this gives ``sn(K) = 1`` and ``cn(K) = 0``, so ``θ = amplitude``
    and ``ω = 0`` as required. Valid for ``|amplitude| < π``; at ``π`` the
    modulus reaches one, the period diverges and the motion is the separatrix.
    """
    if not 0.0 < amplitude < np.pi:
        raise ValueError(
            f"amplitude must lie in (0, pi) for a librating solution, got "
            f"{amplitude}. At pi the pendulum is on the separatrix and the "
            f"period is infinite."
        )
    k = np.sin(amplitude / 2.0)
    m = k * k
    argument = ellipk(m) - NATURAL_FREQUENCY * np.asarray(t, dtype=float)
    sn, cn, _, _ = ellipj(argument, m)
    return 2.0 * np.arcsin(k * sn), -2.0 * NATURAL_FREQUENCY * k * cn


def libration_period(amplitude: float = VALIDATION_AMPLITUDE) -> float:
    """``4 K(m) / ω₀``, the exact period of the undriven libration.

    Compare with the small-angle ``2π/ω₀``, which it exceeds by a third at
    :data:`VALIDATION_AMPLITUDE`. That gap is what makes the closed-form
    comparison a test of the nonlinear field rather than of its linearisation.
    """
    return 4.0 * ellipk(np.sin(amplitude / 2.0) ** 2) / NATURAL_FREQUENCY


def energy(y: NDArray) -> float:
    """``½ω² + ω₀²(1 - cos θ)``, conserved when drag and torque both vanish.

    Kinetic plus potential per unit moment of inertia, with the zero at the
    hanging state. Constant along any trajectory of ``pendulum_dynamics(0.0)``
    driven by zero torque, and equal to ``2ω₀²sin²(amplitude/2)`` on the
    libration above.
    """
    y = np.asarray(y, dtype=float)
    return float(
        0.5 * y[1] ** 2 + NATURAL_FREQUENCY**2 * (1.0 - np.cos(y[0]))
    )


# ---------------------------------------------------------------------------
# The controlled problem
# ---------------------------------------------------------------------------


class SwingUpCost:
    """``½(y_N - target)ᵀ S (y_N - target) + ½ R h Σ w_k u²``.

    The step size and stage weights are carried here rather than by the
    library, which applies ``dJ_du`` verbatim; see C-9.3.
    """

    def __init__(
        self,
        stage_weights: NDArray,
        step_size: float,
        target: NDArray = TARGET,
        energy_weight: float = ENERGY_WEIGHT,
    ) -> None:
        self.stage_weights = np.asarray(stage_weights, dtype=float)
        self.step_size = step_size
        self.target = np.asarray(target, dtype=float)
        self.energy_weight = energy_weight

    def _quadrature_factor(self, stage: int) -> float:
        return self.step_size * float(self.stage_weights[stage]) * self.energy_weight

    def evaluate(self, trajectory, u: NDArray) -> float:
        miss = trajectory.Y[-1][0] - self.target
        stage_energy = np.einsum("k,nkv->", self.stage_weights, u**2)
        return float(
            0.5 * miss @ TERMINAL_WEIGHT @ miss
            + 0.5 * self.energy_weight * self.step_size * stage_energy
        )

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        gradient = np.zeros_like(np.asarray(y_final, dtype=float))
        gradient[0] = TERMINAL_WEIGHT @ (
            np.asarray(y_final, dtype=float)[0] - self.target
        )
        return gradient

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros_like(np.asarray(y, dtype=float))

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self._quadrature_factor(stage) * u_stage

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self._quadrature_factor(stage) * np.eye(1)

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((2, 2))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return TERMINAL_WEIGHT.astype(float).copy()


def build_optimizer(
    n_steps: int = N_STEPS,
    drag: float = DRAG,
    method_factory=gauss2,
) -> GLMOptimizer:
    """Assemble the swing-up on ``gauss2``: one coupled ``(s·n)`` Newton solve per step.

    The Jacobian ``F`` carries ``-ω₀² cos θ``, so it depends on the state and
    no factorisation is reused, within a solve or across calls (C-15.1).
    """
    method = method_factory()
    step_size = T_FINAL / n_steps
    return GLMOptimizer(
        pendulum_dynamics(drag),
        SwingUpCost(method.B[0, :], step_size),
        method,
        t_span=(0.0, T_FINAL),
        N=n_steps,
        y0=Y0,
    )


def solve(n_steps: int = N_STEPS, drag: float = DRAG):
    """Minimise from zero torque with ``trust-ncg``, using the exact Hessian action."""
    optimizer = build_optimizer(n_steps, drag)
    stages = optimizer.method.s
    fun, jac = optimizer.scipy_interface()
    result = minimize(
        fun,
        np.zeros(n_steps * stages),
        jac=jac,
        hessp=optimizer.scipy_hessp(),
        method="trust-ncg",
        options={"gtol": 1e-10, "maxiter": 600},
    )
    return optimizer, result, result.x.reshape(n_steps, stages, 1)


def stage_times(n_steps: int = N_STEPS, method_factory=gauss2) -> NDArray:
    """Absolute time at every stage, shape ``(n_steps, s)``."""
    method = method_factory()
    h = T_FINAL / n_steps
    return np.arange(n_steps)[:, None] * h + h * np.asarray(method.c)[None, :]


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    period = libration_period()
    print(f"Undriven libration at amplitude {VALIDATION_AMPLITUDE} rad:")
    print(f"  exact period      {period:.6f} s")
    print(f"  small-angle 2pi/w0 {2 * np.pi / NATURAL_FREQUENCY:.6f} s"
          f"   ({period / (2 * np.pi / NATURAL_FREQUENCY):.3f}x longer)")
    print(f"  invariant E       {energy(np.array([VALIDATION_AMPLITUDE, 0.0])):.9f}")

    print("\nForward solve against the closed form (undriven, undamped):")
    previous = None
    for n_steps in (10, 20, 40, 80):
        error = _libration_error(n_steps)
        rate = "" if previous is None else f"   rate {np.log2(previous / error):.2f}"
        print(f"  N = {n_steps:3d}   max error {error:.3e}{rate}")
        previous = error

    print("\nEnergy drift over 32 periods, 20 steps per period:")
    for label, ratio, kind in _energy_contrast():
        print(f"  {label:7s} {kind:22s} grows by {ratio:6.2f}x")

    optimizer, result, control = solve()
    final = optimizer.trajectory(control).Y[-1][0]
    print(f"\nSwing-up on gauss2, N = {N_STEPS}, T = {T_FINAL} s:")
    print(f"  objective        {result.fun:.9f}")
    print(f"  theta(T)         {final[0]:.6f} rad   (target {np.pi:.6f})")
    print(f"  omega(T)         {final[1]:.6f} rad/s (target 0)")
    print(f"  peak |torque|    {np.max(np.abs(control)):.6f}")
    print(f"  iterations       {result.nit}")


def _libration_error(n_steps: int) -> float:
    """Max error of the undriven solve against :func:`libration`, over the whole arc.

    Compared at every step node across half a period, not at the endpoint
    alone. A half period ends at the opposite turning point, where
    ``dθ/dt = 0``, and sampling only there corrupts the measured order in
    either direction: ``θ`` alone converges at 4.92, 4.98, 5.00 over the
    meshes below, because a phase error reaches it only at second order, while
    ``max(θ, ω)`` gives 4.89, 3.49, 3.80. Over the whole arc the rates are
    4.11, 4.08, 4.04, which is the method's order.
    """
    from adjungo.methods.runge_kutta import rk4

    horizon = 0.5 * libration_period()
    method = rk4()
    optimizer = GLMOptimizer(
        pendulum_dynamics(drag=0.0),
        SwingUpCost(method.B[0, :], horizon / n_steps),
        method,
        t_span=(0.0, horizon),
        N=n_steps,
        y0=np.array([VALIDATION_AMPLITUDE, 0.0]),
    )
    trajectory = optimizer.trajectory(np.zeros((n_steps, method.s, 1))).Y
    nodes = np.linspace(0.0, horizon, n_steps + 1)
    theta_exact, omega_exact = libration(nodes)
    theta = np.array([trajectory[i][0][0] for i in range(n_steps + 1)])
    omega = np.array([trajectory[i][0][1] for i in range(n_steps + 1)])
    return max(
        float(np.max(np.abs(theta - theta_exact))),
        float(np.max(np.abs(omega - omega_exact))),
    )


def _energy_contrast():
    """Growth of the energy error from one period to 32, for both methods.

    One statistic for both methods: ``max|E(t) - E(0)|`` over the whole
    integration, the largest energy error the run ever attains. Measuring rk4
    by its endpoint error and gauss2 by the spread of its oscillation would
    compare two different quantities, and the contrast could then be an
    artefact of that choice rather than of the methods.
    """
    from adjungo.methods.runge_kutta import rk4

    period = libration_period()
    exact = energy(np.array([VALIDATION_AMPLITUDE, 0.0]))
    results = []
    for label, factory, kind in (
        ("rk4", rk4, "secular drift"),
        ("gauss2", gauss2, "bounded oscillation"),
    ):
        measured = [
            float(np.max(np.abs(_energy_trace(factory, p * period, 20 * p) - exact)))
            for p in (1, 32)
        ]
        results.append((label, measured[1] / measured[0], kind))
    return results


def _energy_trace(method_factory, horizon: float, n_steps: int) -> NDArray:
    """``E`` at every step node of an undriven, undamped solve."""
    method = method_factory()
    optimizer = GLMOptimizer(
        pendulum_dynamics(drag=0.0),
        SwingUpCost(method.B[0, :], horizon / n_steps),
        method,
        t_span=(0.0, horizon),
        N=n_steps,
        y0=np.array([VALIDATION_AMPLITUDE, 0.0]),
    )
    trajectory = optimizer.trajectory(np.zeros((n_steps, method.s, 1)))
    return np.array(
        [energy(trajectory.Y[i][0]) for i in range(n_steps + 1)]
    )


if __name__ == "__main__":
    main()
