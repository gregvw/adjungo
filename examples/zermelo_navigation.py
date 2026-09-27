"""Zermelo navigation: steering through a current, with a closed-form optimum.

Run directly::

    .venv/bin/python -m examples.zermelo_navigation

It imports a sibling (``examples.symbolic``), so running it by path puts
``examples/`` on ``sys.path`` rather than the repository root and the import
fails. ``-m`` resolves the package from the root, which is also how the tests
reach it.

The problem
-----------

A boat holds constant speed ``V`` through the water and steers by choosing its
heading ``θ``. The water itself moves: a linear shear flowing in ``+x`` at
speed ``V y / h``, at rest along ``y = 0``. With state ``y = (x, y)`` and
control ``u = θ``,

.. math::

    \\dot x = V\\cos θ + V y / h, \\qquad \\dot y = V\\sin θ.

Zermelo (1931) asked for the quickest path to a point. This example asks the
fixed-time question instead -- **maximise ``x(T)``** -- because the library
integrates a fixed horizon on a fixed mesh and has no free-final-time
parameter. Both problems have the same navigation formula and the same
character: head across the current early, to reach faster water, then turn
downstream and be carried. The objective is ``J = -x(T)``.

What this example adds
----------------------

Every other example here is **linear in its control**: the torque, the thrust
and the acceleration all enter ``f`` additively, so ``∂²f/∂u²`` vanishes
identically and the contracted second derivative ``F_uu_action`` returns zero
everywhere. This one is not. The heading enters through ``cos θ`` and
``sin θ``, so

.. math::

    F_{uu}(v) = \\sum_l v_l \\frac{∂^2 f_l}{∂θ^2}
              = -V\\,(v_0 \\cos θ + v_1 \\sin θ),

which is the first nonzero ``F_uu`` any example drives.

It is also the *only* nonzero second derivative here. ``f`` is linear in the
state, so ``F_yy = 0``; the state enters ``f`` only through the current, which
does not involve ``θ``, so ``F_yu = 0``; and the cost is terminal and linear,
so every ``d2J`` block is zero too. The whole Hessian of this problem is
dynamics curvature in the control. Zeroing ``F_uu_action`` does not perturb
the Hessian-vector product here -- it annihilates it.

The closed form
---------------

Pontryagin's conditions are unusually simple because ``∂f/∂y`` is a constant
matrix. With ``H = λ·f``,

.. math::

    \\dot λ_x = 0, \\qquad \\dot λ_y = -λ_x V/h, \\qquad λ(T) = (-1, 0),

so ``λ_x ≡ -1`` and ``λ_y(t) = -(V/h)(T - t)``, exactly, with no reference to
the state. Stationarity ``∂H/∂θ = 0`` then gives

.. math::

    \\tan θ^*(t) = (V/h)\\,(T - t),

and ``∂²H/∂θ² = V\\cos θ + (V/h)(T-t)\\,V\\sin θ > 0`` on ``(-π/2, π/2)``, so
this minimises ``H`` as the minimum principle requires. Differentiating gives
Zermelo's navigation formula ``\\dot θ = -(V/h)\\cos^2 θ``, which is what the
general formula reduces to for a current with ``∂u_c/∂y = V/h`` and every
other derivative zero.

Substituting ``θ^*`` and integrating in ``s = (V/h)(T-t)`` closes the
trajectory in elementary functions; see :func:`optimal_position`.

A sharper oracle than usual
---------------------------

Because the costate above never consults the state, **no discretisation error
can reach it**. Working the discrete adjoint out by hand: with ``M = ∂f/∂y``
constant and ``Mᵀv = (0, (V/h)v_x)``, the ``x``-component of every adjoint
stays at its terminal ``-1``, and the stage adjoint carries

.. math::

    Λ_{n,i,y} = λ_{n+1,y} - h\\,(V/h)\\,\\frac{\\sum_j b_j a_{ji}}{b_i},
    \\qquad
    λ_{n,y} = λ_{n+1,y} - h\\,(V/h).

The nodal value is therefore exact for any consistent tableau. The *stage*
value matches ``λ_y(t_n + c_i h)`` if and only if

.. math::

    \\sum_j b_j a_{ji} = b_i\\,(1 - c_i),

Butcher's simplifying assumption ``D(1)`` -- equivalently, the condition for
the discrete adjoint of a Runge-Kutta scheme to be a consistent Runge-Kutta
scheme (Hager, *Numer. Math.* 87 (2000) 247-282). Since the gradient is
``h b_i Gᵀ Λ_{n,i}`` and ``G = (-V\\sin θ, V\\cos θ)``, it vanishes exactly
when ``\\tan θ = Λ_y/Λ_x``.

So for this problem the continuous ``θ^*`` is a stationary point of the
*discrete* problem, on every mesh, to rounding -- but only for tableaux
satisfying ``D(1)``. Over the eight shipped tableaux, on the parameters this
module defines, at ``N = 16``, :func:`adjoint_consistency_defect` predicts
which, and ``main`` reprints the table::

    tableau                D(1) defect   |dJ/dtheta| at theta*
    heun                     0.000e+00               2.082e-17
    rk4                      0.000e+00               2.776e-17
    implicit_midpoint        0.000e+00               5.551e-17
    gauss2                   0.000e+00               2.776e-17
    explicit_euler           1.000e+00               1.567e-02
    sdirk2                   8.579e-02               1.357e-03
    sdirk3                   3.800e-01               6.006e-03
    implicit_trapezoid       2.500e-01               3.955e-03

The four small entries are rounding, not zero, and their last digits depend on
the BLAS in use; C-11.3 forbids pinning them and the test does not. What is
being claimed is the *gap* -- fourteen orders of magnitude, and on which side
of it each tableau falls.

That is worth more than a mesh study. It is a closed-form anchor (C-14.1 tier
2) that binds the adjoint's *stage* weights ``b_j a_{ji} / b_i`` -- the part
of the discrete adjoint a duality test can least distinguish -- to a
predicate computed from the tableau alone, at machine precision, independently
of the mesh.

The value of the optimum still carries ordinary truncation error, so the
fourth-order mesh study remains available: on the same parameters,
``|x_N - x(T)|`` is ``1.289e-07`` at ``N = 8`` and falls at measured orders
3.99, 4.00 and 4.00 through ``N = 64``.

No stage quadrature here
------------------------

``rocket_ascent.py`` and ``pendulum_swing_up.py`` carry ``h * w_k`` in the
objective because the library applies ``dJ_du`` verbatim (C-9.3). This
objective has no running term at all -- ``dJ_du`` is identically zero -- so
that obligation does not arise, and the gradient the optimizer reports is
entirely the adjoint's work.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.optimize import minimize

from adjungo.core.method import GLMethod
from adjungo.methods.runge_kutta import rk4
from adjungo.optimization.interface import GLMOptimizer
from examples.symbolic import SymbolicDynamics

#: Boat speed through the water.
BOAT_SPEED = 1.2
#: Height at which the current matches the boat's own speed; the shear is
#: ``BOAT_SPEED * y / SHEAR_HEIGHT``.
SHEAR_HEIGHT = 0.8
#: Fixed horizon. Maximising ``x(T)`` over it replaces Zermelo's free final
#: time, which this library has no parameter for.
T_FINAL = 1.5
#: Mesh used by :func:`solve` and reported by :func:`main`.
N_STEPS = 24
#: Start at the origin, on the streamline where the water is at rest.
Y0 = np.array([0.0, 0.0])


def shear_rate() -> float:
    """``V / h``, the current's velocity gradient and the only rate here."""
    return BOAT_SPEED / SHEAR_HEIGHT


# ---------------------------------------------------------------------------
# The vector field
# ---------------------------------------------------------------------------


def zermelo_dynamics() -> SymbolicDynamics:
    """``f = (V cos θ + V y / h, V sin θ)``, with its derivatives differentiated.

    Only ``f`` is written down. ``F``, ``G`` and the three contracted second
    derivatives come from :class:`examples.symbolic.SymbolicDynamics`;
    ``tests/test_examples.py`` checks all five against derivatives taken by
    hand, including the ``F_uu_action`` that no other example exercises.
    """
    x, y, heading, time = sp.symbols("x y theta t")
    f = [
        BOAT_SPEED * sp.cos(heading) + BOAT_SPEED * y / SHEAR_HEIGHT,
        BOAT_SPEED * sp.sin(heading),
    ]
    return SymbolicDynamics(f, (x, y), (heading,), time)


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------


def optimal_heading(t: NDArray | float) -> NDArray:
    """``θ*(t) = arctan((V/h)(T - t))``, the continuous optimal heading.

    Steeply across the current at first -- ``arctan`` of ``(V/h)T`` -- easing
    to dead downstream at ``t = T``, when there is no time left to profit from
    climbing further into the faster water. The range is ``(0, π/2)``, so the
    boat never heads backwards.
    """
    return np.arctan(shear_rate() * (T_FINAL - np.asarray(t, dtype=float)))


def optimal_position(t: NDArray | float) -> NDArray:
    """Exact ``(x, y)`` along the optimal path, shape ``(2, ...)``.

    Substituting ``θ*`` and changing variable to ``s = (V/h)(T - t)`` gives
    ``dy/ds = -h s / sqrt(1 + s²)`` and hence ``y = A - h sqrt(1 + s²)`` with
    ``A`` fixed by the initial height. Feeding that back into ``x`` leaves
    only standard integrals, collected in ``g`` below, so that
    ``x(t) = x₀ + g(s) - g(s₀)``.
    """
    s = shear_rate() * (T_FINAL - np.asarray(t, dtype=float))
    s0 = shear_rate() * T_FINAL
    a = Y0[1] + SHEAR_HEIGHT * np.hypot(1.0, s0)

    def g(sigma: NDArray | float) -> NDArray:
        sigma = np.asarray(sigma, dtype=float)
        return (
            -0.5 * SHEAR_HEIGHT * np.arcsinh(sigma)
            - a * sigma
            + 0.5 * SHEAR_HEIGHT * sigma * np.hypot(1.0, sigma)
        )

    return np.stack(
        [Y0[0] + g(s) - g(s0), a - SHEAR_HEIGHT * np.hypot(1.0, s)]
    )


def maximum_range() -> float:
    """``x(T)`` on the optimal path -- the best the boat can do in ``T``.

    Compare ``BOAT_SPEED * T_FINAL``, which is what steering dead downstream
    from ``y = 0`` achieves: the current is at rest on that streamline, so the
    straight run collects nothing from it.
    """
    return float(optimal_position(T_FINAL)[0])


def navigation_defect(t: NDArray | float, dt: float = 1e-5) -> float:
    """Residual of Zermelo's navigation formula ``dθ/dt = -(V/h) cos²θ``.

    Differenced centrally, so this is ``O(dt²)`` rather than zero; the test
    confirms it *scales* that way instead of pinning a value. The general
    formula ``\\dot θ = sin²θ ∂v/∂x + sinθcosθ(∂u/∂x - ∂v/∂y) - cos²θ ∂u/∂y``
    collapses to this one because the shear contributes only ``∂u/∂y = V/h``.
    """
    t = np.asarray(t, dtype=float)
    slope = (optimal_heading(t + dt) - optimal_heading(t - dt)) / (2.0 * dt)
    return float(
        np.max(np.abs(slope + shear_rate() * np.cos(optimal_heading(t)) ** 2))
    )


def adjoint_consistency_defect(method: GLMethod) -> float:
    """``max_i |Σ_j b_j a_{ji} - b_i (1 - c_i)|``: Butcher's ``D(1)``, as a defect.

    Zero exactly when the discrete adjoint reproduces a costate that is linear
    in ``t`` at the *stage* abscissae, which for this problem is the whole of
    the optimality condition. See the module docstring for the derivation and
    for the measured table this predicts.
    """
    b = np.asarray(method.B, dtype=float).ravel()
    return float(
        np.max(
            np.abs(
                b @ np.asarray(method.A, dtype=float)
                - b * (1.0 - np.asarray(method.c, dtype=float).ravel())
            )
        )
    )


# ---------------------------------------------------------------------------
# The controlled problem
# ---------------------------------------------------------------------------


class MaxRangeCost:
    """``J = -x(T)``: purely terminal, and linear in the terminal state.

    Every second derivative of the cost vanishes, so the Hessian this problem
    presents to ``trust-ncg`` is made entirely of ``F_uu`` contracted with the
    adjoint. There is no running term, so C-9.3's stage quadrature does not
    arise.
    """

    def evaluate(self, trajectory, u: NDArray) -> float:
        return -float(trajectory.Y[-1][0][0])

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        gradient = np.zeros_like(np.asarray(y_final, dtype=float))
        gradient[0] = np.array([-1.0, 0.0])
        return gradient

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros_like(np.asarray(y, dtype=float))

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros_like(np.asarray(u_stage, dtype=float))

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros((1, 1))

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((2, 2))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return np.zeros((2, 2))


def build_optimizer(
    n_steps: int = N_STEPS,
    method_factory=rk4,
) -> GLMOptimizer:
    """Assemble the maximum-range problem on ``rk4`` by default.

    ``rk4`` is explicit, fourth order, and satisfies ``D(1)`` -- the last of
    which is what makes the closed-form control exactly optimal here. Pass
    another factory to see that property fail; ``main`` does exactly that.
    """
    return GLMOptimizer(
        zermelo_dynamics(),
        MaxRangeCost(),
        method_factory(),
        t_span=(0.0, T_FINAL),
        N=n_steps,
        y0=Y0,
    )


def stage_times(n_steps: int = N_STEPS, method_factory=rk4) -> NDArray:
    """Absolute time at every stage, shape ``(n_steps, s)``."""
    method = method_factory()
    h = T_FINAL / n_steps
    return np.arange(n_steps)[:, None] * h + h * np.asarray(method.c)[None, :]


def optimal_control(n_steps: int = N_STEPS, method_factory=rk4) -> NDArray:
    """``θ*`` sampled at the stage abscissae, shape ``(n_steps, s, 1)``."""
    return optimal_heading(stage_times(n_steps, method_factory))[:, :, None]


def solve(n_steps: int = N_STEPS, method_factory=rk4):
    """Maximise the range from a dead-downstream start, using the exact Hessian.

    Starting from ``θ ≡ 0`` rather than from zero curvature matters: the
    heading is defined modulo ``2π`` and the problem has a stationary point on
    every branch, so the start selects which one is found.
    """
    optimizer = build_optimizer(n_steps, method_factory)
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


def main() -> None:
    from adjungo.methods import runge_kutta

    print(__doc__.strip().splitlines()[0])
    print()

    print("Problem:")
    print(f"  boat speed V            : {BOAT_SPEED}")
    print(f"  shear V*y/h, h          : {SHEAR_HEIGHT}   (rate V/h = {shear_rate():g})")
    print(f"  horizon T               : {T_FINAL}")
    print(f"  start                   : ({Y0[0]:g}, {Y0[1]:g})")
    print()

    straight = BOAT_SPEED * T_FINAL
    best = maximum_range()
    print("Closed-form optimum:")
    print(f"  initial heading         : {np.degrees(optimal_heading(0.0)):.3f} deg")
    print(f"  final heading           : {np.degrees(optimal_heading(T_FINAL)):.3f} deg")
    print(f"  x(T) steering downstream: {straight:.9f}")
    print(f"  x(T) steering optimally : {best:.9f}   ({best / straight:.3f}x)")
    print(f"  y(T)                    : {optimal_position(T_FINAL)[1]:.9f}")
    print(f"  navigation-formula defect: {navigation_defect(np.linspace(0, T_FINAL, 9)):.2e}")
    print()

    print("Is the continuous theta* stationary for the DISCRETE problem, N = 16?")
    print("  D(1) holds exactly when the discrete adjoint carries a linear")
    print("  costate at the stage abscissae, which is all this problem asks.")
    print(f"    {'tableau':<20}{'D(1) defect':>14}{'|dJ/dtheta| at theta*':>24}")
    for name in (
        "heun",
        "rk4",
        "implicit_midpoint",
        "gauss2",
        "explicit_euler",
        "sdirk2",
        "sdirk3",
        "implicit_trapezoid",
    ):
        factory = getattr(runge_kutta, name)
        optimizer = build_optimizer(16, factory)
        gradient = np.abs(optimizer.gradient(optimal_control(16, factory))).max()
        defect = adjoint_consistency_defect(factory())
        print(f"    {name:<20}{defect:>14.3e}{gradient:>24.3e}")
    print()

    print("The value still carries truncation error, and refines at rk4's order:")
    previous = None
    for n_steps in (8, 16, 32, 64):
        error = abs(-build_optimizer(n_steps).objective_value(optimal_control(n_steps)) - best)
        rate = "" if previous is None else f"   order {np.log2(previous / error):.2f}"
        print(f"    N = {n_steps:3d}   |x_N - x(T)| = {error:.3e}{rate}")
        previous = error
    print()

    optimizer, result, control = solve()
    final = optimizer.trajectory(control).Y[-1][0]
    print(f"Optimizer on rk4, N = {N_STEPS}, from theta = 0:")
    print(f"  x(T)                    : {-result.fun:.9f}")
    print(f"  y(T)                    : {final[1]:.9f}")
    print(f"  |theta - theta*|inf     : {np.abs(control - optimal_control()).max():.2e}")
    print(f"  |grad J| at theta*      : {np.abs(optimizer.gradient(optimal_control())).max():.2e}")
    print(f"  iterations              : {result.nit}")


if __name__ == "__main__":
    main()
