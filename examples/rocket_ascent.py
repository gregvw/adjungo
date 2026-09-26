"""Rocket ascent: reach a target altitude on the least propellant energy.

Run directly::

    .venv/bin/python -m examples.rocket_ascent

Unlike the other examples this one imports a sibling module,
``examples/symbolic.py``. Running the file by path would put ``examples/`` on
``sys.path`` instead of the repository root, so ``-m`` is the invocation that
works; ``tests/test_examples.py`` checks that the line above resolves.

Why this example exists
-----------------------

The other examples write every derivative by hand. That is auditable and it is
what a C++ port will do, but it does not scale past two states, and hand
derivation is where an example acquires a quiet sign error. Here the dynamics
are handed to ``examples/symbolic.py`` as sympy expressions and the five
derivative callbacks are differentiated from them. The generated callbacks are
checked against hand-derived ones in the test, so the convenience does not cost
the verification.

The problem is also the first here with a **closed-form solution for the
dynamics**: under a constant burn rate the state has an elementary
antiderivative, which is the Tsiolkovsky rocket equation. That is a tier-2
oracle under C-14.1 -- a closed-form anchor, not a self-consistency check --
and the test uses it both to verify the forward solve and to observe rk4's
fourth order on this problem.

Provenance
----------

The problem, its parameters and the construction of the target altitude are
taken from the ``rocket-simopt`` tutorial of Sandia's Rapid Optimization
Library (ROL), BSD-3-Clause, at ``tutorial/rocket-simopt`` in
``sandialabs/rol``. ROL poses it as a SimOpt problem on the discrete
trajectory. Posing it here as an ODE-constrained problem in adjungo's terms is
a reformulation, not a translation: nothing is copied, and the discrete
objective below is adjungo's, not ROL's.

Physical problem
----------------

A rocket ascends vertically, burning propellant at rate ``z(t) >= 0``. With
state ``y = (h, v, m)`` -- altitude, velocity, total mass:

.. math::

    \\dot{h} &= v \\\\
    \\dot{v} &= v_e \\frac{z}{m} - g \\\\
    \\dot{m} &= -z

The thrust term ``v_e z / m`` is what makes this nonlinear: it is inversely
proportional to the state component ``m``, so ``df/dy`` depends on the state
and the second derivatives do not vanish. Burning propellant both accelerates
the rocket and lightens it, which is the whole content of the rocket equation.

The objective trades altitude against propellant energy:

.. math::

    J = \\tfrac{1}{2}\\left(h(T) - h_\\text{target}\\right)^2
        + \\tfrac{1}{2}\\alpha \\int_0^T z(t)^2 \\, dt

``h_target`` is the altitude a **constant** burn reaches while consuming
exactly the available fuel over ``[0, T]``. That is ROL's construction and it
is a good one: the target is known to be attainable, the constant profile is a
natural initial guess, and any improvement the optimizer finds is a genuine
statement about the *shape* of the burn rather than about its size.

What this demonstrates
----------------------

The optimizer moves the burn earlier and shuts the engine down before ``T``,
reaching essentially the same altitude on materially less energy. Propellant
burned early is propellant whose velocity contribution has the whole remaining
flight to accumulate into altitude, and it is also mass the rocket does not
have to carry and accelerate afterwards.

On the discrete objective below, with the parameters fixed above, the reported
run reduces ``J`` from 5333.33 to 4400.65 -- the entire reduction coming from
energy, 5333.33 to 4400.64, while the terminal altitude moves by -0.16 m out
of 18218.43 m. The optimal profile stays non-negative on its own; the lower
bound ``z >= 0`` is *not* imposed here, and the run is reported rather than
promised. See the ``ENVELOPE`` note at the end of this docstring.

The discrete objective and C-9.3
--------------------------------

C-9.3 states the stage-quadrature term as ``h * sum_n sum_k w_k * l(...)``,
with ``w_k = B[0, k]`` for a Runge-Kutta method in GLM form. The library does
not apply ``h`` or ``w_k`` on the objective's behalf: ``dJ_du`` is used exactly
as returned (``adjungo/optimization/gradient.py``). An objective that intends
the quadrature must therefore carry both factors itself, and this one does --
unlike ``minimum_energy_oscillator.py`` and ``nonlinear_implicit_control.py``,
whose control term is a plain sum over stages. Both are legitimate discrete
objectives; they are simply different functions, and the gradient each example
reports is the exact gradient of its own.

ENVELOPE
--------

Non-negativity of ``z`` and positivity of ``m`` are properties of the
*solution* here, not constraints imposed on the problem. The dynamics have a
singularity at ``m = 0`` and no meaning for ``m`` below the dry mass. The run
below stays well inside the physical region -- the mass never falls below
34.7 of an initial 100, against a dry mass of 20 -- but a different ``alpha``,
horizon or target could leave it, and nothing in this example would stop it.
Imposing ``z >= 0`` needs bounds, which ``scipy`` supports directly; imposing
a terminal condition on the state needs a constraint Jacobian, which is open
question C-Q7 in ``NUMERICS.md``.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.optimize import minimize

from adjungo import GLMOptimizer
from adjungo.methods.runge_kutta import rk4
from examples.symbolic import SymbolicDynamics

# Parameters from ROL's Rocket.xml.
GRAVITY = 9.8  # m/s^2
EXHAUST_VELOCITY = 1000.0  # m/s
DRY_MASS = 20.0  # kg
FUEL_MASS = 80.0  # kg
INITIAL_MASS = DRY_MASS + FUEL_MASS  # kg
T_FINAL = 60.0  # s
ENERGY_WEIGHT = 100.0  # ROL's "Mass Penalty", alpha

N_STEPS = 60
Y0 = np.array([0.0, 0.0, INITIAL_MASS])

#: The burn rate that consumes exactly the available fuel over the horizon.
CONSTANT_BURN = FUEL_MASS / T_FINAL


def rocket_dynamics() -> SymbolicDynamics:
    """The three-state ascent model, with its derivatives differentiated."""
    h, v, m = sp.symbols("h v m", real=True)
    z = sp.Symbol("z", real=True)
    t = sp.Symbol("t", real=True)

    f = sp.Matrix(
        [
            v,
            EXHAUST_VELOCITY * z / m - GRAVITY,
            -z,
        ]
    )
    return SymbolicDynamics(f, [h, v, m], [z], t)


def constant_burn_state(t: float, rate: float = CONSTANT_BURN) -> NDArray:
    """Closed-form ``(h, v, m)`` at time ``t`` under a constant burn rate.

    With ``m(t) = m0 - z t`` the velocity integrates to the Tsiolkovsky rocket
    equation less the gravity loss,

    .. math::

        v(t) = v_e \\ln\\frac{m_0}{m_0 - z t} - g t,

    and altitude integrates once more. Writing ``mu = m0 - z t``,

    .. math::

        h(t) = v_e \\left[ t (\\ln m_0 + 1)
               + \\frac{\\mu \\ln \\mu - m_0 \\ln m_0}{z} \\right]
               - \\tfrac{1}{2} g t^2,

    which is exact, elementary, and shares no code with ``adjungo.stepping``.
    That independence is what makes it usable as a C-14.1 tier-2 anchor.
    """
    mass = INITIAL_MASS - rate * t
    if mass <= 0.0:
        raise ValueError(
            f"a burn rate of {rate} exhausts all mass by t={INITIAL_MASS / rate}; "
            f"the closed form has no meaning at t={t}"
        )

    velocity = EXHAUST_VELOCITY * np.log(INITIAL_MASS / mass) - GRAVITY * t
    altitude = (
        EXHAUST_VELOCITY
        * (
            t * (np.log(INITIAL_MASS) + 1.0)
            + (mass * np.log(mass) - INITIAL_MASS * np.log(INITIAL_MASS)) / rate
        )
        - 0.5 * GRAVITY * t**2
    )
    return np.array([altitude, velocity, mass])


#: Altitude reached by the constant burn. Attainable by construction.
TARGET_ALTITUDE = float(constant_burn_state(T_FINAL)[0])


class AltitudeMissAndEnergy:
    """``½(h(T) - h_target)² + ½ α ∫ z² dt``, quadrature carried per C-9.3.

    The stage weights and the step size are held here because the library
    applies neither to ``dJ_du``; see the module docstring.
    """

    def __init__(
        self,
        target_altitude: float,
        stage_weights: NDArray,
        step_size: float,
        energy_weight: float = ENERGY_WEIGHT,
    ) -> None:
        self.target_altitude = target_altitude
        self.stage_weights = np.asarray(stage_weights, dtype=float)
        self.step_size = step_size
        self.energy_weight = energy_weight

    def _quadrature_factor(self, stage: int) -> float:
        return self.step_size * float(self.stage_weights[stage]) * self.energy_weight

    def evaluate(self, trajectory, u: NDArray) -> float:
        miss = trajectory.Y[-1][0][0] - self.target_altitude
        energy = 0.5 * self.energy_weight * self.step_size * float(
            np.einsum("k,nkv->", self.stage_weights, u**2)
        )
        return float(0.5 * miss**2 + energy)

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        gradient = np.zeros_like(np.asarray(y_final, dtype=float))
        gradient[0, 0] = y_final[0][0] - self.target_altitude
        return gradient

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros_like(np.asarray(y, dtype=float))

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self._quadrature_factor(stage) * u_stage

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self._quadrature_factor(stage) * np.eye(1)

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((3, 3))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        # Only the altitude component is penalised.
        return np.diag([1.0, 0.0, 0.0])


def build_optimizer(n_steps: int = N_STEPS) -> GLMOptimizer:
    """Assemble the optimizer for the ascent problem on ``n_steps`` steps."""
    method = rk4()
    return GLMOptimizer(
        problem=rocket_dynamics(),
        objective=AltitudeMissAndEnergy(
            target_altitude=TARGET_ALTITUDE,
            stage_weights=method.B[0, :],
            step_size=T_FINAL / n_steps,
        ),
        method=method,
        t_span=(0.0, T_FINAL),
        N=n_steps,
        y0=Y0,
    )


def initial_control(n_steps: int = N_STEPS) -> NDArray:
    """The constant burn, which is ROL's initial guess and defines the target."""
    return np.full((n_steps, rk4().s, 1), CONSTANT_BURN)


def solve(n_steps: int = N_STEPS):
    """Minimise with ``trust-ncg`` and the exact Hessian-vector product.

    ``gtol`` is 1e-6 rather than something smaller because ``trust-ncg``
    reports ``status 2`` ("bad approximation caused failure to predict
    improvement") once the gradient is at the level where the trust-region
    model can no longer resolve an improvement. The run stops with a gradient
    norm near 5e-8 either way; the looser tolerance simply lets it say so.
    """
    optimizer = build_optimizer(n_steps)
    fun, jac = optimizer.scipy_interface()
    hessp = optimizer.scipy_hessp()

    result = minimize(
        fun,
        initial_control(n_steps).ravel(),
        jac=jac,
        hessp=hessp,
        method="trust-ncg",
        options={"maxiter": 500, "gtol": 1e-6},
    )
    return optimizer, result, result.x.reshape(n_steps, rk4().s, 1)


def _energy(u: NDArray, n_steps: int) -> float:
    weights = rk4().B[0, :]
    return float(
        0.5 * ENERGY_WEIGHT * (T_FINAL / n_steps) * np.einsum("k,nkv->", weights, u**2)
    )


def main() -> None:
    optimizer, result, u_optimal = solve()

    u_initial = initial_control()
    exact = constant_burn_state(T_FINAL)
    initial_trajectory = optimizer.trajectory(u_initial)
    optimal_trajectory = optimizer.trajectory(u_optimal)

    print("Rocket ascent to a target altitude on least propellant energy")
    print("=" * 62)
    print(f"  states (h, v, m)         : {optimizer.problem.state_dim}")
    print(f"  control (burn rate z)    : {optimizer.problem.control_dim}")
    print(f"  method                   : rk4, {rk4().s} stages")
    print(f"  steps N                  : {N_STEPS}")
    print(f"  unknowns                 : {u_optimal.size}")
    print()

    print("Forward solve against the closed form (constant burn):")
    computed = initial_trajectory.Y[-1][0]
    for name, got, want in zip(("altitude", "velocity", "mass"), computed, exact):
        print(f"  {name:9s} {got:16.6f}   exact {want:16.6f}   err {abs(got - want):.2e}")
    print()

    print("Optimization:")
    print(f"  converged                : {result.success} ({result.nit} iterations)")
    print(f"  gradient norm            : {np.linalg.norm(result.jac):.3e}")
    print(f"  objective                : {result.fun:.2f}  (from {optimizer.objective_value(u_initial):.2f})")
    print()

    print("What changed:")
    print(
        f"  energy                   : {_energy(u_initial, N_STEPS):.2f}"
        f" -> {_energy(u_optimal, N_STEPS):.2f}"
    )
    print(
        f"  terminal altitude        : {optimal_trajectory.Y[-1][0][0]:.2f} m"
        f"  (target {TARGET_ALTITUDE:.2f} m,"
        f" miss {optimal_trajectory.Y[-1][0][0] - TARGET_ALTITUDE:+.2f} m)"
    )
    print(
        f"  burn rate                : constant {CONSTANT_BURN:.4f}"
        f" -> range [{u_optimal.min():.4f}, {u_optimal.max():.4f}] kg/s"
    )
    print(
        f"  fuel used                : {INITIAL_MASS - optimal_trajectory.Y[-1][0][2]:.2f}"
        f" of {FUEL_MASS:.2f} kg available"
    )
    print(f"  minimum mass in flight   : {optimal_trajectory.Y[:, 0, 2].min():.2f} kg"
          f"  (dry mass {DRY_MASS:.2f} kg)")
    print()
    print("  The engine is throttled up early and shut down before T: propellant")
    print("  burned early buys velocity that has the rest of the flight to become")
    print("  altitude, and mass that no longer has to be carried.")


if __name__ == "__main__":
    main()
