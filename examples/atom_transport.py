"""Atom transport in a shuttled lattice trap, on a partitioned symplectic method.

Run directly::

    .venv/bin/python -m examples.atom_transport

It imports a sibling (``examples.symbolic``), so running it by path puts
``examples/`` on ``sys.path`` rather than the repository root and the import
fails. ``-m`` resolves the package from the root, which is also how the tests
reach it.

The problem
-----------

A neutral atom in the lowest band of an optical lattice, held in a harmonic
trap whose centre is the control. Canonical coordinates ``y = (q, p)``, control
``u`` the trap centre, and a **separable** Hamiltonian in the sense of
`NUMERICS.md` C-8.4:

.. math::

    H(q, p, u, t) = T(p) + V(q, u), \\qquad
    T(p) = J\\,(1 - \\cos p), \\qquad
    V(q, u) = \\tfrac12 κ (q - u)^2,

so that

.. math::

    \\dot q = J \\sin p, \\qquad \\dot p = -κ (q - u).

The task is *shuttling*: move the atom from rest at ``q = 0`` to rest at
``q = d`` in a fixed time, which is the transport step of a neutral-atom or
trapped-ion register. What is actually paid for is the **motional excitation**
left behind -- the atom must arrive *and stop* -- plus the control energy
spent moving the trap.

The kinetic energy is deliberately not quadratic. ``T(p) = J(1 - cos p)`` is
the lowest-band tight-binding dispersion, and it makes ``∂f^q/∂p = J cos p``
depend on the state. With ``T = p²/2`` that block is the identity and a wrong
index in the ``A^q`` stage recursion would act on a constant Jacobian and
cancel; see C-14.5, which makes the same choice for the certification fixture
and for the same reason.

What this example adds
----------------------

It is the first example here that is **not a GLM**. Every other one applies a
single coefficient array ``A`` to the whole state, as ``A ⊗ I``. Störmer-Verlet
applies a different array to each half -- ``A^q`` to the position block and
``A^p`` to the momentum block -- which C-8.4 records is not expressible as a
GLM and is not a matter of carrying more history.

**Why that is worth the machinery here.** The objective is residual motional
energy at the end of the transport: a small difference of larger quantities.
Any energy the integrator invents is added directly to the number being
minimised, so a method that drifts optimises partly against its own artefact.
The contrast below measures that.

**What symplecticity does and does not buy.** A symplectic method preserves the
canonical symplectic form for prescribed controls. That is *not* a guarantee of
energy conservation or of the absence of artificial drift, and C-18.6 says so
explicitly. The energy behaviour printed below is an **observation on this
problem at constant step size**, not an inherited guarantee -- and per C-18.6 it
is not inherited by a variable-step plan either, where prescribed unequal steps
remain symplectic yet can excite a resonance. Read it as a measurement, with its
setup, and not as a theorem.

Two caveats on the comparison. ``rk4`` is fourth order against Verlet's second,
so it is more accurate per step on any short arc; what is being contrasted is
the *growth* of the energy error with horizon, not its size. And both are
measured by one statistic, ``max|E(t) - E(0)|`` over the whole integration,
because measuring one method by its endpoint error and the other by the spread
of its oscillation would compare two different quantities.

The domain is a hypothesis, and it is checked
---------------------------------------------

C-8.4 scopes partitioned methods to separable Hamiltonians. That is not a
stylistic preference: the partitioned tangent and adjoint sweeps **drop** the
coupling blocks by construction, so a problem outside the domain would not be
integrated inaccurately, it would be integrated as a different problem. Per C-7
the solve refuses instead. :func:`refusal_demonstration` adds an ordinary
linear drag ``-γp`` to ``f^p`` -- a perfectly reasonable physical model, which
every other certified family integrates without comment -- and shows the
partitioned route declining it.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.optimize import minimize

from adjungo.core.partitioned import verlet
from adjungo.core.plan import DiscretizationPlan
from adjungo.optimization.interface import GLMOptimizer
from adjungo.solvers.partitioned import SeparabilityViolation
from examples.symbolic import SymbolicDynamics

#: Tight-binding half-bandwidth, trap stiffness, transport distance.
HOPPING = 1.0
STIFFNESS = 4.0
DISTANCE = 1.0

T_FINAL = 3.0
N_STEPS = 60

#: Weight on the terminal miss. The momentum entry is what makes this a
#: transport problem rather than a reachability problem: arriving at ``q = d``
#: with momentum left over is a failure, not a success.
TERMINAL_WEIGHT = np.diag([40.0, 40.0])
ENERGY_WEIGHT = 1e-3

TARGET = np.array([DISTANCE, 0.0])


def transport_dynamics(drag: float = 0.0):
    """``q' = J sin p``, ``p' = -κ(q - u) - γ p``, derivatives from sympy.

    ``drag`` is zero for the transport itself. A nonzero value puts ``p`` into
    ``f^p`` and so leaves C-8.4's separable domain; :func:`refusal_demonstration`
    uses it for exactly that.
    """
    q, p, u, t = sp.symbols("q p u t")
    f = sp.Matrix(
        [
            HOPPING * sp.sin(p),
            -STIFFNESS * (q - u) - drag * p,
        ]
    )

    class TransportProblem(SymbolicDynamics):
        #: Index of the first momentum component. The partition is the
        #: problem's to declare and never the tableau's to guess: ``n // 2``
        #: is right for every canonical Hamiltonian and wrong in silence for
        #: anything else, which is the shape of answer C-7 forbids.
        n_q = 1

    return TransportProblem(f, (q, p), (u,), t)


def energy(y: NDArray, centre: float = 0.0) -> float:
    """``H = J(1 - cos p) + ½κ(q - u)²`` at a fixed trap centre."""
    q, p = float(y[0]), float(y[1])
    return HOPPING * (1.0 - np.cos(p)) + 0.5 * STIFFNESS * (q - centre) ** 2


class TransportCost:
    """``½(y_N - target)ᵀ S (y_N - target) + ½ R h Σ w_k (u_k - ū_k)²``.

    The control is penalised against the straight-line ramp ``ū(t) = d t / T``
    rather than against zero. Penalising against zero would price the transport
    itself, and the cheapest control would be not to move; what should be priced
    is the *deviation* from simply dragging the trap across, which is what
    suppresses the motional excitation.

    The step size and stage weights are carried here rather than by the
    library, which applies ``dJ_du`` verbatim; see C-9.3.
    """

    def __init__(
        self,
        stage_weights: NDArray,
        step_size: float,
        ramp: NDArray,
        target: NDArray = TARGET,
        energy_weight: float = ENERGY_WEIGHT,
    ) -> None:
        self.stage_weights = np.asarray(stage_weights, dtype=float)
        self.step_size = step_size
        self.ramp = np.asarray(ramp, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.energy_weight = energy_weight

    def _quadrature_factor(self, stage: int) -> float:
        return self.step_size * float(self.stage_weights[stage]) * self.energy_weight

    def _deviation(self, u: NDArray) -> NDArray:
        return np.asarray(u, dtype=float) - self.ramp

    def evaluate(self, trajectory, u: NDArray) -> float:
        miss = trajectory.Y[-1][0] - self.target
        d = self._deviation(u)
        stage_energy = np.einsum("k,nkv->", self.stage_weights, d**2)
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
        return self._quadrature_factor(stage) * (
            np.asarray(u_stage, dtype=float) - self.ramp[step, stage]
        )

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return self._quadrature_factor(stage) * np.eye(1)

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((2, 2))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return TERMINAL_WEIGHT.astype(float).copy()


def stage_ramp(method, n_steps: int = N_STEPS, horizon: float = T_FINAL) -> NDArray:
    """``ū(t) = d t / T`` sampled at the stage times of every step.

    Sampled per step through ``c``, not once for the whole horizon, because a
    partitioned method's abscissae are supplied rather than derived: C-8.4
    records that an abscissa is not a row sum of ``A^q`` or of ``A^p``.
    """
    nodes = np.linspace(0.0, horizon, n_steps + 1)
    h = horizon / n_steps
    times = nodes[:-1, None] + h * np.asarray(method.c, dtype=float)[None, :]
    return (DISTANCE * times / horizon)[:, :, None]


def build_optimizer(
    n_steps: int = N_STEPS, method_factory=verlet
) -> tuple[GLMOptimizer, NDArray]:
    """The shuttling problem on Störmer-Verlet. Returns the optimizer and the ramp."""
    method = method_factory()
    plan = DiscretizationPlan.uniform((0.0, T_FINAL), n_steps, method)
    ramp = stage_ramp(method, n_steps)
    cost = TransportCost(method.B[0, :], T_FINAL / n_steps, ramp)
    optimizer = GLMOptimizer(
        transport_dynamics(),
        cost,
        y0=np.zeros(2),
        plan=plan,
    )
    return optimizer, ramp


def solve(n_steps: int = N_STEPS):
    """Minimise with the exact gradient and exact Hessian-vector product."""
    optimizer, ramp = build_optimizer(n_steps)
    fun, jac = optimizer.scipy_interface()
    result = minimize(
        fun,
        ramp.ravel(),
        jac=jac,
        hessp=optimizer.scipy_hessp(),
        method="trust-ncg",
        options={"maxiter": 400, "gtol": 1e-12},
    )
    return optimizer, result, result.x.reshape(ramp.shape)


def excitation(optimizer: GLMOptimizer, control: NDArray) -> float:
    """Motional energy left at the end, measured in the trap it arrives in.

    ``H`` at the final state with the trap centred on the target: the energy
    the atom still carries once transport is over, which is the quantity the
    terminal weight is a proxy for.
    """
    return energy(optimizer.trajectory(control).Y[-1][0], centre=DISTANCE)


def energy_growth(method_factory, periods: tuple[int, int] = (1, 32)) -> float:
    """Growth of ``max|E(t) - E(0)|`` between two horizons, at fixed steps per period.

    One statistic for both methods, and the same number of steps per period at
    each horizon, so what changes between the two measurements is the length of
    the integration and nothing else.
    """
    traces = [
        _energy_trace(method_factory, p * trap_period(), 20 * p)
        for p in periods
    ]
    initial = traces[0][0]
    errors = [float(np.max(np.abs(trace - initial))) for trace in traces]
    return errors[1] / errors[0]


def trap_period() -> float:
    """``2π/√(κ J)``: the small-oscillation period of the undriven trap.

    Linearising ``q' = J sin p ≈ J p`` and ``p' = -κ q`` gives ``q'' = -κJ q``.
    Only a scale for the horizons below; the integration itself uses the full
    nonlinear dispersion.
    """
    return 2.0 * np.pi / np.sqrt(STIFFNESS * HOPPING)


def _energy_trace(method_factory, horizon: float, n_steps: int) -> NDArray:
    """``E`` at every step node of an undriven solve, trap held at the origin."""
    method = method_factory()
    plan = DiscretizationPlan.uniform((0.0, horizon), n_steps, method)
    ramp = np.zeros((n_steps, method.s, 1))
    optimizer = GLMOptimizer(
        transport_dynamics(),
        TransportCost(method.B[0, :], horizon / n_steps, ramp),
        y0=np.array([0.5, 0.0]),
        plan=plan,
    )
    trajectory = optimizer.trajectory(ramp)
    return np.array([energy(trajectory.Y[i][0]) for i in range(n_steps + 1)])


def refusal_demonstration(drag: float = 0.3) -> str:
    """Solve the same problem with a drag term, and return the refusal message.

    ``-γp`` in ``f^p`` is an ordinary damping model and every GLM family here
    integrates it. It is outside C-8.4's separable domain, and the partitioned
    route declines rather than dropping the term it cannot carry.

    The *structural* check is what refuses it, and the reason is specific to
    the first step. The trajectory starts from rest with the ramp still at
    zero, so every quantity in step 0 is exactly zero: the momentum
    substituted for the unwritten half is the incoming ``y^p = 0``, the
    completed stage momentum is zero too, and the value comparison sees
    ``0.0`` against ``0.0``. ``F^pp = -γ = -0.3`` is nonzero everywhere, so
    the Jacobian structure refuses at that same step.

    The value check is not blind to this problem in general. With the
    structural check disabled the solve reaches step 1, where the consumed
    ``f^p`` is ``0.06616667`` and the completed one ``0.06567042``, and the
    comparison refuses there. The two checks are complementary in exactly
    this way; see C-8.4.
    """
    optimizer, ramp = build_optimizer()
    damped = GLMOptimizer(
        transport_dynamics(drag=drag),
        optimizer.objective,
        y0=np.zeros(2),
        plan=optimizer.plan,
    )
    try:
        damped.objective_value(ramp)
    except SeparabilityViolation as exc:
        return str(exc)
    raise AssertionError(
        f"a drag of {drag} put p into f^p and was accepted; C-8.4's domain "
        "check did not fire, and the sweeps would have discarded the term"
    )


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    period = trap_period()
    print(f"Trap: J = {HOPPING}, kappa = {STIFFNESS}, "
          f"small-oscillation period {period:.6f}")
    print(f"Transport: q = 0 -> {DISTANCE} at rest, T = {T_FINAL}, N = {N_STEPS}")

    print("\nUndriven energy error over 32 periods, 20 steps per period:")
    for label, factory in (("verlet", verlet), ("rk4", _rk4)):
        print(f"  {label:7s} max|E - E(0)| grows by "
              f"{energy_growth(factory):8.2f}x")
    print("  (an observation at constant h on this problem, not a guarantee;")
    print("   symplecticity does not imply energy conservation -- C-18.6)")

    optimizer, result, control = solve()
    final = optimizer.trajectory(control).Y[-1][0]
    print(f"\nShuttling on verlet, N = {N_STEPS}, T = {T_FINAL}:")
    print(f"  objective          {result.fun:.9f}")
    print(f"  q(T)               {final[0]:.9f}   (target {DISTANCE})")
    print(f"  p(T)               {final[1]:.9f}   (target 0)")
    print(f"  residual excitation {excitation(optimizer, control):.3e}")
    print(f"  peak |u - ramp|    {np.max(np.abs(control - stage_ramp(verlet()))):.6f}")
    print(f"  iterations         {result.nit}")

    print("\nOutside C-8.4's domain, the solve refuses rather than answering:")
    for line in refusal_demonstration().splitlines():
        print(f"  {line}")


def _rk4():
    from adjungo.methods.runge_kutta import rk4

    return rk4()


if __name__ == "__main__":
    main()
