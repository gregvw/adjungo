"""Double integrator: a problem whose optimal control is known in closed form.

Run directly::

    .venv/bin/python examples/double_integrator.py

Why this example exists
-----------------------

The other examples check *derivatives* against independent references. None of
them checks the **answer**, because none of them knows it. A linear-quadratic
problem does: Pontryagin's conditions are linear, so the continuous optimal
control can be written down, and the discrete problem on a fixed mesh can be
solved exactly by a backward Riccati recursion that shares no code with
``adjungo``.

That gives two references of different kinds, in the order C-14.1 prefers:

1. **Independently assembled discrete optimum.** ``step_maps`` builds the
   one-step affine map ``y_{n+1} = A y_n + B u_n`` from the tableau numbers
   and the continuous coefficient matrices alone, and ``riccati_control``
   minimises the same discrete objective over it. This is the *same* discrete
   problem the optimizer solves, reached by a different route, so agreement is
   a rounding-level statement about the optimizer's answer -- not about
   discretisation.

2. **Closed-form continuous optimum.** ``continuous_optimum`` solves the
   two-point boundary value problem in elementary functions. Comparing against
   it measures discretisation error, which is what the mesh study below
   refines.

A result worth stating
----------------------

For the undamped double integrator the two coincide **exactly**: the discrete
optimal control equals the continuous one sampled at the stage abscissae, to
about twenty unit roundoffs, at every mesh tried from ``N = 5`` to ``N = 160``.
There is no discretisation error to refine away.

That is a property of the problem, not a virtue of the code. With ``u = 0``
in the state equation for ``x`` and no drag, the costate equations give
``lambda_1' = 0`` and ``lambda_2' = -lambda_1``, so ``u* = -lambda_2 / R`` is
**linear in t**. The state is then a cubic, ``u*^2`` is a quadratic, and
``rk4``'s weights ``b = (1/6, 1/3, 1/3, 1/6)`` at abscissae ``(0, ½, ½, 1)``
are Simpson's rule, exact through cubics. Both the propagation and the cost
quadrature are exact for the functions this problem's optimum produces.

An example that only ever tested the exact case would be untestable in the
usual sense -- nothing would distinguish a fourth-order method from a perfect
one. So ``DRAG`` adds a linear drag term ``-c v``, which makes the optimal
costate exponential rather than linear and puts genuine truncation error back
in. The discrete optimum then converges to the continuous one at rk4's order:
measured 4.16, 4.08, 4.04, 4.02, 4.01 over ``N = 5 ... 160``.

The discrete objective and C-9.3
--------------------------------

As in ``rocket_ascent.py``, the stage quadrature ``h * w_k`` is carried by the
objective, because the library applies ``dJ_du`` exactly as returned. The
Riccati reference weights its control cost the same way, ``R_hat = R h
diag(w)``, so the two are minimising the same function.

Uniqueness
----------

The comparison would be meaningless if the discrete problem had a manifold of
minimisers -- ``rk4`` has a repeated abscissa, ``c_2 = c_3 = ½``, so the
question is not idle. The stages enter the step map differently (stage 3 reads
stage 2), and the discrete Hessian is positive definite, smallest eigenvalue
3.3e-3 against a largest of 3.1 at ``N = 5``. The minimiser is unique, and
``tests/test_examples.py`` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from adjungo import GLMOptimizer
from adjungo.core.affine import AffineDynamics
from adjungo.methods.runge_kutta import rk4

T_FINAL = 2.0
Y0 = np.array([0.0, 0.0])  # at rest at the origin
Y_TARGET = np.array([1.0, 0.0])  # one metre away, at rest

#: Terminal weight ``S``. Position is penalised ten times as hard as velocity.
TERMINAL_WEIGHT = np.diag([10.0, 1.0])
#: Control energy weight ``R`` in ``½ R ∫ u² dt``.
ENERGY_WEIGHT = 0.05
#: Linear drag coefficient ``c`` in ``v' = -c v + u``. Zero is the double
#: integrator proper; the nonzero value is what the mesh study uses.
DRAG = 0.8

N_STEPS = 20


def point_mass(drag: float = 0.0) -> AffineDynamics:
    """``x' = v``, ``v' = -c v + u``: a point mass under thrust and drag."""
    return AffineDynamics(
        np.array([[0.0, 1.0], [0.0, -drag]]),
        np.array([[0.0], [1.0]]),
    )


# ---------------------------------------------------------------------------
# Reference 2: the continuous optimum, in closed form
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousOptimum:
    """The solution of the continuous problem's optimality conditions.

    With ``H = ½ R u² + lambda_1 v + lambda_2 (-c v + u)`` the stationarity
    condition is ``u* = -lambda_2 / R`` and the costate obeys

        ``lambda_1' = 0``,  ``lambda_2' = -lambda_1 + c lambda_2``.

    So ``lambda_1 = a`` is constant and ``lambda_2`` is ``b - a t`` when
    ``c = 0`` and ``a/c + K e^{ct}`` otherwise. Two unknowns remain, fixed by
    the transversality condition ``lambda(T) = S (y(T) - y_target)``, which is
    linear in them. ``second`` holds ``b`` or ``K`` accordingly.
    """

    drag: float
    a: float
    second: float

    def control(self, t: NDArray | float) -> NDArray:
        """``u*(t) = -lambda_2(t) / R``."""
        t = np.asarray(t, dtype=float)
        if self.drag == 0.0:
            return (self.a * t - self.second) / ENERGY_WEIGHT
        return (
            -(self.a / self.drag + self.second * np.exp(self.drag * t))
            / ENERGY_WEIGHT
        )

    def state(self, t: NDArray | float) -> NDArray:
        """``(x(t), v(t))`` under ``u*``, integrated in closed form."""
        t = np.asarray(t, dtype=float)
        x0, v0 = Y0
        a, r = self.a, ENERGY_WEIGHT

        if self.drag == 0.0:
            b = self.second
            v = v0 + (a * t**2 / 2.0 - b * t) / r
            x = x0 + v0 * t + (a * t**3 / 6.0 - b * t**2 / 2.0) / r
            return np.stack([x, v], axis=-1)

        c, k = self.drag, self.second
        rise, decay = np.exp(c * t), np.exp(-c * t)
        v = v0 * decay - a / (c**2 * r) * (1.0 - decay) - k / (2 * c * r) * (
            rise - decay
        )
        x = (
            x0
            + v0 * (1.0 - decay) / c
            - a / (c**2 * r) * (t - (1.0 - decay) / c)
            - k / (2 * c**2 * r) * (rise + decay - 2.0)
        )
        return np.stack([x, v], axis=-1)

    @property
    def objective(self) -> float:
        """``½ (y(T) - y_target)ᵀ S (y(T) - y_target) + ½ R ∫ u*² dt``."""
        miss = self.state(T_FINAL) - Y_TARGET
        a, r, t = self.a, ENERGY_WEIGHT, T_FINAL

        if self.drag == 0.0:
            b = self.second
            # ∫₀ᵀ (a t - b)² dt, divided by R².
            integral = (a**2 * t**3 / 3.0 - a * b * t**2 + b**2 * t) / r**2
        else:
            c, k = self.drag, self.second
            rise = np.exp(c * t)
            integral = (
                (a / c) ** 2 * t
                + 2.0 * (a / c) * k * (rise - 1.0) / c
                + k**2 * (rise**2 - 1.0) / (2 * c)
            ) / r**2

        return float(0.5 * miss @ TERMINAL_WEIGHT @ miss + 0.5 * r * integral)


def continuous_optimum(drag: float = 0.0) -> ContinuousOptimum:
    """Solve the two-point boundary value problem for its two constants.

    ``y(T)`` is linear in the two unknowns, so the transversality condition
    ``lambda(T) = S (y(T) - y_target)`` is a 2x2 linear system. Its
    coefficients are read off the closed forms in :class:`ContinuousOptimum`.
    """
    x0, v0 = Y0
    xt, vt = Y_TARGET
    s_x, s_v = TERMINAL_WEIGHT[0, 0], TERMINAL_WEIGHT[1, 1]
    r, t = ENERGY_WEIGHT, T_FINAL

    if drag == 0.0:
        # x(T) = x0 + v0 T + (a T³/6 - b T²/2)/R,  v(T) = v0 + (a T²/2 - bT)/R
        # a = s_x (x(T) - xt);  b - aT = s_v (v(T) - vt)
        system = np.array(
            [
                [1.0 - s_x * (t**3 / 6.0) / r, s_x * (t**2 / 2.0) / r],
                [-t - s_v * (t**2 / 2.0) / r, 1.0 + s_v * t / r],
            ]
        )
        rhs = np.array([s_x * (x0 + v0 * t - xt), s_v * (v0 - vt)])
        a, second = np.linalg.solve(system, rhs)
        return ContinuousOptimum(drag=0.0, a=float(a), second=float(second))

    c = drag
    rise, decay = np.exp(c * t), np.exp(-c * t)
    # Coefficients of (a, K) in x(T) and v(T), and the parts independent of them.
    v_a = -(1.0 - decay) / (c**2 * r)
    v_k = -(rise - decay) / (2 * c * r)
    v_0 = v0 * decay
    x_a = -(t - (1.0 - decay) / c) / (c**2 * r)
    x_k = -(rise + decay - 2.0) / (2 * c**2 * r)
    x_0 = x0 + v0 * (1.0 - decay) / c
    # a = s_x (x(T) - xt);  a/c + K e^{cT} = s_v (v(T) - vt)
    system = np.array(
        [
            [1.0 - s_x * x_a, -s_x * x_k],
            [1.0 / c - s_v * v_a, rise - s_v * v_k],
        ]
    )
    rhs = np.array([s_x * (x_0 - xt), s_v * (v_0 - vt)])
    a, second = np.linalg.solve(system, rhs)
    return ContinuousOptimum(drag=c, a=float(a), second=float(second))


# ---------------------------------------------------------------------------
# Reference 1: the discrete optimum, assembled and solved independently
# ---------------------------------------------------------------------------


def step_maps(n_steps: int, drag: float = 0.0) -> tuple[NDArray, NDArray]:
    """``(A, B)`` in ``y_{n+1} = A y_n + B u_n``, from the tableau alone.

    For linear dynamics the stage equations ``Z = 1 ⊗ y + h (Ā ⊗ I)(M Z +
    C U)`` are linear in ``Z``, so the stage solve is one dense inversion and
    the step map follows from ``y_{n+1} = y_n + h (bᵀ ⊗ I)(M Z + C U)``. This
    is the ``r = 1`` Runge-Kutta case of C-8; nothing here calls into
    ``adjungo.stepping``, which is the point.
    """
    method = rk4()
    h = T_FINAL / n_steps
    s = method.s
    state = point_mass(drag)
    m_c, c_c = np.asarray(state.M), np.asarray(state.C)
    n, nu = m_c.shape[0], c_c.shape[1]

    stage_state = np.kron(np.eye(s), m_c)
    stage_control = np.kron(np.eye(s), c_c)
    coupling = np.kron(method.A, np.eye(n))

    lhs = np.eye(s * n) - h * coupling @ stage_state
    d_z_d_y = np.linalg.solve(lhs, np.kron(np.ones((s, 1)), np.eye(n)))
    d_z_d_u = np.linalg.solve(lhs, h * coupling @ stage_control)

    weights = np.kron(method.B[0, :].reshape(1, s), np.eye(n))
    a_map = np.eye(n) + h * weights @ stage_state @ d_z_d_y
    b_map = h * weights @ (stage_state @ d_z_d_u + stage_control)
    assert b_map.shape == (n, s * nu)
    return a_map, b_map


def riccati_control(n_steps: int, drag: float = 0.0) -> NDArray:
    """The exact discrete optimum, by backward Riccati on the stacked stages.

    Treating one step's ``s`` stage controls as a single vector makes the
    discrete problem an ordinary affine-tracking LQR, whose value function is
    ``V_n(y) = ½ yᵀ P_n y + q_nᵀ y + const``. The recursion is

        ``M   = R_hat + Bᵀ P_{n+1} B``
        ``K   = M⁻¹ Bᵀ P_{n+1} A``,   ``k = M⁻¹ Bᵀ q_{n+1}``
        ``P_n = Aᵀ P_{n+1} A - Aᵀ P_{n+1} B K``
        ``q_n = Aᵀ q_{n+1} - Aᵀ P_{n+1} B k``

    started from ``P_N = S`` and ``q_N = -S y_target``, and the optimal
    control is ``u_n = -K_n y_n - k_n`` along the forward roll. Exact in one
    backward and one forward pass: no iteration, and no shared code with the
    optimizer under test.
    """
    a_map, b_map = step_maps(n_steps, drag)
    method = rk4()
    h = T_FINAL / n_steps
    cost = ENERGY_WEIGHT * h * np.diag(method.B[0, :])

    p_next = TERMINAL_WEIGHT.astype(float).copy()
    q_next = -TERMINAL_WEIGHT @ Y_TARGET
    gains: list[tuple[NDArray, NDArray]] = []
    for _ in range(n_steps):
        moment = cost + b_map.T @ p_next @ b_map
        gain = np.linalg.solve(moment, b_map.T @ p_next @ a_map)
        offset = np.linalg.solve(moment, b_map.T @ q_next)
        gains.append((gain, offset))
        p_next, q_next = (
            a_map.T @ p_next @ a_map - a_map.T @ p_next @ b_map @ gain,
            a_map.T @ q_next - a_map.T @ p_next @ b_map @ offset,
        )
    gains.reverse()

    y = Y0.astype(float).copy()
    control = np.zeros((n_steps, method.s, 1))
    for step, (gain, offset) in enumerate(gains):
        stage = -gain @ y - offset
        control[step] = stage.reshape(method.s, 1)
        y = a_map @ y + b_map @ stage
    return control


def discrete_objective(control: NDArray, n_steps: int, drag: float = 0.0) -> float:
    """``J`` evaluated through the independently assembled step map."""
    a_map, b_map = step_maps(n_steps, drag)
    y = Y0.astype(float).copy()
    for step in range(n_steps):
        y = a_map @ y + b_map @ control[step].ravel()
    miss = y - Y_TARGET
    energy = np.einsum("k,nkv->", rk4().B[0, :], control**2)
    return float(
        0.5 * miss @ TERMINAL_WEIGHT @ miss
        + 0.5 * ENERGY_WEIGHT * (T_FINAL / n_steps) * energy
    )


# ---------------------------------------------------------------------------
# The problem as the package sees it
# ---------------------------------------------------------------------------


class TerminalMissAndEnergy:
    """``½ (y_N - y_target)ᵀ S (y_N - y_target) + ½ R h Σ w_k u²``.

    The step size and stage weights are carried here, not by the library; see
    the module docstring and C-9.3.
    """

    def __init__(
        self,
        stage_weights: NDArray,
        step_size: float,
        target: NDArray = Y_TARGET,
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
        energy = np.einsum("k,nkv->", self.stage_weights, u**2)
        return float(
            0.5 * miss @ TERMINAL_WEIGHT @ miss
            + 0.5 * self.energy_weight * self.step_size * energy
        )

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        gradient = np.zeros_like(np.asarray(y_final, dtype=float))
        gradient[0] = TERMINAL_WEIGHT @ (np.asarray(y_final, dtype=float)[0] - self.target)
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


def build_optimizer(n_steps: int = N_STEPS, drag: float = 0.0) -> GLMOptimizer:
    """Assemble the optimizer for the tracking problem on ``n_steps`` steps."""
    method = rk4()
    return GLMOptimizer(
        problem=point_mass(drag),
        objective=TerminalMissAndEnergy(
            stage_weights=method.B[0, :], step_size=T_FINAL / n_steps
        ),
        method=method,
        t_span=(0.0, T_FINAL),
        N=n_steps,
        y0=Y0,
    )


def solve(n_steps: int = N_STEPS, drag: float = 0.0):
    """Minimise from rest with ``trust-ncg`` and the exact Hessian.

    The reduced objective is quadratic with a positive definite Hessian, so
    the Newton direction is the exact step and the iteration terminates in a
    handful of trust-region adjustments rather than converging asymptotically.
    """
    optimizer = build_optimizer(n_steps, drag)
    fun, jac = optimizer.scipy_interface()
    result = minimize(
        fun,
        np.zeros(n_steps * rk4().s),
        jac=jac,
        hessp=optimizer.scipy_hessp(),
        method="trust-ncg",
        options={"maxiter": 200, "gtol": 1e-12},
    )
    return optimizer, result, result.x.reshape(n_steps, rk4().s, 1)


def stage_times(n_steps: int) -> NDArray:
    """``t_{n,k} = t_n + h c_k``, the times the stage controls act at."""
    h = T_FINAL / n_steps
    return np.arange(n_steps)[:, None] * h + h * np.asarray(rk4().c)[None, :]


def main() -> None:
    method = rk4()

    print("Double integrator against a known optimal control")
    print("=" * 62)
    print("  states (x, v)            : 2")
    print(f"  method                   : rk4, {method.s} stages")
    print(f"  horizon T                : {T_FINAL} s")
    print(
        f"  start -> target          : ({Y0[0]:g}, {Y0[1]:g})"
        f" -> ({Y_TARGET[0]:g}, {Y_TARGET[1]:g})"
    )
    print(
        "  weights S, R             : diag("
        f"{TERMINAL_WEIGHT[0, 0]:g}, {TERMINAL_WEIGHT[1, 1]:g}), {ENERGY_WEIGHT:g}"
    )
    print()

    exact = continuous_optimum()
    optimizer, result, u_scipy = solve()
    u_riccati = riccati_control(N_STEPS)
    u_exact = exact.control(stage_times(N_STEPS))[:, :, None]

    print(f"Undamped (c = 0), N = {N_STEPS}:")
    print(f"  continuous J*            : {exact.objective:.15f}")
    print(f"  Riccati    J             : {discrete_objective(u_riccati, N_STEPS):.15f}")
    print(f"  optimizer  J             : {result.fun:.15f}  ({result.nit} iterations)")
    print(f"  |u_scipy - u_riccati|inf : {np.abs(u_scipy - u_riccati).max():.2e}")
    print(f"  |u_riccati - u*(t_nk)|inf: {np.abs(u_riccati - u_exact).max():.2e}")
    print(f"  |grad J at u_riccati|inf : {np.abs(optimizer.gradient(u_riccati)).max():.2e}")
    print()
    print("  The discrete optimum *is* the continuous one: u* is linear in t,")
    print("  rk4 propagates a cubic exactly and its weights are Simpson's rule.")
    print()

    damped = continuous_optimum(DRAG)
    print(f"Damped (c = {DRAG}): the optimal costate is exponential, so")
    print("discretisation error returns and can be refined away.")
    print(f"  continuous J*            : {damped.objective:.15f}")
    previous = None
    for n_steps in (5, 10, 20, 40, 80, 160):
        error = abs(
            discrete_objective(riccati_control(n_steps, DRAG), n_steps, DRAG)
            - damped.objective
        )
        rate = "" if previous is None else f"{np.log2(previous / error):6.2f}"
        print(f"    N = {n_steps:4d}   |J_N - J*| = {error:.3e}   order {rate}")
        previous = error
    print()
    print("  rk4 recovers its fourth order on the optimum, not merely on a solve.")


if __name__ == "__main__":
    main()
