"""The stage adjoint carries the costate at Hager's transformed abscissa.

``examples/zermelo_navigation.py`` observes a consequence of this: for that
problem the continuous optimal heading is a stationary point of the discrete
problem exactly when the tableau satisfies ``D(1)``. That is an end-to-end
observation on one example. The mechanism underneath it is a property of
:func:`adjungo.stepping.adjoint.adjoint_solve` itself, and it pins the stage
weight ``Σ_j a_{jk} μ_j`` more sharply than any duality test can (C-14.1: a
duality test passes when the tangent and the adjoint share a mistake). This
module states it at unit level, against a closed-form oracle.

The fixture
-----------

Take ``∂f/∂y = M`` constant and **nilpotent**, ``M² = 0``::

    ẏ₀ = κ y₁ + cos θ,    ẏ₁ = sin θ,    M = [[0, κ], [0, 0]]

with a terminal-only linear cost ``J = w · y(T)``. Two things follow.

*The costate is exact and elementary.* ``λ̇ = -Mᵀλ`` with ``λ(T) = w``, and
``exp(-Mᵀ s) = I - Mᵀ s`` for a nilpotent ``M``, so with ``τ = T - t`` the
time to go,

.. math::  λ(τ) = w + τ\\,M^{\\mathsf T} w,

affine in ``τ``, with no reference to the state or the control.

*The discrete adjoint reproduces it.* Every ``μ`` the stage solve returns lies
in the range of ``Mᵀ``, so ``Mᵀ μ_j = 0`` exactly and the stage equation
collapses to ``μ_i = h b_i Mᵀ λ^{n+1}``. Substituting into the weighted
adjoint ``Λ_k = Σ_j a_{jk} μ_j + b_k λ^{n+1}`` gives

.. math::

    Λ^n_k = b_k λ^{n+1} + h\\Big(\\sum_j b_j a_{jk}\\Big) M^{\\mathsf T} λ^{n+1}
          = b_k\\,λ\\big(τ_n - \\bar c_k h\\big),
    \\qquad
    \\bar c_k = 1 - \\frac{\\sum_j b_j a_{jk}}{b_k},

and the nodal update gives ``λ^n = λ(τ_n)`` using ``Σ_j b_j = 1`` alone. So
the adjoint is exact here for *every* consistent tableau -- but the stage
value sits at ``c̄``, Hager's transformed abscissa (*Numer. Math.* **87**
(2000) 247-282, equations (33) and (52)), not at ``c``. ``D(1)`` is the
condition ``c̄ = c``.

This is what makes the fixture worth its narrowness. Discretisation error
cannot hide the stage *time*: any tableau that placed the stage adjoint
anywhere but ``t_n + c̄_k h`` is wrong by an O(h) amount that no mesh
refinement study would attribute to the adjoint rather than to the method's
order. Four of the eight shipped tableaux violate ``D(1)``, so the ``c``- and
``c̄``-sampled references genuinely disagree and the test has a live negative
control.

What this module cannot see
---------------------------

The same nilpotency is a blind spot, and it is worth naming rather than
discovering later. Every ``μ_j`` lies in the range of ``Mᵀ``, so the
coefficient ``A[j,i]`` in each solver's triangular back-substitution
multiplies a vector that ``Mᵀ`` then annihilates. Transposing it to
``A[i,j]`` in :mod:`adjungo.solvers.explicit`, :mod:`~adjungo.solvers.sdirk`
or :mod:`~adjungo.solvers.dirk` leaves all of the assertions here passing.
The monolithic reference sees it -- injected one at a time, those three
defects fail 46, 34 and 27 tests elsewhere in the suite.

What is pinned here is the *driver's* weight ``Σ_j a_{jk} μ_j`` in
:func:`adjungo.stepping.adjoint.adjoint_solve`, which is not filtered through
``Mᵀ``: transposing that one fails 15 of the tests below. Of fourteen defects
injected across the adjoint driver, all four stage solvers and the gradient
assembler, this module detects eleven.

Because ``G`` depends on the control alone, the whole gradient closes too::

    ∇_{u^n_k} J = h b_k\\,G(θ^n_k)^{\\mathsf T} λ(τ_n - \\bar c_k h),

which :func:`test_gradient_matches_the_closed_form_at_the_transformed_abscissa`
checks through the public :class:`~adjungo.GLMOptimizer` path.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from numpy.typing import NDArray

from adjungo import GLMOptimizer
from adjungo.core.method import GLMethod
from adjungo.methods import runge_kutta
from adjungo.stepping import adjoint_solve

EPS = float(np.finfo(float).eps)

T_SPAN = (0.0, 1.3)

#: ``(κ, w)``. Three shear rates, of both signs and spanning a factor of four,
#: each paired with a terminal weight that makes the costate's varying
#: component move by more than the weight's own magnitude over the horizon.
#: That last condition is the fixture's whole point and
#: :func:`test_the_costate_oracle_solves_the_adjoint_equation` asserts it: a
#: costate that barely moves cannot distinguish one stage time from another.
#: The middle case sends ``λ₁`` through zero at ``τ = 1/12``, which exercises
#: a vanishing component against the max-norm scales used below.
FIXTURES = (
    (0.7, np.array([-1.0, 0.25])),
    (3.0, np.array([2.0, -0.5])),
    (-1.5, np.array([-2.5, 0.4])),
)

MESHES = (1, 3, 8, 21)
SEEDS = (0, 5)

# Basis. Every comparison below is a relative one, against the sum of the
# magnitudes of the terms that formed the reference -- the error model for a
# sum is set by the roundings of its terms, not of its result, and sdirk3's
# b = (1.2085, -0.6444, 0.4359) makes Σ_j b_j a_{jk} a cancelling sum.
#
# Worst observed was 5.69 eps -- for the nodal adjoint, both stage forms and
# the gradient alike -- over a sweep wider than this module runs: five (κ, w)
# fixtures x eight tableaux x meshes N ∈ {1, 2, 3, 8, 21} x four seeds, 800
# configurations. Over the three fixtures retained here it is 4.97 eps. 32 eps
# leaves a factor of 5.6, which is the margin for a different BLAS: CI links
# OpenBLAS and development here links Apple Accelerate, and the implicit stage
# solves are LU-based (C-11.3).
COSTATE_BUDGET = 32 * EPS

# Basis. ``c̄`` is formed by a division and a subtraction from stored
# coefficients, so it is a cancelling expression and may not be compared
# against ``c`` exactly (C-11.3): rewriting gauss2's ``√3/6`` as the
# mathematically identical ``1/(2√3)`` moves such residuals by about one ulp.
# Measured, the four satisfying tableaux give exactly 0 and the four violating
# ones give 0.146, 0.343, 0.5 and 1.0 -- fourteen orders apart, so the budget
# is not a close call.
ABSCISSA_BUDGET = 8 * EPS

# Basis. For a tableau with c̄ ≠ c, the c-sampled reference differs from the
# computed stage adjoint by at least 5.26e-03 relative over the fixtures and
# meshes below, and by at least 2.81e-04 over the wider sweep above. 1e-5 sits
# a factor of 28 below the weaker of those and ten orders above
# COSTATE_BUDGET, so this is a floor and not a fitted threshold.
SEPARATION_FLOOR = 1e-5

# Basis. J is exactly affine in y₀ here, so the central difference carries no
# truncation error and its rounding is eps/ε ≈ 2.2e-13 at ε = 1e-3. Observed
# worst 2.28e-13 relative over the eight tableaux at N ∈ {1, 3, 8, 21}.
Y0_STEP = 1e-3
Y0_FD_RTOL = 1e-10


def _tableaux() -> dict[str, GLMethod]:
    """Every zero-argument tableau constructor in :mod:`.methods.runge_kutta`.

    Discovered rather than listed so that a tableau added later is covered
    without an edit here.
    """
    found = {}
    for name, obj in vars(runge_kutta).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        if inspect.signature(obj).parameters:
            continue
        method = obj()
        if isinstance(method, GLMethod):
            found[name] = method
    return found


TABLEAUX = _tableaux()


def transformed_abscissae(method: GLMethod) -> NDArray:
    """Hager's ``c̄_i = 1 - (Σ_j b_j a_{ji}) / b_i``, equations (33), (52).

    Raises on a vanishing weight rather than substituting a value for one
    (C-7): with ``b_i = 0`` the stage carries no adjoint time at all, and the
    division-free identity is the statement that still applies.
    """
    b = method.B[0]
    if np.any(b == 0.0):
        raise ValueError(
            f"transformed abscissae are undefined for a tableau with a zero "
            f"weight: b = {b}."
        )
    return 1.0 - (b @ method.A) / b


class NilpotentShear:
    """``ẏ₀ = κ y₁ + cos θ``, ``ẏ₁ = sin θ``."""

    state_dim = 2
    control_dim = 1

    def __init__(self, kappa: float):
        self.kappa = float(kappa)
        self.M = np.array([[0.0, self.kappa], [0.0, 0.0]])

    def f(self, y, u, t):
        return np.array([self.kappa * y[1] + np.cos(u[0]), np.sin(u[0])])

    def F(self, y, u, t):
        return self.M.copy()

    def G(self, y, u, t):
        return np.array([[-np.sin(u[0])], [np.cos(u[0])]])

    def F_yy_action(self, y, u, t, v):
        return np.zeros((2, 2))

    def F_yu_action(self, y, u, t, v):
        return np.zeros((2, 1))

    def F_uu_action(self, y, u, t, v):
        return np.array([[-(v[0] * np.cos(u[0]) + v[1] * np.sin(u[0]))]])


class TerminalLinear:
    """``J = w · y(T)``, with no running cost."""

    def __init__(self, w: NDArray):
        self.w = np.asarray(w, dtype=float)

    def evaluate(self, trajectory, u):
        return float(self.w @ trajectory.Y[-1, 0])

    def dJ_dy_terminal(self, y_final):
        grad = np.zeros_like(y_final)
        grad[0] = self.w
        return grad

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return np.zeros_like(u_stage)

    def d2J_du2(self, u_stage, step, stage):
        return np.zeros((len(u_stage), len(u_stage)))

    def d2J_dy2(self, y, step):
        return np.zeros((y.shape[0], y.shape[1], y.shape[1]))

    def d2J_dy2_terminal(self, y_final):
        return np.zeros((y_final.shape[0], y_final.shape[1], y_final.shape[1]))


def costate(tau: float, kappa: float, w: NDArray) -> NDArray:
    """``λ(τ) = w + τ Mᵀ w`` at time-to-go ``τ = T - t``.

    Parametrised by time to go rather than by ``t`` on purpose. The adjoint
    runs backward from ``T``, so ``T - t`` is the natural argument, and
    forming it as a difference of two nearly equal numbers near the terminal
    time would put cancellation into the oracle rather than into the quantity
    under test.
    """
    return w + tau * (np.array([[0.0, kappa], [0.0, 0.0]]).T @ w)


def _setup(kappa, w, method, N, seed):
    """Build the optimizer, a random control and a random initial state."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-2.0, 2.0, size=(N, method.s, 1))
    y0 = rng.uniform(-1.0, 1.0, size=2)
    optimizer = GLMOptimizer(
        NilpotentShear(kappa), TerminalLinear(w), method, T_SPAN, N, y0
    )
    return optimizer, u


def _adjoint(optimizer, u):
    """The adjoint of the dispatched route, not of a reconstructed one."""
    return adjoint_solve(
        optimizer.trajectory(u),
        optimizer.objective,
        optimizer.method,
        optimizer.stage_solver,
        optimizer.h,
    )


def test_the_discovered_tableaux_cover_both_sides_of_d1():
    """Guard against a vacuous negative control.

    Several tests below say "the stage time is ``c̄`` and not ``c``". That
    claim is empty on a tableau satisfying ``D(1)``, where the two coincide.
    It is also empty if discovery silently returns nothing.
    """
    assert len(TABLEAUX) >= 8, f"discovery found only {sorted(TABLEAUX)}"

    satisfying, violating = [], []
    for name, method in TABLEAUX.items():
        assert method.r == 1, f"{name} is not a Runge-Kutta tableau"
        gap = np.abs(transformed_abscissae(method) - method.c)
        relative = (gap / (1.0 + np.abs(method.c))).max()
        (satisfying if relative <= ABSCISSA_BUDGET else violating).append(name)

    assert len(satisfying) >= 2, f"no D(1) tableaux among {sorted(TABLEAUX)}"
    assert len(violating) >= 2, f"no D(1) violators among {sorted(TABLEAUX)}"


def test_the_fixture_jacobian_is_nilpotent():
    """``M² = 0`` is what makes the costate affine, so it is checked here.

    An exact comparison is admissible: ``M`` has a single nonzero entry, so
    every term of ``M²`` is a product with a stored zero and vanishes
    structurally rather than by cancellation (C-11.3).
    """
    for kappa, _ in FIXTURES:
        M = NilpotentShear(kappa).M
        assert np.all(M @ M == 0.0)
        assert np.all(M.T @ M.T == 0.0)
        assert M[0, 1] != 0.0, "a zero shear rate makes the costate constant"


def test_the_costate_oracle_solves_the_adjoint_equation():
    """``λ̇ = -Mᵀλ``, ``λ(T) = w``, and ``λ`` is not constant.

    The derivative is checked exactly rather than by difference: ``λ`` is
    affine in ``τ``, so ``dλ/dt = -Mᵀw``, and ``-Mᵀλ(τ) = -Mᵀw`` because
    ``(Mᵀ)² = 0``. The non-constancy assertion is the one that keeps the rest
    of the module from passing trivially -- if ``Mᵀw`` vanished, every stage
    time would give the same answer.
    """
    for kappa, w in FIXTURES:
        MT = NilpotentShear(kappa).M.T
        assert np.array_equal(costate(0.0, kappa, w), w)
        for tau in (0.0, 0.37, 1.3):
            lam = costate(tau, kappa, w)
            assert np.array_equal(-MT @ lam, -MT @ w)
        drift = costate(T_SPAN[1] - T_SPAN[0], kappa, w) - w
        assert np.abs(drift).max() > 0.1 * np.abs(w).max(), (
            f"kappa={kappa} w={w} moves the costate by only {drift} over the "
            f"horizon, which is too little to separate two stage times"
        )


@pytest.mark.parametrize("name", sorted(TABLEAUX))
def test_the_initial_adjoint_is_the_objective_derivative_in_the_initial_state(
    name,
):
    """``Λ[0] = ∂J/∂y₀``, by central differences on the public objective.

    This fixes the *orientation* of the oracle independently of the adjoint's
    own transpose conventions. Asserting that the closed form satisfies
    ``λ̇ = -Mᵀλ`` cannot do that: a sign or transpose error in ``Mᵀ`` would
    appear identically in the oracle and in the check. The derivative of the
    objective with respect to the initial state is a different characterisation
    of the same object, and it is measured through ``objective_value`` alone.
    """
    method = TABLEAUX[name]
    kappa, w = FIXTURES[0]
    expected = costate(T_SPAN[1] - T_SPAN[0], kappa, w)

    for N in MESHES:
        optimizer, u = _setup(kappa, w, method, N, seed=3)
        fd = np.zeros(2)
        for i in range(2):
            for sign in (+1.0, -1.0):
                y0 = optimizer.y0.copy()
                y0[i] += sign * Y0_STEP
                shifted = GLMOptimizer(
                    optimizer.problem, optimizer.objective, method,
                    T_SPAN, N, y0,
                )
                fd[i] += sign * shifted.objective_value(u)
        fd /= 2.0 * Y0_STEP

        adjoint = _adjoint(optimizer, u)
        scale = np.abs(expected).max()
        assert np.abs(fd - expected).max() <= Y0_FD_RTOL * scale
        assert np.abs(adjoint.Lambda[0, 0] - expected).max() <= (
            COSTATE_BUDGET * scale
        )


@pytest.mark.parametrize("name", sorted(TABLEAUX))
def test_nodal_adjoint_reproduces_the_exact_costate(name):
    """``λ^n = λ(τ_n)`` at every node, for every tableau and every mesh.

    Nothing here is asymptotic. The nodal recursion uses ``Σ_j b_j = 1`` and
    nothing else about the tableau, so a first-order method and a fourth-order
    one are held to the same rounding-level budget.
    """
    method = TABLEAUX[name]
    for kappa, w in FIXTURES:
        MTw = np.array([[0.0, kappa], [0.0, 0.0]]).T @ w
        for N in MESHES:
            for seed in SEEDS:
                optimizer, u = _setup(kappa, w, method, N, seed)
                adjoint = _adjoint(optimizer, u)
                h = optimizer.h
                for n in range(N + 1):
                    tau = (N - n) * h
                    expected = costate(tau, kappa, w)
                    scale = (np.abs(w) + tau * np.abs(MTw)).max()
                    assert np.abs(adjoint.Lambda[n, 0] - expected).max() <= (
                        COSTATE_BUDGET * scale
                    ), f"{name} N={N} seed={seed} node {n}"


@pytest.mark.parametrize("name", sorted(TABLEAUX))
def test_stage_adjoint_is_the_costate_at_the_transformed_abscissa(name):
    """``Λ^n_k = b_k λ(τ_n - c̄_k h)``, in both of its forms.

    The division-free form is the identity the algebra produces; the ``c̄``
    form is its interpretation as a time. Both are asserted, because it is the
    second one that carries the content -- that the stage adjoint is a costate
    *sample*, at a specific instant -- and only the first one is safe on a
    tableau with a vanishing weight.
    """
    method = TABLEAUX[name]
    b = method.B[0]
    bA = b @ method.A
    abs_bA = np.abs(b) @ np.abs(method.A)
    cbar = transformed_abscissae(method)

    for kappa, w in FIXTURES:
        MT = np.array([[0.0, kappa], [0.0, 0.0]]).T
        for N in MESHES:
            for seed in SEEDS:
                optimizer, u = _setup(kappa, w, method, N, seed)
                adjoint = _adjoint(optimizer, u)
                h = optimizer.h
                for n in range(N):
                    lam_next = costate((N - n - 1) * h, kappa, w)
                    MT_lam = MT @ lam_next
                    for k in range(method.s):
                        got = adjoint.WeightedAdj[n, k]
                        direct = b[k] * lam_next + h * bA[k] * MT_lam
                        sampled = b[k] * costate(
                            (N - n - cbar[k]) * h, kappa, w
                        )
                        scale = (
                            abs(b[k]) * np.abs(lam_next)
                            + h * abs_bA[k] * np.abs(MT_lam)
                        ).max()
                        where = f"{name} N={N} seed={seed} step {n} stage {k}"
                        assert np.abs(got - direct).max() <= (
                            COSTATE_BUDGET * scale
                        ), f"division-free form, {where}"
                        assert np.abs(got - sampled).max() <= (
                            COSTATE_BUDGET * scale
                        ), f"c-bar sampled form, {where}"


@pytest.mark.parametrize("name", sorted(TABLEAUX))
def test_the_stage_time_is_c_bar_and_not_c(name):
    """The negative control for the test above.

    On a tableau satisfying ``D(1)`` the two abscissae coincide and there is
    nothing to separate, so this asserts agreement instead. On the four that
    do not, sampling the costate at ``c`` is wrong by an amount that stays
    above :data:`SEPARATION_FLOOR` on every mesh here -- the discrepancy is
    ``h (c̄_k - c_k) b_k Mᵀλ``, which shrinks with ``h`` in absolute terms but
    not relative to the stage adjoint's own scale.
    """
    method = TABLEAUX[name]
    b = method.B[0]
    abs_bA = np.abs(b) @ np.abs(method.A)
    cbar = transformed_abscissae(method)
    gap = (np.abs(cbar - method.c) / (1.0 + np.abs(method.c))).max()
    satisfies_d1 = gap <= ABSCISSA_BUDGET

    worst = 0.0
    for kappa, w in FIXTURES:
        MT = np.array([[0.0, kappa], [0.0, 0.0]]).T
        for N in MESHES:
            for seed in SEEDS:
                optimizer, u = _setup(kappa, w, method, N, seed)
                adjoint = _adjoint(optimizer, u)
                h = optimizer.h
                for n in range(N):
                    lam_next = costate((N - n - 1) * h, kappa, w)
                    MT_lam = MT @ lam_next
                    for k in range(method.s):
                        at_c = b[k] * costate(
                            (N - n - method.c[k]) * h, kappa, w
                        )
                        scale = (
                            abs(b[k]) * np.abs(lam_next)
                            + h * abs_bA[k] * np.abs(MT_lam)
                        ).max()
                        deviation = (
                            np.abs(adjoint.WeightedAdj[n, k] - at_c).max()
                            / scale
                        )
                        if satisfies_d1:
                            assert deviation <= COSTATE_BUDGET, (
                                f"{name} satisfies D(1), so sampling at c must "
                                f"agree: {deviation:.3e} at N={N} step {n} "
                                f"stage {k}"
                            )
                        worst = max(worst, deviation)

    if not satisfies_d1:
        assert worst > SEPARATION_FLOOR, (
            f"{name} violates D(1) by {gap:.3e} in its abscissae, but the "
            f"c-sampled costate is within {worst:.3e} of the stage adjoint, "
            f"so this fixture cannot tell the two times apart"
        )


@pytest.mark.parametrize("name", sorted(TABLEAUX))
def test_gradient_matches_the_closed_form_at_the_transformed_abscissa(name):
    """``∇_{u^n_k} J = h b_k G(θ^n_k)ᵀ λ(τ_n - c̄_k h)``, on the public path.

    The oracle shares no code with :mod:`adjungo.stepping`: it is the
    continuous costate, evaluated at a time the tableau determines, contracted
    with a control Jacobian written out by hand. A gradient this exact is not
    a coincidence of a special control -- the controls are random and the
    dynamics are nonlinear in them.
    """
    method = TABLEAUX[name]
    b = method.B[0]
    bA = b @ method.A
    abs_bA = np.abs(b) @ np.abs(method.A)

    for kappa, w in FIXTURES:
        MT = np.array([[0.0, kappa], [0.0, 0.0]]).T
        for N in MESHES:
            for seed in SEEDS:
                optimizer, u = _setup(kappa, w, method, N, seed)
                gradient = optimizer.gradient(u)
                h = optimizer.h
                for n in range(N):
                    lam_next = costate((N - n - 1) * h, kappa, w)
                    MT_lam = MT @ lam_next
                    for k in range(method.s):
                        theta = u[n, k, 0]
                        G = np.array([-np.sin(theta), np.cos(theta)])
                        expected = h * (
                            b[k] * (G @ lam_next)
                            + h * bA[k] * (G @ MT_lam)
                        )
                        scale = h * (
                            abs(b[k]) * (
                                abs(G[0] * lam_next[0])
                                + abs(G[1] * lam_next[1])
                            )
                            + h * abs_bA[k] * (
                                abs(G[0] * MT_lam[0])
                                + abs(G[1] * MT_lam[1])
                            )
                        )
                        assert abs(gradient[n, k, 0] - expected) <= (
                            COSTATE_BUDGET * scale
                        ), (
                            f"{name} N={N} seed={seed} step {n} stage {k}: "
                            f"{gradient[n, k, 0]!r} vs {expected!r}"
                        )
