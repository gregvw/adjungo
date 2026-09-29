"""Partitioned methods: continuous accuracy under refinement (C-14.5 req. 3).

This is a claim about the **mesh**, and it is deliberately in its own module.
Per C-2/C-4 and the AGENTS.md rule, a derivative test holds the mesh fixed and
an order-of-accuracy test refines it; the two must not share a name or a file,
because a plateau in one is a defect and in the other is the expected floor.

Two requirements, because the natural application witness is degenerate.

*The requirement is on the terminal residual norm, not on the objective.* For
``J ∝ ‖r‖²`` whose optimum has ``r = 0``, a method with ``‖r‖ = O(hᵖ)`` gives
``J = O(h²ᵖ)``. Written on ``J``, a second-order method would appear fourth
order and the clause would be satisfied by the wrong evidence.

*The horizon is detuned.* At ``T = 2π·15`` the bare ramp already solves the
problem exactly, every even mode lies in the nullspace, and the position row of
the residual map vanishes: the optimum is a seven-dimensional affine set
containing ``θ = 0``, and any method that stays near it scores arbitrarily well
for a reason that is not accuracy. That case is retained below only as a
labelled check on the degeneracy itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from adjungo.core.partitioned import verlet
from adjungo.core.plan import DiscretizationPlan
from adjungo.methods.runge_kutta import gauss2, implicit_trapezoid, rk4
from adjungo.optimization.interface import GLMOptimizer
from tests.prk_problems import (
    EXCITATION_MODES,
    DrivenOscillator,
    TerminalExcitation,
    excitation,
    oracle_terminal_map,
)

DETUNED = 2.0 * np.pi * 15.25
RESONANT = 2.0 * np.pi * 15.0
TARGET = np.array([1.0, 0.0])


def _terminal_map(method_factory, T: float, N: int):
    """``θ -> r``, the discrete terminal residual of one mesh.

    The problem and the control basis are both affine in ``θ``, so the map is
    affine and ``K + 1`` trajectory evaluations determine it exactly.
    """
    method = method_factory()
    plan = DiscretizationPlan.uniform((0.0, T), N, method)
    optimizer = GLMOptimizer(
        DrivenOscillator(),
        TerminalExcitation(),
        y0=np.zeros(2),
        plan=plan,
    )
    h = T / N

    def residual(theta: np.ndarray) -> np.ndarray:
        x0 = excitation(theta, T)
        u = np.array(
            [
                [[x0(n * h + method.c[k] * h)] for k in range(method.s)]
                for n in range(N)
            ]
        )
        return optimizer.trajectory(u).Y[-1, 0] - TARGET

    d = residual(np.zeros(EXCITATION_MODES))
    R = np.column_stack(
        [
            residual(np.eye(EXCITATION_MODES)[j]) - d
            for j in range(EXCITATION_MODES)
        ]
    )
    return R, d


def _continuous_error(method_factory, T: float, N: int) -> tuple[float, float]:
    """Fit ``θ`` on the mesh's own map; score it against the oracle.

    Returns ``(discrete residual, continuous residual)``. The first is
    reported so that a near-zero discrete fit cannot be mistaken for
    accuracy: it is a property of an eight-parameter basis against a rank-two
    map, not of the method.
    """
    R_h, d_h = _terminal_map(method_factory, T, N)
    theta = -np.linalg.pinv(R_h) @ d_h
    R, d = oracle_terminal_map(T)
    return (
        float(np.linalg.norm(R_h @ theta + d_h)),
        float(np.linalg.norm(R @ theta + d)),
    )


# =====================================================================
# The oracle
# =====================================================================


def test_the_oracle_integrals_match_adaptive_quadrature():
    """The closed forms in ``oracle_terminal_map`` are the stated integrals.

    Adaptive quadrature shares no derivation with the closed forms and no
    discretization with any method under test. It checks the *arithmetic*;
    what establishes the oracle is that ``z(T) = i e^{-iT} ∫₀ᵀ e^{it} x₀ dt``
    satisfies ``z' = -i z + i x₀`` with ``z(0) = 0``, which is a derivation.

    Budget: ``1e-11`` absolute, against a measured worst discrepancy of
    ``4.3e-14`` over all nine columns at the detuned horizon. The margin
    covers the quadrature's own error over 15 oscillation periods, which is
    the larger of the two contributions.
    """
    T = DETUNED
    R, d = oracle_terminal_map(T)

    def z(fn) -> complex:
        re = quad(lambda t: np.cos(t) * fn(t), 0.0, T, limit=800)[0]
        im = quad(lambda t: np.sin(t) * fn(t), 0.0, T, limit=800)[0]
        return 1j * np.exp(-1j * T) * (re + 1j * im)

    for k in range(1, EXCITATION_MODES + 1):
        got = z(lambda t, k=k: np.sin(k * np.pi * t / T))
        assert abs(got - (R[0, k - 1] + 1j * R[1, k - 1])) < 1e-11, k
    assert abs(z(lambda t: t / T) - 1.0 - (d[0] + 1j * d[1])) < 1e-11


def test_gauss2_converges_toward_the_oracle_at_fourth_order():
    """Corroboration only (C-14.5): a shared error would pass this too.

    ``gauss2`` is fourth order, so its discrepancy against the oracle must
    fall by ~16 per halving. Observed here and recorded in C-14.5: ``5.55e-05``
    at 32 steps per period and ``3.48e-06`` at 64, a factor of ``15.97``. The
    factor is what is asserted; the individual values are platform-dependent
    at this precision and are not pinned (C-11.3).
    """
    T = DETUNED
    R, d = oracle_terminal_map(T)
    errors = []
    for steps_per_period in (32, 64):
        steps = round(15.25 * steps_per_period)
        R_h, d_h = _terminal_map(gauss2, T, steps)
        errors.append(
            max(np.abs(R - R_h).max(), np.abs(d - d_h).max())
        )
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.05)


# =====================================================================
# Requirement 3 -- the detuned horizon
# =====================================================================


def test_the_detuned_horizon_is_not_degenerate():
    """Rank 2 and ``‖d‖ != 0``: the fit has something to get wrong.

    The premise of the refinement test below. Without it a passing order
    measurement would say nothing, which is exactly what happens at the
    resonant horizon.
    """
    R, d = oracle_terminal_map(DETUNED)
    assert np.linalg.matrix_rank(R, tol=1e-12) == 2
    assert np.linalg.norm(d) == pytest.approx(1.476e-02, rel=1e-3)


def test_verlet_converges_at_second_order_against_the_continuous_oracle():
    """``‖r‖ = O(h²)`` for Verlet at the detuned horizon (C-4, req. 3).

    On the *terminal residual norm*, never the objective. Observed, and
    recorded in C-14.5: ``3.379e-04``, ``8.381e-05``, ``2.091e-05`` at 60,
    120 and 240 steps, orders ``2.01`` and ``2.00``.

    The mesh is refined here and only here. A discrepancy in this test is a
    statement about the method's order; a discrepancy in
    ``tests/test_partitioned_methods.py`` never is.
    """
    errors = []
    for N in (60, 120, 240):
        discrete, continuous = _continuous_error(verlet, DETUNED, N)
        assert discrete < 1e-12, (
            f"the affine fit did not attain its zero at N={N}: {discrete}. "
            f"The refinement measurement below assumes it does, so that the "
            f"only thing separating the meshes is the method's own error."
        )
        errors.append(continuous)
    orders = [
        np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)
    ]
    assert all(1.9 < p < 2.1 for p in orders), (
        f"Verlet residual norms {errors} give orders {orders}, not 2"
    )


@pytest.mark.parametrize(
    "method_factory,expected",
    [
        pytest.param(implicit_trapezoid, 2, id="implicit_trapezoid"),
        pytest.param(rk4, 4, id="rk4"),
    ],
)
def test_certified_methods_converge_at_their_own_order_on_this_fixture(
    method_factory, expected
):
    """The fixture measures order, not Verlet in particular.

    Without this, a fixture that reported ``2`` for everything -- because of
    an error in the oracle, the fit or the control sampling -- would certify
    Verlet by accident. ``rk4`` separates the possibilities: it must report
    ``4`` here, and any mechanism that forced ``2`` would report ``2``.

    Measured on the finer pair only for ``rk4``: its 60-step error
    (``1.81e-02``) is preasymptotic, larger than Verlet's, and the pair
    ``(60, 120)`` gives an order of ``11.6`` rather than ``4``.
    """
    meshes = (60, 120, 240) if expected == 2 else (120, 240, 480)
    errors = [
        _continuous_error(method_factory, DETUNED, N)[1] for N in meshes
    ]
    orders = [
        np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)
    ]
    assert all(abs(p - expected) < 0.2 for p in orders), (
        f"{method_factory.__name__} residual norms {errors} give orders "
        f"{orders}, not {expected}"
    )


def test_verlet_is_more_accurate_than_rk4_on_the_coarse_mesh():
    """An observed comparison on one coarse mesh, and nothing more.

    At 60 steps Verlet's residual is far smaller than fourth-order ``rk4``'s,
    which is a fact about this fixture at this resolution and not a general
    claim about symplectic methods. It is asserted because it is the
    application-relevant regime -- and bounded loosely, because it is not
    grounds to weaken requirement 3 (C-14.5).
    """
    verlet_error = _continuous_error(verlet, DETUNED, 60)[1]
    rk4_error = _continuous_error(rk4, DETUNED, 60)[1]
    assert rk4_error / verlet_error > 10.0


# =====================================================================
# The resonant horizon -- a labelled degeneracy check, not an accuracy one
# =====================================================================


def test_the_resonant_horizon_is_structurally_degenerate():
    """``T = 2π·15``: rank 1, ``d = 0``, even modes in the nullspace.

    This is the whole content of the resonant case. It is recorded as a
    check on the degeneracy so that the reason requirement 3 detunes the
    horizon is itself verified, rather than being an assertion in prose that
    nothing can contradict.
    """
    R, d = oracle_terminal_map(RESONANT)
    assert np.linalg.matrix_rank(R, tol=1e-12) == 1
    assert np.linalg.norm(d) < 1e-14
    # The position row vanishes identically: q(T) is already on target.
    assert np.abs(R[0]).max() < 1e-13
    # Every even mode is annihilated.
    assert np.abs(R[:, 1::2]).max() < 1e-13


def test_the_resonant_horizon_cannot_measure_an_order():
    """At ``T = 2π·15`` Verlet's error is at rounding and does not refine.

    The negative control for requirement 3, and the reason the horizon is
    detuned. It is *not* that every method scores zero here -- ``rk4`` scores
    ``6.6e-03``, worse than Verlet does at the detuned horizon. It is that
    for a method which stays near the ``θ = 0`` member of the optimal affine
    set, the continuous residual sits at the rounding floor at every mesh, so
    refining tells you nothing about the method's order.

    Observed: ``7.5e-16``, ``8.5e-16``, ``6.1e-16`` at 60, 120 and 240 steps
    -- an apparent order of ``-0.19`` and then ``0.47``. A clause written on
    this case would certify or reject a method by rounding noise.
    """
    errors = [
        _continuous_error(verlet, RESONANT, N)[1] for N in (60, 120, 240)
    ]
    assert all(e < 1e-13 for e in errors), (
        f"Verlet no longer sits at the rounding floor at the resonant "
        f"horizon: {errors}. The degeneracy argument assumes it does."
    )
    orders = [
        np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)
    ]
    assert not all(1.9 < p < 2.1 for p in orders), (
        f"the resonant case reported orders {orders}, which look like a "
        f"genuine second-order measurement. It must not: the quantity it "
        f"measures is at the rounding floor, and a clause written on it "
        f"would be satisfied by noise."
    )
