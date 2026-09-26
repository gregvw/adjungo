"""Where the monolithic reference starts its Newton iteration.

:func:`adjungo.validation.reference.reference_solve` is the tier-1 oracle of
the C-14.1 hierarchy, so a change to it is a change to the instrument the rest
of the suite measures against. The change under test here is confined to the
*starting point*: the acceptance test ``||R||_inf <= tol`` is untouched, and
:func:`test_both_starts_accept_the_same_root` holds that invariant.

Holding every step node and every stage at ``y0`` is a constant guess, so its
distance from the root is the whole excursion of the trajectory. Sweeping the
trajectory forward instead is better on an oscillatory problem and worse on a
stiff one, so the implementation builds both and keeps the one with the
smaller residual. Each test below fixes one of those three claims -- the gain,
the fallback, and the safeguard inside the sweep -- so that removing any one
of them fails a test rather than silently narrowing the oracle.

One deliberate gap. Degrading the sweep's *accuracy* -- for instance
evaluating ``f`` at the step time rather than the stage time -- is not tested
and is not intended to be. The sweep is a starting point, not an answer: the
selection below compares it against the alternative and Newton then runs to
the same ``||R||_inf <= tol``, so a worse sweep costs iterations and cannot
change the result. Both fields used here are autonomous, so that particular
edit is not even observable in them.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import ellipk

import adjungo.validation.reference as ref
from adjungo.methods.runge_kutta import gauss2, rk4, sdirk3
from adjungo.validation.reference import (
    _constant_guess,
    _Layout,
    _residual,
    _swept_guess,
    reference_solve,
)

NATURAL_FREQUENCY = 1.3


class UndrivenPendulum:
    """``theta'' = -w0^2 sin(theta)``, written out by hand.

    Undamped and undriven, so the trajectory oscillates forever instead of
    settling. That is the point: a constant initial guess is wrong by the full
    amplitude at every node, and stays wrong no matter how long the horizon.
    """

    state_dim = 2
    control_dim = 1

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([y[1], -(NATURAL_FREQUENCY**2) * np.sin(y[0]) + u[0]])

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array(
            [[0.0, 1.0], [-(NATURAL_FREQUENCY**2) * np.cos(y[0]), 0.0]]
        )

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[0.0], [1.0]])


class DampedPendulum(UndrivenPendulum):
    """The same field with linear drag, so the two starts can be compared."""

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array(
            [
                y[1],
                -(NATURAL_FREQUENCY**2) * np.sin(y[0]) - 0.25 * y[1] + u[0],
            ]
        )

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array(
            [[0.0, 1.0], [-(NATURAL_FREQUENCY**2) * np.cos(y[0]), -0.25]]
        )


class StiffDecay:
    """``y' = lam y``, the scalar problem that defines stiffness.

    Written out by hand, and deliberately run at a step size where an
    explicit sweep is unstable but an implicit method is not -- which is the
    reason one reaches for ``gauss2`` in the first place.
    """

    state_dim = 1
    control_dim = 1

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([self.lam * y[0] + u[0]])

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[self.lam]])

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[1.0]])


def libration_period(amplitude: float) -> float:
    """``4 K(sin^2(amplitude/2)) / w0``, the exact period from rest.

    Closed form, so the horizons below are stated in real oscillations rather
    than in an arbitrary final time that happens to be long.
    """
    return 4.0 * ellipk(np.sin(amplitude / 2.0) ** 2) / NATURAL_FREQUENCY


def _constant_start(
    u: NDArray, y0: NDArray, problem: object, method: object,
    t0: float, h: float, lay: _Layout,
) -> NDArray:
    """The guess the reference used before: everything at ``y0``."""
    return _constant_guess(y0, lay)


def _vdp_fixture() -> tuple[object, NDArray, NDArray, float, int]:
    """The Van der Pol case whose step size makes a forward sweep diverge."""
    from examples.nonlinear_implicit_control import Y0, build_optimizer

    optimizer = build_optimizer()
    n_steps = 5
    rng = np.random.default_rng(13)
    u = rng.standard_normal((n_steps, gauss2().s, 1))
    return optimizer.problem, Y0, u, optimizer.t_span[1], n_steps


def test_sweeping_reaches_a_horizon_the_constant_start_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Six oscillations at ``h=0.98`` converge swept and refuse constant.

    Amplitude 2.8 rad is well outside the small-angle regime, so the period
    depends on the amplitude and a phase error accumulates; over six periods
    the constant guess is far enough from the root that Newton does not
    recover it within the iteration cap. This is a *reach* claim, and the
    failure it guards against is the reference refusing to answer at all.

    The refusal is loud either way -- the pre-change reference raised rather
    than returning an unconverged answer -- so what is repaired here is the
    oracle's range, not a wrong number.
    """
    problem = UndrivenPendulum()
    method = gauss2()
    amplitude = 2.8
    y0 = np.array([amplitude, 0.0])
    n_steps = 60
    t_final = 6.0 * libration_period(amplitude)
    u = np.zeros((n_steps, method.s, 1))

    monkeypatch.setattr(ref, "_initial_guess", _constant_start)
    with pytest.raises(RuntimeError, match="failed to converge"):
        reference_solve(y0, u, (0.0, t_final), n_steps, problem, method)

    monkeypatch.undo()
    solution = reference_solve(
        y0, u, (0.0, t_final), n_steps, problem, method
    )

    assert solution.residual_norm <= 1e-13, (
        "the swept start must reach the same acceptance test, not a weaker "
        f"one; got ||R||_inf = {solution.residual_norm:.3e}"
    )
    # The constant start exhausts all 50 iterations. Anything near that cap
    # would mean the sweep merely delayed the problem rather than removing it.
    assert solution.iterations <= 15, (
        f"expected the swept start to converge promptly, took "
        f"{solution.iterations} iterations"
    )


def test_the_constant_start_is_kept_where_sweeping_is_worse() -> None:
    """On Van der Pol at ``h=1.2`` the chosen guess is the constant one.

    A forward sweep is explicit, so on a problem whose step size exceeds the
    stability limit it amplifies instead of tracking. Adopting it there would
    hand Newton a far worse starting point than it had before: this fixture
    is the one that first exposed the failure, raising ``LinAlgError:
    Singular matrix`` from an unguarded sweep.

    The implementation therefore compares both candidates on the residual
    Newton is about to reduce and keeps the better one. This test fails if
    that comparison is removed.
    """
    problem, y0, u, t_final, n_steps = _vdp_fixture()
    method = gauss2()
    h = t_final / n_steps
    lay = _Layout(
        n_steps, method.s, method.r, problem.state_dim, problem.control_dim
    )

    chosen = ref._initial_guess(u, y0, problem, method, 0.0, h, lay)
    assert np.array_equal(chosen, _constant_guess(y0, lay)), (
        "the swept guess was adopted on a fixture where sweeping is unstable"
    )

    def defect(w: NDArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(
                np.max(np.abs(_residual(w, u, y0, problem, method, 0.0, h, lay)))
            )

    swept = defect(_swept_guess(u, y0, problem, method, 0.0, h, lay))
    constant = defect(_constant_guess(y0, lay))
    assert swept > constant, (
        "this fixture is only meaningful while sweeping is the worse start "
        f"here; measured swept={swept:.3e} against constant={constant:.3e}"
    )

    # And the reference still answers, exactly as it did before the change.
    solution = reference_solve(
        y0, u, (0.0, t_final), n_steps, problem, method
    )
    assert solution.residual_norm <= 1e-13


def test_the_contraction_check_keeps_a_diverging_sweep_finite() -> None:
    """A Picard sweep is accepted only while it shrinks the stage defect.

    The sweep ``Z <- U Y + h A f(Z)`` contracts only while ``h`` times the
    Lipschitz constant of ``f`` is below one. Above it the iteration runs
    away superlinearly: the unguarded version assembled below reaches ``1e+72``
    after two sweeps on this fixture and ``1e+252`` after three, which is
    what produced a singular Newton Jacobian.

    Comparing against that unguarded iteration is what makes this test able
    to fail. Without the check the guarded and unguarded results coincide.
    """
    problem, y0, u, t_final, n_steps = _vdp_fixture()
    method = gauss2()
    h = t_final / n_steps
    lay = _Layout(
        n_steps, method.s, method.r, problem.state_dim, problem.control_dim
    )

    # The same forward sweep with the contraction check removed, for contrast.
    def unguarded_sweep(sweeps: int) -> float:
        Y = np.zeros((lay.N + 1, lay.r, lay.nx))
        Z = np.zeros((lay.N, lay.s, lay.nx))
        Y[0] = y0
        for n in range(lay.N):
            t_stage = np.array([h * (n + method.c[j]) for j in range(lay.s)])
            explicit = np.array(
                [
                    sum(method.U[i, k] * Y[n, k] for k in range(lay.r))
                    for i in range(lay.s)
                ]
            )
            Zn = explicit
            for _ in range(sweeps):
                f_st = np.array(
                    [
                        problem.f(Zn[j], u[n, j], t_stage[j])
                        for j in range(lay.s)
                    ]
                )
                Zn = explicit + h * (method.A @ f_st)
            Z[n] = Zn
            f_st = np.array(
                [problem.f(Zn[j], u[n, j], t_stage[j]) for j in range(lay.s)]
            )
            for lvl in range(lay.r):
                Y[n + 1, lvl] = sum(
                    method.V[lvl, k] * Y[n, k] for k in range(lay.r)
                ) + h * sum(method.B[lvl, j] * f_st[j] for j in range(lay.s))
        return float(max(np.max(np.abs(Y)), np.max(np.abs(Z))))

    with np.errstate(over="ignore", invalid="ignore"):
        unguarded = unguarded_sweep(sweeps=2)
    assert unguarded > 1e60, (
        "this fixture no longer makes an unguarded sweep diverge, so the "
        "guarded comparison below proves nothing; measured "
        f"{unguarded:.3e}"
    )

    with np.errstate(over="ignore", invalid="ignore"):
        guarded = _swept_guess(u, y0, problem, method, 0.0, h, lay)
    assert np.all(np.isfinite(guarded))
    # The guarded sweep stays within a few orders of the state scale rather
    # than running to 1e+72; the bound separates "stopped" from "diverged".
    assert np.max(np.abs(guarded)) < 1e6, (
        "the contraction check did not stop a diverging sweep; peak "
        f"magnitude {np.max(np.abs(guarded)):.3e}"
    )


@pytest.mark.parametrize("method_factory", [gauss2, sdirk3, rk4])
@pytest.mark.parametrize("n_steps", [20, 80])
def test_both_starts_accept_the_same_root(
    method_factory: object, n_steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the start must not change which solution comes back.

    Newton stops at the first iterate inside an absolute ball
    ``||R||_inf <= tol``, so the two answers cannot be bit-identical and are
    not asserted to be. What must hold is that they are the same root.

    The budget separates two outcomes that differ by ten orders of magnitude.
    Two points inside a ``1e-13`` residual ball sit within ``5e-14`` relative
    of each other here, the largest value over these six cases. A sweep that
    delivered Newton to a *different* root -- the real hazard when starting a
    nonlinear iteration somewhere new -- would differ by the scale of the
    trajectory itself, order ``1``. The ``1e-10`` bound is two thousand times
    the observed rounding and eight orders below a distinct root, so it is
    insensitive to the BLAS backend (C-11.3) while still excluding the defect
    it exists to catch.
    """
    problem = DampedPendulum()
    method = method_factory()
    y0 = np.array([0.0, 0.0])
    rng = np.random.default_rng(3)
    u = 0.5 * rng.standard_normal((n_steps, method.s, 1))

    swept = reference_solve(y0, u, (0.0, 6.0), n_steps, problem, method)
    monkeypatch.setattr(ref, "_initial_guess", _constant_start)
    constant = reference_solve(y0, u, (0.0, 6.0), n_steps, problem, method)

    scale = np.max(np.abs(constant.Y))
    displacement = np.max(np.abs(swept.Y - constant.Y)) / scale
    assert displacement < 1e-10, (
        f"the two starting points reached different solutions: relative "
        f"displacement {displacement:.3e} over state scale {scale:.3e}"
    )
    assert swept.residual_norm <= 1e-13
    assert constant.residual_norm <= 1e-13


def test_an_overflowed_sweep_is_rejected_without_a_finiteness_test() -> None:
    """``y' = -200 y`` at ``h=0.1`` overflows the sweep, and is handled.

    The contraction check bounds the stages within a step, but ``Y`` is
    carried from step to step explicitly, and nothing bounds that. At
    ``lam*h = -20`` it grows without limit and reaches ``nan`` before the end
    of the horizon. This is not an exotic input: a step size far outside the
    explicit stability region is the ordinary operating point for an implicit
    method, so the oracle must survive it.

    No finiteness test guards this. A defect is ``max(abs(.))``, so it is
    ``nan`` or lies in ``[0, inf]``, and both ``nan < x`` and ``inf < x`` are
    false -- the ordinary "is the sweep better?" comparison rejects a runaway
    for the same reason it rejects a poor one. This test fixes that behaviour
    so the redundant guard is not reintroduced on the assumption it is needed.
    """
    problem = StiffDecay(-200.0)
    method = gauss2()
    y0 = np.array([1.0])
    n_steps, t_final = 400, 40.0
    h = t_final / n_steps
    u = np.zeros((n_steps, method.s, 1))
    lay = _Layout(n_steps, method.s, method.r, 1, 1)

    with np.errstate(over="ignore", invalid="ignore"):
        swept = _swept_guess(u, y0, problem, method, 0.0, h, lay)
        swept_defect = np.max(
            np.abs(_residual(swept, u, y0, problem, method, 0.0, h, lay))
        )
    assert not np.isfinite(swept_defect), (
        "this fixture no longer overflows the sweep, so it no longer tests "
        f"the path it was built for; measured {swept_defect:.3e}"
    )

    chosen = ref._initial_guess(u, y0, problem, method, 0.0, h, lay)
    assert np.array_equal(chosen, _constant_guess(y0, lay)), (
        "a runaway sweep was adopted as the starting point"
    )

    solution = reference_solve(
        y0, u, (0.0, t_final), n_steps, problem, method
    )
    assert solution.residual_norm <= 1e-13
    # y(40) = exp(-8000) underflows to zero; the decay must not have blown up.
    assert abs(solution.Y[-1][0][0]) < 1e-12, (
        f"stiff decay did not decay: y_N = {solution.Y[-1][0][0]:.3e}"
    )
