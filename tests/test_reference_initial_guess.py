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
evaluating ``f`` at the step time rather than the stage time -- is not
tested. Both fields used in that comparison are autonomous, so the edit is
not observable in them at all, and the acceptance test still guarantees that
whatever returns is a root of the correct discrete system. It does not
guarantee that it is the *same* root: a different starting point can in
principle reach a different one, and no residual test can exclude that. The
gap is left open knowingly rather than argued away.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import ellipk

import adjungo.validation.reference as ref
from adjungo.core.method import GLMethod
from adjungo.core.plan import DiscretizationPlan
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
    u: NDArray, y0: NDArray, problem: object, lay: _Layout,
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
    lay = _Layout(
        DiscretizationPlan.uniform((0.0, t_final), n_steps, method),
        problem.state_dim,
        problem.control_dim,
    )
    u = lay.pack_stage_input(u, "control")

    chosen = ref._initial_guess(u, y0, problem, lay)
    assert np.array_equal(chosen, _constant_guess(y0, lay)), (
        "the swept guess was adopted on a fixture where sweeping is unstable"
    )

    def defect(w: NDArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(
                np.max(np.abs(_residual(w, u, y0, problem, lay)))
            )

    swept = defect(_swept_guess(u, y0, problem, lay))
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
    lay = _Layout(
        DiscretizationPlan.uniform((0.0, t_final), n_steps, method),
        problem.state_dim,
        problem.control_dim,
    )
    u = lay.pack_stage_input(u, "control")

    # The same forward sweep with the contraction check removed, for contrast.
    def unguarded_sweep(sweeps: int) -> float:
        h = lay.plan.step_size(0)
        Y = np.zeros((lay.N + 1, lay.r, lay.nx))
        Z = np.zeros((lay.total_stages, lay.nx))
        Y[0] = y0
        for n in range(lay.N):
            s = lay.s_at(n)
            un = lay.stage(u, n)
            t_stage = np.array([h * (n + method.c[j]) for j in range(s)])
            explicit = np.array(
                [
                    sum(method.U[i, k] * Y[n, k] for k in range(lay.r))
                    for i in range(s)
                ]
            )
            Zn = explicit
            for _ in range(sweeps):
                f_st = np.array(
                    [
                        problem.f(Zn[j], un[j], t_stage[j])
                        for j in range(s)
                    ]
                )
                Zn = explicit + h * (method.A @ f_st)
            lay.stage(Z, n)[...] = Zn
            f_st = np.array(
                [problem.f(Zn[j], un[j], t_stage[j]) for j in range(s)]
            )
            for lvl in range(lay.r):
                Y[n + 1, lvl] = sum(
                    method.V[lvl, k] * Y[n, k] for k in range(lay.r)
                ) + h * sum(method.B[lvl, j] * f_st[j] for j in range(s))
        return float(max(np.max(np.abs(Y)), np.max(np.abs(Z))))

    with np.errstate(over="ignore", invalid="ignore"):
        unguarded = unguarded_sweep(sweeps=2)
    assert unguarded > 1e60, (
        "this fixture no longer makes an unguarded sweep diverge, so the "
        "guarded comparison below proves nothing; measured "
        f"{unguarded:.3e}"
    )

    with np.errstate(over="ignore", invalid="ignore"):
        guarded = _swept_guess(u, y0, problem, lay)
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

    The budget is set by measurement on these six cases, not by a bound. The
    largest observed displacement is ``5e-14`` relative, and ``1e-10`` sits
    two thousand times above it, far enough to be insensitive to the BLAS
    backend (C-11.3) and still far below any disagreement a reader would
    call a different answer.

    It is worth being exact about what this does and does not establish. A
    residual ball is not a state-error ball: the two are related through
    ``||(dR/dw)^-1||``, which is not bounded here and grows with the horizon.
    Nor need two distinct roots of a nonlinear system be far apart. So this
    is evidence that the starting point does not move the answer *on these
    fixtures*, which is what the consumers of this oracle run, and it is not
    a proof that it cannot do so in general.
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
    u = np.zeros((n_steps, method.s, 1))
    lay = _Layout(
        DiscretizationPlan.uniform((0.0, t_final), n_steps, method), 1, 1
    )
    u = lay.pack_stage_input(u, "control")

    with np.errstate(over="ignore", invalid="ignore"):
        swept = _swept_guess(u, y0, problem, lay)
        swept_defect = np.max(
            np.abs(_residual(swept, u, y0, problem, lay))
        )
    assert not np.isfinite(swept_defect), (
        "this fixture no longer overflows the sweep, so it no longer tests "
        f"the path it was built for; measured {swept_defect:.3e}"
    )

    chosen = ref._initial_guess(u, y0, problem, lay)
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


class DrainingTank:
    """``y' = -sqrt(y)``: Torricelli's law, with the domain guarded.

    ``math.sqrt`` raises on a negative argument rather than returning ``nan``,
    which is what C-7 asks a callback to do instead of inventing a value. The
    closed form ``y(t) = (1 - t/2)^2`` is positive for ``t < 2``, so nothing
    about the problem or the horizon below leaves the domain -- only a trial
    state does.
    """

    state_dim = 1
    control_dim = 1

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([-math.sqrt(y[0]) + u[0]])

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[-0.5 / math.sqrt(y[0])]])

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[1.0]])


def backward_euler() -> GLMethod:
    """Backward Euler as a one-stage, one-value GLM."""
    return GLMethod(
        A=np.array([[1.0]]),
        U=np.array([[1.0]]),
        B=np.array([[1.0]]),
        V=np.array([[1.0]]),
        c=np.array([1.0]),
    )


def test_a_predictor_that_raises_is_a_rejected_candidate() -> None:
    """A domain error in the sweep must not take the solve down with it.

    The predictor evaluates ``f`` at trial states the solution never visits.
    From ``y0 = 1`` at ``h = 1.5`` the first Picard sweep asks for
    ``f(-0.5)``, and a callback that guards its domain raises there -- while
    both the continuous solution and the discrete root stay positive. Any
    failure of an optional predictor has to degrade to the starting point
    that was used before it existed.

    The root is known independently: backward Euler on this field solves
    ``y1 + h sqrt(y1) - y0 = 0``, and ``0.25 + 1.5(0.5) - 1 = 0`` exactly, so
    this checks the answer and not merely the absence of an exception.
    """
    problem = DrainingTank()
    method = backward_euler()
    y0 = np.array([1.0])
    u = np.zeros((1, 1, 1))
    lay = _Layout(DiscretizationPlan.uniform((0.0, 1.5), 1, method), 1, 1)
    u = lay.pack_stage_input(u, "control")

    with pytest.raises(ValueError):
        _swept_guess(u, y0, problem, lay)

    chosen = ref._initial_guess(u, y0, problem, lay)
    assert np.array_equal(chosen, _constant_guess(y0, lay))

    solution = reference_solve(y0, u, (0.0, 1.5), 1, problem, method)
    assert solution.Y[-1][0][0] == pytest.approx(0.25, abs=1e-14), (
        f"expected the known discrete root 0.25, got {solution.Y[-1][0][0]!r}"
    )


class JacobianRefuses(UndrivenPendulum):
    """``f`` is well behaved; ``F`` refuses.

    The solve therefore gets past assembling a residual and fails inside the
    Newton update, which is the second of the two places an error can arise
    and the one a predictor-shaped guard would not reach.
    """

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        raise ValueError("Jacobian unavailable")


@pytest.mark.parametrize(
    "problem, y0, label",
    [
        (DrainingTank(), np.array([-1.0]), "residual evaluation"),
        (JacobianRefuses(), np.array([0.4, 0.0]), "Newton update"),
    ],
)
def test_a_failing_newton_solve_still_raises(
    problem: object, y0: NDArray, label: str
) -> None:
    """The fallback covers the predictor only, never the solve itself.

    Catching broadly around an optional predictor is safe, because its
    failure has a correct answer -- use the other candidate. Catching around
    the iteration would convert a genuine failure into a silent wrong answer,
    which is the defect class C-7 exists to prevent.

    Both places an error can arise are covered. ``DrainingTank`` from a
    negative state fails while the residual is assembled, before any update.
    ``JacobianRefuses`` has a working ``f``, so it reaches the update and
    fails there instead; without it, a guard wrapped around the update would
    go unnoticed.
    """
    method = backward_euler() if problem.state_dim == 1 else gauss2()
    u = np.zeros((1, method.s, problem.control_dim))
    with pytest.raises(ValueError):
        reference_solve(y0, u, (0.0, 1.5), 1, problem, method)


def test_the_contraction_bound_carries_the_tableau() -> None:
    """``h ||A|| L < 1``, not ``h L < 1``.

    The sweep iterates ``Z -> explicit + h A f(Z)``, so its Lipschitz
    constant carries ``||A||``. Dropping it understates the range badly, and
    this fixture is the counterexample: implicit midpoint has
    ``||A||_inf = 1/2``, so ``y' = -1.5 y`` at ``h = 1`` contracts by exactly
    ``h ||A|| L = 0.75`` per sweep while ``h L = 1.5`` would predict growth.

    This pins the statement the docstrings and C-14.3 make, so that the
    weaker claim cannot quietly return.
    """
    A = np.array([[0.5]])
    lam, h = -1.5, 1.0
    lipschitz = abs(lam)
    assert h * lipschitz > 1.0, "fixture must violate the tableau-free bound"
    a_norm = float(np.abs(A).sum(axis=1).max())
    assert h * a_norm * lipschitz < 1.0, "fixture must satisfy the real bound"

    explicit = 1.0
    Zn = explicit
    ratios = []
    defect = abs(Zn - explicit - h * A[0, 0] * lam * Zn)
    for _ in range(5):
        Zn = explicit + h * A[0, 0] * lam * Zn
        nxt = abs(Zn - explicit - h * A[0, 0] * lam * Zn)
        ratios.append(nxt / defect)
        defect = nxt

    assert np.allclose(ratios, h * a_norm * lipschitz, rtol=1e-12), (
        f"expected every sweep to contract by {h * a_norm * lipschitz}, "
        f"measured {ratios}"
    )


def test_the_contraction_bound_is_sufficient_and_not_necessary() -> None:
    """Failing ``h ||A|| L < 1`` does not mean the sweep amplifies.

    The bound is a norm estimate of the iteration matrix
    ``K = h A (x) M``, and for a non-normal ``K`` a norm badly overstates
    what the iteration does. This fixture is the counterexample: implicit
    midpoint at ``h = 1`` with ``M = [[-1/4, 2], [0, -1/4]]`` has
    ``h ||A||_inf L = 9/8 > 1`` while ``rho(K) = 1/8``, and every measured
    defect falls.

    It exists so that "the condition is sufficient" is not quietly upgraded
    to "the condition decides", which would make the amplification claim in
    C-14.3 an implication it cannot support. The genuine amplification
    recorded there is an observation on one fixture, not a consequence of
    this bound failing.
    """
    A = np.array([[0.5]])
    h = 1.0
    M = np.array([[-0.25, 2.0], [0.0, -0.25]])
    lipschitz = float(np.abs(M).sum(axis=1).max())
    a_norm = float(np.abs(A).sum(axis=1).max())
    assert h * a_norm * lipschitz > 1.0, "fixture must fail the sufficient bound"

    K = h * A[0, 0] * M
    spectral_radius = float(max(abs(np.linalg.eigvals(K))))
    assert spectral_radius < 1.0, "fixture must still converge asymptotically"

    explicit = np.array([1.0, 1.0])
    Zn = explicit.copy()
    defects = []
    for _ in range(8):
        defects.append(float(np.max(np.abs(Zn - explicit - K @ Zn))))
        Zn = explicit + K @ Zn

    assert all(b < a for a, b in itertools.pairwise(defects)), (
        f"defects did not fall monotonically despite rho(K)={spectral_radius}: "
        f"{defects}"
    )
    # The contrast is the point: the norm bound permits growth by 1.125 per
    # sweep, and the iteration instead shrinks by roughly rho(K) = 1/8. The
    # observed average factor is ~0.17, between the two and far from the
    # bound, so 0.25 separates "contracts" from "does what the bound allows"
    # with the whole of that gap to spare.
    observed = (defects[-1] / defects[0]) ** (1.0 / (len(defects) - 1))
    assert observed < 0.25, (
        f"observed contraction factor {observed:.4f} does not sit below the "
        f"norm bound {h * a_norm * lipschitz:.4f}, so this fixture does not "
        f"demonstrate that the bound is pessimistic"
    )
    assert h * a_norm * lipschitz > 1.0
