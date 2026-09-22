"""Structural acceptance tests for the implicit stage solvers (milestone M2).

These tests cover properties of the DIRK/SDIRK solve that a gradient
comparison on the C-14 population does not reach:

* the factorization an adjoint consumes is the one its own stage produced
  (precedent R-9);
* a stage equation that does not converge refuses rather than returning its
  last iterate (clause C-7).
"""

import numpy as np
import pytest

from adjungo.core.problem import Linearity, ProblemStructure
from adjungo.methods.runge_kutta import (
    implicit_midpoint,
    implicit_trapezoid,
    sdirk2,
    sdirk3,
)
from adjungo.optimization.interface import GLMOptimizer
from adjungo.solvers.newton import (
    NewtonMixin,
    StageSolveError,
    stage_solve_tolerance,
)
from adjungo.validation import reference_gradient

IMPLICIT_FACTORIES = [
    pytest.param(implicit_midpoint, id="implicit_midpoint"),
    pytest.param(implicit_trapezoid, id="crank_nicolson"),
    pytest.param(sdirk2, id="sdirk2"),
    pytest.param(sdirk3, id="sdirk3"),
]


class StateOnlyJacobian:
    """``f = y^2 + u``: the Jacobian depends on the state and nothing else.

    This is the discriminating problem for precedent R-9. ``F = 2y`` varies
    from stage to stage because the stage values differ, but ``F`` is
    constant in ``u`` and in ``t``. A reuse probe that varies only ``u`` and
    ``t`` therefore sees no variation and wrongly concludes that one
    factorization serves every stage.

    ``G = 1`` is nonzero so the gradient with respect to ``u`` is not
    degenerate.
    """

    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return np.array([y[0] ** 2 + u[0]])

    def F(self, y, u, t):
        return np.array([[2.0 * y[0]]])

    def G(self, y, u, t):
        return np.array([[1.0]])


class TerminalSquareObjective:
    """``J = y_N^2 / 2``: terminal only, so the gradient isolates the adjoint."""

    def J(self, Y, u):
        return 0.5 * float(Y[-1][0] ** 2)

    def dJ_dy_terminal(self, y):
        return np.array([y[0]])

    def dJ_dy(self, y, n):
        return np.zeros(1)

    def dJ_du(self, u, n, k):
        return np.zeros(1)

    def d2J_dy2_terminal(self, y):
        return np.ones((1, 1))

    def d2J_dy2(self, y, n):
        return np.zeros((1, 1))

    def d2J_du2(self, u, n, k):
        return np.zeros((1, 1))


def _state_only_case(method_factory, N=4):
    method = method_factory()
    problem = StateOnlyJacobian()
    objective = TerminalSquareObjective()
    t_span = (0.0, 0.4)
    y0 = np.array([0.5])
    u = 0.3 * np.ones((N, method.s, 1))
    return problem, objective, method, t_span, N, y0, u


# ---------------------------------------------------------------------------
# R-9: each stage's adjoint uses its own matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_factory", IMPLICIT_FACTORIES)
def test_stage_factorizations_are_not_shared_between_stages(method_factory):
    """No two implicit stages may publish the same factorization object.

    Object identity is the check, not numerical closeness: the defect this
    pins was introduced by assigning one stage's LU object to another. If a
    future change reintroduces sharing, this fails even on a problem where
    the shared matrix happens to be numerically right, which is the only way
    to catch it before it reaches a problem where it is wrong.
    """
    problem, objective, method, t_span, N, y0, u = _state_only_case(
        method_factory
    )
    opt = GLMOptimizer(problem, objective, method, t_span, N, y0)
    opt._ensure_forward(u)

    for step, cache in enumerate(opt._trajectory.caches):
        factorizations = cache.stage_factorizations
        if factorizations is None:
            continue
        live = [lu for lu in factorizations if lu is not None]
        identities = [id(lu) for lu in live]
        assert len(set(identities)) == len(identities), (
            f"step {step}: {len(identities)} implicit stages published only "
            f"{len(set(identities))} distinct factorization objects. A shared "
            f"factorization gives the adjoint the transpose of another "
            f"stage's matrix (NUMERICS.md C-5.4, precedent R-9)."
        )


@pytest.mark.parametrize("method_factory", IMPLICIT_FACTORIES)
def test_state_dependent_jacobian_gradient_matches_reference(method_factory):
    """The R-9 reproducer, checked against the independent reference.

    Pre-cure this measured a relative error of 4.24e-05 for ``sdirk2`` and
    4.26e-05 for ``sdirk3``. Those are the numbers that make the identity
    test above load-bearing rather than cosmetic.
    """
    problem, objective, method, t_span, N, y0, u = _state_only_case(
        method_factory
    )
    grad_pkg = GLMOptimizer(
        problem, objective, method, t_span, N, y0
    ).gradient(u)
    grad_ref = reference_gradient(
        y0, u, t_span, N, problem, method, objective
    )

    assert np.max(np.abs(grad_ref)) > 1e-3, "degenerate case: gradient is ~0"
    err = float(np.max(np.abs(grad_pkg - grad_ref))) / max(
        float(np.max(np.abs(grad_ref))), 1.0
    )
    assert err < 1e-9, (
        f"state-dependent Jacobian: gradient differs from the monolithic "
        f"reference by {err:.6e} (relative). A stage is solving its adjoint "
        f"with another stage's matrix (precedent R-9)."
    )


# ---------------------------------------------------------------------------
# C-7: a stage that does not converge refuses
# ---------------------------------------------------------------------------


def test_newton_raises_instead_of_returning_an_unconverged_iterate():
    """C-7 forbids a silent sentinel; an unconverged stage value is one.

    ``r(z) = z^3 - 2`` from ``z_0 = 50`` with a three-iteration budget. The
    Jacobian ``3 z^2`` stays well away from singular throughout, so the
    failure is genuinely "the tolerance was not reached in the allotted
    iterations" and not an arithmetic breakdown that would raise anyway.
    Returning the final iterate would make every downstream derivative the
    derivative of a different discrete map.
    """
    solver = NewtonMixin()

    with pytest.raises(StageSolveError) as excinfo:
        solver.newton_solve(
            residual_fn=lambda z: z**3 - 2.0,
            jacobian_fn=lambda z: np.atleast_2d(3.0 * z[0] ** 2),
            z0=np.array([50.0]),
            max_iter=3,
            context="deliberately truncated stage",
        )

    err = excinfo.value
    assert err.iterations == 3
    assert err.residual > err.tol
    assert "deliberately truncated stage" in str(err)


def test_newton_reports_the_stage_context_on_failure():
    """The refusal must name the stage and time, not just fail.

    A bare convergence error in a multi-stage, multi-step integration is not
    actionable; the message has to locate the stage that failed.

    The stage equation ``Z - h γ e^Z = rhs`` has **no solution** when ``rhs``
    exceeds ``max_Z (Z - h γ e^Z)``. Starting from ``y_0 = 5`` with
    ``h γ ≈ 0.029`` puts ``rhs`` above that maximum, so no iteration count
    would succeed: the solver must refuse rather than return an iterate.
    """
    method = sdirk2()
    N = 4

    class NoStageSolution:
        """``f = e^y + u``: the stage equation has no root for large ``y``."""

        state_dim = 1
        control_dim = 1

        def f(self, y, u, t):
            return np.array([np.exp(y[0]) + u[0]])

        def F(self, y, u, t):
            return np.array([[np.exp(y[0])]])

        def G(self, y, u, t):
            return np.array([[1.0]])

    opt = GLMOptimizer(
        NoStageSolution(),
        TerminalSquareObjective(),
        method,
        (0.0, 0.4),
        N,
        np.array([5.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
        ),
    )

    with pytest.raises(StageSolveError) as excinfo:
        opt._ensure_forward(np.zeros((N, method.s, 1)))

    message = str(excinfo.value)
    # C-5.3 requires all four locators, not merely a failure.
    assert "step 0" in message, "step index missing (C-5.3)"
    assert "stage 0" in message, "stage index missing (C-5.3)"
    assert "t=" in message, "stage time missing"
    assert "||r||_inf" in message, "final residual norm missing (C-5.3)"
    assert "50 iterations" in message, "iteration count missing (C-5.3)"


# ---------------------------------------------------------------------------
# C-5.1: the convergence test is scaled, not absolute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y_scale,expect_residual_above,expect_relative_below",
    [
        pytest.param(1e6, 1e-12, 1e-12, id="large_state"),
        pytest.param(1e-6, 0.0, 1e-12, id="small_state"),
    ],
)
def test_stage_convergence_is_scaled_not_absolute(
    y_scale, expect_residual_above, expect_relative_below
):
    """C-5.1 prohibits an unscaled absolute test; this shows why.

    The stage-like equation ``z - 0.05*sqrt(|z|) = Y`` is solved at two state
    magnitudes nine orders apart.

    * ``Y = 1e6``: the achieved residual is ``1.57e-08``. A fixed absolute
      threshold of ``1e-12`` is **unreachable** here — at magnitude ``1e6``
      the spacing between representable doubles is already ``~1e-10`` — so
      an unscaled test would reject a stage that is accurate to ``1.6e-14``
      relative. The assertion that the residual exceeds ``1e-12`` is the
      evidence for that claim.
    * ``Y = 1e-6``: the achieved residual is ``1.33e-19``. A fixed absolute
      threshold of ``1e-12`` would have accepted a stage with only ``1e-6``
      relative accuracy, six orders worse than certified.

    In both cases the *relative* accuracy achieved is below ``1e-12``, which
    is the property C-5.1 exists to guarantee and which no single absolute
    threshold delivers at both scales.
    """
    solver = NewtonMixin()

    def residual(z):
        return z - 0.05 * np.sqrt(np.abs(z)) - y_scale

    def jacobian(z):
        return np.atleast_2d(1.0 - 0.025 / np.sqrt(np.abs(z[0])))

    z, _ = solver.newton_solve(
        residual,
        jacobian,
        np.array([y_scale]),
        y_scale=y_scale,
        context="scaled-criterion check",
    )

    achieved = float(np.max(np.abs(residual(z))))
    relative = achieved / float(np.max(np.abs(z)))

    assert achieved <= stage_solve_tolerance(z, y_scale)
    assert relative < expect_relative_below, (
        f"relative stage accuracy {relative:.3e} at y_scale={y_scale:.0e} "
        f"is worse than the C-5.1 guarantee."
    )
    if expect_residual_above > 0.0:
        assert achieved > expect_residual_above, (
            f"achieved residual {achieved:.3e} no longer demonstrates that a "
            f"fixed absolute threshold of {expect_residual_above:.0e} is "
            f"unreachable at this magnitude; the C-5.1 rationale needs "
            f"re-deriving."
        )


def test_stage_solve_tolerance_has_both_terms():
    """Neither term of ``rtol*||z|| + atol`` may be dropped.

    Dropping ``atol`` makes the threshold zero at ``z = 0``, which no
    floating-point residual can meet. Dropping ``rtol*||z||`` reintroduces
    exactly the unscaled absolute test C-5.1 prohibits.
    """
    zero = np.zeros(3)
    assert stage_solve_tolerance(zero, 1.0) > 0.0, "atol term is missing"

    big = np.array([1e8])
    assert stage_solve_tolerance(big, 1.0) > stage_solve_tolerance(
        zero, 1.0
    ), "rtol term is missing: the threshold does not grow with ||z||"
