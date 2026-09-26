"""Certification of structure-aware dispatch (milestone M7).

Two optimizations land here, and **neither is visible to an accuracy test**.

1. A stage equation that is affine in its unknown is solved by one linear solve
   instead of by Newton. Both routes reach the same stage values -- Newton to
   its convergence threshold, the direct solve exactly -- so no assertion on a
   gradient, a Hessian, or an observed order rate can tell which one ran.
2. Dynamics-curvature terms that are identically zero are skipped instead of
   being computed. Skipping the addition of an exact zero changes no bit of the
   result, so again no numerical assertion can see it.

The certified quantities are therefore *counts* and *routes*, exactly as they
are for factorization reuse in ``NUMERICS.md`` C-15, whose generalized lesson
(C-15.1) these tests apply: a gate that nothing reads is not evidenced by being
correct. ``needs_newton`` was computed correctly for the whole life of the
project and consumed by nothing, so every implicit method entered Newton
regardless of what it said.

Two facts are asserted by *tripwire* rather than by counting, because a count
only measures the route it is attached to:

* ``test_the_affine_route_does_not_enter_newton_at_all`` replaces
  ``NewtonMixin.newton_solve`` with a function that raises. If any path still
  reaches Newton, the gradient computation fails rather than quietly producing
  a count that happens to look right.
* ``test_the_skip_does_not_call_the_curvature_callbacks`` replaces the instance
  callbacks with functions that raise. Instance attributes do not affect the
  class-identity check that establishes affineness, so the problem is still
  verified while any actual call fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from adjungo import GLMOptimizer, ProblemStructure
from adjungo.core.affine import AffineDynamics, affine_dynamics_verified
from adjungo.core.problem import Linearity
from adjungo.core.requirements import deduce_requirements
from adjungo.methods.runge_kutta import (
    gauss2,
    implicit_midpoint,
    implicit_trapezoid,
    rk4,
    sdirk2,
    sdirk3,
)
from adjungo.solvers import newton as newton_module
from adjungo.solvers.dirk import DIRKStageSolver
from adjungo.solvers.implicit import ImplicitStageSolver
from adjungo.solvers.linear_stage import (
    NonAffineStageEquation,
    linear_stage_solve,
)
from adjungo.solvers.sdirk import SDIRKStageSolver
from adjungo.validation.reference import reference_gradient
from tests.problems import (
    BilinearStateAffine,
    ConstantJacobianQuadraticControl,
    CoupledNonlinear,
    FullCostObjective,
    make_controls,
)

N_STEPS = 6
T_SPAN = (0.2, 0.95)
Y0 = np.array([0.6, -0.3, 0.2])

M_COEFF = np.array(
    [
        [-0.70, 1.30, 0.00],
        [-0.40, -0.25, 0.90],
        [0.15, -0.60, -1.10],
    ]
)
C_COEFF = np.array([[1.0, 0.0], [0.30, 1.0], [-0.20, 0.45]])
B_COEFF = np.array([0.05, -0.10, 0.02])

IMPLICIT_METHODS = [
    implicit_midpoint,
    implicit_trapezoid,
    sdirk2,
    sdirk3,
    gauss2,
]
ALL_METHODS = [rk4, *IMPLICIT_METHODS]

GUARANTEED_METHODS = [
    "f",
    "F",
    "G",
    "F_yy_action",
    "F_yu_action",
    "F_uu_action",
]


def affine_problem() -> AffineDynamics:
    return AffineDynamics(M_COEFF, C_COEFF, B_COEFF)


def build(method_factory, problem=None, structure=None):
    method = method_factory()
    problem = affine_problem() if problem is None else problem
    opt = GLMOptimizer(
        problem,
        FullCostObjective(nx=3, nu=2),
        method,
        t_span=T_SPAN,
        N=N_STEPS,
        y0=Y0,
        problem_structure=structure,
    )
    u = make_controls(N_STEPS, method.s, 2, seed=5, scale=0.4)
    return opt, u, method


# ---------------------------------------------------------------------------
# The representation
# ---------------------------------------------------------------------------


def test_affine_dynamics_reproduces_its_own_coefficients() -> None:
    p = affine_problem()
    y = np.array([0.3, -0.7, 1.1])
    u = np.array([0.4, -0.2])
    assert np.allclose(p.f(y, u, 0.0), M_COEFF @ y + C_COEFF @ u + B_COEFF)
    assert np.array_equal(p.F(y, u, 0.0), M_COEFF)
    assert np.array_equal(p.G(y, u, 0.0), C_COEFF)


def test_affine_dynamics_jacobians_match_central_differences() -> None:
    """The class computes ``F`` and ``G`` from coefficients rather than by
    differentiating ``f``, so the two must be checked against each other."""
    p = affine_problem()
    y = np.array([0.3, -0.7, 1.1])
    u = np.array([0.4, -0.2])
    eps = 1e-6

    F_fd = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        F_fd[:, j] = (p.f(y + e, u, 0.0) - p.f(y - e, u, 0.0)) / (2 * eps)
    assert np.max(np.abs(F_fd - p.F(y, u, 0.0))) < 1e-9

    G_fd = np.zeros((3, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        G_fd[:, j] = (p.f(y, u + e, 0.0) - p.f(y, u - e, 0.0)) / (2 * eps)
    assert np.max(np.abs(G_fd - p.G(y, u, 0.0))) < 1e-9


def test_affine_dynamics_curvature_is_zero_with_the_right_shapes() -> None:
    p = affine_problem()
    y, u, v = np.ones(3), np.ones(2), np.ones(3)
    assert p.F_yy_action(y, u, 0.0, v).shape == (3, 3)
    assert p.F_yu_action(y, u, 0.0, v).shape == (3, 2)
    assert p.F_uu_action(y, u, 0.0, v).shape == (2, 2)
    assert not p.F_yy_action(y, u, 0.0, v).any()
    assert not p.F_yu_action(y, u, 0.0, v).any()
    assert not p.F_uu_action(y, u, 0.0, v).any()


def test_coefficients_cannot_be_written_through() -> None:
    """A coefficient that could change mid-solve would let the M6 store reuse
    the factorization of a matrix that no longer exists."""
    p = affine_problem()
    with pytest.raises(ValueError):
        p.M[0, 0] = 99.0
    with pytest.raises(ValueError):
        p.C[0, 0] = 99.0
    with pytest.raises(ValueError):
        p.b[0] = 99.0


def test_coefficients_cannot_be_rebound() -> None:
    p = affine_problem()
    with pytest.raises(AttributeError):
        p.M = np.eye(3)


def test_coefficients_are_copied_from_the_caller() -> None:
    M = M_COEFF.copy()
    p = AffineDynamics(M, C_COEFF)
    M[0, 0] = 1234.0
    assert p.M[0, 0] == M_COEFF[0, 0]


@pytest.mark.parametrize(
    "M, C, b",
    [
        (np.ones((3, 2)), C_COEFF, None),
        (M_COEFF, np.ones((2, 2)), None),
        (M_COEFF, C_COEFF, np.ones(5)),
    ],
)
def test_affine_dynamics_rejects_inconsistent_shapes(M, C, b) -> None:
    with pytest.raises(ValueError):
        AffineDynamics(M, C, b)


# ---------------------------------------------------------------------------
# Verification is by construction, not by declaration
# ---------------------------------------------------------------------------


def test_an_unmodified_affine_problem_is_verified() -> None:
    assert affine_dynamics_verified(affine_problem())


def test_a_problem_that_is_not_affine_dynamics_is_not_verified() -> None:
    assert not affine_dynamics_verified(ConstantJacobianQuadraticControl())
    assert not affine_dynamics_verified(object())


@pytest.mark.parametrize("name", GUARANTEED_METHODS)
def test_overriding_any_guaranteeing_method_loses_verification(name) -> None:
    """The coefficient arrays describe the dynamics only while the methods
    that read them are the ones this class defines."""
    overridden = type(
        "Overridden",
        (AffineDynamics,),
        {name: lambda self, *args, **kwargs: None},
    )
    assert not affine_dynamics_verified(overridden(M_COEFF, C_COEFF, B_COEFF))


def test_subclassing_without_overriding_keeps_verification() -> None:
    """Adding behaviour is allowed; replacing the guarantee is not."""
    extended = type(
        "Extended", (AffineDynamics,), {"extra": lambda self: 1}
    )
    assert affine_dynamics_verified(extended(M_COEFF, C_COEFF, B_COEFF))


# ---------------------------------------------------------------------------
# Deduction
# ---------------------------------------------------------------------------


def test_an_affine_problem_deduces_all_three_structural_facts() -> None:
    opt, _u, _m = build(sdirk2)
    st = opt.problem_structure
    assert st.state_affine
    assert st.jointly_affine
    assert st.jacobian_constant


def test_linear_does_not_imply_zero_curvature() -> None:
    """``ConstantJacobianQuadraticControl`` is ``f = A y + B(u*u)``.

    Its ``F`` is the constant ``A``, so it satisfies ``Linearity.LINEAR`` --
    "F independent of y, u" -- while ``F_uu_action`` returns ``2 diag(B^T v)``,
    which is not zero. A dispatch rule reading ``LINEAR`` as permission to skip
    curvature would silently return a Gauss-Newton approximation from
    ``assemble_hessian_vector_product``, whose docstring promises an exact
    Hessian. This test pins the counterexample so that such a rule cannot be
    written without a failure.
    """
    problem = ConstantJacobianQuadraticControl()
    v = np.array([0.3, -0.5, 0.9])
    F_uu = problem.F_uu_action(np.zeros(3), np.ones(2), 0.0, v)
    assert np.max(np.abs(F_uu)) > 0.1

    opt, _u, _m = build(sdirk2, problem=problem)
    assert opt.problem_structure.linearity is Linearity.NONLINEAR
    assert not opt.problem_structure.jointly_affine
    assert not affine_dynamics_verified(problem)


def test_jointly_affine_without_state_affine_is_refused() -> None:
    with pytest.raises(ValueError, match="implies state_affine"):
        ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
            state_affine=False,
            jointly_affine=True,
        )


def test_state_affine_alone_deduces_no_newton() -> None:
    """``needs_newton`` follows from state affineness, independently of the
    legacy ``Linearity`` members."""
    structure = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=True,
        state_affine=True,
    )
    req = deduce_requirements(sdirk2(), structure, 3)
    assert not req.needs_newton


# ---------------------------------------------------------------------------
# The gate is consumed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_the_solver_consumes_needs_newton(factory) -> None:
    """Before M7 this flag was computed and read by nothing, so every implicit
    method entered Newton regardless of its value (C-15.1)."""
    opt, _u, _m = build(factory)
    assert opt.stage_solver.needs_newton == opt.requirements.needs_newton
    assert not opt.stage_solver.needs_newton


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_the_affine_route_enters_newton_zero_times(factory) -> None:
    opt, u, _m = build(factory)
    opt.stage_solver.reset_newton_count()
    opt.gradient(u)
    assert opt.stage_solver.newton_entries == 0


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_a_nonlinear_problem_still_enters_newton(factory) -> None:
    """The counter is not vacuously zero."""
    opt, u, _m = build(factory, problem=CoupledNonlinear())
    opt.stage_solver.reset_newton_count()
    opt.gradient(u)
    assert opt.stage_solver.newton_entries > 0


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_the_affine_route_does_not_enter_newton_at_all(
    factory, monkeypatch
) -> None:
    """Completeness, by tripwire rather than by count.

    A count proves only that the counted route was not taken. Replacing
    ``newton_solve`` with a raising function proves that *no* route reached
    Newton, including one that bypassed the counter.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("Newton was entered on the affine route")

    opt, u, _m = build(factory)
    monkeypatch.setattr(newton_module.NewtonMixin, "newton_solve", forbidden)
    grad = opt.gradient(u)
    assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# The answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_affine_route_gradient_matches_the_independent_reference(
    factory,
) -> None:
    """C-14.1. The reference assembles and differentiates the monolithic
    discrete system and shares no code with the stepping path."""
    opt, u, method = build(factory)
    grad = opt.gradient(u)
    ref = reference_gradient(
        Y0,
        u,
        T_SPAN,
        N_STEPS,
        opt.problem,
        method,
        opt.objective,
    )
    scale = max(float(np.max(np.abs(ref))), 1.0)
    assert np.max(np.abs(grad - ref)) / scale < 1e-10


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_affine_and_newton_routes_agree(factory) -> None:
    """Not bit-identity, deliberately.

    Factorization reuse (M6) could demand bit-identity because it factored the
    same matrix; nothing was approximated. A direct linear solve and a Newton
    iteration are *different floating-point computations* that reach the same
    stage values by different routes, so the correct acceptance criterion is
    agreement at a tolerance derived from the Newton convergence threshold,
    not a pinned float.
    """
    forced_newton = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=True,
    )
    opt_affine, u, _m = build(factory)
    opt_newton, _u2, _m2 = build(factory, structure=forced_newton)

    g_affine = opt_affine.gradient(u)
    g_newton = opt_newton.gradient(u)
    scale = max(float(np.max(np.abs(g_newton))), 1.0)
    assert np.max(np.abs(g_affine - g_newton)) / scale < 1e-10

    v = make_controls(N_STEPS, _m.s, 2, seed=11, scale=0.3)
    hv_affine = opt_affine.hessian_vector_product(u, v)
    hv_newton = opt_newton.hessian_vector_product(u, v)
    scale = max(float(np.max(np.abs(hv_newton))), 1.0)
    assert np.max(np.abs(hv_affine - hv_newton)) / scale < 1e-10


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_the_affine_route_still_takes_one_factorization(factory) -> None:
    """M7 composes with M6 rather than displacing it: the direct solve goes
    through the same store, so a constant Jacobian is still factored once for
    the entire solve (C-15)."""
    opt, u, _m = build(factory)
    opt.stage_solver.factorizations.reset_counts()
    opt.gradient(u)
    assert opt.stage_solver.factorizations.factorizations == 1


# ---------------------------------------------------------------------------
# The curvature skip
# ---------------------------------------------------------------------------


def _no_curvature_structure() -> ProblemStructure:
    """Same problem, with the zero-curvature fact withheld."""
    return ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
        state_affine=True,
        jointly_affine=False,
    )


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_skipping_zero_curvature_is_bit_identical(factory) -> None:
    """The skipped terms are additions of exact zero, so this is the removal
    of an operation rather than an approximation of one. The claim is made
    about two computations performed here, not about a stored constant, so it
    holds on any platform and any BLAS."""
    opt_skip, u, method = build(factory)
    opt_full, _u, _m = build(factory, structure=_no_curvature_structure())
    v = make_controls(N_STEPS, method.s, 2, seed=13, scale=0.3)

    hv_skip = opt_skip.hessian_vector_product(u, v)
    hv_full = opt_full.hessian_vector_product(u, v)
    assert np.array_equal(hv_skip, hv_full)


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_the_skip_does_not_call_the_curvature_callbacks(factory) -> None:
    """Tripwire on the instance.

    Affineness is verified from the *class*, so replacing the callbacks on the
    instance leaves the problem verified while making any actual call fail.
    This closes the ergonomic gap reported at the end of M6: a problem with no
    dynamics curvature is differentiated twice without supplying three
    zero-returning callbacks.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("a zero curvature term was computed")

    opt, u, method = build(factory)
    problem = opt.problem
    for name in ("F_yy_action", "F_yu_action", "F_uu_action"):
        object.__setattr__(problem, name, forbidden)
    assert affine_dynamics_verified(problem)

    v = make_controls(N_STEPS, method.s, 2, seed=17, scale=0.3)
    hv = opt.hessian_vector_product(u, v)
    assert np.all(np.isfinite(hv))


def test_objective_curvature_is_not_skipped() -> None:
    """Only the *dynamics* curvature vanishes. An affine plant under a
    quadratic cost has a perfectly nonzero Hessian, and a skip that reached
    the objective would return zero."""
    opt, u, method = build(sdirk2)
    v = make_controls(N_STEPS, method.s, 2, seed=19, scale=0.3)
    hv = opt.hessian_vector_product(u, v)
    assert np.max(np.abs(hv)) > 1e-3


def test_a_declaration_alone_does_not_open_the_skip() -> None:
    """``jointly_affine`` is a public constructor argument and no cheap exact
    check of it exists, so it is believed only when the problem is affine by
    construction. A problem that merely declares it must still supply the
    callbacks, and still has its curvature computed."""

    class DeclaredOnly:
        state_dim = 3
        control_dim = 2
        A = M_COEFF

        def f(self, y, u, t):
            return self.A @ y + C_COEFF @ (u * u)

        def F(self, y, u, t):
            return self.A

        def G(self, y, u, t):
            return 2.0 * C_COEFF * u[np.newaxis, :]

        def F_yy_action(self, y, u, t, v):
            return np.zeros((3, 3))

        def F_yu_action(self, y, u, t, v):
            return np.zeros((3, 2))

        def F_uu_action(self, y, u, t, v):
            return 2.0 * np.diag(C_COEFF.T @ v)

    lying = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
        state_affine=True,
        jointly_affine=True,
    )
    opt, u, method = build(sdirk2, problem=DeclaredOnly(), structure=lying)
    v = make_controls(N_STEPS, method.s, 2, seed=23, scale=0.3)
    hv = opt.hessian_vector_product(u, v)

    ref = reference_gradient(
        Y0, u, T_SPAN, N_STEPS, opt.problem, method, opt.objective
    )
    assert np.all(np.isfinite(hv))
    assert np.all(np.isfinite(ref))

    # The F_uu term was computed, not skipped. Two checks, because either
    # alone could pass vacuously: the lying declaration gives the same answer
    # as honestly withholding the claim, and that answer differs from the one
    # a believed declaration would have produced.
    opt_no_curv, _u, _m = build(
        sdirk2, problem=DeclaredOnly(), structure=_no_curvature_structure()
    )
    assert np.array_equal(hv, opt_no_curv.hessian_vector_product(u, v))

    class NoCurvature(DeclaredOnly):
        def F_uu_action(self, y, u, t, v):
            return np.zeros((2, 2))

    opt_zeroed, _u2, _m2 = build(
        sdirk2, problem=NoCurvature(), structure=_no_curvature_structure()
    )
    hv_without = opt_zeroed.hessian_vector_product(u, v)
    assert np.max(np.abs(hv - hv_without)) > 1e-6


# ---------------------------------------------------------------------------
# The direct solve refuses what it cannot solve
# ---------------------------------------------------------------------------


def test_linear_stage_solve_is_exact_for_an_affine_residual() -> None:
    K = np.array([[2.0, 0.3], [-0.4, 1.5]])
    c = np.array([1.0, -2.0])
    z, lu = linear_stage_solve(
        lambda z: K @ z - c, lambda z: K, np.array([7.0, 7.0])
    )
    assert np.max(np.abs(K @ z - c)) < 1e-12
    assert np.allclose(z, np.linalg.solve(K, c))
    assert lu is not None


def test_linear_stage_solve_result_is_independent_of_the_start() -> None:
    """One Newton step on an affine residual is exact from anywhere, which is
    what licenses factoring at the starting point rather than at the answer."""
    K = np.array([[2.0, 0.3], [-0.4, 1.5]])
    c = np.array([1.0, -2.0])

    def residual(z):
        return K @ z - c

    z_a, _ = linear_stage_solve(residual, lambda z: K, np.zeros(2))
    z_b, _ = linear_stage_solve(residual, lambda z: K, np.array([50.0, -30.0]))
    assert np.max(np.abs(z_a - z_b)) < 1e-12


def test_linear_stage_solve_refuses_a_nonlinear_residual() -> None:
    """The check is on the residual at the returned value, so it establishes
    what matters -- that the stage equation is satisfied -- rather than
    sampling the Jacobian somewhere and generalizing (precedent R-9)."""

    def residual(z):
        return z * z - np.array([4.0, 9.0])

    def jac(z):
        return np.diag(2.0 * z)

    with pytest.raises(NonAffineStageEquation, match="not affine in the state"):
        linear_stage_solve(residual, jac, np.array([1.0, 1.0]))


def test_nonaffine_error_reports_the_residual_and_threshold() -> None:
    err = NonAffineStageEquation("SDIRK stage 2", 3.5e-3, 1.2e-12)
    assert "3.500e-03" in str(err)
    assert "1.200e-12" in str(err)
    assert err.residual == 3.5e-3


def test_linear_stage_solve_publishes_the_matrix_it_solved_with() -> None:
    """C-5.4: the adjoint applies the transpose of the matrix the forward
    stage equation used. For an affine residual the Jacobian is constant, so
    the factorization taken at the starting point *is* the one at the answer,
    and this asserts that identity rather than assuming it."""
    K = np.array([[2.0, 0.3], [-0.4, 1.5]])
    seen: list[np.ndarray] = []

    def factor(matrix):
        seen.append(matrix.copy())
        import scipy.linalg

        return scipy.linalg.lu_factor(matrix)

    _z, _lu = linear_stage_solve(
        lambda z: K @ z - np.array([1.0, -2.0]),
        lambda z: K,
        np.array([5.0, 5.0]),
        factor=factor,
    )
    assert len(seen) == 1
    assert np.array_equal(seen[0], K)


# ---------------------------------------------------------------------------
# Gaps found by defect injection, and the tests that close them
# ---------------------------------------------------------------------------


def _central_difference_hvp(opt, u, v, eps: float = 1e-6) -> np.ndarray:
    """``[H v]`` by differencing the gradient, which shares no code with the
    second-order adjoint path."""
    return (opt.gradient(u + eps * v) - opt.gradient(u - eps * v)) / (2 * eps)


@pytest.mark.parametrize("factory", [sdirk2, gauss2, rk4])
def test_a_declaration_does_not_open_the_second_order_adjoint_skip(
    factory,
) -> None:
    """Closes injection gap M7-6.

    ``adjoint_sensitivity`` and ``assemble_hessian_vector_product`` each have
    their own skip, and they consume *different* curvature blocks: the first
    reads ``F_yy`` and ``F_yu``, the second reads ``F_yu`` and ``F_uu``.

    Every other affine fixture here has zero ``F_yu``, so wrongly skipping in
    ``adjoint_sensitivity`` alone changed no number and no test failed.
    ``BilinearStateAffine`` has nonzero ``F_yu`` and zero ``F_uu``, which is
    the pattern that reaches the first skip and not the second.
    """
    problem = BilinearStateAffine()
    lying = ProblemStructure(
        linearity=Linearity.BILINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=True,
        state_affine=True,
        jointly_affine=True,
    )
    opt, u, method = build(factory, problem=problem, structure=lying)
    v = make_controls(N_STEPS, method.s, 2, seed=29, scale=0.25)

    hv = opt.hessian_vector_product(u, v)
    hv_fd = _central_difference_hvp(opt, u, v)
    scale = max(float(np.max(np.abs(hv_fd))), 1.0)
    assert np.max(np.abs(hv - hv_fd)) / scale < 1e-6

    # Not vacuous: the F_yu term this exercises is materially nonzero.
    F_yu = problem.F_yu_action(
        np.array([0.4, -0.7, 1.1]),
        np.array([0.35, -0.2]),
        0.0,
        np.array([0.3, -0.5, 0.9]),
    )
    assert np.max(np.abs(F_yu)) > 0.1


def test_declaring_linear_does_not_deduce_zero_curvature() -> None:
    """Closes injection gap M7-8.

    ``test_linear_does_not_imply_zero_curvature`` above pinned the
    counterexample but could not reach the defect, because
    ``ConstantJacobianQuadraticControl`` declares no ``linearity`` attribute
    and is therefore deduced ``NONLINEAR``. The dangerous rule lives on the
    ``LINEAR`` branch, so the counterexample has to travel it.
    """

    class DeclaredLinear(ConstantJacobianQuadraticControl):
        linearity = Linearity.LINEAR

    problem = DeclaredLinear()
    opt, u, method = build(sdirk2, problem=problem)
    assert opt.problem_structure.linearity is Linearity.LINEAR
    # What the LINEAR branch legitimately decides: stage equations linear in
    # their unknown. It no longer decides constancy (C-15.1), so the branch is
    # witnessed by the stage route rather than by jacobian_constant.
    assert not opt.requirements.needs_newton
    assert not opt.problem_structure.jointly_affine

    v = make_controls(N_STEPS, method.s, 2, seed=31, scale=0.25)
    hv = opt.hessian_vector_product(u, v)
    hv_fd = _central_difference_hvp(opt, u, v)
    scale = max(float(np.max(np.abs(hv_fd))), 1.0)
    assert np.max(np.abs(hv - hv_fd)) / scale < 1e-6


@pytest.mark.parametrize(
    "solver_class",
    [DIRKStageSolver, SDIRKStageSolver, ImplicitStageSolver],
)
def test_a_solver_built_without_an_opinion_iterates(solver_class) -> None:
    """Closes injection gap M7-12.

    The affine route is only sound for a stage equation that is affine in its
    unknown. A solver constructed with no structural information must
    therefore iterate: the error direction is toward doing more work, never
    toward asserting a structure the problem does not have.

    Only the factory sets this flag in the integrated path, so no end-to-end
    test reaches the constructor default and it must be asserted directly.
    """
    assert solver_class().needs_newton is True


def test_the_dispatch_default_is_also_conservative() -> None:
    """The class attribute backing the flag, asserted separately from the
    constructors that shadow it."""
    assert newton_module.StageDispatchMixin.needs_newton is True
