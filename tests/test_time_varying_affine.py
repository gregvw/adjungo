"""Certification of time-varying affine dynamics ``y' = M(t)y + C(t)u + b(t)``.

``NUMERICS.md`` C-16.2 states ``jointly_affine`` with coefficients that may vary
in time. :class:`~adjungo.core.affine.AffineDynamics` implements the constant
case; this module certifies :class:`~adjungo.core.affine.TimeVaryingAffineDynamics`,
which implements the general one.

The increment separates two facts that the constant case holds simultaneously
and that C-16.1 forbids merging:

* **Zero curvature** still holds, so the C-16.6 skip is still opened — but
  C-17.1 splits the reason in two, and the split matters. The *form*
  ``f = M(·)y + C(·)u + b(·)`` is owned by the class and constructed. The
  *purity* of the three caller-supplied callables is an obligation under C-1:
  the coefficients are invoked with ``t`` and nothing else, which confines the
  interface but not a Python closure. ``test_a_coefficient_closing_over_the_
  control_defeats_the_skip`` holds that boundary against two closed forms so
  the C-17.1 numbers cannot rot.
* **A constant Jacobian does not.** ``F = M(t_i)`` differs between stages at
  distinct abscissae, so C-15 reuse must be off. ``coefficients_constant``
  carries this, and it is part of the verified guarantee rather than a caller
  declaration.

What is certified here, and by what
-----------------------------------

Per C-16.3 the route is invisible to accuracy assertions, so route and count
are certified separately from the numbers:

* *Route* — Newton is never entered, asserted by count and again by tripwire.
* *Count* — one factorization per implicit stage solve per step, with **no**
  reuse. The contrast against the constant case, which takes one factorization
  for the entire solve, is asserted in the same test so that neither number can
  be vacuous.
* *Numbers* — the independent monolithic reference (C-14.1 level 1) and a
  closed-form two-step anchor (C-14.2), which no reference implementation
  participates in.
* *Times* — the coefficients are evaluated at the stage abscissae
  ``t_n + c_i h``. This is the claim with no analogue in the constant case: a
  defect that read ``M`` at the step time instead of the stage time is
  invisible to every constant-coefficient fixture in the suite, which is the
  "population that cannot reach the defect" failure C-16.8 records.
"""

from __future__ import annotations

import numpy as np
import pytest

from adjungo import GLMOptimizer, ProblemStructure
from adjungo.core.affine import (
    AffineDynamics,
    TimeVaryingAffineDynamics,
    affine_dynamics_verified,
)
from adjungo.core.method import StageType
from adjungo.core.problem import Linearity
from adjungo.methods.runge_kutta import (
    explicit_euler,
    gauss2,
    implicit_midpoint,
    implicit_trapezoid,
    rk4,
    sdirk2,
    sdirk3,
)
from adjungo.solvers import newton as newton_module
from adjungo.solvers.dirk import DIRKStageSolver
from adjungo.solvers.factorization import DeclaredStructureViolation
from adjungo.solvers.implicit import ImplicitStageSolver
from adjungo.solvers.sdirk import SDIRKStageSolver
from adjungo.validation.reference import reference_gradient, reference_hessian
from tests.problems import (
    AnchorObjective,
    FullCostObjective,
    make_controls,
)

N_STEPS = 6
T_SPAN = (0.2, 0.95)
Y0 = np.array([0.6, -0.3, 0.2])
NX = 3
NU = 2

M_0 = np.array(
    [
        [-0.70, 1.30, 0.00],
        [-0.40, -0.25, 0.90],
        [0.15, -0.60, -1.10],
    ]
)
#: The time-varying part of ``M``. Deliberately not a multiple of ``M_0`` and
#: not symmetric: a perturbation along ``M_0`` would commute with it and could
#: leave a stage-time defect undetectable in the propagated state.
M_1 = np.array(
    [
        [0.40, -0.20, 0.55],
        [0.10, 0.65, -0.30],
        [-0.35, 0.20, 0.45],
    ]
)
C_0 = np.array([[1.0, 0.0], [0.30, 1.0], [-0.20, 0.45]])
C_1 = np.array([[0.20, -0.50], [0.00, 0.35], [0.60, -0.15]])
B_0 = np.array([0.05, -0.10, 0.02])


def M_of_t(t: float) -> np.ndarray:
    return M_0 + np.sin(1.7 * t) * M_1


def C_of_t(t: float) -> np.ndarray:
    return C_0 + t * C_1


def b_of_t(t: float) -> np.ndarray:
    return B_0 * np.cos(0.9 * t)


IMPLICIT_METHODS = [
    implicit_midpoint,
    implicit_trapezoid,
    sdirk2,
    sdirk3,
    gauss2,
]
ALL_METHODS = [rk4, *IMPLICIT_METHODS]

# Both sides of every reference comparison below solve stage equations that
# are affine in their unknowns, the package by one direct linear solve and the
# monolithic reference by a Newton iteration that is exact in one step on an
# affine residual. Neither side stops at a convergence threshold, so the floor
# is the backward error of the dense LU solves, not STAGE_NEWTON_TOL, and the
# implicit population needs no CONDITIONING_ALLOWANCE.
#
# Measured over ALL_METHODS at N = 6, n = 3, nu = 2 on the fixture in this
# module, relative max-norm against the reference: worst gradient error
# 1.11e-16, worst Hessian-vector error 5.55e-17. AFFINE_RTOL is set five orders
# above that observation, not at it, because the margin must absorb
# conditioning of a stage matrix rather than certify these coefficients.
AFFINE_RTOL = 1.0e-11


def time_varying_problem() -> TimeVaryingAffineDynamics:
    return TimeVaryingAffineDynamics(
        M_of_t, C_of_t, b_of_t, state_dim=NX, control_dim=NU
    )


def constant_problem() -> AffineDynamics:
    return AffineDynamics(M_0, C_0, B_0)


def build(method_factory, problem=None, structure=None):
    method = method_factory()
    problem = time_varying_problem() if problem is None else problem
    opt = GLMOptimizer(
        problem,
        FullCostObjective(nx=NX, nu=NU),
        method,
        t_span=T_SPAN,
        N=N_STEPS,
        y0=Y0,
        problem_structure=structure,
    )
    u = make_controls(N_STEPS, method.s, NU, seed=5, scale=0.4)
    return opt, u, method


def relative_error(got: np.ndarray, ref: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(ref))), 1.0)
    return float(np.max(np.abs(got - ref))) / scale


# ---------------------------------------------------------------------------
# The representation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0.2, 0.575, 0.95])
def test_the_coefficients_are_evaluated_at_the_time_asked_for(t) -> None:
    p = time_varying_problem()
    y = np.array([0.3, -0.7, 1.1])
    u = np.array([0.4, -0.2])
    assert np.allclose(
        p.f(y, u, t), M_of_t(t) @ y + C_of_t(t) @ u + b_of_t(t)
    )
    assert np.allclose(p.F(y, u, t), M_of_t(t))
    assert np.allclose(p.G(y, u, t), C_of_t(t))


def test_the_coefficients_actually_vary() -> None:
    """Guard against a fixture that would make every claim below vacuous."""
    assert not np.allclose(M_of_t(0.2), M_of_t(0.95))
    assert not np.allclose(C_of_t(0.2), C_of_t(0.95))
    assert not np.allclose(b_of_t(0.2), b_of_t(0.95))


def test_a_missing_forcing_is_zero_not_absent() -> None:
    p = TimeVaryingAffineDynamics(
        M_of_t, C_of_t, state_dim=NX, control_dim=NU
    )
    assert np.array_equal(p.b(0.4), np.zeros(NX))
    y = np.array([0.3, -0.7, 1.1])
    u = np.array([0.4, -0.2])
    assert np.allclose(p.f(y, u, 0.4), M_of_t(0.4) @ y + C_of_t(0.4) @ u)


def test_dynamics_curvature_is_structurally_zero() -> None:
    p = time_varying_problem()
    y = np.array([0.3, -0.7, 1.1])
    u = np.array([0.4, -0.2])
    v = np.array([1.0, -2.0, 0.5])
    for t in (0.2, 0.6, 0.95):
        assert np.array_equal(p.F_yy_action(y, u, t, v), np.zeros((NX, NX)))
        assert np.array_equal(p.F_yu_action(y, u, t, v), np.zeros((NX, NU)))
        assert np.array_equal(p.F_uu_action(y, u, t, v), np.zeros((NU, NU)))


def test_a_coefficient_value_is_a_copy_the_caller_cannot_reach() -> None:
    """A callable is free to return the same scratch buffer every call and
    then write into it. Without the copy, a factorization already taken from
    that array would describe a matrix that no longer exists -- the aliasing
    defect C-15.6 records for the factorization store, arriving through the
    problem instead."""
    scratch = np.zeros((NX, NX))

    def M_scratch(t: float) -> np.ndarray:
        scratch[:] = M_0
        return scratch

    p = TimeVaryingAffineDynamics(
        M_scratch, C_of_t, state_dim=NX, control_dim=NU
    )
    first = p.F(Y0, np.zeros(NU), 0.3)
    scratch[0, 0] = 1.0e6
    assert first[0, 0] == M_0[0, 0]


def test_a_coefficient_value_is_not_writeable() -> None:
    p = time_varying_problem()
    value = p.F(Y0, np.zeros(NU), 0.3)
    with pytest.raises(ValueError):
        value[0, 0] = 1.0


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("M", {"M": lambda t: np.zeros((2, 2))}),
        ("C", {"C": lambda t: np.zeros((NX, 5))}),
        ("b", {"b": lambda t: np.zeros(7)}),
    ],
)
def test_a_coefficient_of_the_wrong_shape_is_refused_where_it_is_used(
    name, kwargs
) -> None:
    """Checked at every call rather than once at construction. Validating one
    sample and asserting the shape of every other value would be a probe."""
    spec = {"M": M_of_t, "C": C_of_t, "b": b_of_t} | kwargs
    p = TimeVaryingAffineDynamics(
        spec["M"], spec["C"], spec["b"], state_dim=NX, control_dim=NU
    )
    with pytest.raises(ValueError, match=name):
        p.f(Y0, np.zeros(NU), 0.5)


def test_non_callable_coefficients_are_refused() -> None:
    with pytest.raises(TypeError, match="callable of t alone"):
        TimeVaryingAffineDynamics(
            M_0, C_of_t, state_dim=NX, control_dim=NU
        )


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TimeVaryingAffineDynamics(
            M_of_t, C_of_t, state_dim=0, control_dim=NU
        )


# ---------------------------------------------------------------------------
# Verification by construction
# ---------------------------------------------------------------------------


def test_the_time_varying_representation_is_verified() -> None:
    assert affine_dynamics_verified(time_varying_problem())


def test_the_constant_representation_is_still_verified() -> None:
    """The generalisation of the check must not have cost the original case."""
    assert affine_dynamics_verified(constant_problem())


@pytest.mark.parametrize(
    "member",
    [
        "f",
        "F",
        "G",
        "F_yy_action",
        "F_yu_action",
        "F_uu_action",
        "M",
        "C",
        "b",
        "_coefficient",
        "coefficients_constant",
    ],
)
def test_overriding_any_guaranteed_member_defeats_verification(
    member,
) -> None:
    """The coefficient accessors are guaranteed alongside the computational
    methods because they are the only places the caller's callables are
    invoked. Confining those calls to ``t`` is the whole of the zero-curvature
    argument, so a subclass that intercepts one could pass ``y`` and the
    guarantee would be gone."""
    overridden = type(
        "Overridden",
        (TimeVaryingAffineDynamics,),
        {member: lambda self, *args, **kwargs: None},
    )
    problem = overridden(
        M_of_t, C_of_t, b_of_t, state_dim=NX, control_dim=NU
    )
    assert not affine_dynamics_verified(problem)


def test_an_unrelated_addition_does_not_defeat_verification() -> None:
    """The check is on the guaranteed members, not on the class being
    exactly this one; otherwise it would refuse harmless extension."""
    extended = type(
        "Extended",
        (TimeVaryingAffineDynamics,),
        {"describe": lambda self: "extended"},
    )
    assert affine_dynamics_verified(
        extended(M_of_t, C_of_t, b_of_t, state_dim=NX, control_dim=NU)
    )


def test_a_class_deriving_from_two_roots_is_refused() -> None:
    """Such a class mixes two initialisers and two notions of
    ``coefficients_constant``, and the MRO would silently pick one. Refusing
    costs the caller the general route and nothing else."""
    both = type("Both", (AffineDynamics, TimeVaryingAffineDynamics), {})
    assert not affine_dynamics_verified(both(M_0, C_0, B_0))


def test_the_constancy_claims_differ_between_the_two_representations() -> None:
    assert constant_problem().coefficients_constant is True
    assert time_varying_problem().coefficients_constant is False


# ---------------------------------------------------------------------------
# What the structure deduction concludes
# ---------------------------------------------------------------------------


def test_time_varying_coefficients_do_not_claim_a_constant_jacobian() -> None:
    """Zero curvature and a constant Jacobian are separate axes (C-16.1). The
    deduction previously asserted the second unconditionally for anything
    verified affine, which was correct only because the constant case was the
    only one that existed."""
    opt, _u, _m = build(sdirk2)
    structure = opt.problem_structure
    assert structure.jointly_affine
    assert structure.state_affine
    assert not structure.jacobian_constant
    assert not structure.jacobian_control_dependent
    assert structure.linearity is Linearity.LINEAR


def test_constant_coefficients_still_claim_a_constant_jacobian() -> None:
    opt, _u, _m = build(sdirk2, problem=constant_problem())
    assert opt.problem_structure.jacobian_constant


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_no_reuse_within_a_solve_is_deduced_for_time_varying_coefficients(
    factory,
) -> None:
    """Distinct stage times give distinct matrices (C-17.2). Reuse *across
    calls* is deduced, and is asserted separately under C-17.6."""
    opt, _u, _m = build(factory)
    requirements = opt.requirements
    assert not requirements.needs_newton
    assert not requirements.can_reuse_across_stages
    assert not requirements.can_reuse_across_steps
    assert requirements.factorizations_for_solve(N_STEPS) is None


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_the_affine_route_enters_newton_zero_times(factory) -> None:
    """A time-varying Jacobian removes reuse; it does not remove affineness.
    The stage equation is still linear in its unknown at each fixed stage
    time, so the direct solve is still exact."""
    opt, u, _m = build(factory)
    opt.stage_solver.reset_newton_count()
    opt.gradient(u)
    assert opt.stage_solver.newton_entries == 0


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_the_affine_route_does_not_enter_newton_at_all(
    factory, monkeypatch
) -> None:
    """Tripwire, per C-16.5: a count proves only that the counted route was
    not taken."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Newton was entered on the affine route")

    opt, u, _m = build(factory)
    monkeypatch.setattr(newton_module.NewtonMixin, "newton_solve", forbidden)
    grad = opt.gradient(u)
    assert np.all(np.isfinite(grad))


def expected_factorizations(method, steps: int) -> int:
    """One factorization per implicit stage *solve*, per step, with no reuse.

    Derived from the tableau rather than tabulated, so that the expectation
    states the rule. The sequential families solve one ``n x n`` system per
    stage with a nonzero diagonal entry; a fully implicit tableau solves one
    coupled ``(s*n)`` system per step however many stages it has.

    There is no Newton multiplier. On the affine route each stage solve
    factors exactly once, which is why this count is predictable at all --
    C-15.3 declines to predict a count without reuse precisely because Newton
    iteration counts are a property of Newton, and here there is no Newton.
    """
    if method.stage_type is StageType.EXPLICIT:
        return 0
    if method.stage_type is StageType.IMPLICIT:
        return steps
    implicit_stages = sum(
        1 for i in range(method.s) if method.A[i, i] != 0.0
    )
    return steps * implicit_stages


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_each_implicit_stage_solve_takes_exactly_one_factorization(
    factory,
) -> None:
    """The certified count for this route, and the contrast that keeps it from
    being vacuous.

    C-15.4 reads the constant case's single factorization as the optimisation
    working. Here the same store must *not* reuse, because the matrices
    genuinely differ, and the two numbers are asserted together so that a
    regression in either direction is visible: a store that wrongly reused
    would drop this count to one, and one that stopped reusing would raise the
    constant case's count to this one.
    """
    opt, u, method = build(factory)
    opt.stage_solver.factorizations.reset_counts()
    opt.gradient(u)
    observed = opt.stage_solver.factorizations.factorizations
    assert observed == expected_factorizations(method, N_STEPS)
    assert observed > 1

    opt_const, u_const, _m = build(factory, problem=constant_problem())
    opt_const.stage_solver.factorizations.reset_counts()
    opt_const.gradient(u_const)
    assert opt_const.stage_solver.factorizations.factorizations == 1


# ---------------------------------------------------------------------------
# When the coefficients are evaluated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_coefficients_are_evaluated_at_the_stage_abscissae(factory) -> None:
    """The claim with no analogue in the constant case.

    ``M`` must be read at ``t_n + c_i h``, not at ``t_n``. Every
    constant-coefficient fixture is blind to the difference, and a wrong stage
    time produces a derivative that is wrong by an amount shrinking with ``h``
    -- which C-3 forbids reading as discretization error, and which precedents
    R-1 and R-2 record being misread that way twice.

    Times are compared at 1e-12 absolute rather than exactly. They are O(1)
    and formed by a few flops, so a few ULPs of O(1) is the right budget;
    C-11.3 forbids pinning the bit pattern of an arithmetic result. The
    tolerance is far tighter than the smallest spacing being resolved, which
    is ``h * min_i>j |c_i - c_j|``, of order 1e-2 here.
    """
    seen: list[float] = []

    def M_recording(t: float) -> np.ndarray:
        seen.append(float(t))
        return M_of_t(t)

    problem = TimeVaryingAffineDynamics(
        M_recording, C_of_t, b_of_t, state_dim=NX, control_dim=NU
    )
    opt, u, method = build(factory, problem=problem)
    opt.gradient(u)

    h = (T_SPAN[1] - T_SPAN[0]) / N_STEPS
    expected = sorted(
        {
            T_SPAN[0] + n * h + float(c_i) * h
            for n in range(N_STEPS)
            for c_i in method.c
        }
    )
    observed = sorted(set(seen))

    assert len(observed) == len(expected)
    assert all(
        abs(got - want) < 1e-12
        for got, want in zip(observed, expected, strict=True)
    )


def test_the_abscissa_check_can_fail() -> None:
    """The assertion above compares against a set the method supplies. If the
    stage times were read at ``t_n``, the observed set would be the step times
    and would not match -- shown here rather than assumed, because an
    assertion that cannot fail has never tested anything."""
    method = sdirk2()
    h = (T_SPAN[1] - T_SPAN[0]) / N_STEPS
    stage_times = sorted(
        {
            T_SPAN[0] + n * h + float(c_i) * h
            for n in range(N_STEPS)
            for c_i in method.c
        }
    )
    step_times = sorted({T_SPAN[0] + n * h for n in range(N_STEPS)})
    assert len(stage_times) != len(step_times) or any(
        abs(a - b) > 1e-12
        for a, b in zip(stage_times, step_times, strict=True)
    )


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_gradient_matches_the_independent_reference(factory) -> None:
    """C-14.1 level 1. The reference assembles the whole discrete system as one
    residual and differentiates it densely, sharing no code with
    ``adjungo/stepping/`` or ``adjungo/solvers/``."""
    opt, u, method = build(factory)
    grad = opt.gradient(u)
    ref = reference_gradient(
        Y0, u, T_SPAN, N_STEPS, opt.problem, method, opt.objective
    )
    assert relative_error(grad, ref) < AFFINE_RTOL


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_hessian_vector_product_matches_the_independent_reference(
    factory,
) -> None:
    opt, u, method = build(factory)
    v = make_controls(N_STEPS, method.s, NU, seed=11, scale=0.3)
    hv = opt.hessian_vector_product(u, v)
    H = reference_hessian(
        Y0, u, T_SPAN, N_STEPS, opt.problem, method, opt.objective
    )
    ref = (H @ v.reshape(-1)).reshape(v.shape)
    assert relative_error(hv, ref) < AFFINE_RTOL


def test_gradient_matches_a_closed_form_anchor() -> None:
    """C-14.2, the oracle level no implementation participates in.

    Two explicit Euler steps on the scalar ``y' = m(t) y + c(t) u`` with
    ``J = y_2^2 / 2``. Writing the stage times ``t_0`` and ``t_1 = t_0 + h``,

        y_1 = (1 + h m_0) y_0 + h c_0 u_0
        y_2 = (1 + h m_1) y_1 + h c_1 u_1

    so ``dJ/du_1 = y_2 h c_1`` and ``dJ/du_0 = y_2 (1 + h m_1) h c_0``.

    Two steps rather than one, and distinct coefficient values at the two step
    times, so that the anchor distinguishes ``m_0`` from ``m_1``. A single step
    would be satisfied by any code that evaluated the coefficients at one
    consistent time, which is the defect this whole module exists to exclude.
    """
    t0, t1 = 0.3, 1.1
    steps = 2
    h = (t1 - t0) / steps
    y0_value = 0.8

    def m_scalar(t: float) -> np.ndarray:
        return np.array([[-0.6 + 0.9 * t]])

    def c_scalar(t: float) -> np.ndarray:
        return np.array([[0.4 + 1.3 * t]])

    method = explicit_euler()
    problem = TimeVaryingAffineDynamics(
        m_scalar, c_scalar, state_dim=1, control_dim=1
    )
    opt = GLMOptimizer(
        problem,
        AnchorObjective(),
        method,
        t_span=(t0, t1),
        N=steps,
        y0=np.array([y0_value]),
    )
    u = np.array([[[0.35]], [[-0.55]]])

    tau = [t0, t0 + h]
    m = [float(m_scalar(t)[0, 0]) for t in tau]
    c = [float(c_scalar(t)[0, 0]) for t in tau]

    y1 = (1.0 + h * m[0]) * y0_value + h * c[0] * float(u[0, 0, 0])
    y2 = (1.0 + h * m[1]) * y1 + h * c[1] * float(u[1, 0, 0])

    expected = np.array(
        [
            [[y2 * (1.0 + h * m[1]) * h * c[0]]],
            [[y2 * h * c[1]]],
        ]
    )

    assert abs(opt.objective_value(u) - 0.5 * y2**2) < 1e-14
    assert relative_error(opt.gradient(u), expected) < 1e-13


def test_the_closed_form_anchor_distinguishes_the_two_stage_times() -> None:
    """The anchor above is only an oracle for the stage index if swapping the
    two coefficient times changes its value. Asserted, not assumed."""
    t0, t1 = 0.3, 1.1
    h = (t1 - t0) / 2
    y0_value = 0.8
    u0, u1 = 0.35, -0.55

    def m_scalar(t: float) -> float:
        return -0.6 + 0.9 * t

    def c_scalar(t: float) -> float:
        return 0.4 + 1.3 * t

    def terminal(times: list[float]) -> float:
        m = [m_scalar(t) for t in times]
        c = [c_scalar(t) for t in times]
        y1 = (1.0 + h * m[0]) * y0_value + h * c[0] * u0
        return (1.0 + h * m[1]) * y1 + h * c[1] * u1

    correct = terminal([t0, t0 + h])
    swapped = terminal([t0 + h, t0])
    assert abs(correct - swapped) > 1e-3


def test_a_coefficient_closing_over_the_control_defeats_the_skip() -> None:
    """C-17.1's boundary, held against two closed forms.

    C-17.1 states the purity of ``M``, ``C`` and ``b`` as a C-1 caller
    obligation rather than a constructed fact, because invoking a callable with
    ``t`` alone confines the *interface* and not a Python closure. This test
    exists so that the two numbers C-17.1 quotes are recomputed rather than
    remembered, and so that closing the residue (C-Q6) cannot happen silently:
    this test would then fail, forcing the clause to be rewritten with it.

    The caller keeps one array holding "the current control" and reads it from
    inside ``M`` — an innocent mistake, not a hostile one. Scalar
    ``y' = m(u) y + u`` with ``m = -0.2 + 0.7 u``, ``y_0 = 0.8``,
    ``t_span = (0, 0.5)``, ``N = 1``, explicit Euler, ``J = y_1^2 / 2``,
    evaluated at ``u = 0.3``.

    One explicit Euler step gives ``y_1 = y_0 + h (m u_val) y_0 + h u_val``,
    so with ``h = 0.5`` and ``m = 0.01``, ``y_1 = 0.954``. This package
    differentiates the form it was handed, ``f = M y + C u``, treating ``M`` as
    independent of ``u``:

        dJ/du = y_1 h C = 0.954 * 0.5 = 0.477

    The objective it actually computes varies faster, because ``m`` moves too:

        dy_1/du = h (dm/du * y_0 + 1) = 0.5 (0.7 * 0.8 + 1) = 0.78
        dJ/du = 0.954 * 0.78 = 0.74412

    Both are exact. Nothing raises, which is the point: the failure direction
    is silent. Note the damage is not confined to the skipped curvature — ``G``
    is ``C(t)``, which omits ``(dM/du) y`` outright, so the *first* derivative
    is already wrong.
    """
    h = 0.5
    y0_value = 0.8
    u_value = 0.3

    # The caller's mistake: one array serves as both the optimizer's control
    # and a value read from inside a coefficient said to depend on t alone.
    current = np.array([u_value])

    def m_leaky(t: float) -> np.ndarray:
        return np.array([[-0.2 + 0.7 * current[0]]])

    def c_unit(t: float) -> np.ndarray:
        return np.array([[1.0]])

    problem = TimeVaryingAffineDynamics(
        m_leaky, c_unit, state_dim=1, control_dim=1
    )
    opt = GLMOptimizer(
        problem,
        AnchorObjective(),
        explicit_euler(),
        t_span=(0.0, h),
        N=1,
        y0=np.array([y0_value]),
    )

    def objective(value: float) -> float:
        current[0] = value
        return opt.objective_value(np.array([[[value]]]))

    current[0] = u_value
    reported = float(opt.gradient(np.array([[[u_value]]]))[0, 0, 0])

    m = -0.2 + 0.7 * u_value
    y1 = y0_value + h * (m * y0_value + u_value)
    assert abs(y1 - 0.954) < 1e-14

    # What differentiating the declared form gives, in closed form.
    form_derivative = y1 * h * 1.0
    # What the objective this package computes actually does, in closed form.
    true_derivative = y1 * h * (0.7 * y0_value + 1.0)

    assert abs(form_derivative - 0.477) < 1e-14
    assert abs(true_derivative - 0.74412) < 1e-14

    # The package is self-consistent: it differentiates the form exactly.
    # Basis: no iterative solve on either side, so C-3 admits round-off only.
    assert abs(reported - form_derivative) < 1e-14

    # A central difference of this package's own objective sees the leak.
    # Basis: C-3.2 central difference, error O(eps^2) + O(macheps/eps);
    # at eps = 1e-6 on an O(1) quantity that is ~1e-10, so 1e-7 is loose.
    eps = 1e-6
    measured = (objective(u_value + eps) - objective(u_value - eps)) / (2 * eps)
    assert abs(measured - true_derivative) < 1e-7

    # The obligation is load-bearing: violating it is worth 0.267 here, and
    # the discrepancy is in the gradient, not merely in skipped curvature.
    assert abs(reported - measured) > 0.26

    current[0] = u_value


def test_the_closure_boundary_is_specific_to_the_violation() -> None:
    """The test above is only evidence if an *obedient* coefficient of the
    same shape agrees with the same finite difference. Otherwise it would be
    consistent with the whole route being broken rather than with the closure.
    C-14.2, applied to a discrepancy instead of to an assertion."""
    h = 0.5
    y0_value = 0.8
    u_value = 0.3

    def m_pure(t: float) -> np.ndarray:
        return np.array([[-0.2 + 0.7 * u_value]])

    def c_unit(t: float) -> np.ndarray:
        return np.array([[1.0]])

    opt = GLMOptimizer(
        TimeVaryingAffineDynamics(m_pure, c_unit, state_dim=1, control_dim=1),
        AnchorObjective(),
        explicit_euler(),
        t_span=(0.0, h),
        N=1,
        y0=np.array([y0_value]),
    )

    def objective(value: float) -> float:
        return opt.objective_value(np.array([[[value]]]))

    reported = float(opt.gradient(np.array([[[u_value]]]))[0, 0, 0])

    eps = 1e-6
    measured = (objective(u_value + eps) - objective(u_value - eps)) / (2 * eps)

    # Same mesh, same method, same eps as the test above; only the closure is
    # removed. Basis as above: central difference at eps = 1e-6.
    assert abs(reported - measured) < 1e-7


# ---------------------------------------------------------------------------
# Agreement between routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_the_affine_and_newton_routes_agree(factory) -> None:
    """Not bit-identity: a direct linear solve and a Newton iteration are
    different floating-point computations reaching the same stage values, so
    the criterion is a tolerance derived from the Newton stopping test rather
    than a pinned float (C-11.3)."""
    forced_newton = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=True,
    )
    opt_affine, u, method = build(factory)
    opt_newton, _u, _m = build(factory, structure=forced_newton)

    assert relative_error(opt_affine.gradient(u), opt_newton.gradient(u)) < (
        AFFINE_RTOL
    )

    v = make_controls(N_STEPS, method.s, NU, seed=13, scale=0.3)
    assert (
        relative_error(
            opt_affine.hessian_vector_product(u, v),
            opt_newton.hessian_vector_product(u, v),
        )
        < AFFINE_RTOL
    )


def _no_curvature_structure() -> ProblemStructure:
    """The same problem with the zero-curvature fact withheld, and with the
    constant-Jacobian fact withheld too, because it is false here."""
    return ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
        state_affine=True,
        jointly_affine=False,
    )


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_skipping_zero_curvature_is_bit_identical(factory) -> None:
    """C-16.6. The skipped terms are additions of exact zero, so this is the
    removal of an operation rather than an approximation of one, and the claim
    holds under any BLAS."""
    opt_skip, u, method = build(factory)
    opt_full, _u, _m = build(factory, structure=_no_curvature_structure())
    v = make_controls(N_STEPS, method.s, NU, seed=17, scale=0.3)
    assert np.array_equal(
        opt_skip.hessian_vector_product(u, v),
        opt_full.hessian_vector_product(u, v),
    )


@pytest.mark.parametrize("factory", ALL_METHODS)
def test_the_skip_does_not_call_the_curvature_callbacks(factory) -> None:
    """Tripwire on the instance. Affineness is verified from the class, so the
    problem stays verified while any actual call fails."""

    def forbidden(*args, **kwargs):
        raise AssertionError("a zero curvature term was computed")

    opt, u, method = build(factory)
    for name in ("F_yy_action", "F_yu_action", "F_uu_action"):
        object.__setattr__(opt.problem, name, forbidden)
    assert affine_dynamics_verified(opt.problem)

    v = make_controls(N_STEPS, method.s, NU, seed=19, scale=0.3)
    assert np.all(np.isfinite(opt.hessian_vector_product(u, v)))


def test_objective_curvature_is_not_skipped() -> None:
    """Only the *dynamics* curvature vanishes; a time-varying affine plant
    under a quadratic cost still has a nonzero Hessian."""
    opt, u, method = build(sdirk2)
    v = make_controls(N_STEPS, method.s, NU, seed=23, scale=0.3)
    assert np.max(np.abs(opt.hessian_vector_product(u, v))) > 1e-3


# ---------------------------------------------------------------------------
# The fail-safe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_declaring_a_constant_jacobian_here_is_refused(factory) -> None:
    """C-15.2 is the backstop behind the deduction, and it is load-bearing.

    ``deduce_structure`` reads ``coefficients_constant`` so this declaration
    is never made by the package. A caller can still make it by passing an
    explicit structure, and the store must then refuse rather than solve with
    the transpose of a matrix the forward solve did not use -- precedent R-9,
    whose 4.24e-05 relative gradient error passed the entire suite.
    """
    lying = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
        state_affine=True,
        jointly_affine=True,
    )
    opt, u, _m = build(factory, structure=lying)
    with pytest.raises(DeclaredStructureViolation):
        opt.gradient(u)


# ---------------------------------------------------------------------------
# Reuse across calls (C-17.6)
# ---------------------------------------------------------------------------


def _counts_per_call(opt, method) -> list[int]:
    """Factorizations taken by each of three evaluations at distinct controls.

    The third is a Hessian-vector product rather than a gradient, so that the
    count covers the call an optimizer's inner Krylov loop makes most often.
    """
    store = opt.stage_solver.factorizations
    store.reset_counts()
    counts = []
    for seed in (5, 29, 31):
        u = make_controls(N_STEPS, method.s, NU, seed=seed, scale=0.4)
        before = store.factorizations
        if seed == 31:
            v = make_controls(N_STEPS, method.s, NU, seed=37, scale=0.3)
            opt.hessian_vector_product(u, v)
        else:
            opt.gradient(u)
        counts.append(store.factorizations - before)
    return counts


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_reuse_across_calls_is_deduced_for_time_varying_coefficients(
    factory,
) -> None:
    """``F = M(t)`` depends on neither the state nor the control, so the
    matrix at each stage time is the same on every call (C-17.6)."""
    opt, _u, _m = build(factory)
    requirements = opt.requirements
    assert requirements.can_reuse_across_calls
    assert requirements.factorizations_for_repeated_solve() == 0
    assert opt.stage_solver.key_by_stage_time


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_a_repeated_solve_takes_no_factorizations(factory) -> None:
    """The certified count for C-17.6, with its contrasts.

    The first call takes exactly what C-17.4 certifies for one evaluation;
    every later call on the same mesh takes none, whatever the control. The
    constant case is asserted beside it, as in C-17.4, and a structure that
    withholds the time-only fact must keep refactoring, so that neither
    number can be vacuous.
    """
    opt, _u, method = build(factory)
    first = expected_factorizations(method, N_STEPS)
    assert _counts_per_call(opt, method) == [first, 0, 0]

    opt_const, _u, _m = build(factory, problem=constant_problem())
    assert _counts_per_call(opt_const, method) == [1, 0, 0]

    withheld = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=True,
        state_affine=True,
    )
    opt_withheld, _u, _m = build(factory, structure=withheld)
    assert not opt_withheld.requirements.can_reuse_across_calls
    assert opt_withheld.requirements.factorizations_for_repeated_solve() is None
    assert _counts_per_call(opt_withheld, method) == [first, first, first]


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_reuse_across_calls_reproduces_a_fresh_solve_exactly(factory) -> None:
    """C-15.4 for this route. A reused factorization is the factorization of
    the identical matrix, so derivatives at a later control must equal, bit
    for bit, those of an optimizer that has never factored anything. This
    holds under any BLAS, because both sides run the same LU on the same
    input."""
    opt_reusing, u_first, method = build(factory)
    opt_reusing.gradient(u_first)

    u = make_controls(N_STEPS, method.s, NU, seed=41, scale=0.4)
    v = make_controls(N_STEPS, method.s, NU, seed=43, scale=0.3)
    grad_reused = opt_reusing.gradient(u)
    hvp_reused = opt_reusing.hessian_vector_product(u, v)
    assert opt_reusing.stage_solver.factorizations.reuses > 0

    opt_fresh, _u, _m = build(factory)
    assert np.array_equal(grad_reused, opt_fresh.gradient(u))
    assert np.array_equal(hvp_reused, opt_fresh.hessian_vector_product(u, v))


@pytest.mark.parametrize("factory", IMPLICIT_METHODS)
def test_a_coefficient_that_drifts_between_calls_is_refused(factory) -> None:
    """C-17.3 makes determinism in ``t`` a caller obligation. Reuse across
    calls would turn a violation into a stale factorization, so the C-15.2
    comparison must refuse it. It can, when the later evaluation is at a
    different control: that runs a new forward solve, whose matrix at each
    stage time is compared with the one stored on the first call. The
    boundary of this guard is the next test."""
    scale = np.array([1.0])

    def m_drifting(t: float) -> np.ndarray:
        return M_0 + scale[0] * np.sin(1.7 * t) * M_1

    problem = TimeVaryingAffineDynamics(
        m_drifting, C_of_t, b_of_t, state_dim=NX, control_dim=NU
    )
    opt, u, method = build(factory, problem=problem)
    opt.gradient(u)

    scale[0] = 1.25
    u_next = make_controls(N_STEPS, method.s, NU, seed=47, scale=0.4)
    with pytest.raises(DeclaredStructureViolation) as excinfo:
        opt.gradient(u_next)
    message = str(excinfo.value)
    assert "C-17.3" in message, "the refusal must name the obligation"
    assert "jacobian_constant=True" not in message, (
        "the refusal must name the declaration reuse rested on, and here that "
        "is not a constant Jacobian"
    )


def test_a_matrix_coefficient_closing_over_the_control_is_caught_next_call() -> None:
    """C-17.1's residue, narrowed for ``M`` on an implicit route.

    The first evaluation is silently wrong exactly as C-17.1 records: nothing
    inside one solve can see a coefficient that varies only with the control.
    The next evaluation at a different control is compared against the first
    at every stage time, so the leak is refused there. Nothing is gained for
    ``C`` or ``b``, which are never factored, or for explicit methods, which
    factor nothing; C-17.1's own test covers those and still passes.
    """
    current = np.array([0.3])

    def m_leaky(t: float) -> np.ndarray:
        return np.array([[-0.2 + 0.7 * current[0]]])

    def c_unit(t: float) -> np.ndarray:
        return np.array([[1.0]])

    problem = TimeVaryingAffineDynamics(
        m_leaky, c_unit, state_dim=1, control_dim=1
    )
    opt = GLMOptimizer(
        problem,
        AnchorObjective(),
        implicit_midpoint(),
        t_span=(0.0, 0.5),
        N=1,
        y0=np.array([0.8]),
    )

    grad = opt.gradient(np.array([[[0.3]]]))
    assert np.all(np.isfinite(grad))

    current[0] = 0.45
    with pytest.raises(DeclaredStructureViolation):
        opt.gradient(np.array([[[0.45]]]))



def test_drift_is_not_seen_by_an_evaluation_served_from_the_cache() -> None:
    """The boundary C-17.6 states, held so that it cannot drift from the code.

    The comparison runs only when a factorization is requested. Repeating an
    evaluation at an unchanged control returns the optimizer's cached
    trajectory and adjoint and assembles nothing, so a coefficient that
    drifted in between goes unseen and the earlier gradient comes back. That
    is a violation of the caller's C-17.3 obligation, recorded rather than
    hardened against. If the cache ever re-checks coefficients, this test
    fails and the clause must be rewritten with it.
    """
    scale = np.array([1.0])

    def m_drifting(t: float) -> np.ndarray:
        return M_0 + scale[0] * np.sin(1.7 * t) * M_1

    problem = TimeVaryingAffineDynamics(
        m_drifting, C_of_t, b_of_t, state_dim=NX, control_dim=NU
    )
    opt, u, _method = build(sdirk2, problem=problem)
    first = opt.gradient(u)

    scale[0] = 1.25
    assert np.array_equal(opt.gradient(u), first)


def test_existing_positional_constructor_calls_keep_their_meaning() -> None:
    """``reuse_across_calls`` is keyword-only and comes last.

    Inserted mid-signature, it took the place of an existing argument:
    ``SDIRKStageSolver(False, 1.0)`` meant ``y_scale=1.0`` and instead enabled
    stage-time keying, which on a nonlinear problem raises from the store.
    The values below are chosen to differ from every default, so that a
    shifted argument is visible.
    """
    sdirk = SDIRKStageSolver(False, 2.0, False)
    dirk = DIRKStageSolver(2.0, False, False)
    coupled = ImplicitStageSolver(2.0, False, False)
    for solver in (sdirk, dirk, coupled):
        assert solver.y_scale == 2.0
        assert solver.needs_newton is False
        assert solver.key_by_stage_time is False
        assert solver.factorizations.reuse_enabled is False

    for build_solver in (
        lambda: SDIRKStageSolver(False, 2.0, False, True),
        lambda: DIRKStageSolver(2.0, False, False, True),
        lambda: ImplicitStageSolver(2.0, False, False, True),
    ):
        with pytest.raises(TypeError):
            build_solver()
