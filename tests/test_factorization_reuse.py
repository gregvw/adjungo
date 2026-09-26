"""Certification of declaration-gated factorization reuse (milestone M6).

Reuse is the one optimization in this repository that **no accuracy test can
check**. When it works, the answers are bit-identical to not reusing, because
the matrix is identical; when it silently stops working, the answers are again
bit-identical, because refactoring the same matrix gives the same factors. Only
the cost changes, in either direction, and nothing in a suite of derivative
tests can see that.

So the certified quantity here is the *count*, and `NUMERICS.md` C-15 makes it
one. These tests assert the number of ``lu_factor`` calls observed during a
solve against the number
:meth:`~adjungo.core.requirements.SolverRequirements.factorizations_for_solve`
predicts, and they assert it structurally -- never by timing, which would make
the suite a benchmark and would fail on a loaded machine.

The counts are taken from
:class:`~adjungo.solvers.factorization.FactorizationStore`, which every
implicit stage solver routes through. That is the only place ``lu_factor`` is
reached from in the stepping code, so a count taken there is complete rather
than merely representative.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from adjungo import GLMOptimizer, ProblemStructure
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
from adjungo.solvers.factorization import (
    DeclaredStructureViolation,
    FactorizationStore,
)
from adjungo.validation import reference_gradient, reference_hessian
from tests.problems import (
    ConstantJacobianQuadraticControl,
    FullCostObjective,
    LinearTimeVarying,
    StateDependentJacobian,
    make_controls,
)

N_STEPS = 12
T_SPAN = (0.35, 1.55)
Y0 = np.array([0.4, -0.25, 0.15])

CONSTANT_JACOBIAN = ProblemStructure(
    linearity=Linearity.SEMILINEAR,
    jacobian_constant=True,
    jacobian_control_dependent=False,
    has_second_derivatives=True,
)
UNDECLARED = ProblemStructure(
    linearity=Linearity.NONLINEAR,
    jacobian_constant=False,
    jacobian_control_dependent=True,
    has_second_derivatives=True,
)

IMPLICIT_METHODS = [
    ("implicit_midpoint", implicit_midpoint),
    ("implicit_trapezoid", implicit_trapezoid),
    ("sdirk2", sdirk2),
    ("sdirk3", sdirk3),
    ("gauss2", gauss2),
]


def _build(method, structure):
    problem = ConstantJacobianQuadraticControl()
    objective = FullCostObjective(nx=3, nu=2)
    optimizer = GLMOptimizer(
        problem,
        objective,
        method,
        T_SPAN,
        N_STEPS,
        Y0,
        problem_structure=structure,
    )
    u = make_controls(N_STEPS, method.s, 2, seed=11)
    return optimizer, u


def _store(optimizer) -> FactorizationStore:
    store = getattr(optimizer.stage_solver, "factorizations", None)
    assert store is not None, "the stage solver does not own a store to count"
    return store


# --------------------------------------------------------------------------
# The count is what is predicted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", IMPLICIT_METHODS)
def test_declared_constant_jacobian_factors_once_for_the_whole_solve(
    name, factory
):
    """With a declared-constant Jacobian the count is independent of ``N``.

    Every method here has a single distinct implicit stage matrix: SDIRK by
    its defining constant diagonal, the fully implicit route by solving one
    coupled ``(s*n)`` system, and the one-stage methods trivially. A constant
    ``F`` and a uniform mesh (C-3) then make that matrix the same at every
    stage of every step, so the whole solve takes exactly one factorization.

    The assertion is ``== 1``, not ``<= 2 * N``. A bound that loose would
    still pass if reuse worked within a step and not across steps, which is
    the likeliest way for this to regress.
    """
    method = factory()
    optimizer, u = _build(method, CONSTANT_JACOBIAN)
    requirements = deduce_requirements(method, CONSTANT_JACOBIAN, 3)

    predicted = requirements.factorizations_for_solve(N_STEPS)
    assert predicted == 1, (
        f"{name}: requirements predict {predicted} factorizations; this test "
        "population was chosen so that every method has exactly one distinct "
        "stage matrix"
    )

    store = _store(optimizer)
    store.reset_counts()
    optimizer.gradient(u)

    assert store.factorizations == predicted, (
        f"{name}: observed {store.factorizations} factorizations over "
        f"{N_STEPS} steps, predicted {predicted}"
    )
    assert store.reuses > 0, f"{name}: nothing was actually reused"


@pytest.mark.parametrize("name,factory", IMPLICIT_METHODS)
def test_the_backward_sweeps_take_no_factorizations(name, factory):
    """Adjoint, tangent and second-order adjoint add nothing to the count.

    This is the central structural claim of the whole design: the adjoint
    operator is the transpose of the forward Newton Jacobian, so it is applied
    with ``lu_solve(..., trans=1)`` on the factors the forward solve already
    took. The tangent uses the same factors untransposed, and the second-order
    adjoint uses the same transposed factors with a different right-hand side.

    A Hessian-vector product therefore costs the same number of factorizations
    as an objective evaluation. Were anyone to form and factor a transpose
    explicitly, every accuracy test would still pass -- the transpose of an
    LU-factored matrix and a fresh factorization of the transpose solve the
    same system -- and only this count would move.
    """
    method = factory()
    v = make_controls(N_STEPS, method.s, 2, seed=23)

    # Two optimizers, each starting from a cold store. Reusing one would
    # compare a warm store with a cold one and the second measurement would
    # be zero for the wrong reason.
    forward, u = _build(method, CONSTANT_JACOBIAN)
    forward.objective_value(u)
    forward_only = _store(forward).factorizations

    second_order, _ = _build(method, CONSTANT_JACOBIAN)
    second_order.hessian_vector_product(u, v)
    with_all_four_sweeps = _store(second_order).factorizations

    assert forward_only > 0, "nothing was measured"

    assert with_all_four_sweeps == forward_only, (
        f"{name}: a Hessian-vector product took {with_all_four_sweeps} "
        f"factorizations against {forward_only} for the forward solve alone. "
        "The backward sweeps must reuse the forward factors."
    )


def test_explicit_methods_factor_nothing():
    """An explicit method has no stage system, so it factors nothing.

    A guard against the reuse machinery being reached by a route that should
    never touch it.
    """
    method = rk4()
    structure = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=True,
    )
    requirements = deduce_requirements(method, structure, 3)

    assert requirements.factorizations_per_step == 0
    assert requirements.factorizations_for_solve(N_STEPS) is None, (
        "an explicit method cannot reuse across steps because it takes no "
        "factorizations to reuse; the prediction must be 'not applicable', "
        "not the number zero dressed up as a reuse claim"
    )

    optimizer, u = _build(method, structure)
    assert not hasattr(optimizer.stage_solver, "factorizations")
    optimizer.gradient(u)


@pytest.mark.parametrize("name,factory", IMPLICIT_METHODS)
def test_without_the_declaration_the_count_is_not_predicted(name, factory):
    """No declaration means no reuse, and the prediction says so.

    ``factorizations_for_solve`` returns ``None`` rather than a number,
    because without reuse every Newton iteration factors and the iteration
    count is a property of Newton and the initial guess, not of the dispatch.
    Returning a number there would be asserting something this module does
    not know.
    """
    method = factory()
    requirements = deduce_requirements(method, UNDECLARED, 3)
    assert requirements.factorizations_for_solve(N_STEPS) is None

    optimizer, u = _build(method, UNDECLARED)
    store = _store(optimizer)
    store.reset_counts()
    optimizer.gradient(u)

    assert store.reuses == 0, (
        f"{name}: {store.reuses} factorizations were reused although the "
        "caller declared nothing. Reuse must be gated on the declaration "
        "(R-9), never inferred from what the matrices happen to look like."
    )
    assert store.factorizations >= N_STEPS, (
        f"{name}: only {store.factorizations} factorizations without reuse "
        f"over {N_STEPS} steps; at least one per step is expected"
    )


# --------------------------------------------------------------------------
# Reuse changes no number
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", IMPLICIT_METHODS)
def test_reuse_reproduces_the_unreused_derivatives_exactly(name, factory):
    """Reuse must be bit-identical to not reusing, not merely close.

    This is the acceptance criterion the plan sets for M6: a full re-run
    proving no certified number changed. The demand for *exact* equality is
    not strictness for its own sake, and it is not a pinned float -- nothing
    here records a literal. It is what the operation means. The store returns
    a stored factorization only after verifying that its matrix is identical,
    element for element, to the one it was asked to factor. An identical
    matrix put through a deterministic ``lu_factor`` gives identical factors,
    so an identical right-hand side gives an identical solution.

    That argument is independent of platform and of the linked BLAS, because
    it never compares across two different computations: it compares a
    computation with one whose every input is equal. If this assertion ever
    fails by a small amount, the correct reading is not "roundoff" but that
    some input was not in fact identical, and the store's guard is being
    reached with something it should have rejected.
    """
    method = factory()
    reusing, u = _build(method, CONSTANT_JACOBIAN)
    plain, _ = _build(method, UNDECLARED)
    v = make_controls(N_STEPS, method.s, 2, seed=23)

    assert _store(reusing).reuse_enabled
    assert not _store(plain).reuse_enabled

    assert reusing.objective_value(u) == plain.objective_value(u), (
        f"{name}: reuse changed the objective"
    )
    np.testing.assert_array_equal(
        reusing.gradient(u),
        plain.gradient(u),
        err_msg=f"{name}: reuse changed the gradient",
    )
    np.testing.assert_array_equal(
        reusing.hessian_vector_product(u, v),
        plain.hessian_vector_product(u, v),
        err_msg=f"{name}: reuse changed the Hessian-vector product",
    )


# --------------------------------------------------------------------------
# A false declaration is refused, not absorbed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", [("sdirk3", sdirk3), ("gauss2", gauss2)])
def test_a_false_constancy_declaration_raises(name, factory):
    """Declaring a constant Jacobian for a varying one must refuse.

    This is the R-9 regression. The historical defect inferred constancy from
    a probe at ``(y_history[0], u_i, t_i)``, which varied the control and the
    time but not the state, so a state-dependent Jacobian passed it; the
    adjoint then solved with the transpose of the wrong matrix and the
    gradient was wrong by 4.24e-05 relative while the suite passed.

    :class:`StateDependentJacobian` is that exact shape: ``F`` depends on
    ``y`` alone, so it is control-independent and time-independent and still
    differs at every stage. Reuse must therefore be refused loudly, and what
    refuses it is not a heuristic but the direct observation that two matrices
    declared identical are not.
    """
    problem = StateDependentJacobian()
    objective = FullCostObjective(nx=2, nu=2)
    optimizer = GLMOptimizer(
        problem,
        objective,
        factory(),
        T_SPAN,
        N_STEPS,
        np.array([0.5, -0.4]),
        problem_structure=CONSTANT_JACOBIAN,
    )
    u = make_controls(N_STEPS, factory().s, 2, seed=5)

    with pytest.raises(DeclaredStructureViolation) as excinfo:
        optimizer.gradient(u)

    message = str(excinfo.value)
    assert "jacobian_constant" in message, (
        "the message must name the declaration that is false, so that the "
        "caller knows what to change"
    )
    assert "R-9" in message, "the message must cite the controlling precedent"


def test_the_same_problem_is_correct_when_declared_honestly():
    """The refusal is about the declaration, not about the problem.

    Complement to the test above: with ``jacobian_constant=False`` the very
    same problem and method produce a gradient, and it matches a fixed-mesh
    central difference of the objective the optimizer itself evaluates. A
    guard that refused the problem outright would be over-broad; this shows
    it is not.
    """
    problem = StateDependentJacobian()
    objective = FullCostObjective(nx=2, nu=2)
    method = sdirk3()
    optimizer = GLMOptimizer(
        problem,
        objective,
        method,
        T_SPAN,
        8,
        np.array([0.5, -0.4]),
        problem_structure=UNDECLARED,
    )
    u = make_controls(8, method.s, 2, seed=5)

    analytic = optimizer.gradient(u)

    # Central difference on the fixed mesh: this differentiates exactly the
    # discrete objective the optimizer evaluates, so any disagreement is a
    # defect and not truncation error (C-2, C-3).
    eps = 1e-6
    numeric = np.zeros_like(u)
    for index in np.ndindex(u.shape):
        step = np.zeros_like(u)
        step[index] = eps
        numeric[index] = (
            optimizer.objective_value(u + step)
            - optimizer.objective_value(u - step)
        ) / (2.0 * eps)

    scale = max(float(np.max(np.abs(numeric))), 1.0)
    assert np.max(np.abs(analytic - numeric)) / scale < 1e-7


class _DeclaredLinearTimeVarying(LinearTimeVarying):
    """``LinearTimeVarying`` routed through the ``LINEAR`` branch.

    The parent is ``LINEAR`` by that member's own definition -- ``F`` depends
    on ``t`` alone -- but declares no ``linearity``, so it is deduced
    ``NONLINEAR`` and never reaches the deduction that could misread it.
    Pinning a counterexample is not the same as routing it through the code
    that could misuse it (C-16.8, item 2).
    """

    linearity = Linearity.LINEAR


# Basis (R-13), measured on this fixture across all five implicit families.
#
# Rounding side: package and reference agree to between 5.6e-17 and 2.2e-16
# relative, gradient and Hessian-vector product alike. The two are not the
# same computation done twice. The package walks the trajectory, solving each
# stage matrix in turn -- 10 to 30 factorizations here. The reference forms
# the monolithic residual over the whole trajectory, 60 to 100 unknowns, and
# applies Newton to it; because f is affine in y that converges in exactly one
# iteration to a residual of 3.9e-16 to 6.6e-16. Agreement at rounding between
# a sequential and a monolithic route is the C-14.1 evidence this test wants.
#
# Defect side: a stale factorization reused across these t-varying stage
# matrices -- measured with the C-15.2 comparison bypassed on the Newton
# route, where nothing else notices -- moves the gradient by 6.9e-4 to 2.8e-3.
#
# 1e-11 sits five orders above the rounding and seven below the defect.
_LINEAR_ROUTE_RTOL = 1e-11


@pytest.mark.parametrize("name,factory", IMPLICIT_METHODS)
def test_a_declared_linear_time_varying_problem_is_solved_not_refused(
    name, factory
):
    """``LINEAR`` is not a constancy declaration (C-15.1).

    ``Linearity.LINEAR`` says ``F`` is independent of ``y`` and ``u``; it says
    nothing about ``t``, and ``F = M(t)`` satisfies it (C-16.1). The deduction
    used to read it as ``jacobian_constant=True``, which made the C-15.1
    declaration on the caller's behalf, routed this problem to reuse, and had
    the C-15.2 guard refuse it at the second stage matrix. The guard was right;
    the deduction was not.

    The problem must now be solved on the linear route with reuse off, and
    both derivatives must match the independent reference (C-14.1).
    """
    method = factory()
    problem = _DeclaredLinearTimeVarying()
    objective = FullCostObjective(nx=2, nu=2)
    y0 = np.array([0.6, -0.4])
    optimizer = GLMOptimizer(problem, objective, method, T_SPAN, N_STEPS, y0)
    u = make_controls(N_STEPS, method.s, 2, seed=7)
    v = make_controls(N_STEPS, method.s, 2, seed=8)

    assert optimizer.problem_structure.linearity is Linearity.LINEAR
    assert optimizer.problem_structure.jacobian_constant is False
    assert not optimizer.requirements.needs_newton
    assert not _store(optimizer).reuse_enabled

    grad = optimizer.gradient(u)
    grad_ref = reference_gradient(
        y0, u, T_SPAN, N_STEPS, problem, method, objective
    )
    hessian = reference_hessian(
        y0, u, T_SPAN, N_STEPS, problem, method, objective
    )
    hvp = optimizer.hessian_vector_product(u, v)
    hvp_ref = (hessian.reshape(u.size, u.size) @ v.ravel()).reshape(u.shape)

    for label, value, reference in (
        ("gradient", grad, grad_ref),
        ("Hessian-vector product", hvp, hvp_ref),
    ):
        scale = max(float(np.max(np.abs(reference))), 1.0)
        err = float(np.max(np.abs(value - reference))) / scale
        assert err < _LINEAR_ROUTE_RTOL, (
            f"{name}: {label} differs from the independent reference by "
            f"{err:.3e} (relative, C-3 basis)"
        )


# --------------------------------------------------------------------------
# The store itself
# --------------------------------------------------------------------------


def test_store_disabled_is_a_counting_pass_through():
    store = FactorizationStore(reuse_enabled=False)
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])

    first = store.factor(("k",), matrix)
    second = store.factor(("k",), matrix)

    assert store.factorizations == 2
    assert store.reuses == 0
    assert first is not second


def test_store_returns_the_identical_object_on_a_hit():
    """Reuse must return the same factors, not an equal-looking copy."""
    store = FactorizationStore(reuse_enabled=True)
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])

    first = store.factor(("k",), matrix)
    second = store.factor(("k",), matrix.copy())

    assert first is second
    assert store.factorizations == 1
    assert store.reuses == 1


def test_store_rejects_a_one_ulp_difference():
    """The comparison is exact; there is no tolerance to tune.

    A one-unit-in-the-last-place difference means the two stage matrices are
    different matrices, and reusing across them is an approximation. The
    contract's subject is the derivative of the discrete map as implemented
    (C-2), so an approximation here is a different answer, however small.
    """
    store = FactorizationStore(reuse_enabled=True)
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    store.factor(("k",), matrix)

    perturbed = matrix.copy()
    perturbed[0, 0] = np.nextafter(perturbed[0, 0], np.inf)
    assert perturbed[0, 0] != matrix[0, 0]

    with pytest.raises(DeclaredStructureViolation):
        store.factor(("k",), perturbed)


def test_store_copies_its_reference_matrix():
    """A stored view of a caller's scratch buffer would disable the guard.

    The stage solvers build ``I - h a F`` into temporaries. If the store kept
    a view rather than a copy, the reference would change with the buffer and
    every later comparison would compare the new matrix with itself, so the
    guard would always pass and reuse would become unconditional -- the R-9
    failure reintroduced through aliasing rather than through a probe.
    """
    store = FactorizationStore(reuse_enabled=True)
    scratch = np.array([[2.0, 1.0], [1.0, 3.0]])
    store.factor(("k",), scratch)

    scratch[0, 0] = 99.0

    with pytest.raises(DeclaredStructureViolation):
        store.factor(("k",), scratch)


def test_store_refuses_rather_than_reuses_on_a_key_collision():
    """A key is a hint. Correctness may not depend on choosing it well.

    Two different matrices under one key is a programming error in a solver,
    not in the caller's problem. The error direction that matters is that it
    can only cause a refusal or a redundant factorization, never a silent
    reuse of the wrong factors.
    """
    store = FactorizationStore(reuse_enabled=True)
    store.factor(("collide",), np.array([[2.0, 0.0], [0.0, 2.0]]))

    with pytest.raises(DeclaredStructureViolation):
        store.factor(("collide",), np.array([[5.0, 1.0], [0.0, 4.0]]))


def test_store_reports_the_size_of_the_discrepancy():
    store = FactorizationStore(reuse_enabled=True)
    store.factor(("k",), np.eye(2))

    with pytest.raises(DeclaredStructureViolation) as excinfo:
        store.factor(("k",), np.eye(2) * 1.5)

    assert excinfo.value.largest_difference == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The count must be complete, not merely representative
# --------------------------------------------------------------------------


def test_no_factorization_bypasses_the_store(monkeypatch):
    """Every ``lu_factor`` in a solve is one the store knows about.

    Everything above counts factorizations by asking the store. That measures
    the right thing only if the store is the sole route to ``lu_factor``, and
    nothing so far establishes that it is.

    It matters because the bypass is undetectable by every other test here.
    Defect injection M7 restored one direct ``scipy.linalg.lu_factor`` call in
    :meth:`~adjungo.solvers.newton.NewtonMixin.newton_solve` -- the one taken
    at the converged iterate -- and *nothing failed*. The store still reported
    one factorization, because the extra one never reached it; the answers
    were unchanged, because it factored the same matrix. A real per-stage,
    per-step cost had returned and the count said the optimization was
    working.

    This test counts at ``scipy.linalg.lu_factor`` itself and requires the two
    numbers to agree. The patch is applied to the ``scipy.linalg`` module
    attribute, which is how every call site in ``adjungo`` reaches it; none
    imports the name directly, and a future one that did would show up here as
    a shortfall rather than pass silently.
    """
    calls = {"n": 0}
    real = scipy.linalg.lu_factor

    def counting(matrix, *args, **kwargs):
        calls["n"] += 1
        return real(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)

    method = sdirk3()
    optimizer, u = _build(method, CONSTANT_JACOBIAN)
    v = make_controls(N_STEPS, method.s, 2, seed=23)
    store = _store(optimizer)

    optimizer.hessian_vector_product(u, v)

    assert calls["n"] == store.factorizations, (
        f"{calls['n']} calls to scipy.linalg.lu_factor, but the store counted "
        f"{store.factorizations}. Something factors outside the store, so the "
        "C-15.3 count understates the real cost and cannot detect reuse "
        "regressing."
    )
    assert calls["n"] == 1, (
        f"{calls['n']} factorizations for a whole Hessian-vector product on a "
        "declared-constant Jacobian; exactly one is required"
    )


def test_undeclared_solves_also_route_every_factorization_through_the_store(
    monkeypatch,
):
    """The completeness claim must hold on the non-reusing path too.

    With reuse disabled the store is a pass-through, and a bypass there would
    be equally invisible -- more so, since no count is predicted at all.
    """
    calls = {"n": 0}
    real = scipy.linalg.lu_factor

    def counting(matrix, *args, **kwargs):
        calls["n"] += 1
        return real(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_factor", counting)

    optimizer, u = _build(sdirk3(), UNDECLARED)
    store = _store(optimizer)
    optimizer.gradient(u)

    assert calls["n"] == store.factorizations
    assert calls["n"] > 0


# --------------------------------------------------------------------------
# The deduction rule itself
# --------------------------------------------------------------------------


def test_stage_reuse_is_deduced_from_constancy_alone():
    """``can_reuse_across_stages`` must follow ``jacobian_constant``, nothing else.

    Until M6 this read ``jacobian_constant or not jacobian_control_dependent``.
    The second disjunct is false for precisely the R-9 case: a Jacobian
    depending on the state alone is control-independent and still differs at
    every stage.

    The flag is asserted here directly because no solver currently reads it --
    they consume ``can_reuse_across_steps``, which carries the same value on a
    uniform mesh. That is exactly the condition under which the original
    defect survived: it was wrong for years and inert, so no behavioural test
    could reach it. An unread gate needs a test *of the gate*, or its
    correctness is untested by construction (C-15.1).
    """
    control_independent_but_state_dependent = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
    )

    for method in (sdirk3(), gauss2(), implicit_midpoint()):
        requirements = deduce_requirements(
            method, control_independent_but_state_dependent, 3
        )
        assert requirements.can_reuse_across_stages is False, (
            "control-independence was taken as evidence of constancy; that is "
            "the R-9 defect restated as a deduction rule"
        )
        assert requirements.can_reuse_across_steps is False

        constant = deduce_requirements(method, CONSTANT_JACOBIAN, 3)
        assert constant.can_reuse_across_stages is True
        assert constant.can_reuse_across_steps is True
