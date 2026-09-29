"""C-18.7 acceptance for the discretization plan.

Item 1 of that clause -- the degenerate plan (uniform nodes, one method)
reproduces the present results -- is the rest of the suite, which runs
unchanged through :meth:`DiscretizationPlan.uniform`. This module covers the
parts the existing suite cannot reach:

* **Item 2**, non-uniform ``h_n`` and prescribed method changes, including
  different stage counts, against the C-14.1 tier-1 reference, for gradients
  and Hessian-vector products both. The mesh of each plan is held fixed
  throughout, per C-2.
* **Item 3**, continuous refinement under C-4 against a closed-form solution,
  on a plan refined *as a plan*: a graded mesh whose ratio of largest to
  smallest step stays constant while the whole plan is refined. Item 2
  compares derivatives at one fixed plan and would pass for a discretization
  that converged to the wrong thing.
* **Item 4**, a plan change at unchanged control parameters agreeing with a
  fresh evaluation. Items 2 and 3 use one plan per optimizer, so neither can
  see a stale trajectory, a stale control sampling, or a factorization cache
  that outlived the plan it was keyed to.

The oracle hierarchy is C-14.1's, unchanged: the independent monolithic
reference is the primary derivative oracle here, with a fixed-mesh eps-sweep
beside it rather than in place of it.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize

from adjungo.core.method import StageType
from adjungo.core.plan import DiscretizationPlan
from adjungo.methods.runge_kutta import (
    explicit_euler,
    gauss2,
    heun,
    implicit_midpoint,
    implicit_trapezoid,
    rk4,
    sdirk2,
    sdirk3,
)
from adjungo.optimization import interface
from adjungo.optimization.interface import GLMOptimizer
from adjungo.optimization.parametrization import (
    NodalControl,
    PiecewiseConstantControl,
)
from adjungo.validation import reference_gradient, reference_hessian
from tests.problems import (
    AnchorObjective,
    ConstantJacobianQuadraticControl,
    CoupledNonlinear,
    FullCostObjective,
    ScalarAnchor,
)
from tests.test_factorization_reuse import CONSTANT_JACOBIAN
from tests.test_oracle_gradient import certified_rtol

# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def graded_nodes(t_span: tuple[float, float], N: int, ratio: float) -> np.ndarray:
    """Nodes whose step sizes form a geometric sequence of total ratio ``ratio``.

    ``h_{n+1} / h_n`` is constant, so ``h_max / h_min == ratio`` for every
    ``N``. Refining such a mesh keeps its shape and shrinks every step
    proportionally, which is what makes C-4's ``e ~ h^p`` meaningful on it:
    the family is one mesh refined, not a sequence of unrelated meshes.
    """
    t0, t1 = t_span
    q = ratio ** (1.0 / (N - 1)) if N > 1 else 1.0
    h = q ** np.arange(N)
    nodes = np.empty(N + 1)
    nodes[0] = t0
    nodes[1:] = t0 + (t1 - t0) * np.cumsum(h) / h.sum()
    nodes[-1] = t1  # cumsum leaves the endpoint a rounding away
    return nodes


def mixed_plan(t_span: tuple[float, float], factories, ratio: float = 4.0):
    """A graded mesh whose steps cycle through ``factories``.

    Both axes C-18.1 allows vary at once: the step sizes are all different and
    so are the methods, including their stage counts. A plan that varied only
    one of them could not expose a stage offset computed from a stage count
    that belongs to a different step.
    """
    methods = [factory() for factory in factories]
    return DiscretizationPlan(
        nodes=graded_nodes(t_span, len(methods), ratio),
        methods=tuple(methods),
    )


def packed_controls(plan: DiscretizationPlan, nu: int, seed: int = 0) -> np.ndarray:
    """Controls in the packed ``(sum_n s_n, nu)`` layout, all entries distinct.

    A control that repeated across stages or steps could not expose a stage
    index read from the wrong step's block.
    """
    rng = np.random.default_rng(seed)
    return 0.5 * rng.standard_normal((plan.total_stages, nu))


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Error relative to ``max(|b|_inf, 1)`` (clause C-3)."""
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1.0))


def _plan_rtol(plan: DiscretizationPlan) -> float:
    """The loosest C-3 tolerance any method in the plan carries.

    A plan is a sequence of independent method choices, so the accuracy it can
    be held to is the weakest of theirs: one implicit step anywhere puts the
    whole solve on the C-3.4 Newton-radius budget.
    """
    return max(certified_rtol(method) for method in plan.methods)


#: Plans exercising C-18.1's two axes. Each mixes stage counts, so every
#: consumer of a stage offset is asked for a different answer at each step.
MIXED_PLANS = [
    pytest.param(
        (explicit_euler, heun, rk4, explicit_euler, heun),
        id="explicit_s1_s2_s4",
    ),
    pytest.param(
        (implicit_midpoint, sdirk2, sdirk3, implicit_trapezoid),
        id="implicit_s1_s2_s3_s2",
    ),
    pytest.param(
        (gauss2, sdirk3, gauss2),
        id="fully_implicit_s2_s3_s2",
    ),
    pytest.param(
        (heun, sdirk2, rk4, implicit_midpoint),
        id="explicit_and_implicit_mixed",
    ),
]


#: Plans mixing *methods* and step sizes while holding the stage count fixed.
#: These are the plans on which a scalar objective value exists, because the
#: rectangular ``(N, s, nu)`` layout ``Objective.evaluate`` is specified on
#: still describes them. They are also the first case an adaptive policy would
#: reach for -- switching an explicit method for an implicit one where the
#: problem stiffens -- so they are worth exercising on their own.
SAME_STAGE_COUNT_PLANS = [
    pytest.param(
        (explicit_euler, implicit_midpoint, explicit_euler, implicit_midpoint),
        id="s1_explicit_and_sdirk",
    ),
    pytest.param(
        (heun, implicit_trapezoid, sdirk2, gauss2),
        id="s2_explicit_dirk_sdirk_and_fully_implicit",
    ),
]


def _case(factories, ratio: float = 4.0, seed: int = 0):
    """A C-14 population instance on a mixed plan."""
    t_span = (0.3, 1.1)
    plan = mixed_plan(t_span, factories, ratio)
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    u = packed_controls(plan, problem.control_dim, seed=seed)
    return plan, problem, objective, y0, u


class _WithoutPackedEvaluation:
    """An objective that forwards everything except ``evaluate_packed``.

    Every objective in this repository now offers the packed route, so the
    refusal has nothing to fire on without one that does not. Deleting the
    method from a copy is closer to a third-party objective than editing the
    shipped one would be.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        if name == "evaluate_packed":
            raise AttributeError(name)
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# C-18.7 item 2 -- against the tier-1 reference, at a fixed plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factories", MIXED_PLANS)
def test_gradient_on_a_mixed_plan_matches_the_tier_one_reference(factories):
    """C-18.7 item 2, gradients.

    The reference assembles the whole plan as one algebraic residual and
    differentiates it densely, sharing no code with ``adjungo/stepping/``. It
    accumulates its own stage offsets, so an offset error on either side shows
    up as a disagreement rather than cancelling.
    """
    plan, problem, objective, y0, u = _case(factories)

    got = GLMOptimizer(problem, objective, y0=y0, plan=plan).gradient(u)
    expected = reference_gradient(
        y0, u, problem=problem, objective=objective, plan=plan
    )

    assert got.shape == u.shape
    assert _rel_err(got, expected) < _plan_rtol(plan)


@pytest.mark.parametrize("factories", MIXED_PLANS)
def test_hvp_on_a_mixed_plan_matches_the_tier_one_reference(factories):
    """C-18.7 item 2, Hessian-vector products.

    The gradient can be right while the second-order path reads a stage from
    the wrong step: the sensitivity sweeps carry their own packed arrays.
    Contracting the dense reference Hessian against the same direction tests
    those separately.
    """
    plan, problem, objective, y0, u = _case(factories, seed=3)
    rng = np.random.default_rng(11)
    v = rng.standard_normal(u.shape)

    got = GLMOptimizer(
        problem, objective, y0=y0, plan=plan
    ).hessian_vector_product(u, v)
    H = reference_hessian(y0, u, problem=problem, objective=objective, plan=plan)
    expected = (H @ v.reshape(-1)).reshape(v.shape)

    assert got.shape == v.shape
    assert _rel_err(got, expected) < _plan_rtol(plan)


@pytest.mark.parametrize("factories", SAME_STAGE_COUNT_PLANS)
def test_the_mixed_plan_gradient_survives_a_fixed_mesh_eps_sweep(factories):
    """C-14.1 level 3, beside the reference rather than in place of it.

    Corroboration only: the plan is fixed for the whole sweep, per C-2, so a
    plateau in the relative error across two decades of ``eps`` can only come
    from the derivative, never from the discretization. This is the check
    that would still have something to say if the reference and the package
    ever shared a mistake.

    It runs on the fixed-stage-count plans because it needs the scalar
    objective value, which a varying-stage-count plan does not have; see
    :func:`test_the_objective_value_is_refused_for_a_varying_stage_count`.
    The methods and the step sizes still both vary.
    """
    plan, problem, objective, y0, u = _case(factories, seed=5)
    optimizer = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    grad = optimizer.gradient(u)

    rng = np.random.default_rng(2)
    v = rng.standard_normal(u.shape)
    v /= np.linalg.norm(v)
    directional = float(np.sum(grad * v))

    errors = []
    for eps in (1e-5, 1e-6, 1e-7):
        plus = optimizer.objective_value(u + eps * v)
        minus = optimizer.objective_value(u - eps * v)
        central = (plus - minus) / (2.0 * eps)
        errors.append(abs(central - directional) / max(abs(directional), 1.0))

    # A central difference of an exact derivative decays as eps^2 until
    # cancellation lifts it; what disqualifies a wrong derivative is a floor
    # that does not move. 1e-6 is two decades below the 1e-4-and-worse errors
    # of every derivative defect this repository has cured (R-1, R-2, R-9).
    assert min(errors) < 1e-6


# ---------------------------------------------------------------------------
# C-18.7 item 3 -- continuous refinement under C-4
# ---------------------------------------------------------------------------


class DecayWithConstantForcing:
    """``y' = -k y + u``: a scalar problem with a closed-form solution.

    Used only for the continuous-order study, where what is needed is an
    exact ``y(t)`` rather than the structural coverage of the C-14 population.
    """

    state_dim = 1
    control_dim = 1

    def __init__(self, k: float = 1.3) -> None:
        self.k = float(k)

    def f(self, y, u, t):
        return np.array([-self.k * y[0] + u[0]])

    def F(self, y, u, t):
        return np.array([[-self.k]])

    def G(self, y, u, t):
        return np.array([[1.0]])

    def exact(self, t: float, y0: float, u: float) -> float:
        """``y(t)`` for the constant control ``u``."""
        k = self.k
        return float(u / k + (y0 - u / k) * np.exp(-k * t))


@pytest.mark.parametrize(
    "factory,order",
    [
        pytest.param(explicit_euler, 1, id="explicit_euler_p1"),
        pytest.param(heun, 2, id="heun_p2"),
        pytest.param(implicit_midpoint, 2, id="implicit_midpoint_p2"),
        pytest.param(rk4, 4, id="rk4_p4"),
    ],
)
def test_a_graded_plan_refined_as_a_plan_attains_the_methods_order(factory, order):
    """C-18.7 item 3 / C-4.

    The mesh is graded four-to-one and refined with its grading held fixed, so
    the family is one plan refined rather than a sequence of unrelated meshes.
    ``h`` is reported as the largest step, which is the one the error constant
    is governed by; any fixed multiple of it would give the same slope.

    This is the check item 2 cannot make. Item 2 compares two computations of
    the same discrete objective, and a plan whose stage times were assembled
    from the wrong step's abscissae would still agree with a reference built
    from those same wrong times -- while converging to the wrong function of
    ``t``, or to nothing.
    """
    problem = DecayWithConstantForcing()
    y0 = np.array([0.7])
    u_value = 0.45
    t_span = (0.0, 1.0)

    errors, steps = [], []
    for N in (8, 16, 32, 64):
        method = factory()
        plan = DiscretizationPlan(
            nodes=graded_nodes(t_span, N, ratio=4.0),
            methods=tuple(method for _ in range(N)),
        )
        u = np.full((plan.total_stages, 1), u_value)
        optimizer = GLMOptimizer(
            problem, AnchorObjective(), y0=y0, plan=plan
        )
        y_end = optimizer.trajectory(u).Y[-1, 0, 0]
        exact = problem.exact(t_span[1], y0[0], u_value)
        errors.append(abs(y_end - exact))
        steps.append(float(plan.h.max()))

    rates = [
        np.log(errors[i] / errors[i + 1]) / np.log(steps[i] / steps[i + 1])
        for i in range(len(errors) - 1)
    ]
    # 0.25 admits the pre-asymptotic first pair while excluding the adjacent
    # integer order in both directions, which is what this test must separate.
    assert min(rates) > order - 0.25, f"observed rates {rates}"


# ---------------------------------------------------------------------------
# C-18.7 item 4 -- a plan change at unchanged control parameters
# ---------------------------------------------------------------------------


def _fresh(plan, problem, objective, y0, u):
    """Objective, gradient and HVP from an optimizer that has seen one plan."""
    optimizer = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    rng = np.random.default_rng(7)
    v = rng.standard_normal(u.shape)
    return (
        optimizer.objective_value(u),
        optimizer.gradient(u),
        optimizer.hessian_vector_product(u, v),
        v,
    )


def test_with_plan_at_the_same_control_agrees_with_a_fresh_evaluation():
    """C-18.7 item 4.

    Two plans with the *same* stage counts, so the identical control array is
    admissible under both and nothing but the mesh changes. Re-discretizing an
    optimizer that has already solved the first plan must agree with
    evaluating the second from scratch. A trajectory, a control sampling or a
    factorization cache that outlived the plan it was keyed to shows up here
    and nowhere else in this module.

    Two *fresh* optimizers would not show it: the point is a warm object.
    """
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    t_span = (0.3, 1.1)
    method = sdirk2()
    N = 5

    first = DiscretizationPlan.uniform(t_span, N, method)
    second = DiscretizationPlan(
        nodes=graded_nodes(t_span, N, ratio=3.0),
        methods=tuple(method for _ in range(N)),
    )
    u = packed_controls(first, problem.control_dim, seed=13)
    assert second.total_stages == first.total_stages

    warm = GLMOptimizer(problem, objective, y0=y0, plan=first)
    J_first = warm.objective_value(u)
    g_first = warm.gradient(u)

    moved = warm.with_plan(second)
    J_ref, g_ref, hv_ref, v = _fresh(second, problem, objective, y0, u)

    assert moved.objective_value(u) == pytest.approx(J_ref, rel=1e-14, abs=1e-15)
    assert moved.gradient(u) == pytest.approx(g_ref, rel=1e-13, abs=1e-14)
    assert moved.hessian_vector_product(u, v) == pytest.approx(
        hv_ref, rel=1e-13, abs=1e-14
    )

    # The two plans must genuinely disagree, or the comparison above is
    # satisfied by any implementation that ignores the plan entirely.
    assert not np.allclose(g_ref, g_first, rtol=1e-6, atol=1e-8)

    # And the original is left describing its own plan: with_plan builds a
    # new optimizer rather than mutating this one.
    assert warm.plan is first
    assert warm.objective_value(u) == J_first


def test_rebinding_the_plan_is_refused():
    """C-18.2: the plan a trajectory was recorded against cannot move.

    ``optimizer.plan = other`` would leave the cached trajectory, adjoint,
    stage solvers and factorization stores describing the previous
    discretization, and the cache is keyed on the controls alone. Measured
    before this was refused, on ``y' = u``, ``J = y(1)²/2`` with two Euler
    steps and stage controls ``(1, 3)``: moving the interior node from ``½``
    to ``¼`` and re-evaluating at the same controls returned the first plan's
    ``J = 2.0`` instead of ``3.125``.
    """
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    first = DiscretizationPlan.uniform((0.3, 1.1), 4, sdirk2())
    second = DiscretizationPlan.uniform((0.3, 1.1), 8, sdirk2())

    optimizer = GLMOptimizer(problem, objective, y0=y0, plan=first)
    with pytest.raises(AttributeError, match="read-only"):
        optimizer.plan = second
    assert optimizer.plan is first


def test_changing_the_plan_invalidates_a_cached_factorization():
    """C-18.7 item 4, the C-15 cache specifically.

    ``jacobian_constant`` licenses reuse of one LU factorization across every
    step (C-15.1), so this needs a problem and a declaration that actually
    enable reuse -- otherwise there is no cache to invalidate and the test
    passes for the wrong reason.

    A store keyed to nothing but the problem would survive a change of ``h``,
    and ``I - h a_ii F`` is a different matrix at a different ``h``, so the
    second plan would be solved with the first plan's factors. Counting is
    the only way to see this: refactorizing the same matrix gives the same
    answer, and reusing the wrong one gives a wrong answer that no structural
    identity detects.
    """
    problem = ConstantJacobianQuadraticControl()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    t_span = (0.3, 1.1)
    method = implicit_midpoint()
    N = 4

    coarse = DiscretizationPlan.uniform(t_span, N, method)
    fine = DiscretizationPlan.uniform(t_span, 2 * N, method)
    u_coarse = packed_controls(coarse, problem.control_dim, seed=21)
    u_fine = packed_controls(fine, problem.control_dim, seed=21)

    warm = GLMOptimizer(
        problem, objective, y0=y0, plan=coarse,
        problem_structure=CONSTANT_JACOBIAN,
    )
    warm.gradient(u_coarse)
    store = warm.stage_solver.factorizations
    # Reuse really is on, or the invalidation below tests nothing.
    assert store.factorizations == 1, (
        f"the coarse solve made {store.factorizations} factorizations where "
        f"C-15.1 predicts 1; this fixture is not exercising reuse"
    )

    moved = warm.with_plan(fine)
    after = moved.gradient(u_fine)

    # A new plan means a new solver and a new store; the count starts again.
    assert moved.stage_solver.factorizations.factorizations == 1
    assert moved.stage_solver is not warm.stage_solver

    expected = reference_gradient(
        y0, u_fine, problem=problem, objective=objective, plan=fine
    )
    assert _rel_err(after, expected) < certified_rtol(method)

    # The first store is untouched: nothing was silently re-keyed under it.
    assert store.factorizations == 1


# ---------------------------------------------------------------------------
# The uniform plan's step sizes are exactly equal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t_span,N",
    [
        ((0.0, 1.0), 10),
        ((0.3, 1.1), 7),
        ((-3.0, 10.7), 1000),
        ((100.0, 105.0), 13),
        ((1.0, 0.0), 10),
    ],
)
def test_a_uniform_plan_gives_every_step_the_identical_step_size(t_span, N):
    """Exact equality, not approximate -- C-15.1 compares element for element.

    ``np.diff(np.linspace(a, b, N + 1))`` spans two values one ulp apart. That
    is harmless for accuracy and fatal for factorization reuse: the stage
    matrix ``I - h a_ii F`` built from a step size that differs in its last
    bit is a *different matrix* at that step, the stored one fails the C-15.1
    comparison, and the solve silently refactorizes once per step. Measured
    before ``step_sizes`` was made explicit: 5 factorizations over 12 steps of
    ``implicit_midpoint`` where ``SolverRequirements`` predicts 1.

    This pins the cause. ``tests/test_factorization_reuse.py`` counts the
    factorizations and would fail again if this regressed, but it would not
    say why.
    """
    plan = DiscretizationPlan.uniform(t_span, N, explicit_euler())
    h = plan.h
    assert h.shape == (N,)
    assert np.all(h == h[0]), f"step sizes span {h.min()!r} .. {h.max()!r}"
    assert plan.nodes[0] == t_span[0]
    assert plan.nodes[-1] == t_span[1]


def test_a_uniform_plans_stage_times_are_unchanged_by_the_plan():
    """The plan must not move stage times relative to the pre-C-18 convention.

    ``t_n = t0 + n*h`` with the multiplier ``h`` is what the scalar-``h`` code
    evaluated ``F(t)`` at. Reproducing it bit for bit is what makes C-18.7
    item 1 -- the existing suite as the regression -- a real check on a
    time-varying right-hand side rather than one absorbed into tolerances.
    """
    t_span, N = (0.3, 1.1), 7
    method = sdirk3()
    plan = DiscretizationPlan.uniform(t_span, N, method)
    h = (t_span[1] - t_span[0]) / N

    for n in range(N):
        legacy = t_span[0] + n * h + method.c * h
        assert np.array_equal(plan.stage_times(n), legacy)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_varying_stage_count_refuses_a_rectangular_control():
    """C-7: no rectangular shape describes a plan whose stage count varies.

    Reshaping such a control to the packed length would silently reassign
    stages to the wrong steps, which is exactly the class of error item 2
    exists to catch, so the refusal is loud.
    """
    plan, problem, objective, y0, _ = _case(MIXED_PLANS[0].values[0])
    optimizer = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    rect = np.zeros((plan.N, plan.methods[0].s, problem.control_dim))
    with pytest.raises(ValueError, match="stage count varies"):
        optimizer.gradient(rect)


def test_supplying_both_a_plan_and_the_uniform_arguments_is_refused():
    """C-18.1: two meshes were described; neither may be silently chosen."""
    plan = DiscretizationPlan.uniform((0.0, 1.0), 3, explicit_euler())
    with pytest.raises(TypeError, match="plan"):
        GLMOptimizer(
            ScalarAnchor(),
            AnchorObjective(),
            explicit_euler(),
            (0.0, 1.0),
            3,
            np.array([0.25]),
            plan=plan,
        )


def test_a_single_step_size_is_refused_for_a_graded_plan():
    """C-7: reporting step 0's ``h`` would name it as the solve's step size."""
    plan = DiscretizationPlan(
        nodes=graded_nodes((0.0, 1.0), 4, ratio=3.0),
        methods=tuple(explicit_euler() for _ in range(4)),
    )
    optimizer = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.array([0.25]), plan=plan
    )
    with pytest.raises(ValueError, match="single step size"):
        _ = optimizer.h


def test_a_single_method_is_refused_for_a_mixed_plan():
    """C-7, the same for the tableau: a mixed plan has no one method."""
    plan = mixed_plan((0.0, 1.0), (explicit_euler, heun, rk4))
    optimizer = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.array([0.25]), plan=plan
    )
    with pytest.raises(ValueError, match="single method"):
        _ = optimizer.method


def test_the_envelope_is_enforced_at_every_step_not_only_the_first(monkeypatch):
    """C-6.1 / C-18.1: one uncertified step puts the whole plan outside.

    A guard reading ``plan.methods[0]`` would admit this plan, and the
    uncertified step would be executed by a route with no certified accuracy
    claim.

    Every family in C-6.1 is currently certified, so the uncertified step is
    produced by narrowing the certified set for the duration of the test. That
    tests the loop, which is what changed, and not the contents of the set,
    which ``tests/test_documentation.py`` checks against C-6.1.
    """
    plan = DiscretizationPlan(
        nodes=np.array([0.0, 0.4, 1.0]),
        methods=(explicit_euler(), sdirk2()),
    )
    assert plan.methods[0].stage_type is StageType.EXPLICIT

    monkeypatch.setattr(
        interface, "CERTIFIED_STAGE_TYPES", frozenset({StageType.EXPLICIT})
    )
    with pytest.raises(NotImplementedError, match="at step 1"):
        GLMOptimizer(
            ScalarAnchor(), AnchorObjective(), y0=np.array([0.25]), plan=plan
        )


def test_the_objective_value_needs_a_packed_evaluation_on_a_mixed_plan():
    """C-7, C-18.5: opt in to the packed layout or be refused.

    ``Objective.evaluate`` takes ``(N, s, ν)``. On a plan whose stage count
    varies there is no such array, and handing the objective the packed one
    would not raise: it would read stage ``(n, k)`` from whatever step lies at
    packed row ``n`` and return a plausible number. So an objective that has
    not said it can read the packed layout is refused, and one that has is
    used.

    The derivatives were never affected: their objective callbacks are given
    one stage and its ``(step, stage)`` index.
    """
    plan, problem, objective, y0, u = _case(MIXED_PLANS[0].values[0])

    rectangular_only = _WithoutPackedEvaluation(objective)
    with pytest.raises(NotImplementedError, match="evaluate_packed"):
        GLMOptimizer(
            problem, rectangular_only, y0=y0, plan=plan
        ).objective_value(u)

    optimizer = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    value = optimizer.objective_value(u)

    # Independent sum: the control term accumulated over the packed rows
    # directly, with the state terms taken from the trajectory.
    trajectory = optimizer.trajectory(u)
    Y = trajectory.Y
    d = Y[-1, 0] - objective.y_target
    expected = 0.5 * float(d @ objective.Q_T @ d)
    for n in range(plan.N):
        expected += 0.5 * float(Y[n, 0] @ objective.Q @ Y[n, 0])
    for row in u:
        expected += 0.5 * float(row @ objective.R @ row)
    # Exact: both routes add the same products in the same order.
    assert value == expected

    assert optimizer.gradient(u).shape == u.shape


def test_packed_and_rectangular_evaluation_agree_on_a_uniform_plan():
    """The degenerate plan is where the two routes can be compared at all.

    ``evaluate`` and ``evaluate_packed`` describe one function. A uniform
    plan has both layouts, so an objective offering both must return the same
    number; that is what makes the mixed-plan value above trustworthy, since
    on a mixed plan only one of them can be called.
    """
    plan = DiscretizationPlan.uniform((0.3, 1.1), 5, sdirk3())
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    u = packed_controls(plan, problem.control_dim, seed=3)
    optimizer = GLMOptimizer(
        problem, objective, y0=np.array([0.4, -0.25, 0.15]), plan=plan
    )
    trajectory = optimizer.trajectory(u)

    rectangular = objective.evaluate(
        trajectory, u.reshape(plan.N, plan.methods[0].s, problem.control_dim)
    )
    packed = objective.evaluate_packed(trajectory, u)
    # Exact: the same products, accumulated in the same order.
    assert packed == rectangular


# ---------------------------------------------------------------------------
# C-10 control coordinates on a plan (C-18.5)
# ---------------------------------------------------------------------------


def test_nodal_control_samples_each_step_at_its_own_abscissae():
    """C-10.4, C-18.4: the interpolant and the solver must agree on *when*.

    Closed form. ``y' = u``, ``y(0) = 0``, ``J = y(1)²/2``, nodes ``(0, ½, 1)``,
    explicit Euler then implicit midpoint, both ``s = 1``, and nodal values
    ``θ = (0, ½, 1)``. Step 0 samples at ``c = 0``, giving ``u = θ₀ = 0``; step
    1 samples at ``c = ½``, giving ``u = ½θ₁ + ½θ₂ = ¾``. Both methods have
    ``b = [1]``, so ``y(1) = ½·0 + ½·¾ = ⅜`` and ``J = 9/128 = 0.0703125``.

    Sampling both steps at Euler's ``c = 0`` -- which is what one abscissa
    vector for the whole plan gives, at exactly the right shape -- yields
    ``(0, ½)``, ``y(1) = ¼`` and ``J = 0.03125``. Nothing about the shape
    distinguishes the two.
    """
    plan = DiscretizationPlan(
        nodes=np.array([0.0, 0.5, 1.0]),
        methods=(explicit_euler(), implicit_midpoint()),
    )
    optimizer = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.zeros((1, 1)), plan=plan
    )
    theta = np.array([[0.0], [0.5], [1.0]])

    good = NodalControl.from_plan(plan, control_dim=1)
    assert good.expand(theta).ravel() == pytest.approx([0.0, 0.75], abs=0.0)

    fun, jac = optimizer.scipy_interface(parametrization=good)
    assert fun(theta.ravel()) == pytest.approx(9.0 / 128.0, rel=1e-15)
    # dJ/dθ = y(1) · (½, ¼, ¼) by the chain rule through the two steps.
    assert jac(theta.ravel()) == pytest.approx(
        0.375 * np.array([0.5, 0.25, 0.25]), rel=1e-14
    )


def test_a_parametrization_sampling_the_wrong_abscissae_is_refused():
    """C-7: the mismatch above must not be reachable in silence.

    ``explicit_euler`` and ``implicit_midpoint`` both have ``s = 1``, so the
    map built against either produces an array of exactly the right shape.
    Only the abscissae distinguish them, so only comparing the abscissae
    refuses it.
    """
    plan = DiscretizationPlan(
        nodes=np.array([0.0, 0.5, 1.0]),
        methods=(explicit_euler(), implicit_midpoint()),
    )
    optimizer = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.zeros((1, 1)), plan=plan
    )
    wrong = NodalControl(n_steps=2, control_dim=1, c=plan.methods[0].c)
    assert wrong.stage_shape == (2, 1, 1)  # the shape check cannot see it

    with pytest.raises(ValueError, match="abscissae"):
        optimizer.scipy_interface(parametrization=wrong)
    with pytest.raises(ValueError, match="abscissae"):
        optimizer.scipy_hessp(parametrization=wrong)


def test_a_piecewise_constant_map_needs_no_abscissae():
    """It declares ``None`` rather than reporting abscissae it does not use.

    ``u[n, i] = θ[n]`` whatever ``c`` is, so there is nothing for a plan to
    disagree with, and the map is usable on a plan that changes method at
    every step.
    """
    plan = mixed_plan((0.3, 1.1), MIXED_PLANS[0].values[0], ratio=4.0)
    problem = CoupledNonlinear()
    param = PiecewiseConstantControl(
        plan.N, [m.s for m in plan.methods], problem.control_dim
    )
    assert param.stage_abscissae is None
    assert param.packed_shape == (plan.total_stages, problem.control_dim)

    optimizer = GLMOptimizer(
        problem, FullCostObjective(nx=3, nu=2),
        y0=np.array([0.4, -0.25, 0.15]), plan=plan,
    )
    theta = np.linspace(-0.4, 0.6, plan.N * problem.control_dim)
    fun, jac = optimizer.scipy_interface(parametrization=param)

    # The pullback is the transpose of the expansion, in the flat Euclidean
    # product C-10.4 specifies -- checked here against a directional
    # difference of the objective, which shares no code with pullback.
    g = jac(theta)
    rng = np.random.default_rng(5)
    v = rng.standard_normal(theta.size)
    eps = 1e-6
    fd = (fun(theta + eps * v) - fun(theta - eps * v)) / (2.0 * eps)
    # Central differences on a smooth J: O(eps^2) truncation plus O(u_r/eps)
    # rounding, so ~1e-9 at eps = 1e-6. 1e-7 is a margin over that.
    assert g @ v == pytest.approx(fd, rel=1e-7)


def test_a_mixed_plan_optimizes_through_the_scipy_adapters():
    """C-18.5: the packed C-10 layout is the deliverable, not a refusal.

    A plan mixing stage counts reaches a scalar value, a gradient and a
    descent step. Without the packed layout the parametrization was refused
    outright, so nothing below could run.
    """
    plan = mixed_plan((0.3, 1.1), MIXED_PLANS[0].values[0], ratio=4.0)
    problem = CoupledNonlinear()
    assert len({m.s for m in plan.methods}) > 1

    optimizer = GLMOptimizer(
        problem, FullCostObjective(nx=3, nu=2),
        y0=np.array([0.4, -0.25, 0.15]), plan=plan,
    )
    param = PiecewiseConstantControl(
        plan.N, [m.s for m in plan.methods], problem.control_dim
    )
    fun, jac = optimizer.scipy_interface(parametrization=param)
    hessp = optimizer.scipy_hessp(parametrization=param)

    theta = np.full(plan.N * problem.control_dim, 0.3)
    g = jac(theta)
    assert np.linalg.norm(g) > 0.0
    assert hessp(theta, g).shape == g.shape

    result = scipy.optimize.minimize(
        fun, theta, jac=jac, hessp=hessp, method="trust-ncg",
        options={"gtol": 1e-10, "maxiter": 200},
    )
    assert result.fun < fun(theta)
    assert np.linalg.norm(jac(result.x)) < 1e-8


# ---------------------------------------------------------------------------
# The plan owns its tableaux (C-18.2)
# ---------------------------------------------------------------------------


def test_writing_through_the_callers_tableau_does_not_reach_the_plan():
    """C-18.2, and the C-15.7 aliasing class reached through the method.

    ``GLMethod`` is a frozen dataclass, so ``m.B = ...`` is refused; its
    fields are ordinary writable arrays, so ``m.B[0, 0] = 2.0`` is not. That
    write would change the discretization a trajectory was already recorded
    against. Measured before the plan took a snapshot, on ``y' = u``,
    ``J = y(1)²/2`` with two Euler steps and stage controls ``(1, 3)``:
    ``J = 2.0`` became ``6.125``.
    """
    method = explicit_euler()
    plan = DiscretizationPlan(
        nodes=np.array([0.0, 0.5, 1.0]), methods=(method, method)
    )
    u = np.array([[1.0], [3.0]])
    optimizer = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.zeros((1, 1)), plan=plan
    )
    before = optimizer.objective_value(u)
    assert before == pytest.approx(2.0, rel=1e-15)

    method.B[0, 0] = 2.0
    method.c[0] = 0.75
    assert plan.methods[0].B[0, 0] == 1.0
    assert plan.methods[0].c[0] == 0.0

    after = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.zeros((1, 1)), plan=plan
    ).objective_value(u)
    assert after == before

    for name in ("A", "U", "B", "V", "c"):
        assert not getattr(plan.methods[0], name).flags.writeable


def test_the_snapshot_keeps_one_method_object_shared_across_steps():
    """The snapshot must not cost the C-15.1 factorization reuse.

    ``GLMOptimizer`` keys its stage solvers, and therefore its factorization
    stores, on ``id(method)``. Copying each step's tableau separately would
    give every step its own solver and silently refactorize once per step --
    the same 5-factorizations-over-12-steps failure the explicit
    ``step_sizes`` were introduced to prevent, reached a different way.
    """
    method = implicit_midpoint()
    N = 6
    plan = DiscretizationPlan.uniform((0.3, 1.1), N, method)
    assert len({id(m) for m in plan.methods}) == 1

    problem = ConstantJacobianQuadraticControl()
    optimizer = GLMOptimizer(
        problem, FullCostObjective(nx=3, nu=2),
        y0=np.array([0.4, -0.25, 0.15]), plan=plan,
        problem_structure=CONSTANT_JACOBIAN,
    )
    optimizer.gradient(packed_controls(plan, problem.control_dim, seed=4))
    assert optimizer.stage_solver.factorizations.factorizations == 1


def test_a_map_whose_stages_are_distributed_differently_is_refused():
    """C-7: equal totals are the dangerous case, not the safe one.

    A plan with stage counts ``(1, 4)`` and a map producing ``(4, 1)`` agree
    on ``N``, on ``ν`` and on the total number of stage controls, so every
    size in sight matches and the packed array reshapes without complaint.
    The blocks are then read against the wrong steps: step 0's method gets
    four controls of which it uses one, and step 1's gets one where it needs
    four.

    ``PiecewiseConstantControl`` reports no abscissae, so the stage-count
    comparison is the only thing that can refuse this.
    """
    plan = DiscretizationPlan(
        nodes=np.array([0.3, 0.7, 1.1]), methods=(explicit_euler(), rk4())
    )
    problem = CoupledNonlinear()
    optimizer = GLMOptimizer(
        problem, FullCostObjective(nx=3, nu=2),
        y0=np.array([0.4, -0.25, 0.15]), plan=plan,
    )

    wrong = PiecewiseConstantControl(plan.N, [4, 1], problem.control_dim)
    assert wrong.stage_abscissae is None
    assert wrong.n_stage_controls == plan.total_stages * problem.control_dim
    assert wrong.n_steps == plan.N and wrong.control_dim == problem.control_dim

    with pytest.raises(ValueError, match="stage counts"):
        optimizer.scipy_interface(parametrization=wrong)
    with pytest.raises(ValueError, match="stage counts"):
        optimizer.scipy_hessp(parametrization=wrong)
