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
from adjungo.validation import reference_gradient, reference_hessian
from tests.problems import (
    AnchorObjective,
    CoupledNonlinear,
    FullCostObjective,
    ScalarAnchor,
)
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


def test_a_new_plan_at_the_same_control_agrees_with_a_fresh_evaluation():
    """C-18.7 item 4.

    Two plans with the *same* stage counts, so the identical control array is
    admissible under both and nothing but the mesh changes. Evaluating the
    second plan on an optimizer that has already solved the first must agree
    with evaluating it from scratch. A trajectory, a control sampling or a
    factorization cache that outlived the plan it was keyed to shows up here
    and nowhere else in this module.
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
    warm.objective_value(u)
    warm.gradient(u)

    # Same control parameters, new plan: a second optimizer is how a caller
    # changes a plan today, the plan being frozen by construction (C-18.2).
    moved = GLMOptimizer(problem, objective, y0=y0, plan=second)
    J_ref, g_ref, hv_ref, v = _fresh(second, problem, objective, y0, u)

    assert moved.objective_value(u) == pytest.approx(J_ref, rel=1e-14, abs=1e-15)
    assert moved.gradient(u) == pytest.approx(g_ref, rel=1e-13, abs=1e-14)
    assert moved.hessian_vector_product(u, v) == pytest.approx(
        hv_ref, rel=1e-13, abs=1e-14
    )

    # And the two plans must genuinely disagree, or the comparison above is
    # satisfied by any implementation that ignores the plan entirely.
    assert not np.allclose(g_ref, warm.gradient(u), rtol=1e-6, atol=1e-8)


def test_changing_the_plan_invalidates_a_cached_factorization():
    """C-18.7 item 4, the C-15 cache specifically.

    ``jacobian_constant`` licenses reuse of one LU factorization across every
    step (C-15.1). A cache keyed to nothing but the problem would survive a
    change of ``h``, and ``I - h a_ii F`` is a different matrix at a different
    ``h``, so the second plan would be solved with the first plan's
    factorization. The two gradients would then differ by an amount no
    structural identity detects.
    """
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    t_span = (0.3, 1.1)
    method = implicit_midpoint()
    N = 4

    coarse = DiscretizationPlan.uniform(t_span, N, method)
    fine = DiscretizationPlan.uniform(t_span, 2 * N, method)
    u_coarse = packed_controls(coarse, problem.control_dim, seed=21)
    u_fine = packed_controls(fine, problem.control_dim, seed=21)

    warm = GLMOptimizer(problem, objective, y0=y0, plan=coarse)
    warm.gradient(u_coarse)

    after = GLMOptimizer(problem, objective, y0=y0, plan=fine).gradient(u_fine)
    expected = reference_gradient(
        y0, u_fine, problem=problem, objective=objective, plan=fine
    )
    assert _rel_err(after, expected) < certified_rtol(method)


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


def test_the_objective_value_is_refused_for_a_varying_stage_count():
    """C-7: a packed control must not be read as if it were rectangular.

    ``Objective.evaluate`` takes ``(N, s, nu)``. On a plan whose stage count
    varies there is no such array, and handing the objective the packed one
    would not raise: it would read stage ``(n, k)`` from whatever step lies at
    packed row ``n`` and return a plausible number. The derivatives stay
    available, because their objective callbacks are given one stage and its
    ``(step, stage)`` index.
    """
    plan, problem, objective, y0, u = _case(MIXED_PLANS[0].values[0])
    optimizer = GLMOptimizer(problem, objective, y0=y0, plan=plan)

    with pytest.raises(NotImplementedError, match="stage count varies"):
        optimizer.objective_value(u)

    assert optimizer.gradient(u).shape == u.shape
