"""Affine control parametrizations: the C-10 adapter layer.

The claim under test is C-10.2: for an affine map ``u = P θ + q``,

    g_θ    = Pᵀ g_u
    H_θ v  = Pᵀ H_u (P v)

A transposition error here is the failure mode worth designing against. It
does not raise, does not produce NaN, and does not break the optimiser
outright. It returns a vector of the right shape that is simply not the
gradient, so a line search still makes progress and convergence merely
degrades. Nothing downstream would report it.

The tests are therefore chained so that no operation validates itself:

1. ``expand`` is definitional -- it produces the stage controls the solver
   integrates, so it *is* the discrete problem.
2. ``P`` is materialised from ``expand`` by differencing basis vectors.
   Because the map is affine this difference is exact, not an
   approximation, so the materialised ``P`` inherits ``expand``'s authority.
3. ``push`` is checked against that ``P``.
4. ``pullback`` is checked against ``Pᵀ`` -- that is, against ``push``,
   never against itself.
5. The end-to-end gradient and Hessian in ``θ`` are checked against the
   independently assembled monolithic reference in ``u``, composed with the
   already-validated transpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from adjungo.core.plan import DiscretizationPlan
from adjungo.methods.runge_kutta import (
    explicit_euler,
    heun,
    implicit_midpoint,
    rk4,
)
from adjungo.optimization.interface import GLMOptimizer
from adjungo.optimization.parametrization import (
    AffineControlParametrization,
    NodalControl,
    PiecewiseConstantControl,
)
from adjungo.validation import reference_gradient, reference_hessian
from tests.problems import (
    AnchorObjective,
    CoupledNonlinear,
    FullCostObjective,
    ScalarAnchor,
)

T_SPAN = (0.0, 0.4)
N_STEPS = 5
#: ``CoupledNonlinear`` is a 3-state, 2-control problem (C-14 workhorse).
Y0 = np.array([0.4, -0.25, 0.15])

#: Matches the explicit-method basis in test_oracle_gradient.py (C-3).
RTOL = 1e-11


def _maps():
    """Every shipped parametrization, over every certified explicit tableau."""
    for method_factory in (explicit_euler, heun, rk4):
        method = method_factory()
        nu = CoupledNonlinear.control_dim
        yield (
            f"piecewise-constant/{method_factory.__name__}",
            method,
            PiecewiseConstantControl(N_STEPS, method.s, nu),
        )
        yield (
            f"nodal/{method_factory.__name__}",
            method,
            NodalControl(N_STEPS, nu, method.c),
        )


MAPS = list(_maps())
MAP_IDS = [name for name, _, _ in MAPS]


def _build(method):
    problem = CoupledNonlinear()
    objective = FullCostObjective(
        nx=problem.state_dim, nu=problem.control_dim
    )
    return GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=T_SPAN,
        N=N_STEPS,
        y0=Y0,
    )


def _dense_P(param) -> np.ndarray:
    """Materialise ``P`` from ``expand`` alone.

    Column ``j`` is ``expand(e_j) - expand(0)``. For an affine map this
    difference is the linear part exactly -- there is no truncation error to
    bound, because the second and higher derivatives of an affine map are
    identically zero.

    This deliberately does not call ``push``. ``push`` is the thing being
    checked.
    """
    origin = param.expand(np.zeros(param.n_parameters)).ravel()
    columns = np.empty((param.n_stage_controls, param.n_parameters))
    basis = np.zeros(param.n_parameters)
    for j in range(param.n_parameters):
        basis[j] = 1.0
        columns[:, j] = param.expand(basis).ravel() - origin
        basis[j] = 0.0
    return columns


# ---------------------------------------------------------------------------
# The map is affine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,method,param", MAPS, ids=MAP_IDS)
def test_map_is_affine(name, method, param):
    """``expand(a + b) - expand(a) - expand(b) + expand(0) == 0``.

    C-10.2 applies only to affine maps. If a map were not affine, the
    transpose relation would still hold pointwise for the linearisation but
    the Hessian transform would be missing the C-10.3 curvature term, and
    the result would be a Gauss-Newton operator presented as an exact
    Hessian -- which C-2 forbids.
    """
    rng = np.random.default_rng(20240517)
    a = rng.standard_normal(param.n_parameters)
    b = rng.standard_normal(param.n_parameters)

    residual = (
        param.expand(a + b)
        - param.expand(a)
        - param.expand(b)
        + param.expand(np.zeros(param.n_parameters))
    )
    assert np.max(np.abs(residual)) < 1e-14


@pytest.mark.parametrize("name,method,param", MAPS, ids=MAP_IDS)
def test_push_is_the_linear_part_of_expand(name, method, param):
    """``push(v)`` must equal ``P v`` with ``P`` taken from ``expand``."""
    rng = np.random.default_rng(11)
    P = _dense_P(param)

    for _ in range(4):
        v = rng.standard_normal(param.n_parameters)
        assert np.allclose(param.push(v).ravel(), P @ v, rtol=0, atol=1e-14)


@pytest.mark.parametrize("name,method,param", MAPS, ids=MAP_IDS)
def test_pullback_is_the_exact_transpose_of_push(name, method, param):
    """``⟨P v, g⟩ == ⟨v, Pᵀ g⟩`` for random ``v`` and ``g``.

    This is the test that a transposed-index bug cannot survive. It is
    checked both as a dense matrix identity and as the bilinear identity,
    because the two fail differently: a wrong *shape* of contraction fails
    the matrix check, while a wrong *scaling* (averaging instead of summing
    over stages, say) fails both but is easiest to read off the bilinear
    form.
    """
    rng = np.random.default_rng(12)
    P = _dense_P(param)

    assert np.allclose(
        param.matrix(), P, rtol=0, atol=1e-14
    ), "matrix() disagrees with expand()"

    for _ in range(4):
        v = rng.standard_normal(param.n_parameters)
        g = rng.standard_normal(param.n_stage_controls)

        assert np.allclose(
            param.pullback(g.reshape(param.stage_shape)).ravel(),
            P.T @ g,
            rtol=0,
            atol=1e-14,
        )

        lhs = float(np.dot(param.push(v).ravel(), g))
        rhs = float(np.dot(v, param.pullback(g.reshape(param.stage_shape)).ravel()))
        assert lhs == pytest.approx(rhs, rel=1e-13, abs=1e-15)


# ---------------------------------------------------------------------------
# Derivatives in parameter space, against the independent reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,method,param", MAPS, ids=MAP_IDS)
def test_parameter_gradient_matches_the_reference(name, method, param):
    """``jac(θ)`` must equal ``Pᵀ`` applied to the *reference* ``g_u``.

    The reference (``adjungo/validation/reference.py``) assembles the whole
    discretisation as one residual and solves for the gradient densely. It
    shares no code with the stepping or adjoint implementations, so
    agreement here is not two copies of one mistake.
    """
    optimizer = _build(method)
    rng = np.random.default_rng(13)
    theta = 0.3 * rng.standard_normal(param.n_parameters)

    _, jac = optimizer.scipy_interface(parametrization=param)
    g_theta = jac(theta)

    u = param.expand(theta)
    g_u_ref = reference_gradient(
        Y0, u, T_SPAN, N_STEPS, optimizer.problem, method, optimizer.objective
    )
    expected = _dense_P(param).T @ np.asarray(g_u_ref).ravel()

    scale = max(float(np.max(np.abs(expected))), 1.0)
    error = float(np.max(np.abs(g_theta - expected))) / scale
    assert error < RTOL, f"{name}: relative gradient error {error:.3e}"


@pytest.mark.parametrize("name,method,param", MAPS, ids=MAP_IDS)
def test_parameter_hessp_matches_the_reference(name, method, param):
    """``hessp(θ, v)`` must equal ``Pᵀ H_u (P v)`` with ``H_u`` the reference."""
    optimizer = _build(method)
    rng = np.random.default_rng(14)
    theta = 0.3 * rng.standard_normal(param.n_parameters)
    v = rng.standard_normal(param.n_parameters)

    hessp = optimizer.scipy_hessp(parametrization=param)
    hv_theta = hessp(theta, v)

    u = param.expand(theta)
    P = _dense_P(param)
    H_u = reference_hessian(
        Y0, u, T_SPAN, N_STEPS, optimizer.problem, method, optimizer.objective
    )
    expected = P.T @ (np.asarray(H_u) @ (P @ v))

    scale = max(float(np.max(np.abs(expected))), 1.0)
    error = float(np.max(np.abs(hv_theta - expected))) / scale
    assert error < RTOL, f"{name}: relative HVP error {error:.3e}"


@pytest.mark.parametrize("name,method,param", MAPS, ids=MAP_IDS)
def test_parameter_hessian_is_symmetric(name, method, param):
    """Corroboration only, never primary evidence.

    ``Pᵀ H P`` is symmetric whenever ``H`` is, so this cannot detect a
    wrong-but-symmetric ``H``. This repository has held a Hessian symmetric
    to 8.7e-19 while wrong by 3.8e-3 in value (precedent R-6). It is
    retained because it *can* detect an asymmetric mistake introduced by the
    adapter itself, such as pushing with one map and pulling back with
    another.
    """
    optimizer = _build(method)
    rng = np.random.default_rng(15)
    theta = 0.3 * rng.standard_normal(param.n_parameters)

    hessp = optimizer.scipy_hessp(parametrization=param)
    n = param.n_parameters
    H = np.column_stack([hessp(theta, np.eye(n)[:, j]) for j in range(n)])

    asymmetry = float(np.max(np.abs(H - H.T)))
    assert asymmetry < 1e-11 * max(float(np.max(np.abs(H))), 1.0)


# ---------------------------------------------------------------------------
# Specific structure of each map
# ---------------------------------------------------------------------------


def test_piecewise_constant_holds_the_value_across_stages():
    """Every stage of a step must receive the same value."""
    param = PiecewiseConstantControl(N_STEPS, 3, 2)
    rng = np.random.default_rng(16)
    theta = rng.standard_normal((N_STEPS, 2))

    u = param.expand(theta)
    assert u.shape == (N_STEPS, 3, 2)
    for i in range(3):
        assert np.array_equal(u[:, i, :], theta)


def test_piecewise_constant_pullback_sums_rather_than_averages():
    """The transpose must sum over stages.

    Averaging is the plausible wrong choice, because the forward map
    *broadcasts* and broadcasting feels like a mean. It would scale the
    gradient by ``1/s`` -- an error that leaves the descent direction
    correct and only changes its length, so a line-searching optimiser would
    still converge and hide it.
    """
    s = 4
    param = PiecewiseConstantControl(2, s, 1)
    g = np.ones((2, s, 1))

    assert np.array_equal(param.pullback(g), np.full((2, 1), float(s)))


def test_nodal_interpolates_between_the_bracketing_nodes():
    """Stage values must be the linear interpolant at the tableau's ``c``."""
    method = rk4()
    param = NodalControl(N_STEPS, 1, method.c)
    theta = np.arange(N_STEPS + 1, dtype=float).reshape(-1, 1)

    u = param.expand(theta)
    for n in range(N_STEPS):
        for i, c_i in enumerate(method.c):
            expected = (1.0 - c_i) * theta[n, 0] + c_i * theta[n + 1, 0]
            assert u[n, i, 0] == pytest.approx(expected, rel=0, abs=1e-15)


def test_nodal_reproduces_a_constant_exactly():
    """Partition of unity: interpolation weights must sum to one.

    ``(1 - c) + c == 1`` for every abscissa, so a constant node vector must
    give a constant control. This fails if an abscissa is dropped or an
    index is shifted by one step.
    """
    for factory in (explicit_euler, heun, rk4):
        method = factory()
        param = NodalControl(N_STEPS, 2, method.c)
        theta = np.tile(np.array([2.5, -1.25]), (N_STEPS + 1, 1))

        u = param.expand(theta)
        assert np.allclose(u, np.array([2.5, -1.25]), rtol=0, atol=1e-15)


def test_nodal_uses_the_abscissae_it_was_given():
    """Changing ``c`` alone must change the map.

    This pins the C-10 requirement that the interpolation weights and the
    stage times the solver evaluates come from the same tableau. A
    ``NodalControl`` that ignored ``c`` -- for instance by treating every
    stage as the step's left endpoint -- would still satisfy the
    partition-of-unity test, the transpose test and the affinity test. Only
    a comparison against a perturbed ``c`` detects it.

    The two abscissa vectors have the same length, so the comparison is
    between genuinely different maps of the same shape rather than between
    arrays that cannot be compared at all.
    """
    c_true = rk4().c
    c_shifted = c_true + 0.25
    assert c_true.shape == c_shifted.shape

    theta = np.linspace(0.0, 1.0, N_STEPS + 1).reshape(-1, 1)
    u_true = NodalControl(N_STEPS, 1, c_true).expand(theta)
    u_shifted = NodalControl(N_STEPS, 1, c_shifted).expand(theta)

    assert u_true.shape == u_shifted.shape
    assert not np.allclose(u_true, u_shifted)

    # And the left-endpoint degeneracy specifically.
    u_zero_c = NodalControl(N_STEPS, 1, np.zeros_like(c_true)).expand(theta)
    assert not np.allclose(u_true, u_zero_c)


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


def test_optimizer_refuses_a_mismatched_parametrization():
    """A map built for a different discretisation must be refused.

    Sizes that merely happen to coincide are the dangerous case: a silent
    reshape would permute controls across steps and stages and still return
    finite numbers.
    """
    optimizer = _build(rk4())
    wrong = PiecewiseConstantControl(
        N_STEPS + 1, rk4().s, CoupledNonlinear.control_dim
    )

    with pytest.raises(ValueError, match="stage controls"):
        optimizer.scipy_interface(parametrization=wrong)
    with pytest.raises(ValueError, match="stage controls"):
        optimizer.scipy_hessp(parametrization=wrong)


def test_parametrization_rejects_a_wrongly_sized_parameter_vector():
    param = PiecewiseConstantControl(N_STEPS, 2, 1)
    with pytest.raises(ValueError, match="expects"):
        param.expand(np.zeros(param.n_parameters + 1))


def test_parametrization_rejects_a_wrongly_sized_covector():
    param = PiecewiseConstantControl(N_STEPS, 2, 1)
    with pytest.raises(ValueError, match="expects"):
        param.pullback(np.zeros(param.n_stage_controls + 1))


def test_nodal_rejects_nonfinite_abscissae():
    with pytest.raises(ValueError, match="finite"):
        NodalControl(N_STEPS, 1, np.array([0.0, np.nan]))


@pytest.mark.parametrize("bad", [0, -1])
def test_parametrization_rejects_nonpositive_dimensions(bad):
    with pytest.raises(ValueError):
        PiecewiseConstantControl(bad, 2, 1)
    with pytest.raises(ValueError):
        PiecewiseConstantControl(N_STEPS, bad, 1)
    with pytest.raises(ValueError):
        PiecewiseConstantControl(N_STEPS, 2, bad)


# ---------------------------------------------------------------------------
# The default path is unchanged
# ---------------------------------------------------------------------------


def test_no_parametrization_is_the_identity():
    """Omitting the argument must reproduce the stage-control interface.

    The adapter layer is optional, and adding it must not perturb the
    certified default path.
    """
    optimizer = _build(rk4())
    rng = np.random.default_rng(17)
    u = 0.2 * rng.standard_normal(
        (N_STEPS, rk4().s, CoupledNonlinear.control_dim)
    )

    fun, jac = optimizer.scipy_interface()
    assert fun(u.ravel()) == pytest.approx(optimizer.objective_value(u))
    assert np.array_equal(jac(u.ravel()), optimizer.gradient(u).ravel())

    v = rng.standard_normal(u.size)
    hessp = optimizer.scipy_hessp()
    assert np.array_equal(
        hessp(u.ravel(), v),
        optimizer.hessian_vector_product(u, v.reshape(u.shape)).ravel(),
    )


# ---------------------------------------------------------------------------
# The affine offset
# ---------------------------------------------------------------------------


class ShiftedControl(AffineControlParametrization):
    """``u[n, i, :] = θ[n, :] + 1``: the smallest map with ``q ≠ 0``.

    Both shipped parametrizations have ``q = 0``, so every test above holds
    equally well for a map that confuses ``P θ + q`` with ``P θ``. This one
    does not. It is deliberately minimal — the offset is the only thing that
    distinguishes it from :class:`PiecewiseConstantControl`.
    """

    def __init__(self, n_steps: int, stage_counts, control_dim: int) -> None:
        super().__init__(n_steps, stage_counts, control_dim)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (self.n_steps, self.control_dim)

    def _expand_packed(self, theta):
        arr = self._check_parameter_shape(theta)
        return np.repeat(arr, self.stage_counts, axis=0) + 1.0

    def _push_packed(self, v):
        return np.repeat(self._check_parameter_shape(v), self.stage_counts, 0)

    def _pullback_packed(self, g):
        return np.add.reduceat(g, self.stage_offsets[:-1], axis=0)


def test_the_offset_moves_the_point_and_not_the_direction():
    """``push`` is ``P v``, never ``P v + q``.

    ``expand`` and ``push`` differ by exactly ``q`` for this map, which is
    the whole content of C-10.2's distinction between the two.
    """
    par = ShiftedControl(3, [2, 2, 2], 2)
    v = np.arange(6, dtype=float).reshape(3, 2)

    assert np.allclose(par.push(v), par.expand(v) - 1.0)
    # P is the difference of expand at two points: the offset cancels.
    assert np.allclose(par.push(v), par.expand(v) - par.expand(np.zeros((3, 2))))
    assert np.allclose(par.matrix() @ v.ravel(), par.push(v).ravel())


def test_an_offset_map_returns_the_exact_hessian_on_the_scalar_anchor():
    """Closed form, so the failure is not a matter of tolerance.

    On ``y' = u``, ``y(0) = 0``, one explicit Euler step of length one and
    ``J = y(1)²/2``, the objective in ``θ`` is ``(θ + 1)²/2``. Its gradient
    is ``θ + 1`` and its Hessian is the identity, whatever ``θ`` is.

    A ``push`` that carried the offset would return ``v + 1`` where ``P v``
    is wanted, giving ``H·0 = 1`` and ``H·2 = 3`` instead of ``0`` and
    ``2``. The gradient is unaffected — only the curvature path pushes a
    direction — so nothing else in this file could see it.
    """
    plan = DiscretizationPlan.uniform((0.0, 1.0), 1, explicit_euler())
    optimizer = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.zeros((1, 1)), plan=plan
    )
    par = ShiftedControl(1, [1], 1)
    fun, jac = optimizer.scipy_interface(parametrization=par)
    hessp = optimizer.scipy_hessp(parametrization=par)

    for theta in (0.0, -1.0, 0.75):
        th = np.array([theta])
        assert fun(th) == pytest.approx(0.5 * (theta + 1.0) ** 2, rel=1e-14)
        assert jac(th)[0] == pytest.approx(theta + 1.0, rel=1e-14, abs=1e-15)
        for v in (0.0, 2.0, -1.5):
            assert hessp(th, np.array([v]))[0] == pytest.approx(
                v, rel=1e-14, abs=1e-15
            )


def test_a_map_must_declare_its_linear_part():
    """No default: the wrong one is silent, and only on offset maps.

    A subclass that supplies ``_expand_packed`` and ``_pullback_packed`` but
    not ``_push_packed`` must not be constructible, rather than inheriting
    ``expand`` and being right only by accident of ``q = 0``.
    """

    class NoPush(AffineControlParametrization):
        @property
        def parameter_shape(self) -> tuple[int, ...]:
            return (self.n_steps, self.control_dim)

        def _expand_packed(self, theta):
            arr = self._check_parameter_shape(theta)
            return np.repeat(arr, self.stage_counts, axis=0) + 1.0

        def _pullback_packed(self, g):
            return np.add.reduceat(g, self.stage_offsets[:-1], axis=0)

    with pytest.raises(TypeError, match="_push_packed"):
        NoPush(1, [1], 1)


# ---------------------------------------------------------------------------
# Abscissa ownership
# ---------------------------------------------------------------------------


def test_the_callers_abscissa_array_is_copied_not_aliased():
    """An ordinary write to the caller's own array must not reach the map.

    ``np.asarray(c).ravel()`` may return a view; freezing a view leaves its
    owner writable. The interpolation weights are built once by a copying
    ``np.stack``, so an aliased declaration and the weights can drift apart
    — and the optimizer's abscissa check reads the declaration, so it would
    accept the result.
    """
    c = np.array([0.0])
    par = NodalControl(n_steps=1, control_dim=1, c=c)

    c[0] = 0.5
    assert par.stage_abscissae[0][0] == 0.0
    assert np.array_equal(par.expand(np.array([[0.0], [1.0]])).ravel(), [0.0])

    # The declaration and the weights still agree, so the gate can do its job.
    plan = DiscretizationPlan.uniform((0.0, 1.0), 1, implicit_midpoint())
    optimizer = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), y0=np.zeros((1, 1)), plan=plan
    )
    with pytest.raises(ValueError, match="abscissae"):
        optimizer.scipy_interface(parametrization=par)

    # And the map the plan does declare gives the closed-form value:
    # u = ½·0 + ½·1, J = u²/2.
    fun, _ = optimizer.scipy_interface(
        parametrization=NodalControl.from_plan(plan, 1)
    )
    assert fun(np.array([0.0, 1.0])) == pytest.approx(0.125, rel=1e-14)


def test_the_plans_own_abscissae_are_not_captured_by_reference_either():
    """``from_plan`` must not give the map a window into the plan."""
    plan = DiscretizationPlan.uniform((0.0, 1.0), 2, implicit_midpoint())
    par = NodalControl.from_plan(plan, 1)
    for row in par.stage_abscissae:
        assert not row.flags.writeable
        assert not np.shares_memory(row, plan.methods[0].c)
