"""Oracle tests for the discrete Hessian (units U-M1.3, U-O.1).

Clause C-2 requires ``hessian_vector_product`` to return the **exact** action
of the Hessian of the discrete objective that ``objective_value`` evaluates,
at a fixed mesh. These tests compare the operator, column by column, against
the dense reduced Hessian assembled independently in
:mod:`adjungo.validation.reference`.

Precedent R-3 is the reason symmetry is checked but never relied on: this
repository previously held a Hessian-vector product that was symmetric to
``8.7e-19`` while being wrong by ``3.8e-3``. A consistently wrong symmetric
operator is still symmetric.
"""

from __future__ import annotations

import numpy as np
import pytest

from adjungo.methods.runge_kutta import explicit_euler, heun, rk4
from adjungo.optimization.interface import GLMOptimizer
from adjungo.validation import reference_hessian
from tests.problems import (
    AnchorObjective,
    CoupledNonlinear,
    FullCostObjective,
    ScalarAnchor,
    make_controls,
)

CERTIFIED_RTOL = 1e-11

METHODS = [
    pytest.param(explicit_euler, id="explicit_euler_s1"),
    pytest.param(heun, id="heun_s2"),
    pytest.param(rk4, id="rk4_s4"),
]


def _dense_operator(optimizer: GLMOptimizer, u: np.ndarray) -> np.ndarray:
    """Materialize the Hessian operator by applying it to each basis vector."""
    ncol = u.size
    cols = []
    for j in range(ncol):
        v = np.zeros(ncol)
        v[j] = 1.0
        cols.append(optimizer.hessian_vector_product(u, v.reshape(u.shape)).ravel())
    return np.array(cols).T


def _case(method_factory, N=4, seed=3):
    method = method_factory()
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    t_span = (0.3, 1.1)
    u = make_controls(N, method.s, problem.control_dim, seed=seed)
    return problem, objective, method, t_span, N, y0, u


def test_package_hvp_matches_closed_form_anchor():
    """``H v = h^2 v`` for ``y_1 = y_0 + h u``, ``J = y_1^2 / 2`` (C-14.2)."""
    h = 0.37
    y0 = np.array([0.25])
    u = np.array([[[1.3]]])
    v = np.array([[[1.0]]])
    opt = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), explicit_euler(), (0.0, h), 1, y0
    )
    assert opt.hessian_vector_product(u, v)[0, 0, 0] == pytest.approx(
        h * h, rel=1e-13, abs=1e-15
    )


@pytest.mark.parametrize("method_factory", METHODS)
def test_hvp_matches_independent_reference(method_factory):
    """C-2: the second-order adjoint is the exact discrete Hessian action.

    Compared against a dense reduced Hessian built from the monolithic
    residual and its Lagrangian, which shares no code with
    ``adjungo/stepping/``.
    """
    problem, objective, method, t_span, N, y0, u = _case(method_factory)
    H_ref = reference_hessian(y0, u, t_span, N, problem, method, objective)
    H_pkg = _dense_operator(
        GLMOptimizer(problem, objective, method, t_span, N, y0), u
    )

    assert H_ref.shape == H_pkg.shape == (u.size, u.size)
    assert np.max(np.abs(H_ref)) > 1e-3, "degenerate case: Hessian is ~0"
    scale = max(float(np.max(np.abs(H_ref))), 1.0)
    err = float(np.max(np.abs(H_pkg - H_ref))) / scale
    assert err < CERTIFIED_RTOL, (
        f"Hessian operator differs from the independent reference by "
        f"{err:.6e} (relative, C-3 basis); certified tolerance "
        f"{CERTIFIED_RTOL:.1e}."
    )


@pytest.mark.parametrize("method_factory", METHODS)
def test_hvp_is_symmetric(method_factory):
    """Corroboration only (precedent R-3): symmetry is necessary, not sufficient."""
    problem, objective, method, t_span, N, y0, u = _case(method_factory)
    H = _dense_operator(
        GLMOptimizer(problem, objective, method, t_span, N, y0), u
    )
    scale = max(float(np.max(np.abs(H))), 1.0)
    assert float(np.max(np.abs(H - H.T))) / scale < 1e-12


@pytest.mark.parametrize("method_factory", METHODS)
def test_hvp_finite_difference_sweep(method_factory):
    """Central differences of the exact gradient, reported as an ε-sweep (C-3.3)."""
    problem, objective, method, t_span, N, y0, u = _case(method_factory, N=3)
    opt = GLMOptimizer(problem, objective, method, t_span, N, y0)

    rng = np.random.default_rng(21)
    v = rng.standard_normal(u.shape)
    v /= np.linalg.norm(v)
    Hv = opt.hessian_vector_product(u, v)
    scale = max(float(np.max(np.abs(Hv))), 1.0)
    assert float(np.max(np.abs(Hv))) > 1e-6, "degenerate direction"

    sweep = {}
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        fd = (opt.gradient(u + eps * v) - opt.gradient(u - eps * v)) / (2 * eps)
        sweep[eps] = float(np.max(np.abs(Hv - fd))) / scale

    best = min(sweep, key=lambda e: sweep[e])
    detail = ", ".join(f"eps={e:.0e}: {r:.3e}" for e, r in sweep.items())
    assert sweep[best] < 1e-7, (
        f"central-difference sweep never approached the second-order adjoint; "
        f"best relative agreement {sweep[best]:.3e} at eps={best:.0e}. "
        f"Full sweep: {detail}. An eps-independent plateau indicates a wrong "
        f"operator, not round-off."
    )


def test_hvp_refuses_without_objective_second_derivatives():
    """C-7: a missing curvature callback must refuse, not silently drop a term.

    Dropping ``d2J_dy2_terminal`` would return a Gauss-Newton-like operator
    while the public method still promises the exact Hessian.
    """

    class NoCurvatureObjective(FullCostObjective):
        d2J_dy2_terminal = None  # type: ignore[assignment]

    problem, _, method, t_span, N, y0, u = _case(heun)
    opt = GLMOptimizer(
        problem, NoCurvatureObjective(3, 2), method, t_span, N, y0
    )
    with pytest.raises(NotImplementedError, match="d2J_dy2_terminal"):
        opt.hessian_vector_product(u, np.ones_like(u))


def test_hvp_refuses_without_problem_second_derivatives():
    """C-7: same refusal when the *problem* lacks a curvature callback."""

    class NoCurvatureProblem(CoupledNonlinear):
        F_uu_action = None  # type: ignore[assignment]

    _, objective, method, t_span, N, y0, u = _case(heun)
    opt = GLMOptimizer(
        NoCurvatureProblem(), objective, method, t_span, N, y0
    )
    with pytest.raises(NotImplementedError, match="F_uu_action"):
        opt.hessian_vector_product(u, np.ones_like(u))
