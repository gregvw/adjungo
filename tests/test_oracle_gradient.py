"""Oracle tests for the discrete gradient (units U-O.1 .. U-O.4, U-M1.1).

These tests implement the clause C-14.1 oracle hierarchy:

1. **Closed-form anchor** (C-14.2). ``y_1 = y_0 + h u``, ``J = y_1^2 / 2``
   has ``dJ/du = h y_1`` and ``d2J/du2 = h^2`` exactly. No reference
   implementation participates, so this level cannot be fooled by a shared
   mistake.
2. **Independent monolithic reference** (:mod:`adjungo.validation.reference`).
   The whole discrete system is assembled as one algebraic residual
   ``R(w, u) = 0`` and differentiated densely. It shares no code with
   ``adjungo/stepping/`` or ``adjungo/solvers/``.
3. **Finite differences of the discrete objective.** Weakest level, used only
   to corroborate, and always reported as a sweep over the step size so that
   the observed plateau is visible rather than a single lucky point.

The level-2 tests are the acceptance tests for finding B0 (an incorrect stage
Jacobian index in the adjoint stage recursion). They are constructed so that
they *fail* against the pre-fix code: :func:`test_oracle_detects_the_b0_stage_index_defect`
re-installs the old recursion and asserts a large disagreement, which keeps
the rest of the module from silently degenerating into a tautology.
"""

from __future__ import annotations

import numpy as np
import pytest

from adjungo.methods.runge_kutta import explicit_euler, heun, rk4
from adjungo.optimization.interface import GLMOptimizer
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.stepping.forward import forward_solve
from adjungo.validation import (
    reference_gradient,
    reference_hessian,
    reference_solve,
)
from tests.problems import (
    AnchorObjective,
    CoupledNonlinear,
    FullCostObjective,
    LinearTimeVarying,
    ScalarAnchor,
    make_controls,
)

# C-3.1 certified tolerance for nonlinear problems, relative to
# max(|grad J|_inf, 1). The floor is set by the dense linear solves inside the
# reference, not by the method: both sides evaluate the same discrete map.
CERTIFIED_RTOL = 1e-11

EXPLICIT_METHODS = [
    pytest.param(explicit_euler, id="explicit_euler_s1"),
    pytest.param(heun, id="heun_s2"),
    pytest.param(rk4, id="rk4_s4"),
]


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Error relative to ``max(|b|_inf, 1)`` (clause C-3)."""
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1.0))


def _make_case(method_factory, N=6, seed=0):
    """A C-14 population instance: nonlinear, n_x=3, nu=2, t0 != 0."""
    method = method_factory()
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    t_span = (0.3, 1.1)
    u = make_controls(N, method.s, problem.control_dim, seed=seed)
    return problem, objective, method, t_span, N, y0, u


# ---------------------------------------------------------------------------
# Level 1: closed-form anchor (C-14.2)
# ---------------------------------------------------------------------------


def test_reference_gradient_matches_closed_form_anchor():
    """``dJ/du = h y_1`` exactly for ``y_1 = y_0 + h u``, ``J = y_1^2/2``."""
    h = 0.37
    y0 = np.array([0.25])
    u = np.array([[[1.3]]])
    grad = reference_gradient(
        y0, u, (0.0, h), 1, ScalarAnchor(), explicit_euler(), AnchorObjective()
    )
    expected = h * (y0[0] + h * u[0, 0, 0])
    assert grad.shape == (1, 1, 1)
    assert grad[0, 0, 0] == pytest.approx(expected, rel=1e-14, abs=1e-15)


def test_reference_hessian_matches_closed_form_anchor():
    """``d2J/du2 = h^2`` exactly for the same anchor."""
    h = 0.37
    y0 = np.array([0.25])
    u = np.array([[[1.3]]])
    H = reference_hessian(
        y0, u, (0.0, h), 1, ScalarAnchor(), explicit_euler(), AnchorObjective()
    )
    assert H.shape == (1, 1)
    assert H[0, 0] == pytest.approx(h * h, rel=1e-14, abs=1e-15)


def test_package_gradient_matches_closed_form_anchor():
    """The package itself must reproduce the anchor, not merely the reference."""
    h = 0.37
    y0 = np.array([0.25])
    u = np.array([[[1.3]]])
    opt = GLMOptimizer(
        ScalarAnchor(), AnchorObjective(), explicit_euler(), (0.0, h), 1, y0
    )
    expected = h * (y0[0] + h * u[0, 0, 0])
    assert opt.gradient(u)[0, 0, 0] == pytest.approx(
        expected, rel=1e-14, abs=1e-15
    )


# ---------------------------------------------------------------------------
# Level 2: the independent reference solves the same discrete system
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_factory", EXPLICIT_METHODS)
def test_reference_forward_reproduces_package_forward(method_factory):
    """The monolithic residual must define the package's discrete map.

    If the reference solved a *different* discrete system, agreement of the
    gradients would mean nothing. This test pins the premise: same stage
    times, same external-stage embedding, same output update.
    """
    problem, objective, method, t_span, N, y0, u = _make_case(method_factory)
    solver = GLMOptimizer(
        problem, objective, method, t_span, N, y0
    ).stage_solver
    traj = forward_solve(y0, u, t_span, N, problem, method, solver)
    ref = reference_solve(y0, u, t_span, N, problem, method)

    assert ref.residual_norm <= 1e-13
    assert _rel_err(ref.Y, traj.Y) < 1e-12
    assert _rel_err(ref.Z, traj.Z) < 1e-12


# ---------------------------------------------------------------------------
# Level 2: the acceptance test for B0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_factory", EXPLICIT_METHODS)
@pytest.mark.parametrize("N", [3, 6])
def test_explicit_gradient_matches_independent_reference(method_factory, N):
    """C-2: the adjoint gradient is the exact gradient of the discrete J.

    Measured at a fixed mesh against a reference that shares no code with the
    staged recursion. This is the test that finding B0 fails.
    """
    problem, objective, method, t_span, _, y0, _ = _make_case(method_factory)
    u = make_controls(N, method.s, problem.control_dim, seed=N)

    grad_pkg = GLMOptimizer(
        problem, objective, method, t_span, N, y0
    ).gradient(u)
    grad_ref = reference_gradient(
        y0, u, t_span, N, problem, method, objective
    )

    assert grad_pkg.shape == grad_ref.shape == u.shape
    assert np.max(np.abs(grad_ref)) > 1e-3, "degenerate case: gradient is ~0"
    err = _rel_err(grad_pkg, grad_ref)
    assert err < CERTIFIED_RTOL, (
        f"{method.name if hasattr(method, 'name') else method_factory.__name__}"
        f" N={N}: adjoint gradient differs from the independent monolithic "
        f"reference by {err:.6e} (relative, C-3 basis); certified tolerance is "
        f"{CERTIFIED_RTOL:.1e}."
    )


def test_linear_time_varying_gradient_matches_reference():
    """A Jacobian varying only with ``t`` still differs between stages."""
    method = rk4()
    problem = LinearTimeVarying()
    objective = FullCostObjective(nx=2, nu=2)
    y0 = np.array([0.6, -0.4])
    t_span = (0.25, 1.4)
    N = 5
    u = make_controls(N, method.s, problem.control_dim, seed=7)

    grad_pkg = GLMOptimizer(
        problem, objective, method, t_span, N, y0
    ).gradient(u)
    grad_ref = reference_gradient(
        y0, u, t_span, N, problem, method, objective
    )
    assert _rel_err(grad_pkg, grad_ref) < CERTIFIED_RTOL


# ---------------------------------------------------------------------------
# The discriminating test: the oracle must reject the pre-fix recursion
# ---------------------------------------------------------------------------


def _legacy_adjoint_stages(self, lambda_ext, cache, method, h):
    """The pre-fix recursion, applying ``F_j`` to the stage-coupling term.

    Reproduced verbatim from the state of ``ExplicitStageSolver`` before
    finding B0 was cured, so that the oracle can be shown to reject it.
    """
    s = method.s
    n = cache.Z.shape[1]
    A, B = method.A, method.B
    mu = np.zeros((s, n))
    for i in range(s - 1, -1, -1):
        mu[i] = h * cache.F[i].T @ (B[:, i] @ lambda_ext)
        for j in range(i + 1, s):
            mu[i] += h * A[j, i] * cache.F[j].T @ mu[j]
    return mu


@pytest.mark.parametrize("method_factory", EXPLICIT_METHODS)
def test_oracle_detects_the_b0_stage_index_defect(method_factory, monkeypatch):
    """The stage-index cure must be load-bearing, and only for ``s > 1``.

    For ``s = 1`` the stage-coupling sum is empty, so the two recursions are
    identical and the oracle is *expected* to see no difference. That is
    precisely why the historical suite, which used ``explicit_euler``
    throughout, could not detect the defect. This test records both halves of
    that fact.
    """
    problem, objective, method, t_span, N, y0, u = _make_case(method_factory)
    grad_ref = reference_gradient(
        y0, u, t_span, N, problem, method, objective
    )

    monkeypatch.setattr(
        ExplicitStageSolver, "solve_adjoint_stages", _legacy_adjoint_stages
    )
    grad_legacy = GLMOptimizer(
        problem, objective, method, t_span, N, y0
    ).gradient(u)
    err = _rel_err(grad_legacy, grad_ref)

    if method.s == 1:
        assert err < CERTIFIED_RTOL, (
            "for s = 1 the two recursions are algebraically identical; a "
            "difference here means the legacy reproduction is wrong"
        )
    else:
        assert err > 1e-7, (
            f"the pre-fix recursion disagrees with the oracle by only "
            f"{err:.3e} for s={method.s}; the oracle is not discriminating "
            f"and the B0 acceptance tests are vacuous"
        )


# ---------------------------------------------------------------------------
# Level 3: finite-difference corroboration, reported as a sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_factory", EXPLICIT_METHODS)
def test_gradient_finite_difference_sweep(method_factory):
    """Central differences of the discrete objective in a random direction.

    Clause C-3.2: a single step size proves nothing, because the total error
    is ``O(eps^2) + O(eps_mach |J| / eps)`` and any one sample can sit on
    either branch. The sweep must attain a minimum well below the O(1) scale
    of the directional derivative, and the failure message names the whole
    sweep so that a plateau is diagnosable rather than mysterious.
    """
    problem, objective, method, t_span, N, y0, u = _make_case(
        method_factory, N=4
    )
    opt = GLMOptimizer(problem, objective, method, t_span, N, y0)
    grad = opt.gradient(u)

    rng = np.random.default_rng(11)
    v = rng.standard_normal(u.shape)
    v /= np.linalg.norm(v)
    exact = float(np.sum(grad * v))
    assert abs(exact) > 1e-6, "degenerate direction"

    sweep = {}
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        fd = (
            opt.objective_value(u + eps * v)
            - opt.objective_value(u - eps * v)
        ) / (2.0 * eps)
        sweep[eps] = abs(fd - exact) / abs(exact)

    best_eps = min(sweep, key=lambda e: sweep[e])
    detail = ", ".join(f"eps={e:.0e}: {r:.3e}" for e, r in sweep.items())
    assert sweep[best_eps] < 1e-8, (
        f"central-difference sweep never approached the adjoint directional "
        f"derivative {exact:.8e}; best relative agreement {sweep[best_eps]:.3e} "
        f"at eps={best_eps:.0e}. Full sweep: {detail}. A flat plateau across "
        f"the sweep indicates a wrong derivative, not round-off."
    )
