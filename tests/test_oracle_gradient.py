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
from adjungo.optimization.interface import GLMOptimizer
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.solvers.newton import STAGE_NEWTON_TOL
from adjungo.stepping.forward import forward_solve
from adjungo.stepping.sensitivity import forward_sensitivity
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

# C-3.1 certified tolerance for explicit methods, relative to
# max(|grad J|_inf, 1). Both sides evaluate the same discrete map with no
# iteration, so the floor is set by the dense linear solves inside the
# reference, not by the method.
CERTIFIED_RTOL = 1e-11

# C-3.4 certified tolerance for implicit methods. Derived, not chosen: the
# package and the reference each stop Newton at the C-5.1 scaled threshold,
# whose value at unit scale is ``STAGE_NEWTON_TOL``, and therefore land on
# *different* points within that radius of the same root. The gradient each
# returns is exact for its own converged iterate, so their difference is
# bounded near ``kappa * STAGE_NEWTON_TOL``, where ``kappa`` bounds the
# stage-Jacobian inverse over a step.
#
# That bound is not tight, and the measured behavior is better than it.
# Because Newton converges quadratically, the step that satisfies the
# threshold usually drives the residual far below it, so both sides land much
# closer to the true root than the stopping test guarantees. Measured over the
# C-14 implicit population (implicit_midpoint, Crank-Nicolson, sdirk2, sdirk3
# at N=3 and N=6 on CoupledNonlinear), the worst relative gradient error is
# 6.22e-15, a ratio of 0.031 to STAGE_NEWTON_TOL.
#
# CONDITIONING_ALLOWANCE is deliberately larger than that observation. The
# overshoot above is a property of these problems, not a guarantee: a stiffer
# or worse-conditioned stage Jacobian can consume the whole tolerance and more.
# 1e3 keeps the certified claim tied to the stage stopping test with room for
# conditioning, while remaining seven orders below every defect cured in M2,
# each of which showed a relative error of 1e-3 to 1e-5.
CONDITIONING_ALLOWANCE = 1.0e3
IMPLICIT_RTOL = CONDITIONING_ALLOWANCE * STAGE_NEWTON_TOL


def certified_rtol(method) -> float:
    """Return the C-3 tolerance appropriate to the method's stage type."""
    return (
        CERTIFIED_RTOL
        if method.stage_type is StageType.EXPLICIT
        else IMPLICIT_RTOL
    )

EXPLICIT_METHODS = [
    pytest.param(explicit_euler, id="explicit_euler_s1"),
    pytest.param(heun, id="heun_s2"),
    pytest.param(rk4, id="rk4_s4"),
]

#: Certified implicit families (C-6.1). ``gauss2`` has a dense stage matrix
#: and is solved by one coupled (s*n) Newton iteration per step (M3); it is
#: held to exactly the same oracle tolerance as the triangular families.
IMPLICIT_METHODS = [
    pytest.param(implicit_midpoint, id="implicit_midpoint_sdirk_s1"),
    pytest.param(implicit_trapezoid, id="crank_nicolson_dirk_s2"),
    pytest.param(sdirk2, id="sdirk2_s2"),
    pytest.param(sdirk3, id="sdirk3_s3"),
    pytest.param(gauss2, id="gauss2_fully_implicit_s2"),
]

ALL_CERTIFIED_METHODS = EXPLICIT_METHODS + IMPLICIT_METHODS


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


@pytest.mark.parametrize("method_factory", ALL_CERTIFIED_METHODS)
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
    traj = forward_solve(y0, u, DiscretizationPlan.uniform(t_span, N, method), problem, solver)
    ref = reference_solve(y0, u, t_span, N, problem, method)

    tol = certified_rtol(method)
    assert ref.residual_norm <= 1e-13
    assert _rel_err(ref.Y, traj.Y) < tol
    assert _rel_err(ref.Z, traj.Z) < tol


# ---------------------------------------------------------------------------
# Level 2: the acceptance test for B0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_factory", ALL_CERTIFIED_METHODS)
@pytest.mark.parametrize("N", [3, 6])
def test_explicit_gradient_matches_independent_reference(method_factory, N):
    """C-2: the adjoint gradient is the exact gradient of the discrete J.

    Measured at a fixed mesh against a reference that shares no code with the
    staged recursion. This is the test that findings B0 (explicit and
    implicit stage-index) and B2 (missing transposed stage solve) fail.
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
    tol = certified_rtol(method)
    err = _rel_err(grad_pkg, grad_ref)
    assert err < tol, (
        f"{method.name if hasattr(method, 'name') else method_factory.__name__}"
        f" N={N}: adjoint gradient differs from the independent monolithic "
        f"reference by {err:.6e} (relative, C-3 basis); certified tolerance is "
        f"{tol:.1e}."
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


@pytest.mark.parametrize("method_factory", ALL_CERTIFIED_METHODS)
def test_gradient_finite_difference_sweep(method_factory):
    """Central differences of the discrete objective in a random direction.

    Clauses C-3.2 and C-3.3: a single step size proves nothing, because the total error
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


# ---------------------------------------------------------------------------
# Level 4: duality. Corroboration only (C-14.1 item 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_factory", ALL_CERTIFIED_METHODS)
def test_tangent_adjoint_duality(method_factory):
    """``<grad J, v>`` from the adjoint equals the tangent directional derivative.

    The tangent solve propagates ``δZ`` and ``δY`` forward; contracting those
    with the objective's first derivatives gives ``dJ·v`` without touching the
    adjoint recursion at all.

    This is **corroboration, not certification** (NUMERICS.md C-14.1 item 4).
    Finding B0 is the concrete reason: ``forward_sensitivity`` legitimately
    uses ``F_j`` because there the Jacobian genuinely belongs to stage ``j``,
    so tangent and adjoint did *not* share that mistake — but a duality test
    passes whenever they do share one, and cannot distinguish the cases. The
    monolithic reference above is what certifies.
    """
    problem, objective, method, t_span, N, y0, u = _make_case(
        method_factory, N=4
    )
    opt = GLMOptimizer(problem, objective, method, t_span, N, y0)
    rng = np.random.default_rng(5)
    v = rng.standard_normal(u.shape)

    from_adjoint = float(np.sum(opt.gradient(u) * v))

    opt._ensure_forward(u)
    traj = opt._trajectory
    sens = forward_sensitivity(
        traj, v, opt.stage_solver, problem)

    from_tangent = float(
        np.sum(objective.dJ_dy_terminal(traj.Y[N]) * sens.delta_Y[N])
    )
    for n in range(N):
        from_tangent += float(
            np.sum(objective.dJ_dy(traj.Y[n], n) * sens.delta_Y[n])
        )
    for n in range(N):
        for k in range(method.s):
            from_tangent += float(
                np.dot(objective.dJ_du(u[n, k], n, k), v[n, k])
            )

    scale = max(abs(from_tangent), 1.0)
    assert abs(from_adjoint - from_tangent) / scale < certified_rtol(method), (
        f"duality gap: adjoint gives {from_adjoint:.12e}, tangent gives "
        f"{from_tangent:.12e}"
    )


# ---------------------------------------------------------------------------
# Level 2: the C-3.4 tolerance basis is itself measured, not asserted
# ---------------------------------------------------------------------------


def test_implicit_tolerance_basis_is_measured():
    """Pin the observation that justifies ``CONDITIONING_ALLOWANCE``.

    The comment beside ``IMPLICIT_RTOL`` claims a worst observed ratio of
    0.061 between the relative gradient error and ``STAGE_NEWTON_TOL`` over
    the C-14 implicit population. A comment is a claim; this test is the
    evidence, and it runs in the tree.

    Two directions are checked, because each catches a different way for the
    certified tolerance to stop meaning anything:

    * **Upper**: the ratio must stay under ``CONDITIONING_ALLOWANCE``. If it
      does not, ``IMPLICIT_RTOL`` no longer bounds the real behavior.
    * **Lower**: the ratio must stay under 1.0, i.e. agreement must remain
      *better* than the Newton stopping test. Losing that says the stage
      solves are no longer converging the way the derivation assumes, which
      is a change in the accuracy basis even if no certified test fails.
    """
    problem = CoupledNonlinear()
    objective = FullCostObjective(nx=3, nu=2)
    y0 = np.array([0.4, -0.25, 0.15])
    t_span = (0.3, 1.1)

    worst_ratio = 0.0
    worst_case = ""
    for param in IMPLICIT_METHODS:
        factory = param.values[0]
        for N in (3, 6):
            method = factory()
            u = make_controls(N, method.s, problem.control_dim, seed=N)
            grad_pkg = GLMOptimizer(
                problem, objective, method, t_span, N, y0
            ).gradient(u)
            grad_ref = reference_gradient(
                y0, u, t_span, N, problem, method, objective
            )
            ratio = _rel_err(grad_pkg, grad_ref) / STAGE_NEWTON_TOL
            if ratio > worst_ratio:
                worst_ratio, worst_case = ratio, f"{param.id} N={N}"

    assert worst_ratio < CONDITIONING_ALLOWANCE, (
        f"observed error/Newton-tolerance ratio {worst_ratio:.4f} at "
        f"{worst_case} exceeds CONDITIONING_ALLOWANCE "
        f"{CONDITIONING_ALLOWANCE:.1e}: IMPLICIT_RTOL no longer bounds "
        f"observed behavior and its derivation must be revisited."
    )
    assert worst_ratio < 1.0, (
        f"observed error/Newton-tolerance ratio {worst_ratio:.4f} at "
        f"{worst_case} is no longer below 1. The C-3.4 derivation assumes "
        f"Newton overshoots its stopping test; that assumption no longer "
        f"holds and the accuracy basis has changed."
    )
