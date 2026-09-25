"""The shipped example must run, and its answer must be right.

The README previously carried an example that could not execute: the
objective class referenced ``self.y_target`` without ever setting it, used a
method name the ``Objective`` protocol does not define, returned the wrong
shapes from two derivatives, and omitted the second derivatives the exact
Hessian requires. Nothing in the repository ran it, so nothing caught it.

These tests execute the real example module. The README quotes this module
rather than carrying its own copy of the code, so documentation drift becomes
a test failure.
"""

import pathlib
import re

import numpy as np
import pytest

from adjungo.optimization.interface import GLMOptimizer
from adjungo.validation import reference_gradient, reference_hessian
from examples.minimum_energy_oscillator import (
    CONTROL_WEIGHT,
    TERMINAL_WEIGHT,
    Y0,
    Y_TARGET,
    DampedOscillator,
    MinimumEnergyObjective,
    build_optimizer,
    solve,
)

# ---------------------------------------------------------------------------
# The physical setup is in the regime the docstring claims
# ---------------------------------------------------------------------------


def test_oscillator_is_underdamped_and_resolved_by_the_mesh():
    """The example's scale table must describe the example that runs.

    A docstring claiming ``zeta = 0.1`` and ``h*omega_0 = 0.1`` while the
    constants say otherwise is worse than no docstring: a reader would draw
    conclusions about a system the code does not simulate.
    """
    problem = DampedOscillator()

    assert problem.natural_frequency == pytest.approx(2.0)
    assert problem.damping_ratio == pytest.approx(0.1)
    assert 0.0 < problem.damping_ratio < 1.0, "claimed underdamped"

    optimizer = build_optimizer()
    assert optimizer.h * problem.natural_frequency == pytest.approx(0.1)

    steps_per_period = (2 * np.pi / problem.natural_frequency) / optimizer.h
    assert steps_per_period > 50, (
        f"only {steps_per_period:.1f} steps per natural period; the mesh no "
        f"longer resolves the oscillation the example is about"
    )


def test_oscillator_rejects_nonpositive_mass():
    """An unphysical parameter is refused at construction, not later."""
    with pytest.raises(ValueError, match="mass must be positive"):
        DampedOscillator(mass=0.0)


# ---------------------------------------------------------------------------
# The example's derivatives are the certified ones
# ---------------------------------------------------------------------------


def test_example_gradient_matches_independent_reference():
    """The example is a certification case, not just a demo.

    A README example that runs but computes the wrong gradient is the exact
    failure this repository is trying to stop advertising.
    """
    problem = DampedOscillator()
    objective = MinimumEnergyObjective()
    optimizer = build_optimizer(n_steps=8)
    method = optimizer.method

    rng = np.random.default_rng(3)
    u = rng.standard_normal((8, method.s, 1))

    grad_pkg = optimizer.gradient(u)
    grad_ref = reference_gradient(
        Y0, u, (0.0, optimizer.t_span[1]), 8, problem, method, objective
    )

    assert np.max(np.abs(grad_ref)) > 1e-3, "degenerate case"
    err = np.max(np.abs(grad_pkg - grad_ref)) / max(
        float(np.max(np.abs(grad_ref))), 1.0
    )
    assert err < 1e-11, f"example gradient off by {err:.3e}"


def test_example_hessp_matches_independent_reference():
    """``scipy_hessp`` must deliver the same operator as ``reference_hessian``."""
    problem = DampedOscillator()
    objective = MinimumEnergyObjective()
    optimizer = build_optimizer(n_steps=6)
    method = optimizer.method

    rng = np.random.default_rng(11)
    u = rng.standard_normal((6, method.s, 1))
    H_ref = reference_hessian(
        Y0, u, (0.0, optimizer.t_span[1]), 6, problem, method, objective
    )

    hessp = optimizer.scipy_hessp()
    n_dof = u.size
    H_pkg = np.zeros((n_dof, n_dof))
    for j in range(n_dof):
        e = np.zeros(n_dof)
        e[j] = 1.0
        H_pkg[:, j] = hessp(u.ravel(), e)

    scale = max(float(np.max(np.abs(H_ref))), 1.0)
    assert float(np.max(np.abs(H_pkg - H_ref))) / scale < 1e-11


def test_scipy_hessp_reuses_the_forward_solve_across_directions():
    """The documented reuse claim, checked rather than asserted.

    ``scipy_hessp`` states that only ``u`` selects the trajectory, so a
    Krylov method sweeping many directions at fixed ``u`` pays for one
    forward and adjoint solve. If a future change keyed the cache on the
    direction too, every inner Krylov iteration would silently re-solve the
    nonlinear system and the operator would still be *correct*, so no
    accuracy test would notice.
    """
    optimizer = build_optimizer(n_steps=6)
    calls = {"n": 0}
    original = optimizer.stage_solver.solve_stages

    def counting_solve_stages(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    optimizer.stage_solver.solve_stages = counting_solve_stages  # type: ignore[method-assign]

    rng = np.random.default_rng(2)
    u = rng.standard_normal((6, optimizer.method.s, 1))
    hessp = optimizer.scipy_hessp()

    hessp(u.ravel(), rng.standard_normal(u.size))
    after_first = calls["n"]
    assert after_first > 0, "no forward solve happened at all"

    for _ in range(4):
        hessp(u.ravel(), rng.standard_normal(u.size))

    assert calls["n"] == after_first, (
        f"forward solve re-ran {calls['n'] - after_first} times for four "
        f"extra directions at the same u; the trajectory cache is keyed on "
        f"the direction, which makes Krylov methods needlessly expensive."
    )


# ---------------------------------------------------------------------------
# The example actually solves its stated problem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_hessian", [True, False], ids=["trust-ncg", "l-bfgs-b"])
def test_example_converges_and_improves_on_doing_nothing(use_hessian):
    """Both advertised optimizer routes reach a stationary point.

    The controlled trajectory must end materially closer to the target than
    the uncontrolled one; otherwise the example demonstrates nothing, and a
    gradient that was identically zero would also "converge".
    """
    optimizer, result, u_optimal = solve(n_steps=20, use_hessian=use_hessian)

    assert result.success, f"optimizer did not converge: {result.message}"

    y_controlled = optimizer.trajectory(u_optimal).Y[-1][0]
    y_uncontrolled = optimizer.trajectory(np.zeros_like(u_optimal)).Y[-1][0]

    miss_controlled = np.linalg.norm(y_controlled - Y_TARGET)
    miss_uncontrolled = np.linalg.norm(y_uncontrolled - Y_TARGET)

    assert miss_controlled < 0.1 * miss_uncontrolled, (
        f"control achieved little: terminal miss {miss_controlled:.4e} vs "
        f"{miss_uncontrolled:.4e} uncontrolled"
    )

    grad = optimizer.gradient(u_optimal)
    assert np.max(np.abs(grad)) < 1e-6, (
        f"reported success but |grad|_inf = {np.max(np.abs(grad)):.3e}"
    )


def test_exact_hessian_reaches_a_sharper_stationary_point():
    """Both routes find the same optimum; the exact Hessian locates it better.

    Linear dynamics plus a quadratic cost make ``J(u)`` an exactly quadratic
    form, so both methods converge to the same minimiser and the same
    objective value. They do not resolve it equally well: a Newton method
    supplied with the exact Hessian drives the gradient to rounding level,
    while a quasi-Newton method stops at its own curvature-approximation
    error.

    Iteration counts are deliberately **not** compared. The two algorithms
    use different termination criteria and do different work per iteration,
    so that comparison measures SciPy's stopping rules rather than the
    quality of the Hessian. Final stationarity is the attributable quantity.
    """
    newton_opt, newton_result, u_newton = solve(n_steps=20, use_hessian=True)
    _, lbfgs_result, u_lbfgs = solve(n_steps=20, use_hessian=False)

    assert newton_result.success and lbfgs_result.success

    # Same problem, same minimum.
    assert newton_result.fun == pytest.approx(lbfgs_result.fun, rel=1e-9)

    g_newton = float(np.max(np.abs(newton_opt.gradient(u_newton))))
    g_lbfgs = float(np.max(np.abs(newton_opt.gradient(u_lbfgs))))

    assert g_newton < 1e-14, (
        f"trust-ncg with the exact Hessian left |grad|_inf = {g_newton:.3e}; "
        f"on an exactly quadratic problem it should reach rounding level. "
        f"A Hessian that is merely close would show here."
    )
    assert g_newton < g_lbfgs, (
        f"exact-Hessian stationarity {g_newton:.3e} is no better than "
        f"quasi-Newton's {g_lbfgs:.3e}"
    )


# ---------------------------------------------------------------------------
# Objective bookkeeping
# ---------------------------------------------------------------------------


def test_objective_value_matches_its_own_formula():
    """``evaluate`` must be the function whose derivatives are supplied.

    An objective whose ``evaluate`` disagrees with its ``dJ_*`` methods makes
    every gradient check meaningless, because the two sides would describe
    different functions.
    """
    optimizer = build_optimizer(n_steps=10)
    rng = np.random.default_rng(7)
    u = rng.standard_normal((10, optimizer.method.s, 1))

    y_final = optimizer.trajectory(u).Y[-1][0]
    miss = y_final - Y_TARGET
    expected = 0.5 * TERMINAL_WEIGHT * float(np.dot(miss, miss)) + 0.5 * (
        CONTROL_WEIGHT * float(np.sum(u**2))
    )

    assert optimizer.objective_value(u) == pytest.approx(expected, rel=1e-13)


def test_trajectory_is_a_copy():
    """Mutating the returned trajectory must not corrupt the cache.

    The optimizer keys its cache on ``u``. If it handed out the live arrays,
    a caller mutating them would leave every later gradient describing a
    trajectory no forward solve ever produced, with the cache key still
    matching.
    """
    optimizer = build_optimizer(n_steps=5)
    u = np.zeros((5, optimizer.method.s, 1))

    first = optimizer.trajectory(u)
    first.Y[:] = 12345.0

    second = optimizer.trajectory(u)
    assert not np.allclose(second.Y, 12345.0), "trajectory aliases the cache"
    assert second.Y[0][0] == pytest.approx(Y0)


def test_example_module_runs_as_a_script(capsys):
    """``python examples/minimum_energy_oscillator.py`` must work.

    This is the command the README tells a reader to run.
    """
    from examples.minimum_energy_oscillator import main

    main()
    out = capsys.readouterr().out
    assert "converged                  : True" in out
    assert "damping ratio" in out


# ---------------------------------------------------------------------------
# The README's code must run
# ---------------------------------------------------------------------------


def _readme_quickstart_blocks() -> list[str]:
    """Extract the ```python blocks of the README's "Quick start" section."""
    readme = (
        pathlib.Path(__file__).resolve().parent.parent / "README.md"
    ).read_text()

    start = readme.index("## Quick start")
    end = readme.index("## Validation")
    section = readme[start:end]

    blocks = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    assert blocks, "no python blocks found in the README Quick start section"
    return blocks


def test_readme_quickstart_executes():
    """The README's quick start must run as written.

    Concatenated in document order, the blocks define the problem, define the
    objective, and run the optimisation. A reader copying them in sequence
    gets exactly this script.

    The previous README example could not execute at all: its objective
    referenced ``self.y_target`` without setting it, defined ``evaluate`` with
    the wrong signature contract, returned ``np.zeros((1, 2))`` where a shape
    ``(2,)`` gradient was required, and omitted every second derivative. This
    test exists so that cannot recur silently.
    """
    script = "\n".join(_readme_quickstart_blocks())
    namespace: dict = {}

    # Executing the README is the entire point of this test. The input
    # is a file in this repository, not caller-supplied data.
    exec(compile(script, "README.md:quickstart", "exec"), namespace)  # noqa: S102

    result = namespace["result"]
    assert result.success, f"README example did not converge: {result.message}"

    y_final = namespace["y_final"]
    assert np.all(np.isfinite(y_final))
    assert abs(y_final[0]) < 1e-2, (
        f"README claims the mass reaches x(T) ~ 5.7e-4 m; got {y_final[0]:.3e}"
    )

    optimizer = namespace["optimizer"]
    u_optimal = namespace["u_optimal"]
    grad = optimizer.gradient(u_optimal)
    assert np.max(np.abs(grad)) < 1e-14, (
        f"README claims |grad J|_inf ~ 2e-17; got "
        f"{np.max(np.abs(grad)):.3e}"
    )


def test_readme_does_not_advertise_refused_families():
    """No line of the README may mention a refused family affirmatively.

    The assessment that prompted this work found advertised capabilities
    exceeding the implementation. ``GLMOptimizer`` raises
    ``NotImplementedError`` at construction for multistep (``r > 1``) and
    IMEX methods, so the README may name them only while marking them
    refused.

    Every line is checked, not only table rows. The README this replaced
    advertised these families in a bullet list under "Features", which a
    table-only check would not have caught.
    """
    readme = (
        pathlib.Path(__file__).resolve().parent.parent / "README.md"
    ).read_text()

    # Check whole paragraphs, not raw lines: prose is hard-wrapped, so a
    # sentence that does mark a refusal can straddle a line break. Table
    # rows are self-contained and are checked individually.
    units: list[str] = []
    for paragraph in readme.split("\n\n"):
        rows = [ln for ln in paragraph.splitlines() if ln.lstrip().startswith("|")]
        if rows:
            units.extend(rows)
        else:
            units.append(" ".join(paragraph.split()))

    refused_terms = ("adams", "bdf", "imex", "multistep")
    allowed = ("refused", "reject", "not reachable", "not representable")
    offenders = [
        unit
        for unit in units
        if any(term in unit.lower() for term in refused_terms)
        and not any(ok in unit.lower() for ok in allowed)
    ]

    assert not offenders, (
        "README mentions a refused method family without marking it "
        f"refused: {offenders}"
    )


def test_refused_families_are_actually_refused():
    """Pin the behaviour the README documents.

    The previous test constrains prose. This one constrains the code, so the
    two cannot drift apart.

    Refusal happens at two different places, for two different reasons, and
    both are pinned here:

    - BDF tableaux are representable (``s = 1``, ``r = 2``) and construct
      normally. ``GLMOptimizer`` refuses them because ``r > 1`` has no
      starting procedure, raising ``NotImplementedError``.
    - Adams tableaux need per-history stage coefficients, giving a
      non-square ``A`` of shape ``(1, 2)``. They are refused earlier still,
      by ``GLMethod`` validation, raising ``ValueError``.
    """
    from adjungo.methods.experimental.multistep import adams_bashforth2, bdf2

    # Representable, but not usable: refused by the optimizer envelope.
    with pytest.raises(NotImplementedError):
        GLMOptimizer(
            problem=DampedOscillator(),
            objective=MinimumEnergyObjective(),
            method=bdf2(),
            t_span=(0.0, 1.0),
            N=10,
            y0=np.array([1.0, 0.0]),
        )

    # Not even representable: refused by the tableau dataclass.
    with pytest.raises(ValueError, match="must be square"):
        adams_bashforth2()


# ---------------------------------------------------------------------------
# examples/nonlinear_implicit_control.py
#
# The affine oscillator above exercises neither stage coupling nor a
# non-zero second derivative of ``f``. This example is the implicit,
# genuinely nonlinear counterpart, and is held to the same standard: it is a
# certification case, not a demonstration.
# ---------------------------------------------------------------------------


def _vdp():
    from examples.nonlinear_implicit_control import (
        MinimumEnergyObjective as VdpObjective,
    )
    from examples.nonlinear_implicit_control import (
        VanDerPolOscillator,
    )

    return VanDerPolOscillator(), VdpObjective()


def test_van_der_pol_jacobian_actually_depends_on_the_state():
    """The example's reason for existing, checked rather than asserted.

    If ``df/dy`` did not vary with ``y`` the example would demonstrate
    nothing that the affine oscillator does not, and the R-9 discussion in
    its docstring would be describing a hazard it cannot exhibit.
    """
    problem, _ = _vdp()
    u = np.array([0.3])

    at_a = problem.F(np.array([2.0, 0.5]), u, 0.0)
    at_b = problem.F(np.array([-1.0, 0.5]), u, 0.0)

    assert np.max(np.abs(at_a - at_b)) > 1.0, (
        "the Van der Pol Jacobian is supposed to vary with the state"
    )
    # ...and not with the control, which is what made R-9's probe pass.
    other_u = problem.F(np.array([2.0, 0.5]), np.array([99.0]), 0.0)
    assert np.array_equal(at_a, other_u)


def test_van_der_pol_second_derivatives_match_differences_of_the_jacobian():
    """``F_yy_action`` is hand-derived, so it is checked against ``F``.

    A hand-differentiated second derivative is exactly the kind of term that
    can be wrong while every solve still converges, because the forward
    problem never touches it. ``F`` itself is verified against ``f`` by the
    same construction.

    Central differences are second-order accurate, so at ``eps = 1e-5`` the
    truncation error is ``O(1e-10)`` and the subtractive cancellation floor
    is ``O(1e-11)``; ``1e-7`` is a loose bound on their sum and is a
    statement about the difference formula, not about this machine.
    """
    problem, _ = _vdp()
    y = np.array([1.7, -0.6])
    u = np.array([0.25])
    w = np.array([0.4, -1.3])
    eps = 1e-5

    analytic = problem.F_yy_action(y, u, 0.0, w)

    differenced = np.zeros((2, 2))
    for j in range(2):
        step = np.zeros(2)
        step[j] = eps
        dF = (
            problem.F(y + step, u, 0.0) - problem.F(y - step, u, 0.0)
        ) / (2 * eps)
        differenced[:, j] = w @ dF

    assert np.max(np.abs(analytic - differenced)) < 1e-7, (
        f"F_yy_action disagrees with differences of F:\n{analytic}\n"
        f"{differenced}"
    )

    # And F itself against differences of f, so the reference above is sound.
    F_differenced = np.zeros((2, 2))
    for j in range(2):
        step = np.zeros(2)
        step[j] = eps
        F_differenced[:, j] = (
            problem.f(y + step, u, 0.0) - problem.f(y - step, u, 0.0)
        ) / (2 * eps)
    assert np.max(np.abs(problem.F(y, u, 0.0) - F_differenced)) < 1e-7


def test_nonlinear_example_gradient_matches_independent_reference():
    """C-2 on the fully implicit route, through the shipped example.

    ``reference_gradient`` re-derives the whole discrete problem from the
    monolithic residual and shares no code with ``adjungo/stepping/``, which
    is the strongest oracle available (C-14.1 item 1). The tolerance matches
    the affine example's: a relative ``1e-11`` against a C-3 budget that
    leaves several orders of headroom over the observed agreement.
    """
    from examples.nonlinear_implicit_control import Y0 as VDP_Y0
    from examples.nonlinear_implicit_control import build_optimizer as vdp_build

    problem, objective = _vdp()
    optimizer = vdp_build(n_steps=8)
    method = optimizer.method

    rng = np.random.default_rng(5)
    u = rng.standard_normal((8, method.s, 1))

    grad_pkg = optimizer.gradient(u)
    grad_ref = reference_gradient(
        VDP_Y0, u, (0.0, optimizer.t_span[1]), 8, problem, method, objective
    )

    assert np.max(np.abs(grad_ref)) > 1e-3, "degenerate case"
    err = np.max(np.abs(grad_pkg - grad_ref)) / max(
        float(np.max(np.abs(grad_ref))), 1.0
    )
    assert err < 1e-11, f"example gradient off by {err:.3e}"


def test_nonlinear_example_hessp_matches_independent_reference():
    """The exact Hessian on a problem whose ``F_yy`` is not zero.

    In the affine example every second derivative of ``f`` vanishes, so this
    comparison cannot detect a dropped ``F_yy`` term. Here it can.
    """
    from examples.nonlinear_implicit_control import Y0 as VDP_Y0
    from examples.nonlinear_implicit_control import build_optimizer as vdp_build

    problem, objective = _vdp()
    optimizer = vdp_build(n_steps=5)
    method = optimizer.method

    rng = np.random.default_rng(13)
    u = rng.standard_normal((5, method.s, 1))
    H_ref = reference_hessian(
        VDP_Y0, u, (0.0, optimizer.t_span[1]), 5, problem, method, objective
    )

    hessp = optimizer.scipy_hessp()
    n_dof = u.size
    H_pkg = np.zeros((n_dof, n_dof))
    for j in range(n_dof):
        e = np.zeros(n_dof)
        e[j] = 1.0
        H_pkg[:, j] = hessp(u.ravel(), e)

    scale = max(float(np.max(np.abs(H_ref))), 1.0)
    assert float(np.max(np.abs(H_pkg - H_ref))) / scale < 1e-11


def test_nonlinear_example_reaches_its_target():
    """The example must actually solve the problem it advertises."""
    from examples.nonlinear_implicit_control import Y_TARGET as VDP_TARGET
    from examples.nonlinear_implicit_control import solve as vdp_solve

    optimizer, result, u_optimal = vdp_solve(n_steps=20)

    assert result.success, result.message
    zero = np.zeros_like(u_optimal)
    assert result.fun < 0.05 * optimizer.objective_value(zero), (
        "optimisation barely improved on doing nothing"
    )

    final = optimizer.trajectory(u_optimal).Y[-1][0]
    assert np.max(np.abs(final - VDP_TARGET)) < 0.05, (
        f"control did not bring the state to the target: {final}"
    )


def test_nonlinear_example_module_runs_as_a_script(capsys):
    """``python examples/nonlinear_implicit_control.py`` must work."""
    from examples.nonlinear_implicit_control import main

    main()
    out = capsys.readouterr().out
    assert "Van der Pol" in out
    assert "coupled Newton system" in out


# ---------------------------------------------------------------------------
# examples/factorization_reuse_counts.py
#
# The claim this example makes is about *work*, which no accuracy test can
# check. Its assertions are therefore counts, exactly as C-15 requires.
# ---------------------------------------------------------------------------


def test_reuse_example_matches_its_own_prediction():
    """The declared solve takes the predicted number of factorizations.

    ``factorizations_for_solve`` is the contract's prediction and the store's
    counter is the observation; the example is only honest if they agree.
    The assertion is equality with the prediction, not a loose bound: a bound
    such as ``<= 2 N`` would still pass if reuse worked within a step but not
    across steps, which is the likeliest way for this to regress.
    """
    from adjungo.core.requirements import deduce_requirements
    from adjungo.methods.runge_kutta import sdirk3
    from examples.factorization_reuse_counts import (
        DECLARED_CONSTANT,
        N_STEPS,
        LinearSystem,
        count_factorizations,
    )

    predicted = deduce_requirements(
        sdirk3(), DECLARED_CONSTANT, LinearSystem.state_dim
    ).factorizations_for_solve(N_STEPS)
    assert predicted == 1

    observed, reuses, _ = count_factorizations(DECLARED_CONSTANT)
    assert observed == predicted, (
        f"declared-constant solve took {observed} factorizations, predicted "
        f"{predicted}"
    )
    assert reuses > 0, "nothing was actually reused"


def test_reuse_example_does_not_infer_structure_that_was_not_declared():
    """The undeclared run must not quietly discover the constancy.

    This is the half of the example that records precedent R-9: constancy is
    a property of the problem, which the caller declares, never something the
    library concludes by probing points the stages do not visit.
    """
    from examples.factorization_reuse_counts import (
        DECLARED_CONSTANT,
        UNDECLARED,
        count_factorizations,
    )

    declared, _, _ = count_factorizations(DECLARED_CONSTANT)
    undeclared, undeclared_reuses, _ = count_factorizations(UNDECLARED)

    assert undeclared > declared, (
        "the undeclared solve reused factorizations it was never told it "
        f"could: {undeclared} vs {declared}"
    )
    assert undeclared_reuses == 0


def test_reuse_example_changes_the_work_and_not_the_answer():
    """Reuse is an optimization; the gradient must be unaffected.

    The two routes factor the same matrix a different number of times, so
    they agree to within the reproducibility of LU on one platform. The
    bound is relative and generous rather than bit-exact: per C-11.3 no
    floating-point output is pinned, because the certified LU path is
    permitted to differ in its last ULPs between BLAS backends.
    """
    from examples.factorization_reuse_counts import (
        DECLARED_CONSTANT,
        UNDECLARED,
        count_factorizations,
    )

    _, _, declared_grad = count_factorizations(DECLARED_CONSTANT)
    _, _, undeclared_grad = count_factorizations(UNDECLARED)

    scale = max(float(np.max(np.abs(undeclared_grad))), 1.0)
    err = float(np.max(np.abs(declared_grad - undeclared_grad))) / scale
    assert err < 1e-12, f"reuse changed the gradient by {err:.3e}"


def test_reuse_example_module_runs_as_a_script(capsys):
    """``python examples/factorization_reuse_counts.py`` must work."""
    from examples.factorization_reuse_counts import main

    main()
    out = capsys.readouterr().out
    assert "predicted factorizations" in out
    assert "observed  factorizations" in out
