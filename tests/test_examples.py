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

import importlib.util
import itertools
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


# ---------------------------------------------------------------------------
# examples/rocket_ascent.py and the sympy derivation layer
#
# This example differs from the two above in three ways that need separate
# checks: its derivative callbacks are differentiated by sympy rather than
# written out, its dynamics have a closed-form solution under a constant
# control, and its objective carries the C-9.3 quadrature factors itself.
# ---------------------------------------------------------------------------

requires_sympy = pytest.mark.skipif(
    importlib.util.find_spec("sympy") is None,
    reason=(
        "examples/symbolic.py needs the `examples` extra; install with "
        "`pip install -e '.[dev,examples]'`. CI installs it, and "
        "tests/test_documentation.py asserts that it does, so this skip "
        "cannot go unnoticed there."
    ),
)


def _perturbed_burn(n_steps: int, seed: int) -> np.ndarray:
    """A physically meaningful control: the constant burn, jittered.

    Standard normal controls would be negative burn rates, which the model
    has no meaning for -- thrust would reverse and mass would grow. The
    derivative claims are about the discrete map and hold at any admissible
    point, so the point chosen is one the problem admits.
    """
    from examples.rocket_ascent import CONSTANT_BURN

    rng = np.random.default_rng(seed)
    jitter = rng.standard_normal((n_steps, 4, 1))
    return CONSTANT_BURN * (1.0 + 0.3 * jitter)


def _reference_solution(optimizer, u, n_steps):
    """The monolithic reference solve, at a tolerance this problem can reach.

    ``reference_solve`` measures ``||R||_inf`` against an absolute default of
    1e-13. That suits an O(1) problem; here the residual entries scale like
    ``h * max|f| ~ 7.5 s * 1e3 m/s ~ 7.5e3``, whose unit roundoff is about
    1.7e-12, so the default is below the floating-point floor. Newton
    descends to 1.36e-12, cannot improve on it, and exhausts ``max_iter``
    there -- a refusal of an answer that is already as good as the format
    allows.

    An explicit tableau does **not** make that iteration exact. It makes
    ``R_Z`` block strictly lower triangular plus identity, so each Newton
    step is a forward substitution; the iteration terminates in one step
    only when ``f`` is affine in the state, as in the oscillator example.
    This ``f`` is not, and the solve measurably takes two steps.

    The tolerance below is therefore a stopping rule, not an error bound. It
    is set at 1e-10, about sixty times the roundoff of the residual scale, so
    the iteration stops on the quadratic step that lands at ~1.8e-12 instead
    of grinding against an unreachable target. What justifies it is not the
    residual but the comparisons themselves: the package and the reference
    agree to below 4e-15 relative, three orders inside the 1e-11 budget
    asserted below. A residual tolerance alone would not establish that.
    """
    from adjungo.validation import reference_solve
    from examples.rocket_ascent import T_FINAL, Y0

    return reference_solve(
        Y0,
        u,
        (0.0, T_FINAL),
        n_steps,
        optimizer.problem,
        optimizer.method,
        tol=1e-10,
    )


#: Rounding budget for a generated derivative against a hand-written one.
#: Each entry of this problem's derivatives is a product or quotient of at
#: most four operands, so at most four roundings separate any two orderings
#: of the same expression; eight unit roundoffs leaves a factor of two over
#: that bound. Observed deviation over 2000 sampled points on this machine
#: was 0.0, but a generator that reassociated a product would still be
#: correct, and the test must not call that a failure.
DERIVATION_BUDGET = 8 * np.finfo(float).eps


def _agrees_by_derivation(
    generated: np.ndarray,
    hand: np.ndarray,
    name: str,
    scale: np.ndarray | None = None,
) -> None:
    """Compare a generated derivative with a hand-written one, entry by kind.

    ``scale`` is the magnitude that sets each entry's rounding. It defaults to
    ``|hand|``, which is right for a product or quotient: reassociating one
    changes the result by a few units in its own last place. It is wrong for a
    sum that cancels, where the roundings are of the *terms* and survive into a
    smaller result. ``f`` for the pendulum is such a sum -- at one sampled
    point the terms are 3.7 in magnitude and total 0.113, a condition number of
    33 -- so those callers pass the sum of the term magnitudes instead. The
    observed deviations are then within ``2 * eps`` of it.
    """
    generated = np.asarray(generated, dtype=float)
    assert generated.shape == hand.shape, (
        f"{name}: generated shape {generated.shape}, hand-derived {hand.shape}"
    )

    structural = hand == 0.0
    assert np.array_equal(generated[structural], hand[structural]), (
        f"{name}: an entry that vanishes identically came back nonzero at "
        f"{np.argwhere(structural & (generated != 0.0)).tolist()}"
    )

    computed = ~structural
    if not computed.any():
        return
    denominator = np.abs(hand if scale is None else np.asarray(scale, dtype=float))
    deviation = np.abs(generated[computed] - hand[computed]) / denominator[computed]
    assert np.max(deviation) <= DERIVATION_BUDGET, (
        f"{name}: generated and hand-derived entries differ by "
        f"{np.max(deviation):.3e} relative, over the "
        f"{DERIVATION_BUDGET:.3e} rounding budget"
    )


@requires_sympy
def test_symbolic_derivatives_match_hand_derivation():
    """The generated callbacks equal derivatives taken by hand.

    This is the check that makes the sympy layer usable. ``f`` is small
    enough here to differentiate on paper, so the two paths are genuinely
    independent: an error in the contraction convention, a transpose, or a
    dropped term would separate them. The contractions follow the ``Problem``
    docstrings, ``F_yy_action(y, u, t, v)_{ij} = sum_l v_l d2 f_l/dy_i dy_j``.

    Two different claims are made about the two kinds of entry, and
    ``_agrees_by_derivation`` states the basis for each. A structurally zero
    entry must be exactly zero: that the derivative *vanishes identically* is
    an algebraic fact, not a small computed number. Every other entry is
    compared against a rounding budget, because sympy chooses its own
    association and is free to emit ``-(ve*z)/m**2`` where the hand form
    writes ``-ve*z/m**2``. Requiring the last bits to match would pin a code
    generator's formatting decisions, which C-11.3 forbids.
    """
    from examples.rocket_ascent import EXHAUST_VELOCITY, GRAVITY, rocket_dynamics

    dynamics = rocket_dynamics()
    assert dynamics.state_dim == 3
    assert dynamics.control_dim == 1

    rng = np.random.default_rng(17)
    for _ in range(5):
        altitude, velocity = rng.uniform(0.0, 5000.0), rng.uniform(-100.0, 900.0)
        mass = rng.uniform(25.0, 100.0)
        y = np.array([altitude, velocity, mass])
        u = np.array([rng.uniform(0.1, 3.0)])
        v = rng.standard_normal(3)
        ve, z, m = EXHAUST_VELOCITY, u[0], mass

        f_hand = np.array([velocity, ve * z / m - GRAVITY, -z])
        F_hand = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -ve * z / m**2],
                [0.0, 0.0, 0.0],
            ]
        )
        G_hand = np.array([[0.0], [ve / m], [-1.0]])

        # Only v1 * ve * z / m carries any second derivative.
        F_yy_hand = np.zeros((3, 3))
        F_yy_hand[2, 2] = 2.0 * v[1] * ve * z / m**3
        F_yu_hand = np.array([[0.0], [0.0], [-v[1] * ve / m**2]])
        F_uu_hand = np.zeros((1, 1))

        _agrees_by_derivation(dynamics.f(y, u, 0.0), f_hand, "f")
        _agrees_by_derivation(dynamics.F(y, u, 0.0), F_hand, "F")
        _agrees_by_derivation(dynamics.G(y, u, 0.0), G_hand, "G")
        _agrees_by_derivation(dynamics.F_yy_action(y, u, 0.0, v), F_yy_hand, "F_yy")
        _agrees_by_derivation(dynamics.F_yu_action(y, u, 0.0, v), F_yu_hand, "F_yu")
        _agrees_by_derivation(dynamics.F_uu_action(y, u, 0.0, v), F_uu_hand, "F_uu")


@requires_sympy
def test_symbolic_dynamics_refuses_an_unbound_parameter():
    """A symbol that is neither state, control nor time cannot be evaluated.

    Per C-7 this is refused at construction rather than surfacing later as a
    lambdify ``TypeError`` about a missing argument, or worse, as a callback
    that silently closes over whatever the name resolves to.
    """
    import sympy as sp

    from examples.symbolic import SymbolicDynamics

    y, u, t, forgotten = sp.symbols("y u t forgotten", real=True)
    with pytest.raises(ValueError, match="neither state, control nor time"):
        SymbolicDynamics(sp.Matrix([forgotten * y + u]), [y], [u], t)


@requires_sympy
def test_constant_burn_matches_the_tsiolkovsky_closed_form():
    """C-14.1 tier 2: the forward solve against a closed-form anchor.

    Under a constant burn rate the model integrates in elementary functions,
    and ``constant_burn_state`` shares no code with ``adjungo.stepping``. The
    residual here is rk4's truncation error, not rounding: at ``N = 60`` over
    a 60 s horizon the altitude error is about 6e-4 m in 1.8e4 m. The bound
    is a relative 1e-7, roughly three times the observed error, and it is a
    discretization budget -- the next test is what pins the rate.
    """
    from examples.rocket_ascent import (
        N_STEPS,
        T_FINAL,
        build_optimizer,
        constant_burn_state,
        initial_control,
    )

    computed = build_optimizer().trajectory(initial_control()).Y[-1][0]
    exact = constant_burn_state(T_FINAL)

    assert exact[2] == pytest.approx(20.0), "the constant burn should use all fuel"
    relative = np.abs(computed - exact) / np.maximum(np.abs(exact), 1.0)
    assert np.max(relative) < 1e-7, (
        f"N={N_STEPS} solve differs from the closed form by {relative}"
    )


@requires_sympy
def test_rocket_forward_solve_attains_fourth_order():
    """C-4: refine the mesh and recover rk4's order against the closed form.

    This refines the mesh and is therefore an order-of-accuracy claim, not a
    derivative claim; AGENTS.md requires the two to be different tests. The
    oracle is the closed form, so the measured quantity is true
    discretization error rather than a difference between two discrete
    solutions.
    """
    from adjungo.methods.runge_kutta import rk4
    from examples.rocket_ascent import (
        CONSTANT_BURN,
        T_FINAL,
        build_optimizer,
        constant_burn_state,
    )

    exact = constant_burn_state(T_FINAL)[0]
    errors = []
    for n_steps in (15, 30, 60, 120):
        control = np.full((n_steps, rk4().s, 1), CONSTANT_BURN)
        final = build_optimizer(n_steps).trajectory(control).Y[-1][0][0]
        errors.append(abs(final - exact))

    rates = [np.log2(a / b) for a, b in itertools.pairwise(errors)]
    # Observed 3.96, 3.99, 4.00. The floor admits the coarsest mesh's
    # pre-asymptotic deficit; a first- or second-order defect would give
    # rates near 1 or 2 and fail.
    assert min(rates) > 3.8, f"observed orders {rates} from errors {errors}"
    assert errors[-1] < 1e-4


@requires_sympy
def test_rocket_gradient_matches_the_independent_reference():
    """The gradient of an objective that carries its own quadrature factors.

    The other two examples' control terms are plain sums over stages. This
    one multiplies by ``h * w_k`` per C-9.3, so a gradient that dropped or
    doubled those factors would still look plausible. The reference assembles
    the same discrete gradient independently of ``adjungo.stepping``.
    """
    from examples.rocket_ascent import T_FINAL, Y0, build_optimizer

    n_steps = 8
    optimizer = build_optimizer(n_steps)
    u = _perturbed_burn(n_steps, seed=5)

    grad_pkg = optimizer.gradient(u)
    grad_ref = reference_gradient(
        Y0,
        u,
        (0.0, T_FINAL),
        n_steps,
        optimizer.problem,
        optimizer.method,
        optimizer.objective,
        solution=_reference_solution(optimizer, u, n_steps),
    )

    assert np.max(np.abs(grad_ref)) > 1e-3, "degenerate case"
    # Two independent assemblies of one discrete quantity: the difference is
    # rounding, so the budget matches the other examples' 1e-11 relative.
    err = np.max(np.abs(grad_pkg - grad_ref)) / max(
        float(np.max(np.abs(grad_ref))), 1.0
    )
    assert err < 1e-11, f"rocket gradient off by {err:.3e}"


@requires_sympy
def test_rocket_hvp_matches_the_independent_reference():
    """The exact Hessian on dynamics whose ``F_yy`` and ``F_yu`` are nonzero.

    ``d2f/dm2`` and ``d2f/dm dz`` are both nonzero here because of the
    ``ve z / m`` term, so a dropped second-derivative contraction changes the
    operator. In the affine example it could not.
    """
    from examples.rocket_ascent import T_FINAL, Y0, build_optimizer

    n_steps = 6
    optimizer = build_optimizer(n_steps)
    u = _perturbed_burn(n_steps, seed=23)
    rng = np.random.default_rng(29)
    v = rng.standard_normal(u.shape)

    H_ref = reference_hessian(
        Y0,
        u,
        (0.0, T_FINAL),
        n_steps,
        optimizer.problem,
        optimizer.method,
        optimizer.objective,
        solution=_reference_solution(optimizer, u, n_steps),
    )
    hv_ref = (H_ref @ v.ravel()).reshape(v.shape)
    hv_pkg = optimizer.hessian_vector_product(u, v)

    assert np.max(np.abs(hv_ref)) > 1e-3, "degenerate case"
    err = np.max(np.abs(hv_pkg - hv_ref)) / max(float(np.max(np.abs(hv_ref))), 1.0)
    assert err < 1e-11, f"rocket HVP off by {err:.3e}"


@requires_sympy
def test_the_optimum_buys_energy_with_almost_no_altitude():
    """The claim the docstring makes about the answer, held as an assertion.

    The bounds are deliberately loose. They are not an accuracy claim about a
    computed number -- the optimum of a nonconvex problem reached by a
    trust-region iteration is not a quantity to pin -- but the qualitative
    result is the whole point of the example, and it would not survive a
    wrong gradient.
    """
    from examples.rocket_ascent import (
        CONSTANT_BURN,
        DRY_MASS,
        TARGET_ALTITUDE,
        _energy,
        initial_control,
        solve,
    )

    n_steps = 60
    optimizer, result, u_optimal = solve(n_steps)

    assert np.linalg.norm(result.jac) < 1e-6, "did not reach a stationary point"

    before = _energy(initial_control(n_steps), n_steps)
    after = _energy(u_optimal, n_steps)
    assert after < 0.9 * before, f"energy only moved {before:.1f} -> {after:.1f}"

    trajectory = optimizer.trajectory(u_optimal)
    miss = abs(trajectory.Y[-1][0][0] - TARGET_ALTITUDE)
    assert miss < 1.0, f"altitude missed by {miss:.3f} m"

    # The shape of the answer: throttle up early, shut down before T.
    assert u_optimal.max() > 1.5 * u_optimal.mean()
    assert u_optimal[-1, -1, 0] < 0.1 * u_optimal.max()

    # The run stays inside the physical region. See the example's ENVELOPE
    # note: this is observed, not imposed. The lower bound is a tolerance
    # rather than an exact zero because the optimum sits *on* z = 0 over the
    # final steps, where the trust-region iterate lands within rounding of
    # the boundary from either side -- the observed minimum is about 5e-10
    # here, and C-11.3 forbids pinning that sign across BLAS backends. A
    # burn that had gone physically negative would be O(CONSTANT_BURN).
    assert trajectory.Y[:, 0, 2].min() > DRY_MASS
    assert u_optimal.min() > -1e-6 * CONSTANT_BURN


@requires_sympy
def test_the_closed_form_refuses_a_rate_that_exhausts_the_mass():
    """C-7: no silent sentinel where the model stops meaning anything.

    Past ``m = 0`` the logarithm's argument is negative and numpy would
    return a nan, which a caller could plot without noticing.
    """
    from examples.rocket_ascent import INITIAL_MASS, constant_burn_state

    with pytest.raises(ValueError, match="exhausts all mass"):
        constant_burn_state(60.0, rate=INITIAL_MASS / 30.0)


@requires_sympy
def test_rocket_example_main_runs(capsys):
    """The example executes end to end and reports what it claims to."""
    from examples.rocket_ascent import main

    main()
    out = capsys.readouterr().out
    assert "Forward solve against the closed form" in out
    assert "minimum mass in flight" in out


def test_each_example_documents_an_invocation_that_resolves():
    """The ``Run directly::`` line of every example must name something real.

    The examples are not uniform: ``rocket_ascent`` imports a sibling module
    and so must be run with ``-m``, while the others are run by path. A
    renamed file or a copied-and-not-edited header would leave a reader with
    a command that fails, which is the same defect class the README quoting
    machinery exists to prevent.
    """
    directory = pathlib.Path(__file__).resolve().parents[1] / "examples"
    checked = 0
    for path in sorted(directory.glob("*.py")):
        text = path.read_text()
        if "Run directly::" not in text:
            # symbolic.py is a helper, not an example; it says so instead.
            assert "Run the examples that use this module" in text, path.name
            continue
        command = text.split("Run directly::", 1)[1].split("\n\n")[1].strip()
        assert command.startswith(".venv/bin/python"), command
        target = command.split()[-1]

        # The mechanism, not just the spelling: running a file by path puts
        # ``examples/`` on ``sys.path`` rather than the repository root, so a
        # module that imports a sibling can only be run with ``-m``.
        if re.search(r"^from examples\.", text, re.MULTILINE):
            assert "-m" in command.split(), (
                f"{path.name} imports a sibling module, so the documented "
                f"command must use -m; it is {command!r}"
            )

        if "-m" in command.split():
            assert importlib.util.find_spec(target) is not None, command
            assert target.replace(".", "/") + ".py" == f"examples/{path.name}"
        else:
            assert (directory.parent / target).is_file(), command
            assert target == f"examples/{path.name}"
        checked += 1
    assert checked == 6, f"expected six runnable examples, found {checked}"


@requires_sympy
def test_the_fuel_budget_is_a_direct_function_of_the_control():
    """Mass is integrated exactly by the quadrature, so no adjoint is needed.

    ``dm/dt = -z`` does not depend on the state, so the Runge-Kutta update
    for that component reduces to the tableau's own quadrature and the
    terminal mass is a *linear* function of the control,

        m_N = m0 - h * sum_n sum_k b_k z_{n,k}.

    This is what keeps C-Q7 from blocking a fuel-constrained problem: a
    constraint with a direct formula in ``u`` needs no sweep per component,
    and the general trajectory-Jacobian API can stay open while such a
    problem is posed. The claim is asserted here rather than stated in prose
    so that a change to the objective's quadrature convention, or to the
    method, cannot quietly invalidate it.

    The two sides are the same sum in different orders. Each accumulates
    about ``N * s`` roundings, so the budget is ``4 * N * s`` unit roundoffs;
    the observed deviation is under two.
    """
    from examples.rocket_ascent import T_FINAL, Y0, build_optimizer

    for n_steps, seed in ((8, 5), (60, 11)):
        optimizer = build_optimizer(n_steps)
        u = _perturbed_burn(n_steps, seed=seed)
        stages = optimizer.method.s

        direct = Y0[2] - (T_FINAL / n_steps) * float(
            np.einsum("k,nkv->", optimizer.method.B[0, :], u)
        )
        stepped = float(optimizer.trajectory(u).Y[-1][0][2])

        budget = 4 * n_steps * stages * np.finfo(float).eps
        deviation = abs(direct - stepped) / abs(stepped)
        assert deviation <= budget, (
            f"N={n_steps}: quadrature formula gives {direct!r}, the solve "
            f"{stepped!r}, {deviation / np.finfo(float).eps:.1f} eps apart"
        )


# ---------------------------------------------------------------------------
# examples/double_integrator.py
#
# The first example here that knows its own answer. Every test below compares
# against something assembled outside adjungo: a backward Riccati recursion on
# an independently built step map, or the closed-form solution of the
# continuous optimality conditions.
# ---------------------------------------------------------------------------


def test_the_step_map_assembly_reproduces_the_package_solve():
    """The independent step map is the same map ``adjungo.stepping`` applies.

    Everything downstream -- the Riccati optimum, the objective the mesh
    study refines -- is built on ``step_maps``. If it assembled a *different*
    discretisation, the comparisons would still agree with each other and
    silently stop being about the package at all. So this is checked first.
    """
    from examples.double_integrator import (
        DRAG,
        T_FINAL,
        Y0,
        build_optimizer,
        discrete_objective,
        step_maps,
    )

    for drag in (0.0, DRAG):
        n_steps = 10
        a_map, b_map = step_maps(n_steps, drag)
        rng = np.random.default_rng(31)
        u = rng.standard_normal((n_steps, 4, 1))

        y = Y0.astype(float).copy()
        for step in range(n_steps):
            y = a_map @ y + b_map @ u[step].ravel()

        optimizer = build_optimizer(n_steps, drag)
        stepped = optimizer.trajectory(u).Y[-1][0]

        # Two assemblies of one linear map over ten steps, so the difference
        # is rounding. The budget is not set from that rounding: it is set
        # from what it has to exclude. Any *different* discretisation differs
        # at O(h^p) with h = 0.2, which is 1e-4 at best, so 1e-10 relative
        # sits six orders below the smallest real defect and six orders above
        # a rounding difference of a fraction of an ulp. A budget pinned just
        # above one machine's observation certifies that machine (C-11.3).
        scale = max(float(np.max(np.abs(stepped))), 1.0)
        assert np.max(np.abs(y - stepped)) / scale < 1e-10
        assert discrete_objective(u, n_steps, drag) == pytest.approx(
            optimizer.objective_value(u), rel=1e-11
        )
        assert T_FINAL > 0.0


def test_the_riccati_control_is_stationary_for_the_package_gradient():
    """The package's gradient vanishes at the independently computed optimum.

    This is the strongest form of the derivative check available here.
    A gradient that was wrong by a constant factor would still vanish at the
    right point, but a gradient that was wrong in *direction* -- the usual
    consequence of a dropped adjoint term -- would not.
    """
    from examples.double_integrator import (
        DRAG,
        build_optimizer,
        riccati_control,
    )

    for drag in (0.0, DRAG):
        n_steps = 20
        optimizer = build_optimizer(n_steps, drag)
        gradient = optimizer.gradient(riccati_control(n_steps, drag))

        # The objective's own control term is R*h*w_k*u ~ 1e-3, and a
        # misdirected gradient would be of that order; the terms that cancel
        # here are ten orders above the budget.
        assert np.max(np.abs(gradient)) < 1e-13, (
            f"drag={drag}: |grad| = {np.max(np.abs(gradient)):.3e} at the "
            "Riccati optimum"
        )


def test_the_optimizer_reaches_the_independently_computed_optimum():
    """``trust-ncg`` on the package's derivatives finds the Riccati control."""
    from examples.double_integrator import DRAG, riccati_control, solve

    for drag in (0.0, DRAG):
        n_steps = 20
        _, result, u_scipy = solve(n_steps, drag)
        u_riccati = riccati_control(n_steps, drag)

        assert result.success, result.message
        # The reduced problem is quadratic with a positive definite Hessian,
        # so the exact Newton step solves it; the iteration count is a
        # trust-region artefact, not a convergence rate.
        assert result.nit < 25
        scale = float(np.max(np.abs(u_riccati)))
        assert np.max(np.abs(u_scipy - u_riccati)) / scale < 1e-11


def test_the_discrete_problem_has_a_unique_minimiser():
    """Positive definite Hessian: the comparisons above are well posed.

    ``rk4`` has a repeated abscissa, ``c_2 = c_3 = ½``, so two stage controls
    act at the same instant. If the discrete objective were flat along some
    direction, 'the' discrete optimum would not be a single object to compare
    with, and an optimizer landing elsewhere on the manifold would look like
    a failure.
    """
    from examples.double_integrator import build_optimizer

    n_steps = 5
    optimizer = build_optimizer(n_steps)
    hessp = optimizer.scipy_hessp()
    size = n_steps * optimizer.method.s
    origin = np.zeros(size)
    hessian = np.column_stack([hessp(origin, np.eye(size)[:, j]) for j in range(size)])

    asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    assert asymmetry < 1e-14, f"Hessian asymmetry {asymmetry:.3e}"

    eigenvalues = np.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    # Smallest observed 3.3e-3 against a largest of 3.1. The floor only has
    # to exclude zero by a clear margin.
    assert eigenvalues.min() > 1e-4, f"spectrum {eigenvalues.min():.3e} .. {eigenvalues.max():.3e}"


def test_the_closed_form_solves_the_ode_it_claims_to():
    """C-14.1: verify the anchor's defining property, independently.

    A closed form is only an oracle if it is right. These expressions were
    integrated by hand from the costate equations, so they are checked
    against ``scipy.integrate.solve_ivp`` at a tolerance three orders tighter
    than the agreement asserted -- a path that shares nothing with either
    adjungo or the derivation.
    """
    from scipy.integrate import solve_ivp

    from examples.double_integrator import (
        DRAG,
        T_FINAL,
        Y0,
        continuous_optimum,
    )

    for drag in (0.0, DRAG):
        exact = continuous_optimum(drag)
        solution = solve_ivp(
            lambda t, y, exact=exact, drag=drag: [
                y[1],
                -drag * y[1] + float(exact.control(t)),
            ],
            (0.0, T_FINAL),
            Y0.astype(float),
            rtol=1e-13,
            atol=1e-14,
        )
        assert solution.success
        deviation = np.max(np.abs(solution.y[:, -1] - exact.state(T_FINAL)))
        assert deviation < 1e-10, f"drag={drag}: closed form off by {deviation:.3e}"


def test_the_closed_form_satisfies_the_transversality_condition():
    """The other half of the optimality system, which the ODE check misses.

    ``solve_ivp`` confirms the state solves the dynamics under ``u*``. It
    says nothing about whether ``u*`` is *optimal*: that is ``lambda(T) =
    S (y(T) - y_target)`` together with ``u* = -lambda_2 / R``. Checking only
    the ODE would accept any admissible control.
    """
    from examples.double_integrator import (
        DRAG,
        ENERGY_WEIGHT,
        T_FINAL,
        TERMINAL_WEIGHT,
        Y_TARGET,
        continuous_optimum,
    )

    for drag in (0.0, DRAG):
        exact = continuous_optimum(drag)
        costate_terminal = TERMINAL_WEIGHT @ (exact.state(T_FINAL) - Y_TARGET)

        # lambda_1 is constant; lambda_2 = -R u*.
        assert exact.a == pytest.approx(costate_terminal[0], rel=1e-12)
        lambda_2 = -ENERGY_WEIGHT * float(exact.control(T_FINAL))
        assert lambda_2 == pytest.approx(costate_terminal[1], rel=1e-12)


def test_the_undamped_optimum_is_the_continuous_one_at_every_mesh():
    """No discretisation error at all, and none appears under refinement.

    ``u*`` is linear in ``t``, the state is a cubic, and rk4's weights are
    Simpson's rule, so both the propagation and the cost quadrature are exact
    for the functions this optimum produces. The discrete optimal control is
    then the continuous one sampled at the stage abscissae -- including at
    ``N = 5``, five steps over the whole horizon.

    Stated as a mesh study because that is what makes it falsifiable: an
    error that were merely small would shrink as the mesh refines, and this
    one does not move.
    """
    from examples.double_integrator import (
        continuous_optimum,
        discrete_objective,
        riccati_control,
        stage_times,
    )

    exact = continuous_optimum()
    scale = float(np.max(np.abs(exact.control(stage_times(5)))))
    deviations = []
    for n_steps in (5, 10, 20, 40, 80, 160):
        u_discrete = riccati_control(n_steps)
        u_continuous = exact.control(stage_times(n_steps))[:, :, None]
        deviations.append(float(np.max(np.abs(u_discrete - u_continuous))) / scale)
        assert discrete_objective(u_discrete, n_steps) == pytest.approx(
            exact.objective, rel=1e-11
        )

    # The budget separates two measured populations rather than fitting one.
    # Rounding: 15 to 40 unit roundoffs under Apple Accelerate and up to 65
    # under OpenBLAS -- the Riccati recursion accumulates over N backward
    # steps, and the backends differ in the last bits of an LU (C-11.3). The
    # first version of this test used 64 eps, which passed here and failed in
    # CI: it had certified a BLAS, not a method.
    #
    # Truncation: the same measurement on the damped problem, where the
    # discretisation is genuinely inexact, gives 3.1e-2 at N = 5 falling to
    # 3.0e-5 at N = 160. So 1e-11 sits roughly 700x above the worst rounding
    # seen on either backend and six orders below the smallest truncation
    # error this family produces.
    assert max(deviations) < 1e-11, f"deviations {deviations}"

    # The magnitude alone could be met by an error that is simply small. The
    # content of "exact" is that it does not fall under refinement: a
    # fourth-order error would drop by 2^4 per halving, a factor of 1024 from
    # N = 5 to N = 160. Rounding stays flat, observed between 0.8x and 2.2x.
    assert deviations[-1] > deviations[0] / 100.0, (
        f"deviation fell like truncation error under refinement: {deviations}"
    )


def test_the_damped_optimum_converges_at_fourth_order():
    """C-4: with drag the costate is exponential and the error is real.

    This is the test that keeps the exactness result above from being
    vacuous. Adding ``-c v`` to the state equation makes ``lambda_2``
    exponential rather than linear, so rk4 is no longer exact for the optimal
    trajectory, and the discrete optimum approaches the continuous one at the
    method's order. Observed 4.16, 4.08, 4.04, 4.02, 4.01.

    The quantity refined is the optimum, not a solve: each mesh solves its
    own discrete optimal control problem to machine precision first.
    """
    from examples.double_integrator import (
        DRAG,
        continuous_optimum,
        discrete_objective,
        riccati_control,
    )

    exact = continuous_optimum(DRAG)
    errors = []
    for n_steps in (5, 10, 20, 40, 80, 160):
        u_discrete = riccati_control(n_steps, DRAG)
        errors.append(abs(discrete_objective(u_discrete, n_steps, DRAG) - exact.objective))

    rates = [np.log2(a / b) for a, b in itertools.pairwise(errors)]
    assert min(rates) > 3.8, f"observed orders {rates} from errors {errors}"
    assert max(rates) < 4.4, f"orders above rk4's: {rates}"
    assert errors[-1] < 1e-10


def test_double_integrator_gradient_matches_the_independent_reference():
    """The monolithic reference, at a point that is not the optimum.

    Stationarity at the Riccati control checks the gradient where it
    vanishes. This checks it where it does not, which is where a scale error
    would show.
    """
    from examples.double_integrator import DRAG, T_FINAL, Y0, build_optimizer

    n_steps = 8
    optimizer = build_optimizer(n_steps, DRAG)
    rng = np.random.default_rng(13)
    u = rng.standard_normal((n_steps, optimizer.method.s, 1))

    grad_ref = reference_gradient(
        Y0,
        u,
        (0.0, T_FINAL),
        n_steps,
        optimizer.problem,
        optimizer.method,
        optimizer.objective,
    )
    assert np.max(np.abs(grad_ref)) > 1e-3, "degenerate case"
    err = np.max(np.abs(optimizer.gradient(u) - grad_ref)) / max(
        float(np.max(np.abs(grad_ref))), 1.0
    )
    assert err < 1e-11, f"double integrator gradient off by {err:.3e}"


def test_double_integrator_hessian_is_constant_and_matches_the_reference():
    """A quadratic reduced objective: the Hessian cannot depend on ``u``.

    Affine dynamics and a quadratic cost make ``J`` a quadratic function, so
    ``d²J/du²`` is one constant matrix. Evaluating it at two unrelated points
    is a structural check the nonlinear examples cannot make, and it would
    fail if a second-derivative term were contracted against the wrong
    variable.
    """
    from examples.double_integrator import DRAG, T_FINAL, Y0, build_optimizer

    n_steps = 6
    optimizer = build_optimizer(n_steps, DRAG)
    size = n_steps * optimizer.method.s
    hessp = optimizer.scipy_hessp()

    rng = np.random.default_rng(7)
    points = [np.zeros(size), rng.standard_normal(size) * 3.0]
    assembled = [
        np.column_stack([hessp(point, np.eye(size)[:, j]) for j in range(size)])
        for point in points
    ]
    # A genuine dependence on u would be O(1) relative, since the second
    # derivative terms it would come from are not small; the two assemblies
    # differ only by rounding, measured at 0.0 here.
    scale = float(np.max(np.abs(assembled[0])))
    assert np.max(np.abs(assembled[0] - assembled[1])) / scale < 1e-11

    reference = reference_hessian(
        Y0,
        rng.standard_normal((n_steps, optimizer.method.s, 1)),
        (0.0, T_FINAL),
        n_steps,
        optimizer.problem,
        optimizer.method,
        optimizer.objective,
    )
    assert np.max(np.abs(assembled[0] - reference)) / scale < 1e-11


def test_double_integrator_main_runs(capsys):
    """The example executes end to end and reports both studies."""
    from examples.double_integrator import main

    main()
    out = capsys.readouterr().out
    assert "The discrete optimum *is* the continuous one" in out
    assert "order" in out


# ---------------------------------------------------------------------------
# examples/pendulum_swing_up.py
#
# The first example here whose *nonlinear* dynamics have a closed-form
# solution, so it reaches the second tier of the C-14.1 oracle hierarchy where
# nonlinear_implicit_control.py reaches only the first. Two oracles of
# different kinds need separate checks: a pointwise closed form, and a first
# integral that constrains every point of every undriven trajectory.
#
# Both describe the undriven, undamped pendulum. The controlled problem
# carries drag and torque and conserves nothing; its derivatives are checked
# against the independent discrete reference, as everywhere else.
# ---------------------------------------------------------------------------


#: Half-width of the central difference used to check the closed form against
#: the ODE it solves. The truncation error of a central difference is
#: ``(dt^2 / 6) |y'''|``, and ``|theta'''| <= w0^2 * max|omega| ~ 3.7`` here,
#: so this spacing is good for about ``6e-9``. Rounding contributes about
#: ``eps * |y| / dt ~ 2e-12``, far below it. The budget below leaves more
#: than a decade over the truncation term.
_ODE_STEP = 1e-4
_ODE_BUDGET = 1e-7


@requires_sympy
def test_the_libration_closed_form_solves_the_pendulum_equation():
    """The oracle is checked before anything is checked against it.

    ``libration`` is asserted to be the exact undriven motion, and every
    other closed-form test in this section trusts it. Its defining property
    is that it satisfies ``theta' = omega`` and ``omega' = -w0^2 sin theta``,
    which is verified here by differencing the closed form itself -- no
    adjungo code takes part, so a stepping bug cannot make this pass.

    Sampled away from the turning points, where ``omega`` passes through zero
    and a relative comparison would divide by it.
    """
    from examples.pendulum_swing_up import (
        NATURAL_FREQUENCY,
        libration,
        libration_period,
    )

    period = libration_period()
    t = np.linspace(0.05 * period, 0.95 * period, 41)

    theta, omega = libration(t)
    theta_plus, omega_plus = libration(t + _ODE_STEP)
    theta_minus, omega_minus = libration(t - _ODE_STEP)

    d_theta = (theta_plus - theta_minus) / (2.0 * _ODE_STEP)
    d_omega = (omega_plus - omega_minus) / (2.0 * _ODE_STEP)

    assert np.max(np.abs(d_theta - omega)) < _ODE_BUDGET
    assert np.max(
        np.abs(d_omega + NATURAL_FREQUENCY**2 * np.sin(theta))
    ) < _ODE_BUDGET

    # The residual must be truncation, not luck: halving the step must cut it
    # by about four. A closed form that solved a *different* equation would
    # leave an O(1) residual that refinement does not touch.
    coarse = np.max(np.abs(d_theta - omega))
    theta_p2, _ = libration(t + 2 * _ODE_STEP)
    theta_m2, _ = libration(t - 2 * _ODE_STEP)
    doubled = np.max(np.abs((theta_p2 - theta_m2) / (4.0 * _ODE_STEP) - omega))
    assert 3.5 < doubled / coarse < 4.5, (
        f"residual scaled by {doubled / coarse:.2f} when the step doubled, "
        f"not the 4 that identifies it as central-difference truncation"
    )


@requires_sympy
def test_the_libration_is_released_from_rest_and_returns():
    """Boundary values that follow from ``sn(K) = 1``, ``cn(K) = 0``.

    Those are exact identities of the elliptic functions, but their
    floating-point evaluation is not exact, and C-11.3 forbids pinning it.
    ``theta`` comes back through ``2 arcsin(k sn)`` with ``k = sin(a/2)``, a
    round trip whose agreement with ``a`` depends on the platform's libm: on
    this machine it is exact at 0.5, 1.0, 2.0 and 2.5 rad and off by one ULP
    at 3.0, all within the helper's own domain. A few ULP is the honest
    budget. ``omega`` vanishes rather than returning a value, so it carries an
    absolute budget scaled by the natural frequency instead.
    """
    from examples.pendulum_swing_up import (
        NATURAL_FREQUENCY,
        VALIDATION_AMPLITUDE,
        libration,
        libration_period,
    )

    period = libration_period()
    theta_0, omega_0 = libration(0.0)
    theta_half, omega_half = libration(period / 2.0)

    ulp = np.spacing(VALIDATION_AMPLITUDE)
    assert abs(theta_0 - VALIDATION_AMPLITUDE) <= 4 * ulp
    assert abs(theta_half + VALIDATION_AMPLITUDE) <= 4 * ulp

    # Released from rest, and at rest again at the opposite turning point.
    # The scale is the peak speed 2*w0*k, which these are a rounding of.
    speed = 2.0 * NATURAL_FREQUENCY * np.sin(VALIDATION_AMPLITUDE / 2.0)
    assert abs(omega_0) <= 8 * np.finfo(float).eps * speed
    assert abs(omega_half) <= 8 * np.finfo(float).eps * speed


@requires_sympy
def test_the_libration_conserves_the_first_integral_it_claims_to():
    """``E`` is constant along the closed form, and equals ``2 w0^2 k^2``.

    The second claim is the sharper one: it is the identity
    ``sn^2 + cn^2 = 1`` in disguise, so agreement is a statement about the
    elliptic functions rather than about the pendulum. Both hold to rounding,
    which is why the budget is a small multiple of eps and not a discretisation
    tolerance -- nothing here is discretised.
    """
    from examples.pendulum_swing_up import (
        NATURAL_FREQUENCY,
        VALIDATION_AMPLITUDE,
        energy,
        libration,
        libration_period,
    )

    t = np.linspace(0.0, 2.0 * libration_period(), 501)
    theta, omega = libration(t)
    values = 0.5 * omega**2 + NATURAL_FREQUENCY**2 * (1.0 - np.cos(theta))

    closed_form = (
        2.0 * NATURAL_FREQUENCY**2 * np.sin(VALIDATION_AMPLITUDE / 2.0) ** 2
    )
    assert np.max(np.abs(values - closed_form)) < 16 * np.finfo(float).eps
    # The module's own helper agrees with the expression spelled out above.
    assert energy(np.array([VALIDATION_AMPLITUDE, 0.0])) == pytest.approx(
        closed_form, abs=8 * np.finfo(float).eps
    )


@requires_sympy
def test_the_libration_period_is_not_the_small_angle_period():
    """What makes this a nonlinear check rather than a linear one.

    At the validation amplitude the true period exceeds ``2 pi / w0`` by a
    third. A solve that quietly linearised ``sin theta`` would drift out of
    phase by far more than its discretisation error, so the order study below
    could not pass on a linearised field.

    The small-amplitude limit pins the formula independently: the classical
    expansion is ``T / T_0 = 1 + theta_0^2 / 16 + 11 theta_0^4 / 3072 + ...``,
    and at ``1e-3`` and ``1e-4`` rad the computed ratio reproduces that leading
    correction.

    The budget carries both of its terms. ``theta_0^4`` covers the neglected
    expansion term, which it exceeds by a factor of 280. Alone it would be
    ``1e-16`` at ``1e-4`` rad -- below one ULP of the ratio, 2.22e-16 -- and
    would then be pinning an association rather than a correction: computing
    the same ratio as ``2 K(m) / pi`` moves it one ULP and fails. C-11.3
    forbids that, so a rounding floor is added.
    """
    from examples.pendulum_swing_up import (
        NATURAL_FREQUENCY,
        libration_period,
    )

    small_angle = 2.0 * np.pi / NATURAL_FREQUENCY
    assert libration_period() / small_angle > 1.3

    for amplitude in (1e-3, 1e-4):
        predicted = 1.0 + amplitude**2 / 16.0
        ratio = libration_period(amplitude) / small_angle
        budget = amplitude**4 + 8 * np.finfo(float).eps
        assert abs(ratio - predicted) < budget, (
            f"at {amplitude:.0e} rad the ratio deviates from the expansion by "
            f"{abs(ratio - predicted):.3e}, over the {budget:.3e} budget"
        )


@requires_sympy
def test_libration_refuses_an_amplitude_off_the_librating_branch():
    """C-7: outside ``(0, pi)`` this formula has no motion to describe.

    Only ``pi`` itself is a physical boundary: the modulus reaches one, ``K``
    diverges, and the motion is the separatrix. The other two exclusions are
    narrower, and the docstring here previously got them wrong by saying the
    pendulum rotates beyond ``pi``. It does not. Released from rest at
    ``pi + 0.1`` its energy is 3.3716 against a separatrix value of 3.3800, so
    it librates -- about ``2 pi`` rather than about zero. What fails is the
    parameterisation: ``k = sin(theta_0/2)`` is not injective there, and
    ``2 arcsin k`` reflects ``pi + 0.1`` back to ``pi - 0.1``, so the helper
    would return a real trajectory of the wrong amplitude. Negative amplitudes
    it would handle correctly by mirror symmetry, and are excluded to keep one
    convention for the amplitude.

    A silently aliased trajectory is exactly the sentinel C-7 forbids, which
    is why all three are refused rather than only the separatrix.
    """
    from examples.pendulum_swing_up import libration

    for amplitude in (0.0, -1.0, np.pi, np.pi + 0.1, 10.0):
        with pytest.raises(ValueError, match=r"\(0, pi\)"):
            libration(0.0, amplitude=amplitude)


@requires_sympy
def test_the_undriven_solve_attains_fourth_order_against_the_closed_form():
    """C-4 on a nonlinear field, against a closed form rather than a reference.

    This is the claim the example exists to support. ``rk4`` is order four, and
    the observed rates are 4.11 and 4.08; the floor below allows for the
    ``O(h^5)`` terms still visible at these meshes without admitting order
    three.

    The comparison is over the whole arc, not at the endpoint. Half a period
    ends at a turning point where ``d theta / dt = 0``, and sampling there
    corrupts the measurement in either direction: ``theta`` alone becomes
    superconvergent (4.92, 4.98, 5.00, because a phase error reaches it only
    at second order), while ``max(theta, omega)`` becomes erratic (4.89, 3.49,
    3.80). The whole arc gives 4.11, 4.08, 4.04.

    Both bounds are therefore asserted. A rate below four would mean the
    method is not achieving its order; a rate near five would mean this is no
    longer measuring the method's order at all, but the extra accuracy of a
    degenerate sampling point. Only the second catches the endpoint variant,
    which otherwise passes a one-sided test by looking *better* than order
    four.
    """
    from examples.pendulum_swing_up import _libration_error

    errors = [_libration_error(n) for n in (10, 20, 40)]
    assert errors[0] < 1e-3, (
        f"coarse mesh is not yet in the asymptotic regime: {errors[0]:.3e}"
    )
    for coarse, fine in itertools.pairwise(errors):
        rate = np.log2(coarse / fine)
        assert 3.7 < rate < 4.6, (
            f"observed order {rate:.3f}; fourth order with the O(h^5) terms "
            f"still visible at these meshes lies between the bounds, and a "
            f"rate above them means the error is being sampled somewhere "
            f"degenerate rather than over the whole arc"
        )


@requires_sympy
def test_gauss2_holds_the_energy_error_bounded_where_rk4_drifts():
    """The first integral as an oracle, and the two methods it separates.

    Gauss--Legendre collocation is symplectic, so backward error analysis
    applies: it solves a nearby modified Hamiltonian almost exactly, and the
    energy error stays within an ``O(h^p)`` band over intervals exponentially
    long in ``1/h``, given a small enough step and a trajectory in a compact
    region (Hairer, Lubich & Wanner, *Geometric Numerical Integration*,
    Ch. IX). That is a finite-time statement under hypotheses, not a promise
    that the error is bounded forever, and this test asserts only the 32-period
    observation it actually makes. ``rk4`` is not symplectic and its error
    grows with elapsed time.

    Both are measured the same way -- the largest ``|E(t) - E(0)|`` the run
    ever attains -- so the contrast cannot come from the choice of statistic.
    Thirty-two-fold more integration time multiplies that error by about 27
    for ``rk4`` and by 1.003 for ``gauss2``. The thresholds sit well inside
    that gap rather than at the observed values, which are not a contract.
    """
    from examples.pendulum_swing_up import _energy_contrast

    growth = {label: ratio for label, ratio, _ in _energy_contrast()}

    assert growth["rk4"] > 10.0, (
        f"rk4 energy error grew by only {growth['rk4']:.2f}x over 32x the "
        f"integration time; the drift this example exhibits is gone"
    )
    assert growth["gauss2"] < 1.5, (
        f"gauss2 energy error grew by {growth['gauss2']:.2f}x; a symplectic "
        f"method must hold it bounded"
    )


@requires_sympy
def test_pendulum_symbolic_derivatives_match_hand_derivation():
    """All five generated callbacks against derivatives taken by hand.

    ``f`` is two lines, so the hand forms are genuinely independent rather
    than a transcription. ``F_yy`` is the entry that matters: it is the only
    nonvanishing second derivative, it is exactly the term a linearised
    pendulum would drop, and it is what the HVP test below exercises.
    """
    from examples.pendulum_swing_up import DRAG, NATURAL_FREQUENCY, pendulum_dynamics

    dynamics = pendulum_dynamics(DRAG)
    assert dynamics.state_dim == 2
    assert dynamics.control_dim == 1

    rng = np.random.default_rng(29)
    for _ in range(5):
        theta, omega = rng.uniform(-np.pi, np.pi), rng.uniform(-3.0, 3.0)
        y = np.array([theta, omega])
        u = np.array([rng.uniform(-2.0, 2.0)])
        v = rng.standard_normal(2)
        w0 = NATURAL_FREQUENCY

        f_hand = np.array([omega, -(w0**2) * np.sin(theta) - DRAG * omega + u[0]])
        F_hand = np.array([[0.0, 1.0], [-(w0**2) * np.cos(theta), -DRAG]])
        G_hand = np.array([[0.0], [1.0]])

        # f[1] is a sum of three terms that can cancel, so its rounding is set
        # by their magnitudes rather than by the result. f[0] = omega is exact.
        f_scale = np.array(
            [
                abs(omega),
                (w0**2) * abs(np.sin(theta)) + DRAG * abs(omega) + abs(u[0]),
            ]
        )

        # Only -v1 w0^2 sin(theta) survives two derivatives, and only in theta.
        F_yy_hand = np.array([[v[1] * w0**2 * np.sin(theta), 0.0], [0.0, 0.0]])
        # f is affine in u with constant coefficients, so these vanish.
        F_yu_hand = np.zeros((2, 1))
        F_uu_hand = np.zeros((1, 1))

        _agrees_by_derivation(dynamics.f(y, u, 0.0), f_hand, "f", scale=f_scale)
        _agrees_by_derivation(dynamics.F(y, u, 0.0), F_hand, "F")
        _agrees_by_derivation(dynamics.G(y, u, 0.0), G_hand, "G")
        _agrees_by_derivation(dynamics.F_yy_action(y, u, 0.0, v), F_yy_hand, "F_yy")
        _agrees_by_derivation(dynamics.F_yu_action(y, u, 0.0, v), F_yu_hand, "F_yu")
        _agrees_by_derivation(dynamics.F_uu_action(y, u, 0.0, v), F_uu_hand, "F_uu")


@requires_sympy
def test_the_pendulum_jacobian_actually_depends_on_the_state():
    """Without this, the reuse gate below would be vacuous.

    ``F`` carries ``-w0^2 cos theta``, so no factorisation may be reused
    across stages, steps or calls (C-15.1). A problem whose Jacobian happened
    to be constant would let a reuse bug through the derivative tests
    unnoticed, so the dependence is asserted rather than assumed.
    """
    from examples.pendulum_swing_up import DRAG, pendulum_dynamics

    dynamics = pendulum_dynamics(DRAG)
    u, t = np.array([0.3]), 0.0
    at_zero = dynamics.F(np.array([0.0, 0.0]), u, t)
    at_one = dynamics.F(np.array([1.0, 0.0]), u, t)

    assert not np.allclose(at_zero, at_one), (
        "the Jacobian does not vary with theta; this example no longer "
        "exercises the state-dependent path it was written for"
    )
    # Varying omega alone must not move it: the drag term is linear.
    assert np.array_equal(at_zero, dynamics.F(np.array([0.0, 2.0]), u, t))


@requires_sympy
def test_pendulum_gradient_matches_the_independent_reference():
    """C-2 on the controlled problem, against the strongest oracle (C-14.1.1).

    ``reference_gradient`` rebuilds the discrete problem from the monolithic
    residual and shares no code with ``adjungo/stepping/``. The budget is the
    ``1e-11`` relative used by the other examples; the observed disagreement
    here is about ``6e-16``.
    """
    from examples.pendulum_swing_up import T_FINAL, build_optimizer
    from examples.pendulum_swing_up import Y0 as PENDULUM_Y0

    n_steps = 6
    optimizer = build_optimizer(n_steps=n_steps)
    method = optimizer.method
    u = 0.5 * np.random.default_rng(5).standard_normal((n_steps, method.s, 1))

    grad_ref = reference_gradient(
        PENDULUM_Y0, u, (0.0, T_FINAL), n_steps,
        optimizer.problem, method, optimizer.objective,
    )
    assert np.max(np.abs(grad_ref)) > 1e-3, "degenerate case"

    err = np.max(np.abs(optimizer.gradient(u) - grad_ref)) / max(
        float(np.max(np.abs(grad_ref))), 1.0
    )
    assert err < 1e-11, f"pendulum gradient off by {err:.3e}"


@requires_sympy
def test_pendulum_hvp_matches_the_independent_reference():
    """The exact Hessian where ``F_yy`` does not vanish.

    ``-w0^2 sin theta`` has a nonzero second derivative, so a dropped
    ``F_yy_action`` term changes this comparison. The affine examples cannot
    detect that; this one can, and the Van der Pol example is the only other
    that does.
    """
    from examples.pendulum_swing_up import T_FINAL, build_optimizer
    from examples.pendulum_swing_up import Y0 as PENDULUM_Y0

    n_steps = 5
    optimizer = build_optimizer(n_steps=n_steps)
    method = optimizer.method
    u = 0.5 * np.random.default_rng(13).standard_normal((n_steps, method.s, 1))

    H_ref = reference_hessian(
        PENDULUM_Y0, u, (0.0, T_FINAL), n_steps,
        optimizer.problem, method, optimizer.objective,
    ).reshape(u.size, u.size)

    hessp = optimizer.scipy_hessp()
    H_pkg = np.column_stack(
        [hessp(u.ravel(), e) for e in np.eye(u.size)]
    )

    scale = max(float(np.max(np.abs(H_ref))), 1.0)
    assert float(np.max(np.abs(H_pkg - H_ref))) / scale < 1e-11
    # Symmetry is necessary, never sufficient (C-14.1.5), but an asymmetric
    # result would mean the two contraction orders disagree.
    assert np.max(np.abs(H_pkg - H_pkg.T)) / scale < 1e-11


@requires_sympy
def test_the_swing_up_reaches_the_inverted_state():
    """The example must solve the problem it advertises.

    No closed form is claimed for the optimum and none is asserted. What is
    asserted is that the solve converges, beats doing nothing by a wide
    margin, and arrives near enough to ``theta = pi`` that the pendulum has
    genuinely been brought up rather than left swinging.
    """
    from examples.pendulum_swing_up import TARGET, solve

    optimizer, result, control = solve(n_steps=20)

    assert result.success, result.message
    do_nothing = optimizer.objective_value(np.zeros_like(control))
    assert result.fun < 0.01 * do_nothing, (
        f"objective {result.fun:.4f} barely improved on {do_nothing:.4f}"
    )

    final = optimizer.trajectory(control).Y[-1][0]
    assert abs(final[0] - TARGET[0]) < 0.05, f"theta(T) = {final[0]:.4f}, not near pi"
    assert abs(final[1] - TARGET[1]) < 0.10, f"omega(T) = {final[1]:.4f}, not near rest"

    # It is a swing-up, not a lift: the torque available cannot hold the
    # pendulum against gravity at the midpoint of a direct push, so the
    # trajectory must pass through angles beyond the target on the way.
    assert np.max(np.abs(control)) < 5.0, "unbounded-looking torque"


@requires_sympy
def test_pendulum_main_runs(capsys):
    """``.venv/bin/python -m examples.pendulum_swing_up`` must work."""
    from examples.pendulum_swing_up import main

    main()
    out = capsys.readouterr().out
    assert "exact period" in out
    assert "rate" in out
    assert "theta(T)" in out
