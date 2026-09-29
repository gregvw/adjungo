"""Validation of the C-14.5 certification problem itself.

A certification problem is evidence only if its own callbacks are right. This
module establishes that before any partitioned method exists to be certified,
by two independent routes to every quantity:

1. **From the Hamiltonian.** ``f`` is checked against a symbolic ``J grad H``
   assembled from ``T`` and ``V``, where ``J`` is the canonical symplectic
   matrix. Nothing in that path reads :meth:`ShakenLatticeTrap.f`.
2. **From ``f``.** All five derivative callbacks are checked against
   :class:`examples.symbolic.SymbolicDynamics`, which differentiates the
   sympy expression for ``f`` and contracts as C-9.4 specifies. That is the
   pattern ``tests/test_examples.py`` already applies to the rocket, pendulum
   and Zermelo problems.

The separability structure C-8.4 depends on -- ``f^q`` a function of ``p``
alone, ``f^p`` a function of ``(q, u, t)`` alone -- is asserted directly on the
Jacobian rather than inferred from the model, because it is the hypothesis a
partitioned method is built on and an ordinary-looking Hamiltonian can fail it.

Finally, the problem is put through the certified ``r = 1`` families and its
gradients and HVPs compared with the C-14.1 tier-1 reference. That is not a
test of the partitioned family, which does not exist; it establishes that the
problem is admissible and well conditioned on the routes that do exist, so a
future disagreement can be attributed to the new method rather than to the
fixture.

**Nothing here certifies a partitioned method.** C-6.1 is unchanged and
continues to list the family as not supported.
"""

from __future__ import annotations

import numpy as np
import pytest

from adjungo.methods.runge_kutta import (
    gauss2,
    implicit_midpoint,
    rk4,
    sdirk3,
)
from adjungo.optimization.interface import GLMOptimizer
from adjungo.validation import reference_gradient, reference_hessian
from tests.prk_problems import TRANSPORT_TARGET, ShakenLatticeTrap
from tests.problems import FullCostObjective, make_controls

sp = pytest.importorskip("sympy", reason="the symbolic cross-check needs sympy")

T_SPAN = (0.2, 1.4)
Y0 = np.array([0.35, -0.2, 0.15, 0.4])

#: Evaluation points. Distinct in every component, with momenta that reach
#: past the band's inflection so that ``cos p`` changes sign, and one point
#: with ``u_2 < 0`` so that a dropped sign in ``-2 u_2 (q - d)`` shows.
SAMPLES = [
    (np.array([0.35, -0.2, 0.15, 0.4]), np.array([0.3, 0.7]), 0.2),
    (np.array([-0.9, 1.3, 2.1, -1.8]), np.array([-0.45, 1.2]), 0.85),
    (np.array([1.7, 0.6, -2.4, 0.9]), np.array([0.8, -0.6]), 1.4),
    (np.array([0.05, -1.1, 3.0, 2.6]), np.array([-1.3, 0.25]), 2.75),
]


def _objective() -> FullCostObjective:
    return FullCostObjective(nx=4, nu=2, y_target=TRANSPORT_TARGET)


# ---------------------------------------------------------------------------
# Route 1: f from the Hamiltonian
# ---------------------------------------------------------------------------


def _symbolic_hamiltonian(problem: ShakenLatticeTrap):
    """``(H, y_symbols, u_symbols, t)`` built from the model, not from ``f``.

    Written out here rather than imported so that the comparison has two
    authors' worth of independence: this is ``T + V`` transcribed from the
    module docstring's displayed formulae.
    """
    q1, q2, p1, p2 = sp.symbols("q1 q2 p1 p2")
    u1, u2 = sp.symbols("u1 u2")
    t = sp.Symbol("t")

    d1 = problem.delta * sp.sin(problem.omega_drive * t)
    r1, r2 = q1 - d1, q2
    kinetic = problem.J[0] * (1 - sp.cos(p1)) + problem.J[1] * (1 - sp.cos(p2))
    potential = (
        sp.Rational(1, 2) * (problem.kappa + u2**2) * (r1**2 + r2**2)
        + sp.Rational(1, 4) * problem.alpha * (q1**2 + q2**2) ** 2
        - u1 * q1
    )
    return kinetic + potential, [q1, q2, p1, p2], [u1, u2], t


def _symbolic_f(problem: ShakenLatticeTrap):
    """``f = J grad H`` symbolically, with ``J`` the canonical form."""
    H, y_syms, u_syms, t = _symbolic_hamiltonian(problem)
    q_syms, p_syms = y_syms[:2], y_syms[2:]
    rows = [sp.diff(H, p) for p in p_syms] + [-sp.diff(H, q) for q in q_syms]
    return sp.Matrix(rows), y_syms, u_syms, t


@pytest.mark.parametrize("y,u,t", SAMPLES)
def test_f_is_the_canonical_hamiltonian_field(y, u, t):
    """``f = (dH/dp, -dH/dq)``, against a symbolic ``H`` that never reads ``f``.

    A sign error in the ``p`` block is the classic way to write down a
    Hamiltonian system that integrates plausibly and conserves nothing; it
    would not be caught by any derivative check, because every derivative
    below is taken *of* ``f``.
    """
    problem = ShakenLatticeTrap()
    f_sym, y_syms, u_syms, t_sym = _symbolic_f(problem)
    subs = dict(zip(y_syms, y, strict=True))
    subs.update(dict(zip(u_syms, u, strict=True)))
    subs[t_sym] = t

    expected = np.array(
        [float(expr.subs(subs)) for expr in f_sym], dtype=float
    )
    # Both sides evaluate the same elementary formulae; the gap is rounding
    # in sin/cos and a few multiplies, not method error.
    assert problem.f(y, u, t) == pytest.approx(expected, rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("y,u,t", SAMPLES)
def test_the_hamiltonian_agrees_with_its_kinetic_and_potential_parts(y, u, t):
    """``H = T(p) + V(q, u, t)`` -- separability as an identity, not a label."""
    problem = ShakenLatticeTrap()
    H_sym, y_syms, u_syms, t_sym = _symbolic_hamiltonian(problem)
    subs = dict(zip(y_syms, y, strict=True))
    subs.update(dict(zip(u_syms, u, strict=True)))
    subs[t_sym] = t

    assert problem.hamiltonian(y, u, t) == pytest.approx(
        float(H_sym.subs(subs)), rel=1e-14, abs=1e-15
    )
    assert problem.hamiltonian(y, u, t) == pytest.approx(
        problem.kinetic(y[2:]) + problem.potential(y[:2], u, t),
        rel=1e-14,
        abs=1e-15,
    )


# ---------------------------------------------------------------------------
# Route 2: the five derivative callbacks from f
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def symbolic_twin():
    """``SymbolicDynamics`` differentiating the same ``f``."""
    from examples.symbolic import SymbolicDynamics

    problem = ShakenLatticeTrap()
    f_sym, y_syms, u_syms, t_sym = _symbolic_f(problem)
    return SymbolicDynamics(f_sym, y_syms, u_syms, t_sym)


@pytest.mark.parametrize("y,u,t", SAMPLES)
def test_hand_written_jacobians_match_symbolic_differentiation(
    symbolic_twin, y, u, t
):
    """``F`` and ``G``, by hand and by sympy."""
    problem = ShakenLatticeTrap()
    assert problem.F(y, u, t) == pytest.approx(
        symbolic_twin.F(y, u, t), rel=1e-13, abs=1e-14
    )
    assert problem.G(y, u, t) == pytest.approx(
        symbolic_twin.G(y, u, t), rel=1e-13, abs=1e-14
    )


@pytest.mark.parametrize("y,u,t", SAMPLES)
def test_hand_written_contracted_hessians_match_symbolic_differentiation(
    symbolic_twin, y, u, t
):
    """The three contracted second derivatives, by hand and by sympy.

    ``v`` has a distinct nonzero entry per component: contracting against a
    uniform or sparse vector is how a wrong index in a contraction survives.
    """
    problem = ShakenLatticeTrap()
    v = np.array([0.9, -1.4, 0.6, 2.1])

    assert problem.F_yy_action(y, u, t, v) == pytest.approx(
        symbolic_twin.F_yy_action(y, u, t, v), rel=1e-13, abs=1e-14
    )
    assert problem.F_yu_action(y, u, t, v) == pytest.approx(
        symbolic_twin.F_yu_action(y, u, t, v), rel=1e-13, abs=1e-14
    )
    assert problem.F_uu_action(y, u, t, v) == pytest.approx(
        symbolic_twin.F_uu_action(y, u, t, v), rel=1e-13, abs=1e-14
    )


# ---------------------------------------------------------------------------
# The structure C-8.4 requires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y,u,t", SAMPLES)
def test_the_state_jacobian_is_block_anti_diagonal(y, u, t):
    """C-8.4's domain: ``f^q`` depends on ``p`` alone, ``f^p`` on ``q`` alone.

    Exact zeros, not a tolerance. These blocks are structurally absent rather
    than numerically small, and a partitioned method's correctness rests on
    their being absent; a tolerance here would admit a problem outside the
    domain (C-8.3's spirit, applied to the problem rather than the tableau).
    """
    F = ShakenLatticeTrap().F(y, u, t)
    n_q = ShakenLatticeTrap.n_q
    assert np.array_equal(F[:n_q, :n_q], np.zeros((n_q, n_q))), "df^q/dq != 0"
    assert np.array_equal(F[n_q:, n_q:], np.zeros((n_q, n_q))), "df^p/dp != 0"

    # And the two surviving blocks are not themselves zero, or the assertion
    # above would be satisfied by a problem with no dynamics at all.
    assert np.any(F[:n_q, n_q:] != 0.0)
    assert np.any(F[n_q:, :n_q] != 0.0)


@pytest.mark.parametrize("y,u,t", SAMPLES)
def test_the_control_enters_only_the_momentum_block(y, u, t):
    """``f^q = dT/dp`` carries no control, because ``T`` does not."""
    G = ShakenLatticeTrap().G(y, u, t)
    n_q = ShakenLatticeTrap.n_q
    assert np.array_equal(G[:n_q], np.zeros((n_q, G.shape[1])))
    assert np.any(G[n_q:] != 0.0)


def test_every_structural_feature_C14_requires_is_present():
    """The coverage claim in the module docstring, asserted rather than stated.

    Each of these was chosen to close a way a derivative defect can hide. A
    later simplification of the model that dropped one would leave the
    docstring's justification standing over a problem that no longer supports
    it.
    """
    problem = ShakenLatticeTrap()
    y, u, t = SAMPLES[1]
    v = np.array([0.9, -1.4, 0.6, 2.1])

    assert problem.state_dim != problem.control_dim
    assert problem.state_dim > 1 and problem.control_dim > 1

    # The Jacobian varies with the state *and* with the control.
    y2 = y + np.array([0.3, -0.2, 0.25, 0.1])
    u2 = u + np.array([0.4, 0.3])
    assert not np.allclose(problem.F(y, u, t), problem.F(y2, u, t))
    assert not np.allclose(problem.F(y, u, t), problem.F(y, u2, t))

    # All three contracted Hessians are nonzero, in both partitions for F_yy.
    assert np.any(problem.F_yy_action(y, u, t, v)[:2, :2] != 0.0)
    assert np.any(problem.F_yy_action(y, u, t, v)[2:, 2:] != 0.0)
    assert np.any(problem.F_yu_action(y, u, t, v) != 0.0)
    assert np.any(problem.F_uu_action(y, u, t, v) != 0.0)

    # f depends explicitly on t, and the two axes are distinguishable.
    assert not np.allclose(problem.f(y, u, t), problem.f(y, u, t + 0.37))
    assert problem.J[0] != problem.J[1]

    # The band is traversed past its inflection somewhere in the samples, so
    # d2T/dp2 changes sign and the kinetic block is not effectively quadratic.
    curvatures = [np.cos(sample[0][2]) for sample in SAMPLES]
    assert min(curvatures) < 0.0 < max(curvatures)


# ---------------------------------------------------------------------------
# Admissibility on the routes that exist today
# ---------------------------------------------------------------------------

CERTIFIED = [
    pytest.param(rk4, 1e-11, id="rk4_explicit"),
    pytest.param(implicit_midpoint, 1e-8, id="implicit_midpoint_sdirk"),
    pytest.param(sdirk3, 1e-8, id="sdirk3"),
    pytest.param(gauss2, 1e-8, id="gauss2_fully_implicit"),
]


def _rel_err(a, b):
    """Error relative to ``max(|b|_inf, 1)`` (clause C-3)."""
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1.0))


@pytest.mark.parametrize("factory,rtol", CERTIFIED)
def test_the_gradient_matches_the_tier_one_reference(factory, rtol):
    """The problem is admissible on the certified families.

    Tolerances are the C-3 ones the oracle tests already use: 1e-11 for
    explicit methods, where both sides evaluate the same closed map, and
    1e-8 for implicit ones, where package and reference stop Newton at the
    C-5.1 threshold and land at different points within that radius.
    """
    method, N = factory(), 6
    problem, objective = ShakenLatticeTrap(), _objective()
    u = make_controls(N, method.s, problem.control_dim, seed=4)

    got = GLMOptimizer(problem, objective, method, T_SPAN, N, Y0).gradient(u)
    expected = reference_gradient(
        Y0, u, T_SPAN, N, problem, method, objective
    )
    assert _rel_err(got, expected) < rtol


@pytest.mark.parametrize("factory,rtol", CERTIFIED)
def test_the_hvp_matches_the_tier_one_reference(factory, rtol):
    """Second derivatives too: the three contracted Hessians are exercised."""
    method, N = factory(), 5
    problem, objective = ShakenLatticeTrap(), _objective()
    u = make_controls(N, method.s, problem.control_dim, seed=9)
    rng = np.random.default_rng(17)
    v = rng.standard_normal(u.shape)

    got = GLMOptimizer(
        problem, objective, method, T_SPAN, N, Y0
    ).hessian_vector_product(u, v)
    H = reference_hessian(Y0, u, T_SPAN, N, problem, method, objective)
    expected = (H @ v.reshape(-1)).reshape(v.shape)
    assert _rel_err(got, expected) < rtol


def test_the_problem_is_not_stiff_at_the_meshes_the_study_will_use():
    """A conditioning check, so a later disagreement is not the fixture's.

    If the stage matrices were near-singular at the step sizes C-14.5's
    refinement study uses, a partitioned method's first reported error would
    be dominated by conditioning rather than by the method. The spectral
    radius of ``h F`` is reported against the step, and the C-14.5 study's
    coarsest mesh is the demanding end.
    """
    problem = ShakenLatticeTrap()
    objective = _objective()
    method, N = rk4(), 60
    h = (T_SPAN[1] - T_SPAN[0]) / N
    u = make_controls(N, method.s, problem.control_dim, seed=1, scale=0.5)

    trajectory = GLMOptimizer(
        problem, objective, method, T_SPAN, N, Y0
    ).trajectory(u)

    worst = 0.0
    for n in range(N):
        y = trajectory.Y[n, 0]
        radius = float(
            np.max(np.abs(np.linalg.eigvals(h * problem.F(y, u[n, 0], 0.0))))
        )
        worst = max(worst, radius)

    # A budget, not a measurement: rk4's real-axis stability limit is 2.78
    # and its imaginary-axis limit 2*sqrt(2), so 0.5 leaves the coarsest mesh
    # of the C-14.5 study a factor of five inside the nearer of them.
    assert worst < 0.5, f"spectral radius of h*F reached {worst}"
    assert np.all(np.isfinite(trajectory.Y))
