"""Partitioned Runge-Kutta: structure and fixed-mesh derivative exactness.

Requirements 1 and 2 of NUMERICS.md C-14.5. They are kept in separate,
differently named tests because they fail for different reasons: a tableau can
be exactly symplectic and be executed by a stage solver with a transposed
index, and a stage solver can be exactly right while integrating coefficients
that are not symplectic at all. Requirement 3 -- continuous accuracy under
refinement -- is a claim about the *mesh* and lives in
``tests/test_partitioned_refinement.py``, per C-2/C-4 and the AGENTS.md rule
that a derivative test never refines.
"""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from adjungo.core.method import GLMethod, StageType, TableauDeclarationError
from adjungo.core.partitioned import (
    P,
    PartitionedMethod,
    Q,
    conjugate,
    symplectic_euler,
    verlet,
)
from adjungo.core.plan import DiscretizationPlan
from adjungo.methods.runge_kutta import explicit_euler, rk4
from adjungo.optimization.interface import GLMOptimizer
from adjungo.solvers.partitioned import (
    PartitionedStageSolver,
    SeparabilityViolation,
    partition_of,
)
from adjungo.validation import (
    reference_gradient,
    reference_hessian,
    reference_solve,
)
from tests.prk_problems import TRANSPORT_TARGET, ShakenLatticeTrap
from tests.problems import FullCostObjective, make_controls

METHODS = [
    pytest.param(symplectic_euler, id="symplectic_euler_s1"),
    pytest.param(verlet, id="verlet_s2"),
]

#: Exact coefficients, as rationals, written here independently of
#: ``adjungo.core.partitioned``. A symbolic check that read the package's own
#: ``Fraction`` tables would be checking them against themselves.
EXACT = {
    "symplectic_euler": (
        [[0]],
        [[1]],
        [1],
    ),
    "verlet": (
        [[0, 0], [Fraction(1, 2), Fraction(1, 2)]],
        [[Fraction(1, 2), 0], [Fraction(1, 2), 0]],
        [Fraction(1, 2), Fraction(1, 2)],
    ),
}

#: C-11.3 budget for a check on *stored floats*. Both tableaux here hold
#: exactly representable dyadic rationals, so every product and sum in the
#: paired condition is exact in binary64 and the residual is identically
#: zero -- but the budget is stated rather than the zero asserted, because
#: the next composition method (Blanes-Moan) cancels arithmetically instead
#: of structurally and will need one. Basis: the residual entries are sums of
#: at most ``2s`` products of coefficients bounded by 1, so the accumulated
#: rounding is bounded by ``2s * eps``.
STORED_COEFFICIENT_BUDGET = 4 * np.finfo(float).eps


def _case(method_factory, N=5, seed=5):
    method = method_factory()
    problem = ShakenLatticeTrap()
    objective = FullCostObjective(nx=4, nu=2, y_target=TRANSPORT_TARGET)
    y0 = np.array([0.3, -0.2, 0.1, 0.05])
    plan = DiscretizationPlan.uniform((0.2, 1.1), N, method)
    u = make_controls(N, method.s, problem.control_dim, seed=seed)
    return problem, objective, plan, y0, u


# =====================================================================
# Requirement 1 -- symplecticity, structural, no mesh
# =====================================================================


@pytest.mark.parametrize("method_factory", METHODS)
def test_paired_symplectic_condition_holds_symbolically(method_factory):
    """``D A^p + (A^q)ᵀ D - b bᵀ = 0`` in exact arithmetic (C-14.5 req. 1).

    Symbolic, on rationals transcribed here rather than read from the
    package. No mesh appears and none may: this is an identity in the
    coefficients.
    """
    A_q, A_p, b = EXACT[method_factory().name]
    Aq, Ap, bb = sp.Matrix(A_q), sp.Matrix(A_p), sp.Matrix(b)
    residual = sp.simplify(sp.diag(*bb) * Ap + Aq.T * sp.diag(*bb) - bb * bb.T)
    assert residual == sp.zeros(*residual.shape), (
        f"{method_factory().name} is not symplectic as a pair: {residual}"
    )


@pytest.mark.parametrize("method_factory", METHODS)
def test_conjugate_exchanges_the_pair_symbolically(method_factory):
    """``C(A^q) = A^p`` and ``C(A^p) = A^q`` exactly (C-8.4, C-14.5 req. 1).

    The conjugate is a stronger statement than the paired condition read
    entrywise, and it is checked only where the weights are nonzero, per
    C-8.4: ``C`` divides by ``b_i``.
    """
    A_q, A_p, b = EXACT[method_factory().name]
    Aq, Ap, bb = sp.Matrix(A_q), sp.Matrix(A_p), sp.Matrix(b)
    D = sp.diag(*bb)
    ones = sp.ones(bb.shape[0], 1)

    def C(M):
        return ones * bb.T - D.inv() * M.T * D

    assert sp.simplify(C(Aq) - Ap) == sp.zeros(*Aq.shape)
    assert sp.simplify(C(Ap) - Aq) == sp.zeros(*Aq.shape)


@pytest.mark.parametrize("method_factory", METHODS)
def test_the_pair_is_not_individually_self_conjugate(method_factory):
    """``C(A^q) != A^q``: the symplecticity is a property of the *pair*.

    Without this the previous test would pass for an unpartitioned method
    supplied twice, and the whole partitioned construction would be
    untested by it. ``gauss2`` is exactly that case and is checked below.
    """
    A_q, _, b = EXACT[method_factory().name]
    Aq, bb = sp.Matrix(A_q), sp.Matrix(b)
    D = sp.diag(*bb)
    ones = sp.ones(bb.shape[0], 1)
    conj = ones * bb.T - D.inv() * Aq.T * D
    assert sp.simplify(conj - Aq) != sp.zeros(*Aq.shape)


def test_a_symplectic_unpartitioned_tableau_is_self_conjugate():
    """``gauss2`` satisfies the paired condition with ``A^q = A^p = A``.

    The control for the previous test: it shows that the conjugate identity
    is not vacuous, and that "symplectic" for a partitioned pair reduces to
    the ordinary condition when the two arrays coincide.
    """
    r3 = sp.sqrt(3)
    A = sp.Matrix(
        [
            [sp.Rational(1, 4), sp.Rational(1, 4) - r3 / 6],
            [sp.Rational(1, 4) + r3 / 6, sp.Rational(1, 4)],
        ]
    )
    b = sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2)])
    D = sp.diag(*b)
    assert sp.simplify(D * A + A.T * D - b * b.T) == sp.zeros(2, 2)
    conj = sp.ones(2, 1) * b.T - D.inv() * A.T * D
    assert sp.simplify(conj - A) == sp.zeros(2, 2)


@pytest.mark.parametrize("method_factory", METHODS)
def test_stored_float_coefficients_meet_their_rounding_budget(method_factory):
    """The same identity on the *stored* floats, with a C-11.3 budget.

    A different measurement from the symbolic one, and it does not replace
    it: floating-point coefficients can satisfy this to rounding while
    describing a method that is not symplectic at all.
    """
    method = method_factory()
    residual = method.paired_symplectic_residual()
    assert np.max(np.abs(residual)) <= STORED_COEFFICIENT_BUDGET
    assert np.max(np.abs(conjugate(method.A_q, method.b) - method.A_p)) <= (
        STORED_COEFFICIENT_BUDGET
    )


@pytest.mark.parametrize("method_factory", METHODS)
def test_an_abscissa_is_not_a_partitioned_row_sum(method_factory):
    """``σ^q != c`` or ``σ^p != c`` (C-8.4).

    C-8.1 lets an ordinary tableau's abscissae be checked against ``A``'s row
    sums. A partitioned method has two row-sum vectors and one abscissa
    vector, and they do not agree. An implementation that recovered stage
    times from a row sum would be wrong here, so the disagreement is
    asserted rather than left as a remark.
    """
    method = method_factory()
    assert not np.allclose(method.sigma_q, method.c) or not np.allclose(
        method.sigma_p, method.c
    )


@pytest.mark.parametrize("method_factory", METHODS)
def test_the_adjoint_order_is_the_reverse_of_the_forward_order(
    method_factory,
):
    """The adjoint dependency graph is the forward graph with edges reversed.

    The backward sweeps run ``reversed(dependency_order)``, which is valid
    only if this holds. Proved by index swapping -- a forward edge
    ``Z_i^q <- Z_j^p`` carried by ``A^q_ij`` becomes an adjoint edge
    ``μ_j^p <- μ_i^q`` carried by the same entry -- and asserted here
    directly on the graph, because the proof is what the sweep relies on.
    """
    method = method_factory()
    s = method.s
    forward = {}
    adjoint = {}
    for i in range(s):
        forward[(Q, i)] = {(P, j) for j in range(s) if method.A_q[i, j] != 0.0}
        forward[(P, i)] = {(Q, j) for j in range(s) if method.A_p[i, j] != 0.0}
        # Λ_i^q reads μ_j^q through A^q[j, i]; μ_i^q reads Λ_i^p. So
        # μ_i^q depends on μ_j^p exactly where A^p[j, i] != 0.
        adjoint[(Q, i)] = {(P, j) for j in range(s) if method.A_p[j, i] != 0.0}
        adjoint[(P, i)] = {(Q, j) for j in range(s) if method.A_q[j, i] != 0.0}

    reversed_forward = {node: set() for node in forward}
    for node, deps in forward.items():
        for dep in deps:
            reversed_forward[dep].add(node)
    assert adjoint == reversed_forward

    order = list(method.dependency_order)
    assert set(order) == set(forward)
    seen: set = set()
    for node in order:
        assert forward[node] <= seen, f"{node} runs before its dependencies"
        seen.add(node)
    seen = set()
    for node in reversed(order):
        assert adjoint[node] <= seen, (
            f"{node} runs before its adjoint dependencies"
        )
        seen.add(node)


def test_an_implicit_partitioned_pair_is_refused_at_construction():
    """A cyclic dependency graph is refused, not deferred to a solve (C-6.2).

    ``A^q = A^p = [[1]]`` couples ``Z^q`` and ``Z^p`` to each other, which
    no substitution order resolves. The milestone supports substitution
    only, so the refusal belongs at construction.
    """
    with pytest.raises(NotImplementedError, match="implicit"):
        PartitionedMethod(
            A_q=np.array([[1.0]]),
            A_p=np.array([[1.0]]),
            b=np.array([1.0]),
            c=np.array([0.0]),
        )


# =====================================================================
# Requirement 2 -- fixed-mesh derivative exactness (C-2)
# =====================================================================
#
# The mesh is held fixed in every test below. Per AGENTS.md and C-2/C-4 a
# derivative discrepancy here is never explained by refinement.

#: The reference solves the whole discrete system by Newton and never uses
#: the dependency order, so agreement is evidence about that order too.
#: Budget basis: both sides evaluate the same ~60 floating-point operations
#: per stage in different groupings, so the relative difference is a few
#: ULPs amplified by the condition of the dense solve the reference takes.
CERTIFIED_RTOL = 1e-11


@pytest.mark.parametrize("method_factory", METHODS)
def test_forward_stages_match_the_independent_reference(method_factory):
    """Substitution reproduces the solution of the full coupled system.

    The reference assembles ``R(w, u) = 0`` for the whole trajectory and
    drives it to zero by Newton. It therefore knows nothing of the
    dependency order, and an order that resolved the stages in a sequence
    the equations do not support would disagree here.
    """
    problem, _, plan, y0, u = _case(method_factory)
    ref = reference_solve(y0=y0, u=u, problem=problem, plan=plan)
    opt = GLMOptimizer(
        problem,
        FullCostObjective(nx=4, nu=2, y_target=TRANSPORT_TARGET),
        y0=y0,
        plan=plan,
    )
    traj = opt.trajectory(u)
    assert np.max(np.abs(traj.Y - ref.Y)) < 1e-13
    assert np.max(
        np.abs(np.asarray(traj.Z).reshape(ref.Z.shape) - ref.Z)
    ) < 1e-13


@pytest.mark.parametrize("method_factory", METHODS)
def test_gradient_matches_the_independent_reference(method_factory):
    """Tier-1 oracle for ``dJ/du`` on the nonlinear fixture (C-14.1)."""
    problem, objective, plan, y0, u = _case(method_factory)
    opt = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    g = opt.gradient(u)
    ref = np.asarray(
        reference_gradient(
            y0=y0, u=u, problem=problem, objective=objective, plan=plan
        )
    ).reshape(g.shape)
    assert np.linalg.norm(g - ref) <= CERTIFIED_RTOL * np.linalg.norm(ref)


@pytest.mark.parametrize("method_factory", METHODS)
def test_hessian_vector_product_matches_the_independent_reference(
    method_factory,
):
    """Tier-1 oracle for ``H v``, column by column (C-14.1, C-14.5 req. 2).

    The whole operator is materialized rather than one product checked: a
    single direction can agree while individual columns do not, and the
    dense reference costs nothing extra once assembled.
    """
    problem, objective, plan, y0, u = _case(method_factory)
    opt = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    H = reference_hessian(
        y0=y0, u=u, problem=problem, objective=objective, plan=plan
    )
    cols = []
    for j in range(u.size):
        v = np.zeros(u.size)
        v[j] = 1.0
        cols.append(opt.hessian_vector_product(u, v.reshape(u.shape)).ravel())
    observed = np.array(cols).T
    assert np.linalg.norm(observed - H) <= CERTIFIED_RTOL * np.linalg.norm(H)


@pytest.mark.parametrize("method_factory", METHODS)
def test_gradient_passes_a_fixed_mesh_epsilon_sweep(method_factory):
    """Tier-3 corroboration: order 2, no plateau, mesh held fixed (C-2).

    Stands alongside the reference comparison above, never in place of it
    (C-14.5 req. 2). The mesh is not refined anywhere in this test.
    """
    problem, objective, plan, y0, u = _case(method_factory)
    opt = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    rng = np.random.default_rng(17)
    v = rng.standard_normal(u.shape)
    directional = float(np.sum(opt.gradient(u) * v))

    errors = []
    for eps in (1e-3, 1e-4, 1e-5):
        central = (
            opt.objective_value(u + eps * v) - opt.objective_value(u - eps * v)
        ) / (2.0 * eps)
        errors.append(abs(central - directional))
    orders = [
        np.log10(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)
    ]
    assert all(o > 1.7 for o in orders), (
        f"central differences do not converge at order 2 toward the "
        f"gradient: errors {errors}, observed orders {orders}"
    )


@pytest.mark.parametrize("method_factory", METHODS)
def test_a_nonuniform_plan_of_mixed_methods_matches_the_reference(
    method_factory,
):
    """A partitioned method inside a C-18 plan beside an ordinary one.

    Not an extra flourish: the plan supplies ``U``, ``B``, ``V`` and the
    stage times generically, and this is the only test in which a
    partitioned step's external-stage propagation has to agree with an
    unpartitioned neighbour's on a mesh that also changes ``h``.
    """
    problem = ShakenLatticeTrap()
    objective = FullCostObjective(nx=4, nu=2, y_target=TRANSPORT_TARGET)
    y0 = np.array([0.3, -0.2, 0.1, 0.05])
    partitioned = method_factory()
    nodes = np.array([0.2, 0.35, 0.7, 0.8, 1.1])
    methods = (partitioned, explicit_euler(), partitioned, rk4())
    plan = DiscretizationPlan(nodes=nodes, methods=methods)
    rng = np.random.default_rng(11)
    u = [rng.standard_normal((m.s, 2)) for m in methods]
    packed = np.concatenate(u, axis=0)

    opt = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    g = opt.gradient(packed)
    ref = np.asarray(
        reference_gradient(
            y0=y0, u=packed, problem=problem, objective=objective, plan=plan
        )
    ).reshape(g.shape)
    assert np.linalg.norm(g - ref) <= CERTIFIED_RTOL * np.linalg.norm(ref)


# =====================================================================
# The separable domain, checked where it is relied upon
# =====================================================================


class _CoupledHamiltonian(ShakenLatticeTrap):
    """``f^q`` made to depend on ``q``: outside C-8.4's domain.

    A single extra term, and the resulting system is still a perfectly
    ordinary ODE that every certified family integrates correctly. That is
    exactly why the refusal has to be raised from the partitioned route:
    nothing else about the problem announces it.
    """

    def f(self, y, u, t):
        out = super().f(y, u, t)
        out[0] += 0.5 * y[0]
        return out

    def F(self, y, u, t):
        out = super().F(y, u, t)
        out[0, 0] += 0.5
        return out


class _ControlledKinetic(ShakenLatticeTrap):
    """``f^q`` made to depend on ``u``: also outside the domain.

    Separate from the case above because only one of the two checks can see
    it. The *value* of ``f^q`` is unaffected by the substitution order --
    ``u`` is known before any stage is -- so the comparison of partial
    against complete evaluations agrees, and the structural check on ``G``
    is what refuses it.
    """

    def f(self, y, u, t):
        out = super().f(y, u, t)
        out[0] += 0.25 * u[0]
        return out

    def G(self, y, u, t):
        out = super().G(y, u, t)
        out[0, 0] += 0.25
        return out


class _MomentumDependentForce(ShakenLatticeTrap):
    """``f^p`` made to depend on ``p``: the mirror-image violation.

    Included because the two halves are not handled by the same code. A
    check written for ``f^q`` alone would leave this one integrated.
    """

    def f(self, y, u, t):
        out = super().f(y, u, t)
        out[2] += 0.4 * y[2]
        return out

    def F(self, y, u, t):
        out = super().F(y, u, t)
        out[2, 2] += 0.4
        return out


#: Which check refuses which violation, per method. Recorded explicitly so
#: that deleting one check cannot be masked by the other: the two mechanisms
#: are independent, and each of them is the only one that sees some row here.
#:
#: The empty ``A^q`` row of both methods' first stage means no ``f^q`` is
#: consumed mid-substitution under symplectic Euler at all, which is why its
#: ``f_q(q)`` row is caught structurally while Verlet's is caught by value.
REFUSALS = [
    pytest.param(symplectic_euler, _CoupledHamiltonian, "F^qq", id="se_fq_q"),
    pytest.param(verlet, _CoupledHamiltonian, "before the whole", id="v_fq_q"),
    pytest.param(symplectic_euler, _ControlledKinetic, "G^q", id="se_fq_u"),
    pytest.param(verlet, _ControlledKinetic, "G^q", id="v_fq_u"),
    pytest.param(
        symplectic_euler, _MomentumDependentForce, "before the whole",
        id="se_fp_p",
    ),
    pytest.param(
        verlet, _MomentumDependentForce, "before the whole", id="v_fp_p"
    ),
]


@pytest.mark.parametrize("method_factory,broken,mechanism", REFUSALS)
def test_a_non_separable_problem_is_refused_during_the_solve(
    method_factory, broken, mechanism
):
    """C-8.4's hypothesis is checked on the values the solve used.

    Not probed beforehand: precedent R-9 records a probe that varied the
    control and the time but not the state and so certified a
    state-dependent Jacobian as constant. These checks run on the stage
    values and Jacobians the step actually computed with.

    The refusing mechanism is asserted, not just the refusal. Both checks
    fire on some rows of :data:`REFUSALS` and only one fires on others, so
    without naming it a deleted check would hide behind the survivor.
    """
    problem = broken()
    objective = FullCostObjective(nx=4, nu=2, y_target=TRANSPORT_TARGET)
    y0 = np.array([0.3, -0.2, 0.1, 0.05])
    method = method_factory()
    plan = DiscretizationPlan.uniform((0.2, 1.1), 3, method)
    u = make_controls(3, method.s, 2, seed=2)
    opt = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    with pytest.raises(SeparabilityViolation, match=re.escape(mechanism)):
        opt.objective_value(u)


def test_both_refusal_mechanisms_are_exercised():
    """Neither check is dead code in the table above."""
    mechanisms = {row.values[2] for row in REFUSALS}
    assert mechanisms == {"F^qq", "G^q", "before the whole"}


#: Coupling strengths that a tolerance would swallow. ``1e-14`` is below any
#: plausible threshold; the negative value escapes a one-sided comparison
#: written as ``> tol`` rather than on the magnitude. Both are ordinary
#: physics -- a weak velocity-dependent force and an attractive one -- not
#: adversarial inputs.
WEAK_COUPLINGS = [1e-14, -1.0]


@pytest.mark.parametrize("coupling", WEAK_COUPLINGS)
@pytest.mark.parametrize("method_factory", [symplectic_euler, verlet])
def test_a_weakly_coupled_problem_is_refused_exactly(method_factory, coupling):
    """The structural check is exact, and a small coupling is not zero.

    A problem whose ``f^q`` depends on ``q`` at strength ``1e-14`` is a
    different discrete problem from one where it does not, and the
    partitioned sweeps discard that dependence rather than integrating it
    inaccurately. A tolerance here would return a gradient for a problem the
    caller did not pose, which is the silent-sentinel failure C-7 forbids.
    """

    class WeaklyCoupled(ShakenLatticeTrap):
        def f(self, y, u, t):
            out = super().f(y, u, t)
            out[0] += coupling * y[0]
            return out

        def F(self, y, u, t):
            out = super().F(y, u, t)
            out[0, 0] += coupling
            return out

    objective = FullCostObjective(nx=4, nu=2, y_target=TRANSPORT_TARGET)
    y0 = np.array([0.3, -0.2, 0.1, 0.05])
    method = method_factory()
    plan = DiscretizationPlan.uniform((0.2, 1.1), 3, method)
    u = make_controls(3, method.s, 2, seed=2)
    opt = GLMOptimizer(WeaklyCoupled(), objective, y0=y0, plan=plan)
    with pytest.raises(SeparabilityViolation):
        opt.objective_value(u)


#: Quadratic drag. ``V(q) = OMEGA_SQ q^2 / 2``, so ``V'(q) = OMEGA_SQ q``.
_DRAG = 0.7
_OMEGA_SQ = 1.3
_DRAG_H = 0.25
_DRAG_Q0 = 0.6


class _QuadraticDrag:
    """``f^p = -V'(q) - k p^2``: non-separable, and invisible at ``p = 0``.

    Both refusal mechanisms go blind together here, at a state a physical
    problem reaches -- a particle momentarily at rest under quadratic drag.
    ``F^pp = -2 k p`` vanishes there, so the structural check sees a
    block-anti-diagonal Jacobian. And with the stage momentum exactly zero,
    the mid-substitution and completed evaluations of ``f^p`` agree bit for
    bit, so the comparison sees nothing either.

    What is left is the fill value. ``NaN`` never compares equal, so the
    comparison refuses regardless of the arithmetic; a zero fill would
    accept, integrate a drag term it had dropped, and return a gradient for
    the undamped problem. See the fill in
    :meth:`PartitionedStageSolver.solve_stages`.
    """

    state_dim = 2
    control_dim = 1
    n_q = 1

    def f(self, y, u, t):
        q, p = y
        return np.array([p, -_OMEGA_SQ * q - _DRAG * p**2])

    def F(self, y, u, t):
        _, p = y
        return np.array([[0.0, 1.0], [-_OMEGA_SQ, -2.0 * _DRAG * p]])

    def G(self, y, u, t):
        return np.zeros((2, 1))


def test_the_stage_momentum_of_the_drag_witness_is_exactly_zero():
    """The fixture's defining property, established without the solver.

    Under symplectic Euler ``A^q = [[0]]``, so ``Z^q = y^q``, and
    ``Z^p = y^p + h f^p(Z)``. A zero fill supplies ``p = 0`` to that ``f^p``,
    giving ``y^p - h V'(q0)``, which the initial condition below sets to
    zero. Checked here so that the refusal test cannot pass because the
    fixture drifted into an ordinary violation.
    """
    y0 = np.array([_DRAG_Q0, _DRAG_H * _OMEGA_SQ * _DRAG_Q0])
    with_zero_fill = np.array([y0[0], 0.0])
    f_p = _QuadraticDrag().f(with_zero_fill, np.zeros(1), 0.0)[1]
    assert y0[1] + _DRAG_H * f_p == 0.0
    assert _QuadraticDrag().F(with_zero_fill, np.zeros(1), 0.0)[1, 1] == 0.0


def test_a_violation_hidden_by_a_vanishing_jacobian_is_still_refused():
    """The NaN fill is load-bearing, not decorative.

    Its only job is this case: a violation that both value comparison and
    structural check would miss under any finite fill. Replace ``np.nan``
    with ``0.0`` and this problem is accepted with ``Z = [0.6, 0.0]``.
    """
    solver = PartitionedStageSolver(n_q=1)
    with pytest.raises(SeparabilityViolation, match="before the whole"):
        solver.solve_stages(
            np.array([[_DRAG_Q0, _DRAG_H * _OMEGA_SQ * _DRAG_Q0]]),
            np.zeros((1, 1)),
            0.0,
            _DRAG_H,
            _QuadraticDrag(),
            symplectic_euler(),
        )


def test_the_broken_fixtures_are_integrated_fine_by_a_certified_family():
    """The refusals above are about the *method*, not about the problems.

    Without this, a mistake that made the modified fixtures unusable would
    make the refusal tests pass for the wrong reason.
    """
    objective = FullCostObjective(nx=4, nu=2, y_target=TRANSPORT_TARGET)
    y0 = np.array([0.3, -0.2, 0.1, 0.05])
    for broken in (
        _CoupledHamiltonian,
        _ControlledKinetic,
        _MomentumDependentForce,
    ):
        problem = broken()
        plan = DiscretizationPlan.uniform((0.2, 1.1), 3, rk4())
        u = make_controls(3, 4, 2, seed=2)
        opt = GLMOptimizer(problem, objective, y0=y0, plan=plan)
        g = opt.gradient(u)
        ref = np.asarray(
            reference_gradient(
                y0=y0, u=u, problem=problem, objective=objective, plan=plan
            )
        ).reshape(g.shape)
        assert np.linalg.norm(g - ref) <= CERTIFIED_RTOL * np.linalg.norm(ref)


class _UnpartitionedProblem:
    """A problem with no ``n_q``: the partition is not the tableau's to know."""

    state_dim = 2
    control_dim = 1

    def f(self, y, u, t):
        return np.array([y[1], -y[0] + u[0]])

    def F(self, y, u, t):
        return np.array([[0.0, 1.0], [-1.0, 0.0]])

    def G(self, y, u, t):
        return np.array([[0.0], [1.0]])


def test_a_problem_without_a_declared_partition_is_refused():
    """No default split is guessed (C-7).

    ``state_dim // 2`` is right for every canonical Hamiltonian and wrong in
    silence for anything else, which is the shape of answer C-7 forbids.
    """
    with pytest.raises(NotImplementedError, match="n_q"):
        partition_of(_UnpartitionedProblem())


@pytest.mark.parametrize("n_q", [0, 2, -1])
def test_a_partition_outside_the_state_is_refused(n_q):
    """Both blocks must be nonempty."""

    class Bad(_UnpartitionedProblem):
        pass

    Bad.n_q = n_q
    with pytest.raises(ValueError, match="n_q"):
        partition_of(Bad())


def test_a_glmethod_may_not_declare_the_partitioned_stage_type():
    """``PARTITIONED`` is not a structure of ``A``.

    Without this guard the declaration would fall through to the SDIRK
    branch of ``stage_structure_violations`` and be checked against a
    property that has nothing to do with it.
    """
    with pytest.raises(TableauDeclarationError, match="PARTITIONED"):
        GLMethod(
            A=np.array([[0.0]]),
            U=np.ones((1, 1)),
            B=np.ones((1, 1)),
            V=np.eye(1),
            c=np.array([0.0]),
            declared_stage_type=StageType.PARTITIONED,
        )


def test_a_partitioned_method_reports_its_own_stage_type():
    """The route is selected by ``stage_type``, as for every other family."""
    for factory in (symplectic_euler, verlet):
        assert factory().stage_type is StageType.PARTITIONED


@pytest.mark.parametrize("method_factory", METHODS)
def test_the_plan_freezes_the_partitioned_coefficients(method_factory):
    """A plan owns an immutable snapshot of a partitioned tableau too.

    The freeze is driven by ``coefficient_names``, so a partitioned method
    that named the wrong arrays would leave a writable alias exactly as the
    ordinary tableau did before C-18 (measured there: a recorded ``J = 2.0``
    moved to ``6.125`` with no refusal).
    """
    method = method_factory()
    plan = DiscretizationPlan.uniform((0.0, 1.0), 3, method)
    stored = plan.method_at(0)
    assert stored is not method
    for name in method.coefficient_names:
        assert not getattr(stored, name).flags.writeable, name
    method.A_q[0, 0] = 99.0
    assert stored.A_q[0, 0] != 99.0


@pytest.mark.parametrize("method_factory", METHODS)
def test_the_route_takes_no_factorizations(method_factory):
    """Substitution needs no matrix, so none is factored (C-15).

    Asserted rather than assumed: a partitioned method routed to the coupled
    solver by mistake would return the same numbers at a silently higher
    cost, which no accuracy test in this file could see.
    """
    problem, objective, plan, y0, u = _case(method_factory)
    opt = GLMOptimizer(problem, objective, y0=y0, plan=plan)
    for requirement in opt.step_requirements:
        assert requirement.factorizations_per_step == 0
        assert not requirement.needs_newton
    for solver in opt.stage_solvers:
        assert isinstance(solver, PartitionedStageSolver)
    opt.gradient(u)
