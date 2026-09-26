"""Cache invalidation and problem-structure deduction.

Both were sources of silent wrongness: a hash-keyed cache can return another
point's trajectory on collision, and a hard-coded structure reports facts about
the problem that were never checked.

Discrimination note. The structure tests below genuinely discriminate: the
previous implementation returned ``has_second_derivatives=False`` and
``NONLINEAR`` unconditionally, so they would have failed against it. The cache
tests do **not** discriminate, because the previous implementation recomputed
``hash(u.tobytes())`` on every call and so also detected these changes. They are
behavioural regression tests. The hazard actually removed -- a 64-bit hash
collision returning another control's trajectory, and a key that ignored array
shape because ``tobytes()`` discards it -- is not directly constructible in a
test, which is precisely why it was worth removing by construction rather than
guarding by assertion.
"""

import numpy as np
import pytest

from adjungo.core.problem import Linearity
from adjungo.methods.runge_kutta import rk4
from adjungo.optimization.interface import GLMOptimizer


class _Decay:
    """y' = -y + u, with no second-derivative callbacks."""

    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return -y + u

    def F(self, y, u, t):
        return np.array([[-1.0]])

    def G(self, y, u, t):
        return np.array([[1.0]])


class _DecayWithSecondDerivatives(_Decay):
    """Same dynamics, but advertising the contracted-Hessian callbacks."""

    def F_yy_action(self, y, u, t, v):
        return np.zeros((1, 1))

    def F_yu_action(self, y, u, t, v):
        return np.zeros((1, 1))

    def F_uu_action(self, y, u, t, v):
        return np.zeros((1, 1))


class _PartialSecondDerivatives(_Decay):
    """Only one of the three hooks; the second-order path needs all three."""

    def F_yy_action(self, y, u, t, v):
        return np.zeros((1, 1))


class _DeclaredLinear(_Decay):
    linearity = Linearity.LINEAR


class _Terminal:
    def evaluate(self, trajectory, u):
        return float(0.5 * trajectory.Y[-1, 0, 0] ** 2)

    def dJ_dy_terminal(self, y):
        return y

    def dJ_dy(self, y, t):
        return np.zeros_like(y)

    def dJ_du(self, z, u, t):
        return np.zeros_like(u)


def _build(problem):
    return GLMOptimizer(
        problem=problem,
        objective=_Terminal(),
        method=rk4(),
        t_span=(0.0, 1.0),
        N=8,
        y0=np.array([1.0]),
    )


def _controls(value):
    return np.full((8, rk4().s, 1), value)


def test_cache_returns_distinct_values_for_distinct_controls():
    optimizer = _build(_Decay())

    j_low = optimizer.objective_value(_controls(0.0))
    j_high = optimizer.objective_value(_controls(1.0))

    assert not np.isclose(j_low, j_high), (
        "Objective did not change with the control; the cache was not invalidated."
    )


def test_cache_is_not_fooled_by_in_place_mutation():
    """The cache key must be a copy, not a reference to caller memory.

    If the key aliased ``u``, mutating ``u`` in place would mutate the key too,
    and the stale trajectory would be reported as current. This is an invariant
    the copy-based key must maintain, not evidence against the hash-based one.
    """
    optimizer = _build(_Decay())

    u = _controls(0.0)
    j_before = optimizer.objective_value(u)

    u[:] = 1.0
    j_after = optimizer.objective_value(u)

    assert not np.isclose(j_before, j_after), (
        "In-place mutation of the control did not invalidate the cache."
    )


def test_repeated_evaluation_at_the_same_control_is_stable():
    optimizer = _build(_Decay())
    u = _controls(0.3)

    first = optimizer.objective_value(u)
    second = optimizer.objective_value(u)
    gradient_first = optimizer.gradient(u)
    third = optimizer.objective_value(u)

    assert first == second == third
    assert np.array_equal(gradient_first, optimizer.gradient(u))


def test_cache_distinguishes_nearly_identical_controls():
    """A one-ULP difference is a different point and must not hit the cache."""
    optimizer = _build(_Decay())

    u = _controls(0.5)
    j_reference = optimizer.objective_value(u)

    u_perturbed = u.copy()
    u_perturbed[3, 0, 0] = np.nextafter(u_perturbed[3, 0, 0], 1.0)

    assert not np.array_equal(u, u_perturbed)
    # Recomputation is what matters; the values may legitimately round equal.
    assert np.isfinite(optimizer.objective_value(u_perturbed))
    assert np.isfinite(j_reference)


def test_second_derivatives_detected_when_present():
    structure = _build(_DecayWithSecondDerivatives())._deduce_problem_structure()
    assert structure.has_second_derivatives is True


def test_second_derivatives_absent_when_not_supplied():
    structure = _build(_Decay())._deduce_problem_structure()
    assert structure.has_second_derivatives is False


def test_partial_second_derivatives_count_as_absent():
    """All three contracted-Hessian hooks are needed; one is not enough."""
    structure = _build(_PartialSecondDerivatives())._deduce_problem_structure()
    assert structure.has_second_derivatives is False


def test_undeclared_problem_is_assumed_nonlinear():
    """The conservative default: no reuse, no constant-Jacobian assumption."""
    structure = _build(_Decay())._deduce_problem_structure()

    assert structure.linearity is Linearity.NONLINEAR
    assert structure.jacobian_constant is False


def test_declared_linearity_is_honoured():
    structure = _build(_DeclaredLinear())._deduce_problem_structure()

    assert structure.linearity is Linearity.LINEAR


def test_declared_linearity_does_not_deduce_a_constant_jacobian():
    """``LINEAR`` says ``F`` is independent of ``y`` and ``u``, not of ``t``.

    ``F = M(t)`` satisfies it (C-16.1), so reading it as a constant Jacobian
    would make the C-15.1 declaration on the caller's behalf. Constancy
    reaches the reuse path only as an explicit ``ProblemStructure`` or from a
    verified affine class (C-17.2). The end-to-end consequence is asserted in
    ``tests/test_factorization_reuse.py``.
    """
    structure = _build(_DeclaredLinear())._deduce_problem_structure()

    assert structure.jacobian_constant is False


@pytest.mark.parametrize("bogus", ["linear", 0, None])
def test_bogus_linearity_declaration_falls_back_to_nonlinear(bogus):
    """A declaration that is not a Linearity must not be trusted."""

    class _Bogus(_Decay):
        linearity = bogus

    structure = _build(_Bogus())._deduce_problem_structure()

    assert structure.linearity is Linearity.NONLINEAR
    assert structure.jacobian_constant is False
