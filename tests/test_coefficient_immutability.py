"""Coefficient immutability survives copying — C-15.7.

``AffineDynamics.__init__`` copies its coefficient arrays and sets
``writeable=False`` on them. C-17.3 calls that *ownership*, and it is what
makes it safe for ``F`` to return ``self._M`` itself rather than a copy: the
tape the backward sweeps read holds references to that one buffer, so if it
could change between the forward solve and the adjoint, the adjoint would be
seeded with a matrix the forward solve never used.

``copy.deepcopy`` dropped the flag. NumPy defines ``ndarray.__deepcopy__`` and
it returns a writeable array, so the copy arrived with exactly the ownership
guarantee that the constructor exists to establish already gone. ``pickle``
goes through ``__reduce__``, which records the flag, and ``copy.copy`` shares
the original buffers, so neither of those ever lost it — the deep copy was
alone in this, which is why it went unnoticed.

Why refusing at verification is a guard and not a cure
------------------------------------------------------

The other candidate remedy considered was to check writeability in
``affine_dynamics_verified`` and refuse. Both are needed, and they do
different jobs. Re-freezing is the *cure*; refusing is not, because the
general route — the one a refusal selects — reads ``F`` and retains what it
returns exactly as the affine route does, so an aliasing problem is wrong
there by the same closed-form displacement
(``test_refusing_to_certify_would_not_repair_the_aliasing``).

That is why a mutable coefficient raises :class:`MutableCoefficients` rather
than quietly answering ``False``. It is the one refusal in this module that
does not lead to a correct computation.

"Every route" would be too strong a claim, and is not made. The monolithic
reference path in ``adjungo/validation/reference.py`` assembles its own owned
arrays and copies values out of ``F``, so it is *not* displaced by this
mutation. What matters is not that a route reads ``F`` but that it retains the
object ``F`` returned.

The measurement
---------------

``n = ν = 1``, ``M = 0.4``, ``C = 1``, ``b = 0``, ``y₀ = 0.8``,
``t_span = (0, 0.5)``, ``N = 2`` (so ``h = 0.25``), ``u = (0.3, 0.2)``,
``explicit_euler``, and ``J = ½y₂²``. The coefficient is overwritten with
``M' = 0.9`` on the first call of the backward sweep, which is the earliest
moment at which forward and adjoint can disagree.

With ``a = 1 + hM``: ``y₁ = 0.955``, ``y₂ = 1.1005``, ``λ₂ = y₂``, and
``∂J/∂u₀ = hCλ₂a``. Replacing ``a`` by ``a' = 1 + hM'`` in the adjoint alone
displaces that by ``h²Cy₂(M' − M) = 0.034390625`` exactly, while ``∂J/∂u₁``,
which never multiplies by ``a``, is untouched. The wrong answer is predicted
in closed form rather than merely observed to differ, per C-14.1.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_array

from adjungo import GLMOptimizer, ProblemStructure
from adjungo.core.affine import (
    _COEFFICIENT_BUFFERS,
    _COPY_PROTOCOL_HOOKS,
    AffineDynamics,
    InvalidCoefficients,
    MissingCoefficients,
    MutableCoefficients,
    TimeVaryingAffineDynamics,
    _defined_as,
    affine_dynamics_verified,
    require_immutable_coefficients,
)
from adjungo.core.problem import Linearity
from adjungo.core.requirements import deduce_requirements
from adjungo.methods.runge_kutta import explicit_euler
from adjungo.optimization.gradient import assemble_gradient
from adjungo.solvers.factory import create_stage_solver
from adjungo.stepping import adjoint_solve, forward_solve
from adjungo.validation.reference import reference_gradient

M_CONST = np.array([[0.4]])
C_CONST = np.array([[1.0]])
M_OVERWRITTEN = 0.9

#: The ``C = 2`` fixture of C-15.7, where the displacement is largest and the
#: exact and mutated gradients are furthest apart (0.6781500 against
#: 0.7552125). ``_closed_form(2.0)`` gives both.
C_DOUBLED = np.array([[2.0]])

Y0 = np.array([0.8])
T_SPAN = (0.0, 0.5)
N_STEPS = 2
H = (T_SPAN[1] - T_SPAN[0]) / N_STEPS

U = np.array([[[0.3]], [[0.2]]])

#: Round-off only. Both sides of every comparison against the closed form below
#: run a fixed, short sequence of arithmetic — two Euler steps and two adjoint
#: steps, no iterative solve — so C-3 admits accumulated round-off and nothing
#: else: a few tens of operations on O(1) quantities, bounded well below 1e-13.
EXACT_TOL = 1e-13


def _closed_form(c: float = 1.0) -> tuple[NDArray, float]:
    """The exact gradient, and separately the displacement aliasing causes.

    Returned apart so that the defective answer is the sum of two independently
    derived quantities rather than a number read off a broken run.

    ``c`` is a parameter because the displacement carries **one** factor of it,
    not two: ``g₀ = hCy₂a``, so replacing ``a`` by ``a'`` in the adjoint alone
    displaces it by ``hCy₂(a' − a) = h²Cy₂(M' − M)``. At ``c = 1`` a spurious
    square is invisible, and an earlier draft of C-15.7 carried one.
    """
    a = 1.0 + H * M_CONST[0, 0]
    y1 = Y0[0] * a + H * c * U[0, 0, 0]
    y2 = y1 * a + H * c * U[1, 0, 0]
    lam2 = y2
    exact = np.array([[[H * c * lam2 * a]], [[H * c * lam2]]])
    displacement = H * H * c * y2 * (M_OVERWRITTEN - M_CONST[0, 0])
    return exact, displacement


class OverwritesCoefficientMidSolve:
    """``J = ½y_N²``, overwriting a coefficient as the backward sweep opens.

    ``dJ_dy_terminal`` is the first objective callback the adjoint calls, and
    it runs after the forward solve has completed, so a write here is the
    cleanest available way to make the forward and backward sweeps disagree
    about a coefficient without touching package internals.
    """

    def __init__(self, target: object | None) -> None:
        self.target = target
        self.fired = False

    def _fire(self) -> None:
        if self.target is not None and not self.fired:
            self.fired = True
            self.target._M[0, 0] = M_OVERWRITTEN  # type: ignore[attr-defined]

    def evaluate(self, trajectory: object, u: NDArray) -> float:
        return 0.5 * float(trajectory.Y[-1, 0, 0] ** 2)  # type: ignore[attr-defined]

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        self._fire()
        return np.array(y_final, dtype=float)

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros_like(y)

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros_like(u_stage)

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros((1, 1))

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((1, 1))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return np.ones((1, 1))


class AliasesAMutableBuffer:
    """A plain callback problem returning a live reference to its own buffer.

    Not an :class:`AffineDynamics`, and deliberately so: it is the control for
    ``test_the_aliasing_error_is_route_independent``, showing that the hazard
    belongs to the aliasing rather than to anything the affine route does.
    """

    state_dim = 1
    control_dim = 1

    def __init__(self, c: float = 1.0) -> None:
        self._M = np.array(M_CONST, dtype=float, copy=True)
        self._C = np.array([[c]], dtype=float)

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._M @ y + self._C @ u)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._M

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._C

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((1, 1))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((1, 1))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((1, 1))


GENERAL_ROUTE = ProblemStructure(
    linearity=Linearity.NONLINEAR,
    jacobian_constant=False,
    jacobian_control_dependent=False,
    has_second_derivatives=True,
    state_affine=False,
    jointly_affine=False,
)


def _gradient(
    problem: object,
    objective: object,
    structure: ProblemStructure | None = None,
) -> NDArray:
    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=explicit_euler(),
        y0=Y0,
        t_span=T_SPAN,
        N=N_STEPS,
        problem_structure=structure,
    )
    return optimizer.gradient(U)


def _fresh() -> AffineDynamics:
    return AffineDynamics(M_CONST, C_CONST)


def _buffers(problem: AffineDynamics) -> list[NDArray]:
    return [getattr(problem, name) for name in _COEFFICIENT_BUFFERS]


# --------------------------------------------------------------------------
# The invariant, across every route that produces a second instance
# --------------------------------------------------------------------------


def test_the_constructor_freezes_every_coefficient_buffer() -> None:
    assert all(not buf.flags.writeable for buf in _buffers(_fresh()))


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(copy.copy, id="copy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
        pytest.param(
            lambda p: copy.deepcopy({"problems": [p]})["problems"][0],
            id="deepcopy-nested",
        ),
    ],
)
def test_a_duplicate_keeps_its_coefficients_frozen(duplicate) -> None:
    """The deepcopy cases are the ones that failed before C-15.7.

    Nesting is included because ``deepcopy`` of a container dispatches through
    the memo rather than calling the object's hook directly, and a cure that
    only worked at the top level would be worth little: the realistic way an
    ``AffineDynamics`` gets deep-copied is as a member of something larger.
    """
    assert all(not buf.flags.writeable for buf in _buffers(duplicate(_fresh())))


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(copy.copy, id="copy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_a_duplicate_carries_the_same_coefficient_values(duplicate) -> None:
    """Re-freezing must not be achieved by handing back different numbers."""
    original = _fresh()
    other = duplicate(original)
    assert np.array_equal(other.M, original.M)
    assert np.array_equal(other.C, original.C)
    assert np.array_equal(other.b, original.b)
    assert other.state_dim == original.state_dim
    assert other.control_dim == original.control_dim


def test_a_deep_copy_owns_storage_separate_from_the_original() -> None:
    """Otherwise the fix would be "share the frozen buffer", which is not a
    deep copy and would couple two problems the caller means to keep apart.

    ``is not`` would be too weak on its own: two distinct array objects can be
    views of one block of memory, so independence is asserted against
    ``np.shares_memory``.
    """
    original = _fresh()
    other = copy.deepcopy(original)
    assert all(
        not np.shares_memory(getattr(other, name), getattr(original, name))
        for name in _COEFFICIENT_BUFFERS
    )


def test_a_deep_copy_is_still_verified_and_still_affine() -> None:
    other = copy.deepcopy(_fresh())
    assert affine_dynamics_verified(other)
    assert other.coefficients_constant


def test_the_time_varying_class_survives_a_deep_copy() -> None:
    """It stores callables rather than buffers, so it has no frozen state to
    lose — but its *evaluated* coefficients are frozen per C-17.3, and that
    must still hold through a copy."""
    problem = TimeVaryingAffineDynamics(
        lambda t: np.array([[0.4 + t]]),
        lambda t: np.array([[1.0]]),
        state_dim=1,
        control_dim=1,
    )
    other = copy.deepcopy(problem)
    assert affine_dynamics_verified(other)
    assert not other.M(0.3).flags.writeable
    assert other.M(0.3) == pytest.approx(problem.M(0.3), abs=EXACT_TOL)


def test_every_frozen_buffer_is_named_in_the_refreeze_list() -> None:
    """Guards the failure mode ``_COEFFICIENT_BUFFERS`` exists to prevent: a
    coefficient added to ``__init__`` and not to ``__setstate__`` would be
    writeable after a deep copy and nothing else here would notice."""
    problem = _fresh()
    frozen = {
        name
        for name, value in vars(problem).items()
        if isinstance(value, np.ndarray) and not value.flags.writeable
    }
    assert frozen == set(_COEFFICIENT_BUFFERS)


@pytest.mark.parametrize(
    "b",
    [
        pytest.param(None, id="defaulted"),
        pytest.param([2.0], id="list"),
        pytest.param(np.array([2.0]), id="flat-array"),
        pytest.param(np.array([[2.0]]), id="column-array"),
    ],
)
def test_every_coefficient_buffer_owns_its_storage(b) -> None:
    """A frozen *view* over a writeable base is not immutable data.

    ``np.array(b, copy=True).reshape(-1)`` returns a view, so the original
    construction froze the view and left ``b.base`` writeable: ``p.b`` could
    still be changed, through ``p.b.base[0] = ...``, while every flag on
    ``p.b`` said read-only. The column-array case is the one that exercised
    it, since that is when the reshape has work to do.
    """
    problem = AffineDynamics(M_CONST, C_CONST, b)
    for name in _COEFFICIENT_BUFFERS:
        buffer = getattr(problem, name.lstrip("_"))
        assert buffer.flags.owndata, name
        assert not buffer.flags.writeable, name


def test_an_unpickled_problem_owns_its_storage() -> None:
    """Pickle reconstructs an ``ndarray`` as a view over the pickled ``bytes``,
    so a round trip arrives frozen but not owning. ``__setstate__`` restores
    the constructor's postcondition rather than verification having to reason
    about whether a particular non-array base happens to be immutable."""
    other = pickle.loads(pickle.dumps(_fresh()))
    assert all(buf.flags.owndata for buf in _buffers(other))
    assert affine_dynamics_verified(other)


# --------------------------------------------------------------------------
# What the invariant is worth
# --------------------------------------------------------------------------


def test_a_deep_copy_reproduces_the_gradient_of_the_original() -> None:
    exact, _ = _closed_form()
    other = copy.deepcopy(_fresh())
    assert _gradient(other, OverwritesCoefficientMidSolve(None)) == (
        pytest.approx(exact, abs=EXACT_TOL)
    )


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(lambda p: p, id="original"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(copy.copy, id="copy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_overwriting_a_coefficient_mid_solve_is_refused(duplicate) -> None:
    """The cure, stated as the error direction C-7 requires.

    Before C-15.7 the ``deepcopy`` case here did not raise. It returned a
    gradient displaced by exactly ``0.034390625``, with nothing reported.
    """
    problem = duplicate(_fresh())
    with pytest.raises(ValueError, match="read-only"):
        _gradient(problem, OverwritesCoefficientMidSolve(problem))


@pytest.mark.parametrize("c", [1.0, 2.0])
def test_refusing_to_certify_would_not_repair_the_aliasing(c: float) -> None:
    """Why re-freezing is the cure and a writeability check is only a guard.

    The general route — the one a refusal selects — reads ``F`` and retains
    what it returns exactly as the affine route does. So a problem handing out
    a live reference to a mutable coefficient is wrong there by the *same*
    closed-form displacement. Declining to certify such a problem would pick a
    route already known to be no better, which is why
    :class:`MutableCoefficients` raises instead.

    ``c = 2`` is the case that can fail if the displacement is written with a
    spurious ``C²``: it predicts ``0.154125`` where the truth is ``0.0770625``.
    """
    exact, displacement = _closed_form(c)
    expected = exact + np.array([[[displacement]], [[0.0]]])

    problem = AliasesAMutableBuffer(c)
    assert not affine_dynamics_verified(problem)
    on_general_route = _gradient(
        problem, OverwritesCoefficientMidSolve(problem), GENERAL_ROUTE
    )
    assert on_general_route == pytest.approx(expected, abs=EXACT_TOL)


def test_the_deduced_route_for_such_a_problem_is_the_general_one() -> None:
    """Pins the premise of the test above, which would otherwise be comparing
    two spellings of the same route without saying so. ``AliasesAMutableBuffer``
    is not an ``AffineDynamics``, so deduction cannot reach the affine route
    for it and the explicit ``GENERAL_ROUTE`` is not a second data point."""
    problem = AliasesAMutableBuffer()
    deduced = _gradient(problem, OverwritesCoefficientMidSolve(problem))
    declared = AliasesAMutableBuffer()
    explicit = _gradient(
        declared, OverwritesCoefficientMidSolve(declared), GENERAL_ROUTE
    )
    assert np.array_equal(deduced, explicit)


def test_the_second_control_component_is_untouched_by_the_aliasing() -> None:
    """A displacement that moved every component equally would be consistent
    with a scaling mistake anywhere in the sweep. ``∂J/∂u₁`` never multiplies
    by the state Jacobian, so it is the component that localises the defect to
    the one propagation step that reads the overwritten matrix."""
    exact, _ = _closed_form()
    problem = AliasesAMutableBuffer()
    gradient = _gradient(problem, OverwritesCoefficientMidSolve(problem))
    assert gradient[1, 0, 0] == pytest.approx(exact[1, 0, 0], abs=EXACT_TOL)
    assert gradient[0, 0, 0] != pytest.approx(exact[0, 0, 0], abs=EXACT_TOL)


def test_a_route_that_copies_out_of_F_is_not_displaced() -> None:
    """Bounds the claim above: reading ``F`` is not what does the damage.

    ``validation/reference.py`` assembles its own arrays and copies values out
    of ``F``, so the same mutation at the same moment leaves it exact. What
    decides the matter is whether a route *retains* the object ``F`` returned.
    Without this, C-15.7 would be free to drift back to "every route reads
    ``F``", which is the over-broad form an earlier draft carried.
    """
    exact, _ = _closed_form()
    problem = AliasesAMutableBuffer()
    objective = OverwritesCoefficientMidSolve(problem)
    gradient = reference_gradient(
        y0=Y0,
        u=U,
        t_span=T_SPAN,
        N=N_STEPS,
        problem=problem,
        method=explicit_euler(),
        objective=objective,
    )
    assert objective.fired
    assert np.asarray(gradient) == pytest.approx(exact, abs=EXACT_TOL)


# --------------------------------------------------------------------------
# The cure cannot be undone from a subclass
# --------------------------------------------------------------------------


def test_the_copy_protocol_hooks_cover_what_can_pre_empt_the_refreeze() -> None:
    """A roster check, not a tautology: each name below is separately shown to
    defeat the cure by ``test_a_subclass_overriding_a_copy_hook_is_refused``,
    so dropping one from the tuple fails here *and* there."""
    assert set(_COPY_PROTOCOL_HOOKS) == {
        "__setstate__",
        "__getstate__",
        "__deepcopy__",
        "__copy__",
        "__replace__",
        "__reduce__",
        "__reduce_ex__",
    }


def _subclass_overriding(name: str) -> type:
    def permissive_setstate(self, state):
        self.__dict__.update(state)

    def identity_deepcopy(self, memo):
        clone = AffineDynamics(self.M, self.C, self.b)
        for buf in _COEFFICIENT_BUFFERS:
            getattr(clone, buf).flags.writeable = True
        return clone

    bodies = {
        "__setstate__": permissive_setstate,
        "__getstate__": lambda self: dict(self.__dict__),
        "__deepcopy__": identity_deepcopy,
        "__copy__": lambda self: AffineDynamics(self.M, self.C, self.b),
        "__replace__": lambda self, **changes: AffineDynamics(
            self.M, self.C, self.b
        ),
        "__reduce__": lambda self: (
            AffineDynamics,
            (self.M, self.C, self.b),
        ),
        "__reduce_ex__": lambda self, protocol: (
            AffineDynamics,
            (self.M, self.C, self.b),
        ),
    }
    return type("Overrides", (AffineDynamics,), {name: bodies[name]})


@pytest.mark.parametrize("name", _COPY_PROTOCOL_HOOKS)
def test_a_subclass_overriding_a_copy_hook_is_refused(name: str) -> None:
    """``__setstate__`` is the load-bearing case: a subclass overriding it
    alone leaves every other comparison in ``affine_dynamics_verified``
    passing, so without this the cure is one method away from being undone.
    The rest are refused because each chooses either whether the hook runs or
    what it is handed."""
    subclass = _subclass_overriding(name)
    assert not affine_dynamics_verified(subclass(M_CONST, C_CONST))


def test_a_subclass_defining_an_unrelated_member_is_still_verified() -> None:
    """The false-refusal control. Without it the parametrised refusal above is
    satisfied by a check that refuses every subclass, which would be a
    different and much blunter change than the one made."""

    class CarriesALabel(AffineDynamics):
        def label(self) -> str:
            return "unrelated"

    assert affine_dynamics_verified(CarriesALabel(M_CONST, C_CONST))


# --------------------------------------------------------------------------
# Mutable coefficients are refused loudly, however they arose
# --------------------------------------------------------------------------


@pytest.mark.parametrize("buffer_name", _COEFFICIENT_BUFFERS)
def test_a_subclass_that_replaces_a_buffer_after_init_is_refused(
    buffer_name: str,
) -> None:
    """The gap the class-and-copy-hook checks left open.

    ``self._M = self._M.copy()`` after ``super().__init__`` looks defensive
    and is the opposite: NumPy returns a *writeable* copy, so the instance
    carries mutable coefficients while overriding no guaranteed member and no
    copy hook. Before this refusal such a subclass was certified affine and
    reproduced the full displacement below.
    """

    class ReplacesABuffer(AffineDynamics):
        def __init__(self, M, C, b=None):
            super().__init__(M, C, b)
            setattr(self, buffer_name, getattr(self, buffer_name).copy())

    with pytest.raises(MutableCoefficients, match=buffer_name):
        affine_dynamics_verified(ReplacesABuffer(M_CONST, C_CONST))


def test_the_refusal_is_raised_not_returned() -> None:
    """C-15.7's asymmetry, asserted rather than only documented.

    Every other refusal in ``affine_dynamics_verified`` returns ``False`` and
    the general route then computes the same derivative. This one cannot: the
    general route retains what ``F`` returns in the same way, so answering
    ``False`` would select a route already known to be no better. If this ever
    becomes a quiet ``False``, the test fails rather than the suite merely
    continuing to pass with a silently wrong Hessian available.
    """
    problem = _fresh()
    problem.__dict__["_M"] = problem.M.copy()
    with pytest.raises(MutableCoefficients):
        affine_dynamics_verified(problem)


class CarriesASlot(AffineDynamics):
    """A verified subclass whose reduced state is a two-element tuple.

    Module level rather than local to its test because pickle cannot reach a
    class defined inside a function, and pickle is one of the three
    duplication routes under test.
    """

    __slots__ = ("label",)


def test_a_frozen_view_over_a_writeable_base_is_refused() -> None:
    """The ownership half of the refusal, asserted directly.

    No construction route reaches this state any more — ``__init__`` and
    ``__setstate__`` both produce owning buffers — so behaviour alone cannot
    reach the branch, and C-15.5 requires such a gate be asserted rather than
    left to an accuracy test that can never exercise it. The state is built
    here by hand.

    The first two assertions are the reason the branch is not redundant with
    the ``writeable`` check: every flag on ``view`` says read-only, and the
    data behind it changes anyway.
    """
    base = np.array([[0.4], [0.0]])
    view = base[:1]
    view.flags.writeable = False
    assert not view.flags.writeable
    base[0, 0] = 9.0
    assert view[0, 0] == 9.0

    problem = _fresh()
    problem.__dict__["_M"] = view
    with pytest.raises(MutableCoefficients, match="non-owning"):
        affine_dynamics_verified(problem)


def test_a_verified_subclass_using_slots_survives_every_duplication() -> None:
    """``__setstate__`` must handle the two-element state form.

    A subclass adding ``__slots__`` overrides no guaranteed member, so it is
    verified — and ``object.__reduce_ex__`` reports it as
    ``(instance_dict, slot_values)``. A ``__setstate__`` assuming a plain dict
    raises ``ValueError`` on every copy and pickle of one, which would have
    discarded subclass state in exactly the way choosing ``__setstate__`` over
    ``__deepcopy__`` was meant to avoid.
    """
    original = CarriesASlot(M_CONST, C_CONST)
    original.label = "kept"
    assert affine_dynamics_verified(original)

    for duplicate in (
        copy.copy,
        copy.deepcopy,
        lambda p: pickle.loads(pickle.dumps(p)),
    ):
        other = duplicate(original)
        assert other.label == "kept"
        assert affine_dynamics_verified(other)
        assert all(not buf.flags.writeable for buf in _buffers(other))


def test_a_deep_copy_of_a_refused_subclass_raises_rather_than_refusing_quietly() -> None:
    """Copying must not launder a problem into verification — and refusing it
    quietly is not enough.

    This is the case where the two halves of C-15.7 meet. Overriding a copy
    hook is grounds for a quiet refusal, and deep-copying that subclass is
    exactly what leaves the buffers writeable, so the same object trips an
    eligibility check *and* carries an invalid coefficient. With the
    coefficient check ordered last, the quiet ``False`` arrived first and the
    caller was handed the general route, which is displaced by the same amount
    the affine route would have been. Asserting ``not verified`` here — which
    is what this test did — asserts the outcome C-15.7 identifies as unsafe.
    """
    duplicate = copy.deepcopy(_subclass_overriding("__setstate__")(M_CONST, C_CONST))
    assert all(buf.flags.writeable for buf in _buffers(duplicate))
    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(duplicate)


# --------------------------------------------------------------------------
# Validity is answered before eligibility
# --------------------------------------------------------------------------

#: Four mechanisms across the three eligibility predicates in
#: ``affine_dynamics_verified``, plus the multi-root refusal that precedes
#: them. The copy hook and the lookup hook share one loop and are listed
#: separately because they are different mechanisms, not different predicates.
INELIGIBLE_SHAPES = (
    "two roots",
    "copy hook",
    "lookup hook",
    "guaranteed member",
    "instance shadow",
)


def _ineligible(kind: str) -> AffineDynamics:
    """A correctly constructed problem that fails one eligibility check.

    Constructed normally, so its coefficients start out frozen and owned: the
    two tests below differ only in whether the buffers are then unfrozen,
    which is what isolates the ordering question from the refusal itself.

    Every shape is built through ``AffineDynamics.__init__``, so all of them
    hold the root's coefficient storage and therefore satisfy the precondition
    in ``require_immutable_coefficients``. That is the whole precondition:
    what each shape does to ``f``, ``F``, ``G`` or the attribute machinery is
    an *eligibility* matter and is deliberately not consulted here.
    """
    if kind == "two roots":
        cls = type("BothRoots", (AffineDynamics, TimeVaryingAffineDynamics), {})
        return cls(M_CONST, C_CONST)
    if kind == "copy hook":
        return _subclass_overriding("__setstate__")(M_CONST, C_CONST)
    if kind == "guaranteed member":
        cls = type(
            "OverridesF",
            (AffineDynamics,),
            {"f": lambda self, t, y, u: y},
        )
        return cls(M_CONST, C_CONST)
    if kind == "lookup hook":
        cls = type(
            "DefinesGetattr",
            (AffineDynamics,),
            {"__getattr__": lambda self, name: None},
        )
        return cls(M_CONST, C_CONST)
    if kind == "instance shadow":
        problem = _fresh()
        problem.f = lambda t, y, u: y  # type: ignore[method-assign]
        return problem
    raise AssertionError(f"unknown shape {kind!r}")


@pytest.mark.parametrize("kind", INELIGIBLE_SHAPES)
def test_an_ineligible_problem_with_intact_coefficients_is_refused_quietly(
    kind: str,
) -> None:
    """The control, and the premise of the test below.

    Without it, that test is satisfied by a check that raises for every
    subclass, which would be a far blunter change than the one made — and it
    would also leave these four shapes unproven as *ineligible*, so the pairing
    would demonstrate nothing about ordering.
    """
    assert not affine_dynamics_verified(_ineligible(kind))


@pytest.mark.parametrize("kind", INELIGIBLE_SHAPES)
def test_mutable_coefficients_raise_however_the_problem_is_ineligible(
    kind: str,
) -> None:
    """Validity is not gated on eligibility — C-15.7.

    Each shape above returns ``False`` before the coefficient check would have
    run, had that check stayed last. Since the general route is displaced by
    the same amount as the affine one, every one of those quiet answers was a
    silent wrong gradient waiting on a mutation.
    """
    problem = _ineligible(kind)
    problem.__dict__["_M"] = problem.M.copy()
    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)


def test_the_coefficients_are_read_through_the_roots_own_dict_descriptor() -> None:
    """The marker cannot be answered by the machinery it runs ahead of.

    ``__dict__`` is a lookup hook, and the coefficient check now runs before
    the lookup hooks are compared. Reading the root-initialised marker through
    ``problem.__dict__`` would therefore consult a subclass property whose
    honesty is exactly what has not been established yet — and that subclass
    is quietly refused a few lines later, so it is precisely a case that must
    not be allowed to answer for itself.

    The buffer is unfrozen before the hiding class is installed, so the test
    builds its state through ordinary storage and shares no mechanism with the
    code under test. The subclass below, constructed normally, covers what
    that sequencing cannot.

    Neither shape here can be reached by either widening, which is not
    incidental. Both override ``f``, ``F`` and ``G``, so reader identity sees
    nothing of the root's; and the buffer is *deleted* rather than replaced,
    so presence has no name to resolve. That leaves the marker as the only
    thing able to speak, which is the point: it is the only one of the three
    that *remembers* that three frozen arrays were established here, and so
    the only one that can tell an instance taken apart from one that was
    never assembled.
    """

    class HidesItsDict(AffineDynamics):
        @property
        def __dict__(self):  # type: ignore[override]
            return {}

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(super().f(y, u, t))

        def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().F(y, u, t).copy()

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().G(y, u, t).copy()

    problem = _fresh()
    del AffineDynamics.__dict__["__dict__"].__get__(problem)["_M"]
    problem.__class__ = HidesItsDict

    assert problem.__dict__ == {}
    assert not hasattr(problem, "_M")
    with pytest.raises(MissingCoefficients, match="_M"):
        affine_dynamics_verified(problem)


def test_the_marker_is_written_where_it_is_read() -> None:
    """Reading defensively and writing trustingly protects nothing — C-15.7.

    The root-initialised marker is read through ``AffineDynamics``'s own
    ``__dict__`` descriptor precisely so that no subclass can answer for it,
    but it was *written* through ``self.__dict__``. A subclass exposing a
    different mapping under that name therefore took delivery of the marker
    while the read went to the real instance dictionary and found nothing, so
    the precondition concluded the root initialiser had never run and every
    one of the three call sites fell silent.

    Reached by ordinary construction, which is what the test above cannot
    reach: it installs the hiding class *after* ``__init__``, by which time the
    marker has already been written to the real dictionary. Measured on the
    committed fixture at ``C = 2`` when the buffer was a writeable array:
    ``0.7552125`` against an exact ``0.6781500``, the usual ``0.0770625``.

    As above, the overridden readers and the deleted buffer are what leave the
    marker alone able to speak -- either widening would refuse this instance
    wherever the marker had been delivered, and so could not witness the
    write.
    """

    class KeepsItsOwnMapping(AffineDynamics):
        def __init__(self, M: NDArray, C: NDArray) -> None:
            object.__setattr__(self, "_side", {})
            super().__init__(M, C)
            del self._M

        @property
        def __dict__(self):  # type: ignore[override]
            return object.__getattribute__(self, "_side")

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(super().f(y, u, t))

        def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().F(y, u, t).copy()

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().G(y, u, t).copy()

    problem = KeepsItsOwnMapping(M_CONST, np.array([[2.0]]))
    storage = AffineDynamics.__dict__["__dict__"].__get__(problem)
    assert problem.__dict__ is not storage
    assert "_M" not in storage

    objective = OverwritesCoefficientMidSolve(problem)
    with pytest.raises(MissingCoefficients, match="_M"):
        GLMOptimizer(
            problem, objective, explicit_euler(), T_SPAN, N_STEPS, Y0
        )
    assert not objective.fired


def test_a_subclass_that_never_ran_the_root_initialiser_is_refused_by_name(
) -> None:
    """Inheriting a buffer reader is the obligation; running ``__init__`` is not.

    ``__init__`` carries no guarantee and is not compared, so a subclass can
    inherit ``f``, ``F`` and ``G`` while holding no buffers at all. This shape
    is diagnosed at the route decision, naming the attribute.

    An earlier version let it pass quietly, on the reasoning that an inference
    from the MRO is unsound in the other direction -- an override may delegate
    to ``super()`` and read the buffers anyway -- and so could not be trusted.
    That reasoning confused a *replacement* for the recorded flag with a
    *widening* of it. As a replacement it is indeed unsound, because it would
    have to conclude "reads nothing" from an override it cannot see into. Set
    beside the flag it only ever adds instances to the checked set, and it has
    to: the sibling shape that assigns writeable ``_M``, ``_C`` and ``_b`` and
    inherits all three readers ran the root's ``return self._M`` over storage
    no guard had looked at, verified as affine, and displaced the gradient of
    the C-15.7 fixture by the full ``0.0770625``. The absent-buffer case here
    is the same rule reaching a harmless instance, and it is refused for the
    same reason rather than by a second judgement about which absences matter.
    """

    class SkipsTheInitialiser(AffineDynamics):
        def __init__(self) -> None:
            pass

    problem = SkipsTheInitialiser()

    with pytest.raises(MissingCoefficients, match="_M") as raised:
        affine_dynamics_verified(problem)
    assert isinstance(raised.value, InvalidCoefficients)
    assert not isinstance(raised.value, MutableCoefficients)
    # The remedy differs from the post-construction case, so the message must
    # not claim an initialiser that never ran.
    assert "did not run AffineDynamics.__init__" in str(raised.value)


class InheritsTheReadersWithoutTheInitialiser(AffineDynamics):
    """Assigns the root's buffer names itself and inherits every reader.

    Nothing about this is exotic. It is what a subclass looks like when its
    author builds the coefficients some other way -- from a file, a mesh, a
    parent object -- and sees no reason to route them through
    ``super().__init__``. The arrays are plain and writeable, and the
    inherited ``f``, ``F`` and ``G`` hand them to the tape by identity.

    With the recorded flag as the *sole* precondition this instance was
    invisible: no flag, so no check, so no refusal, while every member that
    reads a buffer was the root's own.
    """

    def __init__(self, c: float) -> None:
        self._M = np.array(M_CONST, dtype=float)
        self._C = np.array([[c]], dtype=float)
        self._b = np.zeros(1)
        self._nu = 1


def test_inheriting_a_reader_obliges_the_buffer_it_reads() -> None:
    """The flag is not the only way to acquire the obligation.

    Measured on the C-15.7 fixture at ``C = 2``: without this the default
    optimizer returned ``0.7552125`` where the closed form gives ``0.6781500``
    -- the ``0.0770625`` displacement of an adjoint seeded with a matrix the
    forward solve never used. The setup is the module docstring's, with ``C``
    doubled; ``_closed_form`` derives both numbers so neither is pinned as a
    literal.
    """
    problem = InheritsTheReadersWithoutTheInitialiser(2.0)
    assert problem.F(Y0, U[:1], 0.0) is problem._M
    assert problem._M.flags.writeable

    with pytest.raises(MutableCoefficients, match="_M"):
        require_immutable_coefficients(problem)
    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)
    with pytest.raises(MutableCoefficients, match="_M"):
        GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        )
    with pytest.raises(MutableCoefficients, match="_M"):
        forward_solve(Y0, U, T_SPAN, N_STEPS, problem, explicit_euler(), None)


class ForgesItsMRO(type):
    """Answers ``__mro__`` with one that does not contain ``AffineDynamics``."""

    @property
    def __mro__(cls) -> tuple[type, ...]:
        return (object,)


class ForgesItsClassDict(type):
    """Answers ``__dict__`` with an empty mapping on every class in the MRO."""

    @property
    def __dict__(cls) -> dict[str, Any]:  # type: ignore[override]
        return {}


def test_a_metaclass_cannot_forge_the_mro_the_precondition_reads() -> None:
    """``cls.__mro__`` is attribute access on a class, so the metaclass answers.

    The reader widening is a question about the MRO, and a metaclass reporting
    one without ``AffineDynamics`` in it left ``_live_buffer_names`` empty
    while Python's own method lookup went on resolving ``f``, ``F`` and ``G``
    to the root's definitions. The gradient was displaced by the usual
    ``0.0770625`` with all three guards quiet.

    Reading the MRO through ``type``'s own getset descriptor gets the real
    one. ``type`` is not a class any problem under test can shadow, which is
    the same move that made the *instance* ``__dict__`` unforgeable.

    This is not the dismissed case of patching ``AffineDynamics`` itself.
    There, both sides of every comparison change together and the class really
    does bind what the check is told; here the class binds one thing and
    reports another.
    """

    class ForgesItsMRO(type):
        @property
        def __mro__(cls) -> tuple[type, ...]:
            return (object,)

    forging = ForgesItsMRO(
        "ForgingSubclass", (AffineDynamics,), {"__module__": __name__}
    )
    problem = object.__new__(forging)
    problem._M = np.array(M_CONST, dtype=float)
    problem._C = np.array([[2.0]], dtype=float)
    problem._b = np.zeros(1)
    problem._nu = 1

    # The forgery works, and method dispatch is unaffected by it: a walk of
    # the reported MRO finds no root, while ``F`` hands back the writeable
    # buffer exactly as before.
    assert AffineDynamics not in forging.__mro__
    assert problem.F(Y0, U[:1], 0.0) is problem._M

    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)


def test_a_metaclass_cannot_forge_the_class_dict_the_comparison_reads() -> None:
    """The same hole, one level down, and it certifies rather than skips.

    ``klass.__dict__`` inside the MRO walk is also attribute access on a
    class. A metaclass answering it with ``{}`` hides the subclass's *own*
    bindings, so the walk falls through to ``AffineDynamics`` and reports the
    root's ``f`` for a class that really binds a nonlinear one — verification
    certifies a problem that is not affine, and C-16.6 then drops the
    curvature terms of a system that has them.

    The existing metaclass regression in ``tests/test_instance_shadowing.py``
    covers the ``__getattribute__`` route into the same place. This is the
    property route, which that one does not reach.
    """

    class ForgesItsClassDict(type):
        @property
        def __dict__(cls) -> dict[str, Any]:  # type: ignore[override]
            return {}

    def nonlinear(self: Any, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._M @ y + self._C @ u + 5.0 * y**2)

    forging = ForgesItsClassDict(
        "ForgingSubclass",
        (AffineDynamics,),
        {"__module__": __name__, "f": nonlinear},
    )
    problem = forging(M_CONST, C_CONST)

    # The forgery works: the class reports none of its own bindings.
    assert dict(forging.__dict__) == {}
    assert _defined_as(forging, "f") is nonlinear
    assert not affine_dynamics_verified(problem)


def test_an_object_dtype_coefficient_is_refused() -> None:
    """``writeable=False`` on an ``object`` array covers the references only.

    The elements behind them stay mutable, so a frozen, owning buffer held
    every guarantee this clause asserts while its single element could still
    be rewritten between the forward solve and the adjoint — the full
    ``0.0770625`` displacement, both flags intact throughout. There is no
    depth at which following the references ends, so the dtype is refused.

    Unreachable through ``AffineDynamics(M, C, b)``, which casts to ``float``.
    It arrives on the path that skips the initialiser and assigns the buffers
    directly, which is the path the reader widening newly obliges.
    """

    class HoldsObjects(AffineDynamics):
        def __init__(self) -> None:
            payload = np.empty((1, 1), dtype=object)
            payload[0, 0] = np.array(M_CONST[0, 0])
            payload.flags.writeable = False
            self._M = payload
            self._C = np.array([[2.0]], dtype=float)
            self._C.flags.writeable = False
            self._b = np.zeros(1)
            self._b.flags.writeable = False
            self._nu = 1

    problem = HoldsObjects()
    assert not problem._M.flags.writeable
    assert problem._M.flags.owndata

    with pytest.raises(MutableCoefficients, match="_M") as raised:
        affine_dynamics_verified(problem)
    assert "object" in str(raised.value)


class MasksPartOfItsCoefficient(AffineDynamics):
    """Holds ``_M`` as a ``MaskedArray``, assigned rather than constructed.

    The class inherits ``F``, which hands ``_M`` to the tape by identity, so
    the buffer is live. Nothing here is a numpy trick — a masked coefficient
    is an ordinary way to say which entries of a state matrix are
    structurally absent — and that is the difficulty: the freeze seals the
    data and leaves the mask writeable, so the entry the coefficient claims
    is absent can become present after the check.
    """

    def __init__(self) -> None:
        self._M = np.ma.masked_array(M_CONST, mask=[[True]], dtype=float)

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def control_dim(self) -> int:
        return 1

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(np.ma.filled(np.ma.asarray(self._M), 0.0) @ y)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.zeros((1, 1))


def test_an_ndarray_subclass_coefficient_is_refused() -> None:
    """``writeable = False`` seals the array, not everything it carries.

    A ``MaskedArray``'s mask is a separate, ordinary, writeable array. Freeze
    the buffer and the mask still turns entries on and off, which changes what
    the coefficient means between the forward solve and the adjoint — the
    C-15.7 defect exactly, reached without any flag being touched.

    Measured before the cure on the module fixture at ``C = 2``, with ``_M``
    a masked ``0.4`` deep-copied so that it verified as affine with its data
    frozen and owning: flipping ``_M.mask[0, 0]`` inside the objective gave a
    gradient of ``0.7706250`` against the exact ``0.6781500``. The size of the
    displacement depends on how masked arithmetic degrades and is not the
    claim; that nothing said anything is.

    There is no general way to enumerate what an arbitrary subclass keeps, so
    the type must be ``ndarray`` exactly.
    """
    problem = MasksPartOfItsCoefficient()
    assert isinstance(problem._M, np.ma.MaskedArray)
    problem._M.flags.writeable = False
    assert problem._M.mask.flags.writeable

    for raises in (
        lambda: require_immutable_coefficients(problem),
        lambda: affine_dynamics_verified(problem),
        lambda: GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ),
    ):
        with pytest.raises(MutableCoefficients, match="_M") as raised:
            raises()
        assert "MaskedArray" in str(raised.value)


def test_the_root_initialiser_refuses_an_ndarray_subclass_argument() -> None:
    """The constructor names it rather than converting it.

    ``np.array(M, dtype=float)`` would discard the mask, so the coefficient
    the caller passed becomes a different matrix with nobody told. That is the
    same silent normalisation restoration was fixed not to do; refusing is one
    rule with the check, which cannot certify such a buffer in any case.

    Ordinary sequences lose nothing by conversion and still are converted.
    """
    masked = np.ma.masked_array(M_CONST, mask=[[True]], dtype=float)
    with pytest.raises(ValueError, match="ndarray subclass"):
        AffineDynamics(masked, C_CONST)
    with pytest.raises(ValueError, match="ndarray subclass"):
        AffineDynamics(M_CONST, masked)
    with pytest.raises(ValueError, match="ndarray subclass"):
        AffineDynamics(M_CONST, C_CONST, np.ma.masked_array([0.0]))

    from_lists = AffineDynamics([[0.4]], [[1.0]], [0.0])
    assert type(from_lists._M) is np.ndarray
    assert not from_lists._M.flags.writeable


def test_the_defensive_copy_preserves_the_subclass_it_copies() -> None:
    """Restoration preserves faithfully so the check can refuse loudly.

    The copy must not run the buffer's own ``copy`` — a subclass may define it
    and one returning ``self`` froze the shared source. Achieving that with
    ``subok=False`` went too far the other way: it silently turned a
    ``MaskedArray`` into a base array, which is to say it turned a buffer the
    check is required to refuse into one that passes, and the restored object
    then computed different dynamics from its original with every guard quiet.
    With ``M = 0.4`` masked and ``y = 0.8`` the original's ``f`` returns
    ``0.0`` and a mask-stripped duplicate's returns ``0.32``.

    ``np.array(..., copy=True, subok=True)`` dispatches to no user-defined
    ``copy``. It does run ``__array_finalize__``, which no subclass-preserving
    construction avoids; that is not part of the safety argument, because the
    buffer it produces is refused by name.
    """
    original = MasksPartOfItsCoefficient()
    restored = copy.deepcopy(original)

    assert isinstance(restored._M, np.ma.MaskedArray)
    assert np.array_equal(restored._M.mask, original._M.mask)
    assert np.array_equal(
        restored.f(Y0, U[0], 0.0), original.f(Y0, U[0], 0.0)
    )
    assert not restored._M.flags.writeable
    assert not np.shares_memory(restored._M, original._M)

    with pytest.raises(MutableCoefficients, match="_M"):
        require_immutable_coefficients(restored)


def test_the_obligation_is_only_the_buffers_the_inherited_readers_read() -> None:
    """``F`` owes ``_M``. It does not owe ``_C`` or ``_b``.

    A flat "any reader inherited means all three are live" rule would refuse
    this subclass for buffers no member of it can reach, which is the
    over-refusal C-16.2 asks to be paid for only where a wrong answer is
    otherwise available.
    """

    class InheritsOnlyF(AffineDynamics):
        def __init__(self) -> None:
            self._M = np.array(M_CONST, dtype=float)
            self._own_c = np.array(C_CONST, dtype=float)

        @property
        def state_dim(self) -> int:
            return 1

        @property
        def control_dim(self) -> int:
            return 1

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(self._M @ y + self._own_c @ u)

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return self._own_c.copy()

    problem = InheritsOnlyF()
    with pytest.raises(MutableCoefficients, match="_M") as raised:
        affine_dynamics_verified(problem)
    assert "_C" not in str(raised.value)

    problem._M.flags.writeable = False
    require_immutable_coefficients(problem)


class MaterialisesLate(AffineDynamics):
    """Ships ``_M`` under another name and rebuilds it on first access.

    Nothing in the restored state is called ``_M``. A rule reading the state
    would find nothing to freeze; asking the *object* runs ``__getattr__`` and
    the buffer exists as a consequence of being asked for.

    Module level because pickle cannot reach a class defined in a function.
    """

    def __init__(self, c: float) -> None:
        super().__init__(M_CONST, [[c]])
        self._payload = np.array(M_CONST, dtype=float)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_payload": self._payload,
            "_C": self._C,
            "_b": self._b,
            "_nu": self._nu,
        }

    def __getattr__(self, name: str) -> Any:
        if name == "_M":
            buffer = np.array(self._payload, dtype=float)
            object.__setattr__(self, "_M", buffer)
            return buffer
        raise AttributeError(name)


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_a_lazily_built_coefficient_is_materialised_and_frozen(
    duplicate: Callable[[Any], Any],
) -> None:
    """Asking the object is not a passive read, and that is the point.

    ``getattr`` runs whatever the object does to answer, lazy construction
    included, so a coefficient that exists in no restored key is built,
    copied and frozen here rather than appearing writeable later. A rule that
    read the restored *state* would have found nothing named ``_M`` and left
    the inherited ``F`` handing out a writeable array.

    The cost of asking is that the buffer is built during restoration instead
    of on first use. Paying it is the trade: this clause exists because what
    the solve will read is the only thing worth checking.
    """
    source = MaterialisesLate(2.0)
    exact, _ = _closed_form(2.0)

    restored = duplicate(source)
    storage = AffineDynamics.__dict__["__dict__"].__get__(restored)
    assert "_M" not in source.__getstate__()
    assert "_M" in storage
    assert not storage["_M"].flags.writeable
    assert storage["_M"].flags.owndata
    assert restored.F(Y0, U[:1], 0.0) is storage["_M"]

    gradient = GLMOptimizer(
        restored,
        OverwritesCoefficientMidSolve(None),
        explicit_euler(),
        T_SPAN,
        N_STEPS,
        Y0,
    ).gradient(U)
    assert np.asarray(gradient) == pytest.approx(exact, abs=EXACT_TOL)

    # And the freeze is the real one: the mutation this module measures is
    # refused aloud on the restored object.
    with pytest.raises(ValueError, match="read-only"):
        restored._M[0, 0] = M_OVERWRITTEN


def test_a_coefficient_that_is_never_materialised_is_refused_by_name() -> None:
    """Lazily *absent* is still absent, and is named rather than deferred.

    The inherited ``f`` reads all three buffers. A restoration leaving one
    unreachable is refused at the route decision naming it, instead of raising
    ``AttributeError`` from inside the first step.
    """

    class KeepsOnlyM(MaterialisesLate):
        def __getstate__(self) -> dict[str, Any]:
            return {"_payload": self._payload, "_nu": self._nu}

    restored = copy.copy(KeepsOnlyM(2.0))
    with pytest.raises(MissingCoefficients, match="_C") as raised:
        affine_dynamics_verified(restored)
    assert "did not run AffineDynamics.__init__" in str(raised.value)


def test_the_defensive_copy_does_not_dispatch_to_the_buffer() -> None:
    """Restoration copies through ``np.array``, not ``buffer.copy()``.

    ``isinstance(..., np.ndarray)`` admits subclasses, and ``.copy()`` is
    theirs to define. One returning ``self`` put the freeze back onto the
    shared source that the copy-before-freeze branch exists to protect; one
    perturbing the data changed a coefficient by ``0.5`` in silence, which is
    the C-7 direction that must not be available.
    """

    class ReturnsItself(np.ndarray):
        def copy(self, order: str = "C") -> ReturnsItself:
            return self

    class PerturbsOnCopy(np.ndarray):
        def copy(self, order: str = "C") -> NDArray:
            return np.asarray(np.asarray(self) + 0.5)

    for subclass in (ReturnsItself, PerturbsOnCopy):

        class KeepsItsOwn(AffineDynamics):
            storage = subclass

            def __init__(self) -> None:
                self._M = np.array(M_CONST, dtype=float).view(self.storage)

            def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
                return np.asarray(np.asarray(self._M) @ y)

            def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
                return np.asarray(self._M).copy()

            def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
                return np.zeros((1, 1))

        source = KeepsItsOwn()
        restored = copy.copy(source)

        assert source._M.flags.writeable, "the source must not be frozen"
        assert not np.shares_memory(source._M, restored._M)
        assert np.asarray(restored._M) == pytest.approx(np.asarray(M_CONST))


def test_partial_coefficient_storage_is_refused_by_name() -> None:
    """All of ``_COEFFICIENT_BUFFERS`` or none: a partial set is taken apart.

    ``AffineDynamics.__init__`` establishes the three together, so an instance
    holding some but not others has been modified after construction. It is
    distinguishable from the absent-entirely case above -- which is quiet --
    and is reported at the route decision rather than several frames later.
    """
    problem = AffineDynamics(M_CONST, C_CONST)
    del problem.__dict__["_b"]

    with pytest.raises(MissingCoefficients, match="_b") as raised:
        affine_dynamics_verified(problem)
    assert isinstance(raised.value, InvalidCoefficients)
    assert not isinstance(raised.value, MutableCoefficients)


# --------------------------------------------------------------------------
# The precondition: the check applies only where the buffers are read
# --------------------------------------------------------------------------


class ReadsNoneOfTheRootBuffers(AffineDynamics):
    """Numerically identical to ``AffineDynamics(M_CONST, C_CONST)``, but it
    reads its own **writeable** arrays and never touches the root's buffers.

    ``f``, ``F`` and ``G`` are the only members that hand one back, so a
    subclass replacing all three makes the root's buffers dead storage. It
    never calls ``super().__init__``, so it has none at all.
    """

    def __init__(self) -> None:
        self._m = np.array(M_CONST, dtype=float)
        self._c = np.array(C_CONST, dtype=float)

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def control_dim(self) -> int:
        return 1

    @property
    def coefficients_constant(self) -> bool:
        return True

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._m @ y + self._c @ u)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._m.copy()

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._c.copy()


#: Module level so the duplication tests below can pickle them. Each answers
#: for ``_M`` through a property, which is the shape that separated the two
#: liveness questions: a descriptor can refuse, and it can decline assignment.
_FROZEN_ANSWER = np.array(M_CONST, dtype=float)
_FROZEN_ANSWER.flags.writeable = False
_WRITEABLE_ANSWER = np.array(M_CONST, dtype=float)


class RefusesToAnswer(ReadsNoneOfTheRootBuffers):
    probes = 0

    @property
    def _M(self) -> NDArray:
        type(self).probes += 1
        raise RuntimeError("this class keeps no root _M")


class AnswersWithAFrozenBuffer(ReadsNoneOfTheRootBuffers):
    @property
    def _M(self) -> NDArray:
        return _FROZEN_ANSWER


class AnswersWithAWriteableBuffer(ReadsNoneOfTheRootBuffers):
    @property
    def _M(self) -> NDArray:
        return _WRITEABLE_ANSWER


class SparseBuildsAndDelegates(AffineDynamics):
    """Builds three coefficients itself and wraps all three readers, so it is
    invisible to the marker and to reader identity alike: only presence sees
    it, and only because a name resolving to anything at all is enough.
    """

    def __init__(self) -> None:
        self._M = csr_array(np.array(M_CONST, dtype=float))
        self._C = csr_array(np.array(C_DOUBLED, dtype=float))
        self._b = np.zeros(1)
        self._b.flags.writeable = False
        self._nu = 1

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return super().f(y, u, t)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return super().F(y, u, t)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return super().G(y, u, t)


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_restoration_and_the_guard_ask_a_refusing_descriptor_the_same_way(
    duplicate: Callable[[Any], Any],
) -> None:
    """One question asked twice must be asked the same way, C-15.7.

    Both the guard and restoration ask what each coefficient name hands out,
    and they asked it differently: the guard absorbed any exception, while
    restoration used ``getattr`` with a default, which absorbs only
    ``AttributeError``. A general-route subclass whose ``_M`` property raises
    anything else was therefore accepted by the optimizer -- it computes the
    closed-form gradient below -- and could not be copied, deep-copied or
    pickled at all.

    That is the asymmetry's harmless direction. The other one is not: a name
    live to the guard and invisible to restoration leaves a copy unfrozen,
    which is why the two now share :func:`_resolve_buffer`.
    """
    RefusesToAnswer.probes = 0

    exact, _ = _closed_form(1.0)
    assert _gradient(
        RefusesToAnswer(), OverwritesCoefficientMidSolve(None)
    ) == pytest.approx(exact, abs=EXACT_TOL)
    assert RefusesToAnswer.probes, "the guard must have asked, or this proves nothing"

    RefusesToAnswer.probes = 0
    restored = duplicate(RefusesToAnswer())
    assert RefusesToAnswer.probes, (
        "restoration must have asked, or this proves nothing"
    )
    assert _gradient(
        restored, OverwritesCoefficientMidSolve(None)
    ) == pytest.approx(exact, abs=EXACT_TOL)


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_a_buffer_already_at_the_postcondition_is_left_where_it_is(
    duplicate: Callable[[Any], Any],
) -> None:
    """Restoration replaces a buffer to gain something, or not at all.

    An unmarked buffer is copied before it is frozen because freezing it
    where it lies would reach back through the sharing a copy leaves. When it
    is already frozen and already owns its storage there is nothing to reach
    back to -- the freeze is a no-op -- and replacing it cost the one thing
    this branch can lose: a read-only property answering for a coefficient
    cannot be assigned to, so an ordinary round trip of a subclass the
    optimizer accepts raised ``property '_M' of ... has no setter``.
    """
    exact, _ = _closed_form(1.0)
    restored = duplicate(AnswersWithAFrozenBuffer())
    assert restored._M is _FROZEN_ANSWER
    assert not restored._M.flags.writeable
    assert _gradient(
        restored, OverwritesCoefficientMidSolve(None)
    ) == pytest.approx(exact, abs=EXACT_TOL)


def test_a_writeable_buffer_that_cannot_be_replaced_is_refused_aloud() -> None:
    """And the source is left as the caller had it.

    The same read-only property over a *writeable* array has no safe outcome:
    the copy cannot be stored, and freezing the original would change an
    object the caller did not ask to change. Restoration refuses, names the
    buffer, and says why -- rather than letting a bare ``AttributeError`` out
    of ``copy.deepcopy``, and rather than freezing the source.
    """
    with pytest.raises(MutableCoefficients, match="_M") as raised:
        copy.deepcopy(AnswersWithAWriteableBuffer())
    assert "cannot be replaced with a frozen copy" in str(raised.value)
    assert _WRITEABLE_ANSWER.flags.writeable, "the caller's array is untouched"


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_restoration_refuses_the_storage_the_guard_refuses(
    duplicate: Callable[[Any], Any],
) -> None:
    """The converse of the rule above, and the direction that is not harmless.

    Restoration obliged only names resolving to an ``ndarray``, so the
    delegating subclass holding ``csr_array`` coefficients round-tripped
    quietly and handed back an object whose ``_M.data`` was writeable. The
    guard refuses that instance; a duplication route that does not is the
    same rule binding in one place and not the other.
    """

    with pytest.raises(MissingCoefficients, match="_M") as raised:
        duplicate(SparseBuildsAndDelegates())
    assert "csr_array" in str(raised.value)


class InheritsFOverARefusingDescriptor(AffineDynamics):
    """Inherits ``F``, so reader identity obliges ``_M``; the descriptor then
    refuses to answer, so presence cannot see it. Module level so the
    duplication routes can pickle it.
    """

    probes = 0

    def __init__(self) -> None:
        self._own_c = np.array(C_CONST, dtype=float)

    @property
    def _M(self) -> NDArray:
        type(self).probes += 1
        raise RuntimeError("this class keeps no root _M")

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def control_dim(self) -> int:
        return 1

    @property
    def coefficients_constant(self) -> bool:
        return True

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._own_c @ u)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._own_c.copy()


def test_an_obliged_name_that_refuses_to_answer_is_refused_by_name() -> None:
    """The guard's last read went round the shared rule, C-15.7.

    ``_root_buffers`` restated it as a bare ``getattr``, which absorbs only
    ``AttributeError``. Reader identity obliges ``_M`` here because ``F`` is
    inherited, and presence cannot see it because the descriptor refuses; so
    the name is live and unresolved, exactly the case the reader map exists
    for. The route decision let the descriptor's ``RuntimeError`` out instead
    of naming the buffer -- loud, but not the contract, and it made "literally
    the same function" false at the one call site that decides.

    Restating a shared rule is how it stops being shared. Every read of a
    coefficient now goes through ``_resolve_buffer``.
    """
    problem = InheritsFOverARefusingDescriptor()
    InheritsFOverARefusingDescriptor.probes = 0

    for refuse in (
        lambda: require_immutable_coefficients(problem),
        lambda: affine_dynamics_verified(problem),
        lambda: _gradient(problem, OverwritesCoefficientMidSolve(None)),
    ):
        with pytest.raises(MissingCoefficients, match="_M") as raised:
            refuse()
        assert "is absent" in str(raised.value)
    assert InheritsFOverARefusingDescriptor.probes, "the descriptor was asked"


def test_the_refusal_to_replace_names_the_reason_it_read() -> None:
    """A failed replacement reports the buffer it found, not the common case.

    The copy-before-freeze branch takes two kinds of buffer: one still
    writeable, and one frozen that does not own its storage. The message
    named the first for both, sending a caller holding a frozen view after a
    remedy that does not apply to it. Each is asserted against the other's
    wording so neither can drift back.
    """
    base = np.array([[0.4, 0.0], [0.0, 0.0]], dtype=float)
    view = base[:1, :1]
    view.flags.writeable = False
    assert not view.flags.owndata and not view.flags.writeable

    class AnswersWithAFrozenView(ReadsNoneOfTheRootBuffers):
        @property
        def _M(self) -> NDArray:
            return view

    with pytest.raises(MutableCoefficients, match="_M") as frozen_view:
        copy.deepcopy(AnswersWithAFrozenView())
    assert "does not own its storage" in str(frozen_view.value)
    assert "writeable, and freezing it" not in str(frozen_view.value)

    with pytest.raises(MutableCoefficients, match="_M") as writeable:
        copy.deepcopy(AnswersWithAWriteableBuffer())
    assert "writeable, and freezing it" in str(writeable.value)
    assert "does not own its storage" not in str(writeable.value)


def test_a_subclass_reading_none_of_the_root_buffers_is_left_alone() -> None:
    """The false-refusal control for running the check first.

    Gating on ``isinstance(problem, AffineDynamics)`` alone assumed that every
    instance of that class uses its buffers. It does not, and this subclass —
    a perfectly good general-route problem — was refused outright with
    ``MissingCoefficients``, which is a regression dressed as a guard. What
    makes it safe is that it holds no root buffers at all: never having run
    ``AffineDynamics.__init__``, it has nothing the tape could alias.

    Note what this fixture does *not* establish. Overriding ``f``, ``F`` and
    ``G`` is not itself what makes it safe — see the delegating subclass below,
    which overrides the same three and reads every buffer through ``super()``.

    Asserting the *gradient* rather than only the absence of a raise, because
    "it did not raise" would also pass if the problem were quietly broken.
    """
    problem = ReadsNoneOfTheRootBuffers()
    assert not affine_dynamics_verified(problem)

    exact, _ = _closed_form(1.0)
    gradient = GLMOptimizer(
        problem,
        OverwritesCoefficientMidSolve(None),
        explicit_euler(),
        T_SPAN,
        N_STEPS,
        Y0,
    ).gradient(U)
    assert np.asarray(gradient) == pytest.approx(exact, abs=EXACT_TOL)


class DelegatesEveryBufferReader(AffineDynamics):
    """Overrides ``f``, ``F`` and ``G`` — and reads every buffer regardless.

    An instrumenting or unit-converting subclass looks exactly like this. Each
    override is a distinct function object, so no comparison of method identity
    against the root can tell it apart from one that replaces the bodies.
    """

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return super().f(y, u, t)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return super().F(y, u, t)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return super().G(y, u, t)


def test_a_delegating_override_is_still_checked() -> None:
    """Method identity cannot witness what a method reads — C-15.7.

    The precondition was briefly "do any of ``f``, ``F``, ``G`` still resolve
    to ``AffineDynamics``' own definitions?". This subclass answers no to all
    three and reads all three buffers anyway, so the check was skipped and the
    displaced gradient returned in silence: ``0.7552125`` against an exact
    ``0.6781500`` on the committed two-step Euler fixture at ``C = 2``.

    The precondition is now the presence of the root's storage, which
    delegation cannot hide. The control is the same class left frozen, which
    must still reach the closed form — without it this test would pass against
    a check that refused every subclass.
    """
    exact, displacement = _closed_form(2.0)
    assert displacement == pytest.approx(0.0770625, abs=EXACT_TOL)

    intact = DelegatesEveryBufferReader(M_CONST, np.array([[2.0]]))
    assert np.asarray(
        GLMOptimizer(
            intact,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ).gradient(U)
    ) == pytest.approx(exact, abs=EXACT_TOL)

    problem = DelegatesEveryBufferReader(M_CONST, np.array([[2.0]]))
    problem.__dict__["_M"].flags.writeable = True
    objective = OverwritesCoefficientMidSolve(problem)

    with pytest.raises(MutableCoefficients, match="_M"):
        GLMOptimizer(
            problem, objective, explicit_euler(), T_SPAN, N_STEPS, Y0
        )
    assert not objective.fired


class RedirectsIntoASlot(AffineDynamics):
    """A subclass whose state populates ``_M`` under a *different* name.

    ``payload`` is an ordinary property whose setter writes the ``_M`` slot,
    and ``__getstate__`` ships the buffer under that name. ``_M`` is therefore
    live on the restored instance and aliased by the inherited ``F``, while
    appearing nowhere among the restored keys. Deciding what to freeze from
    those keys missed it on every duplication route.
    """

    __slots__ = ("_M",)

    @property
    def payload(self) -> NDArray:
        return self._M

    @payload.setter
    def payload(self, value: NDArray) -> None:
        object.__setattr__(self, "_M", value)

    def __getstate__(self) -> tuple[dict[str, Any], dict[str, NDArray]]:
        return (
            {"_C": self._C, "_b": self._b, "_nu": self._nu},
            {"payload": self._M.copy()},
        )


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_restoration_reads_the_object_not_the_state_it_arrived_in(
    duplicate: Callable[[AffineDynamics], AffineDynamics],
) -> None:
    """Restoration asks the restored object, because state is not provenance.

    Three rules were tried that decided what to freeze from the state, and a
    subclass defeated each by shaping it: the marker can be dropped from
    ``__getstate__``, a buffer can be omitted alongside it, and — here — a
    coefficient can be delivered under another name entirely, by a property
    setter that writes the slot. The buffer is live and inherited ``F``
    aliases it; it is simply not among the keys.

    The state is written by the subclass and the object is what the solve will
    read, so the object is what restoration reads: every coefficient name that
    resolves to an array on the finished instance is frozen. That is the same
    rule the check uses, and it asks nobody anything.
    """
    duplicated = duplicate(RedirectsIntoASlot(M_CONST, C_DOUBLED))

    assert duplicated.F(Y0, U[:1], 0.0) is duplicated._M
    assert not duplicated._M.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        duplicated._M[0, 0] = M_OVERWRITTEN


class SerialisesOnlyItsCoefficients(AffineDynamics):
    """A root instance whose ``__getstate__`` drops the root-initialised flag.

    Overriding a copy hook is grounds for a *quiet* refusal of the affine
    route, but this instance still holds the root's buffers and still inherits
    ``F``, so the tape aliases ``self._M`` whichever route runs. Its
    restoration obligation is therefore unchanged by its ineligibility.
    """

    def __getstate__(self) -> dict[str, NDArray]:
        return {"_M": self._M, "_C": self._C, "_b": self._b}


class SerialisesTwoOfThree(AffineDynamics):
    """The same, omitting ``_b`` as well as the flag.

    ``f`` is overridden so that the missing ``_b`` is never read and the
    problem still solves; ``F`` is the root's, so ``_M`` is still retained by
    identity. A rule requiring all three names to arrive before re-freezing
    skipped this and restored ``_M`` writeable.
    """

    def __getstate__(self) -> dict[str, NDArray]:
        return {"_M": self._M, "_C": self._C}

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._M @ y + self._C @ u)


class SerialisesOneOfThree(AffineDynamics):
    """The same again, down to a single name.

    This is the shape that settles the question. By the restored names alone
    it is indistinguishable from ``ReusesTheName`` below, which keeps its own
    writeable ``_M`` and is entitled to. One retains the root's buffer and one
    does not, and no rule counting names can tell them apart — so restoration
    stops counting and freezes the name either way.
    """

    def __getstate__(self) -> dict[str, NDArray]:
        return {"_M": self._M}

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._M @ y + C_DOUBLED @ u)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return C_DOUBLED


@pytest.mark.parametrize(
    ("factory", "serialised"),
    [
        pytest.param(SerialisesOnlyItsCoefficients, ("_M", "_C", "_b"), id="three"),
        pytest.param(SerialisesTwoOfThree, ("_M", "_C"), id="two"),
        pytest.param(SerialisesOneOfThree, ("_M",), id="one"),
    ],
)
def test_restoration_does_not_trust_a_flag_a_subclass_can_drop(
    factory: type[AffineDynamics], serialised: tuple[str, ...]
) -> None:
    """What arrives decides; nothing the subclass answers does.

    ``__setstate__`` gated re-freezing on the restored marker, but the marker
    comes in through ``__getstate__``, which a subclass may replace. One
    returning only its coefficients dropped it, so a genuine root instance was
    restored writeable, the precondition concluded the root initialiser had
    never run, and all three call sites fell silent — the fixture's
    ``0.7552125`` against an exact ``0.6781500``.

    Requiring all three names instead was no better, which is why all three
    arities are exercised here: the same override omits a buffer as easily as
    it omits the marker, and the two-name form reproduced the identical
    displacement.

    Asserting that the retained buffer refuses the write, rather than only
    that it is marked read-only, because refusing the write is the property
    the tape depends on.
    """
    duplicated = copy.deepcopy(factory(M_CONST, C_DOUBLED))

    assert duplicated.F(Y0, U[:1], 0.0) is duplicated._M
    for name in serialised:
        buffer = getattr(duplicated, name)
        assert not buffer.flags.writeable
        assert buffer.flags.owndata
    with pytest.raises(ValueError, match="read-only"):
        duplicated._M[0, 0] = M_OVERWRITTEN


def test_a_full_restoration_records_the_flag_itself() -> None:
    """Where all three arrived the root established them, so the copy says so.

    Passing on the marker it was handed would leave a later check reading what
    a subclass chose to serialise; recording it leaves the check reading what
    restoration established. The duplicate is then indistinguishable from a
    freshly constructed instance, which is what a copy should be.
    """
    duplicated = copy.deepcopy(SerialisesOnlyItsCoefficients(M_CONST, C_CONST))

    storage = AffineDynamics.__dict__["__dict__"].__get__(duplicated)
    assert storage["_affine_root_initialised"] is True

    exact, _ = _closed_form(1.0)
    gradient = GLMOptimizer(
        duplicated,
        OverwritesCoefficientMidSolve(None),
        explicit_euler(),
        T_SPAN,
        N_STEPS,
        Y0,
    ).gradient(U)
    assert np.asarray(gradient) == pytest.approx(exact, abs=EXACT_TOL)


class CopiesIntoItsOwnStorage(AffineDynamics):
    """A verified subclass whose ``_M`` setter stores a *copy* of its argument.

    Writing ``self._M = value`` therefore leaves ``self._M`` bound to an object
    that is equal to ``value`` but is not ``value``. Preserving the argument's
    ``writeable`` flag is what makes it verify after ordinary construction.
    """

    @property
    def _M(self) -> NDArray:  # type: ignore[override]
        return self.__dict__["_store"]

    @_M.setter
    def _M(self, value: NDArray) -> None:
        stored = np.array(value, dtype=float)
        stored.flags.writeable = value.flags.writeable
        self.__dict__["_store"] = stored


class RebuildsAnotherCoefficient(AffineDynamics):
    """A verified subclass whose ``_C`` setter also rebuilds ``_M``.

    Nothing forbids one coefficient's setter from touching another, and this
    one preserves each array's ``writeable`` flag, so ordinary construction
    leaves all three frozen and the instance verifies.
    """

    @property
    def _C(self) -> NDArray:  # type: ignore[override]
        return self.__dict__["_cstore"]

    @_C.setter
    def _C(self, value: NDArray) -> None:
        stored = np.array(value, dtype=float)
        stored.flags.writeable = value.flags.writeable
        self.__dict__["_cstore"] = stored
        if "_M" in self.__dict__:
            rebuilt = np.array(self.__dict__["_M"], dtype=float)
            rebuilt.flags.writeable = value.flags.writeable
            self.__dict__["_M"] = rebuilt


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(CopiesIntoItsOwnStorage, id="stores-a-copy"),
        pytest.param(RebuildsAnotherCoefficient, id="rebuilds-another"),
    ],
)
def test_the_replacement_buffer_is_frozen_as_the_object_resolves_it(
    factory: type[AffineDynamics],
    duplicate: Callable[[AffineDynamics], AffineDynamics],
) -> None:
    """Freezing the value handed to ``setattr`` is not freezing the buffer.

    Restoration replaces a non-owning buffer, and ``setattr`` may run a
    property setter that stores something else — a copy, in the first case
    here. The local name stayed bound to the object that was discarded, so the
    freeze landed on it and the array ``F`` returns came back owning and
    *writeable*: a subclass that verified before the round trip failed the
    check after it.

    The second case is why re-reading in place is not enough either. That
    setter rewrites ``_M`` as a side effect, so ``_M`` — frozen earlier in the
    same loop — was replaced while the loop was still running. Freezing had to
    become a second pass over all three, after no coefficient setter can still
    run.
    """
    original = factory(M_CONST, C_CONST)
    assert affine_dynamics_verified(original)

    duplicated = duplicate(original)

    for name in ("_M", "_C", "_b"):
        buffer = getattr(duplicated, name)
        assert not buffer.flags.writeable
        assert buffer.flags.owndata
    assert duplicated.F(Y0, U[:1], 0.0) is duplicated._M
    assert affine_dynamics_verified(duplicated)
    assert duplicated.M == pytest.approx(M_CONST, abs=EXACT_TOL)


class SlottedM(AffineDynamics):
    """A verified subclass that keeps ``_M`` in a slot rather than the dict.

    Declaring ``__slots__`` overrides no guaranteed member, so this shape is
    eligible for the affine route; ``F`` resolves ``self._M`` through the slot
    exactly as it would through the instance dictionary. Defined at module
    scope because the pickle round trip below cannot serialise a local class.
    """

    __slots__ = ("_M",)


def test_a_slotted_coefficient_is_read_through_its_descriptor() -> None:
    """``__slots__`` storage is present and frozen, and must not be refused.

    A slot of the same name shadows the instance dictionary, so reading the
    root's ``__dict__`` — through ``problem.__dict__`` or through the root's
    own getset descriptor — finds no ``_M`` and reported a perfectly valid,
    frozen buffer as missing. ``AffineDynamics.F`` is ``return self._M``, so
    plain attribute access is what it resolves, and that is what the check
    reads: the same object, obtained the same way.
    """
    problem = SlottedM(M_CONST, C_CONST)
    assert "_M" not in object.__getattribute__(problem, "__dict__")
    assert problem.F(Y0, U[:1], 0.0) is problem._M

    require_immutable_coefficients(problem)
    assert affine_dynamics_verified(problem)

    problem._M.flags.writeable = True
    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
    ],
)
def test_a_slotted_coefficient_survives_duplication(
    duplicate: Callable[[AffineDynamics], AffineDynamics],
) -> None:
    """Verified for one obligation of C-15.7 means verified for both.

    ``__setstate__`` read the coefficients out of the instance dictionary,
    where a subclass declaring ``__slots__ = ("_M",)`` keeps none of them, so
    a shape that verified and solved correctly raised ``KeyError('_M')`` on
    every copy and every pickle round trip. Restoration now resolves storage
    by the same rule verification does.

    Asserting the constructor's full postcondition on the duplicate — frozen
    *and* owning — rather than only that the operation completed, because
    completing is what it would also do if re-freezing were skipped entirely.
    """
    original = SlottedM(M_CONST, C_CONST)
    duplicated = duplicate(original)

    assert isinstance(duplicated, SlottedM)
    assert "_M" not in object.__getattribute__(duplicated, "__dict__")
    assert not duplicated._M.flags.writeable
    assert duplicated._M.flags.owndata
    assert affine_dynamics_verified(duplicated)
    assert duplicated.M == pytest.approx(M_CONST, abs=EXACT_TOL)


def test_a_subclass_without_root_storage_can_still_be_duplicated() -> None:
    """``__setstate__`` is inherited, so it runs on subclasses that owe it
    nothing.

    Re-freezing unconditionally demanded ``_M``, ``_C`` and ``_b`` on every
    copy of every subclass, including one that never called
    ``super().__init__`` and keeps its coefficients elsewhere — a valid
    general-route problem that could not be deep-copied. Re-freezing is now
    conditional on the same recorded marker the check is.
    """
    duplicated = copy.deepcopy(ReadsNoneOfTheRootBuffers())

    exact, _ = _closed_form(1.0)
    gradient = GLMOptimizer(
        duplicated,
        OverwritesCoefficientMidSolve(None),
        explicit_euler(),
        T_SPAN,
        N_STEPS,
        Y0,
    ).gradient(U)
    assert np.asarray(gradient) == pytest.approx(exact, abs=EXACT_TOL)


def test_a_coefficient_property_is_read_through_the_property() -> None:
    """The decoy case, and the reason the instance dictionary is not consulted.

    A subclass may resolve ``_M`` through a property while the root's
    ``__dict__`` still holds the array ``__init__`` froze. Reading the stored
    value then checks a decoy: it is frozen, the check passes, and ``F`` hands
    the tape the writeable array the property returns. Attribute access sees
    what ``F`` sees.
    """

    class PropertyM(AffineDynamics):
        @property  # type: ignore[misc]
        def _M(self) -> NDArray:
            return self._live

        @_M.setter
        def _M(self, value: NDArray) -> None:
            self.__dict__["_M"] = value
            self._live = np.array(value, dtype=float)

    problem = PropertyM(M_CONST, C_CONST)
    assert not problem.__dict__["_M"].flags.writeable
    assert problem.F(Y0, U[:1], 0.0) is problem._live

    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)


def test_an_unread_buffer_is_refused_too_and_that_cost_is_accepted() -> None:
    """The price of not inferring liveness from the override structure.

    This subclass overrides ``f``, the only member that reads ``_b``, so its
    ``_b`` is dead storage and unfreezing it changes no callback. It is
    refused anyway. Deciding otherwise would mean concluding from the override
    structure which buffer is still read, and the delegating subclass above is
    why that conclusion is unsound.

    The trade is deliberate and is stated in C-15.7: a loud refusal of a
    working problem, against never staying quiet about a broken one. Pinned
    here so that narrowing the check later is a visible decision rather than
    an accident. The control confirms the callbacks really are unaffected.
    """

    class OverridesOnlyF(AffineDynamics):
        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(self._M @ y + self._C @ u)

    problem = OverridesOnlyF(M_CONST, C_CONST)
    before = problem.f(Y0, U[:1], 0.0)
    problem.__dict__["_b"].flags.writeable = True

    assert problem.f(Y0, U[:1], 0.0) == pytest.approx(before, abs=EXACT_TOL)
    with pytest.raises(MutableCoefficients, match="_b"):
        affine_dynamics_verified(problem)


def test_unfreezing_after_the_check_is_outside_what_the_check_reaches() -> None:
    """`ENVELOPE`, C-15.7: a check at a moment, not a lifetime guarantee.

    ``flags.writeable`` is settable by anyone at any time, so a caller who
    completes a solve, sets it back to ``True`` and mutates rewrites a tape
    that is already built. No guard placed before a solve can see that, and
    the only construction that would is copying ``F``'s result at every step —
    the aliasing the design exists to avoid. In the C++ port the coefficients
    are ``const`` members and this is a compile error.

    Recorded as a test rather than as prose so that the boundary is measured
    and can fail: if a future change did start copying at retention, this test
    breaks and the envelope statement in C-15.7 has to be revisited. The
    control is the same sequence without the mutation.
    """
    c = 2.0
    exact, displacement = _closed_form(c)

    def run(mutate: bool) -> NDArray:
        problem = AffineDynamics(M_CONST, np.array([[c]]))
        optimizer = GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        )
        optimizer.objective_value(U)
        if mutate:
            problem.M.flags.writeable = True
            problem.M[0, 0] = M_OVERWRITTEN
        return np.asarray(optimizer.gradient(U))

    assert run(mutate=False) == pytest.approx(exact, abs=EXACT_TOL)

    displaced = run(mutate=True)
    assert displaced[0] == pytest.approx(exact[0] + displacement, abs=EXACT_TOL)
    assert displaced[1] == pytest.approx(exact[1], abs=EXACT_TOL)


def test_storage_replaced_by_something_unreadable_is_refused_loudly() -> None:
    """Unreadable is not the same as absent — C-7, C-15.7.

    Coefficients that the check could not interpret were once skipped, and a
    skipped buffer left nothing to check, so the precondition read "no root
    storage" and went quiet. Three sparse or buffer-protocol coefficients
    therefore verified as affine and were aliased by the tape unchecked.

    Once the root initialiser has recorded that it established three frozen
    owning arrays, anything else in their place is a replacement made after
    construction whose aliasing cannot be reasoned about, so it raises. The
    control is the same subclass leaving the buffers alone.
    """

    class ReplacesStorage(AffineDynamics):
        def __init__(self, replacement: object | None) -> None:
            super().__init__(M_CONST, C_CONST)
            if replacement is not None:
                self.__dict__["_M"] = replacement

    assert affine_dynamics_verified(ReplacesStorage(None))

    problem = ReplacesStorage(memoryview(bytearray(8)))
    with pytest.raises(MissingCoefficients, match="_M") as raised:
        affine_dynamics_verified(problem)
    assert "memoryview" in str(raised.value)
    assert not isinstance(raised.value, MutableCoefficients)


def test_the_marker_obliges_a_buffer_no_widening_can_see() -> None:
    """What the recorded flag still does that neither widening can.

    Both widenings look at something the instance currently *is*: which
    readers are the root's, and which names resolve. Neither survives a
    coefficient *removed* from a subclass that also wraps its readers --
    identity sees three overrides, presence has no name to resolve, and the
    buffer the delegating ``super().F`` will reach for is live to nobody.

    The flag is the only one of the three that remembers, so it is the only
    one that can say "this instance had three frozen arrays and now has two".
    Without it this shape is refused quietly for its overrides and fails
    later from inside the first step instead.
    """

    class DelegatesAndDropsStorage(AffineDynamics):
        def __init__(self) -> None:
            super().__init__(M_CONST, C_CONST)
            del self._M

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().f(y, u, t)

        def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().F(y, u, t)

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().G(y, u, t)

    problem = DelegatesAndDropsStorage()
    assert not hasattr(problem, "_M")
    for raises in (
        lambda: require_immutable_coefficients(problem),
        lambda: affine_dynamics_verified(problem),
    ):
        with pytest.raises(MissingCoefficients, match="_M"):
            raises()


def test_the_reader_map_obliges_a_buffer_presence_cannot_see() -> None:
    """And what the reader map does that presence cannot.

    Presence can only add a name that resolves. This subclass never ran the
    root initialiser, so there is no flag, and it holds no ``_M`` at all --
    so presence has nothing to say about the one buffer that matters. The
    inherited ``F`` will reach for it regardless, and knowing that is exactly
    what the reader map is for: the refusal names ``_M`` at the route
    decision instead of raising ``AttributeError`` inside the first step.
    """

    class InheritsOnlyFWithoutStorage(AffineDynamics):
        def __init__(self) -> None:
            self._own_c = np.array(C_CONST, dtype=float)

        @property
        def state_dim(self) -> int:
            return 1

        @property
        def control_dim(self) -> int:
            return 1

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(self._own_c @ u)

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return self._own_c.copy()

    problem = InheritsOnlyFWithoutStorage()
    assert not hasattr(problem, "_M")
    with pytest.raises(MissingCoefficients, match="_M") as raised:
        affine_dynamics_verified(problem)
    assert "did not run AffineDynamics.__init__" in str(raised.value)


def test_a_subclass_that_never_ran_the_root_initialiser_is_not_refused() -> None:
    """A descriptor that refuses to answer is probed, and hands out nothing.

    The presence widening asks whether ``_M`` resolves, and asking is not a
    passive read: a coefficient descriptor *is* executed, and one that raises
    has its exception absorbed. This subclass is a valid general-route
    problem -- it supplies its own ``f``, ``F`` and ``G``, keeps its
    coefficients under different names, and leaves ``_M`` as a descriptor
    that raises -- so the measured probe count is asserted rather than the
    claim, made earlier and untrue, that nothing is touched. (The dimension
    properties are overridden because the root's read ``_M.shape`` too, and
    this class owes them nothing either.)

    Absorbing is the safe direction here and is not a second silent skip. A
    name that cannot be read hands nothing to the tape, so there is nothing
    to alias and nothing to freeze; and if it *were* load-bearing, the reader
    reaching for it during the solve raises from the same descriptor, loudly,
    rather than returning a displaced gradient. What the check must never do
    is turn a buffer it could not read into one it decided was absent, which
    is why unreadable *storage* -- as distinct from an unreadable name --
    raises at C-15.7 instead.

    ``AffineDynamics.__init__`` never ran, so nothing it froze is at stake and
    the flag is absent. The gradient is asserted rather than only the absence
    of a raise, because "it did not raise" would also pass if the problem were
    quietly broken.
    """
    probes = 0

    class NeverInitialisedTheRoot(AffineDynamics):
        def __init__(self) -> None:
            self._m = np.array(M_CONST, dtype=float)
            self._c = np.array(C_CONST, dtype=float)

        @property
        def _M(self) -> NDArray:
            nonlocal probes
            probes += 1
            raise AttributeError("this class keeps no _M")

        @property
        def state_dim(self) -> int:
            return 1

        @property
        def control_dim(self) -> int:
            return 1

        @property
        def coefficients_constant(self) -> bool:
            return True

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(self._m @ y + self._c @ u)

        def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return self._m.copy()

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return self._c.copy()

    problem = NeverInitialisedTheRoot()
    require_immutable_coefficients(problem)
    assert probes == 1
    assert not affine_dynamics_verified(problem)

    exact, _ = _closed_form(1.0)
    gradient = GLMOptimizer(
        problem,
        OverwritesCoefficientMidSolve(None),
        explicit_euler(),
        T_SPAN,
        N_STEPS,
        Y0,
    ).gradient(U)
    assert np.asarray(gradient) == pytest.approx(exact, abs=EXACT_TOL)


def test_a_subclass_reusing_a_coefficient_name_is_refused_by_name() -> None:
    """The one working shape this clause refuses, and what it costs.

    This subclass never calls ``super().__init__``, replaces all three
    readers, and keeps its own writeable ``_M`` under that name, which it
    hands out by copy. Nothing it does is unsafe. It is refused anyway,
    because the alternative is worse: the only rule that distinguishes it
    from ``BuildsAndDelegates`` -- same absent marker, same overridden
    readers, storage the root's inherited ``F`` hands straight to the tape --
    is one that asks whether an override delegates, and an override cannot be
    asked what it reads.

    So the refusal is deliberate, it names the attribute that collided, and
    renaming it lifts the refusal. Restoration already charged this shape the
    same price by freezing ``_M`` on every copy; a rule that binds on copying
    but not on checking is not one rule. Pinned here so that narrowing it
    later is a decision rather than an accident.
    """

    class ReusesTheName(AffineDynamics):
        def __init__(self) -> None:
            self._M = np.array(M_CONST, dtype=float)
            self._c = np.array(C_CONST, dtype=float)

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(self._M @ y + self._c @ u)

        def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return self._M.copy()

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return self._c.copy()

    problem = ReusesTheName()
    assert problem._M.flags.writeable

    for raises in (
        lambda: require_immutable_coefficients(problem),
        lambda: affine_dynamics_verified(problem),
        lambda: GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ),
    ):
        with pytest.raises(MutableCoefficients, match="_M") as raised:
            raises()
        assert "_C" not in str(raised.value)

    # Renaming the collision is the whole remedy, and the problem then runs on
    # the general route to the exact gradient. Nothing about its dynamics
    # changed; only the attribute it keeps them in.
    class KeepsItsStorageUnderItsOwnName(ReusesTheName):
        def __init__(self) -> None:
            super().__init__()
            self._own_m = self._M
            del self._M

        @property
        def state_dim(self) -> int:
            return int(self._own_m.shape[0])

        @property
        def control_dim(self) -> int:
            return int(self._c.shape[1])

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return np.asarray(self._own_m @ y + self._c @ u)

        def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return self._own_m.copy()

    renamed = KeepsItsStorageUnderItsOwnName()
    require_immutable_coefficients(renamed)
    assert affine_dynamics_verified(renamed) is False

    exact, _ = _closed_form(1.0)
    assert np.asarray(
        GLMOptimizer(
            renamed,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ).gradient(U)
    ) == pytest.approx(exact, abs=EXACT_TOL)

    # Copying is a separate obligation and does not raise, so this shape still
    # witnesses the restoration rule. On the restored object it is
    # indistinguishable from `SerialisesOneOfThree`, which retains the root's
    # buffer, so the name is frozen in both. The *source* must survive
    # untouched, which is why a buffer frozen without the marker is copied
    # first: `copy.copy` shares it, and freezing in place reached back through
    # that sharing and froze the original.
    for duplicate in (copy.copy, copy.deepcopy):
        source = ReusesTheName()
        duplicated = duplicate(source)
        assert not duplicated._M.flags.writeable
        assert source._M.flags.writeable
        assert not np.shares_memory(source._M, duplicated._M)
        with pytest.raises(ValueError, match="read-only"):
            duplicated._M[0, 0] = M_OVERWRITTEN


def test_a_subclass_that_builds_its_own_buffers_and_delegates_is_refused(
) -> None:
    """Two ordinary habits, and between them a silent wrong answer.

    Building the coefficients from a file or a mesh instead of through
    ``super().__init__`` is one; wrapping a reader for instrumentation, unit
    conversion or logging is the other. Neither is adversarial and each was
    already accounted for alone -- the marker catches a delegating subclass
    that ran the root initialiser, and reader identity catches one that
    inherits the readers. Together they defeated both: no marker, no reader
    of the root's own, an empty live set, and ``super().F(...)`` handing the
    writeable ``_M`` to the tape.

    Measured on the C-15.7 fixture at ``C = 2`` before the cure: the default
    optimizer returned ``0.7552125`` where the closed form gives
    ``0.6781500``, the full ``0.0770625`` displacement, with every guard
    quiet. ``_closed_form`` derives both so neither is pinned as a literal.
    """

    class BuildsAndDelegates(AffineDynamics):
        def __init__(self) -> None:
            self._M = np.array(M_CONST, dtype=float)
            self._C = np.array(C_DOUBLED, dtype=float)
            self._b = np.zeros(1)
            self._nu = 1

        def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().f(y, u, t)

        def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().F(y, u, t)

        def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
            return super().G(y, u, t)

    problem = BuildsAndDelegates()
    assert problem.F(Y0, U[0], 0.0) is problem._M
    assert problem._M.flags.writeable

    with pytest.raises(MutableCoefficients, match="_M"):
        require_immutable_coefficients(problem)
    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)
    with pytest.raises(MutableCoefficients, match="_M"):
        GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(problem),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        )

    # And once frozen it computes the exact gradient, so the refusal was about
    # the buffer and not about the shape of the subclass.
    for name in ("_M", "_C", "_b"):
        getattr(problem, name).flags.writeable = False
    require_immutable_coefficients(problem)

    exact, _ = _closed_form(2.0)
    assert np.asarray(
        GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ).gradient(U)
    ) == pytest.approx(exact, abs=EXACT_TOL)


def test_a_delegating_subclass_holding_sparse_coefficients_is_refused() -> None:
    """The same shape again, with storage neither widening could type.

    ``BuildsAndDelegates`` above is caught because its three names resolve.
    When presence obliged only names holding an ``ndarray``, swapping those
    for ``csr_array`` -- an ordinary choice for a large linear system -- put
    the instance back outside both widenings: no marker, no reader of the
    root's, no array, an empty live set. Measured before the cure on the
    C-15.7 fixture at ``C = 2``, the default optimizer returned the displaced
    first component against the closed form's, the full displacement, quiet.

    So presence obliges a name that *resolves*, and what it resolves to is
    decided afterwards by the refusal, not by the widening. The distinction
    matters because the two directions are not symmetric: obliging a name
    that turns out fine costs a refusal the caller can read and lift,
    while declining to oblige one costs a wrong gradient nobody sees.
    """

    problem = SparseBuildsAndDelegates()
    # Live to neither widening before the cure: every reader is the
    # subclass's, and no coefficient name holds an ndarray.
    assert all(
        getattr(type(problem), name) is not getattr(AffineDynamics, name)
        for name in ("f", "F", "G")
    )
    assert not any(
        isinstance(getattr(problem, name), np.ndarray) for name in ("_M", "_C")
    )

    for refuse in (
        lambda: require_immutable_coefficients(problem),
        lambda: affine_dynamics_verified(problem),
        lambda: GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ),
    ):
        with pytest.raises(MissingCoefficients, match="_M") as raised:
            refuse()
        assert "csr_array" in str(raised.value)

    # The remedy the message names: dense, frozen, owning buffers. The same
    # subclass then computes the exact gradient, so the refusal was about the
    # storage and not about delegating.
    for name, value in (("_M", M_CONST), ("_C", C_DOUBLED)):
        frozen = np.array(value, dtype=float)
        frozen.flags.writeable = False
        setattr(problem, name, frozen)
    require_immutable_coefficients(problem)

    exact, _ = _closed_form(2.0)
    assert np.asarray(
        GLMOptimizer(
            problem,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ).gradient(U)
    ) == pytest.approx(exact, abs=EXACT_TOL)


def test_the_reverse_multi_root_order_is_refused_quietly() -> None:
    """Deriving from both roots is refused, and C-16.2 requires that refusal to
    be *quiet*.

    With ``TimeVaryingAffineDynamics`` first, the MRO resolves ``f``, ``F`` and
    ``G`` to the time-varying root, which builds its coefficients from the
    stored callables and never reads the constant root's buffers. Only
    ``AffineDynamics.__init__`` would have created those buffers, and it never
    ran. Gating on ``isinstance`` alone turned this into a raise.

    The forward order is the ``"two roots"`` entry in ``INELIGIBLE_SHAPES``
    above, where the same three members do resolve to the constant root and
    the check therefore does apply.
    """

    class TimeVaryingFirst(TimeVaryingAffineDynamics, AffineDynamics):
        pass

    problem = TimeVaryingFirst(
        lambda t: np.array([[0.4]]),
        lambda t: np.array([[1.0]]),
        state_dim=1,
        control_dim=1,
    )
    assert not affine_dynamics_verified(problem)


@pytest.mark.parametrize("supply_structure", (False, True))
@pytest.mark.parametrize("c", (1.0, 2.0))
def test_the_optimizer_refuses_a_copy_that_unfroze_its_coefficients(
    c: float, supply_structure: bool
) -> None:
    """The bypass end to end, on both paths into ``GLMOptimizer``.

    ``affine_dynamics_verified`` is reached only when the structure has to be
    deduced, so a caller supplying an explicit ``ProblemStructure`` bypassed
    the coefficient check entirely and received the displaced gradient in
    silence. Validity is a property of the problem rather than of the route,
    so it is enforced at construction on both paths.

    The exact first component is ``0.6781500`` at ``c = 2`` and the
    displacement ``h²Cy₂(M' − M) = 0.0770625``; both are asserted against the
    closed form so that the refusal is measured against a known answer rather
    than merely observed to happen. ``fired`` records that the refusal
    preceded the solve, so there was no mutation to be displaced by.
    """
    exact, displacement = _closed_form(c)
    assert displacement == pytest.approx(
        {1.0: 0.034390625, 2.0: 0.0770625}[c], abs=EXACT_TOL
    )

    # The fixture is anchored before the refusal is asserted: the same problem,
    # correctly constructed, returns the closed-form gradient. Without this the
    # test could be refusing a configuration that was never going to be right,
    # and the measured displacement would have nothing to be a displacement of.
    intact = AffineDynamics(M_CONST, np.array([[c]]))
    assert np.asarray(
        GLMOptimizer(
            intact,
            OverwritesCoefficientMidSolve(None),
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
        ).gradient(U)
    ) == pytest.approx(exact, abs=EXACT_TOL)

    problem = copy.deepcopy(
        _subclass_overriding("__setstate__")(M_CONST, np.array([[c]]))
    )
    objective = OverwritesCoefficientMidSolve(problem)
    structure = (
        ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
            state_affine=True,
            jointly_affine=True,
        )
        if supply_structure
        else None
    )

    # Construction alone, with no call to gradient. Chaining .gradient(U) onto
    # this would let the forward_solve guard satisfy the assertion and leave
    # this one untested: removing it was injected and no test noticed (C-15.5).
    # The claim here is specifically that the refusal precedes any work.
    with pytest.raises(MutableCoefficients, match="_M"):
        GLMOptimizer(
            problem,
            objective,
            explicit_euler(),
            T_SPAN,
            N_STEPS,
            Y0,
            problem_structure=structure,
        )

    assert not objective.fired


def test_the_exported_stepping_composition_refuses_the_same_problem() -> None:
    """``GLMOptimizer`` is not the only way in — C-15.7.

    ``forward_solve``, ``adjoint_solve`` and ``assemble_gradient`` are exported
    and compose into a gradient without touching ``GLMOptimizer``, so guarding
    only the optimizer left the identical ``0.0770625`` displacement reachable:
    this composition returned ``0.7552125`` against an exact ``0.6781500``.

    ``forward_solve`` is where the invariant actually bites — every step stores
    what ``F`` and ``G`` returned, and ``AffineDynamics`` returns its buffers by
    identity — so checking there covers every composition rather than every
    caller. The control below runs the same composition on an intact problem,
    without which this test would pass against a ``forward_solve`` that refused
    everything.
    """
    method = explicit_euler()
    structure = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
        state_affine=True,
        jointly_affine=True,
    )

    def compose(problem: AffineDynamics, objective: object) -> NDArray:
        solver = create_stage_solver(
            method,
            deduce_requirements(method, structure, problem.state_dim),
            structure,
            y_scale=1.0,
        )
        trajectory = forward_solve(
            Y0, U, T_SPAN, N_STEPS, problem, method, solver
        )
        adjoint = adjoint_solve(trajectory, objective, method, solver, H)
        return np.asarray(
            assemble_gradient(
                trajectory, adjoint, U, objective, method, problem, H
            )
        )

    exact, _ = _closed_form(2.0)
    intact = AffineDynamics(M_CONST, np.array([[2.0]]))
    assert compose(
        intact, OverwritesCoefficientMidSolve(None)
    ) == pytest.approx(exact, abs=EXACT_TOL)

    unfrozen = copy.deepcopy(
        _subclass_overriding("__setstate__")(M_CONST, np.array([[2.0]]))
    )
    objective = OverwritesCoefficientMidSolve(unfrozen)
    with pytest.raises(MutableCoefficients, match="_M"):
        compose(unfrozen, objective)
    assert not objective.fired
