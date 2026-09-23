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

import numpy as np
import pytest
from numpy.typing import NDArray

from adjungo import GLMOptimizer, ProblemStructure
from adjungo.core.affine import (
    _COEFFICIENT_BUFFERS,
    _COPY_PROTOCOL_HOOKS,
    AffineDynamics,
    InvalidCoefficients,
    MissingCoefficients,
    MutableCoefficients,
    TimeVaryingAffineDynamics,
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
    """The read cannot be answered by the machinery it runs ahead of.

    ``__dict__`` is a lookup hook, and the coefficient check now runs before
    the lookup hooks are compared. Reading ``problem.__dict__`` would therefore
    consult a subclass property whose honesty is exactly what has not been
    established yet — and that subclass is quietly refused a few lines later,
    so it is precisely a case that must not be allowed to answer for itself.

    The buffer is unfrozen before the hiding class is installed, so the test
    builds its state through ordinary storage and shares no mechanism with the
    code under test.
    """

    class HidesItsDict(AffineDynamics):
        @property
        def __dict__(self):  # type: ignore[override]
            return {}

    problem = _fresh()
    problem.__dict__["_M"] = problem.M.copy()
    problem.__class__ = HidesItsDict

    assert problem.__dict__ == {}
    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)


def test_a_subclass_that_never_ran_the_root_initialiser_is_left_to_fail_loudly(
) -> None:
    """No buffers means nothing to alias, so this is *not* the guard's business.

    ``__init__`` carries no guarantee and is not compared, so a subclass can
    inherit the buffer-reading methods while holding no buffers at all. An
    earlier version raised ``MissingCoefficients`` here, inferring from the MRO
    that the inherited ``f``/``F``/``G`` would read them. That inference is
    unsound in the other direction -- an override may delegate to ``super()``
    and read them anyway -- so it was dropped, and with it the ability to
    recognise this shape at the route decision.

    Nothing is lost that C-7 cares about. An absent buffer cannot be aliased
    and so cannot displace a gradient; the first read raises ``AttributeError``
    naming the attribute. A wrong answer was never available here, which is why
    the quiet path is the correct one and the diagnosis may be deferred.
    """

    class SkipsTheInitialiser(AffineDynamics):
        def __init__(self) -> None:
            pass

    problem = SkipsTheInitialiser()
    require_immutable_coefficients(problem)
    assert affine_dynamics_verified(problem)

    with pytest.raises(AttributeError, match="_M"):
        forward_solve(Y0, U, T_SPAN, N_STEPS, problem, explicit_euler(), None)


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


def test_a_slotted_coefficient_is_read_through_its_descriptor() -> None:
    """``__slots__`` storage is present and frozen, and must not be refused.

    A slot of the same name shadows the instance dictionary, so reading the
    root's ``__dict__`` — through ``problem.__dict__`` or through the root's
    own getset descriptor — finds no ``_M`` and reported a perfectly valid,
    frozen buffer as missing. ``AffineDynamics.F`` is ``return self._M``, so
    plain attribute access is what it resolves, and that is what the check
    reads: the same object, obtained the same way.
    """

    class SlottedM(AffineDynamics):
        __slots__ = ("_M",)

    problem = SlottedM(M_CONST, C_CONST)
    assert "_M" not in object.__getattribute__(problem, "__dict__")
    assert problem.F(Y0, U[:1], 0.0) is problem._M

    require_immutable_coefficients(problem)
    assert affine_dynamics_verified(problem)

    problem._M.flags.writeable = True
    with pytest.raises(MutableCoefficients, match="_M"):
        affine_dynamics_verified(problem)


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


def test_a_subclass_that_never_ran_the_root_initialiser_is_not_touched() -> None:
    """The precondition is recorded, so no coefficient descriptor is evaluated.

    Asking whether ``_M`` is *present* means evaluating whatever ``_M``
    resolves to on a class that may never have had one. This subclass is a
    valid general-route problem: it supplies its own ``f``, ``F`` and ``G``,
    keeps its coefficients under different names, and leaves ``_M`` as a
    descriptor that must never run. Probing it would raise here, out of a
    check that is supposed to be silent about this shape. (The dimension
    properties are overridden for the same reason: the root's read ``_M.shape``
    too, and this class owes them nothing either.)

    ``AffineDynamics.__init__`` never ran, so nothing it froze is at stake and
    the flag is absent. The gradient is asserted rather than only the absence
    of a raise, because "it did not raise" would also pass if the problem were
    quietly broken.
    """

    class NeverInitialisedTheRoot(AffineDynamics):
        def __init__(self) -> None:
            self._m = np.array(M_CONST, dtype=float)
            self._c = np.array(C_CONST, dtype=float)

        @property
        def _M(self) -> NDArray:
            raise AssertionError("a dead coefficient descriptor was evaluated")

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


def test_a_subclass_may_reuse_a_coefficient_name_for_its_own_storage() -> None:
    """A partial set is damage only if the root established the whole set.

    This subclass never calls ``super().__init__`` and keeps its own writeable
    ``_M`` under that name, which it hands out by copy. Reading presence
    instead of the recorded flag saw one buffer of three and diagnosed an
    instance taken apart after construction, refusing a problem that is not
    the root's to refuse.
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
    require_immutable_coefficients(problem)

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
