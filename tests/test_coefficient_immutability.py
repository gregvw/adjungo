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
displaces that by ``h²C²y₂(M' − M) = 0.034390625`` exactly, while ``∂J/∂u₁``,
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
    MutableCoefficients,
    TimeVaryingAffineDynamics,
    affine_dynamics_verified,
)
from adjungo.core.problem import Linearity
from adjungo.methods.runge_kutta import explicit_euler
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


def test_a_deep_copy_of_a_refused_subclass_stays_refused() -> None:
    """Copying must not launder a problem into verification."""
    subclass = _subclass_overriding("__setstate__")
    assert not affine_dynamics_verified(
        copy.deepcopy(subclass(M_CONST, C_CONST))
    )
