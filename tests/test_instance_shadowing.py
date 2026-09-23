"""Verification is checked at the instance, not only at the class — C-16.9.

``affine_dynamics_verified`` established affineness by comparing the members
of ``type(problem)`` against the root class. That reads a fact about the class
and applies it to an instance, and the two can disagree: ``problem.f = ...``
writes to the instance ``__dict__`` and leaves every class-level comparison
untouched.

What that costs, and what it does not
-------------------------------------

The obvious reproduction is not the defect. Writing a nonlinear ``f`` onto the
instance while leaving ``F`` the affine Jacobian gives a gradient wrong by
``6.395`` — but an ordinary callback problem supplying the same mismatched pair
is wrong by *exactly* the same amount, on the general route, with no affine
verification anywhere near it. That is the universal obligation that ``F`` be
the Jacobian of ``f``, and it is not what this module is about.
``test_an_inconsistent_pair_is_not_a_verification_defect`` pins that, so the
distinction cannot be lost again.

The real cost needs a *consistent* shadow: an instance carrying a complete,
self-consistent nonlinear system. Then the gradient is right — it reads only
``f``, ``F`` and ``G``, which agree — while the route chosen from the stale
class-level fact still asserts ``jointly_affine``, and C-16.6 skips a curvature
term that is now genuinely nonzero. The Hessian is silently wrong.

Why an explicit method
----------------------

On an implicit method the damage is already refused: the stage equation is
nonlinear, so the single linear solve leaves a residual and C-16.4 raises
``NonAffineStageEquation``. An explicit method solves no stage equation and so
has no residual to check, which is why the certification below is built on
``explicit_euler``.
"""

from __future__ import annotations

import numpy as np
import pytest

from adjungo import GLMOptimizer, ProblemStructure
from adjungo.core.affine import (
    AffineDynamics,
    TimeVaryingAffineDynamics,
    affine_dynamics_verified,
)
from adjungo.core.problem import Linearity
from adjungo.methods.runge_kutta import explicit_euler, implicit_midpoint
from adjungo.solvers.linear_stage import NonAffineStageEquation
from tests.problems import AnchorObjective

M_CONST = np.array([[0.4]])
C_CONST = np.array([[1.0]])

Y0 = np.array([0.8])
T_SPAN = (0.0, 0.5)
N_STEPS = 2
H = (T_SPAN[1] - T_SPAN[0]) / N_STEPS

U = np.array([[[0.3]], [[0.2]]])
V = np.array([[[1.0]], [[0.0]]])

#: Round-off only. Both sides of every comparison against a closed form below
#: execute a fixed, short sequence of arithmetic with no iterative solve, so
#: C-3 admits accumulated round-off and nothing else: a few tens of operations
#: on O(1)-to-O(10) quantities, bounded well below 1e-13.
EXACT_TOL = 1e-13

#: Central difference of this package's own objective, used only to corroborate
#: the closed forms. Basis: C-3.2, error O(eps^2) + O(macheps/eps); at
#: eps = 1e-6 on quantities of size O(1) to O(10) this is ~1e-9, so 1e-6 is a
#: deliberately loose corroboration budget and never the primary oracle.
FD_EPS = 1e-6
FD_TOL = 1e-6


def _constant() -> AffineDynamics:
    return AffineDynamics(M_CONST, C_CONST)


def _time_varying() -> TimeVaryingAffineDynamics:
    return TimeVaryingAffineDynamics(
        lambda t: np.array([[0.4 + 0.1 * t]]),
        lambda t: np.array([[1.0]]),
        state_dim=1,
        control_dim=1,
    )


def _consistent_nonlinear_shadow() -> AffineDynamics:
    """An ``AffineDynamics`` instance carrying a complete nonlinear system.

    ``f = 0.4y + u + 5y^2`` with ``F = 0.4 + 10y``, ``G = 1`` and
    ``F_yy_action = 10v``. Every member agrees with every other, so this is not
    the mismatched-pair error; the only thing wrong with it is that the *class*
    still says affine.
    """
    p = AffineDynamics(M_CONST, C_CONST)
    object.__setattr__(
        p, "f", lambda y, u, t: np.array([0.4 * y[0] + u[0] + 5.0 * y[0] ** 2])
    )
    object.__setattr__(p, "F", lambda y, u, t: np.array([[0.4 + 10.0 * y[0]]]))
    object.__setattr__(p, "G", lambda y, u, t: np.array([[1.0]]))
    object.__setattr__(
        p, "F_yy_action", lambda y, u, t, v: np.array([[10.0 * v[0]]])
    )
    object.__setattr__(p, "F_yu_action", lambda y, u, t, v: np.zeros((1, 1)))
    object.__setattr__(p, "F_uu_action", lambda y, u, t, v: np.zeros((1, 1)))
    return p


def _optimizer(problem, method=None, structure=None) -> GLMOptimizer:
    return GLMOptimizer(
        problem,
        AnchorObjective(),
        method if method is not None else explicit_euler(),
        t_span=T_SPAN,
        N=N_STEPS,
        y0=Y0,
        problem_structure=structure,
    )


# ---------------------------------------------------------------------------
# The instance check itself
# ---------------------------------------------------------------------------


def test_an_unmodified_instance_is_still_verified() -> None:
    """The check must not have become a blanket refusal. Without this, every
    assertion below would pass against a function that returned ``False``."""
    assert affine_dynamics_verified(_constant())
    assert affine_dynamics_verified(_time_varying())


@pytest.mark.parametrize("name", ["f", "F", "G"])
def test_a_shadowed_computational_member_is_refused(name: str) -> None:
    problem = _constant()
    object.__setattr__(problem, name, lambda *args: np.zeros((1, 1)))
    assert not affine_dynamics_verified(problem)


@pytest.mark.parametrize("name", ["f", "F", "G", "M", "C", "b", "_coefficient"])
def test_a_shadowed_time_varying_member_is_refused(name: str) -> None:
    """The coefficient accessors are invoked members too. They are where the
    shape check and the defensive copy live (C-17.3), so shadowing one removes
    both while leaving the class untouched."""
    problem = _time_varying()
    object.__setattr__(problem, name, lambda *args: np.zeros((1, 1)))
    assert not affine_dynamics_verified(problem)


def test_plain_attribute_assignment_is_refused_too() -> None:
    """``object.__setattr__`` above is the tripwire's idiom. The ordinary
    idiom a caller would actually reach for must be refused identically."""
    problem = _constant()
    problem.f = lambda y, u, t: np.zeros(1)  # type: ignore[method-assign]
    assert not affine_dynamics_verified(problem)


def test_the_curvature_callbacks_remain_shadowable() -> None:
    """C-16.5 depends on this and would be silently disarmed without it.

    The tripwire replaces these three on the instance with raisers and asserts
    the affine gradient still computes. That is evidence no path reaches a
    curvature term, and it is only obtainable while shadowing them leaves the
    problem verified. Asserted here in its own right so that tightening the
    instance check cannot disarm C-16.5 while C-16.5's own test still passes.
    """
    problem = _constant()
    for name in ("F_yy_action", "F_yu_action", "F_uu_action"):
        object.__setattr__(problem, name, lambda *args: np.zeros((1, 1)))
    assert affine_dynamics_verified(problem)


def test_a_class_hiding_its_instance_dictionary_is_refused() -> None:
    """The instance check reads ``problem.__dict__``, which is itself an
    attribute access and so answerable by the class being checked.

    A subclass defining ``__dict__`` as a property returning ``{}`` shows the
    check an empty mapping while the real storage holds every shadow, and
    reinstates the exact silent Hessian error C-16.9 exists to remove. Found by
    review after the first version of this check shipped; the cure is
    :func:`~adjungo.core.affine._defined_as`, which reads bindings out of the
    MRO without invoking the descriptor protocol.
    """

    class HidesInstanceDictionary(AffineDynamics):
        @property
        def __dict__(self):  # type: ignore[override]
            return {}

    problem = HidesInstanceDictionary(M_CONST, C_CONST)
    object.__setattr__(problem, "f", lambda y, u, t: np.zeros(1))

    # The shadow really is hidden from ordinary access: without _defined_as
    # there would be nothing for the instance check to find.
    assert problem.__dict__ == {}
    assert "f" in AffineDynamics.__dict__["__dict__"].__get__(problem)

    assert not affine_dynamics_verified(problem)


def test_a_class_redefining_attribute_lookup_is_refused() -> None:
    """The comparisons reach the problem through ordinary attribute access, so
    a class that redefines attribute access can answer the check with one
    object and the solver with another.

    These are conservative refusals rather than free ones: a subclass using
    ``__getattr__`` only for unrelated metadata is refused too and pays the
    general route. That is the correct direction under C-16.2, and it is
    asserted rather than assumed by the transparent implementations below,
    which change no behaviour at all and are still refused.
    """

    class InterceptsGetattribute(AffineDynamics):
        def __getattribute__(self, name: str):
            return object.__getattribute__(self, name)

    class InterceptsGetattr(AffineDynamics):
        def __getattr__(self, name: str):
            raise AttributeError(name)

    assert not affine_dynamics_verified(InterceptsGetattribute(M_CONST, C_CONST))
    assert not affine_dynamics_verified(InterceptsGetattr(M_CONST, C_CONST))


def test_a_guaranteed_member_blanked_on_a_subclass_is_refused() -> None:
    """A subclass need not replace a member with a working one to remove the
    guarantee; blanking it is enough, and the class comparison must not treat
    a missing attribute as agreement."""

    class Blanked(AffineDynamics):
        F = None

    assert not affine_dynamics_verified(Blanked(M_CONST, C_CONST))


def test_ordinary_constructions_are_not_refused() -> None:
    """A conservative check is only acceptable if it is conservative about the
    right things. Each of these populates or replaces the instance ``__dict__``
    without touching a guaranteed member, and must stay verified -- otherwise
    the check would quietly cost every caller the optimized route.
    """
    import copy
    import pickle

    class AddsUnrelatedMembers(AffineDynamics):
        def describe(self) -> str:
            return "unrelated"

    annotated = AddsUnrelatedMembers(M_CONST, C_CONST)
    annotated.label = "trajectory A"  # type: ignore[attr-defined]
    annotated.cached_norm = 1.0  # type: ignore[attr-defined]

    cases = {
        "subclass adding members": annotated,
        "shallow copy": copy.copy(_constant()),
        "deep copy": copy.deepcopy(_constant()),
        "pickle round trip": pickle.loads(pickle.dumps(_constant())),
        "time-varying deep copy": copy.deepcopy(_time_varying()),
    }
    for label, problem in cases.items():
        assert affine_dynamics_verified(problem), label


def test_a_guaranteed_name_is_refused_even_when_the_value_is_equivalent() -> None:
    """The check is identity, not behaviour, so rebinding ``f`` to something
    that computes the same thing is still refused. Asserted so the cost of the
    rule is explicit: this is a deliberate false refusal, and C-16.2 requires
    the error direction to run this way."""
    problem = _constant()
    object.__setattr__(problem, "f", AffineDynamics.f.__get__(problem))
    assert not affine_dynamics_verified(problem)


def test_a_metaclass_answering_for_the_class_is_refused() -> None:
    """Why the class comparison reads the MRO instead of calling ``getattr``.

    Attribute access on a *class* goes through its metaclass, so
    ``getattr(cls, "f")`` is answerable by a metaclass exactly as
    ``problem.__dict__`` was answerable by the class. The metaclass below hands
    the check the root's ``f`` while the class really binds a nonlinear one,
    and defeats an implementation that compares via ``getattr``.

    This is the only test that separates
    :func:`~adjungo.core.affine._defined_as` from ``getattr`` for the
    *guaranteed members*; without it, replacing one with the other is an
    injected defect the whole suite misses.
    """

    class AnswersForTheClass(type):
        def __getattribute__(cls, name: str):
            if name in ("f", "F", "G"):
                return AffineDynamics.__dict__[name]
            return type.__getattribute__(cls, name)

    class Evil(AffineDynamics, metaclass=AnswersForTheClass):
        def f(self, y, u, t):
            return np.array([0.4 * y[0] + u[0] + 5.0 * y[0] ** 2])

    problem = Evil(M_CONST, C_CONST)

    # The interception works: ordinary class access reports the root's member,
    # so a getattr-based comparison sees no difference at all.
    assert Evil.f is AffineDynamics.__dict__["f"]
    assert Evil.__dict__["f"] is not AffineDynamics.__dict__["f"]

    assert not affine_dynamics_verified(problem)


def test_a_slot_shadowing_a_guaranteed_member_is_refused() -> None:
    """A slot named after a guaranteed member is a class-level data descriptor,
    so the class check catches it before the instance check is reached.

    (The subclass still has an instance ``__dict__``, inherited from
    ``AffineDynamics``, which declares no ``__slots__``. ``__slots__`` on a
    subclass alone does not remove it. The point here is the descriptor, not
    the absence of storage.)
    """

    class Slotted(AffineDynamics):
        __slots__ = ("f",)

    problem = Slotted(M_CONST, C_CONST)
    assert not affine_dynamics_verified(problem)


# ---------------------------------------------------------------------------
# What the check is worth: the Hessian oracle
# ---------------------------------------------------------------------------


def _closed_form() -> dict[str, np.ndarray | float]:
    """Two explicit Euler steps on ``y' = 0.4y + u + 5y^2``, differentiated by
    hand. C-14.2: the oracle level no implementation participates in.

    With ``h = 0.25`` and ``J = y_2^2/2``, writing ``p = dy_2/dy_1``,

        y_1 = y_0 + h f(y_0, u_0)
        y_2 = y_1 + h f(y_1, u_1)
        p   = 1 + h (0.4 + 10 y_1)          d^2 y_2/dy_1^2 = 10 h
        dJ/du_0 = y_2 p h                   dJ/du_1 = y_2 h

    Differentiating ``dJ/du_0`` again along ``v = (1, 0)``, and writing
    ``q = dy_2/du_0 = p h``,

        (Hv)_0 = q p h + y_2 (10 h) h^2     (Hv)_1 = q h

    The second term of ``(Hv)_0`` is exactly the dynamics-curvature
    contribution: it is what C-16.6's skip drops, and it is what makes the
    wrong answer wrong. Returned separately so the defect can be predicted
    rather than merely observed.
    """
    y1 = Y0[0] + H * (0.4 * Y0[0] + U[0, 0, 0] + 5.0 * Y0[0] ** 2)
    y2 = y1 + H * (0.4 * y1 + U[1, 0, 0] + 5.0 * y1**2)
    p = 1.0 + H * (0.4 + 10.0 * y1)
    q = p * H
    curvature = y2 * (10.0 * H) * H**2
    return {
        "y1": y1,
        "y2": y2,
        "gradient": np.array([[[y2 * p * H]], [[y2 * H]]]),
        "hv": np.array([[[q * p * H + curvature]], [[q * H]]]),
        "hv_without_curvature": np.array([[[q * p * H]], [[q * H]]]),
        "curvature": curvature,
    }


def test_the_closed_form_matches_the_trajectory_it_claims() -> None:
    """Establishes the oracle before anything is compared against it. Without
    this, an algebra slip in ``_closed_form`` would be indistinguishable from
    a defect in the package."""
    exact = _closed_form()
    assert abs(float(exact["y1"]) - 1.755) < EXACT_TOL
    assert abs(float(exact["y2"]) - 5.83053125) < EXACT_TOL
    # The dropped term is the whole of the discrepancy asserted below.
    assert abs(float(exact["curvature"]) - 0.9110205078125) < EXACT_TOL
    assert np.allclose(
        np.asarray(exact["hv"]) - np.asarray(exact["hv_without_curvature"]),
        np.array([[[exact["curvature"]]], [[0.0]]]),
        rtol=0.0,
        atol=EXACT_TOL,
    )


def test_the_gradient_of_the_shadowed_problem_is_itself_exact() -> None:
    """The gradient reads only ``f``, ``F`` and ``G``, which this shadow keeps
    mutually consistent, so it is exact for the nonlinear system the instance
    actually describes. That is what makes the Hessian comparison below
    meaningful: the two derivatives disagree about the problem, and only one
    of them is wrong."""
    reported = _optimizer(_consistent_nonlinear_shadow()).gradient(U)
    assert np.max(np.abs(reported - _closed_form()["gradient"])) < EXACT_TOL


def test_the_hessian_of_a_consistently_shadowed_problem_is_exact() -> None:
    """The defect C-16.9 closes, stated as the property that must hold."""
    hv = _optimizer(_consistent_nonlinear_shadow()).hessian_vector_product(U, V)
    assert np.max(np.abs(hv - _closed_form()["hv"])) < EXACT_TOL


def test_a_central_difference_corroborates_the_closed_form() -> None:
    """C-14.1 places a fixed-mesh eps check below a closed form, so it is used
    here only to catch an algebra slip shared between the two closed-form
    expressions -- which a comparison of one against the other could not."""
    opt = _optimizer(_consistent_nonlinear_shadow())
    plus = _optimizer(_consistent_nonlinear_shadow()).gradient(U + FD_EPS * V)
    minus = _optimizer(_consistent_nonlinear_shadow()).gradient(U - FD_EPS * V)
    measured = (plus - minus) / (2 * FD_EPS)
    assert np.max(np.abs(measured - _closed_form()["hv"])) < FD_TOL

    def objective(scale: float) -> float:
        return opt.objective_value(U + scale * V)

    directional = (objective(FD_EPS) - objective(-FD_EPS)) / (2 * FD_EPS)
    exact = float(np.sum(np.asarray(_closed_form()["gradient"]) * V))
    assert abs(directional - exact) < FD_TOL


def test_opening_the_skip_on_that_problem_would_be_wrong(monkeypatch) -> None:
    """C-14.2: the test above must be shown to fail when the skip is opened.

    Otherwise it is consistent with the curvature term being zero here, and
    would pass against any implementation at all. The skip is forced open by
    declaring ``jointly_affine`` *and* making verification unconditional,
    because :mod:`adjungo.optimization.hessian` requires both -- which is
    itself the M6 rule that a declaration alone never opens it.

    The wrong answer is *predicted*, not merely observed: it must equal the
    closed form with the dynamics-curvature term removed, which is
    ``0.911`` below the exact value in the first component and identical in
    the second. An assertion that the two merely differ would also pass if the
    skip broke something else entirely.
    """
    import adjungo.optimization.hessian as hessian_module
    import adjungo.stepping.sensitivity as sensitivity_module

    for module in (hessian_module, sensitivity_module):
        monkeypatch.setattr(
            module, "affine_dynamics_verified", lambda problem: True
        )

    declared = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
        state_affine=True,
        jointly_affine=True,
    )
    hv = _optimizer(
        _consistent_nonlinear_shadow(), structure=declared
    ).hessian_vector_product(U, V)

    exact = _closed_form()
    assert np.max(np.abs(hv - exact["hv_without_curvature"])) < EXACT_TOL
    assert np.max(np.abs(hv - exact["hv"])) > 0.9


def test_an_implicit_method_already_refused_the_shadow_loudly() -> None:
    """The instance check is needed for the explicit route specifically.

    C-16.4 already catches this on an implicit method: the stage equation is
    nonlinear, so one linear solve cannot satisfy it and the residual test
    raises. Asserted so the scope of C-16.9 is evidence rather than assertion
    -- and so that if the affine route ever stops checking its residual, this
    fails too.
    """
    import adjungo.optimization.interface as interface_module

    problem = _consistent_nonlinear_shadow()
    original = interface_module.affine_dynamics_verified
    interface_module.affine_dynamics_verified = lambda p: True
    try:
        opt = _optimizer(problem, method=implicit_midpoint())
        with pytest.raises(NonAffineStageEquation):
            opt.objective_value(U)
    finally:
        interface_module.affine_dynamics_verified = original


# ---------------------------------------------------------------------------
# The boundary: what is *not* a verification defect
# ---------------------------------------------------------------------------


def test_an_inconsistent_pair_is_not_a_verification_defect() -> None:
    """``F`` must be the Jacobian of ``f``. That obligation is universal, and
    breaking it is equally wrong with no affine class in sight.

    Pinned because the natural reproduction of a shadowing defect breaks this
    obligation as well, and would then attribute to verification an error that
    verification never had anything to do with. Both sides here are the *same*
    wrong number, which is the whole point.
    """

    class InconsistentCallbacks:
        state_dim = 1
        control_dim = 1

        def f(self, y, u, t):
            return np.array([0.4 * y[0] + u[0] + 5.0 * y[0] ** 2])

        def F(self, y, u, t):
            return np.array([[0.4]])  # not the Jacobian of f, deliberately

        def G(self, y, u, t):
            return np.array([[1.0]])

        def F_yy_action(self, y, u, t, v):
            return np.zeros((1, 1))

        def F_yu_action(self, y, u, t, v):
            return np.zeros((1, 1))

        def F_uu_action(self, y, u, t, v):
            return np.zeros((1, 1))

    shadowed = _constant()
    object.__setattr__(
        shadowed,
        "f",
        lambda y, u, t: np.array([0.4 * y[0] + u[0] + 5.0 * y[0] ** 2]),
    )

    def first_gradient(problem) -> float:
        return float(_optimizer(problem).gradient(U)[0, 0, 0])

    def first_fd(problem) -> float:
        opt = _optimizer(problem)

        def objective(scale: float) -> float:
            return opt.objective_value(U + scale * V)

        return (objective(FD_EPS) - objective(-FD_EPS)) / (2 * FD_EPS)

    callback_error = abs(
        first_gradient(InconsistentCallbacks())
        - first_fd(InconsistentCallbacks())
    )
    shadow_error = abs(first_gradient(shadowed) - first_fd(shadowed))

    assert callback_error > 6.0
    # The same number, reached with and without an affine class involved.
    # Basis: both routes execute identical arithmetic on identical inputs, so
    # C-11.3 permits an exact comparison only up to reduction order; 1e-12 is
    # a relative budget on an O(6) quantity.
    assert abs(callback_error - shadow_error) < 1e-12
