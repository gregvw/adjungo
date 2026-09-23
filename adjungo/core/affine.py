"""Affine dynamics ``y' = M y + C u + b``, as a representation rather than a claim.

Why this is a class and not a flag
----------------------------------

Milestone M6 accepted a *declaration* that the Jacobian is constant because it
could **verify** that declaration: comparing two assembled stage matrices
element for element costs ``O(n^2)`` and guards an ``O(n^3)`` factorization, so
the check is asymptotically cheaper than the work it protects. See
``NUMERICS.md`` C-15. The governing clause for this module is C-16, and
C-16.2 is the ruling this file implements.

No comparably cheap check exists for the statement "the dynamics have zero
curvature". Evaluating ``f`` at some sample points and observing that it looks
affine is a **probe**: it asks what ``f`` does at points and then generalizes to
points the computation never visits. Generalizing a probe is exactly the defect
recorded as precedent R-9, where a Jacobian probe that varied ``u`` and ``t``
but never ``y`` produced a 4.24e-05 relative gradient error with the whole test
suite passing.

So the structural facts here are not declared and then checked. They are
**constructive**. This class owns ``M``, ``C`` and ``b`` and computes ``f``,
``F``, ``G`` and all three contracted-Hessian actions from them. A caller cannot
instantiate it with non-affine dynamics, because there is nothing to instantiate
it with except the three coefficient arrays. Affineness is a property of this
class, provable by reading it, rather than an assertion by the caller that
nothing can check.

What this class licenses
------------------------

1. ``f`` is affine in ``y``, so every implicit stage equation is affine in its
   unknown and is solved exactly by one linear solve. Newton is not entered.
2. ``F = M`` is constant, so the M6 factorization store reuses a single
   factorization for an entire solve.
3. ``f_yy``, ``f_yu`` and ``f_uu`` all vanish identically, so the
   dynamics-curvature contributions to the second-order adjoint and to the
   Hessian-vector product are structurally zero.

Point 3 is the one that requires care. It licenses skipping the *dynamics*
curvature only. **Objective curvature is unaffected** and is still assembled in
full: an affine plant under a quadratic cost has a perfectly nonzero Hessian.

The trap this class exists to avoid
-----------------------------------

``Linearity.LINEAR`` is documented as "F independent of y, u". That gives
``f(y, u, t) = M(t) y + b(u, t)``: affine in the **state**, with ``b`` free to
be any function of the control whatsoever. It does **not** give ``f_uu = 0``.

The counterexample is already in the test suite.
``tests/problems.py::ConstantJacobianQuadraticControl`` is
``f = A y + B (u * u)``. Its ``F`` is the constant ``A``, so it satisfies
``LINEAR`` by that member's own definition, and its ``F_uu_action`` returns
``2 diag(B^T v)``, which is not zero. A dispatch rule of the form "``LINEAR``
implies skip curvature" would return a Gauss-Newton approximation from a
function whose docstring promises an exact Hessian.

That is why this module distinguishes two facts that the single enum conflates:

``state_affine``
    ``f(y, u, t) = M(u, t) y + b(u, t)``. Stage equations are linear in their
    unknowns for fixed controls. Says nothing about curvature in ``u``.

``jointly_affine``
    ``f(y, u, t) = M(t) y + C(t) u + b(t)``. All three dynamics-curvature blocks
    vanish. This is equivalent to zero curvature, not merely sufficient for it:
    a function whose second derivatives vanish identically is affine.

``ConstantJacobianQuadraticControl`` is ``state_affine`` and not
``jointly_affine``, which is precisely the distinction that keeps it from
losing its ``F_uu`` term.

Constant and time-varying coefficients are different facts
----------------------------------------------------------

C-16.2 states ``jointly_affine`` as ``f = M(t) y + C(t) u + b(t)``, with the
coefficients free to vary in time. :class:`AffineDynamics` implements the
constant-coefficient case; :class:`TimeVaryingAffineDynamics` implements the
general one. Both have identically zero curvature, so both open the C-16.6
skip.

They differ on one axis only, and it is an axis C-16.1 requires be kept
separate from the others: ``jacobian_constant``. For ``M(t)`` the stage
Jacobian ``F = M(t_i)`` differs between stages of a single step whenever the
abscissae differ, so the C-15 factorization store must not reuse across them.
That fact is carried by ``coefficients_constant``, which is part of each
class's guarantee rather than a caller declaration, and which
:meth:`~adjungo.optimization.interface.GLMOptimizer.deduce_structure` reads to
set ``jacobian_constant``.

The error direction if this were wrong is toward refusal, not toward a wrong
number: C-15.2 compares the stored matrix with the requested one element for
element, so a time-varying problem that reached the reuse path would raise
``DeclaredStructureViolation`` at the second distinct stage matrix rather than
solve with the wrong operator. The separation here keeps that guard from ever
being needed, but does not replace it.
"""

from collections.abc import Callable
from typing import Any, TypeGuard

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AffineDynamics",
    "TimeVaryingAffineDynamics",
    "affine_dynamics_verified",
]


class AffineDynamics:
    """``y' = M y + C u + b`` with constant coefficients.

    Args:
        M: State matrix, shape ``(n, n)``.
        C: Control matrix, shape ``(n, nu)``.
        b: Constant forcing, shape ``(n,)``. Defaults to zero.

    The coefficient arrays are **copied at construction** and exposed through
    read-only properties backed by non-writeable buffers. Neither rebinding
    ``problem.M`` nor writing into ``problem.M[0, 0]`` is possible. This matters
    because the route selected for this problem asserts that ``F`` is the same
    matrix at every stage of every step; a coefficient that could change between
    steps would make the M6 factorization store reuse a factorization of a
    matrix that no longer exists.
    """

    def __init__(
        self, M: NDArray, C: NDArray, b: NDArray | None = None
    ) -> None:
        M_arr = np.array(M, dtype=float, copy=True)
        C_arr = np.array(C, dtype=float, copy=True)

        if M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError(
                f"M must be square (n, n); got shape {M_arr.shape}"
            )
        n = M_arr.shape[0]
        if C_arr.ndim != 2 or C_arr.shape[0] != n:
            raise ValueError(
                f"C must have shape (n, nu) with n = {n}; "
                f"got shape {C_arr.shape}"
            )

        b_arr = (
            np.zeros(n)
            if b is None
            else np.array(b, dtype=float, copy=True).reshape(-1)
        )
        if b_arr.shape != (n,):
            raise ValueError(
                f"b must have shape ({n},); got shape {b_arr.shape}"
            )

        for arr in (M_arr, C_arr, b_arr):
            arr.flags.writeable = False

        self._M = M_arr
        self._C = C_arr
        self._b = b_arr
        self._nu = int(C_arr.shape[1])

    @property
    def M(self) -> NDArray:
        """State matrix, read-only."""
        return self._M

    @property
    def C(self) -> NDArray:
        """Control matrix, read-only."""
        return self._C

    @property
    def b(self) -> NDArray:
        """Constant forcing, read-only."""
        return self._b

    @property
    def coefficients_constant(self) -> bool:
        """``True``: ``M`` is the same matrix at every stage of every step.

        This is what licenses C-15 factorization reuse, and it is a property of
        this class rather than a caller declaration. The coefficient arrays are
        copied at construction and are not writeable, so there is no way for
        ``F`` to return two different matrices over the life of the object.
        """
        return True

    @property
    def state_dim(self) -> int:
        return int(self._M.shape[0])

    @property
    def control_dim(self) -> int:
        return int(self._nu)

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self._M @ y + self._C @ u + self._b)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._M

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._C

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(n, n)``.

        Defined rather than omitted so that the route which skips this term can
        be compared against the route which calls it. The two must agree
        exactly, because skipping is the removal of an addition of zero, not an
        approximation of one.
        """
        n = self.state_dim
        return np.zeros((n, n))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(n, nu)``."""
        return np.zeros((self.state_dim, self._nu))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(nu, nu)``."""
        return np.zeros((self._nu, self._nu))


#: The methods whose bodies carry the affine guarantee. If a subclass replaces
#: any of them, the guarantee is gone and the structural route must not be
#: taken. ``state_dim`` and ``control_dim`` are excluded deliberately: they are
#: dimensions, and a wrong one raises a shape error immediately rather than
#: producing a plausible wrong number.
_GUARANTEED_METHODS = (
    "f",
    "F",
    "G",
    "F_yy_action",
    "F_yu_action",
    "F_uu_action",
)


class TimeVaryingAffineDynamics:
    """``y' = M(t) y + C(t) u + b(t)``, coefficients supplied as functions of
    time.

    Args:
        M: Callable ``t -> (n, n)`` state matrix.
        C: Callable ``t -> (n, nu)`` control matrix.
        b: Callable ``t -> (n,)`` forcing. ``None`` means identically zero.
        state_dim: ``n``. Required explicitly rather than discovered by
            calling ``M`` once, because calling a coefficient at one time and
            asserting the shape of its value at every other time is a probe,
            and generalising a probe is precedent R-9. Given ``n`` and ``nu``,
            every coefficient value is checked at the time it is actually
            used.
        control_dim: ``nu``.

    What is constructive here, and what is a caller obligation
    ----------------------------------------------------------

    :class:`AffineDynamics` can be read for affineness because it owns its
    coefficients. This class does not own them — the caller supplies three
    functions this module never sees the body of — so the guarantee has to be
    stated more carefully than that one, and the two halves must not be run
    together.

    **Constructive.** The *form* of ``f`` is fixed by this class and the caller
    cannot change it: ``f = M(·)y + C(·)u + b(·)``, with ``y`` and ``u``
    entering linearly and exactly once, verified by the same class-identity
    check that protects :class:`AffineDynamics`. The caller supplies coefficient
    *values*, not the functional form. That is a far narrower freedom than the
    one C-16.2 refuses, where ``jointly_affine`` is asserted of an opaque ``f``
    about whose form nothing at all is known.

    **A caller obligation.** Given that form, the curvature blocks vanish if and
    only if ``M``, ``C`` and ``b`` depend on ``t`` alone. This class confines
    the *interface* — no ``y`` and no ``u`` is ever passed to a coefficient —
    but confining the interface does not confine a Python closure. A callable
    may capture the control array the optimizer is differentiating and read it,
    and then ``f`` is not affine in ``u`` while every check here still passes.

    It would be wrong to write that a function never given ``y`` cannot depend
    on ``y``. It cannot depend on ``y`` *through this interface*, which is a
    weaker statement and is the one that holds.

    C-1's caller-trust model is what makes the remainder admissible: inputs are
    data-only, and a callable that reads the optimizer's own control array is
    not a data-only input. The obligation is narrow, checkable by reading three
    short functions, and it is the caller's. No cheap exact check for it exists
    on this side — a coefficient that varies only when ``u`` varies is
    *constant* throughout any single gradient evaluation, so no
    consistency comparison within a solve can see it.

    **The failure direction is silent**, which is why it is stated here rather
    than left implicit. Measured: ``y' = m(u)y + u`` with
    ``m = -0.2 + 0.7u`` smuggled in through a closure, ``y₀ = 0.8``,
    ``t_span = (0, 0.5)``, ``N = 1``, explicit Euler, ``J = y₁²/2``, at
    ``u = 0.3``. This class returns ``0.477``; a central difference of the
    objective at ``ε = 1e-6`` gives ``0.744``. Nothing raises.

    What this class does *not* license
    ----------------------------------

    ``F`` is **not** constant. ``coefficients_constant`` is ``False``, so
    ``jacobian_constant`` is deduced ``False`` and the C-15 factorization store
    reuses nothing. Each implicit stage assembles and factors its own matrix.
    That is not a regression against :class:`AffineDynamics`; it is the
    arithmetic the problem actually requires, since ``I - h a_ii M(t_i)``
    genuinely differs between stages at distinct abscissae.

    Determinism is a caller obligation
    ----------------------------------

    ``M``, ``C`` and ``b`` must be deterministic functions of ``t``: the same
    ``t`` must yield the same value. A coefficient that drifted between calls
    would make the adjoint apply the transpose of a matrix the forward solve
    did not use, which is the C-5.4 failure mode.

    This obligation is not checked exhaustively — that would require comparing
    against every earlier call — but it is not unguarded either. On the affine
    stage route, C-16.4 evaluates the residual after the solve using fresh
    coefficient values and tests it against the C-5.1 threshold, so a
    coefficient that changed between assembling ``K`` and evaluating ``R``
    raises :class:`~adjungo.solvers.linear_stage.NonAffineStageEquation`
    instead of returning an unconverged stage value. The error direction is
    toward a loud refusal.

    That guard is real but partial: ``F`` and ``G`` are evaluated once more
    after the residual check, to be stored for the backward sweeps, and a
    coefficient that changed on *that* call would not be caught. Determinism is
    an obligation, not a verified fact.
    """

    def __init__(
        self,
        M: Callable[[float], NDArray],
        C: Callable[[float], NDArray],
        b: Callable[[float], NDArray] | None = None,
        *,
        state_dim: int,
        control_dim: int,
    ) -> None:
        for name, fn in (("M", M), ("C", C), ("b", b)):
            if fn is not None and not callable(fn):
                raise TypeError(
                    f"{name} must be a callable of t alone; got "
                    f"{type(fn).__name__}. For coefficients that do not vary "
                    f"in time, use AffineDynamics, which takes arrays and "
                    f"reuses one factorization for the whole solve."
                )

        n = int(state_dim)
        nu = int(control_dim)
        if n <= 0 or nu <= 0:
            raise ValueError(
                f"state_dim and control_dim must be positive; got "
                f"state_dim={n}, control_dim={nu}"
            )

        self._M_fn = M
        self._C_fn = C
        self._b_fn = b
        self._n = n
        self._nu = nu

    def _coefficient(
        self,
        fn: Callable[[float], NDArray],
        t: float,
        shape: tuple[int, ...],
        name: str,
    ) -> NDArray:
        """Evaluate one coefficient at ``t`` and check its shape there.

        The value is copied and made non-writeable. Copying costs ``O(n^2)``
        and guards the ``O(n^3)`` solve that consumes it, the same ratio C-15.2
        uses to justify its element-for-element comparison. It matters because
        a caller callable is free to return the same scratch buffer on every
        call and then write into it; the returned array would then change
        underneath a factorization already taken from it.
        """
        value = np.array(fn(float(t)), dtype=float, copy=True)
        if value.shape != shape:
            raise ValueError(
                f"{name}({t!r}) must have shape {shape}; got {value.shape}"
            )
        value.flags.writeable = False
        return value

    @property
    def coefficients_constant(self) -> bool:
        """``False``: ``M(t)`` may differ at every stage abscissa.

        Read by ``deduce_structure`` to set ``jacobian_constant``, which gates
        C-15 factorization reuse. Returning ``True`` here for a genuinely
        time-varying ``M`` would not produce a wrong answer: C-15.2 compares
        the stored matrix with the requested one element for element and
        raises ``DeclaredStructureViolation`` on the first disagreement.
        """
        return False

    @property
    def state_dim(self) -> int:
        return self._n

    @property
    def control_dim(self) -> int:
        return self._nu

    def M(self, t: float) -> NDArray:
        """State matrix at ``t``, shape ``(n, n)``, read-only."""
        return self._coefficient(self._M_fn, t, (self._n, self._n), "M")

    def C(self, t: float) -> NDArray:
        """Control matrix at ``t``, shape ``(n, nu)``, read-only."""
        return self._coefficient(self._C_fn, t, (self._n, self._nu), "C")

    def b(self, t: float) -> NDArray:
        """Forcing at ``t``, shape ``(n,)``, read-only."""
        if self._b_fn is None:
            return np.zeros(self._n)
        return self._coefficient(self._b_fn, t, (self._n,), "b")

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.asarray(self.M(t) @ y + self.C(t) @ u + self.b(t))

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.M(t)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self.C(t)

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(n, n)``.

        Defined rather than omitted for the same reason as on
        :class:`AffineDynamics`: the route that skips this term must be
        comparable against the route that calls it, and C-16.6 requires the two
        to agree bit for bit.
        """
        return np.zeros((self._n, self._n))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(n, nu)``."""
        return np.zeros((self._n, self._nu))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """Identically zero, shape ``(nu, nu)``."""
        return np.zeros((self._nu, self._nu))


#: The members whose bodies carry each root class's guarantee, keyed by the
#: class that defines them. A module-level mapping rather than a class
#: attribute, because a class attribute naming the members to check could
#: itself be replaced by the subclass being checked.
#:
#: ``coefficients_constant`` is guarded alongside the computational methods
#: because it decides whether C-15 reuse is enabled. The coefficient accessors
#: of the time-varying class are guarded because they are where the shape check
#: and the defensive copy live, and because they are the only places the
#: caller's callables are invoked — which is what confines those callables to
#: ``t`` and so establishes zero curvature.
_GUARANTEED_MEMBERS: dict[type, tuple[str, ...]] = {
    AffineDynamics: _GUARANTEED_METHODS + ("coefficients_constant",),
    TimeVaryingAffineDynamics: _GUARANTEED_METHODS
    + ("coefficients_constant", "M", "C", "b", "_coefficient"),
}

#: The guaranteed members the affine route never calls, and which may
#: therefore be replaced **on the instance** without disturbing the guarantee.
#:
#: This exemption is not a convenience. C-16.5 requires a tripwire that
#: replaces exactly these three with functions that raise and then asserts a
#: **Hessian-vector product** still computes — evidence, unobtainable from a
#: counter, that no path reaches a curvature term. A gradient would prove
#: nothing here, since the gradient never consults these callbacks on any
#: route. The tripwire only works if shadowing them leaves the problem
#: verified, so the exemption is load-bearing and must not be closed without
#: amending C-16.5.
#:
#: It is sound **on the route verification selects**, and that qualifier is
#: the whole of it: the deduced structure sets ``jointly_affine=True``, under
#: which C-16.6 skips these three, and a replacement that is never called
#: cannot change an answer. C-16.9 states the boundary. Entry points that do
#: not consult the deduced structure — an explicit ``ProblemStructure`` with
#: ``jointly_affine=False``, :func:`adjungo.validation.reference.reference_hessian`,
#: or ``assemble_hessian_vector_product``/``adjoint_sensitivity`` called
#: directly with ``structure=None`` — call these callbacks and would read a
#: shadow. Those are outside the exemption's justification, and C-1 governs
#: them.
_SKIPPED_ON_THE_AFFINE_ROUTE = ("F_yy_action", "F_yu_action", "F_uu_action")

#: Attribute lookup itself is part of the guarantee. Every comparison below
#: would otherwise reach the problem through ordinary attribute access, so a
#: class that redefines how attribute access works can answer the check with
#: one object and the solver with another. ``__dict__`` is here for the same
#: reason and is not hypothetical: a subclass defining ``__dict__`` as a
#: property returning ``{}`` hides its own instance shadows from the check
#: below, and reinstates the C-16.9 silent Hessian error exactly.
#:
#: These are conservative refusals, not free ones. A subclass using
#: ``__getattr__`` for unrelated metadata is refused too, and pays the general
#: route for it. That is the correct direction under C-16.2.
_LOOKUP_HOOKS = ("__getattribute__", "__getattr__", "__dict__")

#: Distinguishes "defined on neither" from "defined on one". The lookup hooks
#: are defined on neither root, so the comparison needs a default on both
#: sides. An object sentinel rather than ``None`` so that "not defined" and
#: "defined as ``None``" stay distinct, which costs nothing and removes a
#: question.
_MISSING: Any = object()


def _defined_as(cls: type, name: str) -> Any:
    """The object ``name`` is bound to in ``cls``'s MRO, found without
    invoking the descriptor protocol.

    ``getattr(cls, name)`` is not usable here. It is itself attribute access,
    so it consults the very machinery this module is trying to verify: a class
    that defines ``__dict__`` as a property gets that property *called*, and
    reports whatever it likes. Walking ``__mro__`` and reading each class's own
    ``__dict__`` reads the binding instead of its result.

    Class ``__dict__`` is a ``mappingproxy`` reached through the metaclass, so
    it is not interceptable by the class being checked. This is the one lookup
    in this module that has to be trusted, and it is the narrowest available.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return _MISSING


def affine_dynamics_verified(
    problem: Any,
) -> TypeGuard[AffineDynamics | TimeVaryingAffineDynamics]:
    """Whether ``problem`` is affine *by construction*, checked exactly.

    Being an instance of an affine root class is not by itself sufficient. A
    subclass can override ``f`` with anything at all, and then the coefficients
    describe nothing. This checks that every member carrying the guarantee is
    still the one that root class defines, by object identity on the attribute
    found through the MRO.

    The check is exact and has no failure mode toward acceptance: an overridden
    member is a different object, so the answer is ``False`` and the general
    route is taken. Error direction is toward doing more work, never toward
    claiming a structure the problem does not have. This mirrors the M6 rule
    that the declaration selects the route and an exact comparison establishes
    the fact (``NUMERICS.md`` C-15.2, C-16.2).

    A class deriving from **more than one** root is refused outright rather
    than verified against whichever one the MRO happens to resolve first. Such
    a class mixes two initialisers and two notions of ``coefficients_constant``,
    and the cost of refusing it is that a caller who wrote one gets the general
    route.

    Three things are checked, and the second is the one that is easy to omit:

    1. **The class.** Every guaranteed member bound in the MRO is the object
       the root class binds, read by :func:`_defined_as` rather than by
       ``getattr`` so that the descriptor protocol cannot answer for it.
    2. **The instance.** No *invoked* guaranteed member is shadowed in the
       problem's own ``__dict__``. Checking only the class leaves
       ``problem.f = something_else`` verified. What that costs is narrower
       than it first appears and is worth stating exactly: an instance
       carrying a *consistent* nonlinear system still produces an exact
       gradient, because the gradient reads only ``f``, ``F`` and ``G`` and
       those agree — but the route is chosen from the stale class-level fact,
       so C-16.6 skips a curvature term that is now nonzero and the **Hessian**
       is silently wrong. C-16.9 measures it at ``0.911``. An inconsistent
       ``f``/``F`` pair is a different thing entirely and not this: it is
       equally wrong on the general route, and verification never bore on it.
       The members in :data:`_SKIPPED_ON_THE_AFFINE_ROUTE` are exempt, because
       C-16.5's tripwire requires shadowing them to leave the problem verified.
    3. **Attribute lookup.** ``__getattribute__``, ``__getattr__`` and
       ``__dict__`` are the root's, so the members compared above are the
       members the solver will later receive, and the instance storage read in
       step 2 is the real one.

    Per C-1 this is not a claim of robustness against a hostile caller, and it
    is not one. What it does not reach: patching the **root class itself**
    (``AffineDynamics.f = ...`` changes both sides of every comparison, so
    every comparison still passes), and, for the time-varying class, the caller
    keeping a reference to a coefficient callable and rebinding what it
    computes. Both are beyond a cheap exact check. What it does reach is the
    ordinary case — a monkeypatch left in place, a cached bound method, a
    subclass that means well — and the error direction remains one way, toward
    the general route.

    The return type narrows to the root classes rather than being a plain
    ``bool``. That is the point of the check: a caller that has passed it may
    read ``coefficients_constant`` directly, instead of reaching for it with a
    ``getattr`` default that would silently choose a route for a problem that
    did not supply the fact. C-7 forbids that kind of default.
    """
    roots = [
        root for root in _GUARANTEED_MEMBERS if isinstance(problem, root)
    ]
    if len(roots) != 1:
        return False
    root = roots[0]
    cls = type(problem)

    for name in _LOOKUP_HOOKS:
        if _defined_as(cls, name) is not _defined_as(root, name):
            return False

    guaranteed = _GUARANTEED_MEMBERS[root]
    if any(
        _defined_as(cls, name) is not _defined_as(root, name)
        for name in guaranteed
    ):
        return False

    # Safe only because ``__dict__`` was compared above: the descriptor
    # reached here is the root's, so this returns real instance storage rather
    # than whatever a subclass would prefer the check to see. A class using
    # __slots__ has no instance dict, and a slot shadowing a guaranteed member
    # is a class-level descriptor already refused above.
    shadowed = getattr(problem, "__dict__", {})
    return not any(
        name in shadowed
        for name in guaranteed
        if name not in _SKIPPED_ON_THE_AFFINE_ROUTE
    )
