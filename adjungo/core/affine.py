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
    "InvalidCoefficients",
    "MissingCoefficients",
    "MutableCoefficients",
    "SubstitutedCoefficients",
    "TimeVaryingAffineDynamics",
    "affine_dynamics_verified",
    "require_immutable_coefficients",
]


class InvalidCoefficients(RuntimeError):
    """An affine problem's live coefficient buffers cannot be relied on.

    Base for the two ways that happens — the data can be rewritten, or it was
    never established. Both are refusals that **raise rather than return**
    ``False``, which is what separates them from every other refusal in this
    module and is the substance of C-15.7.

    The others are safe to answer quietly because the general route then
    computes the same derivative, only more slowly. These are not: the general
    route reads ``F`` and retains what it returns exactly as the affine route
    does, so a problem whose coefficients can change mid-solve is wrong by the
    same amount on both. Declining to certify it would pick a route already
    known to give no better an answer — the silent-sentinel shape C-7 forbids.

    That asymmetry also decides *when* the check runs: before the eligibility
    checks rather than after them, since a problem refused quietly for some
    unrelated reason is handed the general route and displaced there by the
    same amount. See :func:`require_immutable_coefficients` for the
    precondition that makes running first safe.
    """

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name


class MutableCoefficients(InvalidCoefficients):
    """A coefficient buffer can still be written — C-15.7.

    ``AffineDynamics`` freezes its coefficients at construction so that ``F``
    may return ``self._M`` itself rather than a copy. The backward sweeps then
    read a tape of references to that one buffer, and the object's whole claim
    to be affine by construction rests on the buffer not changing under them.

    Reached in practice by a subclass that replaces a buffer after
    ``super().__init__`` — ``self._M = self._M.copy()`` looks defensive and
    produces a writeable array — and by deep-copying a subclass that overrides
    a copy hook, where the override both unfreezes the buffers and is itself
    grounds for a quiet refusal.
    """

    def __init__(self, cls: type, name: str, reason: str) -> None:
        super().__init__(
            name,
            f"{cls.__name__} instance has a coefficient buffer {name!r} "
            f"that is {reason}. AffineDynamics freezes its coefficients at "
            f"construction because F returns the buffer itself and the "
            f"backward sweeps hold references to it; a buffer that can be "
            f"rewritten between the forward solve and the adjoint seeds the "
            f"adjoint with a matrix the forward solve never used "
            f"(NUMERICS.md C-15.7). This cannot be answered by taking the "
            f"general route, which reads F the same way. Build the problem "
            f"with AffineDynamics(M, C, b) and do not rebind or replace "
            f"{name!r} afterwards.",
        )


class SubstitutedCoefficients(InvalidCoefficients):
    """A setter stored an array other than the one it was handed — C-15.7.

    Distinct from :class:`MutableCoefficients` because the array in place is
    perfectly frozen; what is wrong is its provenance. The root freezes each
    coefficient before assigning it, so an instance holding exactly those
    arrays holds memory no subclass code has ever seen writeable. A setter
    that substitutes storage of its own breaks that chain, and it can — with
    no more than an ordinary copy, a retained view, and a line matching the
    argument's ``writeable`` flag — leave a writeable alias of the tape's
    memory behind a frozen owner that every check accepts.

    A separate class because the remedy is different: nothing here is
    writeable and nothing is missing, so neither of those messages would send
    the caller anywhere useful.
    """

    def __init__(self, cls: type, name: str, stored: object = None) -> None:
        found = (
            "nothing"
            if stored is _MISSING or stored is None
            else f"a different {type(stored).__name__}"
        )
        super().__init__(
            name,
            f"{cls.__name__} defines a setter for the coefficient buffer "
            f"{name!r} that does not store the array it is handed: after "
            f"assignment, {name!r} resolves to {found}. AffineDynamics "
            f"freezes each coefficient before assigning it, so that the array "
            f"F returns and the backward sweeps hold is one no subclass has "
            f"seen writeable; a setter that stores a copy instead can keep a "
            f"writeable handle onto that copy, and the buffer then changes "
            f"under a tape that verified as immutable (NUMERICS.md C-15.7). "
            f"Relocating the array is fine -- store {name!r} itself under any "
            f"key or slot you like. To normalise or reshape it, do so before "
            f"calling super().__init__, where the array is still your own."
        )


class MissingCoefficients(InvalidCoefficients):
    """The root's coefficient storage no longer resolves to an ``ndarray``.

    ``AffineDynamics.__init__`` establishes all of ``_COEFFICIENT_BUFFERS`` as
    frozen, owning arrays and records that it did, so once that record is
    present anything else found in their place -- deleted, or replaced by a
    sparse matrix, a ``memoryview``, or any other storage whose aliasing
    cannot be reasoned about -- is a modification made after construction.

    It raises rather than being skipped. Skipping is the silent direction:
    coefficients the check could not read were once treated as coefficients
    that were not there, so the problem verified as affine and the tape
    aliased them unchecked.

    Not an error when the name is live to none of the three mechanisms.
    Nothing then retains a buffer of this class's, and
    :func:`require_immutable_coefficients` says nothing -- C-16.2 requires
    that refusal to be quiet. An instance that inherits ``f``, ``F`` or ``G``
    while holding nothing for them to read *is* an error, and is named here
    rather than left to raise ``AttributeError`` several frames later; so is
    one holding unusable storage under a name the root would have owned.

    A separate class from :class:`MutableCoefficients` because the remedy is
    different and the word would otherwise be false: nothing here is mutable.
    """

    def __init__(
        self, cls: type, name: str, found: object = None, rooted: bool = True
    ) -> None:
        found_desc = (
            "is absent"
            if found is None
            else f"resolves to {type(found).__name__}, not ndarray"
        )
        buffers = ", ".join(map(repr, _COEFFICIENT_BUFFERS))
        origin = (
            f"{cls.__name__} has AffineDynamics coefficient storage "
            f"established -- {buffers} as frozen, owning arrays -- but "
            f"{name!r} now {found_desc}. Something has replaced it since, and "
            f"its aliasing cannot be checked"
            if rooted
            else f"{cls.__name__} did not run AffineDynamics.__init__, but "
            f"{name!r} is one of the root's coefficient names and "
            f"{name!r} {found_desc}. It is live either because the class "
            f"inherits a member that reads it or because the instance holds "
            f"it"
        )
        super().__init__(
            name,
            f"{origin}, so the first solve would retain it unguarded "
            f"(NUMERICS.md C-15.7). Leave the root's buffers as constructed, "
            f"or do not call super().__init__ at all and supply f, F and G "
            f"yourself.",
        )


#: The instance attributes holding frozen coefficient data. Named once because
#: ``__setstate__`` has to re-freeze exactly what ``__init__`` froze, and a
#: fourth buffer added to one and not the other would be silently writeable
#: after a deep copy.
_COEFFICIENT_BUFFERS = ("_M", "_C", "_b")


#: The root members that hand a coefficient buffer to a caller, and which
#: buffers each one reads. These three and no others: ``M``, ``C`` and ``b``
#: return by identity too, but nothing on the solve path reads them, and
#: counting them would refuse a subclass that replaces all of ``f``, ``F`` and
#: ``G`` while leaving the public accessors inherited -- which is exactly the
#: certified shape that keeps its own writeable storage under one of these
#: names.
#:
#: The mapping is per-member rather than a flat set so that inheriting one
#: reader does not demand the buffers the other two would have read. ``F``
#: owes ``_M`` alone.
_ROOT_BUFFER_READERS: dict[str, tuple[str, ...]] = {
    "f": _COEFFICIENT_BUFFERS,
    "F": ("_M",),
    "G": ("_C",),
}


#: Records that all of :data:`_COEFFICIENT_BUFFERS` have been established on
#: *this* instance as frozen, owning ``ndarray``\ s. ``AffineDynamics.__init__``
#: writes it, and so does ``__setstate__`` when restoration has put all three
#: into that state -- which it may do for an instance whose initialiser never
#: ran. So it records the postcondition, not the provenance: read it as "root
#: coefficient storage is established here", never as "the constructor ran".
#: Written into ``__dict__`` directly so that no subclass property or slot can
#: intercept it.
#:
#: This is the primary precondition for the immutability invariant, and it is
#: recorded rather than inferred because neither available inference answers
#: it alone. Asking which of ``f``, ``F``, ``G`` are still the root's own
#: definitions is defeated by an override that delegates to ``super()``, so it
#: cannot *replace* the flag -- though it is sound as a widening beside it,
#: which is what :func:`_live_buffer_names` does with it. Asking whether the
#: buffers are *present* means evaluating ``_M``, ``_C`` and ``_b`` on a
#: subclass that may never have had them -- running descriptors that are not
#: ours to run, and reading a partial set as damage when it is simply a
#: subclass using one of the same private names.
_ROOT_INITIALISED = "_affine_root_initialised"


#: Returned by :func:`_resolve_buffer` for a coefficient name that hands out
#: nothing, and the default on both sides of :func:`_defined_as`'s comparison.
#: An object sentinel rather than ``None`` so that "not there" and "there, and
#: it is ``None``" stay distinct, which costs nothing and removes a question.
_MISSING: Any = object()


def _resolve_buffer(problem: object, name: str) -> Any:
    """What ``name`` hands out on *problem*, or :data:`_MISSING` if nothing.

    Every part of this clause asks this one question, so it is asked in one
    place. Verification uses it to decide which buffers are live and
    ``__setstate__`` to decide which to re-freeze, and those two must not
    differ: a name that is live to the check and invisible to restoration
    leaves a copy unfrozen, and a name that is invisible to the check and
    live to restoration makes an ordinary ``copy.deepcopy`` fail on a problem
    the optimizer accepts. Both were observed before this was shared.

    Reading is not passive -- a coefficient descriptor is executed here -- and
    one that refuses to answer hands out nothing rather than propagating.
    ``getattr`` with a default absorbs only ``AttributeError``, and a property
    raising anything else made duplication raise out of a valid general-route
    problem while the check stayed quiet about the same instance.

    The direction is safe: a name that cannot be read aliases nothing, and if
    it were load-bearing the reader reaching for it during the solve raises
    from the same descriptor, loudly. What is never absorbed is storage that
    *is* readable and is not a frozen owning ``ndarray``; that raises.
    """
    try:
        return getattr(problem, name, _MISSING)
    except Exception:  # noqa: BLE001 - a reader refusing to answer hands out nothing
        return _MISSING


def _resolving_buffer_names(problem: object) -> tuple[str, ...]:
    """The root coefficient names that hand something out on *problem*.

    The shared half of the two liveness questions. ``__setstate__`` freezes
    exactly these, because they are what exists to be frozen, and
    :func:`_live_buffer_names` adds to them the names an inherited reader
    obliges but that resolve to nothing -- which restoration cannot conjure
    and the guard at the route decision refuses by name instead.
    """
    return tuple(
        name
        for name in _COEFFICIENT_BUFFERS
        if _resolve_buffer(problem, name) is not _MISSING
    )


def _require_stored_as_handed(problem: object, name: str, handed: Any) -> None:
    """Refuse a setter that stored something other than what it was handed.

    The one rule that makes the freeze mean anything. Everything else in this
    clause arranges for the array a coefficient resolves to be frozen; this
    arranges for that array to be *the root's own*, frozen before any
    subclass code saw it. Those are not the same promise, and only the second
    one can be kept.

    A setter is ordinary code and may allocate storage of its own. If it
    does, it can derive a handle from that storage while it is still writeable
    and then match the argument's ``writeable`` flag onto the copy it stores.
    The result passes every check this clause makes -- the coefficient
    resolves to an owning, frozen array -- while the subclass retains a
    writeable alias of the very memory the tape holds. Measured on the C-15.7
    fixture at ``C = 2``, from plain construction, from ``copy.copy`` and from
    a default-protocol pickle: the problem verified and the optimizer returned
    ``0.7552125`` against a closed-form ``0.6781500``.

    No ordering reaches it. Restoration and the constructor both freeze before
    they assign, but the setter picks which memory becomes the coefficient,
    and it can pick memory it has already aliased. The only closure is to
    refuse the substitution.

    What that leaves is narrower than it first appears, and the difference is
    worth stating. Once the coefficient is the array the root froze, NumPy
    refuses to make a writeable view of it -- by ``view``, ``as_strided``,
    ``reshape`` or ``frombuffer`` alike -- so a setter cannot derive a
    writeable handle from what it was handed. It *can* unfreeze that array
    first, take the handle, and freeze it again before storing it. Nothing
    distinguishes the result, and nothing here tries to: deliberate
    unfreezing is the excluded class C-15.7 names, and this is it happening a
    few lines earlier than the usual example. The rule closes substitution,
    which was reachable without intending it; it does not close sabotage.

    Nor does it reach a subclass that never calls ``super().__init__``. There
    is no root array to compare against there, and requiring one would refuse
    every unrooted general-route problem that happens to hold these names --
    a shape this clause deliberately supports. Such a subclass owns its
    coefficients outright, and its buffers are still checked for being frozen
    and owning; their provenance is its own affair.

    The cost is narrow and loud. A setter may still *relocate* what it is
    handed -- into another key, into a slot, under another name -- and several
    supported subclasses do. It may not store a substitute. A subclass that
    wants to normalise its coefficients does so before ``super().__init__``,
    where the array is still its own.
    """
    stored = _resolve_buffer(problem, name)
    if stored is not handed:
        raise SubstitutedCoefficients(type(problem), name, stored)


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

    The immutability survives copying and pickling; see ``__setstate__`` and
    C-15.7 for why that needs saying at all.
    """

    def __init__(
        self, M: NDArray, C: NDArray, b: NDArray | None = None
    ) -> None:
        # An ``ndarray`` subclass is refused rather than converted. Converting
        # is what ``np.array(..., dtype=float)`` does, and for a MaskedArray it
        # discards the mask: the caller's coefficient would silently become a
        # different matrix. Refusing here keeps one rule with the check, which
        # cannot certify a subclass buffer at all — see
        # ``require_immutable_coefficients``. A list, tuple or scalar loses
        # nothing by being converted and still may be.
        for label, value in (("M", M), ("C", C), ("b", b)):
            if isinstance(value, np.ndarray) and type(value) is not np.ndarray:
                raise ValueError(
                    f"{label} must be an ndarray or ordinary sequence; got "
                    f"{type(value).__name__}, an ndarray subclass. Converting "
                    f"it would discard state the coefficient depends on (a "
                    f"MaskedArray's mask), and that state cannot be frozen "
                    f"with the buffer (NUMERICS.md C-15.7). Pass "
                    f"np.asarray(...) of the values you mean."
                )

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
            else np.array(np.reshape(b, -1), dtype=float, copy=True)
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
        # Checked after all three, not after each: one coefficient's setter is
        # free to rewrite another, and a subclass that does exists.
        for name, handed in (("_M", M_arr), ("_C", C_arr), ("_b", b_arr)):
            _require_stored_as_handed(self, name, handed)
        self._nu = int(C_arr.shape[1])
        #: Set last, so it is present only if every buffer above was
        #: established and frozen. Written through the root's own ``__dict__``
        #: descriptor, which is how :func:`_root_buffers` reads it: a subclass
        #: exposing a different mapping as ``__dict__`` would otherwise take
        #: delivery of the marker while the read went to the real instance
        #: dictionary and found nothing, silencing the check on precisely the
        #: kind of subclass it exists for. Reading defensively and writing
        #: trustingly protects nothing.
        _AFFINE_INSTANCE_DICT.__get__(self)[_ROOT_INITIALISED] = True

    def __setstate__(self, state: Any) -> None:
        """Restore state and re-freeze the coefficient buffers — C-15.7.

        ``copy.deepcopy`` does not preserve ``writeable=False``: NumPy defines
        ``ndarray.__deepcopy__``, and it returns a writeable array. Pickle,
        which goes through ``__reduce__``, does preserve the flag, and
        ``copy.copy`` shares the original buffers, so for an instance this
        class constructed, only the deep copy unfreezes. A subclass supplying
        its own ``__getstate__`` can unfreeze any of the three, which is why
        the rule below does not depend on which route ran.

        This hook is the one both the copy and pickle protocols run **by
        default**, so re-freezing here covers every route that does not replace
        the protocol itself. A subclass defining ``__reduce__``, ``__copy__``
        or ``__deepcopy__`` bypasses it, which is why those are among the
        hooks verification compares. Reconstructing through ``__deepcopy__``
        instead would have discarded state a subclass added, which is a worse
        failure for a smaller gain.

        The two-element form of ``state`` has to be handled for that reason to
        hold: ``object.__reduce_ex__`` reports a subclass carrying ``__slots__``
        as ``(instance_dict, slot_values)``, and such a subclass is verified,
        since adding a slot overrides no guaranteed member. Assuming a plain
        dict here would raise on every copy of one — discarding its state in a
        different and louder way.

        Restoration resolves the buffers by the **same rule verification uses**
        — attribute access, which follows a slot or a property exactly as ``F``
        does — and restores them the same way. Literally the same rule: both go
        through :func:`_resolve_buffer`, because two spellings of one
        question drifted apart once already. ``getattr`` with a default
        absorbs only ``AttributeError``, so a general-route subclass whose
        ``_M`` property raised anything else was accepted by the optimizer
        and could not be copied, deep-copied or pickled at all; and in the
        other direction, obliging only names holding an ``ndarray`` here
        let a delegating subclass with ``csr_array`` coefficients -- which
        the guard refuses -- round-trip quietly into a writeable copy. Reading them out of the instance
        dictionary instead re-froze whatever happened to be stored there, which
        for a subclass declaring ``__slots__ = ("_M",)`` is nothing at all:
        that shape verified and solved correctly but raised ``KeyError`` on
        every ``copy.copy``, ``copy.deepcopy`` and pickle round trip. Certifying
        a shape for one obligation of this clause and not the other is the
        inconsistency, not the exception.

        **Which** buffers are frozen is asked of the restored object, never of
        the state it arrived in. Three rules that read the state were tried and
        a subclass defeated each by shaping it: the marker can be dropped from
        ``__getstate__``; a buffer can be omitted alongside it, so requiring
        all three fails too; and a coefficient can be delivered under another
        name entirely, by a property setter that writes the slot, so it is live
        and aliased by the inherited ``F`` while appearing in no restored key.
        Each produced the displacement this clause measures through the public
        optimizer with every guard quiet. The state is written by the subclass
        and the object is what the solve will read, so every coefficient name
        that resolves to an array on the finished instance is frozen.

        That binds what exists when this runs, and nothing later. A subclass
        can rebuild a coefficient lazily on first access, so there is no array
        here to freeze; what closes that is the guard at the route decision,
        which asks the same question again once the buffer exists. That is
        also the one half of liveness restoration does *not* take from the
        guard: a name obliged by an inherited reader but resolving to nothing
        cannot be conjured here, so it stays the route decision's to refuse.
        Restoration freezes what resolves; the guard obliges what must.

        Where the marker survived, the root established all three and all three
        are required, so anything else raises, which is the C-7 direction:
        there is no correct value to substitute. Restoration then records the
        marker itself rather than passing on the one it was handed, so a later
        check reads what restoration established.

        Freezing is a **second pass** over re-read values. ``setattr`` runs a
        subclass property setter, and one that stores a copy of its argument
        leaves the caller's name bound to the object that was discarded, while
        one that rewrites a *different* coefficient replaces a buffer frozen
        earlier in the same loop. Both returned owning, *writeable* buffers
        from an ordinary round trip of a subclass that verified before it.
        Nothing is frozen until no coefficient setter can still run. Acting on
        a value fetched by a path other than the one that will be read is the
        mistake this clause keeps making, in each of its parts.

        **The replacement is nevertheless frozen before the setter sees it**,
        which is a different claim and does not soften that one. The second
        pass asks what the object resolves to once every setter has run, and
        so only ever reaches the owner. A setter is free to derive a *second*
        handle from its argument -- a view, a reshape, a slice kept for fast
        access -- and ``writeable`` is a per-array flag that does not reach
        views already made, so a handle taken from a writeable replacement
        stayed writeable over the coefficient's own memory while the owner was
        frozen behind it. Handing the setter a read-only array closes that at
        the source, a view of a read-only array being read-only.

        Without the marker, a buffer that is still writeable is **copied
        before it is frozen** even when it already owns its storage. ``copy.copy`` shares the array with
        the original, so freezing in place reached back through that sharing
        and froze the source — changing an object the caller did not ask to
        change. The marked path keeps sharing, because there both arrays are
        frozen already and there is nothing to lose. The copy is made by
        constructing a base array rather than by calling the buffer's own
        ``copy``, which a subclass of ``ndarray`` is free to define: one
        returning ``self`` put the freeze straight back onto the source this
        branch exists to protect.

        The cost is narrow and loud. A subclass may reuse a coefficient name
        for writeable storage of its own — nothing forbids it, and one does —
        but its copies cannot be written to, and will answer ``assignment
        destination is read-only``. That is the trade this clause makes
        everywhere: refuse a working problem audibly rather than stay quiet
        about a broken one.

        A buffer already at the constructor's postcondition -- owning and
        frozen -- is left exactly where it is, because replacing it would gain
        nothing and costs the only thing this branch can lose: a read-only
        property answering for a coefficient cannot be assigned to, and an
        ordinary round trip of a subclass the optimizer accepts raised
        ``property '_M' ... has no setter``. When the buffer is writeable and
        the storage still refuses assignment there is no safe outcome -- the
        copy cannot be stored and freezing the original would change an object
        the caller did not ask to change -- so restoration refuses, names the
        buffer and says why, rather than letting a bare ``AttributeError`` out
        of ``copy.deepcopy``.

        A buffer that does not own its storage is replaced by one that does,
        rather than merely frozen. Pickle reconstructs an ``ndarray`` as a view
        over the pickled ``bytes``, and while those particular bytes are
        immutable, "frozen view over something else" is a weaker property than
        the constructor establishes and a more expensive one to check. The copy
        is ``O(n²)`` and restores exactly the constructor's postcondition, which
        is what :func:`require_immutable_coefficients` is then able to assert
        cheaply.
        """
        slots: dict[str, Any] | None = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slots = state
        storage = _AFFINE_INSTANCE_DICT.__get__(self)
        if state:
            storage.update(state)
        if slots:
            for name, value in slots.items():
                object.__setattr__(self, name, value)

        rooted = bool(storage.get(_ROOT_INITIALISED, False))
        names = _COEFFICIENT_BUFFERS if rooted else _resolving_buffer_names(self)
        if not names:
            return

        handed: list[tuple[str, NDArray[np.float64]]] = []
        for name in names:
            buffer = _resolve_buffer(self, name)
            if not isinstance(buffer, np.ndarray):
                continue
            if buffer.flags.owndata and not buffer.flags.writeable:
                # Already the constructor's postcondition. Replacing it would
                # gain nothing and costs the one thing this branch can lose:
                # a read-only property answering for a coefficient cannot be
                # assigned to, and an ordinary round trip of a subclass the
                # optimizer accepts raised ``property has no setter``.
                continue
            if rooted and buffer.flags.owndata:
                continue
            # Not ``buffer.copy()``: an ``ndarray`` subclass may override it,
            # and one returning ``self`` put the freeze back on the shared
            # source this branch exists to protect, while one perturbing the
            # data changed a coefficient silently. This dispatches to no
            # user-defined ``copy``. It does run ``__array_finalize__``, which
            # is unavoidable for any subclass-preserving construction;
            # ``subok=True`` is kept because normalising here would quietly
            # turn a buffer the check must refuse into one that passes.
            replacement = np.array(buffer, copy=True, subok=True)
            # Frozen *before* the setter sees it, not only afterwards. A setter
            # is free to derive a second handle from its argument -- a view, a
            # reshape, a slice kept for fast access -- and a handle taken from
            # a writeable array stays writeable when the array it came from is
            # frozen later, because that flag is per-array and does not
            # propagate to views already made. The second pass then froze the
            # owner while the derived view kept writing to the same memory, and
            # a default-protocol pickle round trip of such a subclass verified
            # and returned the displacement C-15.7 measures. Freezing here
            # closes that at the source for a handle derived from *this*
            # array: a view of a read-only array is read-only. A setter that
            # allocates storage of its own instead is beyond any ordering, and
            # is refused below.
            replacement.flags.writeable = False
            try:
                setattr(self, name, replacement)
            except Exception as exc:
                # The reason is read off the buffer, not assumed. This branch
                # takes both a writeable buffer and a frozen one that does not
                # own its storage, and naming the wrong one sends the caller
                # after the wrong remedy.
                reason = (
                    "writeable, and freezing it where it lies would reach back "
                    "through the sharing a copy leaves and freeze the original"
                    if buffer.flags.writeable
                    else "frozen but does not own its storage, so it has to be "
                    "replaced by one that does"
                )
                raise MutableCoefficients(
                    type(self),
                    name,
                    f"{reason}; and it cannot be replaced with a frozen copy "
                    f"({type(exc).__name__}: {exc}). Give the attribute a "
                    f"setter, or leave the root's buffers as constructed",
                ) from exc
            handed.append((name, replacement))

        # Restoration does not run ``__init__``, so this is where the same
        # substitution is refused on the way back in; without it a pickle
        # reintroduces the subclass the constructor turns away. Deferred to
        # here for the reason the constructor defers it: one coefficient's
        # setter may replace another, so a name checked the instant it was
        # assigned is checked before the setter that rewrites it has run.
        for name, replacement in handed:
            _require_stored_as_handed(self, name, replacement)
        for name in names:
            buffer = _resolve_buffer(self, name)
            if not isinstance(buffer, np.ndarray) or not buffer.flags.owndata:
                raise MissingCoefficients(
                    type(self),
                    name,
                    None if buffer is _MISSING else buffer,
                    rooted,
                )
            buffer.flags.writeable = False
        if names == _COEFFICIENT_BUFFERS:
            storage[_ROOT_INITIALISED] = True

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

#: Copying is part of the guarantee for the same reason attribute lookup is.
#: ``__setstate__`` re-establishes the coefficient immutability that
#: ``copy.deepcopy`` would otherwise drop (C-15.7), and every entry here can
#: stop it running or hand back an object the constructor would never have
#: produced: ``__deepcopy__``, ``__copy__`` and ``__replace__`` pre-empt the
#: copy protocol outright, ``__reduce__`` and ``__reduce_ex__`` choose the
#: reconstructor and what is passed to it, and ``__getstate__`` chooses what
#: there is to restore. A subclass overriding ``__setstate__`` alone leaves
#: every other comparison in this module passing, so without this tuple the
#: cure is one method away from being undone.
#:
#: ``__replace__`` is the ``copy.replace`` hook added in Python 3.13. Neither
#: root defines it, so ``copy.replace`` on one raises; it is listed because a
#: subclass that defines it acquires a duplication route that runs none of the
#: others, and the policy here refuses the mechanism rather than judging each
#: implementation.
#:
#: Checked on both roots although only :class:`AffineDynamics` currently holds
#: frozen buffers. The question is whether copying can yield an instance
#: differing from a constructed one, and that is a question about both.
_COPY_PROTOCOL_HOOKS = (
    "__setstate__",
    "__getstate__",
    "__deepcopy__",
    "__copy__",
    "__replace__",
    "__reduce__",
    "__reduce_ex__",
)

#: ``type``'s own getset descriptors for ``__mro__`` and ``__dict__``, bound
#: once. Reading ``cls.__mro__`` or ``klass.__dict__`` is attribute access on a
#: class and so goes through the *metaclass*, which may define either as a
#: property; these are the real slots, and ``type`` is not a class any problem
#: under test gets to shadow.
_TYPE_MRO: Any = type.__dict__["__mro__"]
_TYPE_DICT: Any = type.__dict__["__dict__"]


def _defined_as(cls: type, name: str) -> Any:
    """The object ``name`` is bound to in ``cls``'s MRO, found without
    invoking the descriptor protocol.

    ``getattr(cls, name)`` is not usable here. It is itself attribute access,
    so it consults the very machinery this module is trying to verify: a class
    that defines ``__dict__`` as a property gets that property *called*, and
    reports whatever it likes. Walking ``__mro__`` and reading each class's own
    ``__dict__`` reads the binding instead of its result.

    Both of those are read through ``type``'s own getset descriptors rather
    than by attribute access on the class. ``cls.__mro__`` and
    ``klass.__dict__`` are attribute lookups on a *class*, so they consult the
    **metaclass**, and a metaclass defining either as a property answers them:
    one reported a fabricated MRO while Python's own method lookup went on
    reaching the root's ``f``, ``F`` and ``G``, and the gradient was displaced
    by the ``0.0770625`` of C-15.7 with every guard quiet. Going through
    ``type.__dict__`` gets the real slots, which a metaclass cannot shadow
    because ``type`` is where the descriptor lives.
    """
    for klass in _TYPE_MRO.__get__(cls):
        own = _TYPE_DICT.__get__(klass)
        if name in own:
            return own[name]
    return _MISSING


#: ``AffineDynamics``'s own ``__dict__`` getset descriptor, bound once.
#: :data:`_ROOT_INITIALISED` is the root's bookkeeping about itself, so no
#: subclass may answer for it: one defining ``__dict__`` as a property
#: returning ``{}`` would otherwise hide the marker and silence the check.
#:
#: Only the marker is read this way. The *buffers* are read by ordinary
#: attribute access, because that is how ``F`` resolves them -- see
#: :func:`_root_buffers`.
_AFFINE_INSTANCE_DICT: Any = AffineDynamics.__dict__["__dict__"]


#: The buffers a problem actually reads, obtained the way its own methods
#: obtain them. ``AffineDynamics.F`` is ``return self._M``, so
#: ``getattr(problem, "_M")`` is by construction the object ``F`` hands to the
#: tape -- through a ``__slots__`` descriptor, a property, or a subclass
#: ``__getattribute__`` alike, because ``F`` goes through those too.
#:
#: Two sounder-looking reads were tried and are wrong. ``problem.__dict__``
#: misses a slotted buffer and can be answered by a subclass ``__dict__``
#: property. The root's own ``__dict__`` descriptor fixes the second but not
#: the first, and worse, it reads storage that a property or slot may have
#: displaced: a frozen decoy in the instance dict passed the check while a
#: writeable array reached ``F``.
def _live_buffer_names(problem: object) -> tuple[str, ...]:
    """Which coefficient buffers ``problem`` can still hand to the tape.

    Empty means the root's storage is dead on this instance: the precondition
    failing, not a refusal, and C-16.2 requires it to be *quiet*.

    Two facts answer this, and the second was missing until a review
    reproduced its absence. The first is the recorded
    :data:`_ROOT_INITIALISED` flag: if the root initialiser ran it created all
    three buffers, so all three are live, whatever the subclass has done
    since. It is read through the root's own ``__dict__`` descriptor rather
    than by attribute access, because unlike the buffers it is not something
    any subclass is entitled to answer for.

    The flag alone was the whole precondition, on the reasoning that an
    instance whose root initialiser never ran holds nothing the tape could
    alias. That is true only of a subclass that also *replaces the members
    which hand a buffer back*. One that assigns ``self._M``, ``self._C`` and
    ``self._b`` itself and inherits ``f``, ``F`` and ``G`` runs the root's
    ``return self._M`` over writeable storage, and the check skipped it
    entirely: ``affine_dynamics_verified`` returned ``True`` and the affine
    route returned a gradient displaced by the full ``0.0770625`` of the
    C-15.7 fixture.

    So the second fact is which of those three members the MRO still resolves
    to the root's own definition, and that also says *which* buffers are live
    rather than demanding all three: ``F`` reads only ``_M``, ``G`` only
    ``_C``, and ``f`` reads all three. A subclass overriding ``f`` and ``G``
    but not ``F`` owes ``_M`` and nothing else.

    This is not the MRO test rejected earlier as a *replacement* for the flag.
    That one is unsound in the quiet direction, because an override may
    delegate:

        def F(self, y, u, t): return super().F(y, u, t)

    Three distinct function objects, every buffer read. **Method identity
    cannot witness what a method reads**, and it still cannot. Used as a
    widening rather than as the whole precondition it only ever adds
    instances to the checked set.

    A second widening closes what method identity cannot see. A subclass that
    both declines the root initialiser *and* delegates all three readers to
    the root's has no marker and no reader of its own, and was left with an
    empty live set while `super().F(...)` handed out its writeable ``_M``:
    the full ``0.0770625`` displacement of the C-15.7 fixture, through the
    public optimizer, with every guard quiet. Each half of that shape is
    ordinary on its own -- coefficients built from a file or a mesh,
    instrumentation wrapped round a reader -- so the combination is not
    exotic. So **any root coefficient name that resolves at all is live**,
    which is the same question ``__setstate__`` asks of a restored object,
    asked here for the same reason: the storage is what the tape will alias,
    whatever the class says about who reads it.

    It is resolution, not type, that makes a name live. Restricting it to
    ``ndarray`` values left the same delegating shape holding three mutable
    ``csr_array`` coefficients invisible to both widenings, and it returned
    the same displacement. Whether the value is usable is decided afterwards
    by ``_root_buffers``, which raises rather than skipping; deciding it here
    would be skipping.

    Both widenings are needed, and what identity adds is a name that does not
    resolve at all. A subclass inheriting ``F`` and holding no ``_M`` is
    refused *by name* at the route decision rather than raising
    ``AttributeError`` from inside the first step.

    And the flag remains load-bearing beside both, for the one thing neither
    can do: remember. Each widening asks what the instance currently is; only
    the flag knows that three frozen arrays were established here and that
    two are left. A subclass that delegates its readers and then deletes
    ``_M`` is invisible to identity and to presence, and is refused by name
    only because the marker survives.

    The cost is a shape that is refused although it works: a subclass that
    replaces all three readers and reuses one of the root's buffer names for
    writeable storage of its own. It pays the standing trade of this clause,
    it is told exactly which name collided, and renaming the attribute lifts
    the refusal. Restoration already charged it the same price, and a rule
    that binds on copying but not on checking is not one rule.

    Asking is not a passive read: a coefficient descriptor is executed here,
    and one that raises has its exception absorbed. That direction is safe --
    a name that cannot be read hands nothing to the tape, and if it were
    load-bearing the reader reaching for it during the solve raises from the
    same descriptor, loudly. Unreadable *storage*, as distinct from an
    unreadable name, is a different matter and raises below.

    Once a buffer is live, anything found in its place other than an
    ``ndarray`` is storage whose aliasing cannot be reasoned about -- a sparse
    matrix, a ``memoryview``, a deleted attribute -- and raises rather than
    being skipped.
    """
    if not isinstance(problem, AffineDynamics):
        return ()
    if _AFFINE_INSTANCE_DICT.__get__(problem).get(_ROOT_INITIALISED, False):
        return _COEFFICIENT_BUFFERS
    cls = type(problem)
    live = {
        name
        for reader, read in _ROOT_BUFFER_READERS.items()
        if _defined_as(cls, reader) is _defined_as(AffineDynamics, reader)
        for name in read
    }
    live.update(_resolving_buffer_names(problem))
    return tuple(name for name in _COEFFICIENT_BUFFERS if name in live)


def _root_buffers(problem: object) -> dict[str, NDArray] | None:
    """The live coefficient buffers, or ``None`` if there are none.

    Resolved through :func:`_resolve_buffer`, not by a bare ``getattr``. This
    read was the one place the shared rule was restated rather than reused,
    and a name obliged by an inherited reader whose descriptor raised anything
    but ``AttributeError`` let that exception out of the route decision
    instead of the ``MissingCoefficients`` naming it.
    """
    names = _live_buffer_names(problem)
    if not names:
        return None
    rooted = _AFFINE_INSTANCE_DICT.__get__(problem).get(_ROOT_INITIALISED, False)
    buffers: dict[str, NDArray] = {}
    for name in names:
        buffer = _resolve_buffer(problem, name)
        if not isinstance(buffer, np.ndarray):
            raise MissingCoefficients(
                type(problem),
                name,
                None if buffer is _MISSING else buffer,
                rooted,
            )
        buffers[name] = buffer
    return buffers


def require_immutable_coefficients(problem: object) -> None:
    """Raise unless the coefficient buffers ``problem`` reads are frozen and
    own their storage.

    The invariant exists because ``AffineDynamics.F`` and ``G`` hand back
    ``self._M`` and ``self._C`` *by identity* and the tape retains them;
    ``f`` reads all three to build a fresh array, which is why an unfrozen
    ``_b`` still matters. It is conditional on that storage being live at all,
    which :func:`_root_buffers` answers first.

    The condition is the storage, not which member reads which value *within*
    a buffer. Which buffers are live is answered by :func:`_live_buffer_names`,
    from the recorded flag widened by the readers the MRO still resolves to the
    root's own; the cost is refusing a subclass that keeps the root's storage,
    unfreezes a buffer, and then overrides every member that would have read it
    -- a loud refusal of a working problem, in exchange for never staying quiet
    about a broken one.

    Four conditions, and the last three are the ones that are easy to miss.

    ``writeable = False`` seals the ``ndarray`` it is set on. It does not seal
    anything that array merely carries or points at, and two of these
    conditions exist because it does not.

    An ``ndarray`` **subclass** may hold state of its own that the flag never
    reaches. A ``MaskedArray``'s mask is an ordinary writeable array, and
    flipping one entry after the check changed what the coefficient means
    mid-solve: a deep copy verified as affine with its data frozen and owning,
    and the gradient came back ``0.7706250`` against an exact ``0.6781500`` on
    the C-15.7 fixture at ``C = 2``. There is no general way to enumerate what
    a subclass keeps, so the type must be ``ndarray`` exactly. The certified
    envelope is numeric coefficients, so this refuses nothing C-6 admits.

    ``np.array(b, copy=True).reshape(-1)`` returns a *view*: freezing it leaves
    ``arr.base`` writeable, so the values behind a buffer reported as read-only
    could still be rewritten. Requiring ``owndata`` removes the question rather
    than walking a base chain to answer it.

    An ``object`` dtype makes ``writeable=False`` mean almost nothing. The
    flag covers the array of references; every object behind those references
    stays mutable, and a frozen owning ``object`` array holding a
    zero-dimensional ``ndarray`` gave the full ``0.0770625`` displacement with
    all three guards passing and both flags intact throughout. There is no
    depth at which chasing this ends, so the dtype is refused instead.
    ``AffineDynamics.__init__`` casts to ``float``, so this is reachable only
    on the path that never ran it.

    ``O(1)`` per buffer, once per route decision, against the ``O(n³)`` solve
    it guards. It is a check at a moment, not a lifetime guarantee; C-15.7
    records what that does and does not reach.
    """
    buffers = _root_buffers(problem)
    if buffers is None:
        return
    for name, buffer in buffers.items():
        if type(buffer) is not np.ndarray:
            raise MutableCoefficients(
                type(problem),
                name,
                f"a {type(buffer).__name__}, an ndarray subclass whose own "
                f"state the writeable flag does not reach -- a MaskedArray's "
                f"mask stays writeable and changes what the coefficient "
                f"means",
            )
        if buffer.flags.writeable:
            raise MutableCoefficients(type(problem), name, "writeable")
        if not buffer.flags.owndata:
            raise MutableCoefficients(
                type(problem), name, "non-owning (its base is writeable)"
            )
        if buffer.dtype.hasobject:
            raise MutableCoefficients(
                type(problem),
                name,
                f"of dtype {buffer.dtype!s}, whose elements stay mutable "
                f"however the array is flagged",
            )


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

    Four things are checked, and the order is load-bearing. The **first** is a
    question of validity and is the only one that raises; the three after it
    are questions of eligibility and are answered quietly. The second of those
    three is the one that is easy to omit:

    1. **The coefficients themselves.** Every buffer is present, frozen, and
       owns its storage. This one **raises** :class:`MutableCoefficients`
       instead of returning ``False``, and that is why it runs first rather
       than last. The three below are safe to answer quietly because the
       general route computes the same derivative more slowly, whereas the
       general route reads ``F`` and retains what it returns exactly as the
       affine route does. There is no route that repairs a rewritable
       coefficient, so declining to certify it would choose a route already
       known to be no better — the silent sentinel C-7 forbids. Ordered after
       the others, it was reachable only once they had all passed, so any
       problem that was both mutable *and* ineligible for some unrelated
       reason bypassed it and took the general route silently. C-15.7 measures
       that bypass at ``0.0770625`` on the two-step fixture.
    2. **The class.** Every guaranteed member bound in the MRO is the object
       the root class binds, read by :func:`_defined_as` rather than by
       ``getattr`` so that the descriptor protocol cannot answer for it.
    3. **The instance.** No *invoked* guaranteed member is shadowed in the
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
    4. **Attribute lookup.** ``__getattribute__``, ``__getattr__`` and
       ``__dict__`` are the root's, so the members compared above are the
       members the solver will later receive, and the instance storage read in
       step 3 is the real one. The copy and pickle hooks are compared here too,
       so that a copy of a verified problem is one the constructor could have
       produced — in particular one whose coefficients are still immutable
       (C-15.7).

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
    # Validity, before eligibility. A rewritable coefficient is wrong on every
    # route that retains what ``F`` returns, so unlike the three checks below
    # this one may not be reached only when the problem is otherwise a
    # candidate for the affine route: an earlier quiet refusal would hand the
    # caller the general route, which is displaced by exactly the same amount
    # (C-15.7). It carries its own precondition, so problems that never read
    # the root's buffers pass through it untouched.
    require_immutable_coefficients(problem)

    roots = [
        root for root in _GUARANTEED_MEMBERS if isinstance(problem, root)
    ]
    if len(roots) != 1:
        return False
    root = roots[0]
    cls = type(problem)

    for name in _LOOKUP_HOOKS + _COPY_PROTOCOL_HOOKS:
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
