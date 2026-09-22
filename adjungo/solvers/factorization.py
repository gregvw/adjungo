"""Declaration-gated reuse of LU factorizations.

An implicit stage solve factors ``I - h a_ii F`` at cost ``O(n^3)``. When the
state Jacobian ``F`` does not vary, that matrix is the same at every stage of
every step, and the factorization can be computed once.

Two things make this dangerous, and this module is shaped by both.

**It must not be decided by a runtime probe.** Precedent R-9 records what
happened when it was. An earlier :class:`SDIRKStageSolver` evaluated ``F`` at
``(y_history[0], u_i, t_i)`` for each stage and, on agreement, shared one
factorization. The probe varied ``u`` and ``t`` but never the state, so a
Jacobian depending on ``y`` alone passed it and the adjoint then solved with
``(I - h g F_0)^T`` at every stage. For ``f = y^2 + u`` at ``N = 4`` the
relative gradient error against the monolithic reference was 4.24e-05, seven
orders above the certified tolerance, while the whole suite still passed.

**A declaration alone is a promise, not a fact.** ``ProblemStructure`` is
supplied by the caller. If ``jacobian_constant=True`` is declared for a problem
whose Jacobian is not constant, naive reuse reproduces R-9 exactly, with the
same silence.

The resolution is that the declaration selects the *route* and an exact
comparison establishes the *fact*. Before any stored factorization is returned,
the matrix it was taken from is compared, element for element, with the matrix
the caller is asking to factor now. Reuse is returned only when they are
identical; otherwise the declaration is false and
:class:`DeclaredStructureViolation` is raised.

This is not a probe. A probe asks what ``F`` would be somewhere the computation
does not go, and generalizes. This compares the two matrices that the
computation actually uses, and generalizes nothing: if they are equal, then
reusing the factorization is not an approximation of refactoring, it *is*
refactoring, to the last bit.

The comparison is exact, with no tolerance. That is not a tolerance choice
avoided by strictness; it is the actual condition. If two stage matrices differ
by one unit in the last place, then factoring the second is a different
computation from reusing the first, and the contract (C-2, C-3) is about the
derivative of the discrete map as implemented. Clause C-12 forbids inventing a
tolerance where the governing condition is exact.

Cost: the matrix must still be formed, at ``O(n^2)``, to be compared. Only the
``O(n^3)`` factorization is saved. The comparison is asymptotically cheaper
than the work it guards, so verification never costs more than it protects.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

__all__ = ["DeclaredStructureViolation", "FactorizationStore"]


class DeclaredStructureViolation(RuntimeError):
    """A declared :class:`ProblemStructure` is contradicted by the problem.

    Raised when ``jacobian_constant=True`` was declared and two stage matrices
    that the declaration says are identical are observed not to be.

    This is a hard guard with no override. The alternative to raising is to
    return a derivative computed with the transpose of a matrix the forward
    solve did not use, which is wrong by an amount that shrinks with ``h`` and
    is therefore readily mistaken for discretization error (C-3 forbids that
    reading). Declaring structure the problem does not have is a caller error,
    and the caller can always correct it by declaring
    ``jacobian_constant=False``, which costs accuracy nothing.
    """

    def __init__(self, key: Hashable, largest_difference: float) -> None:
        super().__init__(
            f"ProblemStructure declared jacobian_constant=True, but the stage "
            f"matrix for {key!r} differs between uses by {largest_difference:.3e} "
            f"in the largest element. The declaration is false for this "
            f"problem. Reusing the factorization would make the adjoint solve "
            f"with the transpose of a matrix the forward solve did not use "
            f"(NUMERICS.md R-9). Declare jacobian_constant=False; the "
            f"derivatives will be identical and only the factorization count "
            f"will rise."
        )
        self.key = key
        self.largest_difference = largest_difference


class FactorizationStore:
    """Owns every LU factorization taken during a solve, and counts them.

    One store is held by a stage solver for its whole lifetime, so reuse spans
    stages *and* steps *and* repeated ``gradient``/``hessian_vector_product``
    calls: if the Jacobian is constant, none of those vary the stage matrix.

    The counters are not diagnostics. `NUMERICS.md` C-15 makes the observed
    factorization count a certified quantity with a predicted value, because a
    reuse optimization that silently stops reusing is invisible to every
    accuracy test in the suite -- the answers stay exactly right and only the
    cost changes. ``tests/test_factorization_reuse.py`` asserts the count.
    """

    def __init__(self, *, reuse_enabled: bool) -> None:
        #: Whether the caller declared a constant Jacobian. When false the
        #: store is a pass-through that still counts.
        self.reuse_enabled = reuse_enabled
        self._entries: dict[Hashable, tuple[NDArray, Any]] = {}
        #: Number of ``lu_factor`` calls actually made.
        self.factorizations = 0
        #: Number of times a stored factorization was returned instead.
        self.reuses = 0

    def reset_counts(self) -> None:
        """Zero the counters, keeping the stored factorizations.

        Used to measure a single sweep without discarding valid reuse.
        """
        self.factorizations = 0
        self.reuses = 0

    def factor(self, key: Hashable, matrix: NDArray) -> Any:
        """Return an LU factorization of ``matrix``, reusing when sound.

        Args:
            key: Lookup hint identifying which stage matrix this is, normally
                ``(diagonal coefficient, h)``. The key is *only* a hint. It is
                never trusted: a stored entry is returned only after its
                matrix compares equal to ``matrix``. A key that is too coarse
                therefore causes a refusal, and a key that is too fine causes
                a redundant factorization. Neither can cause a wrong answer.
            matrix: The matrix to factor, exactly as it would be factored
                without this store.

        Raises:
            DeclaredStructureViolation: If reuse is enabled and a stored
                matrix for ``key`` differs from ``matrix``.
        """
        if not self.reuse_enabled:
            self.factorizations += 1
            return scipy.linalg.lu_factor(matrix)

        stored = self._entries.get(key)
        if stored is not None:
            reference, lu = stored
            if reference.shape != matrix.shape or not np.array_equal(
                reference, matrix
            ):
                difference = (
                    float(np.max(np.abs(reference - matrix)))
                    if reference.shape == matrix.shape
                    else float("inf")
                )
                raise DeclaredStructureViolation(key, difference)
            self.reuses += 1
            return lu

        self.factorizations += 1
        lu = scipy.linalg.lu_factor(matrix)
        # The reference is copied. The callers build stage matrices in
        # scratch buffers, and a stored view would be mutated under us,
        # after which every later comparison would trivially succeed and the
        # guard would be silently disabled.
        self._entries[key] = (np.array(matrix, copy=True), lu)
        return lu
