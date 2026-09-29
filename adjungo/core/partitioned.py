"""Partitioned Runge-Kutta methods on a canonical ``(q, p)`` split.

The convention is NUMERICS.md C-8.4. A partitioned method applies a *different*
coefficient array to each half of the state over a **common** stage index,
with common weights ``b`` and a single abscissa vector ``c``::

    Z_i^q = y_n^q + h Σ_j A^q_ij f^q(Z_j, u_j, t_n + c_j h)
    Z_i^p = y_n^p + h Σ_j A^p_ij f^p(Z_j, u_j, t_n + c_j h)
    y_{n+1} = y_n + h Σ_i b_i f(Z_i, u_i, t_n + c_i h)

Only the coefficients are partitioned. Each ``f`` argument is the whole stage
vector, and the output update is unpartitioned -- one ``b`` for both blocks --
so ``U``, ``B`` and ``V`` are the ordinary ``r = 1`` Runge-Kutta ones and the
external-stage propagation needs nothing new.

Exact coefficients
------------------

The tableaux are written as :class:`fractions.Fraction` and converted once.
Every coefficient of both methods is a dyadic rational, so the conversion is
exact and the stored floats satisfy the structural identities to the last bit
rather than to a tolerance. That is a property of these two tableaux and is
not assumed of a composition; C-14.5 requires the structural checks to be made
symbolically on the exact data, where they hold for a reason, and C-11.3
governs any check made on the stored floats instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from adjungo.core.method import StageType

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "SYMPLECTIC_EULER_EXACT",
    "VERLET_EXACT",
    "Block",
    "PartitionedMethod",
    "PartitionedTableauError",
    "conjugate",
    "paired_symplectic_residual",
    "symplectic_euler",
    "verlet",
]

#: Index of the ``q`` block in a ``(block, stage)`` pair.
Q = 0
#: Index of the ``p`` block.
P = 1
#: A ``(block, stage)`` pair naming one half of one stage vector.
Block = tuple[int, int]

_BLOCK_NAME = ("q", "p")


class PartitionedTableauError(ValueError):
    """A partitioned tableau is malformed or outside the supported structure.

    Raised at construction, never deferred to a solve, so that a method that
    cannot be executed is never handed to a caller as though it could be
    (C-6.2).
    """


def _exact_array(rows: Sequence[Sequence[Fraction]] | Sequence[Fraction]) -> NDArray:
    """Convert exact rationals to float64, refusing any that does not fit.

    A coefficient that is not a dyadic rational would be rounded here, and the
    structural identities would then hold only to a tolerance. Both supported
    tableaux are dyadic, so the refusal never fires for them; it fires for a
    tableau whose author assumed it would.
    """
    arr = np.array(rows, dtype=float)
    flat = np.ravel(np.array(rows, dtype=object))
    for value, stored in zip(flat, arr.ravel(), strict=True):
        if Fraction(stored) != Fraction(value):
            raise PartitionedTableauError(
                f"coefficient {value} is not exactly representable in "
                f"float64 (stored {stored!r}). The C-14.5 structural checks "
                f"are made on the exact coefficients; a tableau that rounds "
                f"here needs a C-11.3 rounding budget instead."
            )
    return arr


def paired_symplectic_residual(
    A_q: NDArray, A_p: NDArray, b: NDArray
) -> NDArray:
    """``D A^p + (A^q)ᵀ D − b bᵀ``, which vanishes for a symplectic pair.

    Entrywise this is ``b_i A^p_ij + b_j A^q_ji − b_i b_j`` (C-8.4). It is
    **not** the unpartitioned condition applied twice: it couples the two
    arrays, so each is a statement about the other, and neither array alone
    satisfies anything.
    """
    D = np.diag(b)
    residual: NDArray = D @ A_p + A_q.T @ D - np.outer(b, b)
    return residual


def conjugate(A: NDArray, b: NDArray) -> NDArray:
    """``C(A) = 𝟙 bᵀ − D⁻¹ Aᵀ D``, entrywise ``b_j − (b_j / b_i) A_ji``.

    Defined only for nonzero weights, as C-8.4 states. For a symplectic pair
    conjugation **exchanges the members**: ``C(A^q) = A^p`` and
    ``C(A^p) = A^q``. The familiar unpartitioned statement that a symplectic
    tableau is its own conjugate is the degenerate case ``A^q = A^p``, and
    does not generalise without the exchange made explicit.
    """
    if np.any(b == 0.0):
        raise ValueError(
            "the conjugate C(A) = 𝟙 bᵀ − D⁻¹ Aᵀ D requires nonzero weights; "
            f"got b = {b}. C-8.4 states the hypothesis and C-14.5 applies "
            "the check only where it holds."
        )
    # C(A)_ij = b_j − (b_j / b_i) A_ji.
    conj: NDArray = np.outer(np.ones_like(b), b) - (A.T * b[None, :]) / b[:, None]
    return conj


@dataclass(frozen=True)
class PartitionedMethod:
    """A partitioned Runge-Kutta tableau over a canonical ``(q, p)`` split.

    The surface ``U``, ``B``, ``V``, ``c``, ``s``, ``r`` matches
    :class:`~adjungo.core.method.GLMethod`, so external-stage propagation,
    stage-time computation and the discretization plan treat both alike.
    There is deliberately **no** ``A``: no single array describes the stage
    coupling, and supplying one -- either member of the pair, or their mean --
    would be a value that looks like an answer (C-7).

    The partition itself belongs to the *problem*, not to the tableau: these
    coefficients are dimension-free and are checked against
    ``problem.n_q`` when a solve is set up.
    """

    A_q: NDArray
    A_p: NDArray
    b: NDArray
    c: NDArray
    name: str = "partitioned"

    #: Derived in ``__post_init__`` from ``b``, never supplied. They exist so
    #: that the plan, the stepping sweeps and the reference can read one
    #: surface for both method kinds. ``U``, ``V`` and ``B`` are *not*
    #: partitioned: C-8.4 shares the weights and the external stage between
    #: the blocks, which is why external-stage propagation needed no change.
    U: NDArray = field(init=False, repr=False, compare=False)
    B: NDArray = field(init=False, repr=False, compare=False)
    V: NDArray = field(init=False, repr=False, compare=False)

    #: Arrays a plan must copy and freeze. Named on the class so that
    #: ``_frozen_tableau`` needs no knowledge of which method type it holds.
    coefficient_names: tuple[str, ...] = field(
        default=("A_q", "A_p", "b", "c", "U", "B", "V"),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for attr in ("A_q", "A_p", "b", "c"):
            value = getattr(self, attr)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"PartitionedMethod.{attr} must be a numpy array, got "
                    f"{type(value).__name__}."
                )

        for attr in ("A_q", "A_p"):
            A = getattr(self, attr)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise PartitionedTableauError(
                    f"PartitionedMethod.{attr} must be square (s, s); got "
                    f"shape {A.shape}."
                )
        if self.A_q.shape != self.A_p.shape:
            raise PartitionedTableauError(
                f"A_q and A_p share one stage index, so they must have the "
                f"same shape; got {self.A_q.shape} and {self.A_p.shape} "
                f"(NUMERICS.md C-8.4)."
            )

        s = self.A_q.shape[0]
        for attr, shape in (("b", (s,)), ("c", (s,))):
            actual = getattr(self, attr).shape
            if actual != shape:
                raise PartitionedTableauError(
                    f"PartitionedMethod.{attr} must have shape {shape} for "
                    f"s={s}; got {actual}. The weights and abscissae are "
                    f"common to both partitions (NUMERICS.md C-8.4)."
                )

        for attr in ("A_q", "A_p", "b", "c"):
            if not np.all(np.isfinite(getattr(self, attr))):
                raise PartitionedTableauError(
                    f"PartitionedMethod.{attr} has a non-finite entry: "
                    f"{getattr(self, attr)}."
                )

        object.__setattr__(self, "U", np.ones((s, 1)))
        object.__setattr__(self, "B", self.b.reshape(1, s).copy())
        object.__setattr__(self, "V", np.eye(1))

        # Raises when the stage system is not solvable by substitution, which
        # is the supported structure of this milestone (C-6.2).
        _ = self.dependency_order

    # -- the GLMethod surface the generic stepping code uses ---------------

    @cached_property
    def s(self) -> int:
        """Number of internal stages."""
        return int(self.A_q.shape[0])

    @cached_property
    def r(self) -> int:
        """Number of external stages. Always one: C-8.4 is an ``r = 1``
        family, and multistep history is C-Q4's open question."""
        return 1

    @cached_property
    def stage_type(self) -> StageType:
        """Always :attr:`StageType.PARTITIONED`.

        Not a classification of structure, as it is for ``GLMethod``, but a
        statement that no single stage matrix exists. Both supported methods
        happen to be solvable by substitution -- see
        :attr:`dependency_order` -- but that is a property of the *pair*, not
        something either array shows, so it does not make them ``EXPLICIT``
        and must not route them there.
        """
        return StageType.PARTITIONED

    # -- structural properties ---------------------------------------------

    @cached_property
    def sigma_q(self) -> NDArray:
        """Row sums of ``A^q``. **Not** the abscissae; see C-8.4."""
        return self.A_q.sum(axis=1)

    @cached_property
    def sigma_p(self) -> NDArray:
        """Row sums of ``A^p``. **Not** the abscissae; see C-8.4."""
        return self.A_p.sum(axis=1)

    def paired_symplectic_residual(self) -> NDArray:
        """``D A^p + (A^q)ᵀ D − b bᵀ`` on the **stored** coefficients.

        A check on stored floats carries a C-11.3 rounding budget and is a
        different measurement from the symbolic one C-14.5 requires. Both are
        made; neither substitutes for the other.
        """
        return paired_symplectic_residual(self.A_q, self.A_p, self.b)

    @cached_property
    def dependency_order(self) -> tuple[Block, ...]:
        """An order in which the ``2s`` stage half-vectors become computable.

        Within the separable domain of C-8.4, ``f^q`` reads only ``p`` and
        ``f^p`` only ``(q, u, t)``. So ``Z_i^q`` needs ``Z_j^p`` exactly where
        ``A^q_ij ≠ 0``, and ``Z_i^p`` needs ``Z_j^q`` exactly where
        ``A^p_ij ≠ 0``. The dependency graph is bipartite between the blocks,
        and separability is what makes it sparser than the ``s × s`` coupling
        either array shows on its own.

        Both supported methods are acyclic here and therefore **explicit**,
        which is not visible from either array alone: symplectic Euler's
        ``A^p = [[1]]`` has a nonzero diagonal, and Verlet's ``A^q`` is not
        lower triangular. Neither observation is about the method's
        explicitness, because the diagonal of ``A^p`` couples ``Z_i^p`` to
        ``Z_i^q``, never to itself.

        Zeros are exact, as everywhere a tableau's structure decides a route
        (C-8.3): a coefficient near zero is a different method from one at
        zero, and rounding it away would integrate the method nobody wrote.
        """
        s = self.s
        needs: dict[Block, set[Block]] = {}
        for i in range(s):
            needs[(Q, i)] = {(P, j) for j in range(s) if self.A_q[i, j] != 0.0}
            needs[(P, i)] = {(Q, j) for j in range(s) if self.A_p[i, j] != 0.0}

        order: list[Block] = []
        done: set[Block] = set()
        remaining = set(needs)
        while remaining:
            ready = sorted(k for k in remaining if needs[k] <= done)
            if not ready:
                stuck = ", ".join(
                    f"Z_{i}^{_BLOCK_NAME[blk]}"
                    for blk, i in sorted(remaining)
                )
                raise NotImplementedError(
                    f"the partitioned stage system of '{self.name}' is "
                    f"implicit: no order computes {stuck} by substitution. "
                    f"This milestone implements explicit partitioned methods "
                    f"only; an implicit partitioned stage solve is outside "
                    f"NUMERICS.md C-8.4's delivered scope and needs its own "
                    f"certification, not merely another tableau."
                )
            order.extend(ready)
            done.update(ready)
            remaining -= set(ready)
        return tuple(order)


# --- The two supported tableaux, exactly -----------------------------------

_F = Fraction

#: Symplectic Euler, ``s = 1``. ``c = (0)`` while ``σ^q = 0`` and ``σ^p = 1``:
#: an abscissa is not a row sum (C-8.4).
SYMPLECTIC_EULER_EXACT = {
    "A_q": ((_F(0),),),
    "A_p": ((_F(1),),),
    "b": (_F(1),),
    "c": (_F(0),),
}

#: Störmer-Verlet as Lobatto IIIA-IIIB, ``s = 2``. ``c = (0, 1)`` while
#: ``σ^q = (0, 1)`` and ``σ^p = (1/2, 1/2)``. A convention requiring the two
#: partitions to share row sums would exclude this method (C-8.4).
VERLET_EXACT = {
    "A_q": ((_F(0), _F(0)), (_F(1, 2), _F(1, 2))),
    "A_p": ((_F(1, 2), _F(0)), (_F(1, 2), _F(0))),
    "b": (_F(1, 2), _F(1, 2)),
    "c": (_F(0), _F(1)),
}


def _build(exact: dict, name: str) -> PartitionedMethod:
    return PartitionedMethod(
        A_q=_exact_array(exact["A_q"]),
        A_p=_exact_array(exact["A_p"]),
        b=_exact_array(exact["b"]),
        c=_exact_array(exact["c"]),
        name=name,
    )


def symplectic_euler() -> PartitionedMethod:
    """Symplectic Euler: first order, ``s = 1``.

    ``Z^q = y^q`` and ``Z^p = y^p + h f^p(Z)``, so the position is sampled
    before the momentum is advanced. The pair is symplectic; neither array is
    symplectic on its own.
    """
    return _build(SYMPLECTIC_EULER_EXACT, "symplectic_euler")


def verlet() -> PartitionedMethod:
    """Störmer-Verlet as Lobatto IIIA-IIIB: second order, ``s = 2``.

    ``A^q`` is not lower triangular, so the method looks implicit as an
    ordinary tableau and is not. Within the separable domain the stage system
    resolves in the order ``Z_0^q``, ``Z_0^p``, ``Z_1^p``, ``Z_1^q``; see
    :attr:`PartitionedMethod.dependency_order`.
    """
    return _build(VERLET_EXACT, "verlet")
