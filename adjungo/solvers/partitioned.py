"""Partitioned stage solver for separable Hamiltonians (C-8.4).

The step implementation for a :class:`~adjungo.core.partitioned.PartitionedMethod`.
It executes and differentiates one step: forward stages, tangent stages, the
first- and second-order adjoint stages, and the weighted adjoint the gradient
contracts against.

Why this cannot reuse the ordinary route
----------------------------------------

Every other solver in this package advances ``Z_i = U_i y + h Σ_j A_ij f_j``
with one scalar ``A_ij`` multiplying the whole state. Here the multiplier is
``diag(A^q_ij I, A^p_ij I)``, which is not a scalar and not expressible as
``A ⊗ I``.

What separability buys, and what it costs
-----------------------------------------

Within C-8.4's domain ``f^q`` reads only ``p`` and ``f^p`` only ``(q, u, t)``.
The stage system's dependency graph is then bipartite between the blocks, and
for both supported methods it is acyclic -- so the stages resolve by
substitution even though neither ``A^q`` nor ``A^p`` is triangular.

The cost is that the hypothesis becomes load-bearing, and how it is discharged
depends on what the problem is willing to declare.

Two routes to a stage value
---------------------------

**Split (preferred).** A problem declaring ``f_q(p, u, t)`` and ``f_p(q, u, t)``
-- see :class:`~adjungo.core.problem.SeparableProblem` -- is asked for each
half from exactly the arguments that half reads. Nothing is substituted and
nothing is invented. That settles exactly one of the domain's requirements --
that neither half needs the state half the method has not computed -- by not
offering it rather than by checking it. Every other requirement remains a
caller hypothesis or a runtime check.

**Whole-vector (fallback).** A problem declaring only ``f`` needs a whole
state vector, so the step's incoming state supplies the half no stage has
written yet. This route carries an assumption the split route does not: that
``f`` is *evaluable* at a state mixing the incoming half with the stage's own
half, control and time. Separability constrains what ``f`` reads, not where it
is defined, and three valid Hamiltonians were refused by successive attempts
to invent the missing half -- ``NaN`` (an ``f`` built as a matrix product
returns ``NaN`` across the whole row, even through a mathematically absent
dependency), a fixed finite constant (``q(log q - 1)`` needs ``q > 0``), and
the incoming state itself (``(q-t)(log(q-t) - 1)`` moves its domain with
``t``, so a position valid on arrival is invalid at the stage time). Validity
is joint in ``(y, u, t)``, so no held value can be guaranteed admissible at a
stage the problem has not been asked about; the assumption cannot be
discharged from inside this route, which is why the split exists. It holds by
construction for a total ``f`` -- affine, polynomial, trigonometric -- which
is what the shipped problems are.

What is checked
---------------

Neither route takes separability on trust, and neither probes for it --
precedent R-9 is what a probe buys. Two checks run on the values actually
computed with, at every step of every solve:

1. What the stage was built from is compared, **exactly**, against ``f`` at
   the completed stage. On the fallback route that is every consumed value,
   establishing that the block did not read the half that was missing: a
   problem whose ``f^q`` reads ``q`` returns a different number once the real
   ``q`` is known. On the split route it is both halves at every stage,
   establishing that the split and ``f`` are the same function, which matters
   because the step update, the Jacobians and the independent reference all
   go through ``f``. Checking only consumed values made the refusal a
   property of the tableau: symplectic Euler's ``A^q`` row is empty, so it
   consumes no ``f_q`` and accepted an arbitrarily wrong one.
2. ``F`` and ``G`` are checked for the exact block structure the domain
   asserts -- ``F^qq = 0``, ``F^pp = 0``, ``G^q = 0`` -- at the converged
   stages. That structure is what licenses the explicit block slicing in the
   tangent and adjoint sweeps, which is how those avoid needing a poison value
   at all.

Check 1 constrains ``f`` where it is evaluated; check 2 constrains its
derivative there. Neither implies the other, and the derivative sweeps rest on
the second. Together they establish the block dependencies C-8.4 names; they do
not establish that ``f`` derives from a Hamiltonian, which remains the caller's
hypothesis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from adjungo.core.partitioned import P, PartitionedMethod, Q
from adjungo.solvers.base import StageSolver, StepCache

if TYPE_CHECKING:
    from adjungo.core.problem import Problem

__all__ = [
    "PartitionedStageSolver",
    "SeparabilityViolation",
    "partition_of",
    "split_rhs_of",
]


class SeparabilityViolation(ValueError):
    """The problem is outside C-8.4's separable domain.

    Raised from the values a solve actually used, not from a probe. A
    partitioned method applied outside the domain does not fail loudly on its
    own -- it integrates a different problem -- so the domain is checked where
    it is relied upon.
    """


def split_rhs_of(problem: Problem) -> tuple[Callable, Callable] | None:
    """The problem's split right-hand side, or ``None`` if it declares none.

    Returns ``(f_q, f_p)``, which is indexed by the block constants of this
    module because ``Q`` is ``0`` and ``P`` is ``1``. See
    :class:`~adjungo.core.problem.SeparableProblem` for why the split exists.

    Declaring exactly one half is refused rather than half-honoured. It is
    an authoring error with no correct reading: silently ignoring the
    declared one would discard what the author wrote, and silently using it
    would apply the split to one block and the fallback to the other, which
    is neither path's contract.
    """
    f_q = getattr(problem, "f_q", None)
    f_p = getattr(problem, "f_p", None)
    if f_q is None and f_p is None:
        return None
    if f_q is None or f_p is None:
        declared, missing = ("f_p", "f_q") if f_q is None else ("f_q", "f_p")
        raise SeparabilityViolation(
            f"{type(problem).__name__} declares {declared} but not "
            f"{missing}. C-8.4's split right-hand side is declared as a "
            f"pair: f_q(p, u, t) and f_p(q, u, t). Declare both to take the "
            f"split route, or neither to be solved through f alone."
        )
    return f_q, f_p


def partition_of(problem: Problem) -> int:
    """``n_q``, the size of the position block, declared by the problem.

    The partition is a property of the *system*, not of the tableau: the
    coefficients in :mod:`adjungo.core.partitioned` are dimension-free. A
    problem that does not declare one is refused at construction rather than
    given a default, because ``state_dim // 2`` is right for every canonical
    Hamiltonian and wrong in silence for anything else (C-7).
    """
    n_q = getattr(problem, "n_q", None)
    if n_q is None:
        raise NotImplementedError(
            f"a partitioned method needs the problem's canonical split: "
            f"{type(problem).__name__} declares no n_q. Set n_q to the "
            f"number of position components, so that y = (y[:n_q], y[n_q:]) "
            f"is the (q, p) partition of NUMERICS.md C-8.4."
        )
    n_q = int(n_q)
    n = int(problem.state_dim)
    if not 0 < n_q < n:
        raise ValueError(
            f"n_q must lie strictly inside the state: got n_q={n_q} for "
            f"state_dim={n}. C-8.4 partitions into two nonempty blocks."
        )
    return n_q


class PartitionedStageSolver(StageSolver["PartitionedMethod"]):
    """Substitution in the partitioned dependency order (C-8.4).

    The order is a property of the tableau and is computed once, at method
    construction; an implicit partitioned stage system is refused there rather
    than here, so this class only ever sees a system it can solve.
    """

    def __init__(self, n_q: int) -> None:
        self.n_q = int(n_q)

    # -- block helpers ------------------------------------------------------

    def _slices(self, n: int) -> tuple[slice, slice]:
        return slice(0, self.n_q), slice(self.n_q, n)

    @staticmethod
    def _array(method: PartitionedMethod, block: int) -> NDArray:
        return method.A_q if block == Q else method.A_p

    # -- forward ------------------------------------------------------------

    def solve_stages(
        self,
        y_history: NDArray,
        u_stages: NDArray,
        t_n: float,
        h: float,
        problem: Problem,
        method: PartitionedMethod,
        step: int | None = None,
    ) -> tuple[NDArray, StepCache]:
        """Fill the ``2s`` stage half-vectors in dependency order."""
        s, n = method.s, int(problem.state_dim)
        sl = self._slices(n)
        y = y_history[0]
        t_stage = t_n + method.c * h

        # NaN marks an entry no node has written yet. No callback is ever
        # shown one: the split route passes only written halves, and the
        # whole-vector route replaces the unwritten half in `_consume`. The
        # NaN is here so that an unwritten entry escaping into the returned
        # array is visible rather than plausible (C-7).
        Z = np.full((s, n), np.nan)
        written: set[tuple[int, int]] = set()
        consumed: list[tuple[int, int, NDArray]] = []
        split = split_rhs_of(problem)

        for block, i in method.dependency_order:
            A = self._array(method, block)
            acc = np.array(y[sl[block]], dtype=float)
            for j in range(s):
                if A[i, j] == 0.0:  # exact, as everywhere structure routes
                    continue
                f_j = self._consume(
                    Z, written, y, split, problem, u_stages[j], t_stage[j],
                    sl, block, j,
                )
                if split is None:
                    consumed.append((block, j, f_j.copy()))
                acc = acc + h * A[i, j] * f_j
            Z[i, sl[block]] = acc
            written.add((block, i))

        f_complete = [
            np.asarray(
                problem.f(Z[i], u_stages[i], t_stage[i]), dtype=float
            )
            for i in range(s)
        ]
        if split is None:
            self._check_consumed_values_match_the_whole(
                consumed, f_complete, sl, step
            )
        else:
            self._check_the_split_is_the_same_function_as_f(
                split, Z, f_complete, u_stages, t_stage, sl, step
            )

        F_list = [
            np.asarray(problem.F(Z[i], u_stages[i], t_stage[i]), dtype=float)
            for i in range(s)
        ]
        G_list = [
            np.asarray(problem.G(Z[i], u_stages[i], t_stage[i]), dtype=float)
            for i in range(s)
        ]
        self._check_block_structure(F_list, G_list, sl, step)

        return Z, StepCache(Z=Z, F=F_list, G=G_list)

    def _consume(
        self,
        Z: NDArray,
        written: set[tuple[int, int]],
        y: NDArray,
        split: tuple[Callable, Callable] | None,
        problem: Problem,
        u_stage: NDArray,
        t: float,
        sl: tuple[slice, slice],
        block: int,
        j: int,
    ) -> NDArray:
        """``f^block`` at stage ``j``, before every half of that stage is known.

        The half this block *reads* is always written already -- that is
        what ``dependency_order`` establishes, and
        ``test_the_half_a_consumed_block_needs_is_always_written_already``
        asserts it on the tableaux. Only the block's own half can still be
        missing, and inside C-8.4's domain the block does not read it.

        **Split route.** When the problem declares ``f_q``/``f_p``, the
        conjugate half is all that is passed and nothing is invented. The
        two halves are held to ``f`` afterwards, by
        ``_check_the_split_is_the_same_function_as_f``.

        **Whole-vector route.** Otherwise ``f`` needs a whole state vector,
        and the step's incoming state `y` supplies the missing half. This
        path carries an assumption C-8.4 states explicitly: that ``f`` is
        evaluable at a state mixing the incoming half with this stage's
        half, control and time. Three valid Hamiltonians have been refused
        by successive attempts to invent that half -- ``NaN`` (an ``f``
        built as a matrix product returns ``NaN`` for the whole row), a
        fixed finite constant (``log q`` needs ``q > 0``), and the incoming
        state itself (``log(q - t)`` moves its domain with ``t``, so a
        position valid on arrival is invalid at the stage time). The
        assumption cannot be discharged from within this route, which is
        why the split exists.
        """
        other = 1 - block
        if (other, j) not in written:
            raise SeparabilityViolation(
                f"internal: f^{'q' if block == Q else 'p'} at stage {j} was "
                f"consumed before the half it reads was written. The "
                f"dependency order is wrong for this tableau; see C-8.4."
            )
        if split is not None:
            return np.asarray(
                split[block](Z[j][sl[other]], u_stage, t), dtype=float
            )

        substituted = np.array(Z[j], dtype=float)
        if (block, j) not in written:
            substituted[sl[block]] = y[sl[block]]
        return np.asarray(
            problem.f(substituted, u_stage, t), dtype=float
        )[sl[block]]

    @staticmethod
    def _check_consumed_values_match_the_whole(
        consumed: list[tuple[int, int, NDArray]],
        f_complete: list[NDArray],
        sl: tuple[slice, slice],
        step: int | None,
    ) -> None:
        """Every ``f`` value used mid-substitution must survive completion.

        The whole-vector route's half of check (1). Establishes that the
        block did not read the half that was missing when it was evaluated:
        a problem whose ``f^q`` reads ``q`` returns a different number once
        the real ``q`` is known.

        Exact equality, not a tolerance. Under separability the two
        evaluations are the same arithmetic on the same inputs, so they agree
        bit for bit; a tolerance here would admit a problem that genuinely
        depends on the missing half by a small amount, which is a different
        discrete problem rather than a rounding difference.
        """
        for block, j, used in consumed:
            final = f_complete[j][sl[block]]
            if np.array_equal(used, final):
                continue
            name = "q" if block == Q else "p"
            where = "" if step is None else f" at step {step}"
            raise SeparabilityViolation(
                f"f^{name} was evaluated at stage {j}{where} before the "
                f"whole stage vector was known, and changed once it was: "
                f"{used} became {final}. A partitioned method advances the "
                f"blocks in turn, which is valid only inside C-8.4's "
                f"separable domain, where f^q depends on p alone and f^p on "
                f"(q, u, t) alone. This problem is outside it; use an "
                f"unpartitioned method, or amend C-8.4."
            )

    @staticmethod
    def _check_the_split_is_the_same_function_as_f(
        split: tuple[Callable, Callable],
        Z: NDArray,
        f_complete: list[NDArray],
        u_stages: NDArray,
        t_stage: NDArray,
        sl: tuple[slice, slice],
        step: int | None,
    ) -> None:
        """``f_q`` and ``f_p`` must equal ``f``, at every completed stage.

        The split route's half of check (1). The stage is built from the
        split, while the step update, the Jacobians and the independent
        reference all go through ``f``; a split that drifted from ``f``
        would make the forward step and its derivatives describe different
        problems, with nothing else positioned to notice.

        **Both halves at every stage, not only the ones the tableau
        consumed.** Symplectic Euler's ``A^q`` row is empty, so it never
        evaluates ``f_q`` while resolving its stage; checking only consumed
        values would accept an arbitrarily wrong ``f_q`` under that method
        and refuse it under Verlet. No wrong answer follows -- an unconsumed
        value is by definition one the stage was not built from -- but the
        author would learn of the error only on changing method, which is
        the opposite of where a reference implementation should place the
        cost (C-1).

        Evaluating at the completed stage rather than at the moment of
        consumption loses nothing: each half of ``Z`` is written once and
        never revised, so the argument a consumed call received is the same
        array this one passes.
        """
        for i in range(len(f_complete)):
            for block, name in ((Q, "q"), (P, "p")):
                got = np.asarray(
                    split[block](Z[i][sl[1 - block]], u_stages[i], t_stage[i]),
                    dtype=float,
                )
                final = f_complete[i][sl[block]]
                if np.array_equal(got, final):
                    continue
                where = "" if step is None else f" at step {step}"
                raise SeparabilityViolation(
                    f"f_{name} disagrees with f at stage {i}{where}: the "
                    f"split returned {got} where f returned {final}. "
                    f"C-8.4's split route builds the stage from f_{name} "
                    f"while the step update, the Jacobians and the "
                    f"reference all use f, so the two must be the same "
                    f"function. Fix the split, or withdraw it and be solved "
                    f"through f alone."
                )

    @staticmethod
    def _check_block_structure(
        F_list: list[NDArray],
        G_list: list[NDArray],
        sl: tuple[slice, slice],
        step: int | None,
    ) -> None:
        """``F`` block anti-diagonal and ``G^q = 0``, exactly.

        This is the structure the tangent and adjoint sweeps slice against.
        Exactness follows C-8.3's treatment of structure everywhere else: a
        block that is nearly zero belongs to a different problem from one that
        is zero, and the sweeps would silently drop its contribution.
        """
        q, p = sl
        checks = (
            ("F^qq = ∂f^q/∂q", lambda M: M[q, q]),
            ("F^pp = ∂f^p/∂p", lambda M: M[p, p]),
        )
        where = "" if step is None else f" at step {step}"
        for i, F in enumerate(F_list):
            for label, take in checks:
                block = take(F)
                if np.any(block != 0.0):
                    raise SeparabilityViolation(
                        f"{label} is nonzero at stage {i}{where}:\n{block}\n"
                        f"C-8.4's domain is a separable Hamiltonian "
                        f"H = T(p) + V(q, u, t), whose state Jacobian is "
                        f"block anti-diagonal. The partitioned tangent and "
                        f"adjoint sweeps drop this block by construction, so "
                        f"a nonzero one is not integrated -- it is discarded."
                    )
        for i, G in enumerate(G_list):
            if np.any(G[q, :] != 0.0):
                raise SeparabilityViolation(
                    f"G^q = ∂f^q/∂u is nonzero at stage {i}{where}:\n"
                    f"{G[q, :]}\nUnder C-8.4 the control enters through "
                    f"V(q, u, t) only, so f^q = ∂T/∂p carries no control "
                    f"dependence. The partitioned sweeps drop this block."
                )

    # -- first-order adjoint ------------------------------------------------

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: PartitionedMethod,
        h: float,
    ) -> NDArray:
        """Substitution in the **reverse** dependency order.

        The adjoint stage relation is the C-8.4 substitution of
        ``A[i,j] → diag(A^q_ij I, A^p_ij I)`` into the ordinary one::

            μ_i = h F_i^T Λ_i,
            Λ_i^q = Σ_j A^q_ji μ_j^q + Σ_l B[l,i] λ_l^q

        and likewise for ``p``. The coupling arrays enter transposed in the
        stage index, so the adjoint's dependency graph is the forward graph
        with every edge reversed, and ``reversed(dependency_order)`` solves
        it. That is a statement about the graph, not an assumption: it is
        asserted directly in ``tests/test_partitioned_methods.py``.

        ``F_i^T`` is block anti-diagonal, so ``μ_i^q`` reads ``Λ_i^p`` alone.
        The two halves are sliced out rather than multiplied against a
        placeholder, which is why this sweep needs no poison value.
        """
        s = method.s
        n = cache.Z.shape[1]
        q, p = self._slices(n)
        other = {Q: p, P: q}
        own = {Q: q, P: p}

        mu = np.zeros((s, n))
        for block, i in reversed(method.dependency_order):
            lam = self._weighted_adjoint_block(
                mu, lambda_ext, method, i, 1 - block
            )
            # (F^T Λ)_own = F[other rows, own cols]^T Λ_other.
            mu[i, own[block]] = h * (
                cache.F[i][other[block], own[block]].T @ lam
            )
        return mu

    def _weighted_adjoint_block(
        self,
        mu: NDArray,
        lambda_ext: NDArray,
        method: PartitionedMethod,
        i: int,
        block: int,
    ) -> NDArray:
        """``Λ_i^block = Σ_j A^block_ji μ_j^block + Σ_l B[l,i] λ_l^block``.

        One block only, because the adjoint sweep needs the opposite half of
        ``Λ_i`` before this half exists.
        """
        n = mu.shape[1]
        sl = self._slices(n)[block]
        A = self._array(method, block)
        weighted: NDArray = A[:, i] @ mu[:, sl] + method.B[:, i] @ lambda_ext[:, sl]
        return weighted

    def weighted_adjoint(
        self,
        mu: NDArray,
        lambda_ext: NDArray,
        method: PartitionedMethod,
    ) -> NDArray:
        """``Λ_k``, both halves, with each weighted by its own array.

        The gradient contracts ``G_k`` against this, and the second-order
        sweep contracts the problem's curvature callbacks against it, so a
        single wrong weight here reaches every derivative the step produces.
        """
        s, n = method.s, mu.shape[1]
        out = np.empty((s, n))
        for block in (Q, P):
            sl = self._slices(n)[block]
            for k in range(s):
                out[k, sl] = self._weighted_adjoint_block(
                    mu, lambda_ext, method, k, block
                )
        return out

    # -- tangent ------------------------------------------------------------

    def solve_tangent_stages(
        self,
        delta_y_history: NDArray,
        du_stages: NDArray,
        cache: StepCache,
        method: PartitionedMethod,
        h: float,
        step: int,
    ) -> NDArray:
        """Differentiate the forward substitution, in the same order.

        ``δZ_i^q = δy^q + h Σ_j A^q_ij F_j[q, p] δZ_j^p``: the control term
        drops because ``G^q = 0``, and the ``q`` column of ``F_j``'s ``q``
        rows is zero. Both were checked on these very ``F_j`` and ``G_j``.
        """
        s = method.s
        n = cache.Z.shape[1]
        q, p = self._slices(n)
        other = {Q: p, P: q}
        own = {Q: q, P: p}

        dZ = np.zeros((s, n))
        dy = delta_y_history[0]
        for block, i in method.dependency_order:
            A = self._array(method, block)
            acc = np.array(dy[own[block]], dtype=float)
            for j in range(s):
                if A[i, j] == 0.0:
                    continue
                contribution = (
                    cache.F[j][own[block], other[block]] @ dZ[j, other[block]]
                )
                if block == P:
                    contribution = contribution + (
                        cache.G[j][own[block], :] @ du_stages[j]
                    )
                acc = acc + h * A[i, j] * contribution
            dZ[i, own[block]] = acc
        return dZ

    # -- second-order adjoint ----------------------------------------------

    def solve_adjoint_sensitivity_stages(
        self,
        delta_lambda_ext: NDArray,
        gamma: NDArray,
        cache: StepCache,
        method: PartitionedMethod,
        h: float,
    ) -> NDArray:
        """The first-order adjoint system with ``Γ`` added to each side.

        Same operator, same order; only the right-hand side differs. ``Γ_i``
        carries both halves -- it contracts the problem's curvature against
        the weighted adjoint and is not itself partitioned -- so each half of
        ``δμ_i`` takes its own half of ``Γ_i``.
        """
        s = method.s
        n = cache.Z.shape[1]
        q, p = self._slices(n)
        other = {Q: p, P: q}
        own = {Q: q, P: p}

        dmu = np.zeros((s, n))
        for block, i in reversed(method.dependency_order):
            dlam = self._weighted_adjoint_block(
                dmu, delta_lambda_ext, method, i, 1 - block
            )
            dmu[i, own[block]] = (
                h * (cache.F[i][other[block], own[block]].T @ dlam)
                + gamma[i, own[block]]
            )
        return dmu
