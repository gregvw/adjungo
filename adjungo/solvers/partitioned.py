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

The cost is that the hypothesis becomes load-bearing. Substitution evaluates
``f`` at a stage vector one of whose halves is not yet known, which is correct
**only** because the half that is read does not depend on it. This module does
not take that on trust, and does not probe for it either -- precedent R-9 is
what a probe buys. Two checks run on the values actually computed with, at
every step of every solve:

1. The unknown half is filled with ``NaN``, and every ``f`` value consumed
   during substitution is compared, **exactly**, against a re-evaluation at
   the completed stage. A problem whose ``f^q`` reads ``q`` either poisons the
   value it returns or returns a different number; either way the comparison
   fails.
2. ``F`` and ``G`` are checked for the exact block structure the domain
   asserts -- ``F^qq = 0``, ``F^pp = 0``, ``G^q = 0`` -- at the converged
   stages. That structure is what licenses the explicit block slicing in the
   tangent and adjoint sweeps, which is how those avoid needing a poison value
   at all.

Check 1 constrains ``f`` where it is evaluated; check 2 constrains its
derivative there. Neither implies the other, and the derivative sweeps rest on
the second.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from adjungo.core.partitioned import P, PartitionedMethod, Q
from adjungo.solvers.base import StageSolver, StepCache

if TYPE_CHECKING:
    from adjungo.core.problem import Problem

__all__ = ["PartitionedStageSolver", "SeparabilityViolation", "partition_of"]


class SeparabilityViolation(ValueError):
    """The problem is outside C-8.4's separable domain.

    Raised from the values a solve actually used, not from a probe. A
    partitioned method applied outside the domain does not fail loudly on its
    own -- it integrates a different problem -- so the domain is checked where
    it is relied upon.
    """


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

        # NaN, not zero: a half that has not been computed must not be
        # mistakable for one that has. Any read of it that reaches the answer
        # shows up in the comparison below (C-7).
        Z = np.full((s, n), np.nan)
        consumed: list[tuple[int, int, NDArray]] = []

        for block, i in method.dependency_order:
            A = self._array(method, block)
            acc = np.array(y[sl[block]], dtype=float)
            for j in range(s):
                if A[i, j] == 0.0:  # exact, as everywhere structure routes
                    continue
                f_j = np.asarray(
                    problem.f(Z[j], u_stages[j], t_stage[j]), dtype=float
                )[sl[block]]
                consumed.append((block, j, f_j.copy()))
                acc = acc + h * A[i, j] * f_j
            Z[i, sl[block]] = acc

        f_complete = [
            np.asarray(
                problem.f(Z[i], u_stages[i], t_stage[i]), dtype=float
            )
            for i in range(s)
        ]
        self._check_substitution_was_exact(consumed, f_complete, sl, step)

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

    @staticmethod
    def _check_substitution_was_exact(
        consumed: list[tuple[int, int, NDArray]],
        f_complete: list[NDArray],
        sl: tuple[slice, slice],
        step: int | None,
    ) -> None:
        """Every ``f`` value used mid-substitution must survive completion.

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
