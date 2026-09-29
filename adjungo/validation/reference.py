"""Monolithic reference for the discrete optimal-control system (U-O.1).

This module re-derives the *entire* discrete problem as a single algebraic
system and differentiates it densely.  It exists to provide an oracle that
is independent, in formulation, of the staged forward/adjoint recursions in
:mod:`adjungo.stepping`.

Contract
--------
Controlling clause C-14.1 ("oracle hierarchy") requires a truth source that
shares no code with ``adjungo/stepping/``.  Duality (tangent/adjoint inner
product agreement) is *insufficient* on its own, because tangent and adjoint
implementations can share a mistake and still be mutually consistent.  This
module is that independent source.

The discrete system
-------------------
Let ``N`` be the number of steps, ``h = (t1 - t0) / N``, and let the GLM be
given by the tableau blocks ``A`` (s x s), ``U`` (s x r), ``B`` (r x s),
``V`` (r x r) with abscissae ``c`` (s,).  The unknowns are

    w = ( y^[0], ..., y^[N],  Z^0, ..., Z^{N-1} )

where ``y^[n]`` has shape ``(r, n_x)`` and ``Z^n`` has shape ``(s, n_x)``.
The residual ``R(w, u) = 0`` has three families of blocks:

``R_init``
    ``y^[0] - E y_0``  where ``E`` embeds the initial condition in external
    stage 0 and zeroes the remaining ``r - 1`` external stages.

``R_Y[n]``  (n = 0 .. N-1)
    ``y^[n+1] - V y^[n] - h B f(Z^n, u^n, t^n)``

``R_Z[n,i]``  (n = 0 .. N-1, i = 0 .. s-1)
    ``Z^n_i - sum_k U[i,k] y^[n]_k - h sum_j A[i,j] f(Z^n_j, u^n_j, t^n_j)``

with ``t^n_j = t0 + n h + c_j h``.  These are exactly the equations
implemented by :func:`adjungo.stepping.forward.forward_solve`; reproducing
them is the point.

The reduced derivatives
-----------------------
With ``S = dw/du = -R_w^{-1} R_u`` the reduced gradient is

    dJ/du = J_u + J_w S = J_u - J_w R_w^{-1} R_u

and, introducing ``p`` from ``R_w^T p = J_w^T`` and the Lagrangian
``L = J - p^T R``, the reduced Hessian is

    d2J/du2 = [S; I]^T  grad^2_{(w,u)} L  [S; I].

Both are formed densely.  Cost is ``O(nw^3)``; this is a verification tool,
not a solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from adjungo.core.partitioned import PartitionedMethod

if TYPE_CHECKING:
    from adjungo.core.objective import Objective
    from adjungo.core.plan import DiscretizationPlan, StepMethod
    from adjungo.core.problem import Problem

__all__ = [
    "ReferenceSolution",
    "reference_gradient",
    "reference_hessian",
    "reference_solve",
]


@dataclass(frozen=True)
class ReferenceSolution:
    """Converged solution of the monolithic discrete system.

    Attributes:
        Y: External stages, shape ``(N + 1, r, n_x)``.
        Z: Internal stages, shape ``(N, s, n_x)``.
        residual_norm: Final ``||R(w, u)||_inf`` from the Newton solve.
        iterations: Newton iterations taken.
    """

    Y: NDArray
    Z: NDArray
    residual_norm: float
    iterations: int


class _Layout:
    """Index bookkeeping for the monolithic unknown vector.

    The residual vector uses the *same* layout as the unknown vector, so the
    Jacobian ``R_w`` is square with the natural block structure:

    * rows ``[0, r*n_x)``            -- ``R_init``
    * rows at ``idx_Y(n+1, l)``      -- ``R_Y[n]`` component ``l``
    * rows at ``idx_Z(n, i)``        -- ``R_Z[n, i]``

    Stage-indexed storage is packed, because a plan's steps need not carry
    the same stage count (C-18.1). The offsets are **accumulated here**, from
    the plan's method sequence, rather than read from
    ``DiscretizationPlan.stage_offsets``. That is not duplication for its own
    sake: C-14.1 puts this reference at tier 1 precisely because it shares no
    assembly with ``adjungo/stepping``, and an indexing mistake copied into
    both sides is exactly the class of error a tier-1 oracle exists to catch.
    What it *does* read from the plan is the plan itself -- nodes, step
    lengths and methods -- because that is the specification of the discrete
    problem both sides must solve, not an implementation of it.
    """

    def __init__(
        self, plan: DiscretizationPlan, nx: int, nu: int, n_q: int | None = None
    ) -> None:
        self.plan = plan
        self.N = plan.N
        self.nx = nx
        self.nu = nu
        self.n_q = n_q

        r_values = {m.r for m in plan.methods}
        if len(r_values) != 1:
            raise NotImplementedError(
                f"the reference requires one external-stage count for the "
                f"whole plan, got {sorted(r_values)} (NUMERICS.md C-18.3)."
            )
        self.r = r_values.pop()

        offsets = [0]
        for method in plan.methods:
            offsets.append(offsets[-1] + method.s)
        self.z_off = tuple(offsets)
        self.total_stages = offsets[-1]

        self.n_ext = (self.N + 1) * self.r * nx
        self.n_int = self.total_stages * nx
        self.nw = self.n_ext + self.n_int
        self.nctrl = self.total_stages * nu

        self.coupling = tuple(
            _stage_coupling(m, nx, n_q) for m in plan.methods
        )

    def coupling_at(self, n: int) -> NDArray:
        """``(s, s, nx)`` diagonals of the stage-coupling operators.

        Entry ``[i, j]`` is the diagonal of the operator multiplying
        ``h f(Z_j)`` in the equation for ``Z_i``. For an ordinary tableau
        that operator is ``A[i,j] I``, so every entry of the diagonal is the
        same number and the elementwise products below reduce to the scalar
        multiplications they replace. For a partitioned method it is
        ``diag(A^q_ij I, A^p_ij I)`` and they do not (C-8.4).
        """
        return self.coupling[n]

    def s_at(self, n: int) -> int:
        """Stage count of step ``n``."""
        return self.plan.methods[n].s

    def method_at(self, n: int) -> StepMethod:
        """Tableau executing step ``n``."""
        return self.plan.methods[n]

    def h_at(self, n: int) -> float:
        """Step length ``h_n``."""
        return float(self.plan.h[n])

    def t_at(self, n: int) -> float:
        """Time at the start of step ``n``."""
        return float(self.plan.nodes[n])

    def stage(self, packed: NDArray, n: int) -> NDArray:
        """The ``(s_n, ...)`` block of a packed stage array."""
        start = self.z_off[n]
        return packed[start:start + self.s_at(n)]

    def idx_Y(self, n: int, l: int) -> slice:
        """Slice of ``w`` holding external stage ``l`` at step ``n``."""
        start = (n * self.r + l) * self.nx
        return slice(start, start + self.nx)

    def idx_Z(self, n: int, i: int) -> slice:
        """Slice of ``w`` holding internal stage ``i`` of step ``n``."""
        start = self.n_ext + (self.z_off[n] + i) * self.nx
        return slice(start, start + self.nx)

    def idx_u(self, n: int, k: int) -> slice:
        """Slice of the flattened control holding ``u^n_k``."""
        start = (self.z_off[n] + k) * self.nu
        return slice(start, start + self.nu)

    def pack(self, Y: NDArray, Z: NDArray) -> NDArray:
        return np.concatenate([Y.reshape(-1), Z.reshape(-1)])

    def unpack(self, w: NDArray) -> tuple[NDArray, NDArray]:
        Y = w[: self.n_ext].reshape(self.N + 1, self.r, self.nx)
        Z = w[self.n_ext :].reshape(self.total_stages, self.nx)
        return Y, Z

    def pack_stage_input(self, arr: NDArray, name: str) -> NDArray:
        """Accept a rectangular ``(N, s, ...)`` stage array, or a packed one.

        Callers predating C-18 pass ``(N, s, nu)`` controls, and a plan with
        one stage count throughout can be viewed that way. The check is
        written against this layout's own ``total_stages``.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim >= 3:
            counts = {m.s for m in self.plan.methods}
            if len(counts) != 1 or a.shape[:2] != (self.N, self.s_at(0)):
                raise ValueError(
                    f"{name} has shape {a.shape}, which does not describe "
                    f"this plan's stages (NUMERICS.md C-18.5)."
                )
            return a.reshape(self.total_stages, *a.shape[2:])
        if a.ndim < 1 or a.shape[0] != self.total_stages:
            raise ValueError(
                f"{name} has shape {a.shape}, expected leading length "
                f"{self.total_stages}."
            )
        return a


def _stage_times(lay: _Layout, step: int) -> NDArray:
    """Stage times ``t_n + c_j h_n`` for step ``step``.

    Matches ``adjungo.stepping.forward.forward_solve``'s convention (C-8.1)
    while being computed here from the plan's nodes and step lengths.
    """
    method = lay.method_at(step)
    return lay.t_at(step) + method.c * lay.h_at(step)


def _embed_initial(y0: NDArray, r: int, nx: int) -> NDArray:
    """Embed ``y0`` in external stage 0; remaining external stages are zero.

    Matches ``adjungo.stepping.forward._initialize_external_stages``.
    """
    Y0 = np.zeros((r, nx))
    Y0[0] = y0
    return Y0


def _stage_coupling(
    method: StepMethod, nx: int, n_q: int | None
) -> NDArray:
    """``(s, s, nx)``: the diagonal of each stage-coupling operator.

    Writing the coupling as a diagonal operator rather than a scalar is what
    lets one assembly serve both method kinds. It is not a generalisation for
    its own sake: substituting ``A[i,j] I -> diag(A^q_ij I, A^p_ij I)`` is
    exactly the definition of a partitioned method (C-8.4), and the only
    place the two kinds differ in this module.
    """
    s = method.s
    if isinstance(method, PartitionedMethod):
        if n_q is None:
            raise ValueError(
                "the reference needs n_q for a partitioned method; the "
                "partition belongs to the problem (NUMERICS.md C-8.4)."
            )
        if not 0 < n_q < nx:
            raise ValueError(
                f"n_q must lie strictly inside the state: got n_q={n_q} "
                f"for nx={nx}."
            )
        out = np.empty((s, s, nx))
        out[:, :, :n_q] = method.A_q[:, :, None]
        out[:, :, n_q:] = method.A_p[:, :, None]
        return out
    return np.repeat(method.A[:, :, None], nx, axis=2)


def _apply_coupling(coup: NDArray, f_st: NDArray) -> NDArray:
    """``sum_j 𝒜_ij f_j`` for all ``i``, from the diagonals in ``coup``."""
    applied: NDArray = np.einsum("ijn,jn->in", coup, f_st)
    return applied


def _residual(
    w: NDArray,
    u: NDArray,
    y0: NDArray,
    problem: Problem,
    lay: _Layout,
) -> NDArray:
    """Assemble ``R(w, u)``."""
    Y, Z = lay.unpack(w)
    R = np.zeros(lay.nw)

    Y0_target = _embed_initial(y0, lay.r, lay.nx)
    for l in range(lay.r):
        R[lay.idx_Y(0, l)] = Y[0, l] - Y0_target[l]

    for n in range(lay.N):
        method = lay.method_at(n)
        U, B, V = method.U, method.B, method.V
        coup = lay.coupling_at(n)
        h = lay.h_at(n)
        s = lay.s_at(n)
        t_stage = _stage_times(lay, n)
        Zn, un = lay.stage(Z, n), lay.stage(u, n)
        f_st = np.array(
            [problem.f(Zn[j], un[j], t_stage[j]) for j in range(s)]
        )

        for i in range(s):
            res = Zn[i].copy()
            for k in range(lay.r):
                res -= U[i, k] * Y[n, k]
            for j in range(s):
                res -= h * coup[i, j] * f_st[j]
            R[lay.idx_Z(n, i)] = res

        for l in range(lay.r):
            res = Y[n + 1, l].copy()
            for k in range(lay.r):
                res -= V[l, k] * Y[n, k]
            for j in range(s):
                res -= h * B[l, j] * f_st[j]
            R[lay.idx_Y(n + 1, l)] = res

    return R


def _jacobians(
    Z: NDArray,
    u: NDArray,
    problem: Problem,
    lay: _Layout,
) -> tuple[NDArray, NDArray]:
    """Packed stage Jacobians ``df/dy`` and ``df/du``, one block per stage."""
    F = np.zeros((lay.total_stages, lay.nx, lay.nx))
    G = np.zeros((lay.total_stages, lay.nx, lay.nu))
    for n in range(lay.N):
        t_stage = _stage_times(lay, n)
        Zn, un = lay.stage(Z, n), lay.stage(u, n)
        Fn, Gn = lay.stage(F, n), lay.stage(G, n)
        for j in range(lay.s_at(n)):
            Fn[j] = problem.F(Zn[j], un[j], t_stage[j])
            Gn[j] = problem.G(Zn[j], un[j], t_stage[j])
    return F, G


def _dR_dw(F: NDArray, lay: _Layout) -> NDArray:
    """Dense ``dR/dw``."""
    Jw = np.zeros((lay.nw, lay.nw))
    I_nx = np.eye(lay.nx)

    for l in range(lay.r):
        Jw[lay.idx_Y(0, l), lay.idx_Y(0, l)] = I_nx

    for n in range(lay.N):
        method = lay.method_at(n)
        U, B, V = method.U, method.B, method.V
        coup = lay.coupling_at(n)
        h = lay.h_at(n)
        s = lay.s_at(n)
        Fn = lay.stage(F, n)

        for i in range(s):
            row = lay.idx_Z(n, i)
            Jw[row, lay.idx_Z(n, i)] += I_nx
            for k in range(lay.r):
                Jw[row, lay.idx_Y(n, k)] += -U[i, k] * I_nx
            for j in range(s):
                Jw[row, lay.idx_Z(n, j)] += -h * coup[i, j][:, None] * Fn[j]

        for l in range(lay.r):
            row = lay.idx_Y(n + 1, l)
            Jw[row, lay.idx_Y(n + 1, l)] += I_nx
            for k in range(lay.r):
                Jw[row, lay.idx_Y(n, k)] += -V[l, k] * I_nx
            for j in range(s):
                Jw[row, lay.idx_Z(n, j)] += -h * B[l, j] * Fn[j]

    return Jw


def _dR_du(G: NDArray, lay: _Layout) -> NDArray:
    """Dense ``dR/du``."""
    Ju = np.zeros((lay.nw, lay.nctrl))
    for n in range(lay.N):
        method = lay.method_at(n)
        B = method.B
        coup = lay.coupling_at(n)
        h = lay.h_at(n)
        s = lay.s_at(n)
        Gn = lay.stage(G, n)
        for j in range(s):
            col = lay.idx_u(n, j)
            for i in range(s):
                Ju[lay.idx_Z(n, i), col] += -h * coup[i, j][:, None] * Gn[j]
            for l in range(lay.r):
                Ju[lay.idx_Y(n + 1, l), col] += -h * B[l, j] * Gn[j]
    return Ju


def _stage_defect(
    Zn: NDArray,
    explicit: NDArray,
    problem: Problem,
    u_step: NDArray,
    t_stage: NDArray,
    coup: NDArray,
    h: float,
    s: int,
) -> float:
    """How far ``Zn`` is from satisfying one step's stage equations."""
    f_st = np.array([problem.f(Zn[j], u_step[j], t_stage[j]) for j in range(s)])
    return float(np.max(np.abs(Zn - explicit - h * _apply_coupling(coup, f_st))))


def _constant_guess(y0: NDArray, lay: _Layout) -> NDArray:
    """Every step node and every stage at the initial state."""
    Y = np.zeros((lay.N + 1, lay.r, lay.nx))
    Y[:] = _embed_initial(y0, lay.r, lay.nx)
    Z = np.zeros((lay.total_stages, lay.nx))
    Z[:] = y0
    return lay.pack(Y, Z)


def _swept_guess(
    u: NDArray,
    y0: NDArray,
    problem: Problem,
    lay: _Layout,
    max_sweeps: int = 4,
) -> NDArray:
    """Sweep a candidate ``w`` forward along the trajectory.

    Each step takes its stages from the explicit part ``sum_k U[i,k] Y[n,k]``
    and then applies Picard corrections, which are fixed-point updates built
    from the tableau alone. No linear system is solved, so no implicit
    equation is smuggled into the predictor, and this shares no code with
    ``adjungo/stepping`` as C-14.1 requires of this reference. The clause
    governing this starting point is C-14.3.

    The map is ``Z -> explicit + h A f(Z)``, so a sufficient condition for it
    to contract is ``h ||A|| L < 1`` on the states it visits, where ``L`` is
    a Lipschitz constant for ``f``. The tableau belongs in that bound and
    dropping it understates the useful range badly: implicit midpoint has
    ``||A||_inf = 1/2``, so ``y' = -1.5 y`` at ``h = 1`` contracts by exactly
    ``0.75`` per sweep even though ``h L = 1.5``.

    That bound is sufficient and not necessary, so failing it predicts
    nothing. It is a norm bound on the iteration matrix, and for a
    non-normal one it is pessimistic: implicit midpoint at ``h=1`` with
    ``f(y) = My``, ``M = [[-1/4, 2], [0, -1/4]]``, has ``h||A||L = 9/8`` yet
    an iteration matrix of spectral radius ``1/8``, and its defects fall.

    Some sweeps really do amplify, which is the case this code has to
    survive. Measured, on the Van der Pol fixture of
    ``examples/nonlinear_implicit_control.py`` at ``h=1.2``: two unguarded
    sweeps reach ``3e+72`` and three reach ``1e+252``.

    Neither constant is available here -- ``L`` is a property of a caller's
    callback on a region not known in advance -- so no such bound is
    evaluated. A sweep is instead kept only while it is observed to reduce
    the stage defect it is trying to zero. That is a monotonicity test, not a
    proof of contraction: one decrease does not make a map a contraction. It
    is used because it needs nothing beyond the values already computed and
    because it stops the iteration on the step where growth begins, which is
    what the caller needs from it. It also absorbs an overflowed sweep
    without a separate test for one, since a non-finite candidate has a
    non-finite defect and ``nan < defect`` is false.

    The caller must still compare this against the alternative. The forward
    propagation of ``Y`` is explicit, so it can amplify from step to step even
    when no sweep is accepted within a step, and this function has no basis
    for judging the trajectory as a whole.
    """
    Y = np.zeros((lay.N + 1, lay.r, lay.nx))
    Z = np.zeros((lay.total_stages, lay.nx))
    Y[0] = _embed_initial(y0, lay.r, lay.nx)

    for n in range(lay.N):
        method = lay.method_at(n)
        U, B, V = method.U, method.B, method.V
        coup = lay.coupling_at(n)
        h = lay.h_at(n)
        s = lay.s_at(n)
        t_stage = _stage_times(lay, n)
        un = lay.stage(u, n)
        explicit = np.array(
            [sum(U[i, k] * Y[n, k] for k in range(lay.r)) for i in range(s)]
        )

        Zn = explicit
        defect = _stage_defect(
            Zn, explicit, problem, un, t_stage, coup, h, s
        )
        for _ in range(max_sweeps):
            f_st = np.array(
                [problem.f(Zn[j], un[j], t_stage[j]) for j in range(s)]
            )
            candidate = explicit + h * _apply_coupling(coup, f_st)
            candidate_defect = _stage_defect(
                candidate, explicit, problem, un, t_stage, coup, h, s
            )
            if not candidate_defect < defect:
                break
            Zn, defect = candidate, candidate_defect

        lay.stage(Z, n)[...] = Zn
        f_st = np.array(
            [problem.f(Zn[j], un[j], t_stage[j]) for j in range(s)]
        )
        for l in range(lay.r):
            Y[n + 1, l] = sum(
                V[l, k] * Y[n, k] for k in range(lay.r)
            ) + h * sum(B[l, j] * f_st[j] for j in range(s))

    return lay.pack(Y, Z)


def _initial_guess(
    u: NDArray,
    y0: NDArray,
    problem: Problem,
    lay: _Layout,
) -> NDArray:
    """Choose where Newton starts, by measured residual.

    Holding every node at ``y0`` puts the starting point a whole trajectory
    excursion away from the root, which costs iterations on a short horizon
    and reach on a long one: the undriven pendulum of
    ``examples/pendulum_swing_up.py`` over sixteen periods leaves Newton at
    ``||R||_inf = 7.5e+07`` after fifty iterations, and the reference then
    refuses rather than return a half-converged answer.

    Sweeping forward fixes that where the sweep is stable and makes it worse
    where it is not, so neither candidate is right unconditionally. Both are
    built and the one with the smaller ``||R||_inf`` is used. That is the same
    quantity Newton is driving to ``tol``, it is cheap beside the dense solve
    of each iteration, and it bounds the damage: the swept guess is adopted
    only on evidence that it is the better starting point.

    This changes only where Newton starts; the acceptance test is untouched.
    A returned answer has been measured against the same ``tol`` on the same
    residual, including when no Newton update was needed -- ``iterations=0``
    reports a starting point that was checked and found converged, not one
    that was assumed to be.

    The accepted point is not bit-identical to the one the constant start
    reached, and cannot be: Newton stops at the first iterate inside an
    absolute residual ball of radius ``tol``, and which point that is depends
    on where it began. Across the gauss2, sdirk3 and rk4 fixtures of the
    damped pendulum at ``N=20`` and ``N=80`` the two accepted roots differ by
    at most ``5e-14`` relative. That is a measurement on those fixtures, not
    a bound: a residual ball is not a state-error ball, and the two are
    related only through ``||(dR/dw)^-1||``, which grows with the horizon.
    Consumers of this reference compare derivatives at rounding level, so the
    displacement is reported here rather than asserted away.

    A starting point can in principle select a different root, and no
    residual test can rule that out. What the acceptance test does guarantee
    is that whatever comes back is a root of the correct discrete system.
    Sweeping is expected to *help* here rather than hurt, because a guess
    that follows the dynamics lands near the solution branch continuous with
    ``y0``, whereas a constant guess has no such affinity -- but that is a
    reasoned expectation, not a theorem, and it is not relied on.
    """
    constant = _constant_guess(y0, lay)

    # The predictor is optional, and it evaluates ``f`` at trial states the
    # solution never visits. A callback that guards its domain -- as C-7 asks
    # it to -- will raise there: ``y' = -sqrt(y)`` from ``y0 = 1`` with
    # ``h = 1.5`` is asked for ``f(-0.5)`` on the first sweep, though both the
    # continuous and the discrete solutions stay positive. Any failure of the
    # predictor is therefore a rejected candidate, not a failure of the solve.
    # Newton itself runs outside this guard, so a genuine error there is still
    # raised.
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            swept = _swept_guess(u, y0, problem, lay)
            swept_defect = np.max(
                np.abs(_residual(swept, u, y0, problem, lay))
            )
    except Exception:  # noqa: BLE001 - see above; any predictor failure falls back
        return constant

    constant_defect = np.max(
        np.abs(_residual(constant, u, y0, problem, lay))
    )
    if swept_defect < constant_defect:
        return swept
    return constant


def _resolve_plan(
    t_span: tuple[float, float] | None,
    N: int | None,
    method: StepMethod | None,
    plan: DiscretizationPlan | None,
) -> DiscretizationPlan:
    """Accept either an explicit plan or the uniform ``(t_span, N, method)``.

    The uniform arguments are a plan like any other (C-18.1). Supplying both
    forms is refused rather than resolved by precedence: the reference would
    otherwise validate the package against a discretization the caller did
    not name.
    """
    from adjungo.core.plan import DiscretizationPlan as _Plan

    uniform = (t_span, N, method)
    if plan is None:
        missing = [
            name
            for name, value in zip(
                ("t_span", "N", "method"), uniform, strict=True
            )
            if value is None
        ]
        if missing:
            raise TypeError(
                f"the reference needs either plan= or all of t_span, N, "
                f"method; missing {', '.join(missing)}."
            )
        assert t_span is not None and N is not None and method is not None
        return _Plan.uniform(t_span, N, method)
    if any(value is not None for value in uniform):
        raise TypeError(
            "the reference takes either plan= or (t_span, N, method), not "
            "both."
        )
    return plan


def reference_solve(
    y0: NDArray,
    u: NDArray,
    t_span: tuple[float, float] | None = None,
    N: int | None = None,
    problem: Problem | None = None,
    method: StepMethod | None = None,
    tol: float = 1e-13,
    max_iter: int = 50,
    *,
    plan: DiscretizationPlan | None = None,
) -> ReferenceSolution:
    """Solve the monolithic discrete system ``R(w, u) = 0`` by Newton's method.

    For an explicit tableau ``R_Z`` is block strictly lower triangular plus
    identity, so each Newton step is a forward substitution. That structure
    is not exactness: the iteration terminates in one step only when ``f`` is
    affine in the state, and a nonlinear ``f`` under an explicit tableau still
    iterates -- measurably two steps for ``examples/rocket_ascent.py``.
    Convergence is quadratic in either case.

    ``tol`` is an **absolute** bound on ``||R||_inf``, so it carries the units
    and scale of ``h * f``. A problem whose residual entries are of order
    ``1e4`` cannot reach the default at all, and will exhaust ``max_iter``
    against a target below its own rounding floor; pass a ``tol`` derived from
    that scale. A residual bound is in any case a stopping rule and not an
    error bound on a derivative computed from the result. See open question
    C-Q8 in ``NUMERICS.md``.

    Args:
        y0: Initial state, shape ``(n_x,)``.
        u: Controls, shape ``(N, s, nu)``.
        t_span: ``(t0, t1)``.
        N: Number of steps.
        problem: Problem callbacks.
        method: GLM tableau.
        tol: Target ``||R||_inf``, absolute. See above.
        max_iter: Newton iteration cap.

    Returns:
        The converged :class:`ReferenceSolution`.

    Raises:
        RuntimeError: If Newton does not reach ``tol`` within ``max_iter``.
            The reference never returns an unconverged answer, because a
            silently unconverged reference is worse than no reference
            (clause C-6, no silent sentinels).
    """
    if problem is None:
        raise TypeError("reference_solve requires problem.")
    plan = _resolve_plan(t_span, N, method, plan)
    lay = _Layout(
        plan,
        problem.state_dim,
        problem.control_dim,
        n_q=getattr(problem, "n_q", None),
    )
    u = lay.pack_stage_input(u, "control")
    w = _initial_guess(u, y0, problem, lay)

    res_norm = np.inf
    for it in range(1, max_iter + 1):
        R = _residual(w, u, y0, problem, lay)
        res_norm = float(np.max(np.abs(R)))
        if res_norm <= tol:
            Yc, Zc = lay.unpack(w)
            return ReferenceSolution(
                Y=Yc.copy(), Z=Zc.copy(), residual_norm=res_norm, iterations=it - 1
            )
        _, Zc = lay.unpack(w)
        F, _ = _jacobians(Zc, u, problem, lay)
        Jw = _dR_dw(F, lay)
        w = w - np.linalg.solve(Jw, R)

    raise RuntimeError(
        f"reference_solve: Newton failed to converge in {max_iter} iterations; "
        f"final ||R||_inf = {res_norm:.6e} (tol = {tol:.6e})"
    )


def _objective_state_gradient(
    Y: NDArray, objective: Objective, lay: _Layout
) -> NDArray:
    """``dJ/dw`` for the discrete objective implemented by the package.

    The package's discrete objective (see
    :func:`adjungo.stepping.adjoint.adjoint_solve` and
    :func:`adjungo.optimization.gradient.assemble_gradient`) is

        J = Phi(y^[N]) + sum_{n=0}^{N-1} phi_n(y^[n]) + sum_{n,k} psi(u^n_k)

    so ``dJ/dZ = 0``.  A stage-state running cost is C-9.3 work and is not
    yet part of the protocol; when it lands, this function must gain the
    corresponding ``dJ/dZ`` block.
    """
    g = np.zeros(lay.nw)
    N = lay.N
    term = np.asarray(objective.dJ_dy_terminal(Y[N]))
    for l in range(lay.r):
        g[lay.idx_Y(N, l)] = term[l]
    for n in range(N):
        run = np.asarray(objective.dJ_dy(Y[n], n))
        for l in range(lay.r):
            g[lay.idx_Y(n, l)] = run[l]
    return g


def _objective_control_gradient(
    u: NDArray, objective: Objective, lay: _Layout
) -> NDArray:
    """``dJ/du`` holding the state fixed (the explicit control dependence)."""
    g = np.zeros(lay.nctrl)
    for n in range(lay.N):
        un = lay.stage(u, n)
        for k in range(lay.s_at(n)):
            g[lay.idx_u(n, k)] = np.asarray(objective.dJ_du(un[k], n, k))
    return g


def reference_gradient(
    y0: NDArray,
    u: NDArray,
    t_span: tuple[float, float] | None = None,
    N: int | None = None,
    problem: Problem | None = None,
    method: StepMethod | None = None,
    objective: Objective | None = None,
    solution: ReferenceSolution | None = None,
    *,
    plan: DiscretizationPlan | None = None,
) -> NDArray:
    """Exact discrete reduced gradient ``dJ/du`` of the monolithic system.

    Args:
        y0: Initial state, shape ``(n_x,)``.
        u: Controls, shape ``(N, s, nu)``.
        t_span: ``(t0, t1)``.
        N: Number of steps.
        problem: Problem callbacks.
        method: GLM tableau.
        objective: Objective callbacks.
        solution: Optional precomputed forward solution; recomputed if absent.

    Returns:
        Gradient with shape ``(N, s, nu)``, matching the layout of ``u``.
    """
    if problem is None or objective is None:
        raise TypeError("reference_gradient requires problem and objective.")
    shape_in = np.shape(u)
    plan = _resolve_plan(t_span, N, method, plan)
    lay = _Layout(
        plan,
        problem.state_dim,
        problem.control_dim,
        n_q=getattr(problem, "n_q", None),
    )
    u = lay.pack_stage_input(u, "control")

    sol = solution or reference_solve(y0, u, problem=problem, plan=plan)
    F, G = _jacobians(sol.Z, u, problem, lay)

    Rw = _dR_dw(F, lay)
    Ru = _dR_du(G, lay)
    Jw = _objective_state_gradient(sol.Y, objective, lay)
    Ju = _objective_control_gradient(u, objective, lay)

    # p solves Rw^T p = Jw  ->  dJ/du = Ju - p^T Ru
    p = np.linalg.solve(Rw.T, Jw)
    grad = Ju - Ru.T @ p
    out_shape = shape_in if len(shape_in) == 3 else (lay.total_stages, lay.nu)
    return np.asarray(grad, dtype=float).reshape(out_shape)


def _lagrange_contraction(p: NDArray, lay: _Layout) -> NDArray:
    """Per-stage adjoint weight ``q[n, j]`` contracting the residual Hessian.

    Only the ``f`` terms of ``R`` are nonlinear, and ``f(Z^n_j, u^n_j, .)``
    appears in ``R_Z[n,i]`` with coefficient ``-h A[i,j]`` and in
    ``R_Y[n,l]`` with coefficient ``-h B[l,j]``.  Hence

        q[n, j] = -h ( sum_i A[i,j] p_{R_Z[n,i]} + sum_l B[l,j] p_{R_Y[n,l]} )

    and ``sum_m p_m grad^2 R_m`` restricted to stage ``(n, j)`` equals the
    contraction of ``grad^2 f`` against ``q[n, j]``.
    """
    q = np.zeros((lay.total_stages, lay.nx))
    for n in range(lay.N):
        method = lay.method_at(n)
        B = method.B
        coup = lay.coupling_at(n)
        h = lay.h_at(n)
        s = lay.s_at(n)
        qn = lay.stage(q, n)
        for j in range(s):
            acc = np.zeros(lay.nx)
            for i in range(s):
                acc += coup[i, j] * p[lay.idx_Z(n, i)]
            for l in range(lay.r):
                acc += B[l, j] * p[lay.idx_Y(n + 1, l)]
            qn[j] = -h * acc
    return q


def _require(objective: Objective, name: str) -> Any:
    fn = getattr(objective, name, None)
    if fn is None:
        raise NotImplementedError(
            f"reference_hessian requires objective.{name}(); the objective "
            f"{type(objective).__name__} does not provide it. Second-order "
            "verification needs the full C-9 objective protocol."
        )
    return fn


def reference_hessian(
    y0: NDArray,
    u: NDArray,
    t_span: tuple[float, float] | None = None,
    N: int | None = None,
    problem: Problem | None = None,
    method: StepMethod | None = None,
    objective: Objective | None = None,
    solution: ReferenceSolution | None = None,
    *,
    plan: DiscretizationPlan | None = None,
) -> NDArray:
    """Exact dense discrete reduced Hessian ``d2J/du2``.

    Requires second derivatives from both the problem
    (``F_yy_action``, ``F_yu_action``, ``F_uu_action``) and the objective
    (``d2J_du2``, ``d2J_dy2``, ``d2J_dy2_terminal``).

    Args:
        y0: Initial state, shape ``(n_x,)``.
        u: Controls, shape ``(N, s, nu)``.
        t_span: ``(t0, t1)``.
        N: Number of steps.
        problem: Problem callbacks including second derivatives.
        method: GLM tableau.
        objective: Objective callbacks including second derivatives.
        solution: Optional precomputed forward solution.

    Returns:
        Symmetric matrix of shape ``(N*s*nu, N*s*nu)`` in the flattened
        control ordering of :meth:`_Layout.idx_u`.

    Raises:
        NotImplementedError: If a required second-derivative callback is
            missing.
    """
    if problem is None or objective is None:
        raise TypeError("reference_hessian requires problem and objective.")
    plan = _resolve_plan(t_span, N, method, plan)
    lay = _Layout(
        plan,
        problem.state_dim,
        problem.control_dim,
        n_q=getattr(problem, "n_q", None),
    )
    u = lay.pack_stage_input(u, "control")
    N = lay.N

    d2J_dy2_terminal = _require(objective, "d2J_dy2_terminal")
    d2J_dy2 = _require(objective, "d2J_dy2")
    for cb in ("F_yy_action", "F_yu_action", "F_uu_action"):
        if getattr(problem, cb, None) is None:
            raise NotImplementedError(
                f"reference_hessian requires problem.{cb}(); the problem "
                f"{type(problem).__name__} does not provide it."
            )

    sol = solution or reference_solve(y0, u, problem=problem, plan=plan)
    F, G = _jacobians(sol.Z, u, problem, lay)

    Rw = _dR_dw(F, lay)
    Ru = _dR_du(G, lay)
    gw = _objective_state_gradient(sol.Y, objective, lay)

    p = np.linalg.solve(Rw.T, gw)
    S = -np.linalg.solve(Rw, Ru)

    q = _lagrange_contraction(p, lay)

    # Lagrangian second derivatives: L = J - p^T R.
    L_ww = np.zeros((lay.nw, lay.nw))
    L_wu = np.zeros((lay.nw, lay.nctrl))
    L_uu = np.zeros((lay.nctrl, lay.nctrl))

    term_H = np.asarray(d2J_dy2_terminal(sol.Y[N]))
    L_ww[lay.idx_Y(N, 0), lay.idx_Y(N, 0)] += term_H
    for n in range(N):
        L_ww[lay.idx_Y(n, 0), lay.idx_Y(n, 0)] += np.asarray(
            d2J_dy2(sol.Y[n], n)
        )
        un = lay.stage(u, n)
        for k in range(lay.s_at(n)):
            L_uu[lay.idx_u(n, k), lay.idx_u(n, k)] += np.asarray(
                objective.d2J_du2(un[k], n, k)
            )

    # -sum_m p_m grad^2 R_m contributions, stage-local.
    for n in range(N):
        t_stage = _stage_times(lay, n)
        Zn, un, qn = lay.stage(sol.Z, n), lay.stage(u, n), lay.stage(q, n)
        for j in range(lay.s_at(n)):
            zj, uj, tj = Zn[j], un[j], t_stage[j]
            qj = qn[j]
            L_ww[lay.idx_Z(n, j), lay.idx_Z(n, j)] += -np.asarray(
                problem.F_yy_action(zj, uj, tj, qj)
            )
            cross = -np.asarray(problem.F_yu_action(zj, uj, tj, qj))
            L_wu[lay.idx_Z(n, j), lay.idx_u(n, j)] += cross
            L_uu[lay.idx_u(n, j), lay.idx_u(n, j)] += -np.asarray(
                problem.F_uu_action(zj, uj, tj, qj)
            )

    H = S.T @ L_ww @ S + S.T @ L_wu + L_wu.T @ S + L_uu
    return np.asarray(0.5 * (H + H.T), dtype=float)
