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

if TYPE_CHECKING:
    from adjungo.core.method import GLMethod
    from adjungo.core.objective import Objective
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
    """

    def __init__(self, N: int, s: int, r: int, nx: int, nu: int) -> None:
        self.N = N
        self.s = s
        self.r = r
        self.nx = nx
        self.nu = nu
        self.n_ext = (N + 1) * r * nx
        self.n_int = N * s * nx
        self.nw = self.n_ext + self.n_int
        self.nctrl = N * s * nu

    def idx_Y(self, n: int, l: int) -> slice:
        """Slice of ``w`` holding external stage ``l`` at step ``n``."""
        start = (n * self.r + l) * self.nx
        return slice(start, start + self.nx)

    def idx_Z(self, n: int, i: int) -> slice:
        """Slice of ``w`` holding internal stage ``i`` of step ``n``."""
        start = self.n_ext + (n * self.s + i) * self.nx
        return slice(start, start + self.nx)

    def idx_u(self, n: int, k: int) -> slice:
        """Slice of the flattened control holding ``u^n_k``."""
        start = (n * self.s + k) * self.nu
        return slice(start, start + self.nu)

    def pack(self, Y: NDArray, Z: NDArray) -> NDArray:
        return np.concatenate([Y.reshape(-1), Z.reshape(-1)])

    def unpack(self, w: NDArray) -> tuple[NDArray, NDArray]:
        Y = w[: self.n_ext].reshape(self.N + 1, self.r, self.nx)
        Z = w[self.n_ext :].reshape(self.N, self.s, self.nx)
        return Y, Z


def _stage_times(t0: float, h: float, c: NDArray, step: int) -> NDArray:
    """Stage times ``t0 + step*h + c_j*h``.

    Matches ``adjungo.stepping.forward.forward_solve`` exactly.
    """
    return t0 + step * h + c * h


def _embed_initial(y0: NDArray, r: int, nx: int) -> NDArray:
    """Embed ``y0`` in external stage 0; remaining external stages are zero.

    Matches ``adjungo.stepping.forward._initialize_external_stages``.
    """
    Y0 = np.zeros((r, nx))
    Y0[0] = y0
    return Y0


def _residual(
    w: NDArray,
    u: NDArray,
    y0: NDArray,
    problem: Problem,
    method: GLMethod,
    t0: float,
    h: float,
    lay: _Layout,
) -> NDArray:
    """Assemble ``R(w, u)``."""
    Y, Z = lay.unpack(w)
    R = np.zeros(lay.nw)

    Y0_target = _embed_initial(y0, lay.r, lay.nx)
    for l in range(lay.r):
        R[lay.idx_Y(0, l)] = Y[0, l] - Y0_target[l]

    A, U, B, V, c = method.A, method.U, method.B, method.V, method.c

    for n in range(lay.N):
        t_stage = _stage_times(t0, h, c, n)
        f_st = np.array(
            [problem.f(Z[n, j], u[n, j], t_stage[j]) for j in range(lay.s)]
        )

        for i in range(lay.s):
            res = Z[n, i].copy()
            for k in range(lay.r):
                res -= U[i, k] * Y[n, k]
            for j in range(lay.s):
                res -= h * A[i, j] * f_st[j]
            R[lay.idx_Z(n, i)] = res

        for l in range(lay.r):
            res = Y[n + 1, l].copy()
            for k in range(lay.r):
                res -= V[l, k] * Y[n, k]
            for j in range(lay.s):
                res -= h * B[l, j] * f_st[j]
            R[lay.idx_Y(n + 1, l)] = res

    return R


def _jacobians(
    Z: NDArray,
    u: NDArray,
    problem: Problem,
    method: GLMethod,
    t0: float,
    h: float,
    lay: _Layout,
) -> tuple[NDArray, NDArray]:
    """Stage Jacobians ``F[n, j] = df/dy`` and ``G[n, j] = df/du``."""
    F = np.zeros((lay.N, lay.s, lay.nx, lay.nx))
    G = np.zeros((lay.N, lay.s, lay.nx, lay.nu))
    for n in range(lay.N):
        t_stage = _stage_times(t0, h, method.c, n)
        for j in range(lay.s):
            F[n, j] = problem.F(Z[n, j], u[n, j], t_stage[j])
            G[n, j] = problem.G(Z[n, j], u[n, j], t_stage[j])
    return F, G


def _dR_dw(
    F: NDArray, method: GLMethod, h: float, lay: _Layout
) -> NDArray:
    """Dense ``dR/dw``."""
    A, U, B, V = method.A, method.U, method.B, method.V
    Jw = np.zeros((lay.nw, lay.nw))
    I_nx = np.eye(lay.nx)

    for l in range(lay.r):
        Jw[lay.idx_Y(0, l), lay.idx_Y(0, l)] = I_nx

    for n in range(lay.N):
        for i in range(lay.s):
            row = lay.idx_Z(n, i)
            Jw[row, lay.idx_Z(n, i)] += I_nx
            for k in range(lay.r):
                Jw[row, lay.idx_Y(n, k)] += -U[i, k] * I_nx
            for j in range(lay.s):
                Jw[row, lay.idx_Z(n, j)] += -h * A[i, j] * F[n, j]

        for l in range(lay.r):
            row = lay.idx_Y(n + 1, l)
            Jw[row, lay.idx_Y(n + 1, l)] += I_nx
            for k in range(lay.r):
                Jw[row, lay.idx_Y(n, k)] += -V[l, k] * I_nx
            for j in range(lay.s):
                Jw[row, lay.idx_Z(n, j)] += -h * B[l, j] * F[n, j]

    return Jw


def _dR_du(
    G: NDArray, method: GLMethod, h: float, lay: _Layout
) -> NDArray:
    """Dense ``dR/du``."""
    A, B = method.A, method.B
    Ju = np.zeros((lay.nw, lay.nctrl))
    for n in range(lay.N):
        for j in range(lay.s):
            col = lay.idx_u(n, j)
            for i in range(lay.s):
                Ju[lay.idx_Z(n, i), col] += -h * A[i, j] * G[n, j]
            for l in range(lay.r):
                Ju[lay.idx_Y(n + 1, l), col] += -h * B[l, j] * G[n, j]
    return Ju


def reference_solve(
    y0: NDArray,
    u: NDArray,
    t_span: tuple[float, float],
    N: int,
    problem: Problem,
    method: GLMethod,
    tol: float = 1e-13,
    max_iter: int = 50,
) -> ReferenceSolution:
    """Solve the monolithic discrete system ``R(w, u) = 0`` by Newton's method.

    The Newton iteration is exact for explicit tableaux (``R_Z`` is block
    strictly lower triangular plus identity) and converges quadratically
    otherwise.

    Args:
        y0: Initial state, shape ``(n_x,)``.
        u: Controls, shape ``(N, s, nu)``.
        t_span: ``(t0, t1)``.
        N: Number of steps.
        problem: Problem callbacks.
        method: GLM tableau.
        tol: Target ``||R||_inf``.
        max_iter: Newton iteration cap.

    Returns:
        The converged :class:`ReferenceSolution`.

    Raises:
        RuntimeError: If Newton does not reach ``tol`` within ``max_iter``.
            The reference never returns an unconverged answer, because a
            silently unconverged reference is worse than no reference
            (clause C-6, no silent sentinels).
    """
    t0, t1 = t_span
    h = (t1 - t0) / N
    lay = _Layout(N, method.s, method.r, problem.state_dim, problem.control_dim)

    Y = np.zeros((N + 1, lay.r, lay.nx))
    Y[:] = _embed_initial(y0, lay.r, lay.nx)
    Z = np.zeros((N, lay.s, lay.nx))
    Z[:] = y0
    w = lay.pack(Y, Z)

    res_norm = np.inf
    for it in range(1, max_iter + 1):
        R = _residual(w, u, y0, problem, method, t0, h, lay)
        res_norm = float(np.max(np.abs(R)))
        if res_norm <= tol:
            Yc, Zc = lay.unpack(w)
            return ReferenceSolution(
                Y=Yc.copy(), Z=Zc.copy(), residual_norm=res_norm, iterations=it - 1
            )
        _, Zc = lay.unpack(w)
        F, _ = _jacobians(Zc, u, problem, method, t0, h, lay)
        Jw = _dR_dw(F, method, h, lay)
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
        for k in range(lay.s):
            g[lay.idx_u(n, k)] = np.asarray(objective.dJ_du(u[n, k], n, k))
    return g


def reference_gradient(
    y0: NDArray,
    u: NDArray,
    t_span: tuple[float, float],
    N: int,
    problem: Problem,
    method: GLMethod,
    objective: Objective,
    solution: ReferenceSolution | None = None,
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
    t0, t1 = t_span
    h = (t1 - t0) / N
    lay = _Layout(N, method.s, method.r, problem.state_dim, problem.control_dim)

    sol = solution or reference_solve(y0, u, t_span, N, problem, method)
    F, G = _jacobians(sol.Z, u, problem, method, t0, h, lay)

    Rw = _dR_dw(F, method, h, lay)
    Ru = _dR_du(G, method, h, lay)
    Jw = _objective_state_gradient(sol.Y, objective, lay)
    Ju = _objective_control_gradient(u, objective, lay)

    # p solves Rw^T p = Jw  ->  dJ/du = Ju - p^T Ru
    p = np.linalg.solve(Rw.T, Jw)
    grad = Ju - Ru.T @ p
    return np.asarray(grad, dtype=float).reshape(N, lay.s, lay.nu)


def _lagrange_contraction(
    p: NDArray, method: GLMethod, h: float, lay: _Layout
) -> NDArray:
    """Per-stage adjoint weight ``q[n, j]`` contracting the residual Hessian.

    Only the ``f`` terms of ``R`` are nonlinear, and ``f(Z^n_j, u^n_j, .)``
    appears in ``R_Z[n,i]`` with coefficient ``-h A[i,j]`` and in
    ``R_Y[n,l]`` with coefficient ``-h B[l,j]``.  Hence

        q[n, j] = -h ( sum_i A[i,j] p_{R_Z[n,i]} + sum_l B[l,j] p_{R_Y[n,l]} )

    and ``sum_m p_m grad^2 R_m`` restricted to stage ``(n, j)`` equals the
    contraction of ``grad^2 f`` against ``q[n, j]``.
    """
    A, B = method.A, method.B
    q = np.zeros((lay.N, lay.s, lay.nx))
    for n in range(lay.N):
        for j in range(lay.s):
            acc = np.zeros(lay.nx)
            for i in range(lay.s):
                acc += A[i, j] * p[lay.idx_Z(n, i)]
            for l in range(lay.r):
                acc += B[l, j] * p[lay.idx_Y(n + 1, l)]
            q[n, j] = -h * acc
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
    t_span: tuple[float, float],
    N: int,
    problem: Problem,
    method: GLMethod,
    objective: Objective,
    solution: ReferenceSolution | None = None,
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
    t0, t1 = t_span
    h = (t1 - t0) / N
    lay = _Layout(N, method.s, method.r, problem.state_dim, problem.control_dim)

    d2J_dy2_terminal = _require(objective, "d2J_dy2_terminal")
    d2J_dy2 = _require(objective, "d2J_dy2")
    for cb in ("F_yy_action", "F_yu_action", "F_uu_action"):
        if getattr(problem, cb, None) is None:
            raise NotImplementedError(
                f"reference_hessian requires problem.{cb}(); the problem "
                f"{type(problem).__name__} does not provide it."
            )

    sol = solution or reference_solve(y0, u, t_span, N, problem, method)
    F, G = _jacobians(sol.Z, u, problem, method, t0, h, lay)

    Rw = _dR_dw(F, method, h, lay)
    Ru = _dR_du(G, method, h, lay)
    gw = _objective_state_gradient(sol.Y, objective, lay)

    p = np.linalg.solve(Rw.T, gw)
    S = -np.linalg.solve(Rw, Ru)

    q = _lagrange_contraction(p, method, h, lay)

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
        for k in range(lay.s):
            L_uu[lay.idx_u(n, k), lay.idx_u(n, k)] += np.asarray(
                objective.d2J_du2(u[n, k], n, k)
            )

    # -sum_m p_m grad^2 R_m contributions, stage-local.
    for n in range(N):
        t_stage = _stage_times(t0, h, method.c, n)
        for j in range(lay.s):
            zj, uj, tj = sol.Z[n, j], u[n, j], t_stage[j]
            qj = q[n, j]
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
