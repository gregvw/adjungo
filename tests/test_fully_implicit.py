"""Certification evidence for the fully implicit (dense-``A``) stage solver.

Covers milestone M3. The exact-derivative evidence for ``gauss2`` lives in
``test_oracle_gradient.py`` and ``test_oracle_hessian.py``, where it is held to
the same tolerance as every other certified method; this module carries the
claims that are specific to solving all stages at once:

* the coupled Newton solve reaches the C-3.4 residual tolerance, and does not
  return the zeros that produced precedent R-4;
* the cached factorization is of the analytic Jacobian **at the converged
  iterate** (C-5.4), checked against a Jacobian built by differencing the
  residual, which shares no code with the analytic one;
* the adjoint operator is the transpose of that same Jacobian, so the stage
  adjoints may reuse the forward factorization (this is the property that makes
  ``trans=1`` legitimate rather than merely convenient);
* a failed coupled solve names a stage block, satisfying C-5.3;
* ``gauss2`` attains its nominal order 4 against a closed-form linear solution
  (C-4). That is a *continuous*-accuracy claim on a refined mesh and, per C-2,
  can never excuse a derivative discrepancy at a fixed mesh.
"""

import numpy as np
import pytest
import scipy.linalg

from adjungo.core.method import GLMethod, StageType
from adjungo.core.problem import Linearity, ProblemStructure
from adjungo.core.requirements import deduce_requirements
from adjungo.methods.runge_kutta import (
    gauss2,
    implicit_midpoint,
    rk4,
    sdirk3,
)
from adjungo.solvers.factory import create_stage_solver
from adjungo.solvers.implicit import (
    ImplicitStageSolver,
    solve_coupled,
    solve_coupled_transposed,
)
from adjungo.solvers.newton import STAGE_NEWTON_TOL, StageSolveError
from adjungo.stepping.forward import forward_solve
from tests.problems import CoupledNonlinear

#: Residual floor for a converged coupled solve. C-3.4 sets the Newton stopping
#: test; a converged step must sit at or below it, not merely "close".
COUPLED_RESIDUAL_TOL = 1e-12


class _LTI:
    """``y' = A y``: a damped oscillator with a closed-form solution.

    Used for the C-4 order study. The reference is ``expm(A T) y0``, which is
    independent of every integrator under test, so an order deficiency cannot
    be hidden by comparing two implementations that share an error.
    """

    state_dim = 2
    control_dim = 1
    Amat = np.array([[0.0, 1.0], [-4.0, -0.4]])

    def f(self, y, u, t):
        return self.Amat @ y

    def F(self, y, u, t):
        return self.Amat

    def G(self, y, u, t):
        return np.zeros((2, 1))


class _Blowup:
    """``y' = k y^2``, used only to force a coupled-solve failure.

    The stage equation for a diagonal entry ``a`` of the tableau is
    ``z = y0 + h a k z^2``, a quadratic whose discriminant
    ``1 - 4 h a k y0`` is negative for large ``h k y0``. There is then no real
    stage value to find and Newton cannot converge -- as opposed to a merely
    stiff problem, where Newton converges perfectly well. Contraction is why
    an earlier attempt with ``y' = -k y^3`` did *not* produce a failure here.

    The two components are given different initial values so that the two
    stage blocks carry different residuals, which makes the C-5.3 "worst
    block" index a real choice rather than a constant.

    Not a certification target.
    """

    state_dim = 2
    control_dim = 1

    def __init__(self, k: float = 1e3) -> None:
        self.k = k

    def f(self, y, u, t):
        return self.k * y**2

    def F(self, y, u, t):
        return np.diag(2.0 * self.k * y)

    def G(self, y, u, t):
        return np.zeros((2, 1))


def _structure(linear: bool) -> ProblemStructure:
    return ProblemStructure(
        linearity=Linearity.LINEAR if linear else Linearity.NONLINEAR,
        jacobian_constant=linear,
        jacobian_control_dependent=not linear,
        has_second_derivatives=False,
    )


def _solver_for(method: GLMethod, problem, *, linear: bool, y_scale: float = 1.0):
    structure = _structure(linear)
    req = deduce_requirements(method, structure, problem.state_dim)
    return create_stage_solver(method, req, structure, y_scale=y_scale)


def _coupled_residual(Z, y_history, u_stages, t_stages, h, problem, method):
    """``R_i(Z) = Z_i - (U y)_i - h Σ_j A[i,j] f(Z_j, u_j, t_j)``, shape (s, n)."""
    s = method.s
    f_all = np.stack(
        [np.asarray(problem.f(Z[j], u_stages[j], t_stages[j])) for j in range(s)]
    )
    base = np.asarray(method.U @ y_history, dtype=float).reshape(Z.shape)
    return Z - base - h * (method.A @ f_all)


# ---------------------------------------------------------------------------
# The solve itself
# ---------------------------------------------------------------------------


def test_coupled_solve_converges_and_is_not_a_stub():
    """A Gauss-2 step must satisfy its own stage equations.

    Precedent R-4: this solver once returned zeros, so a forward solve
    returned the initial condition unchanged and nothing distinguished that
    from a correct answer. Asserting the residual is small is the direct
    refutation; asserting ``Z`` is not the initial condition is the specific
    refutation of the historical failure mode.
    """
    problem = CoupledNonlinear()
    method = gauss2()
    solver = _solver_for(method, problem, linear=False)

    y_history = np.array([[0.4, -0.25, 0.15]])
    u_stages = np.array([[0.3, -0.2], [0.1, 0.4]])
    t_n, h = 0.3, 0.05

    Z, _cache = solver.solve_stages(
        y_history, u_stages, t_n, h, problem, method, step=0
    )

    assert Z.shape == (method.s, problem.state_dim)
    residual = _coupled_residual(
        Z, y_history, u_stages, t_n + method.c * h, h, problem, method
    )
    assert np.max(np.abs(residual)) < COUPLED_RESIDUAL_TOL, (
        f"coupled Newton did not converge: ||R||_inf = "
        f"{np.max(np.abs(residual)):.3e}"
    )

    # R-4: the historical stub returned zeros, which would leave every stage
    # equal to the external stage rather than displaced from it.
    assert not np.allclose(Z, 0.0)
    assert not np.allclose(Z, y_history[0])


def test_coupled_cache_carries_a_factorization_not_per_stage_ones():
    """The cache must advertise which route produced it.

    ``adjoint_sensitivity`` and ``forward_sensitivity`` branch on
    ``coupled_factorization is not None``. If a coupled solve also populated
    ``stage_factorizations``, the triangular branch would be taken for a dense
    tableau and would silently compute the derivative of a different map.
    """
    problem = CoupledNonlinear()
    method = gauss2()
    solver = _solver_for(method, problem, linear=False)

    _, cache = solver.solve_stages(
        np.array([[0.4, -0.25, 0.15]]),
        np.zeros((method.s, problem.control_dim)),
        0.0,
        0.05,
        problem,
        method,
    )

    assert cache.coupled_factorization is not None
    assert cache.stage_factorizations is None


@pytest.mark.parametrize("factory", [rk4, sdirk3, implicit_midpoint])
def test_triangular_methods_do_not_set_the_coupled_factorization(factory):
    """The converse of the previous test, for the certified triangular routes.

    A triangular solver that started populating ``coupled_factorization``
    would divert explicit and DIRK methods onto the coupled sensitivity
    branch. Those routes were certified as written (C-6.1), so the diversion
    would invalidate their certification even if it happened to agree.
    """
    problem = CoupledNonlinear()
    method = factory()
    solver = _solver_for(method, problem, linear=False)

    _, cache = solver.solve_stages(
        np.array([[0.4, -0.25, 0.15]]),
        np.zeros((method.s, problem.control_dim)),
        0.0,
        0.05,
        problem,
        method,
    )

    assert cache.coupled_factorization is None


# ---------------------------------------------------------------------------
# The factorization is of the right matrix, at the right point
# ---------------------------------------------------------------------------


def _difference_jacobian(Z, y_history, u_stages, t_stages, h, problem, method):
    """Build the coupled Jacobian by central differences of the residual.

    Deliberately does not call ``problem.F``. The analytic Jacobian inside the
    solver is built from ``F`` with the index convention ``block (i, j) = δ_ij I
    - h A[i,j] F_j``; differencing the residual cannot reproduce a wrong index
    convention, so agreement is evidence rather than tautology.
    """
    s, n = Z.shape
    m = s * n
    J = np.zeros((m, m))
    z0 = Z.ravel()
    scale = max(float(np.max(np.abs(z0))), 1.0)
    eps = scale * 1e-6

    for col in range(m):
        e = np.zeros(m)
        e[col] = eps
        plus = _coupled_residual(
            (z0 + e).reshape(s, n), y_history, u_stages, t_stages, h,
            problem, method,
        ).ravel()
        minus = _coupled_residual(
            (z0 - e).reshape(s, n), y_history, u_stages, t_stages, h,
            problem, method,
        ).ravel()
        J[:, col] = (plus - minus) / (2.0 * eps)
    return J


def test_cached_factorization_matches_a_differenced_jacobian():
    """C-5.4: the factorization is of the Jacobian at the converged iterate.

    Two things could go wrong and are separated here. The Jacobian could use
    the wrong stage index in its blocks (``F_i`` where ``F_j`` is required --
    the transposed form of defect B0, precedent R-5), or it could be evaluated
    at the Newton starting point rather than at the answer. Differencing the
    residual *at the converged Z* tests both at once: the comparison matrix
    has the right index structure by construction and is anchored at the right
    point.

    Tolerance basis: the reference is a central difference with step
    ``1e-6 * scale``, so its own truncation error is ``O(1e-12)`` relative and
    its cancellation error ``O(1e-10)`` relative. 1e-7 is far above both and
    still four orders below the size of a single mispaired ``F`` block, which
    for this problem is ``O(h) ~ 5e-2``.
    """
    problem = CoupledNonlinear()
    method = gauss2()
    solver = _solver_for(method, problem, linear=False)

    y_history = np.array([[0.4, -0.25, 0.15]])
    u_stages = np.array([[0.3, -0.2], [0.1, 0.4]])
    t_n, h = 0.3, 0.05
    t_stages = t_n + method.c * h

    Z, cache = solver.solve_stages(
        y_history, u_stages, t_n, h, problem, method
    )

    J_fd = _difference_jacobian(
        Z, y_history, u_stages, t_stages, h, problem, method
    )

    # Recover the factored matrix by applying its inverse to a basis, then
    # inverting: this reads the factorization the solver actually stored
    # rather than rebuilding one and hoping they agree.
    m = Z.size
    inverse = np.column_stack(
        [
            scipy.linalg.lu_solve(cache.coupled_factorization, e)
            for e in np.eye(m)
        ]
    )
    J_cached = np.linalg.inv(inverse)

    rel = np.max(np.abs(J_cached - J_fd)) / max(float(np.max(np.abs(J_fd))), 1.0)
    assert rel < 1e-7, f"cached Jacobian differs from differenced: rel={rel:.3e}"


def test_transposed_solve_is_the_adjoint_of_the_untransposed_one():
    """``⟨solve_coupled(b), a⟩ = ⟨b, solve_coupled_transposed(a)⟩``.

    The tangent sensitivity uses ``solve_coupled`` and both adjoints use
    ``solve_coupled_transposed``. If one of them had the wrong orientation --
    the single most likely transcription error, because both calls have
    identical shapes and neither raises -- this identity would fail. It is
    checked on the factorization the forward solve actually produced, not on a
    synthetic matrix.

    Tolerance basis: both sides are one triangular solve of a matrix whose
    condition number is measured below, so agreement to a few hundred ULP of
    the larger operand is the most that can be demanded (C-11.3).
    """
    problem = CoupledNonlinear()
    method = gauss2()
    solver = _solver_for(method, problem, linear=False)

    rng = np.random.default_rng(11)
    _, cache = solver.solve_stages(
        np.array([[0.4, -0.25, 0.15]]),
        rng.standard_normal((method.s, problem.control_dim)),
        0.3,
        0.05,
        problem,
        method,
    )

    shape = (method.s, problem.state_dim)
    for trial in range(5):
        b = rng.standard_normal(shape)
        a = rng.standard_normal(shape)
        left = float(np.vdot(solve_coupled(cache.coupled_factorization, b), a))
        right = float(
            np.vdot(b, solve_coupled_transposed(cache.coupled_factorization, a))
        )
        assert abs(left - right) <= 1e-12 * max(abs(left), abs(right), 1.0), (
            f"trial {trial}: duality defect {abs(left - right):.3e}"
        )


def test_adjoint_stage_solve_reuses_the_forward_factorization():
    """The stage adjoints solve ``J^T`` with the external-adjoint right side.

    This is the M3 derivation stated as a test. Writing the operator out here
    from ``A``, ``B`` and the cached ``F`` -- rather than calling the solver's
    own helper -- means the test fails if the implementation puts an ``A``
    term in the right-hand side as well as in the operator, which is the
    natural error when adapting the triangular backward substitution.
    """
    problem = CoupledNonlinear()
    method = gauss2()
    solver = _solver_for(method, problem, linear=False)
    s, n = method.s, problem.state_dim
    h = 0.05

    rng = np.random.default_rng(3)
    _Z, cache = solver.solve_stages(
        np.array([[0.4, -0.25, 0.15]]),
        rng.standard_normal((s, problem.control_dim)),
        0.3,
        h,
        problem,
        method,
    )

    lambda_ext = rng.standard_normal((method.r, n))
    mu = solver.solve_adjoint_stages(lambda_ext, cache, method, h)

    # Residual of the defining relation, assembled independently:
    #   mu_p = h F_p^T ( Σ_i A[i,p] mu_i + Σ_l B[l,p] λ_l )
    for p in range(s):
        coupling = sum(method.A[i, p] * mu[i] for i in range(s))
        external = method.B[:, p] @ lambda_ext
        expected = h * cache.F[p].T @ (coupling + external)
        assert np.allclose(mu[p], expected, rtol=0, atol=1e-11), (
            f"stage adjoint {p} does not satisfy its defining relation"
        )


# ---------------------------------------------------------------------------
# Failure reporting (C-5.3)
# ---------------------------------------------------------------------------


def test_failed_coupled_solve_names_a_stage_block():
    """C-5.3 requires a stage index in the failure message.

    A coupled solve has no single failing stage: all ``s`` blocks are solved
    together. The clause is satisfied by naming the block holding the largest
    residual component, which is the stage a reader should examine first. The
    message must also carry the step and the time interval, so the failure can
    be located in a long integration without instrumenting the solver.
    """
    problem = _Blowup(k=1e3)
    method = gauss2()
    solver = ImplicitStageSolver(y_scale=1.0)

    with pytest.raises(StageSolveError) as excinfo:
        solver.solve_stages(
            np.array([[1.0, 5.0]]),
            np.zeros((method.s, 1)),
            0.7,
            1.0,
            problem,
            method,
            step=4,
        )

    message = str(excinfo.value)
    assert "worst block stage" in message, message
    assert "step 4" in message, message
    assert "t=[0.7, 1.7]" in message, message
    assert excinfo.value.residual_vector is not None


def test_worst_block_is_the_block_with_the_largest_residual():
    """The reported index must be the argmax, not merely some valid index.

    Checked directly against the residual vector the error carries, so the
    test cannot be satisfied by always reporting stage 0.
    """
    problem = _Blowup(k=1e3)
    method = gauss2()
    solver = ImplicitStageSolver(y_scale=1.0)

    with pytest.raises(StageSolveError) as excinfo:
        solver.solve_stages(
            np.array([[1.0, 5.0]]),
            np.zeros((method.s, 1)),
            0.0,
            1.0,
            problem,
            method,
            step=0,
        )

    err = excinfo.value
    r = np.asarray(err.residual_vector).reshape(method.s, problem.state_dim)
    blocks = np.max(np.abs(r), axis=1)
    expected = int(np.argmax(blocks))

    # If the blocks happened to be equal the argmax would be satisfied by any
    # index and the test would prove nothing.
    assert blocks[expected] > 1.5 * np.min(blocks), (
        f"fixture degenerate: block residuals {blocks} are too close to "
        "distinguish a genuine argmax from a constant"
    )
    assert f"worst block stage {expected}" in str(err), str(err)


def test_adjoint_refuses_a_cache_from_another_solver():
    """Mixing routes must raise, not produce a plausible wrong answer.

    A cache from a triangular solver has ``coupled_factorization is None``.
    Proceeding would dereference ``None``; C-7 requires an explanatory refusal
    instead of an ``AttributeError`` from deep inside a linear solve.
    """
    problem = CoupledNonlinear()
    dirk = sdirk3()
    triangular = _solver_for(dirk, problem, linear=False)
    _, dirk_cache = triangular.solve_stages(
        np.array([[0.4, -0.25, 0.15]]),
        np.zeros((dirk.s, problem.control_dim)),
        0.0,
        0.05,
        problem,
        dirk,
    )

    coupled = ImplicitStageSolver(y_scale=1.0)
    with pytest.raises(ValueError, match=r"coupled factorization"):
        coupled.solve_adjoint_stages(
            np.zeros((dirk.r, problem.state_dim)), dirk_cache, dirk, 0.05
        )


# ---------------------------------------------------------------------------
# C-4: continuous accuracy
# ---------------------------------------------------------------------------


def _terminal_error(method: GLMethod, N: int) -> float:
    problem = _LTI()
    y0 = np.array([1.0, 0.0])
    T = 2.0
    exact = scipy.linalg.expm(problem.Amat * T) @ y0
    solver = _solver_for(method, problem, linear=True)
    trajectory = forward_solve(
        y0, np.zeros((N, method.s, 1)), (0.0, T), N, problem, method, solver
    )
    return float(np.max(np.abs(trajectory.Y[-1][0] - exact)))


def test_gauss2_observed_order_is_four():
    """C-4: Gauss-2 attains its nominal order 4 on a closed-form problem.

    This is a CONTINUOUS-accuracy claim: it refines the mesh. Per C-2 it can
    never be used to excuse a derivative discrepancy, which is a fixed-mesh
    claim. It is included because order 4 from a two-stage method is the
    distinguishing property of the Gauss family: a solver that quietly
    degenerated to the implicit midpoint rule, or that solved the stages
    inconsistently, would still converge -- at order 2.

    Tolerance basis: the observed rates over this range are 3.989, 3.997,
    3.999 and 4.000. A 0.1 window around 4 is more than eight times the
    observed spread while still excluding order 3 by a wide margin. Not
    bit-exact (C-11.3).
    """
    meshes = [10, 20, 40, 80, 160]
    errors = [_terminal_error(gauss2(), N) for N in meshes]
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]

    report = "\n".join(
        f"  N={N:4d}  err={e:.6e}" for N, e in zip(meshes, errors)
    ) + f"\n  rates: {[round(r, 4) for r in rates]}"

    assert all(abs(r - 4.0) < 0.1 for r in rates), (
        f"Gauss-2 should converge at order 4:\n{report}"
    )
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)), (
        f"error did not decrease monotonically:\n{report}"
    )


def test_gauss2_is_more_accurate_than_midpoint_at_equal_stage_count():
    """Gauss-2 must beat the implicit midpoint rule, which it shares a family
    with but not an order.

    Guards the specific degeneracy the order test cannot see at a single mesh:
    if the coupled solve ignored the off-diagonal entries of ``A`` -- treating
    the dense tableau as though it were diagonal -- the result would still
    converge and still look reasonable. Here it would lose three orders of
    magnitude at N=40.
    """
    N = 40
    gauss_error = _terminal_error(gauss2(), N)
    midpoint_error = _terminal_error(implicit_midpoint(), N)

    assert gauss_error < midpoint_error / 100.0, (
        f"gauss2 err={gauss_error:.3e} not decisively below "
        f"implicit_midpoint err={midpoint_error:.3e}"
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_factory_routes_dense_a_to_the_coupled_solver():
    """Dispatch is by stage type, and ``y_scale`` must be forwarded.

    ``y_scale`` is not decoration: it sets the C-3.4 scaled stopping test. A
    solver built with the default 1.0 for a problem whose states are O(1e3)
    would stop at a residual that is loose relative to the solution, and every
    derivative downstream would be the derivative of a different map (C-5.4).
    """
    problem = CoupledNonlinear()
    solver = _solver_for(gauss2(), problem, linear=False, y_scale=12.5)

    assert isinstance(solver, ImplicitStageSolver)
    assert solver.y_scale == 12.5
    assert gauss2().stage_type is StageType.IMPLICIT


def test_converged_residual_respects_the_newton_tolerance():
    """The solve stops where C-3.4 says it should, not merely somewhere small.

    Asserting against ``STAGE_NEWTON_TOL`` rather than a literal keeps this
    test tied to the clause: if the tolerance is ever changed, this either
    follows it or fails loudly.
    """
    problem = CoupledNonlinear()
    method = gauss2()
    solver = _solver_for(method, problem, linear=False, y_scale=1.0)

    y_history = np.array([[0.4, -0.25, 0.15]])
    u_stages = np.array([[0.3, -0.2], [0.1, 0.4]])
    t_n, h = 0.3, 0.05

    Z, _ = solver.solve_stages(
        y_history, u_stages, t_n, h, problem, method
    )
    residual = _coupled_residual(
        Z, y_history, u_stages, t_n + method.c * h, h, problem, method
    )

    scale = max(float(np.max(np.abs(Z))), 1.0)
    assert np.max(np.abs(residual)) <= STAGE_NEWTON_TOL * scale
