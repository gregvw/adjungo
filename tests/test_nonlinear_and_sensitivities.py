"""Tests for mildly nonlinear problems and sensitivity equations.

Tests:
1. Nonlinear Crank-Nicolson (needs Newton iteration)
2. Forward sensitivity: δy from δu
3. Adjoint sensitivity: δλ from δy
4. Hessian-vector products
"""

import numpy as np

from adjungo.core.problem import Linearity, ProblemStructure
from adjungo.methods.runge_kutta import explicit_euler, implicit_trapezoid
from adjungo.optimization.interface import GLMOptimizer
from adjungo.stepping.sensitivity import adjoint_sensitivity, forward_sensitivity
from adjungo.validation import reference_gradient, reference_solve


class MildlyNonlinearProblem:
    """
    Mildly nonlinear problem: dy/dt = -y + u - 0.1*y^3

    The cubic term provides mild nonlinearity but keeps the problem stable.
    Linearization: F = -1 - 0.3*y^2
    """

    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        """Dynamics: dy/dt = -y + u - 0.1*y^3."""
        return -y + u - 0.1 * y**3

    def F(self, y, u, t):
        """Jacobian: ∂f/∂y = -1 - 0.3*y^2."""
        return np.array([[-1.0 - 0.3 * y[0]**2]])

    def G(self, y, u, t):
        """Control Jacobian: ∂f/∂u = 1."""
        return np.array([[1.0]])

    def F_yy_action(self, y, u, t, v):
        """Second derivative: ∂²f/∂y² [v] = -0.6*y*v."""
        return np.array([[-0.6 * y[0] * v[0]]])

    def F_yu_action(self, y, u, t, v):
        """Mixed derivative: sum_l v_l d2 f_l / dy du = 0, shape (n, nu)."""
        return np.zeros((1, 1))

    def F_uu_action(self, y, u, t, v):
        """Second derivative: sum_l v_l d2 f_l / du du = 0, shape (nu, nu)."""
        return np.zeros((1, 1))


class QuadraticDragProblem:
    """
    Quadratic drag: dy/dt = -0.1*y*|y| + u

    Common in fluid dynamics, provides smooth nonlinearity.
    Linearization: F = -0.2*|y|
    """

    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        """Dynamics: dy/dt = -0.1*y*|y| + u."""
        return -0.1 * y * np.abs(y) + u

    def F(self, y, u, t):
        """Jacobian: ∂f/∂y = -0.2*|y|."""
        return np.array([[-0.2 * np.abs(y[0])]])

    def G(self, y, u, t):
        """Control Jacobian: ∂f/∂u = 1."""
        return np.array([[1.0]])


class SimpleObjective:
    """J = 0.5 * (y(T) - y_target)^2 + 0.5 * R * sum(u^2)."""

    def __init__(self, y_target, R=0.1):
        self.y_target = y_target
        self.R = R

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0, 0]
        return 0.5 * (y_final - self.y_target) ** 2 + 0.5 * self.R * np.sum(u ** 2)

    def dJ_dy_terminal(self, y_final):
        return np.array([[y_final[0, 0] - self.y_target]])

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return self.R * u_stage

    def d2J_du2(self, u_stage, step, stage):
        return self.R * np.eye(len(u_stage))

    def d2J_dy2(self, y, step):
        """No running state cost, so the running state Hessian vanishes."""
        return np.zeros((1, 1))

    def d2J_dy2_terminal(self, y_final):
        """Terminal cost is 0.5*(y - target)^2, so J_yy = 1."""
        return np.ones((1, 1))


def test_mildly_nonlinear_crank_nicolson():
    """Crank-Nicolson on a nonlinear problem, forward and gradient.

    Unskipped by unit U-M2.2: the DIRK/SDIRK solvers now Newton-solve the
    true stage equation instead of linearizing it.

    The original version of this test used ``u = 0`` from ``y0 = 0``, for
    which the exact solution is identically zero, so ``abs(y_final) < 0.1``
    held no matter what the solver did. It now uses a nonzero control and
    checks the gradient against the independent monolithic reference
    (NUMERICS.md C-14.1 item 1).
    """
    problem = MildlyNonlinearProblem()
    objective = SimpleObjective(y_target=1.0, R=0.1)
    method = implicit_trapezoid()
    t_span = (0.0, 2.0)
    N = 20
    y0 = np.array([0.0])

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=t_span,
        N=N,
        y0=y0,
        problem_structure=ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
        ),
    )

    rng = np.random.default_rng(17)
    u = 0.6 + 0.2 * rng.standard_normal((N, method.s, 1))

    optimizer._ensure_forward(u)
    traj = optimizer._trajectory
    ref = reference_solve(y0, u, t_span, N, problem, method)

    assert np.all(np.isfinite(traj.Y))
    assert np.max(np.abs(traj.Y[-1])) > 1e-3, "degenerate: trajectory stayed at 0"
    assert np.max(np.abs(traj.Y - ref.Y)) < 1e-10, (
        "Crank-Nicolson forward solve disagrees with the monolithic residual; "
        "the stage equations are not being solved as written"
    )

    grad = optimizer.gradient(u)
    grad_ref = reference_gradient(
        y0, u, t_span, N, problem, method, objective
    )
    scale = max(float(np.max(np.abs(grad_ref))), 1.0)
    assert float(np.max(np.abs(grad - grad_ref))) / scale < 1e-10


def test_mildly_nonlinear_explicit_euler():
    """Test that explicit Euler works with mild nonlinearity."""
    problem = MildlyNonlinearProblem()
    objective = SimpleObjective(y_target=1.0, R=0.1)
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 2.0),
        N=100,  # More steps for stability
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
        ),
    )

    # Small control to reach target
    u = np.ones((100, 1, 1)) * 0.5

    # Forward solve should work
    optimizer._ensure_forward(u)
    y_final = optimizer._trajectory.Y[-1, 0, 0]

    # Should be able to reach near target with appropriate control
    assert abs(y_final) < 2.0  # Reasonable bound

    # Gradient validation via finite differences
    grad_adjoint = optimizer.gradient(u)

    eps = 1e-6
    J0 = optimizer.objective_value(u)
    u_pert = u.copy()
    u_pert[50, 0, 0] += eps
    J_pert = optimizer.objective_value(u_pert)
    grad_fd_50 = (J_pert - J0) / eps

    # Should match for nonlinear problem too
    assert np.isclose(grad_adjoint[50, 0, 0], grad_fd_50, rtol=1e-3, atol=1e-5), \
        f"Adjoint: {grad_adjoint[50, 0, 0]:.6f}, FD: {grad_fd_50:.6f}"


def test_quadratic_drag_explicit():
    """Test quadratic drag with explicit method."""
    problem = QuadraticDragProblem()
    objective = SimpleObjective(y_target=2.0, R=0.01)
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 5.0),
        N=100,
        y0=np.array([0.0]),
    )

    # Control to reach target
    u = np.ones((100, 1, 1)) * 0.5

    # Optimize
    for _ in range(20):
        grad = optimizer.gradient(u)
        u = u - 0.05 * grad

    # Check convergence
    optimizer._ensure_forward(u)
    y_final = optimizer._trajectory.Y[-1, 0, 0]

    # Should get reasonably close to target
    error = abs(y_final - 2.0)
    assert error < 0.5, f"Final error: {error:.6f}"


def test_forward_sensitivity_finite_difference():
    """Test forward sensitivity δy against finite differences."""
    problem = MildlyNonlinearProblem()
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=SimpleObjective(y_target=1.0),
        method=method,
        t_span=(0.0, 1.0),
        N=20,
        y0=np.array([0.0]),
    )

    # Baseline control
    u = np.ones((20, 1, 1)) * 0.3
    optimizer._ensure_forward(u)
    trajectory = optimizer._trajectory

    # Perturbation direction
    delta_u = np.zeros((20, 1, 1))
    delta_u[10, 0, 0] = 1.0  # Pulse at step 10

    # Forward sensitivity: δy from δu
    sens = forward_sensitivity(
        trajectory, delta_u, optimizer.stage_solver,
        optimizer.problem)

    # Finite difference validation
    eps = 1e-6
    u_pert = u + eps * delta_u
    optimizer._ensure_forward(u_pert)
    y_pert = optimizer._trajectory.Y

    delta_y_fd = (y_pert - trajectory.Y) / eps

    # Should match
    # Note: sensitivity gives δy at final time
    assert np.allclose(sens.delta_Y[-1], delta_y_fd[-1], rtol=1e-3, atol=1e-5), \
        f"Sensitivity: {sens.delta_Y[-1]}, FD: {delta_y_fd[-1]}"


def test_adjoint_sensitivity_finite_difference():
    """Test adjoint sensitivity δλ against finite differences.

    Cured by unit U-M1.3. Previously ``sensitivity.py`` set
    ``delta_Lambda[N] = 0`` where the derivation requires
    ``J_yy^terminal δy^[N]``, so δλ recovered only about 3% of its true
    magnitude and the error showed an ε-independent plateau at 3.14e-2.

    Before this test was made real it asserted only ``is not None`` on two
    dataclass fields that are unconditionally assigned arrays, so it could never
    fail and concealed the defect entirely.
    """
    problem = MildlyNonlinearProblem()
    objective = SimpleObjective(y_target=1.0)
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=20,
        y0=np.array([0.0]),
    )

    # Baseline
    u = np.ones((20, 1, 1)) * 0.3
    optimizer._ensure_adjoint(u)
    trajectory = optimizer._trajectory
    adjoint = optimizer._adjoint

    # State perturbation direction
    delta_u = np.zeros((20, 1, 1))
    delta_u[10, 0, 0] = 1.0

    # Forward sensitivity to get δy

    sens = forward_sensitivity(
        trajectory, delta_u, optimizer.stage_solver,
        optimizer.problem)

    # Adjoint sensitivity: δλ from δy
    adj_sens = adjoint_sensitivity(
        trajectory, adjoint, sens, u, delta_u,
        optimizer.stage_solver, optimizer.problem, optimizer.objective)

    # Central difference on the adjoint, at a FIXED mesh (NUMERICS.md C-2).
    # Fresh optimizers are built per perturbation rather than poking the private
    # cache key, so this test does not depend on cache-invalidation internals.
    def _adjoint_at(u_eval):
        opt = GLMOptimizer(
            problem=MildlyNonlinearProblem(),
            objective=SimpleObjective(y_target=1.0),
            method=explicit_euler(),
            t_span=(0.0, 1.0),
            N=20,
            y0=np.array([0.0]),
        )
        opt._ensure_adjoint(u_eval)
        return opt._adjoint.Lambda.copy()

    eps = 1e-5
    delta_lambda_fd = (
        _adjoint_at(u + eps * delta_u) - _adjoint_at(u - eps * delta_u)
    ) / (2 * eps)

    scale = max(np.max(np.abs(delta_lambda_fd)), 1.0)
    err = np.max(np.abs(adj_sens.delta_Lambda - delta_lambda_fd))

    # Tolerance basis (C-3.2): central difference on a smooth reduced map at
    # eps=1e-5 carries truncation O(eps^2)~1e-10 and cancellation
    # O(eps_mach*|L|/eps)~1e-11, so 1e-6 relative is loose by several decades
    # and any failure is a genuine defect rather than differencing noise.
    assert err / scale < 1e-6, (
        f"adjoint sensitivity delta_Lambda disagrees with central differences: "
        f"max abs error {err:.6e}, reference scale {scale:.6e}. "
        f"delta_Lambda[N] = {adj_sens.delta_Lambda[-1]} "
        f"(a nonzero terminal term J_yy*delta_y is required)."
    )


def test_hessian_vector_product_finite_difference():
    """Test Hessian-vector product [∇²J]v against central differences.

    Cured by unit U-M1.3. The operator previously omitted the terminal and
    running ``J_yy δy`` terms and contracted the problem's second-derivative
    callbacks over the wrong tensor index; finite differences of the gradient
    plateaued at 3.82152e-3 independently of ε. The dense operator was
    symmetric to 8.7e-19 throughout, which is why symmetry is corroboration
    and never a correctness argument (NUMERICS.md precedent R-3).

    Central differences are used rather than the one-sided quotient the
    original test used: a forward difference carries O(ε) truncation, which at
    ε=1e-5 is the same order as the defect it was supposed to detect.
    """
    problem = MildlyNonlinearProblem()
    objective = SimpleObjective(y_target=1.0, R=0.1)
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=10,
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=False,
            has_second_derivatives=True,  # Required for Hessian
        ),
    )

    u = np.ones((10, 1, 1)) * 0.5
    v = np.random.default_rng(4).standard_normal((10, 1, 1)) * 0.1

    Hv = optimizer.hessian_vector_product(u, v)

    # Central difference of the exact gradient. Truncation is O(eps^2)~1e-10
    # and cancellation O(eps_mach*|g|/eps)~1e-11, so 1e-6 relative is loose by
    # several decades and any failure is a genuine defect.
    eps = 1e-5
    Hv_fd = (
        optimizer.gradient(u + eps * v) - optimizer.gradient(u - eps * v)
    ) / (2 * eps)

    scale = max(float(np.max(np.abs(Hv_fd))), 1.0)
    err = float(np.max(np.abs(Hv - Hv_fd)))
    assert err / scale < 1e-6, (
        f"Hessian-vector product disagrees with central differences of the "
        f"gradient: max abs error {err:.6e}, reference scale {scale:.6e}."
    )


def test_gradient_nonlinear_vs_linear():
    """Compare gradient computation for linear vs. mildly nonlinear."""
    # Test that nonlinear adjoint reduces to linear adjoint for small deviations

    # Linear problem
    class LinearProblem:
        state_dim = 1
        control_dim = 1

        def f(self, y, u, t):
            return -y + u

        def F(self, y, u, t):
            return np.array([[-1.0]])

        def G(self, y, u, t):
            return np.array([[1.0]])

    # Near-zero control (nonlinear term ~0)
    u = np.random.randn(20, 1, 1) * 0.01  # Very small

    method = explicit_euler()
    objective = SimpleObjective(y_target=0.5, R=0.1)

    # Linear optimizer
    opt_linear = GLMOptimizer(
        problem=LinearProblem(),
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=20,
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    # Nonlinear optimizer (with tiny cubic term)
    opt_nonlinear = GLMOptimizer(
        problem=MildlyNonlinearProblem(),
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=20,
        y0=np.array([0.0]),
    )

    # Gradients should be nearly identical for small states
    grad_linear = opt_linear.gradient(u)
    grad_nonlinear = opt_nonlinear.gradient(u)

    # Very close match expected
    assert np.allclose(grad_linear, grad_nonlinear, rtol=1e-2, atol=1e-4), \
        f"Max diff: {np.max(np.abs(grad_linear - grad_nonlinear)):.6e}"
