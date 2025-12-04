"""Test Linear Time-Invariant Control with Crank-Nicolson.

This serves as a stepping stone to multistep methods:
- Linear problem: f(y,u,t) = Ay + Bu
- Time-invariant: A, B constant
- Crank-Nicolson: Implicit trapezoidal rule (2nd order)
- Analytical solutions available for validation
"""

import numpy as np
import pytest
from scipy.linalg import expm

from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.core.method import GLMethod, StageType
from adjungo.methods.runge_kutta import implicit_trapezoid


class LTICProblem:
    """Linear Time-Invariant Control: dy/dt = Ay + Bu."""

    def __init__(self, A, B):
        """
        Args:
            A: State matrix (n, n)
            B: Control matrix (n, m)
        """
        self.A = A
        self.B = B
        self.state_dim = A.shape[0]
        self.control_dim = B.shape[1]

    def f(self, y, u, t):
        """Dynamics: dy/dt = Ay + Bu."""
        return self.A @ y + self.B @ u

    def F(self, y, u, t):
        """Jacobian w.r.t. state: ∂f/∂y = A."""
        return self.A

    def G(self, y, u, t):
        """Jacobian w.r.t. control: ∂f/∂u = B."""
        return self.B


class QuadraticCost:
    """Quadratic objective: J = 0.5 * (y(T) - y_target)^T Q (y(T) - y_target) + 0.5 * sum(u^T R u)."""

    def __init__(self, y_target, Q=None, R=None):
        """
        Args:
            y_target: Target final state
            Q: Terminal state weight matrix (default: identity)
            R: Control weight matrix (default: identity)
        """
        self.y_target = y_target
        self.n = len(y_target)
        self.Q = Q if Q is not None else np.eye(self.n)
        self.R = R  # Will be set based on control dimension

    def evaluate(self, trajectory, u):
        """Evaluate J = 0.5 * e^T Q e + 0.5 * sum(u^T R u)."""
        y_final = trajectory.Y[-1, 0]
        e = y_final - self.y_target
        terminal_cost = 0.5 * e.T @ self.Q @ e

        if self.R is None:
            self.R = np.eye(u.shape[-1])
        control_cost = 0.5 * np.sum([u_step.T @ self.R @ u_step for u_step in u.reshape(-1, u.shape[-1])])

        return terminal_cost + control_cost

    def dJ_dy_terminal(self, y_final):
        """Terminal gradient: ∂J/∂y(T) = Q * (y(T) - y_target)."""
        e = y_final[0] - self.y_target
        return np.array([self.Q @ e])

    def dJ_dy(self, y, step):
        """No running state cost."""
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        """Control gradient: ∂J/∂u = R * u."""
        if self.R is None:
            self.R = np.eye(len(u_stage))
        return self.R @ u_stage

    def d2J_du2(self, u_stage, step, stage):
        """Control Hessian: ∂²J/∂u² = R."""
        if self.R is None:
            self.R = np.eye(len(u_stage))
        return self.R


def analytical_solution_lti(A, B, y0, u_func, t_span, N):
    """
    Analytical solution for LTI system: dy/dt = Ay + Bu.

    Solution: y(t) = e^(A*t) * y0 + ∫₀ᵗ e^(A*(t-τ)) * B * u(τ) dτ

    For piecewise constant control u_k on [t_k, t_{k+1}]:
    y(t_{k+1}) = e^(A*h) * y_k + (e^(A*h) - I) * A^(-1) * B * u_k

    If A is singular, use: ∫₀ʰ e^(A*τ) dτ ≈ h*I for small eigenvalues.
    """
    h = (t_span[1] - t_span[0]) / N
    y = y0.copy()
    trajectory = [y0]

    # Matrix exponential
    eAh = expm(A * h)

    # Compute ∫₀ʰ e^(A*τ) dτ
    # For general case, use: (e^(A*h) - I) * A^(-1)
    # But if A is nearly singular, use series expansion
    try:
        A_inv_B = np.linalg.solve(A, B)
        integral_term = (eAh - np.eye(len(A))) @ A_inv_B
    except np.linalg.LinAlgError:
        # A is singular, use ∫₀ʰ e^(A*τ) dτ ≈ h*I for zero eigenvalues
        integral_term = h * B

    for k in range(N):
        t = t_span[0] + k * h
        u_k = u_func(t, k, 0)
        y = eAh @ y + integral_term @ u_k
        trajectory.append(y)

    return np.array(trajectory)


def test_crank_nicolson_lti_forward_solve():
    """Test Crank-Nicolson forward solve against analytical solution."""
    # Simple 2D system: harmonic oscillator with damping
    # dy/dt = [0, 1; -1, -0.1] * y + [0; 1] * u
    A = np.array([[0.0, 1.0], [-1.0, -0.1]])
    B = np.array([[0.0], [1.0]])

    problem = LTICProblem(A, B)
    method = implicit_trapezoid()  # Crank-Nicolson

    y0 = np.array([1.0, 0.0])
    t_span = (0.0, 2.0)
    N = 20

    # Zero control
    u_zero = lambda t, step, stage: np.array([0.0])

    # Analytical solution
    y_analytical = analytical_solution_lti(A, B, y0, u_zero, t_span, N)

    # Numerical solution with Crank-Nicolson
    optimizer = GLMOptimizer(
        problem=problem,
        objective=QuadraticCost(y_target=np.zeros(2)),
        method=method,
        t_span=t_span,
        N=N,
        y0=y0,
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    # Use zero control
    u = np.zeros((N, method.s, 1))
    optimizer._ensure_forward(u)
    y_numerical = optimizer._trajectory.Y[:, 0, :]

    # Should match to high accuracy (Crank-Nicolson is 2nd order)
    assert np.allclose(y_numerical, y_analytical, rtol=1e-4, atol=1e-6)


def test_crank_nicolson_lti_gradient_validation():
    """Test adjoint gradient against finite differences for LTIC."""
    # Simple scalar system for easy debugging
    A = np.array([[-0.5]])
    B = np.array([[1.0]])

    problem = LTICProblem(A, B)
    y_target = np.array([1.0])
    objective = QuadraticCost(y_target, Q=np.array([[1.0]]), R=np.array([[0.1]]))

    method = implicit_trapezoid()
    N = 10
    t_span = (0.0, 1.0)
    y0 = np.array([0.0])

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=t_span,
        N=N,
        y0=y0,
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    # Random control
    np.random.seed(42)
    u = np.random.randn(N, method.s, 1) * 0.1

    # Adjoint gradient
    grad_adjoint = optimizer.gradient(u)

    # Finite difference gradient
    eps = 1e-6
    grad_fd = np.zeros_like(u)
    J0 = optimizer.objective_value(u)

    for step in range(N):
        for stage in range(method.s):
            u_pert = u.copy()
            u_pert[step, stage, 0] += eps
            J_pert = optimizer.objective_value(u_pert)
            grad_fd[step, stage, 0] = (J_pert - J0) / eps

    # Should match to high accuracy
    assert np.allclose(grad_adjoint, grad_fd, rtol=1e-4, atol=1e-6), \
        f"Gradient mismatch:\nAdjoint: {grad_adjoint.ravel()}\nFD: {grad_fd.ravel()}"


def test_crank_nicolson_lti_optimal_control():
    """Test that gradient descent improves objective for LTIC."""
    # 2D oscillator: drive to target state
    A = np.array([[0.0, 1.0], [-1.0, -0.1]])
    B = np.array([[0.0], [1.0]])

    problem = LTICProblem(A, B)
    y_target = np.array([1.0, 0.0])
    objective = QuadraticCost(y_target, Q=10.0 * np.eye(2), R=0.1 * np.eye(1))

    method = implicit_trapezoid()
    N = 30
    t_span = (0.0, 3.0)
    y0 = np.array([0.0, 0.0])

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=t_span,
        N=N,
        y0=y0,
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    # Start with zero control
    u = np.zeros((N, method.s, 1))
    J_history = []

    # Gradient descent
    for iteration in range(20):
        J = optimizer.objective_value(u)
        grad = optimizer.gradient(u)
        J_history.append(J)

        # Simple line search
        alpha = 0.1
        u_new = u - alpha * grad
        J_new = optimizer.objective_value(u_new)

        if J_new < J:
            u = u_new

    # Should reduce objective significantly
    assert J_history[-1] < 0.5 * J_history[0], \
        f"Optimization failed to improve: J0={J_history[0]:.6f}, Jf={J_history[-1]:.6f}"

    # Final state should be close to target
    optimizer._ensure_forward(u)
    y_final = optimizer._trajectory.Y[-1, 0]
    error = np.linalg.norm(y_final - y_target)

    # With good control, should get reasonably close
    assert error < 0.5, f"Final error too large: {error:.6f}"


def test_lti_energy_conservation():
    """Test energy relationships for conservative LTI system."""
    # Conservative system (skew-symmetric A): energy should be conserved with zero control
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])  # Harmonic oscillator
    B = np.array([[0.0], [1.0]])

    problem = LTICProblem(A, B)
    method = implicit_trapezoid()

    y0 = np.array([1.0, 0.0])
    t_span = (0.0, 10.0)
    N = 100

    optimizer = GLMOptimizer(
        problem=problem,
        objective=QuadraticCost(y_target=np.zeros(2)),
        method=method,
        t_span=t_span,
        N=N,
        y0=y0,
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    # Zero control - energy should be conserved
    u = np.zeros((N, method.s, 1))
    optimizer._ensure_forward(u)

    # Compute energy at each step: E = 0.5 * y^T * y
    trajectory = optimizer._trajectory.Y[:, 0, :]
    energy = 0.5 * np.sum(trajectory ** 2, axis=1)

    # For conservative system with Crank-Nicolson, energy should be conserved
    # (up to discretization error)
    energy_deviation = np.std(energy) / np.mean(energy)

    assert energy_deviation < 0.01, \
        f"Energy not conserved: std/mean = {energy_deviation:.6f}"


def test_lti_controllability_matrix():
    """Test that controllable system can reach target states."""
    # Controllable system
    A = np.array([[0.0, 1.0], [0.0, 0.0]])  # Double integrator
    B = np.array([[0.0], [1.0]])

    # Check controllability matrix rank
    C = np.hstack([B, A @ B])
    rank = np.linalg.matrix_rank(C)
    assert rank == 2, "System should be controllable"

    problem = LTICProblem(A, B)

    # Try to reach different target states
    targets = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    ]

    method = implicit_trapezoid()
    N = 20
    t_span = (0.0, 2.0)
    y0 = np.array([0.0, 0.0])

    for y_target in targets:
        objective = QuadraticCost(y_target, Q=100.0 * np.eye(2), R=0.01 * np.eye(1))

        optimizer = GLMOptimizer(
            problem=problem,
            objective=objective,
            method=method,
            t_span=t_span,
            N=N,
            y0=y0,
            problem_structure=ProblemStructure(
                linearity=Linearity.LINEAR,
                jacobian_constant=True,
                jacobian_control_dependent=False,
                has_second_derivatives=False,
            ),
        )

        # Optimize
        u = np.zeros((N, method.s, 1))
        for _ in range(30):
            grad = optimizer.gradient(u)
            u = u - 0.1 * grad

        # Check final state
        optimizer._ensure_forward(u)
        y_final = optimizer._trajectory.Y[-1, 0]
        error = np.linalg.norm(y_final - y_target)

        assert error < 0.2, \
            f"Failed to reach target {y_target}: error = {error:.6f}"


if __name__ == "__main__":
    # Run tests
    test_crank_nicolson_lti_forward_solve()
    print("✅ Forward solve matches analytical solution")

    test_crank_nicolson_lti_gradient_validation()
    print("✅ Gradients match finite differences")

    test_crank_nicolson_lti_optimal_control()
    print("✅ Gradient descent improves objective")

    test_lti_energy_conservation()
    print("✅ Energy conserved for conservative system")

    test_lti_controllability_matrix()
    print("✅ Controllable system reaches targets")

    print("\n🎉 All LTIC Crank-Nicolson tests pass!")
