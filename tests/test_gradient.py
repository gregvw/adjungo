"""Tests for gradient computation via finite differences."""

import numpy as np
import pytest

from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler, rk4


class SimpleProblem:
    """dy/dt = -y + u"""

    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return -y + u

    def F(self, y, u, t):
        return np.array([[-1.0]])

    def G(self, y, u, t):
        return np.array([[1.0]])


class SimpleObjective:
    """J = 0.5 * (y(T) - 1)^2 + 0.5 * sum(u^2)"""

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0, 0]
        return 0.5 * (y_final - 1.0) ** 2 + 0.5 * np.sum(u ** 2)

    def dJ_dy_terminal(self, y_final):
        return np.array([[y_final[0, 0] - 1.0]])

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return u_stage

    def d2J_du2(self, u_stage, step, stage):
        return np.eye(1)


def test_gradient_finite_difference_explicit_euler():
    """Test gradient via finite difference with explicit Euler."""
    problem = SimpleProblem()
    objective = SimpleObjective()
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=10,
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    u = np.random.randn(10, 1, 1) * 0.1
    grad_adjoint = optimizer.gradient(u)

    # Finite difference gradient
    eps = 1e-6
    grad_fd = np.zeros_like(u)
    J0 = optimizer.objective_value(u)

    for i in range(10):
        for j in range(1):
            for k in range(1):
                u_pert = u.copy()
                u_pert[i, j, k] += eps
                J_pert = optimizer.objective_value(u_pert)
                grad_fd[i, j, k] = (J_pert - J0) / eps

    # Should match within tolerance
    assert np.allclose(grad_adjoint, grad_fd, rtol=1e-4, atol=1e-6)


def test_gradient_finite_difference_rk4():
    """Test gradient via finite difference with RK4."""
    problem = SimpleProblem()
    objective = SimpleObjective()
    method = rk4()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=5,
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    u = np.random.randn(5, 4, 1) * 0.1
    grad_adjoint = optimizer.gradient(u)

    # Finite difference gradient
    eps = 1e-6
    grad_fd = np.zeros_like(u)
    J0 = optimizer.objective_value(u)

    for step in range(5):
        for stage in range(4):
            u_pert = u.copy()
            u_pert[step, stage, 0] += eps
            J_pert = optimizer.objective_value(u_pert)
            grad_fd[step, stage, 0] = (J_pert - J0) / eps

    assert np.allclose(grad_adjoint, grad_fd, rtol=1e-4, atol=1e-6)


def test_gradient_zero_for_optimal_control():
    """Test that gradient is zero at optimal control (if known)."""
    # For linear-quadratic problem with simple dynamics,
    # we can construct a case where u=0 is optimal

    class ZeroOptimalProblem:
        state_dim = 1
        control_dim = 1

        def f(self, y, u, t):
            return u  # dy/dt = u

        def F(self, y, u, t):
            return np.array([[0.0]])

        def G(self, y, u, t):
            return np.array([[1.0]])

    class ZeroOptimalObjective:
        """J = 0.5 * y(T)^2 + 0.5 * sum(u^2), with y(0)=0"""

        def evaluate(self, trajectory, u):
            y_final = trajectory.Y[-1, 0, 0]
            return 0.5 * y_final ** 2 + 0.5 * np.sum(u ** 2)

        def dJ_dy_terminal(self, y_final):
            return np.array([[y_final[0, 0]]])

        def dJ_dy(self, y, step):
            return np.zeros_like(y)

        def dJ_du(self, u_stage, step, stage):
            return u_stage

        def d2J_du2(self, u_stage, step, stage):
            return np.eye(1)

    problem = ZeroOptimalProblem()
    objective = ZeroOptimalObjective()
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=10,
        y0=np.array([0.0]),
    )

    u_zero = np.zeros((10, 1, 1))
    grad = optimizer.gradient(u_zero)

    # At u=0, gradient should be zero (optimal solution)
    # y(T) = 0 since y(0)=0 and u=0
    assert np.allclose(grad, 0.0, atol=1e-10)


def test_gradient_different_controls_different_gradients():
    """Test that different controls produce different gradients."""
    problem = SimpleProblem()
    objective = SimpleObjective()
    method = rk4()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=5,
        y0=np.array([0.0]),
    )

    u1 = np.ones((5, 4, 1)) * 0.1
    u2 = np.ones((5, 4, 1)) * 0.5

    grad1 = optimizer.gradient(u1)
    grad2 = optimizer.gradient(u2)

    # Gradients should be different for different controls
    assert not np.allclose(grad1, grad2)


def test_scipy_interface():
    """Test scipy interface for optimization."""
    problem = SimpleProblem()
    objective = SimpleObjective()
    method = explicit_euler()

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=5,
        y0=np.array([0.0]),
    )

    fun, jac = optimizer.scipy_interface()

    u = np.random.randn(5, 1, 1) * 0.1
    u_flat = u.ravel()

    # Test function interface
    J = fun(u_flat)
    assert isinstance(J, float)

    # Test jacobian interface
    grad_flat = jac(u_flat)
    assert grad_flat.shape == u_flat.shape

    # Should match direct calls
    J_direct = optimizer.objective_value(u)
    grad_direct = optimizer.gradient(u)

    assert np.isclose(J, J_direct)
    assert np.allclose(grad_flat, grad_direct.ravel())


def test_gradient_shape_consistency():
    """Test that gradient has same shape as control."""
    problem = SimpleProblem()
    objective = SimpleObjective()

    for method_func, s in [(explicit_euler, 1), (rk4, 4)]:
        method = method_func()
        N = 8

        optimizer = GLMOptimizer(
            problem=problem,
            objective=objective,
            method=method,
            t_span=(0.0, 1.0),
            N=N,
            y0=np.array([0.0]),
        )

        u = np.random.randn(N, s, 1) * 0.1
        grad = optimizer.gradient(u)

        assert grad.shape == u.shape, f"Method {method_func.__name__}: shapes don't match"
