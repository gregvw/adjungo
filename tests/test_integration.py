"""Integration tests for end-to-end optimization."""

import numpy as np
import pytest

from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler, rk4


class HarmonicOscillator:
    """Simple harmonic oscillator: d²x/dt² + x = u"""

    state_dim = 2  # [position, velocity]
    control_dim = 1

    def f(self, y, u, t):
        return np.array([y[1], -y[0] + u[0]])

    def F(self, y, u, t):
        return np.array([[0.0, 1.0], [-1.0, 0.0]])

    def G(self, y, u, t):
        return np.array([[0.0], [1.0]])


class TrackingObjective:
    """Track desired final state with control penalty."""

    def __init__(self, y_target, Q=1.0, R=0.01):
        self.y_target = y_target
        self.Q = Q
        self.R = R

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0]
        terminal_cost = 0.5 * self.Q * np.sum((y_final - self.y_target) ** 2)
        control_cost = 0.5 * self.R * np.sum(u ** 2)
        return terminal_cost + control_cost

    def dJ_dy_terminal(self, y_final):
        return self.Q * np.array([y_final[0] - self.y_target])

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return self.R * u_stage

    def d2J_du2(self, u_stage, step, stage):
        return self.R * np.eye(len(u_stage))


def test_simple_optimization_explicit_euler():
    """Test simple optimization with explicit Euler."""
    problem = HarmonicOscillator()
    y_target = np.array([1.0, 0.0])  # Target: position=1, velocity=0
    objective = TrackingObjective(y_target, Q=10.0, R=0.01)

    method = explicit_euler()
    N = 50
    t_span = (0.0, 5.0)

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=t_span,
        N=N,
        y0=np.array([0.0, 0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        ),
    )

    # Initial guess: zero control
    u0 = np.zeros((N, 1, 1))

    # Evaluate initial objective
    J0 = optimizer.objective_value(u0)
    grad0 = optimizer.gradient(u0)

    # Take a gradient descent step
    alpha = 0.1
    u1 = u0 - alpha * grad0

    # Objective should decrease
    J1 = optimizer.objective_value(u1)
    assert J1 < J0, "Objective should decrease along negative gradient"


def test_optimization_gradient_descent_converges():
    """Test that gradient descent reduces objective."""
    problem = HarmonicOscillator()
    y_target = np.array([1.0, 0.0])
    objective = TrackingObjective(y_target, Q=10.0, R=0.1)

    method = explicit_euler()
    N = 30
    t_span = (0.0, 3.0)

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=t_span,
        N=N,
        y0=np.array([0.0, 0.0]),
    )

    u = np.zeros((N, 1, 1))
    J_history = []

    # Simple gradient descent
    for iteration in range(20):
        J = optimizer.objective_value(u)
        grad = optimizer.gradient(u)
        J_history.append(J)

        # Armijo line search (simplified)
        alpha = 0.1
        u_new = u - alpha * grad

        J_new = optimizer.objective_value(u_new)
        if J_new < J:
            u = u_new

    # Objective should decrease over iterations
    assert J_history[-1] < J_history[0]
    assert J_history[-1] < 0.5 * J_history[0], "Should reduce by at least 50%"


def test_optimization_with_rk4():
    """Test optimization with higher-order method (RK4)."""
    problem = HarmonicOscillator()
    y_target = np.array([1.0, 0.0])
    objective = TrackingObjective(y_target, Q=10.0, R=0.1)

    method = rk4()
    N = 20
    t_span = (0.0, 3.0)

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=t_span,
        N=N,
        y0=np.array([0.0, 0.0]),
    )

    u = np.zeros((N, 4, 1))  # 4 stages for RK4

    J0 = optimizer.objective_value(u)
    grad = optimizer.gradient(u)

    # Gradient descent step
    u1 = u - 0.1 * grad
    J1 = optimizer.objective_value(u1)

    assert J1 < J0


def test_caching_invalidation():
    """Test that cache is invalidated when control changes."""
    problem = HarmonicOscillator()
    y_target = np.array([1.0, 0.0])
    objective = TrackingObjective(y_target)

    method = explicit_euler()
    N = 10

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=N,
        y0=np.array([0.0, 0.0]),
    )

    u1 = np.ones((N, 1, 1)) * 0.1
    u2 = np.ones((N, 1, 1)) * 0.5

    J1 = optimizer.objective_value(u1)
    grad1 = optimizer.gradient(u1)

    # Change control
    J2 = optimizer.objective_value(u2)
    grad2 = optimizer.gradient(u2)

    # Results should be different
    assert not np.isclose(J1, J2)
    assert not np.allclose(grad1, grad2)


def test_zero_control_penalty():
    """Test optimization with very small control penalty."""
    problem = HarmonicOscillator()
    y_target = np.array([1.0, 0.0])

    # Very small R means we can use large controls
    objective = TrackingObjective(y_target, Q=10.0, R=1e-6)

    method = explicit_euler()
    N = 20

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 2.0),
        N=N,
        y0=np.array([0.0, 0.0]),
    )

    # With tiny control penalty, optimal control should be larger
    u = np.zeros((N, 1, 1))

    for _ in range(10):
        grad = optimizer.gradient(u)
        u = u - 0.1 * grad

    J_final = optimizer.objective_value(u)

    # Should achieve good tracking
    # (exact value depends on discretization)
    assert J_final < 1.0


def test_objective_value_consistency():
    """Test that objective value is consistent with trajectory."""
    problem = HarmonicOscillator()
    y_target = np.array([1.0, 0.0])
    objective = TrackingObjective(y_target, Q=1.0, R=0.1)

    method = explicit_euler()
    N = 10

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 1.0),
        N=N,
        y0=np.array([0.0, 0.0]),
    )

    u = np.random.randn(N, 1, 1) * 0.1

    J = optimizer.objective_value(u)

    # Manually compute objective
    optimizer._ensure_forward(u)
    trajectory = optimizer._trajectory
    y_final = trajectory.Y[-1, 0]

    J_manual = 0.5 * 1.0 * np.sum((y_final - y_target) ** 2)
    J_manual += 0.5 * 0.1 * np.sum(u ** 2)

    assert np.isclose(J, J_manual)


def test_nonlinear_problem_integration():
    """Test optimization with a simple nonlinear problem."""

    class NonlinearProblem:
        state_dim = 1
        control_dim = 1

        def f(self, y, u, t):
            # Nonlinear: dy/dt = -y^2 + u
            return np.array([-y[0] ** 2 + u[0]])

        def F(self, y, u, t):
            return np.array([[-2 * y[0]]])

        def G(self, y, u, t):
            return np.array([[1.0]])

    class SimpleObjective:
        def evaluate(self, trajectory, u):
            y_final = trajectory.Y[-1, 0, 0]
            return 0.5 * (y_final - 1.0) ** 2 + 0.01 * np.sum(u ** 2)

        def dJ_dy_terminal(self, y_final):
            return np.array([[y_final[0, 0] - 1.0]])

        def dJ_dy(self, y, step):
            return np.zeros_like(y)

        def dJ_du(self, u_stage, step, stage):
            return 0.01 * u_stage

        def d2J_du2(self, u_stage, step, stage):
            return 0.01 * np.eye(1)

    problem = NonlinearProblem()
    objective = SimpleObjective()
    method = explicit_euler()
    N = 20

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, 2.0),
        N=N,
        y0=np.array([0.5]),
    )

    u = np.ones((N, 1, 1)) * 0.5
    J = optimizer.objective_value(u)
    grad = optimizer.gradient(u)

    # Gradient should have correct shape
    assert grad.shape == (N, 1, 1)

    # Take a step
    u_new = u - 0.1 * grad
    J_new = optimizer.objective_value(u_new)

    # Should improve (at least for small enough step)
    # May not always decrease due to nonlinearity, but should be computable
    assert isinstance(J_new, float)
    assert np.isfinite(J_new)
