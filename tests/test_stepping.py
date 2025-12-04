"""Tests for forward and adjoint stepping algorithms."""

import numpy as np
import pytest

from adjungo.stepping.forward import forward_solve
from adjungo.stepping.adjoint import adjoint_solve
from adjungo.stepping.trajectory import Trajectory
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.methods.runge_kutta import explicit_euler, rk4


class LinearProblem:
    """Simple linear problem: dy/dt = A*y + B*u"""

    state_dim = 2
    control_dim = 1

    def __init__(self):
        self.A_mat = np.array([[-1.0, 0.0], [0.0, -2.0]])
        self.B_mat = np.array([[1.0], [1.0]])

    def f(self, y, u, t):
        return self.A_mat @ y + self.B_mat @ u

    def F(self, y, u, t):
        return self.A_mat

    def G(self, y, u, t):
        return self.B_mat


class QuadraticObjective:
    """Objective: J = 0.5 * ||y_final - y_target||^2 + 0.5 * ||u||^2"""

    def __init__(self, y_target, weight_u=1.0):
        self.y_target = y_target
        self.weight_u = weight_u

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0]  # Final external stage
        terminal_cost = 0.5 * np.sum((y_final - self.y_target) ** 2)
        control_cost = 0.5 * self.weight_u * np.sum(u ** 2)
        return terminal_cost + control_cost

    def dJ_dy_terminal(self, y_final):
        # Return gradient for all external stages (r, n)
        grad = np.zeros_like(y_final)
        grad[0] = y_final[0] - self.y_target
        return grad

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return self.weight_u * u_stage

    def d2J_du2(self, u_stage, step, stage):
        return self.weight_u * np.eye(len(u_stage))


def test_forward_solve_explicit_euler():
    """Test forward solve with explicit Euler."""
    problem = LinearProblem()
    method = explicit_euler()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 1.0])
    u = np.zeros((10, 1, 1))  # N=10, s=1, ν=1
    t_span = (0.0, 1.0)
    N = 10

    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.Y.shape == (11, 1, 2)  # N+1, r, n
    assert trajectory.Z.shape == (10, 1, 2)  # N, s, n
    assert len(trajectory.caches) == 10
    assert trajectory.N == 10


def test_forward_solve_rk4():
    """Test forward solve with RK4."""
    problem = LinearProblem()
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 1.0])
    u = np.zeros((10, 4, 1))  # N=10, s=4, ν=1
    t_span = (0.0, 1.0)
    N = 10

    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)

    assert trajectory.Y.shape == (11, 1, 2)
    assert trajectory.Z.shape == (10, 4, 2)  # 4 stages per step
    assert trajectory.N == 10


def test_forward_solve_with_nonzero_control():
    """Test that nonzero control affects trajectory."""
    problem = LinearProblem()
    method = explicit_euler()
    solver = ExplicitStageSolver()

    y0 = np.array([0.0, 0.0])
    u_zero = np.zeros((10, 1, 1))
    u_nonzero = np.ones((10, 1, 1))
    t_span = (0.0, 1.0)
    N = 10

    traj_zero = forward_solve(y0, u_zero, t_span, N, problem, method, solver)
    traj_nonzero = forward_solve(y0, u_nonzero, t_span, N, problem, method, solver)

    # Trajectories should be different
    assert not np.allclose(traj_zero.Y, traj_nonzero.Y)


def test_adjoint_solve_basic():
    """Test adjoint solve produces correct shapes."""
    problem = LinearProblem()
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 1.0])
    u = np.zeros((10, 4, 1))
    t_span = (0.0, 1.0)
    N = 10

    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
    objective = QuadraticObjective(y_target=np.array([0.0, 0.0]))

    h = (t_span[1] - t_span[0]) / N
    adjoint = adjoint_solve(trajectory, objective, method, solver, h)

    assert adjoint.Lambda.shape == (11, 1, 2)  # N+1, r, n
    assert adjoint.Mu.shape == (10, 4, 2)  # N, s, n
    assert adjoint.WeightedAdj.shape == (10, 4, 2)  # N, s, n


def test_adjoint_zero_terminal_condition():
    """Test adjoint with zero terminal gradient."""
    problem = LinearProblem()
    method = explicit_euler()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 1.0])
    u = np.zeros((5, 1, 1))
    t_span = (0.0, 1.0)
    N = 5

    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)

    # Objective with zero terminal gradient
    class ZeroTerminalObjective:
        def dJ_dy_terminal(self, y_final):
            return np.zeros_like(y_final)

        def dJ_dy(self, y, step):
            return np.zeros_like(y)

    objective = ZeroTerminalObjective()
    h = (t_span[1] - t_span[0]) / N
    adjoint = adjoint_solve(trajectory, objective, method, solver, h)

    # With zero terminal condition and no running cost, adjoints should be zero
    assert np.allclose(adjoint.Lambda, 0.0)
    assert np.allclose(adjoint.Mu, 0.0)


def test_trajectory_properties():
    """Test Trajectory dataclass properties."""
    problem = LinearProblem()
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 1.0])
    u = np.zeros((10, 4, 1))
    t_span = (0.0, 1.0)
    N = 10

    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)

    assert trajectory.N == 10
    assert trajectory.n == 2
    assert trajectory.r == 1
    assert trajectory.s == 4


def test_forward_solve_callable_control():
    """Test forward solve with callable control function."""
    problem = LinearProblem()
    method = explicit_euler()
    solver = ExplicitStageSolver()

    y0 = np.array([0.0, 0.0])

    # Time-varying control: u(t) = sin(2πt)
    def u_func(t, step, stage):
        return np.array([np.sin(2 * np.pi * t)])

    t_span = (0.0, 1.0)
    N = 10

    trajectory = forward_solve(y0, u_func, t_span, N, problem, method, solver)

    assert trajectory.Y.shape == (11, 1, 2)
    # State should be affected by sinusoidal control
    assert not np.allclose(trajectory.Y[-1], y0)


def test_weighted_adjoint_computation():
    """Test that weighted adjoints are computed correctly."""
    problem = LinearProblem()
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 1.0])
    u = np.zeros((5, 4, 1))
    t_span = (0.0, 1.0)
    N = 5

    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
    objective = QuadraticObjective(y_target=np.array([0.0, 0.0]))

    h = (t_span[1] - t_span[0]) / N
    adjoint = adjoint_solve(trajectory, objective, method, solver, h)

    # Verify weighted adjoint formula: Λ_k = Σ_j a_{jk} μ_j + Σ_j b_{jk} λ_j
    for step in range(N):
        for k in range(method.s):
            expected = (
                method.A[:, k] @ adjoint.Mu[step]
                + method.B[:, k] @ adjoint.Lambda[step + 1]
            )
            assert np.allclose(adjoint.WeightedAdj[step, k], expected)
