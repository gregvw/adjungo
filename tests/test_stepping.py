"""Tests for forward and adjoint stepping algorithms."""

import numpy as np

from adjungo.core.plan import DiscretizationPlan
from adjungo.methods.runge_kutta import explicit_euler, rk4
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.stepping.adjoint import adjoint_solve
from adjungo.stepping.forward import forward_solve
from adjungo.stepping.trajectory import Trajectory


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

    trajectory = forward_solve(y0, u, DiscretizationPlan.uniform(t_span, N, method), problem, solver)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.Y.shape == (11, 1, 2)  # N+1, r, n
    # Stage storage is packed as (sum_n s_n, n) per C-18.5; for a plan with
    # one stage count throughout it is also viewable as (N, s, n).
    assert trajectory.Z.shape == (10, 2)
    assert trajectory.Z_rect.shape == (10, 1, 2)  # N, s, n
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

    trajectory = forward_solve(y0, u, DiscretizationPlan.uniform(t_span, N, method), problem, solver)

    assert trajectory.Y.shape == (11, 1, 2)
    assert trajectory.Z.shape == (40, 2)  # 10 steps x 4 stages, packed
    assert trajectory.Z_rect.shape == (10, 4, 2)  # 4 stages per step
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

    traj_zero = forward_solve(y0, u_zero, DiscretizationPlan.uniform(t_span, N, method), problem, solver)
    traj_nonzero = forward_solve(y0, u_nonzero, DiscretizationPlan.uniform(t_span, N, method), problem, solver)

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

    trajectory = forward_solve(y0, u, DiscretizationPlan.uniform(t_span, N, method), problem, solver)
    objective = QuadraticObjective(y_target=np.array([0.0, 0.0]))

    adjoint = adjoint_solve(trajectory, objective, solver)

    assert adjoint.Lambda.shape == (11, 1, 2)  # N+1, r, n
    assert adjoint.Mu.shape == (40, 2)  # packed (sum_n s_n, n)
    assert adjoint.WeightedAdj.shape == (40, 2)  # packed


def test_adjoint_zero_terminal_condition():
    """Test adjoint with zero terminal gradient."""
    problem = LinearProblem()
    method = explicit_euler()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 1.0])
    u = np.zeros((5, 1, 1))
    t_span = (0.0, 1.0)
    N = 5

    trajectory = forward_solve(y0, u, DiscretizationPlan.uniform(t_span, N, method), problem, solver)

    # Objective with zero terminal gradient
    class ZeroTerminalObjective:
        def dJ_dy_terminal(self, y_final):
            return np.zeros_like(y_final)

        def dJ_dy(self, y, step):
            return np.zeros_like(y)

    objective = ZeroTerminalObjective()
    adjoint = adjoint_solve(trajectory, objective, solver)

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

    trajectory = forward_solve(y0, u, DiscretizationPlan.uniform(t_span, N, method), problem, solver)

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

    trajectory = forward_solve(y0, u_func, DiscretizationPlan.uniform(t_span, N, method), problem, solver)

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

    trajectory = forward_solve(y0, u, DiscretizationPlan.uniform(t_span, N, method), problem, solver)
    objective = QuadraticObjective(y_target=np.array([0.0, 0.0]))

    adjoint = adjoint_solve(trajectory, objective, solver)
    plan = trajectory.plan

    # Verify weighted adjoint formula: Λ_k = Σ_j a_{jk} μ_j + Σ_j b_{jk} λ_j
    for step in range(N):
        Mu_step = plan.stages(adjoint.Mu, step)
        W_step = plan.stages(adjoint.WeightedAdj, step)
        for k in range(method.s):
            expected = (
                method.A[:, k] @ Mu_step
                + method.B[:, k] @ adjoint.Lambda[step + 1]
            )
            assert np.allclose(W_step[k], expected)
