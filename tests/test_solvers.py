"""Tests for stage solvers."""

import numpy as np
import pytest

from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.solvers.sdirk import SDIRKStageSolver
from adjungo.core.method import GLMethod
from adjungo.methods.runge_kutta import explicit_euler, rk4, sdirk2


class SimpleProblem:
    """Linear test problem: dy/dt = -y + u"""

    state_dim = 2
    control_dim = 1

    def f(self, y, u, t):
        return np.array([-y[0] + u[0], -2 * y[1] + u[0]])

    def F(self, y, u, t):
        return np.array([[-1.0, 0.0], [0.0, -2.0]])

    def G(self, y, u, t):
        return np.array([[1.0], [1.0]])


def test_explicit_solver_forward_euler():
    """Test explicit solver with forward Euler on linear problem."""
    method = explicit_euler()
    solver = ExplicitStageSolver()
    problem = SimpleProblem()

    y_history = np.array([[1.0, 1.0]])  # (r=1, n=2)
    u_stages = np.array([[0.0]])  # (s=1, ν=1)
    t_n = 0.0
    h = 0.1

    Z, cache = solver.solve_stages(y_history, u_stages, t_n, h, problem, method)

    # For explicit Euler: Z = y_prev
    assert Z.shape == (1, 2)
    assert cache.Z.shape == (1, 2)
    assert len(cache.F) == 1
    assert len(cache.G) == 1
    assert cache.F[0].shape == (2, 2)
    assert cache.G[0].shape == (2, 1)


def test_explicit_solver_rk4():
    """Test explicit solver with RK4."""
    method = rk4()
    solver = ExplicitStageSolver()
    problem = SimpleProblem()

    y_history = np.array([[1.0, 1.0]])
    u_stages = np.zeros((4, 1))  # 4 stages
    t_n = 0.0
    h = 0.1

    Z, cache = solver.solve_stages(y_history, u_stages, t_n, h, problem, method)

    assert Z.shape == (4, 2)
    assert len(cache.F) == 4
    assert len(cache.G) == 4


def test_explicit_solver_adjoint():
    """Test explicit solver adjoint stage solve."""
    method = rk4()
    solver = ExplicitStageSolver()
    problem = SimpleProblem()

    y_history = np.array([[1.0, 1.0]])
    u_stages = np.zeros((4, 1))
    t_n = 0.0
    h = 0.1

    Z, cache = solver.solve_stages(y_history, u_stages, t_n, h, problem, method)

    # Adjoint with zero terminal condition
    lambda_ext = np.zeros((1, 2))
    mu = solver.solve_adjoint_stages(lambda_ext, cache, method, h)

    assert mu.shape == (4, 2)


def test_sdirk_solver_basic():
    """Test SDIRK solver on linear problem."""
    method = sdirk2()
    solver = SDIRKStageSolver()
    problem = SimpleProblem()

    y_history = np.array([[1.0, 1.0]])
    u_stages = np.zeros((2, 1))
    t_n = 0.0
    h = 0.1

    Z, cache = solver.solve_stages(y_history, u_stages, t_n, h, problem, method)

    assert Z.shape == (2, 2)
    assert cache.factorization is not None  # Should have cached factorization
    assert len(cache.F) == 2
    assert len(cache.G) == 2


def test_sdirk_solver_adjoint():
    """Test SDIRK solver adjoint with factorization reuse."""
    method = sdirk2()
    solver = SDIRKStageSolver()
    problem = SimpleProblem()

    y_history = np.array([[1.0, 1.0]])
    u_stages = np.zeros((2, 1))
    t_n = 0.0
    h = 0.1

    Z, cache = solver.solve_stages(y_history, u_stages, t_n, h, problem, method)

    # Adjoint should reuse factorization
    lambda_ext = np.ones((1, 2))
    mu = solver.solve_adjoint_stages(lambda_ext, cache, method, h)

    assert mu.shape == (2, 2)


def test_solver_cache_structure():
    """Test that solver cache contains all required data."""
    method = rk4()
    solver = ExplicitStageSolver()
    problem = SimpleProblem()

    y_history = np.array([[1.0, 1.0]])
    u_stages = np.zeros((4, 1))
    t_n = 0.0
    h = 0.1

    Z, cache = solver.solve_stages(y_history, u_stages, t_n, h, problem, method)

    # Verify cache structure
    assert hasattr(cache, "Z")
    assert hasattr(cache, "F")
    assert hasattr(cache, "G")
    assert hasattr(cache, "factorization")
    assert hasattr(cache, "stage_matrix")

    # Verify dimensions
    assert cache.Z.shape == (4, 2)
    assert all(F.shape == (2, 2) for F in cache.F)
    assert all(G.shape == (2, 1) for G in cache.G)
