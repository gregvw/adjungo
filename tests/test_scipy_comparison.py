"""Comparison tests against scipy.integrate solvers."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from adjungo.stepping.forward import forward_solve
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.methods.runge_kutta import explicit_euler, rk4, heun


class LinearODE:
    """Linear ODE: dy/dt = A*y + B*u for comparison with scipy."""

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.state_dim = A.shape[0]
        self.control_dim = B.shape[1]

    def f(self, y, u, t):
        return self.A @ y + self.B @ u

    def F(self, y, u, t):
        return self.A

    def G(self, y, u, t):
        return self.B


class HarmonicOscillator:
    """Harmonic oscillator: d²x/dt² + x = 0 -> [x', v'] = [v, -x]"""

    state_dim = 2
    control_dim = 0

    def f(self, y, u, t):
        return np.array([y[1], -y[0]])

    def F(self, y, u, t):
        return np.array([[0.0, 1.0], [-1.0, 0.0]])

    def G(self, y, u, t):
        return np.zeros((2, 0))


class ExponentialDecay:
    """Simple exponential: dy/dt = -k*y"""

    def __init__(self, k=1.0):
        self.k = k
        self.state_dim = 1
        self.control_dim = 0

    def f(self, y, u, t):
        return np.array([-self.k * y[0]])

    def F(self, y, u, t):
        return np.array([[-self.k]])

    def G(self, y, u, t):
        return np.zeros((1, 0))


def test_explicit_euler_vs_scipy_exponential():
    """Compare explicit Euler with scipy's RK23 on exponential decay."""
    problem = ExponentialDecay(k=1.0)
    method = explicit_euler()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0])
    t_span = (0.0, 1.0)
    N = 100  # Need many steps for Euler to be accurate

    # Our implementation
    u = np.zeros((N, 1, 0))  # No control
    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
    y_final_ours = trajectory.Y[-1, 0]

    # Scipy solution
    def scipy_rhs(t, y):
        return problem.f(y, np.array([]), t)

    sol = solve_ivp(scipy_rhs, t_span, y0, method='RK23', dense_output=True)
    y_final_scipy = sol.y[:, -1]

    # Analytical solution: y(1) = e^(-1) ≈ 0.3679
    y_analytical = np.exp(-1.0)

    print(f"Analytical: {y_analytical:.6f}")
    print(f"Ours (Euler): {y_final_ours[0]:.6f}")
    print(f"Scipy (RK23): {y_final_scipy[0]:.6f}")

    # Euler with 100 steps should be within 5% of analytical
    assert np.abs(y_final_ours[0] - y_analytical) < 0.05


def test_rk4_vs_scipy_harmonic_oscillator():
    """Compare RK4 with scipy's RK45 on harmonic oscillator."""
    problem = HarmonicOscillator()
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t_span = (0.0, 2 * np.pi)  # One full period
    N = 50

    # Our implementation
    u = np.zeros((N, 4, 0))  # No control, 4 stages for RK4
    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
    y_final_ours = trajectory.Y[-1, 0]

    # Scipy solution
    def scipy_rhs(t, y):
        return problem.f(y, np.array([]), t)

    sol = solve_ivp(scipy_rhs, t_span, y0, method='RK45', rtol=1e-8, atol=1e-10)
    y_final_scipy = sol.y[:, -1]

    # After one period, should return to initial state: [1, 0]
    # (accounting for numerical error)
    print(f"Initial: [{y0[0]:.6f}, {y0[1]:.6f}]")
    print(f"Ours (RK4): [{y_final_ours[0]:.6f}, {y_final_ours[1]:.6f}]")
    print(f"Scipy (RK45): [{y_final_scipy[0]:.6f}, {y_final_scipy[1]:.6f}]")
    print(f"Analytical: [1.0, 0.0]")

    # Both should be close to [1, 0]
    assert np.allclose(y_final_ours, y0, atol=0.01)
    assert np.allclose(y_final_scipy, y0, atol=1e-6)

    # Our RK4 should be reasonably close to scipy's RK45
    # (RK45 is adaptive and more accurate)
    assert np.allclose(y_final_ours, y_final_scipy, atol=0.01)


def test_heun_vs_scipy_linear_system():
    """Compare Heun's method with scipy on 2D linear system."""
    A = np.array([[-0.5, 0.0], [0.0, -1.0]])
    B = np.zeros((2, 0))
    problem = LinearODE(A, B)

    method = heun()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 2.0])
    t_span = (0.0, 2.0)
    N = 40

    # Our implementation
    u = np.zeros((N, 2, 0))  # No control, 2 stages for Heun
    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
    y_final_ours = trajectory.Y[-1, 0]

    # Scipy solution
    def scipy_rhs(t, y):
        return A @ y

    sol = solve_ivp(scipy_rhs, t_span, y0, method='RK45', rtol=1e-8)
    y_final_scipy = sol.y[:, -1]

    # Analytical: y(t) = [e^(-0.5*t), 2*e^(-t)]
    y_analytical = np.array([np.exp(-0.5 * 2.0), 2.0 * np.exp(-2.0)])

    print(f"Analytical: [{y_analytical[0]:.6f}, {y_analytical[1]:.6f}]")
    print(f"Ours (Heun): [{y_final_ours[0]:.6f}, {y_final_ours[1]:.6f}]")
    print(f"Scipy (RK45): [{y_final_scipy[0]:.6f}, {y_final_scipy[1]:.6f}]")

    # All should match analytical within tolerance
    assert np.allclose(y_final_ours, y_analytical, rtol=0.01)
    assert np.allclose(y_final_scipy, y_analytical, rtol=1e-6)


def test_trajectory_at_intermediate_points():
    """Compare trajectory at multiple time points, not just final."""
    problem = ExponentialDecay(k=2.0)
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([3.0])
    t_span = (0.0, 1.0)
    N = 20

    # Our implementation
    u = np.zeros((N, 4, 0))
    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)

    # Scipy solution
    def scipy_rhs(t, y):
        return problem.f(y, np.array([]), t)

    sol = solve_ivp(scipy_rhs, t_span, y0, method='RK45',
                    t_eval=np.linspace(t_span[0], t_span[1], N + 1),
                    rtol=1e-8)

    # Compare at each time step
    h = (t_span[1] - t_span[0]) / N
    for i in range(N + 1):
        t = t_span[0] + i * h
        y_ours = trajectory.Y[i, 0, 0]
        y_scipy = sol.y[0, i]
        y_analytical = 3.0 * np.exp(-2.0 * t)

        # Our RK4 should be close to both scipy and analytical
        assert np.abs(y_ours - y_analytical) < 0.01
        assert np.abs(y_scipy - y_analytical) < 1e-6
        assert np.abs(y_ours - y_scipy) < 0.01


def test_with_control_input():
    """Test system with control input (scipy comparison of uncontrolled part)."""
    # dy/dt = -y + u, with u(t) = sin(t)
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    problem = LinearODE(A, B)

    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([0.0])
    t_span = (0.0, 2.0)
    N = 40

    # Control function: u(t) = sin(t)
    def u_func(t, step, stage):
        return np.array([np.sin(t)])

    # Our implementation with control
    trajectory = forward_solve(y0, u_func, t_span, N, problem, method, solver)
    y_final_ours = trajectory.Y[-1, 0, 0]

    # Scipy solution with same control
    def scipy_rhs(t, y):
        u = np.array([np.sin(t)])
        return A @ y + B @ u

    sol = solve_ivp(scipy_rhs, t_span, y0, method='RK45', rtol=1e-8)
    y_final_scipy = sol.y[0, -1]

    print(f"Ours (RK4): {y_final_ours:.6f}")
    print(f"Scipy (RK45): {y_final_scipy:.6f}")

    # Should match within RK4 accuracy
    assert np.abs(y_final_ours - y_final_scipy) < 0.01


def test_stiff_problem_comparison():
    """Test on a mildly stiff problem (within explicit method capability)."""
    # dy/dt = -10*y, with y(0) = 1
    problem = ExponentialDecay(k=10.0)
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0])
    t_span = (0.0, 0.5)  # Short time to avoid instability with explicit method
    N = 100  # Need fine steps for stability

    # Our implementation
    u = np.zeros((N, 4, 0))
    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
    y_final_ours = trajectory.Y[-1, 0, 0]

    # Scipy solution
    def scipy_rhs(t, y):
        return problem.f(y, np.array([]), t)

    sol = solve_ivp(scipy_rhs, t_span, y0, method='RK45', rtol=1e-8)
    y_final_scipy = sol.y[0, -1]

    # Analytical: y(0.5) = e^(-5)
    y_analytical = np.exp(-5.0)

    print(f"Analytical: {y_analytical:.6f}")
    print(f"Ours (RK4): {y_final_ours:.6f}")
    print(f"Scipy (RK45): {y_final_scipy:.6f}")

    # Both should be close to analytical
    assert np.abs(y_final_ours - y_analytical) < 0.01
    assert np.abs(y_final_scipy - y_analytical) < 1e-6


def test_conservation_of_energy():
    """Test energy conservation in harmonic oscillator."""
    problem = HarmonicOscillator()
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0, 0.0])  # x=1, v=0
    t_span = (0.0, 10 * np.pi)  # 5 full periods
    N = 200

    # Our implementation
    u = np.zeros((N, 4, 0))
    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)

    # Compute energy at each time step: E = 0.5*(v^2 + x^2)
    energies = []
    for i in range(N + 1):
        x, v = trajectory.Y[i, 0]
        E = 0.5 * (v**2 + x**2)
        energies.append(E)

    E0 = energies[0]
    E_final = energies[-1]

    print(f"Initial energy: {E0:.6f}")
    print(f"Final energy: {E_final:.6f}")
    print(f"Max energy: {max(energies):.6f}")
    print(f"Min energy: {min(energies):.6f}")

    # RK4 should conserve energy reasonably well (symplectic-ish)
    # Energy should not drift by more than a few percent over 5 periods
    assert np.abs(E_final - E0) < 0.1 * E0


def test_convergence_with_refinement():
    """Test that error decreases with finer time steps (convergence test)."""
    problem = ExponentialDecay(k=1.0)
    method = rk4()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0])
    t_span = (0.0, 1.0)

    # Analytical solution
    y_exact = np.exp(-1.0)

    errors = []
    step_sizes = []

    for N in [10, 20, 40, 80]:
        u = np.zeros((N, 4, 0))
        trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
        y_final = trajectory.Y[-1, 0, 0]

        error = np.abs(y_final - y_exact)
        h = (t_span[1] - t_span[0]) / N

        errors.append(error)
        step_sizes.append(h)

        print(f"N={N:3d}, h={h:.4f}, error={error:.2e}")

    # RK4 is 4th order: error ~ h^4
    # Doubling steps should reduce error by ~16x
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        # Should be roughly (h_i / h_{i+1})^4 = 2^4 = 16
        assert ratio > 10  # At least an order of magnitude improvement
        assert ratio < 25  # Not too much (would indicate something weird)


@pytest.mark.parametrize("N", [10, 20, 50])
def test_euler_accuracy_vs_scipy_parametrized(N):
    """Parametrized test: Euler accuracy for different step counts."""
    problem = ExponentialDecay(k=1.0)
    method = explicit_euler()
    solver = ExplicitStageSolver()

    y0 = np.array([1.0])
    t_span = (0.0, 1.0)

    # Our implementation
    u = np.zeros((N, 1, 0))
    trajectory = forward_solve(y0, u, t_span, N, problem, method, solver)
    y_final_ours = trajectory.Y[-1, 0, 0]

    # Analytical
    y_exact = np.exp(-1.0)

    error = np.abs(y_final_ours - y_exact)
    h = (t_span[1] - t_span[0]) / N

    # Euler is first order: error ~ h
    expected_error = h  # Rough estimate

    print(f"N={N}, h={h:.4f}, error={error:.4e}, expected~{expected_error:.4e}")

    # Error should be O(h)
    assert error < 10 * expected_error
