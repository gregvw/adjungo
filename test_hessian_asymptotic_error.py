"""Test if there's a non-zero asymptotic error in the Hessian as h→0."""

import numpy as np
from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler


class MildlyNonlinearProblem:
    """dy/dt = -y + u - 0.1*y^3"""
    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return -y + u - 0.1 * y**3

    def F(self, y, u, t):
        return np.array([[-1.0 - 0.3 * y[0]**2]])

    def G(self, y, u, t):
        return np.array([[1.0]])

    def F_yy_action(self, y, u, t, v):
        return np.array([[-0.6 * y[0] * v[0]]])

    def F_yu_action(self, y, u, t, v_u):
        return np.zeros((1,))

    def F_uu_action(self, y, u, t, v_u):
        return np.zeros((1,))


class SimpleObjective:
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


print("=" * 80)
print("HESSIAN ASYMPTOTIC ERROR TEST")
print("=" * 80)
print()
print("Refining to very fine discretization to see if error → 0 or approaches a")
print("non-zero constant (which would indicate a systematic bug)")
print()

problem = MildlyNonlinearProblem()
objective = SimpleObjective(y_target=1.0, R=0.1)
method = explicit_euler()

T = 1.0
N_values = [10, 20, 40, 80, 160, 320, 640]

print("Testing first control variable at t=0:")
print()
print(f"{'N':<8} {'h':<12} {'Exact':<15} {'FD':<15} {'|Exact-FD|':<15} {'Rate':<8}")
print("-" * 80)

eps_fd = 1e-7  # Very fine FD perturbation

errors = []
exact_values = []
fd_values = []

for N in N_values:
    h = T / N

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, T),
        N=N,
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
        ),
    )

    u = np.ones((N, 1, 1)) * 0.5
    v = np.ones((N, 1, 1)) * 0.1

    # Exact Hessian for first control
    Hv_exact = optimizer.hessian_vector_product(u, v)
    exact_0 = Hv_exact[0, 0, 0]

    # FD approximation
    grad_0 = optimizer.gradient(u)
    grad_eps = optimizer.gradient(u + eps_fd * v)
    Hv_fd = (grad_eps - grad_0) / eps_fd
    fd_0 = Hv_fd[0, 0, 0]

    error = abs(exact_0 - fd_0)

    exact_values.append(exact_0)
    fd_values.append(fd_0)
    errors.append(error)

    # Rate
    if len(errors) > 1:
        rate = np.log(errors[-2] / errors[-1]) / np.log(N_values[len(errors)-2] / N)
        rate_str = f"{rate:.2f}"
    else:
        rate_str = "-"

    print(f"{N:<8} {h:<12.6f} {exact_0:<15.8f} {fd_0:<15.8f} {error:<15.6e} {rate_str:<8}")

print()
print("-" * 80)
print("Asymptotic Analysis:")
print()

if len(errors) >= 5:
    # Check if error is approaching zero or a constant
    last_3_errors = errors[-3:]
    error_reduction = last_3_errors[0] / last_3_errors[-1]

    print(f"Error at N={N_values[-3]}: {errors[-3]:.6e}")
    print(f"Error at N={N_values[-1]}: {errors[-1]:.6e}")
    print(f"Reduction over last 3 refinements: {error_reduction:.2f}x")
    print()

    # Extrapolate to h→0
    print("Extrapolated values as h→0:")
    # Fit last few points to a + b*h form
    from numpy.polynomial import Polynomial

    h_vals = np.array([T/N for N in N_values[-4:]])
    exact_last = np.array(exact_values[-4:])
    fd_last = np.array(fd_values[-4:])

    # Linear fit
    p_exact = Polynomial.fit(h_vals, exact_last, 1)
    p_fd = Polynomial.fit(h_vals, fd_last, 1)

    exact_limit = p_exact(0.0)
    fd_limit = p_fd(0.0)
    asymptotic_error = abs(exact_limit - fd_limit)

    print(f"  Exact (adjoint):     {exact_limit:.8f}")
    print(f"  FD (ground truth):   {fd_limit:.8f}")
    print(f"  Asymptotic error:    {asymptotic_error:.8f}")
    print()

    if asymptotic_error > 1e-4:
        print(f"❌ NON-ZERO asymptotic error: {asymptotic_error:.6e}")
        print(f"   → Exact and FD converge to DIFFERENT values")
        print(f"   → This proves a systematic implementation bug")
        print()
        print(f"   The bug introduces a O(h⁰) error of ~{asymptotic_error:.6e}")
    elif asymptotic_error > 1e-6:
        print(f"⚠️  Small asymptotic error: {asymptotic_error:.6e}")
        print(f"   → May be extrapolation error or subtle bug")
    else:
        print(f"✅ Asymptotic error near zero: {asymptotic_error:.6e}")
        print(f"   → Both converge to same value")
        print(f"   → Implementation likely correct, just slow convergence")

print()
print("=" * 80)
print("FULL VECTOR NORM TEST")
print("=" * 80)
print()

print(f"{'N':<8} {'h':<12} {'||Hv_exact - Hv_FD||':<20} {'Rate':<8}")
print("-" * 60)

errors_norm = []

for N in N_values:
    h = T / N

    optimizer = GLMOptimizer(
        problem=problem,
        objective=objective,
        method=method,
        t_span=(0.0, T),
        N=N,
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
        ),
    )

    u = np.ones((N, 1, 1)) * 0.5
    v = np.ones((N, 1, 1)) * 0.1

    Hv_exact = optimizer.hessian_vector_product(u, v)
    grad_0 = optimizer.gradient(u)
    grad_eps = optimizer.gradient(u + eps_fd * v)
    Hv_fd = (grad_eps - grad_0) / eps_fd

    error_norm = np.linalg.norm(Hv_exact - Hv_fd)
    errors_norm.append(error_norm)

    if len(errors_norm) > 1:
        rate = np.log(errors_norm[-2] / errors_norm[-1]) / np.log(N_values[len(errors_norm)-2] / N)
        rate_str = f"{rate:.2f}"
    else:
        rate_str = "-"

    print(f"{N:<8} {h:<12.6f} {error_norm:<20.6e} {rate_str:<8}")

print()
avg_rate = np.mean([np.log(errors_norm[i] / errors_norm[i+1]) / np.log(N_values[i+1] / N_values[i])
                    for i in range(len(errors_norm)-1)])
print(f"Average convergence rate: {avg_rate:.2f}")
print()

if avg_rate < 0.8:
    print(f"❌ Convergence rate {avg_rate:.2f} < 0.8")
    print(f"   → Dominated by O(h^{avg_rate:.2f}) or O(1) error")
    print(f"   → Systematic implementation bug")
elif avg_rate < 1.5:
    print(f"✅ Convergence rate {avg_rate:.2f} ≈ 1.0")
    print(f"   → O(h) convergence as expected for explicit Euler")
    print(f"   → Implementation likely correct")
else:
    print(f"✅ Convergence rate {avg_rate:.2f} > 1.5")
    print(f"   → Better than O(h) convergence")
    print(f"   → Implementation correct")
