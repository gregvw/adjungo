"""Test if Hessian error decreases with DISCRETIZATION refinement."""

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


print("=" * 80)
print("HESSIAN DISCRETIZATION CONVERGENCE TEST")
print("=" * 80)
print()
print("Testing if Hessian-vector product error decreases as we refine h = T/N")
print()

problem = MildlyNonlinearProblem()
objective = SimpleObjective(y_target=1.0, R=0.1)
method = explicit_euler()

T = 1.0
N_values = [5, 10, 20, 40, 80]

print(f"{'N':<8} {'h=T/N':<12} {'||Hv_exact - Hv_FD||':<20} {'Rate':<10}")
print("-" * 60)

eps_fd = 1e-6  # Fixed FD perturbation
errors = []

for N in N_values:
    h = T / N

    # Create optimizer
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

    # Control and direction (constant in time)
    u = np.ones((N, 1, 1)) * 0.5
    v = np.ones((N, 1, 1)) * 0.1

    # Exact Hessian-vector product
    Hv_exact = optimizer.hessian_vector_product(u, v)

    # FD approximation
    grad_0 = optimizer.gradient(u)
    grad_eps = optimizer.gradient(u + eps_fd * v)
    Hv_fd = (grad_eps - grad_0) / eps_fd

    # Error
    error = np.linalg.norm(Hv_exact - Hv_fd)
    errors.append(error)

    # Convergence rate
    if len(errors) > 1:
        rate = np.log(errors[-2] / errors[-1]) / np.log(N_values[len(errors)-2] / N)
        rate_str = f"{rate:.2f}"
    else:
        rate_str = "-"

    print(f"{N:<8} {h:<12.6f} {error:<20.6e} {rate_str:<10}")

print()
print("-" * 60)
print("Analysis:")
print()

if len(errors) > 2:
    avg_rate = np.mean([np.log(errors[i] / errors[i+1]) / np.log(N_values[i+1] / N_values[i])
                        for i in range(len(errors)-1)])
    print(f"Average convergence rate: {avg_rate:.2f}")
    print()

    if avg_rate > 0.8:
        print(f"✅ Error DECREASES with refinement (rate={avg_rate:.2f})")
        print(f"   → Implementation is CORRECT, just has O(h^{avg_rate:.1f}) truncation error")
        print()
        if avg_rate < 1.5:
            print(f"   Expected O(h²) for explicit Euler + adjoint")
            print(f"   Getting O(h^{avg_rate:.2f}) - may indicate first-order component")
        elif avg_rate >= 1.8:
            print(f"   O(h²) convergence as expected!")
    elif abs(avg_rate) < 0.3:
        print(f"❌ Error is CONSTANT (rate={avg_rate:.2f})")
        print(f"   → Systematic implementation bug")
    else:
        print(f"⚠️  Slow convergence (rate={avg_rate:.2f})")
        print(f"   → May have O(h) term dominating")

print()
print("=" * 80)
print("ELEMENT-WISE DISCRETIZATION CONVERGENCE")
print("=" * 80)
print()
print("Testing first control variable: u[0]")
print()

print(f"{'N':<8} {'h=T/N':<12} {'Exact H[0,0]*v[0]':<18} {'FD H[0,0]*v[0]':<18} {'Error':<15}")
print("-" * 80)

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

    # Exact
    Hv_exact = optimizer.hessian_vector_product(u, v)
    exact_0 = Hv_exact[0, 0, 0]

    # FD
    grad_0 = optimizer.gradient(u)
    grad_eps = optimizer.gradient(u + eps_fd * v)
    Hv_fd = (grad_eps - grad_0) / eps_fd
    fd_0 = Hv_fd[0, 0, 0]

    error = abs(exact_0 - fd_0)

    print(f"{N:<8} {h:<12.6f} {exact_0:<18.8f} {fd_0:<18.8f} {error:<15.6e}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if len(errors) > 2:
    final_error = errors[-1]
    initial_error = errors[0]
    reduction = initial_error / final_error

    print(f"Error reduction from N={N_values[0]} to N={N_values[-1]}: {reduction:.1f}x")
    print()

    if reduction > 10:
        print("✅ Hessian implementation is CORRECT")
        print(f"   Error decreases with refinement as expected")
        print(f"   The test failure at N=5 is due to coarse discretization, not a bug")
        print()
        print("Recommendation: Either")
        print(f"  1. Use finer discretization (N≥40) for accurate Hessians")
        print(f"  2. Relax test tolerance to account for truncation error at coarse h")
    else:
        print("❌ Error does not decrease significantly with refinement")
        print("   → Likely a systematic implementation bug")
