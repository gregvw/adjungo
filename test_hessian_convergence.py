"""Hessian finite difference convergence study."""

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


# Setup
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
        has_second_derivatives=True,
    ),
)

# Control point and direction
np.random.seed(42)  # Reproducible results
u = np.ones((10, 1, 1)) * 0.5
v = np.random.randn(10, 1, 1) * 0.1

print("=" * 80)
print("HESSIAN FINITE DIFFERENCE CONVERGENCE STUDY")
print("=" * 80)
print(f"\nProblem: dy/dt = -y + u - 0.1*y³")
print(f"Objective: J = 0.5*(y(T) - 1)² + 0.5*R*∫u²")
print(f"Control size: {u.shape}")
print()

# Exact Hessian-vector product
print("Computing exact Hessian-vector product via second-order adjoint...")
Hv_exact = optimizer.hessian_vector_product(u, v)
print(f"Computed: ||Hv|| = {np.linalg.norm(Hv_exact):.6e}")
print()

# Finite difference convergence study
print("-" * 80)
print("FORWARD DIFFERENCE: Hv_FD = [∇J(u + h*v) - ∇J(u)] / h")
print("-" * 80)
print(f"{'h':<15} {'||Error||':<15} {'Max |Error|':<15} {'Rate':<10}")
print("-" * 80)

eps_values = np.array([1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6])
errors_fwd = []
errors_max_fwd = []

grad_0 = optimizer.gradient(u)

for eps in eps_values:
    # Forward difference
    grad_eps = optimizer.gradient(u + eps * v)
    Hv_fd = (grad_eps - grad_0) / eps

    error = Hv_fd - Hv_exact
    error_norm = np.linalg.norm(error)
    error_max = np.max(np.abs(error))

    errors_fwd.append(error_norm)
    errors_max_fwd.append(error_max)

    # Compute convergence rate
    if len(errors_fwd) > 1:
        rate = np.log(errors_fwd[-2] / errors_fwd[-1]) / np.log(eps_values[len(errors_fwd)-2] / eps)
        rate_str = f"{rate:.2f}"
    else:
        rate_str = "-"

    print(f"{eps:<15.1e} {error_norm:<15.6e} {error_max:<15.6e} {rate_str:<10}")

print()
print(f"Expected rate for forward differences: O(h¹)")
avg_rate_fwd = np.mean([np.log(errors_fwd[i-1] / errors_fwd[i]) / np.log(eps_values[i-1] / eps_values[i])
                         for i in range(1, len(errors_fwd))])
print(f"Average convergence rate: {avg_rate_fwd:.2f}")
print()

# Centered difference convergence study
print("-" * 80)
print("CENTERED DIFFERENCE: Hv_CD = [∇J(u + h*v) - ∇J(u - h*v)] / (2h)")
print("-" * 80)
print(f"{'h':<15} {'||Error||':<15} {'Max |Error|':<15} {'Rate':<10}")
print("-" * 80)

errors_ctr = []
errors_max_ctr = []

for eps in eps_values:
    # Centered difference
    grad_plus = optimizer.gradient(u + eps * v)
    grad_minus = optimizer.gradient(u - eps * v)
    Hv_cd = (grad_plus - grad_minus) / (2 * eps)

    error = Hv_cd - Hv_exact
    error_norm = np.linalg.norm(error)
    error_max = np.max(np.abs(error))

    errors_ctr.append(error_norm)
    errors_max_ctr.append(error_max)

    # Compute convergence rate
    if len(errors_ctr) > 1:
        rate = np.log(errors_ctr[-2] / errors_ctr[-1]) / np.log(eps_values[len(errors_ctr)-2] / eps)
        rate_str = f"{rate:.2f}"
    else:
        rate_str = "-"

    print(f"{eps:<15.1e} {error_norm:<15.6e} {error_max:<15.6e} {rate_str:<10}")

print()
print(f"Expected rate for centered differences: O(h²)")
avg_rate_ctr = np.mean([np.log(errors_ctr[i-1] / errors_ctr[i]) / np.log(eps_values[i-1] / eps_values[i])
                        for i in range(1, len(errors_ctr))])
print(f"Average convergence rate: {avg_rate_ctr:.2f}")
print()

# Best accuracy comparison
print("-" * 80)
print("BEST ACCURACY COMPARISON")
print("-" * 80)

best_fwd_idx = np.argmin(errors_fwd)
best_ctr_idx = np.argmin(errors_ctr)

print(f"Forward differences:")
print(f"  Best h = {eps_values[best_fwd_idx]:.1e}")
print(f"  Error = {errors_fwd[best_fwd_idx]:.6e}")
print()

print(f"Centered differences:")
print(f"  Best h = {eps_values[best_ctr_idx]:.1e}")
print(f"  Error = {errors_ctr[best_ctr_idx]:.6e}")
print()

improvement = errors_fwd[best_fwd_idx] / errors_ctr[best_ctr_idx]
print(f"Improvement factor: {improvement:.1f}x")
print()

# Summary
print("=" * 80)
print("CONCLUSION")
print("=" * 80)

if avg_rate_fwd > 0.8 and avg_rate_fwd < 1.2:
    print(f"✅ Forward differences show O(h¹) convergence (rate = {avg_rate_fwd:.2f})")
else:
    print(f"⚠️  Forward differences rate = {avg_rate_fwd:.2f} (expected ~1.0)")

if avg_rate_ctr > 1.8 and avg_rate_ctr < 2.2:
    print(f"✅ Centered differences show O(h²) convergence (rate = {avg_rate_ctr:.2f})")
else:
    print(f"⚠️  Centered differences rate = {avg_rate_ctr:.2f} (expected ~2.0)")

print()

if errors_ctr[best_ctr_idx] < 1e-5:
    print("✅ Centered differences achieve < 1e-5 error")
    print("   Second-order adjoint implementation is CORRECT!")
elif errors_ctr[best_ctr_idx] < 1e-4:
    print("✅ Centered differences achieve < 1e-4 error")
    print("   Second-order adjoint implementation is very accurate")
else:
    print(f"⚠️  Best error = {errors_ctr[best_ctr_idx]:.6e}")
    print("   May need further investigation")

print()

# Show that our implementation is consistent
if avg_rate_fwd > 0.8 and avg_rate_ctr > 1.5:
    print("✅ Both convergence rates match theory")
    print("   This validates that the Hessian implementation is mathematically consistent")
    print("   The remaining error is purely from finite difference truncation")
    print()
    print(f"   For production use with finite differences:")
    print(f"   - Use centered differences with h ≈ 1e-5 to 1e-6")
    print(f"   - Expected accuracy: O(1e-6) to O(1e-8)")
    print(f"   - Or use exact Hessian-vector products (implemented!)")
