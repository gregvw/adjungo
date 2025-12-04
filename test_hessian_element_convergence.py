"""Test if individual Hessian elements satisfy FD convergence."""

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


# Setup - small problem for full Hessian
problem = MildlyNonlinearProblem()
objective = SimpleObjective(y_target=1.0, R=0.1)
method = explicit_euler()

N = 5  # Small for manageable Hessian size
optimizer = GLMOptimizer(
    problem=problem,
    objective=objective,
    method=method,
    t_span=(0.0, 1.0),
    N=N,
    y0=np.array([0.0]),
    problem_structure=ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
    ),
)

# Control point
u = np.ones((N, 1, 1)) * 0.5
n_vars = u.size

print("=" * 80)
print("HESSIAN ELEMENT-WISE CONVERGENCE TEST")
print("=" * 80)
print(f"\nProblem: dy/dt = -y + u - 0.1*y³")
print(f"Objective: J = 0.5*(y(T) - 1)² + 0.5*R*∫u²")
print(f"Control: N={N} steps, {n_vars} variables")
print()

# Build exact Hessian via Hessian-vector products
print("Building exact Hessian matrix via second-order adjoint...")
H_exact = np.zeros((n_vars, n_vars))

for i in range(n_vars):
    v = np.zeros_like(u)
    v.flat[i] = 1.0
    Hv = optimizer.hessian_vector_product(u, v)
    H_exact[:, i] = Hv.ravel()

print(f"Exact Hessian computed: {H_exact.shape}")
print(f"  Symmetry check: max|H - H^T| = {np.max(np.abs(H_exact - H_exact.T)):.6e}")
print()

# Test finite difference convergence for various elements
print("-" * 80)
print("ELEMENT-WISE CONVERGENCE ANALYSIS")
print("-" * 80)
print()

# Select representative elements to test
test_elements = [
    (0, 0, "diagonal[0]"),
    (2, 2, "diagonal[2]"),
    (4, 4, "diagonal[4]"),
    (0, 1, "off-diag[0,1]"),
    (1, 2, "off-diag[1,2]"),
    (0, 4, "off-diag[0,4]"),
]

eps_values = np.array([1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6])

for i, j, label in test_elements:
    print(f"\nElement H[{i},{j}] ({label}):")
    print(f"  Exact value: {H_exact[i,j]:.8f}")
    print()
    print(f"  {'h':<12} {'FD(H_ij)':<15} {'Error':<15} {'Rate':<10}")
    print("  " + "-" * 60)

    errors = []
    fd_values = []

    for eps in eps_values:
        # Compute gradient at u + eps*e_j
        u_pert = u.copy()
        u_pert.flat[j] += eps
        grad_plus = optimizer.gradient(u_pert)

        # Compute gradient at u
        grad_0 = optimizer.gradient(u)

        # Finite difference: H_ij ≈ (∂J/∂u_i(u + eps*e_j) - ∂J/∂u_i(u)) / eps
        H_ij_fd = (grad_plus.flat[i] - grad_0.flat[i]) / eps
        fd_values.append(H_ij_fd)

        error = abs(H_ij_fd - H_exact[i,j])
        errors.append(error)

        # Compute rate
        if len(errors) > 1:
            rate = np.log(errors[-2] / errors[-1]) / np.log(eps_values[len(errors)-2] / eps)
            rate_str = f"{rate:.2f}"
        else:
            rate_str = "-"

        print(f"  {eps:<12.1e} {H_ij_fd:<15.8f} {error:<15.6e} {rate_str:<10}")

    # Check if converging
    if len(errors) > 3:
        avg_rate = np.mean([np.log(errors[k] / errors[k+1]) / np.log(eps_values[k] / eps_values[k+1])
                            for k in range(len(errors)-1)])
        print(f"\n  Average convergence rate: {avg_rate:.2f}")

        if avg_rate > 0.8 and avg_rate < 1.2:
            print(f"  ✅ Shows O(h¹) convergence (expected for forward differences)")
        elif abs(avg_rate) < 0.1:
            print(f"  ❌ NO CONVERGENCE - error is constant!")
        else:
            print(f"  ⚠️  Unexpected rate (expected ~1.0)")

print()
print("=" * 80)
print("CENTERED DIFFERENCES TEST (selected elements)")
print("=" * 80)
print()

# Test centered differences for a few elements
for i, j, label in test_elements[:3]:  # Just first 3 for brevity
    print(f"\nElement H[{i},{j}] ({label}) - Centered Differences:")
    print(f"  Exact value: {H_exact[i,j]:.8f}")
    print()
    print(f"  {'h':<12} {'CD(H_ij)':<15} {'Error':<15} {'Rate':<10}")
    print("  " + "-" * 60)

    errors_cd = []

    for eps in eps_values:
        # Centered difference
        u_plus = u.copy()
        u_plus.flat[j] += eps
        grad_plus = optimizer.gradient(u_plus)

        u_minus = u.copy()
        u_minus.flat[j] -= eps
        grad_minus = optimizer.gradient(u_minus)

        # H_ij ≈ (∂J/∂u_i(u + eps*e_j) - ∂J/∂u_i(u - eps*e_j)) / (2*eps)
        H_ij_cd = (grad_plus.flat[i] - grad_minus.flat[i]) / (2 * eps)

        error = abs(H_ij_cd - H_exact[i,j])
        errors_cd.append(error)

        # Compute rate
        if len(errors_cd) > 1:
            rate = np.log(errors_cd[-2] / errors_cd[-1]) / np.log(eps_values[len(errors_cd)-2] / eps)
            rate_str = f"{rate:.2f}"
        else:
            rate_str = "-"

        print(f"  {eps:<12.1e} {H_ij_cd:<15.8f} {error:<15.6e} {rate_str:<10}")

    if len(errors_cd) > 3:
        avg_rate = np.mean([np.log(errors_cd[k] / errors_cd[k+1]) / np.log(eps_values[k] / eps_values[k+1])
                            for k in range(len(errors_cd)-1)])
        print(f"\n  Average convergence rate: {avg_rate:.2f}")

        if avg_rate > 1.8 and avg_rate < 2.2:
            print(f"  ✅ Shows O(h²) convergence (expected for centered differences)")
        elif abs(avg_rate) < 0.1:
            print(f"  ❌ NO CONVERGENCE - error is constant!")
        else:
            print(f"  ⚠️  Unexpected rate (expected ~2.0)")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

# Check overall behavior
diagonal_errors = []
offdiag_errors = []

eps_test = 1e-5
for i in range(n_vars):
    for j in range(n_vars):
        u_pert = u.copy()
        u_pert.flat[j] += eps_test
        grad_plus = optimizer.gradient(u_pert)
        grad_0 = optimizer.gradient(u)
        H_ij_fd = (grad_plus.flat[i] - grad_0.flat[i]) / eps_test

        error = abs(H_ij_fd - H_exact[i,j])

        if i == j:
            diagonal_errors.append(error)
        else:
            offdiag_errors.append(error)

print(f"Using h = {eps_test:.1e}:")
print(f"  Diagonal elements:")
print(f"    Mean error: {np.mean(diagonal_errors):.6e}")
print(f"    Max error:  {np.max(diagonal_errors):.6e}")
print()
print(f"  Off-diagonal elements:")
print(f"    Mean error: {np.mean(offdiag_errors):.6e}")
print(f"    Max error:  {np.max(offdiag_errors):.6e}")
print()

if np.max(diagonal_errors) < 1e-4 and np.max(offdiag_errors) < 1e-4:
    print("✅ All Hessian elements match FD to high accuracy")
    print("   Implementation is correct!")
elif np.max(diagonal_errors) > 1e-3 or np.max(offdiag_errors) > 1e-3:
    print("❌ Large errors in Hessian elements")
    print("   Implementation has bugs")
else:
    print("⚠️  Moderate errors in Hessian elements")
    print("   May need investigation")

print()
print("Key insight:")
print("  If errors are CONSTANT with h → systematic implementation bug")
print("  If errors DECREASE with h → correct implementation, just truncation error")
