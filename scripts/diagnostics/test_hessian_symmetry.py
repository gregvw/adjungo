"""Test Hessian symmetry and d²J/du² contribution."""

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
    N=5,  # Small for fast testing
    y0=np.array([0.0]),
    problem_structure=ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
    ),
)

# Control point
u = np.ones((5, 1, 1)) * 0.5

print("=" * 70)
print("HESSIAN SYMMETRY TEST")
print("=" * 70)
print(f"\nProblem: dy/dt = -y + u - 0.1*y³")
print(f"Objective: J = 0.5*(y(T) - 1)² + 0.5*R*∫u²")
print(f"Control size: {u.shape} (N=5, s=1, ν=1)")
print()

# Build full Hessian matrix by computing H*e_i for each basis vector
n_control = u.size
print(f"Building full Hessian matrix ({n_control} × {n_control})...")

H = np.zeros((n_control, n_control))

for i in range(n_control):
    # Basis vector
    v = np.zeros_like(u)
    v.flat[i] = 1.0

    # Hessian-vector product
    Hv = optimizer.hessian_vector_product(u, v)
    H[:, i] = Hv.ravel()

print(f"Hessian matrix computed.\n")

# Check symmetry
print("-" * 70)
print("SYMMETRY CHECK")
print("-" * 70)

symmetry_error = np.max(np.abs(H - H.T))
print(f"Max |H - H^T|: {symmetry_error:.6e}")

if symmetry_error < 1e-10:
    print("✅ Hessian is symmetric to machine precision")
elif symmetry_error < 1e-6:
    print("✅ Hessian is approximately symmetric (numerical errors)")
else:
    print("❌ Hessian is NOT symmetric - implementation error!")
    print(f"\nLargest asymmetry at:")
    i, j = np.unravel_index(np.argmax(np.abs(H - H.T)), H.shape)
    print(f"  H[{i},{j}] = {H[i,j]:.6e}")
    print(f"  H[{j},{i}] = {H[j,i]:.6e}")
    print(f"  Difference: {H[i,j] - H[j,i]:.6e}")

print()

# Check d²J/du² contribution
print("-" * 70)
print("d²J/du² CONTRIBUTION")
print("-" * 70)

# Pure objective Hessian (no constraints)
H_objective = np.zeros((n_control, n_control))
for step in range(5):
    for stage in range(1):
        idx = step * 1 + stage
        H_objective[idx, idx] = objective.R

print(f"Objective Hessian (d²J/du²):")
print(f"  Expected: R * I = {objective.R} * I")
print(f"  Diagonal of H: {np.diag(H)}")
print()

# Extract just the d²J/du² contribution from our Hessian
# by looking at diagonal (for quadratic objective with R*I structure)
H_diag = np.diag(H)
objective_contribution = np.mean(H_diag)

print(f"Average diagonal element: {objective_contribution:.6f}")
print(f"Expected (R = {objective.R}): {objective.R:.6f}")
print(f"Difference: {objective_contribution - objective.R:.6e}")
print()

if np.allclose(H_diag, objective.R, atol=1e-3):
    print("✅ d²J/du² = R*I is included correctly")
else:
    print("⚠️  d²J/du² diagonal has variations (coupling from constraints)")

print()

# Eigenvalue check (should all be positive for convex problem)
print("-" * 70)
print("POSITIVE DEFINITENESS CHECK")
print("-" * 70)

eigvals = np.linalg.eigvalsh(H)
min_eigval = np.min(eigvals)
max_eigval = np.max(eigvals)

print(f"Eigenvalue range: [{min_eigval:.6e}, {max_eigval:.6e}]")

if min_eigval > 0:
    print(f"✅ Hessian is positive definite (convex)")
elif min_eigval > -1e-10:
    print(f"✅ Hessian is positive semi-definite (numerically)")
else:
    print(f"⚠️  Hessian has negative eigenvalues (non-convex or error)")

print()

# Finite difference validation
print("-" * 70)
print("FINITE DIFFERENCE VALIDATION")
print("-" * 70)

print("Computing full Hessian via finite differences on gradient...")

eps = 1e-5
H_fd = np.zeros((n_control, n_control))

grad_0 = optimizer.gradient(u).ravel()

for i in range(n_control):
    v = np.zeros_like(u)
    v.flat[i] = eps

    grad_eps = optimizer.gradient(u + v).ravel()
    H_fd[:, i] = (grad_eps - grad_0) / eps

print(f"Finite difference Hessian computed.\n")

# Compare
diff = H - H_fd
max_error = np.max(np.abs(diff))
rel_error = max_error / (np.max(np.abs(H_fd)) + 1e-10)

print(f"Max absolute error: {max_error:.6e}")
print(f"Max relative error: {rel_error:.6e}")
print()

if max_error < 1e-4:
    print("✅ Hessian matches finite differences (< 1e-4)")
elif max_error < 1e-3:
    print("⚠️  Hessian close to finite differences (< 1e-3)")
else:
    print("❌ Hessian differs significantly from finite differences")

print()

# Show where largest errors are
print("Largest errors:")
errors_flat = np.abs(diff).ravel()
top_indices = np.argsort(errors_flat)[-5:][::-1]

for idx in top_indices:
    i, j = np.unravel_index(idx, H.shape)
    print(f"  H[{i},{j}]: exact={H[i,j]:.6e}, FD={H_fd[i,j]:.6e}, "
          f"error={diff[i,j]:.6e}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)

if symmetry_error < 1e-6 and max_error < 1e-3:
    print("✅ Hessian is symmetric and matches finite differences")
    print("   Second-order implementation is correct!")
elif symmetry_error < 1e-6:
    print("✅ Hessian is symmetric")
    print("⚠️  Some numerical differences from finite differences")
else:
    print("❌ Hessian symmetry violation indicates implementation error")

print()
