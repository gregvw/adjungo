"""Test if adding terminal Hessian term fixes the bug."""

import numpy as np
from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler
from adjungo.stepping.sensitivity import forward_sensitivity


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

    def d2J_dy2_terminal(self):
        """Second derivative of terminal cost w.r.t. state."""
        return np.array([[1.0]])  # For φ(y) = 0.5*(y - y_target)²


print("=" * 80)
print("TERMINAL HESSIAN CONTRIBUTION TEST")
print("=" * 80)
print()

problem = MildlyNonlinearProblem()
objective = SimpleObjective(y_target=1.0, R=0.1)
method = explicit_euler()

N = 20
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

u = np.ones((N, 1, 1)) * 0.5
v = np.ones((N, 1, 1)) * 0.1

print(f"Problem: N={N} time steps")
print(f"Control: u = {u[0,0,0]}, v = {v[0,0,0]}")
print()

# Compute existing Hessian (WITHOUT terminal term)
Hv_original = optimizer.hessian_vector_product(u, v)

# FD reference
eps_fd = 1e-7
grad_0 = optimizer.gradient(u)
grad_eps = optimizer.gradient(u + eps_fd * v)
Hv_fd = (grad_eps - grad_0) / eps_fd

error_original = np.linalg.norm(Hv_original - Hv_fd)

print("-" * 80)
print("ORIGINAL IMPLEMENTATION (without terminal Hessian)")
print("-" * 80)
print(f"||Hv_original - Hv_FD|| = {error_original:.6e}")
print()

# Now compute the MISSING terminal Hessian contribution manually
print("-" * 80)
print("COMPUTING TERMINAL HESSIAN CONTRIBUTION")
print("-" * 80)
print()

# We need (∂y/∂u)^T (∂²φ/∂y²) (∂y/∂u)
# This requires computing ∂y(T)/∂u_i for each control variable

# Compute sensitivity of y(T) to each control variable
optimizer._ensure_adjoint(u)
trajectory = optimizer._trajectory

# Build the full sensitivity matrix: ∂y(T)/∂u
# Shape: (n_state, n_controls)
n_controls = N * 1 * 1  # N steps, 1 stage, 1 control per stage
dy_du = np.zeros((1, n_controls))  # scalar state, so n_state=1

for i in range(N):
    # Create unit perturbation at control i
    v_i = np.zeros((N, 1, 1))
    v_i[i, 0, 0] = 1.0

    # Forward sensitivity
    sens_i = forward_sensitivity(
        trajectory, v_i, method, optimizer.stage_solver, problem, optimizer.h
    )

    # Extract terminal state sensitivity
    dy_du[0, i] = sens_i.delta_Y[-1, 0, 0]

print(f"Sensitivity matrix ∂y(T)/∂u: shape = {dy_du.shape}")
print(f"  ||∂y(T)/∂u|| = {np.linalg.norm(dy_du):.6f}")
print()

# Terminal Hessian: H_terminal = (∂y/∂u)^T (∂²φ/∂y²) (∂y/∂u)
d2J_dy2 = objective.d2J_dy2_terminal()  # For quadratic: [[1.0]]
H_terminal_matrix = dy_du.T @ d2J_dy2 @ dy_du

print(f"Terminal Hessian contribution: shape = {H_terminal_matrix.shape}")
print(f"  ||H_terminal|| = {np.linalg.norm(H_terminal_matrix):.6f}")
print(f"  Sparsity: {np.count_nonzero(H_terminal_matrix)} / {H_terminal_matrix.size} non-zero")
print()

# Compute H_terminal @ v
H_terminal_v = H_terminal_matrix @ v.ravel()
H_terminal_v = H_terminal_v.reshape((N, 1, 1))

print(f"Terminal contribution to Hv:")
print(f"  ||H_terminal @ v|| = {np.linalg.norm(H_terminal_v):.6e}")
print()

# Corrected Hessian-vector product
Hv_corrected = Hv_original + H_terminal_v

error_corrected = np.linalg.norm(Hv_corrected - Hv_fd)

print("-" * 80)
print("CORRECTED IMPLEMENTATION (with terminal Hessian)")
print("-" * 80)
print(f"||Hv_corrected - Hv_FD|| = {error_corrected:.6e}")
print()

print("-" * 80)
print("COMPARISON")
print("-" * 80)
print(f"Original error:  {error_original:.6e}")
print(f"Corrected error: {error_corrected:.6e}")
print(f"Improvement:     {error_original / error_corrected:.2f}x")
print()

if error_corrected < 1e-4:
    print("✅ ERROR FIXED! Terminal Hessian was the missing piece.")
elif error_corrected < error_original / 2:
    print("✅ SIGNIFICANT IMPROVEMENT! Terminal Hessian is part of the fix.")
    print(f"   Remaining error may be from discretization or other terms.")
elif error_corrected < error_original * 0.9:
    print("⚠️  MODEST IMPROVEMENT. Terminal Hessian helps but not enough.")
else:
    print("❌ NO IMPROVEMENT. Terminal Hessian is not the issue.")

print()
print("=" * 80)
print("ELEMENT-WISE COMPARISON")
print("=" * 80)
print()

print(f"{'Step':<6} {'Original':<15} {'Corrected':<15} {'FD':<15} {'Err (orig)':<12} {'Err (corr)':<12}")
print("-" * 90)

for step in range(min(N, 10)):  # Show first 10 steps
    orig = Hv_original[step, 0, 0]
    corr = Hv_corrected[step, 0, 0]
    fd = Hv_fd[step, 0, 0]
    err_orig = abs(orig - fd)
    err_corr = abs(corr - fd)

    print(f"{step:<6} {orig:<15.8f} {corr:<15.8f} {fd:<15.8f} {err_orig:<12.6e} {err_corr:<12.6e}")

print()
print("=" * 80)
print("TERMINAL HESSIAN MATRIX STRUCTURE")
print("=" * 80)
print()

# Show a few elements of the terminal Hessian
print("Sample elements of H_terminal:")
print(f"  H[0,0]   = {H_terminal_matrix[0,0]:.8f}")
print(f"  H[0,N/2] = {H_terminal_matrix[0,N//2]:.8f}")
print(f"  H[0,N-1] = {H_terminal_matrix[0,N-1]:.8f}")
print(f"  H[N-1,N-1] = {H_terminal_matrix[N-1,N-1]:.8f}")
print()

# Check if it's rank-1
U, S, Vt = np.linalg.svd(H_terminal_matrix)
print(f"Singular values of H_terminal (largest 5):")
for i, s in enumerate(S[:5]):
    print(f"  σ_{i} = {s:.8f}")
print()

if S[1] / S[0] < 1e-10:
    print("✅ H_terminal is rank-1 as expected (σ₁/σ₀ < 1e-10)")
else:
    print(f"⚠️  H_terminal appears to have rank > 1 (σ₁/σ₀ = {S[1]/S[0]:.2e})")
