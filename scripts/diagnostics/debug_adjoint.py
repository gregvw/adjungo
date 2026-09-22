"""Debug script to understand adjoint gradient mismatch."""

import numpy as np
from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler

class SimpleProblem:
    """dy/dt = -y + u"""
    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return -y + u

    def F(self, y, u, t):
        return np.array([[-1.0]])

    def G(self, y, u, t):
        return np.array([[1.0]])


class SimpleObjective:
    """J = 0.5 * (y(T) - 1)^2 + 0.5 * sum(u^2)"""

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0, 0]
        return 0.5 * (y_final - 1.0) ** 2 + 0.5 * np.sum(u ** 2)

    def dJ_dy_terminal(self, y_final):
        return np.array([[y_final[0, 0] - 1.0]])

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return u_stage

    def d2J_du2(self, u_stage, step, stage):
        return np.eye(1)


# Simple test case
np.random.seed(42)
problem = SimpleProblem()
objective = SimpleObjective()
method = explicit_euler()

optimizer = GLMOptimizer(
    problem=problem,
    objective=objective,
    method=method,
    t_span=(0.0, 1.0),
    N=3,  # Just 3 steps for debugging
    y0=np.array([0.0]),
    problem_structure=ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=False,
    ),
)

# Simple control
u = np.ones((3, 1, 1)) * 0.1

# Compute gradient via adjoint
grad_adjoint = optimizer.gradient(u)

print("=" * 60)
print("ADJOINT GRADIENT")
print("=" * 60)
print(f"Adjoint gradient:\n{grad_adjoint.ravel()}")

# Compute gradient via finite difference
eps = 1e-6
grad_fd = np.zeros_like(u)
J0 = optimizer.objective_value(u)

for i in range(3):
    u_pert = u.copy()
    u_pert[i, 0, 0] += eps
    J_pert = optimizer.objective_value(u_pert)
    grad_fd[i, 0, 0] = (J_pert - J0) / eps

print("\n" + "=" * 60)
print("FINITE DIFFERENCE GRADIENT")
print("=" * 60)
print(f"FD gradient:\n{grad_fd.ravel()}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Difference:\n{(grad_adjoint - grad_fd).ravel()}")
print(f"Relative error: {np.linalg.norm(grad_adjoint - grad_fd) / np.linalg.norm(grad_fd):.6f}")

# Let's also inspect the adjoint trajectory
optimizer._ensure_adjoint(u)
adjoint = optimizer._adjoint
trajectory = optimizer._trajectory

print("\n" + "=" * 60)
print("TRAJECTORY INFO")
print("=" * 60)
print(f"Final state: y(T) = {trajectory.Y[-1, 0, 0]:.6f}")
print(f"Terminal adjoint: λ(T) = {adjoint.Lambda[-1, 0, 0]:.6f}")
print(f"\nMu values:")
for i in range(3):
    print(f"  μ[{i}] = {adjoint.Mu[i, 0, 0]:.6f}")
print(f"\nWeighted adjoints:")
for i in range(3):
    print(f"  Λ[{i}] = {adjoint.WeightedAdj[i, 0, 0]:.6f}")

# Check gradient formula manually
print("\n" + "=" * 60)
print("MANUAL GRADIENT CALCULATION")
print("=" * 60)
h = 1.0 / 3.0
for i in range(3):
    cache = trajectory.caches[i]
    dJ_du = u[i, 0, 0]  # From objective
    G_k = cache.G[0]  # (1, 1)
    Lambda_k = adjoint.WeightedAdj[i, 0, 0]

    grad_manual = dJ_du - h * G_k.T @ np.array([Lambda_k])

    print(f"Step {i}:")
    print(f"  dJ/du = {dJ_du:.6f}")
    print(f"  G_k = {G_k[0, 0]:.6f}")
    print(f"  Λ_k = {Lambda_k:.6f}")
    print(f"  h * G^T * Λ = {h * G_k.T @ np.array([Lambda_k])[0]:.6f}")
    print(f"  grad_manual = {grad_manual[0]:.6f}")
    print(f"  grad_adjoint[{i}] = {grad_adjoint[i, 0, 0]:.6f}")
    print(f"  grad_fd[{i}] = {grad_fd[i, 0, 0]:.6f}")
