"""Debug adjoint sensitivity computation."""

import numpy as np
from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler
from adjungo.stepping.sensitivity import forward_sensitivity, adjoint_sensitivity


class SimpleLinearProblem:
    """dy/dt = -y + u (linear for easier debugging)"""
    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return -y + u

    def F(self, y, u, t):
        return np.array([[-1.0]])

    def G(self, y, u, t):
        return np.array([[1.0]])

    def F_yy_action(self, y, u, t, v):
        return np.zeros((1,))

    def F_yu_action(self, y, u, t, v_u):
        return np.zeros((1,))

    def F_uu_action(self, y, u, t, v_u):
        return np.zeros((1,))


class SimpleObjective:
    def __init__(self, R=0.1):
        self.R = R

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0, 0]
        return 0.5 * y_final**2 + 0.5 * self.R * np.sum(u**2)

    def dJ_dy_terminal(self, y_final):
        return np.array([[y_final[0, 0]]])

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return self.R * u_stage

    def d2J_du2(self, u_stage, step, stage):
        return self.R * np.eye(len(u_stage))


# Setup
problem = SimpleLinearProblem()
objective = SimpleObjective(R=0.1)
method = explicit_euler()

optimizer = GLMOptimizer(
    problem=problem,
    objective=objective,
    method=method,
    t_span=(0.0, 1.0),
    N=3,
    y0=np.array([0.0]),
    problem_structure=ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
    ),
)

# Control and direction
u = np.ones((3, 1, 1)) * 0.5
v = np.zeros((3, 1, 1))
v[1, 0, 0] = 1.0  # Perturbation at step 1

print("=" * 70)
print("DEBUGGING ADJOINT SENSITIVITY")
print("=" * 70)
print(f"\nProblem: dy/dt = -y + u (linear)")
print(f"Control: u = {u.ravel()}")
print(f"Perturbation: v = {v.ravel()}")
print()

# Forward and adjoint solve
optimizer._ensure_adjoint(u)
trajectory = optimizer._trajectory
adjoint = optimizer._adjoint

print("Forward trajectory:")
print(f"  Y = {trajectory.Y[:, 0, 0]}")
print()

print("Adjoint:")
print(f"  Lambda = {adjoint.Lambda[:, 0, 0]}")
print(f"  WeightedAdj = {adjoint.WeightedAdj[:, 0, 0]}")
print()

# Forward sensitivity
sens = forward_sensitivity(
    trajectory, v, method, optimizer.stage_solver, optimizer.problem, optimizer.h
)

print("Forward sensitivity:")
print(f"  delta_Y = {sens.delta_Y[:, 0, 0]}")
print(f"  delta_Z = {sens.delta_Z[:, 0, 0]}")
print()

# Adjoint sensitivity
adj_sens = adjoint_sensitivity(
    trajectory, adjoint, sens, u, v, method,
    optimizer.stage_solver, optimizer.problem, optimizer.h,
    optimizer.t_span[0], optimizer.objective
)

print("Adjoint sensitivity:")
print(f"  delta_Lambda = {adj_sens.delta_Lambda[:, 0, 0]}")
print(f"  delta_Mu = {adj_sens.delta_Mu[:, 0, 0]}")
print(f"  delta_WeightedAdj = {adj_sens.delta_WeightedAdj[:, 0, 0]}")
print()

# Hessian-vector product
Hv = optimizer.hessian_vector_product(u, v)

print("Hessian-vector product:")
print(f"  Hv = {Hv.ravel()}")
print()

# Finite difference
eps = 1e-6
grad_0 = optimizer.gradient(u)
grad_eps = optimizer.gradient(u + eps * v)
Hv_fd = (grad_eps - grad_0) / eps

print("Finite difference:")
print(f"  Hv_FD = {Hv_fd.ravel()}")
print()

print("Comparison:")
print(f"  Difference = {(Hv - Hv_fd).ravel()}")
print(f"  Max error = {np.max(np.abs(Hv - Hv_fd)):.6e}")
print()

# Check if delta_WeightedAdj is being computed correctly
print("Checking delta_WeightedAdj formula:")
print(f"  Method A = {method.A}")
print(f"  Method B = {method.B}")
print()

for step in range(3):
    print(f"Step {step}:")
    print(f"  delta_Mu = {adj_sens.delta_Mu[step, :, 0]}")
    print(f"  delta_Lambda[{step+1}] = {adj_sens.delta_Lambda[step+1, 0, 0]}")

    # Manual computation
    delta_Λ_manual = (
        method.A[:, 0] @ adj_sens.delta_Mu[step] +
        method.B[:, 0] @ adj_sens.delta_Lambda[step + 1]
    )
    print(f"  delta_Λ (manual) = {delta_Λ_manual[0]}")
    print(f"  delta_WeightedAdj (stored) = {adj_sens.delta_WeightedAdj[step, 0, 0]}")
    print()
