"""Ultra-simple test: 1 step, 1 stage, dy/dt = u."""

import numpy as np
from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler

class TrivialProblem:
    """dy/dt = u"""
    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return u

    def F(self, y, u, t):
        return np.array([[0.0]])  # ∂f/∂y = 0

    def G(self, y, u, t):
        return np.array([[1.0]])  # ∂f/∂u = 1


class SimpleObjective:
    """J = 0.5 * y(T)^2 + 0.5 * u^2"""

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0, 0]
        return 0.5 * y_final ** 2 + 0.5 * np.sum(u ** 2)

    def dJ_dy_terminal(self, y_final):
        return np.array([[y_final[0, 0]]])

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return u_stage

    def d2J_du2(self, u_stage, step, stage):
        return np.eye(1)


# Setup
problem = TrivialProblem()
objective = SimpleObjective()
method = explicit_euler()

optimizer = GLMOptimizer(
    problem=problem,
    objective=objective,
    method=method,
    t_span=(0.0, 1.0),
    N=1,  # Single step!
    y0=np.array([0.0]),
    problem_structure=ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=False,
    ),
)

# Test with u = 1.0
u = np.array([[[1.0]]])  # (N=1, s=1, ν=1)

# Forward solve
traj = optimizer._ensure_forward(u)
y_final = optimizer._trajectory.Y[-1, 0, 0]
print(f"Forward solve:")
print(f"  u = {u[0, 0, 0]:.6f}")
print(f"  y(T) = y(0) + h*u = 0 + 1.0*1.0 = {y_final:.6f}")
print(f"  J = 0.5*y(T)^2 + 0.5*u^2 = 0.5*1^2 + 0.5*1^2 = {objective.evaluate(optimizer._trajectory, u):.6f}")

# Analytical gradient
# J(u) = 0.5*(h*u)^2 + 0.5*u^2 = 0.5*u^2*(h^2 + 1)
# dJ/du = u*(h^2 + 1)
h = 1.0
grad_analytical = u[0, 0, 0] * (h**2 + 1)
print(f"\nAnalytical gradient:")
print(f"  dJ/du = u*(h^2 + 1) = {u[0, 0, 0]:.6f} * {h**2 + 1:.6f} = {grad_analytical:.6f}")

# Adjoint gradient
grad_adjoint = optimizer.gradient(u)[0, 0, 0]
print(f"\nAdjoint gradient:")
print(f"  dJ/du = {grad_adjoint:.6f}")

# Finite difference
eps = 1e-7
u_pert = u.copy()
u_pert[0, 0, 0] += eps
J0 = optimizer.objective_value(u)
J_pert = optimizer.objective_value(u_pert)
grad_fd = (J_pert - J0) / eps
print(f"\nFinite difference:")
print(f"  dJ/du ≈ {grad_fd:.6f}")

# Debug info
optimizer._ensure_adjoint(u)
adjoint = optimizer._adjoint
print(f"\nAdjoint details:")
print(f"  λ(T) = dJ/dy(T) = y(T) = {adjoint.Lambda[-1, 0, 0]:.6f}")
print(f"  μ = h * F^T * B * λ = {adjoint.Mu[0, 0, 0]:.6f}")
print(f"  Λ = A*μ + B*λ = {adjoint.WeightedAdj[0, 0, 0]:.6f}")
print(f"  Gradient formula: dJ/du - h*G^T*Λ")
print(f"    = {u[0, 0, 0]:.6f} - {h:.6f}*{1.0:.6f}*{adjoint.WeightedAdj[0, 0, 0]:.6f}")
print(f"    = {u[0, 0, 0] - h * 1.0 * adjoint.WeightedAdj[0, 0, 0]:.6f}")
