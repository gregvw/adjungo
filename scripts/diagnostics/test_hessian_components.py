"""Analyze Hessian components to find the bug."""

import numpy as np
from adjungo.optimization.interface import GLMOptimizer
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.methods.runge_kutta import explicit_euler
from adjungo.stepping.sensitivity import forward_sensitivity, adjoint_sensitivity


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


# Setup
problem = MildlyNonlinearProblem()
objective = SimpleObjective(y_target=1.0, R=0.1)
method = explicit_euler()

optimizer = GLMOptimizer(
    problem=problem,
    objective=objective,
    method=method,
    t_span=(0.0, 1.0),
    N=3,  # Small for debugging
    y0=np.array([0.0]),
    problem_structure=ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
    ),
)

# Control and direction
u = np.ones((3, 1, 1)) * 0.5
v = np.ones((3, 1, 1)) * 0.1  # Simple direction for debugging

print("=" * 70)
print("HESSIAN COMPONENT ANALYSIS")
print("=" * 70)
print()

# Forward and adjoint
optimizer._ensure_adjoint(u)
trajectory = optimizer._trajectory
adjoint = optimizer._adjoint

print("Forward trajectory:")
print(f"  Y = {trajectory.Y[:, 0, 0]}")
print()

print("Adjoint:")
print(f"  Lambda = {adjoint.Lambda[:, 0, 0]}")
print(f"  Mu = {adjoint.Mu[:, 0, 0]}")
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

# Check if adjoint sensitivity is being driven properly
print("-" * 70)
print("DIAGNOSING WHY delta_Mu IS ZERO")
print("-" * 70)
print()

# Manual computation for step 2 (last step)
step = 2
cache = trajectory.caches[step]
Lambda_k = adjoint.WeightedAdj[step]

print(f"Step {step}:")
print(f"  Lambda_k = {Lambda_k[0, 0]}")
print(f"  delta_Z = {sens.delta_Z[step, 0, 0]}")
print(f"  delta_u = {v[step, 0, 0]}")
print()

# Compute Gamma
if hasattr(problem, 'F_yy_action'):
    y_k = trajectory.Z[step, 0]
    u_k = u[step, 0]
    t_k = optimizer.t_span[0] + step * optimizer.h

    F_yy_dZ = problem.F_yy_action(y_k, u_k, t_k, sens.delta_Z[step, 0])
    F_yu_du = problem.F_yu_action(y_k, u_k, t_k, v[step, 0])

    print(f"  F_yy_dZ = {F_yy_dZ[0]}")
    print(f"  F_yu_du = {F_yu_du[0]}")
    print(f"  Lambda_k^T @ (F_yy_dZ + F_yu_du) = {Lambda_k[0].T @ (F_yy_dZ + F_yu_du)}")

    Gamma_k = optimizer.h * Lambda_k[0].T @ (F_yy_dZ + F_yu_du)
    print(f"  Gamma_k = {Gamma_k[0]}")
else:
    Gamma_k = 0.0
    print(f"  Gamma_k = 0 (no second derivatives)")

print()

# Compute RHS for adjoint sensitivity
delta_lambda_ext = adj_sens.delta_Lambda[step + 1]
print(f"  delta_Lambda[{step+1}] = {delta_lambda_ext[0, 0]}")

# For explicit Euler: A=0, B=1
# A^T δμ = B^T δλ + Γ
# 0 = 1 * δλ[step+1] + Γ
# So: δμ should come from terminal BC propagating back

# But actually for explicit Euler with A=0:
# The formula becomes: 0 = B^T δλ + Γ which is satisfied trivially
# And: δλ[n-1] = U^T δμ + V^T δλ[n] + J_yy δy

print(f"  Method: A = {method.A}, B = {method.B}")
print(f"  Method: U = {method.U}, V = {method.V}")
print()

# The issue: For explicit Euler (A=0), the adjoint sensitivity reduces to:
# δμ is not solved from A^T δμ = ..., instead it's determined differently

print("For explicit Euler (A=[[0]]):")
print("  A^T δμ = B^T δλ + Γ  becomes:  0 = δλ[n+1] + Γ")
print("  This doesn't determine δμ!")
print()
print("  The δμ should come from the propagation formula:")
print("  δλ[n-1] = U^T δμ + V^T δλ[n] + J_yy δy")
print("  But with U=[[1]], V=[[1]], this gives:")
print("  δλ[n-1] = δμ + δλ[n] + J_yy δy")
print()
print("  For explicit methods, δμ might need to be computed differently!")
print()

# Hessian components
print("-" * 70)
print("HESSIAN COMPONENTS")
print("-" * 70)
print()

hvp = np.zeros_like(u)

for step in range(3):
    cache = trajectory.caches[step]
    Lambda_k = adjoint.WeightedAdj[step]

    # J_uu component
    J_uu = objective.d2J_du2(u[step, 0], step, 0) @ v[step, 0]
    print(f"Step {step}:")
    print(f"  J_uu δu = {J_uu[0]:.6f}")

    # H_uΛ component
    H_uLambda = -optimizer.h * cache.G[0].T @ adj_sens.delta_WeightedAdj[step, 0]
    print(f"  -h G^T δΛ = {H_uLambda[0]:.6f}")

    # H_uZ component (should be zero for F_yu=0)
    if hasattr(problem, 'F_yu_action'):
        F_yu_Lambda = problem.F_yu_action(trajectory.Z[step, 0], u[step, 0],
                                          optimizer.t_span[0] + step * optimizer.h,
                                          Lambda_k[0])
        H_uZ_val = -optimizer.h * sens.delta_Z[step, 0].T @ F_yu_Lambda
        H_uZ = H_uZ_val if np.isscalar(H_uZ_val) else H_uZ_val[0]
        print(f"  -h F_yu[Λ]^T δZ = {H_uZ:.6f}")
    else:
        H_uZ = 0.0

    # H_uu component (should be zero for F_uu=0)
    if hasattr(problem, 'F_uu_action'):
        F_uu_du = problem.F_uu_action(trajectory.Z[step, 0], u[step, 0],
                                      optimizer.t_span[0] + step * optimizer.h,
                                      v[step, 0])
        H_uu_val = -optimizer.h * Lambda_k[0].T @ F_uu_du
        H_uu = H_uu_val if np.isscalar(H_uu_val) else H_uu_val[0]
        print(f"  -h F_uu[Λ] δu = {H_uu:.6f}")
    else:
        H_uu = 0.0

    total = J_uu[0] + H_uLambda[0] + H_uZ + H_uu
    print(f"  Total = {total:.6f}")
    print()

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("The issue is that delta_WeightedAdj is all zeros!")
print("This means the adjoint sensitivity backward sweep is not working.")
print()
print("For explicit methods, the system A^T δμ = B^T δλ + Γ degenerates")
print("because A = 0. We may need special handling for explicit methods.")

# Now compute full Hessian via our method and FD
print("=" * 70)
print("COMPARISON TO FINITE DIFFERENCES")
print("=" * 70)
print()

Hv_exact = optimizer.hessian_vector_product(u, v)
print(f"Exact Hessian-vector product: {Hv_exact.ravel()}")
print()

eps = 1e-6
grad_0 = optimizer.gradient(u)
grad_eps = optimizer.gradient(u + eps * v)
Hv_fd = (grad_eps - grad_0) / eps

print(f"FD Hessian-vector product:    {Hv_fd.ravel()}")
print()

print(f"Difference: {(Hv_exact - Hv_fd).ravel()}")
print(f"Error: {np.linalg.norm(Hv_exact - Hv_fd):.6e}")
print()

# Compare component by component
print("Component-wise comparison:")
for step in range(3):
    print(f"  Step {step}: Exact = {Hv_exact[step,0,0]:.6f}, FD = {Hv_fd[step,0,0]:.6f}, "
          f"Diff = {Hv_exact[step,0,0] - Hv_fd[step,0,0]:.6f}")
