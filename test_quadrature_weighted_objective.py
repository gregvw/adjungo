"""Test if proper quadrature weighting fixes the Hessian."""

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


class QuadratureWeightedObjective:
    """Objective with PROPER quadrature weights.

    For DBO with quadrature: ∫₀ᵀ R*u² dt ≈ Σₙ h*Σᵢ b_i*R*u_{n,i}²
    """

    def __init__(self, y_target, R, h, method):
        self.y_target = y_target
        self.R = R
        self.h = h
        self.method = method

    def evaluate(self, trajectory, u):
        y_final = trajectory.Y[-1, 0, 0]
        terminal_cost = 0.5 * (y_final - self.y_target) ** 2

        # Proper quadrature-weighted control penalty
        control_cost = 0.0
        N, s, nu = u.shape
        for n in range(N):
            for i in range(s):
                # Quadrature: h * b_i * R * u_{n,i}²
                control_cost += self.h * self.method.B[0, i] * 0.5 * self.R * u[n, i, 0]**2

        return terminal_cost + control_cost

    def dJ_dy_terminal(self, y_final):
        return np.array([[y_final[0, 0] - self.y_target]])

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        # Gradient with quadrature weight
        return self.h * self.method.B[0, stage] * self.R * u_stage

    def d2J_du2(self, u_stage, step, stage):
        # Hessian with quadrature weight
        return self.h * self.method.B[0, stage] * self.R * np.eye(len(u_stage))


class UnweightedObjective:
    """Standard objective WITHOUT quadrature weights (current implementation)."""

    def __init__(self, y_target, R):
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
method = explicit_euler()
N = 5
h = 1.0 / N
u = np.ones((N, 1, 1)) * 0.5

print("=" * 80)
print("QUADRATURE-WEIGHTED OBJECTIVE TEST")
print("=" * 80)
print(f"\nExplicit Euler: b = {method.B[0,:]}, c = {method.c}")
print(f"Time step: h = {h}")
print(f"Regularization: R = 0.1")
print()

# Test 1: Unweighted objective (current implementation)
print("-" * 80)
print("TEST 1: UNWEIGHTED OBJECTIVE (current)")
print("-" * 80)
print()

objective_unweighted = UnweightedObjective(y_target=1.0, R=0.1)

optimizer_unweighted = GLMOptimizer(
    problem=problem,
    objective=objective_unweighted,
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

print("Expected diagonal Hessian: d²J/du² = R = 0.1")
print()

# Build Hessian diagonal
v = np.zeros_like(u)
v[0, 0, 0] = 1.0
Hv = optimizer_unweighted.hessian_vector_product(u, v)
print(f"Exact H[0,0] = {Hv[0,0,0]:.6f}")

# FD check
eps = 1e-5
grad_0 = optimizer_unweighted.gradient(u)
grad_eps = optimizer_unweighted.gradient(u + eps * v)
H00_fd = (grad_eps[0,0,0] - grad_0[0,0,0]) / eps
print(f"FD    H[0,0] = {H00_fd:.6f}")
print(f"Error        = {abs(Hv[0,0,0] - H00_fd):.6e}")
print()

# Test 2: Quadrature-weighted objective (proper DBO)
print("-" * 80)
print("TEST 2: QUADRATURE-WEIGHTED OBJECTIVE (proper DBO)")
print("-" * 80)
print()

objective_weighted = QuadratureWeightedObjective(
    y_target=1.0, R=0.1, h=h, method=method
)

optimizer_weighted = GLMOptimizer(
    problem=problem,
    objective=objective_weighted,
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

print(f"Expected diagonal Hessian: d²J/du² = h*b*R = {h}*1*0.1 = {h*0.1:.6f}")
print()

# Build Hessian diagonal
Hv_weighted = optimizer_weighted.hessian_vector_product(u, v)
print(f"Exact H[0,0] = {Hv_weighted[0,0,0]:.6f}")

# FD check
grad_0_weighted = optimizer_weighted.gradient(u)
grad_eps_weighted = optimizer_weighted.gradient(u + eps * v)
H00_fd_weighted = (grad_eps_weighted[0,0,0] - grad_0_weighted[0,0,0]) / eps
print(f"FD    H[0,0] = {H00_fd_weighted:.6f}")
print(f"Error        = {abs(Hv_weighted[0,0,0] - H00_fd_weighted):.6e}")
print()

# Comparison
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

print("Unweighted objective:")
print(f"  Expected: R = 0.1")
print(f"  Exact:    {Hv[0,0,0]:.6f}")
print(f"  FD:       {H00_fd:.6f}")
print(f"  Match:    {np.isclose(Hv[0,0,0], H00_fd, atol=1e-4)}")
print()

print("Quadrature-weighted objective:")
print(f"  Expected: h*R = {h*0.1:.6f}")
print(f"  Exact:    {Hv_weighted[0,0,0]:.6f}")
print(f"  FD:       {H00_fd_weighted:.6f}")
print(f"  Match:    {np.isclose(Hv_weighted[0,0,0], H00_fd_weighted, atol=1e-4)}")
print()

print("=" * 80)
print("CONVERGENCE TEST WITH QUADRATURE WEIGHTS")
print("=" * 80)
print()

print("Testing if quadrature weighting fixes h-independence...")
print()
print(f"{'h':<12} {'||Exact - FD||':<15} {'Rate':<10}")
print("-" * 40)

eps_test = 1e-5
errors_weighted = []
h_values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

for h_test in h_values:
    # Rebuild with new h
    v_test = np.zeros((1, 1, 1))
    v_test[0, 0, 0] = 1.0
    u_test = np.ones((1, 1, 1)) * 0.5

    obj_test = QuadratureWeightedObjective(1.0, 0.1, h_test, method)
    opt_test = GLMOptimizer(
        problem=problem,
        objective=obj_test,
        method=method,
        t_span=(0.0, h_test),  # Single step
        N=1,
        y0=np.array([0.0]),
        problem_structure=ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=False,
            has_second_derivatives=True,
        ),
    )

    Hv_test = opt_test.hessian_vector_product(u_test, v_test)
    grad_0_test = opt_test.gradient(u_test)
    grad_eps_test = opt_test.gradient(u_test + eps_test * v_test)
    H_fd_test = (grad_eps_test[0,0,0] - grad_0_test[0,0,0]) / eps_test

    error = abs(Hv_test[0,0,0] - H_fd_test)
    errors_weighted.append(error)

    if len(errors_weighted) > 1:
        rate = np.log(errors_weighted[-2] / errors_weighted[-1]) / \
               np.log(h_values[len(errors_weighted)-2] / h_test)
        rate_str = f"{rate:.2f}"
    else:
        rate_str = "-"

    print(f"{h_test:<12.1e} {error:<15.6e} {rate_str:<10}")

print()
if all(abs(errors_weighted[i] - errors_weighted[0]) < 1e-10 for i in range(len(errors_weighted))):
    print("❌ Error still constant with h - quadrature weighting doesn't fix the bug")
else:
    print("✅ Error varies with h - quadrature weighting might help!")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if np.isclose(Hv[0,0,0], H00_fd, atol=1e-4) and \
   np.isclose(Hv_weighted[0,0,0], H00_fd_weighted, atol=1e-4):
    print("✅ Both objectives show exact Hessian matching FD")
    print("   → Quadrature weighting is not the issue")
elif np.isclose(Hv_weighted[0,0,0], H00_fd_weighted, atol=1e-4) and \
     not np.isclose(Hv[0,0,0], H00_fd, atol=1e-4):
    print("✅ Quadrature weighting FIXES the Hessian bug!")
    print("   → Need to use proper quadrature-weighted objectives")
else:
    print("⚠️  Neither objective matches FD properly")
    print("   → Bug is elsewhere (not just quadrature weighting)")
