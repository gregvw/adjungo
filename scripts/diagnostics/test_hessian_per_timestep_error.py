"""Check which time steps have the worst Hessian errors."""

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
print("PER-TIMESTEP HESSIAN ERROR ANALYSIS")
print("=" * 80)
print()

problem = MildlyNonlinearProblem()
objective = SimpleObjective(y_target=1.0, R=0.1)
method = explicit_euler()

T = 1.0
N_test = 20  # Moderate refinement

print(f"Testing with N={N_test} time steps (h={T/N_test})")
print()

optimizer = GLMOptimizer(
    problem=problem,
    objective=objective,
    method=method,
    t_span=(0.0, T),
    N=N_test,
    y0=np.array([0.0]),
    problem_structure=ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=False,
        has_second_derivatives=True,
    ),
)

u = np.ones((N_test, 1, 1)) * 0.5
v = np.ones((N_test, 1, 1)) * 0.1

# Compute Hessian-vector products
Hv_exact = optimizer.hessian_vector_product(u, v)

eps_fd = 1e-7
grad_0 = optimizer.gradient(u)
grad_eps = optimizer.gradient(u + eps_fd * v)
Hv_fd = (grad_eps - grad_0) / eps_fd

print("Element-wise comparison:")
print()
print(f"{'Step':<6} {'t':<8} {'Exact':<15} {'FD':<15} {'Error':<15} {'Rel Error':<12}")
print("-" * 80)

errors = []
rel_errors = []

for step in range(N_test):
    t = step * T / N_test
    exact_val = Hv_exact[step, 0, 0]
    fd_val = Hv_fd[step, 0, 0]
    error = abs(exact_val - fd_val)
    rel_error = error / abs(fd_val) if abs(fd_val) > 1e-10 else 0.0

    errors.append(error)
    rel_errors.append(rel_error)

    print(f"{step:<6} {t:<8.3f} {exact_val:<15.8f} {fd_val:<15.8f} {error:<15.6e} {rel_error:<12.2%}")

print()
print("-" * 80)
print("Error Statistics:")
print()

errors_array = np.array(errors)
rel_errors_array = np.array(rel_errors)

print(f"Total L2 error:     {np.linalg.norm(errors_array):.6e}")
print(f"Mean error:         {np.mean(errors_array):.6e}")
print(f"Max error:          {np.max(errors_array):.6e} (step {np.argmax(errors_array)})")
print(f"Mean relative:      {np.mean(rel_errors_array):.2%}")
print(f"Max relative:       {np.max(rel_errors_array):.2%} (step {np.argmax(rel_errors_array)})")
print()

# Check if error grows with time
first_third = errors[:N_test//3]
last_third = errors[-N_test//3:]

print(f"Average error in first 1/3 of time steps: {np.mean(first_third):.6e}")
print(f"Average error in last 1/3 of time steps:  {np.mean(last_third):.6e}")
print(f"Ratio (last/first):                       {np.mean(last_third) / np.mean(first_third):.2f}x")
print()

if np.mean(last_third) > 1.5 * np.mean(first_third):
    print("⚠️  Error GROWS with time step index")
    print("   → Bug accumulates through forward propagation")
    print("   → Likely in how sensitivity couples across time steps")
elif np.mean(first_third) > 1.5 * np.mean(last_third):
    print("⚠️  Error LARGER at early time steps")
    print("   → Bug in terminal condition or backward propagation")
else:
    print("✅ Error relatively uniform across time")

print()
print("=" * 80)
print("CONVERGENCE RATE PER TIME STEP")
print("=" * 80)
print()

# Test a few representative time steps
test_steps = [0, N_test//4, N_test//2, 3*N_test//4, N_test-1]
N_values = [10, 20, 40, 80, 160]

for test_step_index in test_steps:
    print(f"\nTime step index {test_step_index}/{N_test-1} (t ≈ {test_step_index/N_test:.2f}T):")
    print(f"{'N':<8} {'h':<12} {'Error':<15} {'Rate':<8}")
    print("-" * 50)

    step_errors = []

    for N in N_values:
        # Map test_step_index from N_test grid to N grid
        # At N_test=20, step 0 is t=0, step 10 is t=0.5, step 19 is t=0.95
        # We want same physical time on different grids
        t_physical = test_step_index * T / N_test
        actual_step = int(round(t_physical * N / T))
        if actual_step >= N:
            actual_step = N - 1

        h = T / N

        opt = GLMOptimizer(
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

        u_test = np.ones((N, 1, 1)) * 0.5
        v_test = np.ones((N, 1, 1)) * 0.1

        Hv_ex = opt.hessian_vector_product(u_test, v_test)
        g0 = opt.gradient(u_test)
        ge = opt.gradient(u_test + eps_fd * v_test)
        Hv_fd_test = (ge - g0) / eps_fd

        if actual_step < N:
            error = abs(Hv_ex[actual_step, 0, 0] - Hv_fd_test[actual_step, 0, 0])
        else:
            error = 0.0

        step_errors.append(error)

        if len(step_errors) > 1:
            rate = np.log(step_errors[-2] / step_errors[-1]) / np.log(N_values[len(step_errors)-2] / N)
            rate_str = f"{rate:.2f}"
        else:
            rate_str = "-"

        print(f"{N:<8} {h:<12.6f} {error:<15.6e} {rate_str:<8}")

    if len(step_errors) > 2:
        avg_rate = np.mean([np.log(step_errors[i] / step_errors[i+1]) /
                            np.log(N_values[i+1] / N_values[i])
                            for i in range(len(step_errors)-1)])

        if avg_rate > 0.8:
            status = f"✅ O(h^{avg_rate:.1f})"
        else:
            status = f"❌ O(h^{avg_rate:.1f})"
        print(f"  Average rate: {avg_rate:.2f} {status}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

# Determine pattern
early_errors = np.array([errors[i] for i in range(min(3, N_test))])
late_errors = np.array([errors[i] for i in range(max(N_test-3, 0), N_test)])

early_avg = np.mean(early_errors)
late_avg = np.mean(late_errors)

print(f"Early time steps: mean error = {early_avg:.6e}")
print(f"Late time steps:  mean error = {late_avg:.6e}")
print()

if late_avg > 2 * early_avg:
    print("❌ Error GROWS significantly with time step index")
    print("   → Bug likely in forward state coupling to Hessian")
    print("   → Each step accumulates error from previous steps")
    print()
    print("Hypothesis: Missing terminal Hessian term that couples all controls")
    print("   The terminal cost J = 0.5*(y(T) - y_target)² creates dense coupling")
    print("   via the chain rule: ∂²J/∂u_i∂u_j = (∂y/∂u_i)^T (∂²J/∂y²) (∂y/∂u_j)")
elif early_avg > 2 * late_avg:
    print("❌ Error LARGER at early time steps")
    print("   → Bug likely in adjoint backward propagation or terminal condition")
elif np.max(rel_errors) > 0.5:
    print("❌ Large relative errors detected")
    print("   → Systematic implementation bug")
else:
    print("✅ Errors relatively uniform")
    print("   → Likely correct implementation with truncation error")
