"""Debug Crank-Nicolson implementation."""

import numpy as np
from adjungo.methods.runge_kutta import implicit_trapezoid
from adjungo.solvers.factory import create_stage_solver
from adjungo.core.requirements import deduce_requirements
from adjungo.core.problem import ProblemStructure, Linearity

# Check method classification
method = implicit_trapezoid()
print(f"Method: {method}")
print(f"Stage type: {method.stage_type}")
print(f"s = {method.s}")
print(f"A = \n{method.A}")
print(f"B = {method.B}")
print(f"c = {method.c}")
print()

# Check solver selection
problem_structure = ProblemStructure(
    linearity=Linearity.LINEAR,
    jacobian_constant=True,
    jacobian_control_dependent=False,
    has_second_derivatives=False,
)

requirements = deduce_requirements(method, problem_structure, state_dim=2)
solver = create_stage_solver(method, requirements, problem_structure)

print(f"Solver: {solver.__class__.__name__}")
print()

# Manual Crank-Nicolson step
A_matrix = np.array([[0.0, 1.0], [-1.0, -0.1]])
B_matrix = np.array([[0.0], [1.0]])
y0 = np.array([1.0, 0.0])
h = 0.1
u = np.array([0.0])

# Analytical Crank-Nicolson:
# y_1 = y_0 + h/2 * [f(y_0, u_0) + f(y_1, u_1)]
# y_1 = y_0 + h/2 * [A*y_0 + B*u_0 + A*y_1 + B*u_1]
# y_1 - h/2 * A * y_1 = y_0 + h/2 * (A*y_0 + B*u_0 + B*u_1)
# (I - h/2 * A) * y_1 = (I + h/2 * A) * y_0 + h/2 * B * (u_0 + u_1)

I = np.eye(2)
lhs_matrix = I - 0.5 * h * A_matrix
rhs_value = (I + 0.5 * h * A_matrix) @ y0  # Zero control

y1_analytical = np.linalg.solve(lhs_matrix, rhs_value)

print(f"Manual Crank-Nicolson step:")
print(f"y0 = {y0}")
print(f"y1 = {y1_analytical}")
print()

# Now use the solver
class TestProblem:
    state_dim = 2
    control_dim = 1

    def f(self, y, u, t):
        return A_matrix @ y + B_matrix @ u

    def F(self, y, u, t):
        return A_matrix

    def G(self, y, u, t):
        return B_matrix

problem = TestProblem()
y_history = np.array([y0])
u_stages = np.array([[0.0], [0.0]])  # Two stages
t_n = 0.0

Z, cache = solver.solve_stages(y_history, u_stages, t_n, h, problem, method)

print(f"Solver output:")
print(f"Z[0] (first stage) = {Z[0]}")
print(f"Z[1] (second stage) = {Z[1]}")
print()

# Compute y_{n+1} using GLM formula
y_next = method.V @ y_history[0] + method.B @ np.array([problem.f(Z[i], u_stages[i], t_n + method.c[i]*h) for i in range(method.s)])
print(f"y_{{n+1}} via GLM = {y_next[0]}")
print()

# Compare
print(f"Analytical y1: {y1_analytical}")
print(f"Numerical y1:  {y_next[0]}")
print(f"Match: {np.allclose(y1_analytical, y_next[0])}")
