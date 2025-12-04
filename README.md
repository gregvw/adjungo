# Adjungo

General Linear Method optimization library for optimal control problems.

## Overview

Adjungo provides a framework for solving optimal control problems using General Linear Methods (GLMs) with support for:

- **Forward state solving**: Efficient propagation of state trajectories
- **Backward adjoint computation**: Gradient computation via adjoint methods
- **Sensitivity analysis**: First and second-order sensitivity equations
- **Optimization interface**: Compatible with SciPy and other optimizers

## Features

### Method Support

- **Runge-Kutta methods**: Explicit and implicit RK methods
- **Linear multistep methods**: Adams-Bashforth, Adams-Moulton, BDF
- **IMEX methods**: Implicit-explicit pairs for stiff problems
- **Custom GLM tableaux**: Define your own general linear methods

### Solver Optimizations

- **Factorization reuse**: SDIRK methods reuse LU factorizations across stages
- **Automatic dispatch**: Solver selection based on method structure
- **Efficient adjoints**: Share factorizations between forward and adjoint solves

### Problem Types

- Linear, bilinear, quasilinear, and nonlinear dynamics
- Control-dependent and control-independent Jacobians
- Optional second derivatives for Newton-type optimization

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from adjungo import GLMOptimizer
from adjungo.methods.runge_kutta import rk4

# Define your problem
class MyProblem:
    @property
    def state_dim(self):
        return 2

    @property
    def control_dim(self):
        return 1

    def f(self, y, u, t):
        # Dynamics: ẏ = f(y, u, t)
        return np.array([y[1], u[0] - y[0]])

    def F(self, y, u, t):
        # State Jacobian ∂f/∂y
        return np.array([[0, 1], [-1, 0]])

    def G(self, y, u, t):
        # Control Jacobian ∂f/∂u
        return np.array([[0], [1]])

# Define objective function
class MyObjective:
    def evaluate(self, trajectory, u):
        # J = ||y(T) - y_target||^2 + ||u||^2
        y_final = trajectory.Y[-1, 0]
        return np.sum((y_final - self.y_target)**2) + 0.01 * np.sum(u**2)

    def dJ_dy_terminal(self, y_final):
        return np.array([2 * (y_final - self.y_target), [0]])

    def dJ_dy(self, y, step):
        return np.zeros((1, 2))

    def dJ_du(self, u_stage, step, stage):
        return 0.02 * u_stage

    def d2J_du2(self, u_stage, step, stage):
        return 0.02 * np.eye(1)

# Setup optimizer
problem = MyProblem()
objective = MyObjective()
method = rk4()

optimizer = GLMOptimizer(
    problem=problem,
    objective=objective,
    method=method,
    t_span=(0.0, 10.0),
    N=100,
    y0=np.array([1.0, 0.0]),
)

# Use with scipy.optimize
from scipy.optimize import minimize

fun, jac = optimizer.scipy_interface()
u_init = np.zeros((100, 4, 1))  # N=100 steps, s=4 stages, ν=1 control

result = minimize(fun, u_init.ravel(), jac=jac, method='L-BFGS-B')
u_optimal = result.x.reshape(100, 4, 1)
```

## Architecture

The library is organized into several modules:

- `adjungo.core`: Problem and method specifications
- `adjungo.algebra`: Linear algebra backend abstractions
- `adjungo.solvers`: Stage equation solvers (explicit, DIRK, SDIRK, implicit)
- `adjungo.stepping`: Forward/backward stepping algorithms
- `adjungo.optimization`: Gradient and Hessian assembly
- `adjungo.methods`: Standard method tableaux library
- `adjungo.utils`: Utility functions

## Documentation

See the `docs/` directory for detailed mathematical formulation and implementation notes:

- `docs/architecture.md`: High-level architecture overview
- `docs/python_implementation.md`: Detailed implementation guide
- `docs/glm_opt.tex`: Mathematical framework
- `docs/linalg_requirements.tex`: Linear algebra optimization strategies

## Development

Run tests:

```bash
pytest
```

Format code:

```bash
black adjungo/
```

Type checking:

```bash
mypy adjungo/
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
