# Adjungo

General Linear Method optimization library for optimal control problems.

## Why "adjungo"?

The name comes from the Latin verb adjungō: "I join to" (from ad- "to, toward" + jungō "I join, yoke"). This is the direct ancestor of the English word "adjoint"—the French adjoindre descends from the Latin infinitive adjungere.
The etymology captures the mathematical relationship precisely. In the adjoint method, we don't merely combine equations symmetrically; we attach the adjoint system to the primal problem, exploiting its structure to compute gradients efficiently. The directional sense of ad- reflects this dependency.
The first-person singular form follows Latin verbs that survive in English as active declarations: audio ("I hear"), video ("I see"), adjungo ("I join to"). The library announces what it does.

## Overview

Adjungo computes **exact discrete derivatives** for optimal-control problems
discretised by General Linear Methods.

"Exact discrete" is the whole claim, and it is narrower and stronger than it
may sound. The gradient and Hessian-vector product adjungo returns are the
derivatives of the objective **as the code actually discretises it, at the
mesh you give it** — not approximations to the derivatives of the underlying
continuous problem. A disagreement with a finite difference of that same
discrete objective is a defect, never "time-discretisation error". Convergence
to the continuous problem is a separate claim, tested separately.

The governing contract is [`NUMERICS.md`](NUMERICS.md). Where this README and
that document disagree, `NUMERICS.md` is authoritative.

## What is supported

Adjungo **refuses at construction** rather than returning a plausible-looking
number from an uncertified path. A method family appears below as certified
only when a completed milestone validated its forward solve, gradient **and**
Hessian-vector product against an independently assembled reference.

| Method family | Status |
|---|---|
| Explicit Runge–Kutta (`explicit_euler`, `heun`, `rk4`) | **certified** |
| DIRK (`implicit_trapezoid` / Crank–Nicolson) | **certified** |
| SDIRK (`implicit_midpoint`, `sdirk2`, `sdirk3`) | **certified** |
| Fully implicit, dense `A` (`gauss2`) | **certified** — one coupled `(s·n)` Newton solve per step |
| BDF (`bdf2`, `bdf3`) | **refused** — `r > 1` has no starting procedure |
| Adams (`adams_bashforth2`, `adams_moulton2`) | **refused** — tableau not representable |
| IMEX / additive splitting | **refused** |

Tableaux for the refused families are kept under
`adjungo/methods/experimental/` as reference material, not as endorsement.
`GLMOptimizer` raises `NotImplementedError` for the ones that are
representable; the Adams tableaux need per-history stage coefficients and are
rejected earlier, by `GLMethod` validation itself.

### Problem types

Linear, bilinear, quasilinear and nonlinear dynamics; control-dependent and
control-independent Jacobians.

Second derivatives (`F_yy_action`, `F_yu_action`, `F_uu_action`, and the
objective's `d2J_dy2` and `d2J_dy2_terminal`) are **required** for the exact
Hessian-vector product. If they are missing, adjungo raises. It does not
silently fall back to a Gauss–Newton operator, because that is a different
operator and the method promises the exact Hessian.

### Control parametrisation

Derivatives are computed with respect to the *stage* controls
`u ∈ ℝ^(N×s×ν)`, which is where the exactness claim is stated. That is rarely
the layer you want to optimise in: stage controls are tied to the tableau, so
switching from `heun` to `rk4` changes the number of unknowns even though the
physical control being sought did not.

An optional adapter layer maps a parameter vector `θ` to stage controls and
maps derivatives back. For an affine map `u = Pθ + q`, `g_θ = Pᵀg_u` and
`H_θv = PᵀH_u(Pv)`. Two maps ship, both affine:

```python
from adjungo import NodalControl, PiecewiseConstantControl

# One value per step, held across that step's stages: N*nu unknowns,
# independent of the tableau.
theta_map = PiecewiseConstantControl(N, method.s, control_dim=1)

# Or: values at the N+1 step nodes, linearly interpolated to the stage
# abscissae, giving a control continuous across step boundaries.
theta_map = NodalControl(N, control_dim=1, c=method.c)

fun, jac = optimizer.scipy_interface(parametrization=theta_map)
hessp = optimizer.scipy_hessp(parametrization=theta_map)
result = minimize(fun, np.zeros(theta_map.n_parameters), jac=jac,
                  hessp=hessp, method="trust-ncg")
u_stages = theta_map.expand(result.x)
```

The gradient returned is the **coordinate** derivative in the flat variable
vector, not a Riesz representative under an `h`-weighted inner product.
`scipy.optimize.minimize` interprets `jac` that way, so returning a
representative would cause silently wrong steps rather than an error.

Nonlinear parametrisations are specified in `NUMERICS.md` C-10.3 and are not
implemented. They need an extra curvature term; omitting it yields a
Gauss-Newton operator, which may not be presented as the exact Hessian.

### Not yet implemented

- Factorisation **reuse** across stages or steps. Each stage is factored at its
  own converged iterate. An earlier reuse path was removed after it was found
  to hand the adjoint another stage's matrix; see precedent R-9 in
  `NUMERICS.md`.
- Nonlinear control parametrisation, sparse operators, and checkpointing.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Quick start

The complete, runnable version of the code below is
[`examples/minimum_energy_oscillator.py`](examples/minimum_energy_oscillator.py),
which is executed by the test suite. Run it with:

```bash
.venv/bin/python examples/minimum_energy_oscillator.py
```

It drives a damped mass–spring oscillator (`m = 1 kg`, `k = 4 N/m`,
`c = 0.4 N·s/m`, so `ω₀ = 2 rad/s` and `ζ = 0.1`) from rest at `x = 1 m` to
rest at the origin over 4 s, minimising control energy.

### Define the dynamics

```python
import numpy as np

class DampedOscillator:
    state_dim = 2      # y = (position, velocity)
    control_dim = 1    # u = applied force

    def __init__(self, mass=1.0, stiffness=4.0, damping=0.4):
        self.mass, self.stiffness, self.damping = mass, stiffness, damping

    def f(self, y, u, t):
        x, v = y[0], y[1]
        return np.array(
            [v, (u[0] - self.stiffness * x - self.damping * v) / self.mass]
        )

    def F(self, y, u, t):          # df/dy
        return np.array([[0.0, 1.0],
                         [-self.stiffness / self.mass, -self.damping / self.mass]])

    def G(self, y, u, t):          # df/du
        return np.array([[0.0], [1.0 / self.mass]])

    # Required for the exact Hessian. Exactly zero here because f is affine
    # in (y, u) -- a fact about this problem, not a simplification. adjungo
    # cannot infer it, and raises rather than assuming it.
    def F_yy_action(self, y, u, t, v): return np.zeros((2, 2))
    def F_yu_action(self, y, u, t, v): return np.zeros((2, 1))
    def F_uu_action(self, y, u, t, v): return np.zeros((1, 1))
```

### Define the objective

```python
class MinimumEnergyObjective:
    """J = 0.5 * q_T * |y(T) - y_target|^2 + 0.5 * r * sum(u^2)."""

    def __init__(self, y_target=np.zeros(2), terminal_weight=100.0,
                 control_weight=0.01):
        self.y_target = y_target
        self.terminal_weight = terminal_weight
        self.control_weight = control_weight

    def evaluate(self, trajectory, u):
        miss = trajectory.Y[-1][0] - self.y_target
        return float(0.5 * self.terminal_weight * np.dot(miss, miss)
                     + 0.5 * self.control_weight * np.sum(u ** 2))

    def dJ_dy_terminal(self, y_final):
        return self.terminal_weight * (y_final - self.y_target)

    def dJ_dy(self, y, step):                 # no running state cost
        return np.zeros(2)

    def dJ_du(self, u_stage, step, stage):
        return self.control_weight * u_stage

    def d2J_du2(self, u_stage, step, stage):
        return self.control_weight * np.eye(1)

    def d2J_dy2(self, y, step):
        return np.zeros((2, 2))

    def d2J_dy2_terminal(self, y_final):
        return self.terminal_weight * np.eye(2)
```

### Optimise

```python
from scipy.optimize import minimize
from adjungo import GLMOptimizer
from adjungo.methods.runge_kutta import rk4

method = rk4()
N = 80

optimizer = GLMOptimizer(
    problem=DampedOscillator(),
    objective=MinimumEnergyObjective(),
    method=method,
    t_span=(0.0, 4.0),
    N=N,
    y0=np.array([1.0, 0.0]),
)

fun, jac = optimizer.scipy_interface()
u0 = np.zeros(N * method.s * optimizer.problem.control_dim)

# Exact Hessian-vector products, so a Newton-Krylov method is usable directly.
result = minimize(fun, u0, jac=jac, hessp=optimizer.scipy_hessp(),
                  method="trust-ncg", options={"gtol": 1e-10})

u_optimal = result.x.reshape(N, method.s, optimizer.problem.control_dim)
y_final = optimizer.trajectory(u_optimal).Y[-1][0]
```

On the shipped parameters this converges in 5 trust-region iterations to
`|∇J|_∞ ≈ 2e-17`, bringing the mass from `x = 1 m` to `x(T) ≈ 5.7e-4 m`.
`L-BFGS-B` with `jac` alone also works and finds the same minimiser, but
stops several orders short in stationarity.

## Validation

Correctness is established against an **independently assembled** reference
(`adjungo/validation/reference.py`) that writes the entire discretisation as a
single residual `R(w, u) = 0` and forms

```
dJ/du = J_u - J_w (dR/dw)^-1 dR/du
```

by a dense solve. It shares no code with the stepping and adjoint
implementations, so it cannot share a mistake with them. Duality
(dot-product) tests and Hessian symmetry are used as corroboration only: this
repository has held a Hessian symmetric to `8.7e-19` while it was wrong by
`3.8e-3` in value.

```bash
.venv/bin/python -m pytest
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

| Document | What it is |
|---|---|
| [`NUMERICS.md`](NUMERICS.md) | The approved contract. What is certified, what is refused, the accuracy basis, and the defect precedents. Authoritative where anything else disagrees. |
| [`AGENTS.md`](AGENTS.md) | Paths, commands and local conventions for working in this repository. |
| [`docs/architecture.md`](docs/architecture.md) | Module map, the four sweeps, the `glm_opt.tex`-to-code symbol map, an explicit list of what is **not** built, and what a C++ reimplementation must preserve. |
| [`docs/adjoint_sensitivity_insight.md`](docs/adjoint_sensitivity_insight.md) | Why the adjoint and adjoint-sensitivity solves are linear and share one factorization, and the stage-index trap that makes a wrong version look correct. |
| [`docs/glm_opt.tex`](docs/glm_opt.tex) | The mathematical derivation the code implements. |
| [`docs/runge_kutta_opt.tex`](docs/runge_kutta_opt.tex) | The Runge-Kutta special case, second-order conditions. |
| [`docs/multistep_opt.tex`](docs/multistep_opt.tex) | Multistep optimality conditions. The corresponding solver path is **refused**; this is theory, not a description of working code. |
| [`docs/linalg_requirements.tex`](docs/linalg_requirements.tex) | Linear algebra requirements by method family. |
| [`docs/history/`](docs/history/README.md) | Superseded development reports, kept as evidence. Not maintained; see the index for where each durable claim now lives. |

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
