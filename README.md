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
| Partitioned (`symplectic_euler`, `verlet`) | **certified** — separable `H = T(p) + V(q, u, t)` only; the domain is checked on the values the step used |
| BDF (`bdf2`, `bdf3`) | **refused** — `r > 1` has no starting procedure |
| Adams (`adams_bashforth2`, `adams_moulton2`) | **refused** — tableau not representable |
| IMEX / additive splitting | **refused** |

Tableaux for the refused families are kept under
`adjungo/methods/experimental/` as reference material, not as endorsement.
`GLMOptimizer` raises `NotImplementedError` for the ones that are
representable; the Adams tableaux need per-history stage coefficients and are
rejected earlier, by `GLMethod` validation itself.

### Discretisation plans

A solve does not have to use one mesh and one method. A `DiscretizationPlan`
carries the time nodes and a method for **each** step, so step sizes may vary
and the method may change from step to step — an explicit method where the
problem is mild, an implicit one where it stiffens.

```python
from adjungo.core.plan import DiscretizationPlan

# The uniform case, unchanged, and still the convenience form.
plan = DiscretizationPlan.uniform((0.0, 1.0), 100, rk4())

# Or nodes and methods given explicitly.
plan = DiscretizationPlan(
    nodes=np.array([0.0, 0.1, 0.3, 0.7, 1.0]),
    methods=(rk4(), rk4(), sdirk2(), sdirk2()),
)
optimizer = GLMOptimizer(problem, objective, y0=y0, plan=plan)
```

The plan is **prescribed**, not adapted: it is an input, frozen for the
duration of a solve, and the backward sweep reads the same plan the forward
sweep executed. Adjungo returns exact derivatives of the discrete objective
*that plan* defines. Choosing a plan from an evolving trajectory is a policy
the caller writes; Adjungo does not yet supply one.

`optimizer.plan` is read-only, and the plan copies and freezes each tableau it
is given. Re-discretising is `optimizer.with_plan(new_plan)`, which returns a
new optimizer: the trajectory, adjoint, stage solvers and factorization stores
are all keyed to a plan, and none of them would survive rebinding the
attribute.

Stage-indexed arrays are stored packed, as `(Σₙ sₙ, ·)`. When every step has
the same stage count this is the familiar `(N, s, ν)` array reshaped, and
controls may be supplied in either layout. The control parametrisations below
work in either case — build one with `NodalControl.from_plan(plan, ν)` so that
it interpolates to each step's *own* abscissae, which is the thing a plan that
changes method makes possible to get wrong at exactly the right shape.

One thing is refused rather than guessed when the stage count **varies**: the
scalar objective value, because `Objective.evaluate` is specified on the
rectangular layout, and handing it the packed array would have it read stage
`(n, k)` out of whatever step lies at packed row `n`. An objective that
implements `evaluate_packed` is used instead and has no such limit. Gradients
and Hessian-vector products never needed either — their objective callbacks
receive one stage and its `(step, stage)` index.

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

### Factorisation reuse

An LU factorisation is reused across stages and steps only when the caller
declares the structure that makes reuse exact:

```python
from adjungo import GLMOptimizer, ProblemStructure
from adjungo.core.problem import Linearity

structure = ProblemStructure(
    linearity=Linearity.SEMILINEAR,
    jacobian_constant=True,      # the declaration that permits reuse
    jacobian_control_dependent=False,
    has_second_derivatives=True,
)
optimizer = GLMOptimizer(problem, objective, method, t_span, N, y0,
                         problem_structure=structure)
```

Two things are then true. Reuse is *declared*, never inferred: an earlier
version probed `F` to discover constancy, varied only `u` and `t`, and let a
state-dependent Jacobian through, which handed the adjoint another stage's
matrix (precedent R-9 in `NUMERICS.md`). And the declaration is *checked*:
before any stored factorisation is returned, the matrix it was taken from is
compared with the matrix now being asked for, element for element. A
declaration contradicted by the problem raises rather than reusing.

With a constant Jacobian declared, an entire solve takes one factorisation
instead of one per stage per Newton iteration per step. The counts are
asserted, not timed — refactoring the same matrix gives the same answer, so
no accuracy test can detect a failure to reuse. See
`tests/test_factorization_reuse.py` and `NUMERICS.md` C-15.

A Jacobian that depends on time alone, such as that of
`TimeVaryingAffineDynamics`, gets a narrower reuse. Its stage matrices differ
from one stage time to the next, so nothing is shared within a solve. But on a
fixed mesh the matrix at each stage time is the same on every call, whatever
the control. The first gradient therefore takes one factorisation per implicit
stage solve, and every later gradient or Hessian-vector product takes none.
The same exact comparison guards every reuse. So a coefficient that changes
between calls is refused when a later solve assembles a matrix at a stage time
already stored. An evaluation repeated at an unchanged control is served from
the optimizer's cache and checks nothing. See `NUMERICS.md` C-17.6.

### Not yet implemented

- Factorisation reuse for a **varying** Jacobian (modified Newton, lagged
  Jacobian). Reuse under a declared-constant Jacobian is implemented and is
  described above; what is missing is reuse where the matrix genuinely
  changes and a stale one would be used deliberately.
- **Higher-order partitioned methods.** Symplectic Euler and Störmer–Verlet
  are certified and described above. Compositions such as Blanes–Moan are not
  built: their paired symplectic condition cancels arithmetically rather than
  vanishing structurally, so it needs a stated rounding budget under
  `NUMERICS.md` C-11.3 before it can be asserted.
- **Non-separable Hamiltonians** under a partitioned method. `NUMERICS.md`
  C-8.4 scopes the family to `H = T(p) + V(q, u, t)`; anything else is refused
  during the solve rather than integrated, because the partitioned sweeps drop
  the coupling blocks by construction.
- Nonlinear control parametrisation, sparse operators, and checkpointing.
- **Mesh adaptation.** Prescribed non-uniform steps and per-step methods are
  implemented and described above; what is missing is a policy that builds a
  new plan from a trajectory's error or stiffness indicators, and the
  warm-start transfer between plans that would go with it.
- **Multistep startup.** `r > 1` is refused at construction, as the table
  above records: it needs a defined history representation and a certified
  starting procedure. `NUMERICS.md` C-Q4 holds the open question.

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
- `adjungo.validation`: The independently assembled reference the Validation
  section above describes. It shares no code with `stepping` or `optimization`,
  which is the property that makes it an oracle rather than a second opinion.
- `adjungo.utils`: Utility functions

## Documentation

| Document | What it is |
|---|---|
| [`NUMERICS.md`](NUMERICS.md) | The approved contract. What is certified, what is refused, the accuracy basis, and the defect precedents. Authoritative where anything else disagrees. |
| [`AGENTS.md`](AGENTS.md) | Paths, commands and local conventions for working in this repository. |
| [`docs/architecture.md`](docs/architecture.md) | Module map, the four sweeps, the `glm_opt.tex`-to-code symbol map, an explicit list of what is **not** built, and what a C++ reimplementation must preserve. |
| [`docs/adjoint_sensitivity_insight.md`](docs/adjoint_sensitivity_insight.md) | Why the adjoint and adjoint-sensitivity solves are linear and share one factorization, and the stage-index trap that makes a wrong version look correct. |
| [`docs/evidence/coefficient_immutability.md`](docs/evidence/coefficient_immutability.md) | The failure history and injection table behind C-15.7's coefficient-custody rules. Cited from the clause; kept out of it because the narrative is long and the clause is normative. |
| [`docs/glm_opt.tex`](docs/glm_opt.tex) | The mathematical derivation the code implements. |
| [`docs/runge_kutta_opt.tex`](docs/runge_kutta_opt.tex) | The Runge-Kutta special case, second-order conditions. Ten of its equations carried the same stage-index defect as the code (precedent R-5) and were corrected; its correction notice records what changed and why. |
| [`docs/multistep_opt.tex`](docs/multistep_opt.tex) | Multistep optimality conditions. The corresponding solver path is **refused**; this is theory, not a description of working code. |
| [`docs/linalg_requirements.tex`](docs/linalg_requirements.tex) | Linear algebra requirements by method family. A **design** document for the eventual C++ library, not a description of this code. Its status preamble lists four rules it stated that were later refuted, and the clauses that correct them. |
| [`docs/history/`](docs/history/README.md) | Superseded development reports, kept as evidence. Not maintained; see the index for where each durable claim now lives. |

## Examples

All seven are executed by the test suite, and each is checked against the
independently assembled reference rather than merely run.

| Example | What it demonstrates |
|---|---|
| [`minimum_energy_oscillator.py`](examples/minimum_energy_oscillator.py) | The quick start above. Explicit Runge–Kutta (`rk4`) on dynamics affine in `(y, u)`, with both optimizer routes: gradient-only and the exact Hessian-vector product. |
| [`nonlinear_implicit_control.py`](examples/nonlinear_implicit_control.py) | The fully implicit route. A controlled Van der Pol oscillator under `gauss2`: coupled stages solved as one `(s·n)` Newton system per step, and a Jacobian that genuinely depends on the state, so the second derivatives of `f` are not zero. |
| [`factorization_reuse_counts.py`](examples/factorization_reuse_counts.py) | Declaration-gated factorisation reuse, counted rather than timed. One factorisation for the whole solve when `jacobian_constant=True` is declared, against one per stage per Newton iteration per step when it is not — with an identical gradient either way. |
| [`rocket_ascent.py`](examples/rocket_ascent.py) | Derivative callbacks differentiated by sympy rather than written out, via [`examples/symbolic.py`](examples/symbolic.py), and checked against hand-derived ones. A Tsiolkovsky rocket ascent with a closed-form solution under constant burn, an objective that carries the C-9.3 stage quadrature itself, and second derivatives of `f` that do not vanish. Needs the `examples` extra: `pip install -e '.[dev,examples]'`. |
| [`double_integrator.py`](examples/double_integrator.py) | The only example that knows its own answer. A linear-quadratic tracking problem checked against two references: a backward Riccati recursion on an independently assembled step map, and the closed-form solution of the continuous optimality conditions. For the undamped problem these coincide *exactly* — `u*` is linear in `t` and rk4's weights are Simpson's rule — so adding linear drag makes the costate exponential and recovers a fourth-order mesh study of the **optimum**, not of a solve. |
| [`pendulum_swing_up.py`](examples/pendulum_swing_up.py) | The only example whose *nonlinear* dynamics have a closed-form solution. A torque-driven pendulum, swung from hanging to inverted under `gauss2`. Undriven and undamped it librates exactly as a Jacobi elliptic function, with a period a third longer than the small-angle `2π/ω₀`, so the fourth-order mesh study could not pass on a linearised `sin θ`. It also conserves `E = ½ω² + ω₀²(1 − cos θ)`, a first integral that constrains every point rather than one endpoint — and separates the methods: over 32× the integration time `rk4` lets the energy error grow 27-fold where symplectic `gauss2` holds it to 1.003. The swing-up optimum itself has no closed form, and none is claimed. Needs the `examples` extra. |
| [`zermelo_navigation.py`](examples/zermelo_navigation.py) | The only example whose field is **nonlinear in the control**, and the only one that drives `F_uu`. A boat of fixed speed steers through a linear shear current to reach as far downstream as it can in a fixed time. Because `f` is linear in the state and the cost is terminal and linear, `F_yy`, `F_yu` and every `d2J` block vanish identically: removing `F_uu_action` does not perturb the Hessian, it zeroes it. Pontryagin gives `tan θ*(t) = (V/h)(T − t)`, Zermelo's navigation formula `θ̇ = −(V/h)cos²θ`, and the trajectory in elementary functions. The costate never consults the state, so that closed-form heading is stationary for the **discrete** problem too, on any mesh, to rounding — but only for tableaux satisfying Butcher's `D(1)`, `Σⱼbⱼaⱼᵢ = bᵢ(1−cᵢ)`. Four shipped tableaux satisfy it and four do not, and the example predicts which from the tableau alone. That binds the discrete adjoint's stage weights, which a duality test cannot separate from the tangent's. Needs the `examples` extra. |
| [`atom_transport.py`](examples/atom_transport.py) | The only example that is **not a GLM**. Shuttling a trapped atom from rest to rest under Störmer–Verlet, which applies `A^q` to the position block and `A^p` to the momentum block — not expressible as a single `A ⊗ I` (`NUMERICS.md` C-8.4). The Hamiltonian is separable with a tight-binding kinetic energy `T(p) = J(1 − cos p)`, so `∂f^q/∂p` depends on the state; with a quadratic `T` that block would be the identity and a wrong index in the `A^q` recursion would cancel. What is priced is the residual **motional excitation**, a small difference of larger quantities, so any energy the integrator invents is added straight to the objective. Undriven at 20 steps per period, `max|E − E(0)|` grows 1.00× between horizons of 1 and 32 periods under Verlet, where fourth-order `rk4`'s grows 29.9× — a growth factor at constant `h`, not a guarantee and not a claim about the error's size; C-18.6 records that symplecticity implies neither energy conservation nor that a variable-step plan inherits this. It also shows the domain check: adding ordinary linear drag leaves C-8.4's separable domain and the solve refuses rather than silently discarding the term. Needs the `examples` extra. |

## Development

All commands invoke the virtual environment's interpreter explicitly; there is
no assumption that a shell has been activated.

```bash
.venv/bin/python -m pytest                            # full suite
.venv/bin/python -m ruff check adjungo tests examples # lint
.venv/bin/python -m mypy adjungo                      # type check
```

The full suite runs in a few seconds, so there is no reason to select a subset.
[`AGENTS.md`](AGENTS.md) carries the working rules this repository enforces:
the defect-injection procedure, where exploratory scripts live, and the
tolerance and documentation conventions.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome. [`AGENTS.md`](AGENTS.md) is the contributor guide;
[`NUMERICS.md`](NUMERICS.md) is the normative contract, and any accuracy,
envelope or degeneracy claim must cite a clause that exists there.
