Looking at the documentation, this is a well-thought-out mathematical framework. Let me devise a Python project plan that captures the key abstractions while economizing computational work across the various solver types.

## Project Architecture

```
glm_opt/
├── core/
│   ├── problem.py          # Problem protocols (callbacks)
│   ├── objective.py        # Objective function protocols
│   ├── method.py           # GLMethod dataclass + tableau classification
│   └── requirements.py     # SolverRequirements deduction
├── algebra/
│   ├── protocols.py        # LinearAlgebraBackend protocol
│   ├── dense.py            # NumPy/SciPy implementation
│   └── operators.py        # Matrix-free operator wrappers
├── solvers/
│   ├── base.py             # BaseStageSolver ABC
│   ├── explicit.py         # ExplicitStageSolver
│   ├── dirk.py             # DIRKStageSolver
│   ├── sdirk.py            # SDIRKStageSolver (factorization reuse)
│   ├── implicit.py         # FullyImplicitStageSolver
│   ├── newton.py           # NewtonSolver mixin
│   └── factory.py          # Solver dispatch logic
├── stepping/
│   ├── forward.py          # Forward state propagation
│   ├── adjoint.py          # Backward adjoint propagation
│   ├── sensitivity.py      # State/adjoint sensitivity
│   └── trajectory.py       # Trajectory storage management
├── optimization/
│   ├── gradient.py         # Gradient assembly
│   ├── hessian.py          # Hessian-vector product
│   └── interface.py        # Optimizer-facing API
├── methods/
│   ├── runge_kutta.py      # RK tableau library
│   ├── multistep.py        # LMM tableau library
│   ├── glm.py              # General GLM tableaux
│   └── imex.py             # IMEX pairs
└── utils/
    └── kronecker.py        # Block Kronecker utilities
```

---

## Core Abstractions

### 1. Problem Protocol (Callbacks)

```python
# core/problem.py
from typing import Protocol, Optional
from numpy.typing import NDArray

class Problem(Protocol):
    """User provides callbacks; solver never forms full Jacobians unless needed."""
    
    @property
    def state_dim(self) -> int: ...
    
    @property  
    def control_dim(self) -> int: ...
    
    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """RHS evaluation: ẏ = f(y, u, t)"""
        ...
    
    # First derivatives (required for implicit methods and optimization)
    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """State Jacobian: ∂f/∂y, shape (n, n)"""
        ...
    
    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """Control Jacobian: ∂f/∂u, shape (n, ν)"""
        ...
    
    # Second derivatives (optional - for second-order optimization)
    def F_yy_action(self, y: NDArray, u: NDArray, t: float, 
                    v: NDArray) -> NDArray:
        """Contracted Hessian: Σ_ℓ v_ℓ ∂²f_ℓ/∂y∂y, shape (n, n)"""
        ...
    
    def F_yu_action(self, y: NDArray, u: NDArray, t: float,
                    v: NDArray) -> NDArray:
        """Contracted Hessian: Σ_ℓ v_ℓ ∂²f_ℓ/∂y∂u, shape (n, ν)"""
        ...
    
    def F_uu_action(self, y: NDArray, u: NDArray, t: float,
                    v: NDArray) -> NDArray:
        """Contracted Hessian: Σ_ℓ v_ℓ ∂²f_ℓ/∂u∂u, shape (ν, ν)"""
        ...


class ProblemStructure:
    """Deduced from problem or specified by user."""
    
    class Linearity(Enum):
        LINEAR = auto()       # F independent of y, u
        BILINEAR = auto()     # F = H + uV
        QUASILINEAR = auto()  # F depends on u only
        SEMILINEAR = auto()   # F constant + nonlinear part
        NONLINEAR = auto()    # General
    
    linearity: Linearity
    jacobian_constant: bool           # F independent of y
    jacobian_control_dependent: bool  # F depends on u
    has_second_derivatives: bool      # F_yy, F_yu, F_uu available
```

### 2. Method Specification

```python
# core/method.py
from dataclasses import dataclass, field
from functools import cached_property
from enum import Enum, auto
import numpy as np

class StageType(Enum):
    EXPLICIT = auto()   # A strictly lower triangular
    DIRK = auto()       # A lower triangular, varying diagonal
    SDIRK = auto()      # A lower triangular, constant diagonal γ
    IMPLICIT = auto()   # A dense

class PropType(Enum):
    IDENTITY = auto()   # V = I (RK)
    SHIFT = auto()      # V is shift matrix (Adams/BDF)
    TRIANGULAR = auto() # V lower triangular
    DENSE = auto()      # V general

@dataclass(frozen=True)
class GLMethod:
    """General Linear Method tableau."""
    A: np.ndarray  # (s, s) - internal stage coefficients
    U: np.ndarray  # (s, r) - history → stages
    B: np.ndarray  # (r, s) - stages → output
    V: np.ndarray  # (r, r) - history → output
    c: np.ndarray  # (s,)   - abscissae
    
    @cached_property
    def s(self) -> int:
        return self.A.shape[0]
    
    @cached_property
    def r(self) -> int:
        return self.V.shape[0]
    
    @cached_property
    def stage_type(self) -> StageType:
        return _classify_stage_structure(self.A)
    
    @cached_property
    def prop_type(self) -> PropType:
        return _classify_prop_structure(self.V)
    
    @cached_property
    def sdirk_gamma(self) -> Optional[float]:
        if self.stage_type == StageType.SDIRK:
            return self.A[0, 0]
        return None
    
    @cached_property
    def explicit_stage_indices(self) -> list[int]:
        """Stages i where a_{ii} = 0 (can be evaluated explicitly)."""
        return [i for i in range(self.s) if np.isclose(self.A[i, i], 0)]


def _classify_stage_structure(A: np.ndarray) -> StageType:
    s = A.shape[0]
    
    # Check strictly lower triangular
    if np.allclose(A, np.tril(A, -1)):
        return StageType.EXPLICIT
    
    # Check lower triangular
    if np.allclose(A, np.tril(A)):
        diag = np.diag(A)
        nonzero_diag = diag[~np.isclose(diag, 0)]
        
        if len(nonzero_diag) == 0:
            return StageType.EXPLICIT
        
        # All nonzero diagonal entries equal?
        if np.allclose(nonzero_diag, nonzero_diag[0]):
            return StageType.SDIRK
        return StageType.DIRK
    
    return StageType.IMPLICIT
```

### 3. Solver Requirements Deduction

```python
# core/requirements.py
@dataclass
class SolverRequirements:
    """What computational primitives are needed per time step."""
    
    # Stage solve requirements
    needs_newton: bool
    newton_system_size: int  # n for DIRK, ns for fully implicit
    
    # Factorization strategy
    factorizations_per_step: int
    factorization_size: int
    can_reuse_across_stages: bool   # SDIRK with constant Jacobian
    can_reuse_across_steps: bool    # Linear problem
    
    # What to store for optimization
    store_jacobians: bool
    store_stage_values: bool
    trajectory_vectors_per_step: int  # ns + nr for (Z, y)


def deduce_requirements(method: GLMethod, 
                        problem: ProblemStructure) -> SolverRequirements:
    """Dispatch logic from linalg_requirements.tex."""
    
    is_explicit = method.stage_type == StageType.EXPLICIT
    is_linear = problem.linearity in (Linearity.LINEAR, Linearity.BILINEAR, 
                                       Linearity.QUASILINEAR)
    
    if is_explicit:
        return SolverRequirements(
            needs_newton=False,
            newton_system_size=0,
            factorizations_per_step=0,
            factorization_size=0,
            can_reuse_across_stages=False,
            can_reuse_across_steps=False,
            store_jacobians=True,  # Still need F, G for adjoints
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r
        )
    
    needs_newton = not is_linear
    n = ...  # state dimension, passed in or inferred
    
    if method.stage_type == StageType.SDIRK:
        # Key optimization: single factorization if Jacobian constant
        can_reuse_stages = problem.jacobian_constant or not problem.jacobian_control_dependent
        return SolverRequirements(
            needs_newton=needs_newton,
            newton_system_size=n,
            factorizations_per_step=1 if can_reuse_stages else method.s,
            factorization_size=n,
            can_reuse_across_stages=can_reuse_stages,
            can_reuse_across_steps=problem.jacobian_constant,
            store_jacobians=True,
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r
        )
    
    # ... similar for DIRK, IMPLICIT
```

---

## Stage Solvers

The key insight: **factorization reuse**. The matrix `A_n` (for forward) and `A_n^T` (for adjoint) share the same LU factorization.

```python
# solvers/base.py
from abc import ABC, abstractmethod

class StageSolver(ABC):
    """Solves the stage equations for one time step."""
    
    @abstractmethod
    def solve_stages(self, 
                     y_history: NDArray,      # (r, n) external stages
                     u_stages: NDArray,       # (s, ν) controls at stages
                     t_n: float,
                     h: float,
                     problem: Problem,
                     method: GLMethod) -> tuple[NDArray, StepCache]:
        """
        Returns:
            Z: (s, n) internal stage values
            cache: Stored data for adjoint/sensitivity (Jacobians, factorizations)
        """
        ...
    
    @abstractmethod
    def solve_adjoint_stages(self,
                             lambda_ext: NDArray,  # (r, n) external adjoints
                             cache: StepCache,
                             method: GLMethod) -> NDArray:
        """
        Solve A^T μ = B^T λ for stage adjoints.
        Returns: μ (s, n)
        """
        ...


@dataclass
class StepCache:
    """Cached data from forward solve, reused in adjoint/sensitivity."""
    Z: NDArray                          # (s, n) stage values
    F: list[NDArray]                    # s Jacobians, each (n, n)
    G: list[NDArray]                    # s control Jacobians, each (n, ν)
    factorization: Optional[Any] = None # LU of stage matrix (reusable)
    stage_matrix: Optional[NDArray] = None
```

### Explicit Solver

```python
# solvers/explicit.py
class ExplicitStageSolver(StageSolver):
    """Forward substitution for strictly lower triangular A."""
    
    def solve_stages(self, y_history, u_stages, t_n, h, problem, method):
        s, n = method.s, problem.state_dim
        A, U, c = method.A, method.U, method.c
        
        Z = np.zeros((s, n))
        F_list, G_list = [], []
        
        for i in range(s):
            # Z_i = Σ_j U[i,j] y_j + h Σ_{j<i} A[i,j] f_j
            Z[i] = U[i] @ y_history
            for j in range(i):
                Z[i] += h * A[i, j] * self._f_cached[j]
            
            # Evaluate and cache
            t_stage = t_n + c[i] * h
            self._f_cached[i] = problem.f(Z[i], u_stages[i], t_stage)
            F_list.append(problem.F(Z[i], u_stages[i], t_stage))
            G_list.append(problem.G(Z[i], u_stages[i], t_stage))
        
        return Z, StepCache(Z=Z, F=F_list, G=G_list)
    
    def solve_adjoint_stages(self, lambda_ext, cache, method):
        # Backward substitution for A^T (upper triangular)
        # μ_i = h Σ_j a_{ji} F_i^T μ_j + h b_i F_i^T λ
        ...
```

### SDIRK Solver (Factorization Reuse)

```python
# solvers/sdirk.py
class SDIRKStageSolver(StageSolver):
    """
    Exploits constant diagonal γ: factor (I - hγF) once per step.
    For linear problems with constant F, factor once for all steps.
    """
    
    def __init__(self, reuse_across_steps: bool = False):
        self._global_factorization = None
        self._reuse_across_steps = reuse_across_steps
    
    def solve_stages(self, y_history, u_stages, t_n, h, problem, method):
        gamma = method.sdirk_gamma
        s, n = method.s, problem.state_dim
        
        Z = np.zeros((s, n))
        F_list, G_list = [], []
        
        # First implicit stage: need factorization
        # (I - h γ F) Z_i = rhs
        
        factorization = None
        
        for i in range(s):
            t_stage = t_n + method.c[i] * h
            
            # Build RHS: U[i] @ y_history + h Σ_{j<i} A[i,j] f_j
            rhs = method.U[i] @ y_history
            for j in range(i):
                rhs += h * method.A[i, j] * self._f_cached[j]
            
            if np.isclose(method.A[i, i], 0):
                # Explicit stage
                Z[i] = rhs
            else:
                # Implicit stage: solve (I - hγF)Z = rhs
                if factorization is None:
                    # First implicit stage: compute and factor
                    F_i = problem.F(Z[i-1] if i > 0 else y_history[0], 
                                   u_stages[i], t_stage)
                    stage_matrix = np.eye(n) - h * gamma * F_i
                    factorization = scipy.linalg.lu_factor(stage_matrix)
                
                Z[i] = scipy.linalg.lu_solve(factorization, rhs)
            
            self._f_cached[i] = problem.f(Z[i], u_stages[i], t_stage)
            F_list.append(problem.F(Z[i], u_stages[i], t_stage))
            G_list.append(problem.G(Z[i], u_stages[i], t_stage))
        
        return Z, StepCache(Z=Z, F=F_list, G=G_list, 
                           factorization=factorization)
    
    def solve_adjoint_stages(self, lambda_ext, cache, method):
        """
        Key insight: A^T has same LU factors (transposed solve).
        scipy.linalg.lu_solve with trans=1 solves A^T x = b.
        """
        # μ solves: (I - hγF)^T μ = B^T λ + coupling terms
        # Use cache.factorization with trans=1
        ...
```

### Newton Solver for Nonlinear Problems

```python
# solvers/newton.py
class NewtonMixin:
    """Mixin providing Newton iteration for nonlinear stage equations."""
    
    def newton_solve(self,
                     residual_fn: Callable[[NDArray], NDArray],
                     jacobian_fn: Callable[[NDArray], NDArray],
                     z0: NDArray,
                     tol: float = 1e-10,
                     max_iter: int = 10) -> tuple[NDArray, NDArray]:
        """
        Returns: (z_solution, final_jacobian_factorization)
        
        The final factorization is cached for adjoint/sensitivity solves.
        """
        z = z0.copy()
        for iteration in range(max_iter):
            r = residual_fn(z)
            if np.linalg.norm(r) < tol:
                break
            
            J = jacobian_fn(z)
            lu = scipy.linalg.lu_factor(J)
            dz = scipy.linalg.lu_solve(lu, -r)
            z += dz
        
        # Return final factorization for reuse
        return z, lu


class DIRKNewtonSolver(NewtonMixin, StageSolver):
    """DIRK stages with Newton iteration for nonlinear f."""
    
    def solve_stages(self, y_history, u_stages, t_n, h, problem, method):
        # Each stage: Newton solve for Z_i given Z_{1..i-1}
        # Cache the final Jacobian factorization from each stage
        ...
```

---

## Stepping Algorithms

### Forward Propagation

```python
# stepping/forward.py
@dataclass
class Trajectory:
    """Full trajectory storage for optimization."""
    Y: NDArray          # (N+1, r, n) external stages at each step
    Z: NDArray          # (N, s, n) internal stages at each step  
    caches: list[StepCache]  # Per-step cached data
    
    @property
    def N(self) -> int:
        return len(self.caches)


def forward_solve(y0: NDArray,
                  u: NDArray,           # (N, s, ν) or callable
                  t_span: tuple[float, float],
                  N: int,
                  problem: Problem,
                  method: GLMethod,
                  stage_solver: StageSolver) -> Trajectory:
    """
    Algorithm 1: Forward state solve from glm_opt.tex Section 7.
    
    For n = 1, ..., N:
        1. Solve A_n Z^n = U y^[n-1] for internal stages
        2. Update y^[n] = V y^[n-1] + B_n Z^n
        3. Store F_k^n, G_k^n for all stages
    """
    h = (t_span[1] - t_span[0]) / N
    s, r, n = method.s, method.r, problem.state_dim
    
    # Initialize external stages (may need starting procedure for r > 1)
    Y = np.zeros((N + 1, r, n))
    Y[0] = _initialize_external_stages(y0, method)
    
    Z = np.zeros((N, s, n))
    caches = []
    
    for step in range(N):
        t_n = t_span[0] + step * h
        u_stages = _get_stage_controls(u, step, method, h)
        
        # Solve stage equations
        Z[step], cache = stage_solver.solve_stages(
            Y[step], u_stages, t_n, h, problem, method
        )
        caches.append(cache)
        
        # Propagate external stages: y^[n] = V y^[n-1] + B Z^n
        f_stages = np.array([problem.f(Z[step, k], u_stages[k], 
                                       t_n + method.c[k] * h)
                            for k in range(s)])
        Y[step + 1] = method.V @ Y[step] + h * (method.B @ f_stages)
    
    return Trajectory(Y=Y, Z=Z, caches=caches)
```

### Backward Adjoint Propagation

```python
# stepping/adjoint.py
@dataclass  
class AdjointTrajectory:
    """Adjoint variables for all steps."""
    Lambda: NDArray     # (N+1, r, n) external stage adjoints
    Mu: NDArray         # (N, s, n) internal stage adjoints
    WeightedAdj: NDArray  # (N, s, n) Λ_k^n = Σ_j a_{jk} μ_j + Σ_j b_{jk} λ_j


def adjoint_solve(trajectory: Trajectory,
                  objective: Objective,
                  method: GLMethod,
                  stage_solver: StageSolver) -> AdjointTrajectory:
    """
    Algorithm 2: Backward adjoint solve from glm_opt.tex Section 7.
    
    For n = N, ..., 1:
        1. Solve A_n^T μ^n = B_n^T λ^n for stage adjoints
        2. Update λ^{n-1} = U^T μ^n + V^T λ^n + ∂J/∂y^[n-1]
        3. Compute Λ_k^n for all k
    """
    N = trajectory.N
    s, r, n = method.s, method.r, trajectory.Y.shape[-1]
    
    Lambda = np.zeros((N + 1, r, n))
    Mu = np.zeros((N, s, n))
    WeightedAdj = np.zeros((N, s, n))
    
    # Terminal condition
    Lambda[N] = objective.dJ_dy_terminal(trajectory.Y[N])
    
    for step in range(N - 1, -1, -1):
        cache = trajectory.caches[step]
        
        # Solve A^T μ = B^T λ (reuses factorization from forward!)
        Mu[step] = stage_solver.solve_adjoint_stages(
            Lambda[step + 1], cache, method
        )
        
        # Compute weighted adjoint: Λ_k = Σ_j a_{jk} μ_j + Σ_j b_{jk} λ_j
        for k in range(s):
            WeightedAdj[step, k] = (
                method.A[:, k] @ Mu[step] +  # Σ_j a_{jk} μ_j
                method.B[:, k] @ Lambda[step + 1]  # Σ_j b_{jk} λ_j
            )
        
        # Update: λ^{n-1} = U^T μ^n + V^T λ^n + ∂J/∂y^[n-1]
        Lambda[step] = (
            method.U.T @ Mu[step] +
            method.V.T @ Lambda[step + 1] +
            objective.dJ_dy(trajectory.Y[step], step)
        )
    
    return AdjointTrajectory(Lambda=Lambda, Mu=Mu, WeightedAdj=WeightedAdj)
```

### Sensitivity Equations

```python
# stepping/sensitivity.py
def forward_sensitivity(trajectory: Trajectory,
                        delta_u: NDArray,
                        method: GLMethod,
                        stage_solver: StageSolver) -> SensitivityTrajectory:
    """
    Algorithm 3: Forward state sensitivity from glm_opt.tex Section 7.
    
    Given δu, compute δy, δZ via:
        A_n δZ^n = U δy^[n-1] + Φ^n
        δy^[n] = V δy^[n-1] + B_n δZ^n + Ψ^n
    
    where Φ, Ψ contain the control-derivative forcing terms.
    
    Key: Uses SAME factorization as forward solve (cache.factorization)
    """
    ...


def adjoint_sensitivity(trajectory: Trajectory,
                        adjoint: AdjointTrajectory,
                        sensitivity: SensitivityTrajectory,
                        delta_u: NDArray,
                        method: GLMethod,
                        stage_solver: StageSolver,
                        problem: Problem) -> AdjointSensitivityTrajectory:
    """
    Algorithm 4: Backward adjoint sensitivity from glm_opt.tex Section 7.
    
    Solve:
        A_n^T δμ^n = B_n^T δλ^n + Γ^n
        δλ^{n-1} = U^T δμ^n + V^T δλ^n + J_{yy} δy^[n-1]
    
    where Γ^n contains second-derivative terms:
        Γ_k^n = h[F_{yy}^{n,k}[Λ_k^n] δZ_k^n + F_{yu}^{n,k}[Λ_k^n] δu_k^n]
    
    Key: Uses SAME factorization as adjoint solve (cache.factorization, trans=1)
    """
    ...
```

---

## Optimization Interface

```python
# optimization/interface.py
class GLMOptimizer:
    """
    Provides J(u), ∇J(u), H(u)·v to outer optimizer.
    """
    
    def __init__(self,
                 problem: Problem,
                 objective: Objective,
                 method: GLMethod,
                 t_span: tuple[float, float],
                 N: int,
                 y0: NDArray):
        self.problem = problem
        self.objective = objective
        self.method = method
        self.t_span = t_span
        self.N = N
        self.y0 = y0
        
        # Deduce requirements and create appropriate solver
        self.requirements = deduce_requirements(
            method, 
            deduce_problem_structure(problem)
        )
        self.stage_solver = create_stage_solver(method, self.requirements)
        
        # Cached trajectory (invalidated when u changes)
        self._trajectory: Optional[Trajectory] = None
        self._adjoint: Optional[AdjointTrajectory] = None
        self._u_hash: Optional[int] = None
    
    def objective_value(self, u: NDArray) -> float:
        """J(u) - runs forward solve if needed."""
        self._ensure_forward(u)
        return self.objective.evaluate(self._trajectory, u)
    
    def gradient(self, u: NDArray) -> NDArray:
        """∇J(u) - runs forward + adjoint if needed."""
        self._ensure_adjoint(u)
        return assemble_gradient(
            self._trajectory, 
            self._adjoint,
            u,
            self.objective,
            self.method,
            self.problem
        )
    
    def hessian_vector_product(self, u: NDArray, v: NDArray) -> NDArray:
        """[∇²J(u)]v - full second-order computation."""
        self._ensure_adjoint(u)
        
        # Forward sensitivity: δy, δZ from δu = v
        sensitivity = forward_sensitivity(
            self._trajectory, v, self.method, self.stage_solver
        )
        
        # Backward adjoint sensitivity: δλ, δμ
        adj_sensitivity = adjoint_sensitivity(
            self._trajectory, self._adjoint, sensitivity,
            v, self.method, self.stage_solver, self.problem
        )
        
        return assemble_hessian_vector_product(
            self._trajectory, self._adjoint,
            sensitivity, adj_sensitivity,
            u, v,
            self.objective, self.method, self.problem
        )
    
    # For scipy.optimize or similar
    def scipy_interface(self) -> tuple[Callable, Callable]:
        """Returns (fun, jac) for scipy.optimize.minimize."""
        def fun(u_flat):
            u = u_flat.reshape(self.N, self.method.s, self.problem.control_dim)
            return self.objective_value(u)
        
        def jac(u_flat):
            u = u_flat.reshape(self.N, self.method.s, self.problem.control_dim)
            return self.gradient(u).ravel()
        
        return fun, jac
```

---

## Gradient Assembly

```python
# optimization/gradient.py
def assemble_gradient(trajectory: Trajectory,
                      adjoint: AdjointTrajectory,
                      u: NDArray,
                      objective: Objective,
                      method: GLMethod,
                      problem: Problem) -> NDArray:
    """
    From glm_opt.tex equation for ∇_{u_k^n} Ĵ:
    
    ∇_{u_k^n} Ĵ = ∂J/∂u_k^n - h (G_k^n)^T Λ_k^n
    
    where Λ_k^n = Σ_i a_{ik} μ_i^n + Σ_i b_{ik} λ_i^n is the weighted adjoint.
    """
    N, s, nu = u.shape
    grad = np.zeros_like(u)
    h = ...  # step size
    
    for step in range(N):
        cache = trajectory.caches[step]
        
        for k in range(s):
            # ∂J/∂u contribution (if objective depends on u directly)
            grad[step, k] = objective.dJ_du(u[step, k], step, k)
            
            # Constraint contribution: -h G_k^T Λ_k
            G_k = cache.G[k]  # (n, ν)
            Lambda_k = adjoint.WeightedAdj[step, k]  # (n,)
            
            grad[step, k] -= h * G_k.T @ Lambda_k
    
    return grad
```

---

## Hessian-Vector Product Assembly

```python
# optimization/hessian.py
def assemble_hessian_vector_product(
        trajectory: Trajectory,
        adjoint: AdjointTrajectory,
        sensitivity: SensitivityTrajectory,
        adj_sensitivity: AdjointSensitivityTrajectory,
        u: NDArray,
        delta_u: NDArray,
        objective: Objective,
        method: GLMethod,
        problem: Problem) -> NDArray:
    """
    From glm_opt.tex:
    
    [∇²Ĵ]δu = J_{uu}δu + H_{uΛ}δΛ + H_{uZ}δZ + H_{uu}^{constr}δu
    
    At stage (n, k):
        (H_{uΛ}δΛ)_k^n = -h (G_k^n)^T δΛ_k^n
        (H_{uZ}δZ)_k^n = -h F_{yu}^{n,k}[Λ_k^n]^T δZ_k^n  
        (H_{uu}^{constr}δu)_k^n = -h F_{uu}^{n,k}[Λ_k^n] δu_k^n
    """
    N, s, nu = u.shape
    hvp = np.zeros_like(u)
    h = ...
    
    for step in range(N):
        cache = trajectory.caches[step]
        
        for k in range(s):
            # J_{uu} δu (from objective)
            hvp[step, k] = objective.d2J_du2(u[step, k], step, k) @ delta_u[step, k]
            
            # -h G_k^T δΛ_k
            delta_Lambda_k = adj_sensitivity.delta_WeightedAdj[step, k]
            hvp[step, k] -= h * cache.G[k].T @ delta_Lambda_k
            
            # -h F_{yu}[Λ_k]^T δZ_k (requires second derivatives)
            if problem.has_second_derivatives:
                Lambda_k = adjoint.WeightedAdj[step, k]
                delta_Z_k = sensitivity.delta_Z[step, k]
                
                F_yu_Lambda = problem.F_yu_action(
                    trajectory.Z[step, k], 
                    u[step, k],
                    ...,  # time
                    Lambda_k
                )
                hvp[step, k] -= h * F_yu_Lambda.T @ delta_Z_k
                
                # -h F_{uu}[Λ_k] δu_k
                F_uu_Lambda = problem.F_uu_action(
                    trajectory.Z[step, k],
                    u[step, k], 
                    ...,
                    Lambda_k
                )
                hvp[step, k] -= h * F_uu_Lambda @ delta_u[step, k]
    
    return hvp
```

---

## Dispatch Logic

```python
# solvers/factory.py
def create_stage_solver(method: GLMethod,
                        requirements: SolverRequirements,
                        problem_structure: ProblemStructure) -> StageSolver:
    """
    Decision tree from linalg_requirements.tex Section 6.
    """
    
    if method.stage_type == StageType.EXPLICIT:
        return ExplicitStageSolver()
    
    if requirements.needs_newton:
        # Nonlinear problem
        if method.stage_type == StageType.IMPLICIT:
            return FullyImplicitNewtonSolver()
        else:
            return DIRKNewtonSolver()
    
    # Linear/bilinear/quasilinear
    if method.stage_type == StageType.SDIRK:
        return SDIRKStageSolver(
            reuse_across_steps=requirements.can_reuse_across_steps
        )
    
    if method.stage_type == StageType.DIRK:
        return DIRKLinearSolver()
    
    return FullyImplicitLinearSolver()
```

---

## IMEX Extension

```python
# solvers/imex.py
@dataclass
class IMEXMethod:
    """Additive method: f = f_E + f_I with separate tableaux."""
    explicit: GLMethod   # A^E strictly lower triangular
    implicit: GLMethod   # A^I with structure as classified
    
    # Shared
    U: NDArray
    V: NDArray
    B_E: NDArray
    B_I: NDArray


class IMEXStageSolver(StageSolver):
    """
    Stage equations become:
        Z_i = history + h Σ_j a^E_{ij} f_E(Z_j) + h Σ_j a^I_{ij} f_I(Z_j)
    
    For i-th stage:
        (I - h a^I_{ii} F_I) Z_i = rhs + h Σ_{j<i} [a^E_{ij} f_E(Z_j) + a^I_{ij} f_I(Z_j)]
    
    The explicit parts just add to the RHS before solving the implicit part.
    """
    
    def __init__(self, implicit_solver: StageSolver):
        self.implicit_solver = implicit_solver
    
    def solve_stages(self, y_history, u_stages, t_n, h, 
                     problem_E: Problem, problem_I: Problem,
                     method: IMEXMethod):
        # Evaluate explicit parts, add to RHS, then solve implicit
        ...
```

---

## Key Optimizations Summary

| Solver Type | Factorizations/Step | Reuse Opportunity |
|-------------|--------------------|--------------------|
| Explicit | 0 | N/A |
| SDIRK + Linear | 1 | Same factor for all stages; if F constant, all steps |
| SDIRK + Nonlinear | s (Newton) | Modified Newton: lag Jacobian updates |
| DIRK + Linear | s | Each stage separate |
| DIRK + Nonlinear | s × Newton iters | Lag Jacobian within Newton |
| Implicit + Linear | 1 (size ns) | If F constant, all steps |
| Implicit + Nonlinear | Newton iters | Lag Jacobian |

**Factorization sharing between passes:**
- Forward state and forward sensitivity share `A_n` factorization
- Backward adjoint and backward adjoint sensitivity share `A_n^T` factorization (same LU, transposed solve)

This design gives you a clean separation of concerns while maximizing computational reuse. Shall I elaborate on any particular component?
