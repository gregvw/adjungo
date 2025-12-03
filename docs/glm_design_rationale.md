// glm_design_rationale.md
// Design Rationale: GLM Solver with Customizable Functions

# Why Customizable Functions for Time Integration?

## The Problem with Virtual Inheritance (Tempus et al.)

Tempus and similar ODE solver frameworks use virtual inheritance for extensibility:

```cpp
// Typical virtual inheritance approach
class StepperBase {
public:
    virtual void takeStep(double dt) = 0;
    virtual ~StepperBase() = default;
};

class StepperExplicitRK : public StepperBase {
    virtual void evalRHS(Vector& f, Vector const& y, double t) = 0;
    void takeStep(double dt) override {
        // calls evalRHS through vtable
    }
};

class MyProblem : public StepperExplicitRK {
    void evalRHS(Vector& f, Vector const& y, double t) override {
        // user implementation
    }
};
```

**Problems:**

1. **Vtable overhead on hot paths**: Every RHS evaluation goes through a virtual 
   dispatch. For explicit methods with small state vectors, this can dominate.

2. **No inlining**: The compiler cannot inline `evalRHS` because it doesn't know
   the concrete type at compile time. This prevents vectorization, loop fusion,
   and other critical optimizations.

3. **Rigid hierarchy**: Adding new extension points requires modifying the base 
   class hierarchy. Want to add Jacobian-vector products? New virtual function.
   Want matrix-free adjoints? Another virtual function. The hierarchy grows.

4. **Object slicing risks**: Passing `StepperBase&` loses type information.
   Dynamic casting is required to recover it.

5. **Composition over inheritance issues**: Virtual inheritance makes composition
   awkward. You can't easily wrap or decorate steppers.

6. **Testing complexity**: Mocking requires creating concrete derived classes.

## The Customizable Functions Solution

With the `custom` keyword, we define extension points as free functions:

```cpp
template<typename State, typename Control, typename Time, typename RHS>
custom auto evaluate_rhs(
    RHS const& rhs,
    State const& y,
    Control const& u,
    Time t
) -> State;
```

**Key insight**: This looks like a virtual function from the user's perspective
(they provide their implementation), but the compiler sees a concrete function
call when the types are known.

### Static Dispatch When Possible

```cpp
// User provides a concrete type
struct MyQuantumSystem {
    auto f(VectorXcd const& y, double u, double t) const -> VectorXcd {
        return (H + u * V) * y;
    }
};

// Solver instantiated with concrete type
GLMSolver<VectorXcd, double, double, 4, 1, MyQuantumSystem, EigenBackend> solver;

// Compiler knows the exact type: evaluate_rhs<VectorXcd, double, double, MyQuantumSystem>
// This can be inlined, vectorized, etc.
```

### Dynamic Dispatch When Needed

```cpp
// Type-erased wrapper for runtime polymorphism
struct AnyRHS {
    std::any impl;
    std::function<VectorXd(VectorXd const&, double, double)> f_ptr;
    // ...
};

// Customize for type-erased case
template<>
custom auto evaluate_rhs<VectorXd, double, double, AnyRHS>(
    AnyRHS const& rhs, VectorXd const& y, double u, double t
) -> VectorXd {
    return rhs.f_ptr(y, u, t);  // goes through std::function
}

// User chooses: concrete type for performance, AnyRHS for flexibility
```

### The Best of Both Worlds

| Aspect | Virtual Inheritance | Customizable Functions |
|--------|---------------------|------------------------|
| Static dispatch | ✗ | ✓ (when type known) |
| Dynamic dispatch | ✓ | ✓ (when needed) |
| Inlining | ✗ | ✓ |
| Vectorization | ✗ (usually) | ✓ |
| Extension points | Modify base class | Add new `custom` function |
| Composition | Awkward | Natural |
| Type safety | Runtime errors | Compile-time errors |

## Specific Benefits for GLM Optimal Control

### 1. Hot Path Optimization

The RHS evaluation `f(y, u, t)` is called:
- `s` times per time step (internal stages)
- `N` time steps
- Many optimizer iterations

For a 100-dimensional problem with 1000 time steps, a 4-stage method, and 100 
optimizer iterations, that's `s × N × iters = 4 × 1000 × 100 = 400,000` RHS 
evaluations. Even 10ns of vtable overhead per call adds up.

With customizable functions, the compiler can:
- Inline the RHS into the stage loop
- Vectorize over state dimensions
- Fuse operations with the stage linear algebra

### 2. Jacobian Computation Flexibility

Some users want to:
- Provide explicit Jacobians (quantum control: `F = H + uV`)
- Use automatic differentiation (general nonlinear systems)
- Use finite differences (legacy codes)
- Use matrix-free Jacobian-vector products (large-scale PDEs)

With customizable functions, each is a different customization:

```cpp
// Explicit Jacobian
template<>
custom auto evaluate_jacobian_y<...>(ExplicitJacobianRHS const& rhs, ...) {
    return rhs.jacobian_y(y, u, t);
}

// AD-generated (e.g., with Enzyme or dual numbers)
template<>
custom auto evaluate_jacobian_y<...>(AutoDiffRHS<F> const& rhs, ...) {
    return autodiff::jacobian(rhs.f, wrt(y), at(y, u, t));
}

// Matrix-free (only provides Jv products)
template<>
custom auto apply_jacobian_y<...>(MatrixFreeRHS const& rhs, ..., Vector const& v) {
    return rhs.Jv(y, u, t, v);  // no explicit Jacobian ever formed
}
```

The solver queries capabilities at compile time:

```cpp
if constexpr (has_explicit_jacobian<RHS>) {
    auto J = evaluate_jacobian_y(rhs, y, u, t);
    // ... use explicit matrix
} else if constexpr (has_jacobian_vector_product<RHS>) {
    // ... use Krylov methods
} else {
    // ... fall back to finite differences
}
```

### 3. Linear Algebra Backend Swapping

Different problems need different solvers:
- Small dense: Eigen LU
- Large sparse: Trilinos (Ifpack2 + Belos)
- GPU: Kokkos + cuSOLVER
- Distributed: Tpetra + MueLu

Each is a customization of the linear algebra primitives:

```cpp
// Eigen backend
template<>
custom auto factor<MatrixXd, PartialPivLU<MatrixXd>, EigenBackend>(...);
template<>
custom auto solve<VectorXd, PartialPivLU<MatrixXd>, EigenBackend>(...);

// Trilinos backend
template<>
custom auto factor<Tpetra::CrsMatrix<>, Ifpack2::Preconditioner<>, TrilinosBackend>(...);
template<>
custom auto solve<Tpetra::MultiVector<>, Belos::SolverManager<>, TrilinosBackend>(...);
```

The solver code is written once, generic over these customization points.

### 4. Second-Order Optimization

For Newton-CG or trust-region methods, we need Hessian-vector products:
- `F_yy[v]`: Hessian of `f` w.r.t. `y`, contracted with vector `v`
- `F_yu[v]`: Mixed Hessian
- `F_uu[v]`: Control Hessian

For bilinear problems, these simplify dramatically:
- `F_yy[v] = 0` (linear in y)
- `F_yu[v] = V` (constant)
- `F_uu[v] = 0` (linear in u)

With customizable functions, the bilinear structure can be exploited:

```cpp
// General nonlinear: use AD for Hessians
template<>
custom auto hessian_yy_action<...>(GeneralRHS const& rhs, ..., Vector const& v) {
    return autodiff::hessian_action(rhs.f, wrt(y), at(y,u,t), v);
}

// Bilinear: compile-time zero
template<>
custom auto hessian_yy_action<...>(BilinearRHS const& rhs, ..., Vector const& v) {
    return Matrix::Zero(n, n);  // compiler can eliminate entirely
}
```

The solver's Hessian-vector product routine benefits from inlining these,
potentially eliminating entire matrix operations for bilinear problems.

## Implementation Strategy

### Phase 1: Python Prototype

Use JAX for:
- AD-based Jacobians and Hessians
- JIT compilation for performance
- GPU support for proof of concept

This validates the mathematical framework.

### Phase 2: C++ Core with Customizable Functions

1. Define the customization points (RHS, Jacobians, Hessians, linear algebra)
2. Implement the generic GLM stepper
3. Implement the optimization routines (gradient, Hessian-vector product)
4. Provide default customizations for common cases

### Phase 3: Backend Integrations

- Eigen backend (dense, small-scale)
- Trilinos backend (sparse, large-scale, distributed)
- Kokkos backend (GPU)

### Phase 4: AD Integration

- Enzyme for source-to-source AD
- Dual number fallback
- User-provided derivatives as always

## Comparison with Tempus

| Feature | Tempus | GLM-Custom |
|---------|--------|------------|
| RK methods | ✓ | ✓ |
| Multistep | Partial | ✓ (GLM framework) |
| GLM | ✗ | ✓ |
| PRK/Symplectic | Limited | ✓ (partitioned GLM) |
| IMEX | ✓ | ✓ (additive GLM) |
| Adjoint | External (ROL) | Built-in |
| Second-order | External (ROL) | Built-in |
| Dispatch | Virtual | Static when possible |
| Inlining | ✗ | ✓ |
| Matrix-free | Awkward | Natural |

## Conclusion

Customizable functions give us:
1. **Zero-overhead abstraction**: Static dispatch when types are known
2. **Flexibility when needed**: Dynamic dispatch available via type erasure
3. **Clean extension points**: No base class modification required
4. **Composition**: Backends, RHS, objectives all orthogonal
5. **Optimization**: Compiler can see through abstractions

This is exactly what a high-performance, flexible time integration library needs.
