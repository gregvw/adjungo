# Key Insight: Adjoint Sensitivity is a Linear Problem

> **Status.** The central claim of this note -- that the adjoint and the adjoint
> sensitivity are linear in their unknowns and can share one factorization -- is
> correct, is what the code does, and is recorded as C-5 and C-13 item 4 in
> `NUMERICS.md`. The stage recursion originally written below was **not**
> correct; it is corrected in place and the error is described, because it is a
> trap a reimplementation can fall into and pass most of its tests.

## User's Observation

> "In general, the adjoint sensitivity will depend on the state, state sensitivity, and adjoint. However, it is more like solving the adjoint equation again with an enhanced RHS since unlike the general state equation, the adjoint equation is already linear"

This is a **crucial simplification** for implementation!

## Why This Matters

### Forward Problem (Nonlinear)
```
Z^n = U y^{n-1} + h A f(Z^n, u^n)    [NONLINEAR in Z]
```
- Requires Newton iteration for implicit methods
- Jacobian changes at each Newton step
- Computationally expensive

### Adjoint Problem (Linear!)
```
A^T μ^n = B^T λ^n + forcing    [LINEAR in μ]
```
- **Already linear** even for nonlinear forward problems!
- Factorization computed once, reused
- Backward substitution for explicit methods

### Adjoint Sensitivity (Also Linear!)
```
A^T δμ^n = B^T δλ^n + Γ^n    [LINEAR in δμ]
```
- **Same linear structure** as adjoint problem
- **Same factorization** can be reused
- Only difference: enhanced RHS with second derivatives

## Implementation Strategy

### For Explicit Methods
Both adjoint and adjoint sensitivity use **backward substitution**:

```python
# Adjoint:
for i in range(s-1, -1, -1):
    weighted = B[:, i] @ lambda_ext            # Terminal
    for j in range(i+1, s):
        weighted += A[j, i] * mu[j]            # Coupling
    mu[i] = h * F[i].T @ weighted              # ONE F_i, applied last

# Adjoint sensitivity (same structure!):
for i in range(s-1, -1, -1):
    weighted = B[:, i] @ delta_lambda_ext
    for j in range(i+1, s):
        weighted += A[j, i] * delta_mu[j]
    delta_mu[i] = h * F[i].T @ weighted + Gamma[i]   # Gamma is the enhanced RHS
```

### The trap: which stage is the Jacobian evaluated at?

The recursion is

```
mu_i = h F_i^T ( sum_j A[j,i] mu_j  +  sum_l B[l,i] lambda_l )
```

`F_i` sits **outside** the sum. Every term in the bracket is multiplied by the
Jacobian of the stage being solved for, stage `i` -- not by the Jacobian of the
stage the term came from.

An earlier version of this note distributed the transpose into the loop and
wrote `A[j,i] * F[j].T @ mu[j]`, applying `F_j`. That is the transpose of the
wrong forward Jacobian. It is easy to write because the forward stage residual
`R_i = Z_i - U y - h sum_j A[i,j] f(Z_j, ...)` genuinely does carry `F_j` in
block `(i,j)`; transposing swaps the roles, and the index that survives is `i`.

This defect is nearly invisible in testing. It **cannot** be seen at all on:

- any problem with a state-independent Jacobian, where `F_i == F_j` identically,
- any one-stage method, where there is no off-diagonal coupling at all,
- the duality identity `<delta_y_N, w> == <grad, delta_u>`, if the tangent
  sweep is derived from the same wrong recursion -- both sides move together.

It was caught only by comparing against an independently assembled discrete
Jacobian on a problem with a state-dependent `F`. That is why `NUMERICS.md`
C-14.1 ranks the independent reference above the duality test, and why C-3 fixes
the mesh: a wrong stage index produces an error that shrinks with `h` and is
therefore easy to mistake for discretization error.

Implemented in `adjungo/solvers/explicit.py`; regression coverage in
`tests/test_adjoint_stage_coupling.py`.

### For Implicit Methods
Both reuse **same transpose factorization**:

```python
# Adjoint (SDIRK example):
factor = lu_factor(I - h*gamma*F)        # from the FORWARD solve; not refactored
for i in range(s-1, -1, -1):
    rhs = h * F[i].T @ (B[:, i] @ lambda_ext + coupling_terms)
    mu[i] = lu_solve(factor, rhs, trans=1)   # transposed solve, same factors

# Adjoint sensitivity (same factorization!):
for i in range(s-1, -1, -1):
    rhs = h * F[i].T @ (B[:, i] @ delta_lambda_ext + coupling_terms) + Gamma[i]
    delta_mu[i] = lu_solve(factor, rhs, trans=1)
```

The transpose is a solve option, not a separate matrix: `trans=1` reuses the
same `L` and `U` factors. Forming and factorizing the transpose explicitly is
not merely wasteful, it is a different computation in floating point, and a
port that does it will not reproduce the certified numbers.

For the fully implicit case the same statement holds at block level. The
forward Newton system has blocks `dR_i/dZ_j = delta_ij I - h A[i,j] F_j`, of
size `s*n`. The adjoint operator is **exactly** its transpose, so one `s*n`
factorization from the forward solve serves the adjoint, the tangent and the
adjoint sensitivity. See `NUMERICS.md` C-6.1.

## The Enhanced RHS: Γ

The only new computation is the second-derivative forcing:

```
Γ_k^n = h [F_{yy}^{n,k}[Λ_k^n] δZ_k^n + F_{yu}^{n,k}[Λ_k^n] δu_k^n]
```

Where:
- `F_{yy}[Λ]`: Hessian-vector product ∂²f/∂y² [Λ]
- `F_{yu}[Λ]`: Mixed derivative ∂²f/∂y∂u [Λ]
- `Λ_k`: Weighted adjoint at stage k
- `δZ_k`: State sensitivity at stage k
- `δu_k`: Control perturbation at stage k

## Cost Analysis

Counting only the dominant dense linear algebra, with `n` the state dimension
and `s` the number of internal stages:

**Adjoint solve:**
- Explicit: `O(s^2 n^2)` -- backward substitution, no factorization
- DIRK / SDIRK: `s` triangular solves of size `n`, `O(s n^2)`, reusing the
  forward factors; SDIRK has one such factorization per step, DIRK has `s`
- Fully implicit: one transposed solve of size `s n`, `O((s n)^2)`, reusing the
  single `O((s n)^3)` forward factorization

**Adjoint sensitivity:**
- Same structure and same cost as the adjoint solve, with no new factorization
- Plus `O(s n^2)` for the second-derivative forcing `Gamma`, which is
  matrix-free: it needs the *actions* `F_yy[Lambda] dZ` and `F_yu[Lambda] du`,
  never the tensors themselves

**Total cost for adjoint sensitivity is about 1x an adjoint solve.**

These counts are enforced, not assumed. `NUMERICS.md` C-15 makes the observed
number of `lu_factor` calls a certified quantity, and
`tests/test_factorization_reuse.py` asserts it against the number
`SolverRequirements.factorizations_for_solve` predicts -- structurally, never by
timing.

When the caller declares a constant Jacobian, the measured count for a whole
solve is **1**, independent of the number of steps, for every certified
implicit family; a Hessian-vector product takes no more than an objective
evaluation. Before this was measured, `sdirk3` took 48 factorizations over 8
steps while the deduction claimed 8.

## Key Advantages

1. **No Newton iteration needed** - already linear!
2. **Reuse factorizations** - computed during adjoint solve
3. **Same code structure** - just enhanced RHS
4. **Efficient** - no additional factorization cost

## Comparison to Forward Sensitivity

| Aspect | Forward Sensitivity | Adjoint Sensitivity |
|--------|-------------------|-------------------|
| **Linearity** | Linearization of nonlinear problem | Already linear! |
| **Implicit solve** | May need Newton iteration | Just linear solve |
| **Cost** | ≈ 1× forward solve | ≈ 1× adjoint solve |
| **Direction** | Forward in time | Backward in time |
| **Factorization** | Reuse from forward | Reuse from adjoint |

## Implementation status

All of the following are implemented and certified for the explicit, DIRK,
SDIRK and fully implicit families. See `NUMERICS.md` C-6.1 for the current
table and for what "certified" requires.

- [x] Adjoint sensitivity is a linear problem (this document)
- [x] Second-derivative forcing `Gamma`: `F_yy[Lambda] dZ` and `F_yu[Lambda] du`
- [x] Backward linear solve
  - [x] Explicit: backward substitution
  - [x] SDIRK: reuse the transposed factorization
  - [x] DIRK: reuse the transposed factorization per stage
  - [x] Fully implicit: one transposed `s*n` solve
- [x] External-stage propagation `dlambda^{n-1} = U^T dmu + V^T dlambda + J_yy dy`

Refused, not implemented: `r > 1` multistep and IMEX/additive. Their tableaux
are present in `adjungo/methods/experimental/` and the optimizer refuses them at
construction rather than returning a number it cannot justify.

## Conclusion

**Adjoint sensitivity is another backward linear solve with an enhanced
right-hand side.** It has the same solver structure as the adjoint, reuses the
adjoint's factorization, and costs about as much.

What remained genuinely hard was not the linear algebra but getting the
recursion's indices and signs right, because the wrong ones still produce a
plausible, mesh-convergent number. The second-derivative terms `Gamma` are
evaluated through `problem.F_yy_action()` and `problem.F_yu_action()`, the same
callbacks the Hessian-vector product needs, and they are never assembled as
tensors.
