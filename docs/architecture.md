# Architecture

What the code is, how the pieces fit, and what a reimplementation must preserve.

This document describes **delivered** structure. Where something is planned but
not built, it says so. The authority for what is certified is
[`NUMERICS.md`](../NUMERICS.md) C-6.1; if this file and that table ever
disagree, that table wins.

---

## What the library computes

Given a discretization of

```
minimize  J(y, u)   subject to   y' = f(y, u, t),   y(t_0) = y_0
```

by a General Linear Method on a **fixed** mesh of `N` steps, `adjungo` returns
three quantities to an outer optimizer:

| Quantity | Method |
|---|---|
| `J(u)` | `GLMOptimizer.objective_value` |
| `grad J(u)` | `GLMOptimizer.gradient` |
| `H(u) v` | `GLMOptimizer.hessian_vector_product` |

The defining property, and the reason the project exists, is that the gradient
and the Hessian-vector product are the **exact derivatives of the discrete
objective that `objective_value` actually evaluates** -- not discretizations of
the continuous adjoint equations. On a fixed mesh they agree with a finite
difference of `objective_value` to roundoff, and they do so for every mesh,
including coarse ones where the discrete and continuous problems differ a lot.

This is the *discrete adjoint* (differentiate-then-discretize's opposite:
discretize-then-differentiate). It matters to an optimizer: a gradient that is
merely an `O(h^p)` approximation of the true gradient of the function being
minimized will stall the line search at a point that is not a stationary point
of anything.

`NUMERICS.md` C-2 and C-3 state this as a contract, and forbid explaining any
derivative discrepancy as time-discretization error.

---

## Module map

```
adjungo/
  core/            what the caller declares
    problem.py         Problem protocol: f, F=df/dy, G=df/du, and the three
                       second-derivative actions F_yy[w], F_yu[w], F_uu[w].
                       ProblemStructure: the caller's declaration of linearity,
                       Jacobian constancy and control dependence.
    objective.py       Objective protocol: terminal and running cost, their
                       first derivatives, and their second-derivative actions.
    method.py          GLMethod: tableaux A, U, B, V, abscissae c; StageType
                       and PropType classification of a tableau.
    requirements.py    deduce_requirements(method, structure, state_dim):
                       decides, from the declaration alone, what the solver
                       must be able to do.

  algebra/         how linear systems are represented and solved
    protocols.py       LinearAlgebraBackend protocol.
    dense.py           The one implementation: NumPy/SciPy dense LU.
    operators.py       Matrix-free operator wrappers.

  solvers/         one step's stage equations
    base.py            StageSolver protocol and the per-step cache.
    explicit.py        A strictly lower triangular: forward substitution.
    dirk.py            A lower triangular: one n-by-n factorization per stage.
    sdirk.py           A lower triangular with constant diagonal: one n-by-n
                       factorization per step, shared by all stages.
    implicit.py        A dense: one (s*n)-by-(s*n) coupled Newton solve.
    newton.py          Newton iteration with explicit convergence criteria.
    factorization.py   FactorizationStore: declaration-gated LU reuse,
                       verified by exact comparison, with the counters
                       C-15 certifies.
    factory.py         Dispatch from requirements to a stage solver.

  stepping/        the four sweeps over the whole trajectory
    forward.py         forward_solve            -> Trajectory
    adjoint.py         adjoint_solve            -> AdjointTrajectory
    sensitivity.py     forward_sensitivity      -> SensitivityTrajectory
                       adjoint_sensitivity      -> AdjointSensitivityTrajectory
    trajectory.py      Storage. Everything is kept; see "Checkpointing" below.

  optimization/    assembly and the caller-facing surface
    gradient.py        assemble_gradient
    hessian.py         assemble_hessian_vector_product
    interface.py       GLMOptimizer; CERTIFIED_STAGE_TYPES; the refusal gate.
    parametrization.py AffineControlParametrization and its two concrete
                       forms, NodalControl and PiecewiseConstantControl.

  methods/         tableau library
    runge_kutta.py     Explicit RK, DIRK, SDIRK, Gauss collocation.
    glm.py             General GLM tableaux.
    experimental/      RETAINED BUT REFUSED. The tableaux are correct; the
                       solver paths are not certified, so GLMOptimizer refuses
                       these methods at construction rather than returning a
                       derivative it cannot justify.
      multistep.py       r > 1 linear multistep tableaux.
      imex.py            Additive / IMEX pairs.

  validation/
    reference.py       An independently assembled discrete reference. Shares
                       no code with stepping/ or optimization/. This is the
                       primary oracle of NUMERICS.md C-14.1.

  utils/
    kronecker.py       Block Kronecker helpers.
```

---

## Control flow

`GLMOptimizer.__init__` is the gate. It calls `deduce_requirements` and refuses
at construction if the method's `StageType` is not in `CERTIFIED_STAGE_TYPES`.
Nothing downstream re-checks, and nothing downstream has to: **the gate and the
numerical route consume the same validated snapshot**, taken once.

```
gradient(u)                      hessian_vector_product(u, v)
  |                                |
  forward_solve ..................  forward_solve
  |                                |
  |                                forward_sensitivity(v)     <- tangent
  |                                |
  adjoint_solve ..................  adjoint_solve
  |                                |
  |                                adjoint_sensitivity(v)     <- second-order
  |                                |
  assemble_gradient                assemble_hessian_vector_product
```

`H v` costs two forward sweeps and two backward sweeps, independent of the
control dimension, and -- on a declared-constant Jacobian -- exactly as many
matrix factorizations as a single objective evaluation, which is one. Assembling the Hessian column by column instead would cost
`O(N s nu)` tangent solves; see `NUMERICS.md` C-13 item 9.

### Refusal, not silent approximation

A method the library cannot differentiate exactly raises at construction. It
does not fall back to a finite difference, and it does not return a number with
a warning. `NUMERICS.md` C-1. This is why the certified family list is short
and why adding to it requires oracle evidence rather than a passing forward
solve.

---

## The four sweeps

With `Z` the internal stages of a step, `y^[n]` the external stages, and `h` the
(fixed) step:

**Forward.** `R_i(Z) = Z_i - sum_k U[i,k] y^[n]_k - h sum_j A[i,j] f(Z_j,u_j,t_j)`,
then `y^[n] = V y^[n-1] + h B f(Z)`. The Newton Jacobian has blocks
`dR_i/dZ_j = delta_ij I - h A[i,j] F_j`.

**Adjoint.** `mu_p = h F_p^T ( sum_i A[i,p] mu_i + sum_l B[l,p] lambda_l )`,
`lambda^{n-1} = V^T lambda^n + U^T mu^n + dJ/dy`. Block `(p,i)` of this operator
is `delta_pi I - h A[i,p] F_p^T`, which is exactly the transpose of the forward
Jacobian block `(i,p)`.

**Therefore the adjoint reuses the forward factorization**, via a transposed
solve (`trans=1`), never a refactorization of an explicitly formed transpose.
The same factors serve the tangent and the adjoint sensitivity.

**Tangent.** `rhs_i = U[i] dy + h sum_j A[i,j] G_j du_j`, untransposed solve.

**Adjoint sensitivity.** Same operator as the adjoint, right-hand side enriched
by `Gamma_k = h ( F_yy[Lambda_k] dZ_k + F_yu[Lambda_k] du_k )`. See
[`adjoint_sensitivity_insight.md`](adjoint_sensitivity_insight.md).

---

## Symbol map: `glm_opt.tex` to code

A reimplementer reading the derivation alongside the code needs this. **`V` is
overloaded** -- the two uses are unrelated.

| In `glm_opt.tex` | In the code | Note |
|---|---|---|
| `A`, `U`, `B`, `V` | `method.A`, `method.U`, `method.B`, `method.V` | GLM tableaux |
| `V` (bilinear form, Sec. 4) | no code symbol | **Different `V`.** A pairing used only in the derivation. It is not `method.V`. |
| `Z^n` | `trajectory.Z[n]` | internal stages |
| `y^[n]` | `trajectory.Y[n]` | external stages |
| `mu^n` | `adjoint.Mu[n]` | internal adjoint |
| `lambda^n` | `adjoint.Lambda[n]` | external adjoint |
| `Lambda_k^n` | `adjoint.WeightedAdj[n,k]` | weighted adjoint, the argument of the second-derivative actions |
| `F`, `G` | `problem.F`, `problem.G` | `df/dy`, `df/du` |
| `F_yy[w]`, `F_yu[w]`, `F_uu[w]` | `problem.F_yy_action`, `F_yu_action`, `F_uu_action` | actions, never tensors |
| `delta Z`, `delta y` | `sensitivity.delta_Z`, `delta_Y` | tangent |
| `delta mu`, `delta lambda` | `adj_sensitivity.delta_Mu`, `delta_Lambda` | second-order adjoint |

The document's derivation is complex-valued and Hermitian throughout (`^ast`);
the implementation is real, so every `^H` reads as `^T`. The document's
Lagrangian sign convention was corrected in place -- it now adjoins constraints
with a minus sign, which is the convention the code implements. A `+`
convention flips `mu`, `lambda`, `Lambda` and the gradient's constraint term
together, and is self-consistent; the two must not be mixed. The relation
`A^H mu = B^H lambda` is homogeneous in the multipliers and therefore cannot
distinguish them, which is how the original inconsistency survived review.

---

## What is not built

Stated here so that this document cannot be read as advertising.

| Not built | Status |
|---|---|
| Automatic differentiation of user callbacks | Not planned. The caller supplies derivatives. |
| Additive / IMEX splitting | Tableaux retained in `methods/experimental/`; refused. |
| `r > 1` multistep | Tableaux retained; refused. Needs a certified starting procedure. |
| Partitioned methods (PRK, Nystrom) | Not built. |
| Sparse or matrix-free linear algebra | Not built. `algebra/protocols.py` is the seam it would enter through. |
| Reuse for a *varying* Jacobian (modified Newton, lagged Jacobian) | Not built, and not planned without an application benchmark. It would trade exactness for cost, which C-2 does not permit by default. |
| Checkpointing | Not built. Storage is `O(N s n)`, everything retained. |
| Generic scalar type | A C++ concern, not a Python one. See below. |

---

## For a C++ reimplementation

This Python package is a pilot study. `NUMERICS.md` C-13 is the normative list
of what must carry forward; it currently has nine items. The short version:

1. **The stage index on the Jacobian.** Forward block `(i,j)` carries `F_j`;
   the adjoint applies `F_i`, outside the sum. Getting this wrong still
   converges, and still shows the method's design order.
2. **The adjoint operator is the forward Jacobian's transpose.** Use a
   transposed triangular solve on the existing factors. Do not form the
   transpose and refactor: it is a different floating-point computation.
3. **All the `A`-coupling lives in the operator.** Adding an `A` term to the
   adjoint right-hand side as well double-counts. This is the natural slip when
   adapting the triangular branch to the dense one.
4. **The Newton Jacobian must be the true one, including off-diagonal blocks.**
   Dropping them yields a quasi-Newton iteration that converges to the *same*
   stage values -- the forward solve stays correct and the order test still
   passes -- but the adjoint reuses that factorization and the derivatives are
   then wrong. A certification resting on forward accuracy alone misses this
   entirely; it was found only by defect injection.
5. **Second derivatives are actions, never tensors.** `F_yy[w] v`, never `F_yy`.
6. **The Hessian-vector product is matrix-free and independent of `nu`.**
7. **Refuse rather than approximate.** The gate belongs at construction.
8. **The oracle is an independently assembled discrete reference.** Duality and
   Hessian symmetry are corroboration; they pass when the tangent and adjoint
   sweeps share a mistake.
9. **Hold the mesh fixed when testing derivatives.** A wrong stage index
   produces an error that shrinks with `h`.
10. **Gate factorization reuse on a declaration, and then verify the
    declaration.** Reuse is worth having -- it turns 48 factorizations into 1
    for `sdirk3` -- but it must never be decided by probing the problem (R-9),
    and a caller's declaration is a promise rather than a fact. Compare the
    stored matrix with the one being asked for, exactly, before returning
    stored factors. Then *count* the factorizations and assert the count:
    reuse is invisible to every accuracy test, working or broken, because
    refactoring the same matrix gives the same answer. See C-15.

Ergonomics that need not carry forward: the Python `Problem` protocol currently
requires a linear problem to supply three second-derivative callbacks that
return zero. A C++ interface should default them.
