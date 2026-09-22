# NUMERICS.md — Adjungo domain contract

Status of this document: **normative**. Every accuracy, envelope, or degeneracy
claim made by Adjungo code, tests, or documentation must cite a clause here.
Clauses are stable identifiers; do not renumber them.

Fact-status labels used below: `APPROVED` (ruled by the scientific owner),
`OBSERVED` (reproduced in this repository), `DERIVED` (follows mathematically
from an approved clause), `OPEN` (no controlling decision yet).

---

## C-1 Artifact class and caller-trust model — `APPROVED`

Adjungo is a **reference / pilot implementation** for an eventual C++ library. It
is not a hardened general-purpose library.

Caller-trust model: **data-only inputs and designed extension points**. The
`Problem` and `Objective` protocols are user-supplied callbacks consumed by
value. Adjungo does not buy robustness against hostile callers. A review finding
that presupposes a stronger adversary than this is a proposed amendment to this
clause, not a defect report.

Consequences:

- Prefer auditable, directly readable formulations over Python-specific
  cleverness. The code is read by a future reimplementer.
- Public API shape, ownership, and error reporting should map onto C++ without
  requiring Python semantics.
- See [C-13](#c-13-what-a-c-reimplementation-must-preserve).

---

## C-2 Primary accuracy claim — `APPROVED`

For every **certified** method family (see [C-6](#c-6-envelope-enforcement)), at a
**fixed mesh**:

- `GLMOptimizer.gradient(u)` returns the **exact gradient** of
  `GLMOptimizer.objective_value(u)` with respect to the discrete control
  variables `u ∈ ℝ^(N×s×ν)`;
- `GLMOptimizer.hessian_vector_product(u, v)` returns the **exact action** of the
  Hessian of that same discrete function.

"Exact" means: equal up to the floating-point floor derived in
[C-3](#c-3-accuracy-basis-and-derived-floors). This is a statement about the
*discrete* objective the code actually evaluates. It is **not** a statement about
the continuous optimal-control problem; that is [C-4](#c-4-continuous-accuracy).

**A derivative discrepancy is never explained by time-discretization error.**
Discretization error changes *which* function is being differentiated; it does
not change the fact that the adjoint must differentiate that function exactly.

---

## C-3 Accuracy basis and derived floors — `APPROVED`

Accuracy is measured relative to `max(‖∇J‖_∞, 1)`.

### C-3.1 Certified tolerance, nonlinear problems

`1e-9` relative, demonstrated by a fixed-mesh central-difference ε-sweep showing
**observed order 2 and no plateau**. An ε-independent error plateau is a
falsification of [C-2](#c-2-primary-accuracy-claim), not a tolerance question.

### C-3.2 The LTI-quadratic anchor and its real floor — `DERIVED`

For a linear time-invariant problem with a quadratic cost, the discrete reduced
objective is **exactly quadratic in `u`**, so the central difference quotient has
**zero truncation error in exact arithmetic**.

It does **not** follow that the observed error is zero. In floating point the
achievable floor is set by:

1. **Cancellation in the difference quotient.** `J(u+εe) − J(u−εe)` subtracts two
   nearly equal quantities, each carrying relative error `~ε_mach`. The quotient
   therefore carries absolute error of order

   ```
   ε_round  ≈  ε_mach · |J| / ε
   ```

   which *grows* as ε shrinks. At `|J| ~ 1` and `ε = 1e-6` this is `~1e-10`.

2. **Stage-solve residual.** For implicit methods the discrete objective is only
   evaluated as accurately as the stage equations are solved. This contributes at
   the [C-5](#c-5-nonlinear-stage-solves) residual level, which is why C-5's
   tolerance is pinned well below C-3.1's.

3. **Accumulation over `N` steps**, contributing a factor that grows at worst
   linearly in `N` for the reductions performed here.

**Therefore:** tests assert against a floor computed from the above with the
derivation written beside the assertion, and choose ε near the minimiser of
`ε_round + ε_trunc`. No test may assert "exact" or use a bare `ε_mach` as a
domain tolerance.

### C-3.3 Reporting

A derivative check reports the ε-sweep table and the observed order, not a single
pass/fail at one ε. A single-ε check cannot distinguish a wrong adjoint from
finite-difference noise; this repository has twice mistaken one for the other.

---

## C-4 Continuous accuracy — `APPROVED`

Stated and tested **separately** from C-2. Each certified method attains its
nominal classical order of convergence on a closed-form or manufactured problem,
under mesh refinement.

Tests for C-4 are named so they can never be confused with C-2 tests. C-4 tests
refine the mesh; C-2 tests hold it fixed.

**C-4 never justifies a C-2 discrepancy.**

---

## C-5 Nonlinear stage solves — `APPROVED`

### C-5.1 Convergence criterion

Newton iterations converge on a **scaled** residual:

```
‖r(Z)‖_∞  ≤  rtol · ‖Z‖_∞  +  atol
```

with `rtol = 1e-13`, `atol = 1e-13 · y_scale`, where `y_scale` is the
characteristic state magnitude (default: `max(‖y₀‖_∞, 1)`).

An unscaled absolute test on raw magnitudes is prohibited: it is simultaneously
too strict for large states and too loose for small ones.

### C-5.2 Tolerance ordering — `DERIVED`

C-5.1's tolerance is set **well below** C-3.1's certified derivative tolerance,
so that stage-solve error cannot masquerade as derivative error. If the two were
comparable, a C-2 failure would be unattributable.

### C-5.3 Failure reporting

Exhausting the iteration budget **raises**. The exception names the step index,
the stage index, the final residual norm, and the iteration count. Silent
non-convergence is prohibited; see [C-7](#c-7-no-silent-sentinels).

### C-5.4 Converged-iterate derivatives — `APPROVED`

The Jacobian stored for adjoint and sensitivity use **must be evaluated at the
returned iterate**, and the stored factorization must be the factorization of
that Jacobian.

This is the gate/route identity requirement: the operator that is certified and
the operator that is differentiated must be the same object. A factorization
computed at iterate `k` and returned alongside iterate `k+1` violates this clause
even when the residual is small.

---

## C-6 Envelope enforcement — `APPROVED`

### C-6.1 Certified families

A method family is **certified** only when a completed milestone has validated
forward solve, gradient, **and** Hessian-vector product for it, against the
[C-14](#c-14-the-certification-test-population) population.

| Family | Status |
|---|---|
| Explicit Runge–Kutta | pending milestone M1 |
| DIRK | pending milestone M2 |
| SDIRK | pending milestone M2 |
| Fully implicit (dense `A`) | pending milestone M3 |
| Linear multistep, `r > 1` | **not supported** |
| IMEX / additive splitting | **not supported** |

### C-6.2 Hard refusals

The following are refused at `GLMOptimizer` construction or at stage-solver
construction. These are **hard guards**: no override, no warning-and-proceed.

- `method.r > 1`;
- additive / IMEX tableaux;
- an objective requiring second derivatives the problem does not supply;
- any method family not yet certified by a completed milestone.

Refusal raises `NotImplementedError` with a message naming the unsupported
feature and the supported alternatives.

### C-6.3 Retention of uncertified tableaux

Uncertified tableau *definitions* are **retained** under
`adjungo/methods/experimental/` for future work. They are unreachable from
`GLMOptimizer` by C-6.2. Retention is not endorsement: nothing in
`experimental/` carries any accuracy claim.

---

## C-7 No silent sentinels — `APPROVED`

No certified path may return a zero array, a NaN, an infinity, or a default value
in place of a computed result.

A placeholder that returns zeros is a **contract violation**, not an incomplete
feature, because the caller cannot distinguish it from a correct answer. Such
code is deleted and replaced by a C-6.2 refusal until it is implemented.

NaN and infinity are permitted as *data* only where a clause explicitly defines
that output semantics. No clause currently does.

---

## C-8 GLM convention — `APPROVED`

Adjungo represents a General Linear Method by the tableau `(A, U, B, V, c)`.

### C-8.1 Step form

```
Z_i      = Σ_k U_ik y_k^[n-1]  +  h Σ_j A_ij f(Z_j, u_j, t_n + c_j h)
y^[n]    = V y^[n-1]           +  h ( B f_stages )
```

**The factor `h` is applied by the stepping code and is never absorbed into
`A` or `B`.** A tableau whose external vector stores `h·f` history must respect
this when its `V` and `B` blocks are written.

### C-8.2 Shape invariants

```
A : (s, s)      U : (s, r)      B : (r, s)      V : (r, r)      c : (s,)
```

where `s` is the number of internal stages and `r` the number of external stages.
These are enforced at `GLMethod` construction. A tableau storing per-history
coefficients in the rows of `A` is malformed, not merely unconventional.

---

## C-9 Objective decomposition — `APPROVED`

The discrete objective is the sum of exactly three kinds of term, which have
**distinct meanings** and must not be conflated:

```
J  =  Φ( y^[N] )                                   (terminal)
   +  Σ_n  φ_n( y^[n] )                            (node-based)
   +  h Σ_n Σ_k  w_k · ℓ( Z_k^n, u_k^n, t_n + c_k h )   (stage quadrature)
```

### C-9.1 Terminal cost `Φ`

Evaluated once, at the final external state. Carries no `h` and no weight.

### C-9.2 Node-based terms `φ_n`

For **sampled observations and penalties** evaluated at external states. These
are *not* a quadrature: they carry **no `h` factor and no weight `w`**. A
misfit against data sampled at step boundaries is a node-based term.

### C-9.3 Stage quadrature `ℓ`

For **continuous running costs**. The weights `w_k` are the method's quadrature
weights and `h` appears **once**, exactly as written above. This placement is
normative: a reimplementation that folds `h` into `w`, or applies it twice,
produces a different discrete objective and therefore a different exact gradient.

For a Runge–Kutta method in GLM form, `w_k = B[0, k]`.

### C-9.4 Required derivatives

Each of the three kinds requires first, second, **and mixed** derivatives:

| Term | First | Second | Mixed |
|---|---|---|---|
| Terminal `Φ` | `∂Φ/∂y` | `∂²Φ/∂y²` | — |
| Node `φ_n` | `∂φ/∂y` | `∂²φ/∂y²` | — |
| Stage `ℓ` | `∂ℓ/∂Z`, `∂ℓ/∂u` | `∂²ℓ/∂Z²`, `∂²ℓ/∂u²` | `∂²ℓ/∂Z∂u` |

An objective that omits a required derivative is refused under C-6.2 rather than
silently treated as zero.

---

## C-10 Control coordinates and inner product — `APPROVED`

### C-10.1 Two layers

- **Internal layer.** Derivatives are computed with respect to the stage controls
  `u ∈ ℝ^(N×s×ν)`. All of C-2 is stated in this layer.
- **Adapter layer.** The optimization adapter maps a parameter vector `θ` to
  stage controls and returns derivatives in `θ`.

### C-10.2 Affine parametrizations — `DERIVED`

For `u = P θ + q`:

```
g_θ    = Pᵀ g_u
H_θ v  = Pᵀ H_u (P v)
```

Piecewise-constant-per-step and nodal-interpolation-to-abscissae are both affine
and are therefore both expressed by `P`.

### C-10.3 Nonlinear parametrizations — `APPROVED`, unimplemented

A nonlinear map `u = m(θ)` requires an additional curvature term:

```
H_θ v  =  (∂m/∂θ)ᵀ H_u (∂m/∂θ) v  +  Σ_i (g_u)_i  ∂²m_i/∂θ² v
```

The second term is **not optional**. Omitting it yields a Gauss–Newton
approximation, which is a different operator and may not be presented as the
exact Hessian under C-2.

### C-10.4 Inner product — `APPROVED`

The adapter returns the **coordinate derivative** in the flat parameter vector:
the quantity `∂J/∂θ_i` such that `J(θ + δ) ≈ J(θ) + Σ_i (∂J/∂θ_i) δ_i`.

It does **not** return a Riesz representative under any weighted inner product.
An `h`-weighted L² representative is natural for this problem class and is
therefore an easy mistake: `scipy.optimize.minimize` interprets `jac` as the
coordinate derivative of the supplied `fun`, so returning a representative would
cause silently wrong steps rather than an error.

If a weighted representative is ever exposed, it is a separate, differently named
method, and this clause is amended.

---

## C-11 Platform, backend, and pinning — `APPROVED`

### C-11.1 Recording

CI records the linked BLAS/LAPACK backend in its log
(`numpy.show_config()` plus the resolved LAPACK provider).

### C-11.2 Routes used — `OBSERVED`

Certified routes use **LU factorization only**: `getrf` / `getrs` via
`scipy.linalg.lu_factor` / `lu_solve`, and `gesv` via `numpy.linalg.solve`.

They do **not** call `dgesvd` or `dgesdd`, the routines for which legacy Apple
Accelerate LAPACK holds defect evidence. Development machines may therefore link
Accelerate without affecting certified routes. Any future work that introduces an
SVD-based route must revisit this clause and pin OpenBLAS.

### C-11.3 Pinning policy — `APPROVED`

**No fixture pins bit-exact floating-point output.** Development machines link
Apple Accelerate; CI links OpenBLAS; the two differ in the last ULPs of
LU-based reductions.

Every numeric pin is a tolerance with its **basis written beside it** — either a
derived bound under C-3, or a stated relative/ULP budget. A pin with no stated
basis certifies a machine, not a method, and is a review finding.

---

## C-12 Degeneracy policies — `APPROVED`

| Configuration | Policy |
|---|---|
| `N = 0` or `h = 0` | **Invalid.** Hard guard, refuse |
| `t_span[1] == t_span[0]` | **Invalid.** Hard guard, refuse |
| `a_ii = 0` inside an otherwise implicit tableau | **Counts as an explicit stage.** No solve is attempted |
| `I − h·a_ii·F` singular or numerically singular | **Refuse**, with a diagnostic naming step and stage. Never a pseudo-inverse or regularized fallback |
| Newton non-convergence | **Refuse** per C-5.3 |
| `ν = 0` (no controls) | **`OPEN`.** Currently unreachable through the public API |

---

## C-13 What a C++ reimplementation must preserve

Recorded under C-1. A port that changes any of these produces different numbers:

1. **The C-8.1 placement of `h`** — applied by the stepping code, never absorbed
   into `A` or `B`.
2. **The stage-index discipline of the adjoint recursion.** The correct relation
   is

   ```
   μ_k = h F_kᵀ ( Σ_i a_ik μ_i  +  Σ_l b_lk λ_l )  =  h F_kᵀ Λ_k
   ```

   `F_kᵀ` factors out of the **entire** weighted sum. Applying `F_jᵀ` to the
   coupling term from stage `j` is wrong, and is invisible to any test that uses
   a single-stage method or a constant Jacobian. This exact error existed in this
   repository and was not caught by 89 passing tests. See
   [C-14](#c-14-the-certification-test-population).

   Note the asymmetry: the *tangent* recursion correctly uses `F_j`, because
   there the Jacobian genuinely belongs to the stage being differentiated.

3. **C-5.4 converged-iterate derivatives** — the factorization the adjoint uses
   is the factorization of the Jacobian at the returned iterate.
4. **The C-9 objective decomposition**, including exactly where `h` and `w` enter.
5. **The C-10.4 coordinate-derivative convention.**
6. **The C-6.2 hard refusals**, which must not degrade into warnings in a
   performance-oriented port.
7. **The independent reference of C-14.1**, which should be ported alongside the
   solver rather than reinvented.

---

## C-14 The certification test population — `APPROVED`

A derivative check is only as strong as the problem it runs on. Certification
under C-2 requires problems that **jointly** exhibit:

| Property | What it catches |
|---|---|
| `s > 1` | Stage-coupling errors, invisible at `s = 1` |
| State- **and** control-varying Jacobian | Stage-index errors, invisible when `F` is constant |
| `n ≠ ν`, both `> 1` | Transpose and shape errors that square matrices hide |
| Nonzero curvature (`F_yy`, `F_yu`, `F_uu` all non-vanishing) | An untested second-order path |
| All three C-9 term kinds simultaneously, unequal weights | Missing running/terminal contributions |
| `t_span[0] ≠ 0` | Time arguments defaulting to zero |
| Non-uniform controls across stages and steps | Index and broadcast errors |

### C-14.1 Oracle hierarchy

In decreasing order of authority:

1. **Independently assembled discrete reference.** The whole discrete system as a
   single residual `R(w, u) = 0` with `w = (Z¹…Z^N, y^[0]…y^[N])`, assembled
   directly from the tableau and the problem callbacks, giving

   ```
   dJ/du = J_u − J_w (∂R/∂w)⁻¹ ∂R/∂u
   ```

   by a dense solve. This shares **no code** with `adjungo/stepping/`, so it
   cannot share a mistake with the tangent or adjoint implementations.

   Implemented — `OBSERVED` — as `adjungo/validation/reference.py`
   (`reference_solve`, `reference_gradient`, `reference_hessian`), exercised by
   `tests/test_oracle_gradient.py`. Two properties make it load-bearing rather
   than decorative:

   - `test_reference_forward_reproduces_package_forward` pins the premise that
     the residual defines *the same* discrete map the package steps, to
     `< 1e-12` relative. Without this, agreement of gradients would prove
     nothing.
   - `test_oracle_detects_the_b0_stage_index_defect` re-installs the pre-fix
     recursion and requires the oracle to reject it, so the acceptance tests
     cannot decay into tautologies.

2. **Closed-form anchors.** Notably the one-step case
   `y₁ = y₀ + h u`, `J = ½ y₁²`, giving `g = h y₁` and `H v = h² v`. This settles
   sign conventions with no floating-point argument.

3. **Fixed-mesh ε-sweep** per C-3.3.

4. **Duality (dot-product) test.** `⟨∇J, v⟩` from the adjoint versus the
   directional derivative from the tangent solve. **Corroboration only**: it can
   pass when tangent and adjoint share a mistake.

5. **Structural checks.** Hessian symmetry. Necessary, never sufficient — this
   repository held a symmetric Hessian accurate to `8.7e-19` in symmetry while
   being wrong by `3.8e-3` in value.

---

## Open questions

| ID | Question | Blocks |
|---|---|---|
| C-Q3 | Is `ν = 0` (uncontrolled trajectory) a supported configuration? | Nothing currently |
| C-Q4 | Do multistep external vectors hold `y` history or `h·f` history under C-8.1? | Any future `r > 1` work |
| C-Q5 | Characteristic state scale `y_scale` for C-5.1 — is `max(‖y₀‖_∞, 1)` adequate for problems with large transients? | Tightening C-5 |

---

## Precedents

**R-1 — A single-ε finite-difference check cannot certify an adjoint.**
Two defects in this repository (a DIRK adjoint that never applied the transposed
stage solve, and a second-order adjoint missing its terminal term) were
attributed to time-discretization error on the basis of one-sided checks at a
single ε. Both showed ε-independent plateaus under a sweep. Superseded by C-3.3.

**R-2 — Test populations conceal stage-index errors.**
The adjoint stage recursion applied the wrong stage Jacobian to its coupling
term. Eighty-nine tests passed. The error is identically invisible at `s = 1`
(empty coupling loop) and when `F` is constant (`F_i ≡ F_j`). Superseded by C-14.

**R-3 — Symmetry is not correctness.**
A Hessian-vector product was symmetric to `8.7e-19` while being wrong by
`3.8e-3`. A consistently wrong symmetric operator is still symmetric. Superseded
by C-14.1 item 5.

**R-4 — A placeholder that returns zeros is a contract violation.**
`ImplicitStageSolver` returned `Z = 0` with no warning, so a Gauss-2 forward
solve returned the initial condition unchanged. Superseded by C-7.

**R-5 — The B0 stage-index defect, cured and measured.** — `OBSERVED`
Differentiating the stage equations gives
`μ_i = h Fᵢᵀ ( Σ_j A[j,i] μ_j + Σ_l B[l,i] λ_l )`. The stage Jacobian is
evaluated at stage `i` and factors out of the **entire** weighted sum, because
`Z_i` enters every stage equation only through `f(Z_i, …)`. The code applied
`F_jᵀ` to the coupling term in four places: `solvers/explicit.py`,
`solvers/dirk.py`, `solvers/sdirk.py`, and `stepping/sensitivity.py`.

Measured on `CoupledNonlinear` (`n = 3`, `ν = 2`), `N = 6`, `t_span = (0.3, 1.1)`,
against `reference_gradient`, as relative error on the C-3 basis:

| Method | `s` | Before | After |
|---|---|---|---|
| `explicit_euler` | 1 | 4.34e-17 | 4.34e-17 |
| `heun` | 2 | **2.28e-05** | 1.67e-16 |
| `rk4` | 4 | **1.13e-05** | 1.58e-16 |

Two lessons are recorded rather than the fix alone:

1. The `s = 1` column is unchanged **by construction**, and 13 of the 14
   quarantined diagnostic scripts used `explicit_euler`. The historical
   investigation was structurally incapable of seeing this defect.
2. Under the C-3.3 ε-sweep the defect presented as a **flat plateau** — relative
   disagreement `2.907e-03` at every ε from `1e-2` to `1e-7`, varying only in
   the fourth significant figure. An ε-independent plateau is the signature of a
   wrong derivative. It is never round-off, and per C-2 it is never
   time-discretization error.

`forward_sensitivity` (`stepping/sensitivity.py`) was **not** affected: it
legitimately uses `F_j`, because there the Jacobian genuinely belongs to stage
`j`. Tangent and adjoint therefore did not share this mistake, which is why
duality would not have caught it either — see C-14.1 item 4.
