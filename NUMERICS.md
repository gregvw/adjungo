# NUMERICS.md — Adjungo domain contract

Status of this document: **normative**. Every accuracy, envelope, or degeneracy
claim made by Adjungo code, tests, or documentation must cite a clause here.
Clauses are stable identifiers; do not renumber them.

Fact-status labels used below: `APPROVED` (ruled by the scientific owner),
`OBSERVED` (reproduced in this repository), `DERIVED` (follows mathematically
from an approved clause), `OPEN` (no controlling decision yet).

---

<a id="c-1"></a>
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

<a id="c-2-primary-accuracy-claim"></a>

<a id="c-2"></a>
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

<a id="c-3-accuracy-basis-and-derived-floors"></a>

<a id="c-3"></a>
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

### C-3.4 Certified tolerance, implicit methods — `DERIVED`

Package-versus-reference comparison for an implicit method has a floor that
explicit methods do not have, and it is **not** machine epsilon.

Both sides solve the same stage equations, but each stops Newton at the
**scaled** [C-5.1](#c-5-nonlinear-stage-solves) threshold
`rtol·‖Z‖_∞ + atol` (`adjungo/solvers/newton.py::stage_solve_tolerance`).
Neither lands on the exact root. `STAGE_NEWTON_TOL` is that threshold's value
at unit scale, `NEWTON_RTOL + NEWTON_ATOL_FACTOR = 2e-13`. Writing `κ` for a
bound on the stage-Jacobian inverse over a step,

```
‖Z*_pkg − Z*_ref‖  ≲  κ · STAGE_NEWTON_TOL
```

Each side's gradient is the *exact* derivative of the map realised at its own
converged iterate — C-2 is not weakened — but the two iterates differ at the
above order, and so do the two gradients.

**Therefore:** the implicit certified tolerance is a stated multiple of the
Newton stopping tolerance, not an independent number:

```
IMPLICIT_RTOL = CONDITIONING_ALLOWANCE × STAGE_NEWTON_TOL = 1e3 × 2e-13 = 2e-10
```

Tests **import** `STAGE_NEWTON_TOL` rather than hardcoding a value, so loosening
Newton convergence cannot silently loosen a correctness claim.

`OBSERVED`: over the C-14 implicit population (`implicit_midpoint`,
Crank-Nicolson, `sdirk2`, `sdirk3` at `N = 3` and `N = 6` on `CoupledNonlinear`)
the worst relative gradient error is `6.22e-15`, a ratio of **0.031** to
`STAGE_NEWTON_TOL`. Agreement is *better* than the stopping test guarantees
because Newton's quadratic convergence drives the final residual far below
`tol`. That overshoot is a property of these problems, not a guarantee, which is
why `CONDITIONING_ALLOWANCE` is set well above it.

This observation is pinned by
`tests/test_oracle_gradient.py::test_implicit_tolerance_basis_is_measured`,
which fails if the ratio rises above `CONDITIONING_ALLOWANCE` (the bound stops
holding) **or** above `1.0` (the derivation's overshoot assumption stops
holding). A comment is a claim; that test is the evidence.

---

<a id="c-4-continuous-accuracy"></a>

<a id="c-4"></a>
## C-4 Continuous accuracy — `APPROVED`

Stated and tested **separately** from C-2. Each certified method attains its
nominal classical order of convergence on a closed-form or manufactured problem,
under mesh refinement.

Tests for C-4 are named so they can never be confused with C-2 tests. C-4 tests
refine the mesh; C-2 tests hold it fixed.

**C-4 never justifies a C-2 discrepancy.**

---

<a id="c-5-nonlinear-stage-solves"></a>

<a id="c-5"></a>
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

`OBSERVED` — implemented as `adjungo/solvers/newton.py::stage_solve_tolerance`;
`y_scale` is captured by the stage solver **at construction** from `y₀` and is
never re-read from mutable state, so the threshold a stage is certified against
cannot drift as the trajectory evolves.

The rationale is measured, not asserted, by
`tests/test_implicit_solvers.py::test_stage_convergence_is_scaled_not_absolute`
on `z − 0.05√|z| = Y` at two magnitudes nine orders apart:

| `Y` | achieved residual | achieved relative accuracy | a fixed `1e-12` absolute test would |
|---|---|---|---|
| `1e6` | `1.57e-08` | `1.6e-14` | **reject** it — unreachable, since the double spacing at `1e6` is already `~1e-10` |
| `1e-6` | `1.33e-19` | `5.3e-17` | **accept** anything down to `1e-6` relative |

Both terms are load-bearing and neither may be dropped: without `atol` the
threshold is zero at `Z = 0`; without `rtol·‖Z‖_∞` the prohibited absolute test
returns. This is pinned by `test_stage_solve_tolerance_has_both_terms`.

### C-5.2 Tolerance ordering — `DERIVED`

C-5.1's tolerance is set **well below** the certified derivative tolerance, so
that stage-solve error cannot masquerade as derivative error. If the two were
comparable, a C-2 failure would be unattributable.

The ordering is now explicit rather than coincidental: the implicit certified
tolerance is *defined* as `1e3 ×` the C-5.1 threshold at unit scale
([C-3.4](#c-34-certified-tolerance-implicit-methods--derived)), so three orders
of separation hold by construction and cannot be lost by editing one number.

### C-5.3 Failure reporting

Exhausting the iteration budget **raises**. The exception names the step index,
the stage index, the final residual norm, and the iteration count. Silent
non-convergence is prohibited; see [C-7](#c-7-no-silent-sentinels).

`OBSERVED` — built by `adjungo/solvers/newton.py::stage_context`, which also
includes the stage time because that is what a reader of a physical model
recognises. Example:

```
SDIRK step 0, stage 0, t=0.0292893: Newton failed to converge in 50
iterations; final ||r||_inf = 1.464575e+01, tolerance 1.464575e-12
```

All four required locators are asserted individually by
`tests/test_implicit_solvers.py::test_newton_reports_the_stage_context_on_failure`,
so a refactor that drops one fails rather than merely degrading a message.

**Amendment (M3), `APPROVED`.** A fully implicit tableau solves all `s` stages
in one coupled `(s·n)` system, so there is no single failing stage to name: the
iteration either converges for every stage or for none. For that route the stage
locator is satisfied by naming the **block carrying the largest residual
component**, which is the stage a reader should examine first.

The amendment narrows rather than relaxes the clause. The step index, the time
interval, the final residual norm, and the iteration count are still required
verbatim, and the reported block index must be the argmax of the per-block
residual, not an arbitrary valid index. Example:

```
fully implicit step 4, coupled stage system, t=[0.7, 1.7], worst block stage 1
(||r_1||_inf = 2.874923e+01): Newton failed to converge in 50 iterations; final
||r||_inf = 2.874923e+01, tolerance 1.188503e-13
```

`OBSERVED` — `tests/test_fully_implicit.py::test_failed_coupled_solve_names_a_stage_block`
asserts each locator, and `::test_worst_block_is_the_block_with_the_largest_residual`
recomputes the argmax from the residual vector the exception carries. The latter
first asserts that the two blocks differ by more than 50%, so a solver that
always reported stage 0 could not pass by coincidence.

### C-5.4 Converged-iterate derivatives — `APPROVED`

The Jacobian stored for adjoint and sensitivity use **must be evaluated at the
returned iterate**, and the stored factorization must be the factorization of
that Jacobian.

This is the gate/route identity requirement: the operator that is certified and
the operator that is differentiated must be the same object. A factorization
computed at iterate `k` and returned alongside iterate `k+1` violates this clause
even when the residual is small.

---

<a id="c-6-envelope-enforcement"></a>

<a id="c-6"></a>
## C-6 Envelope enforcement — `APPROVED`

### C-6.1 Certified families

A method family is **certified** only when a completed milestone has validated
forward solve, gradient, **and** Hessian-vector product for it, against the
[C-14](#c-14-the-certification-test-population) population.

| Family | Status |
|---|---|
| Explicit Runge–Kutta | **certified** — M1 (`explicit_euler`, `heun`, `rk4`) |
| DIRK | **certified** — M2 (`implicit_trapezoid` / Crank–Nicolson) |
| SDIRK | **certified** — M2 (`implicit_midpoint`, `sdirk2`, `sdirk3`) |
| Fully implicit (dense `A`) | **certified** — M3 (`gauss2`) |
| Linear multistep, `r > 1` | **not supported** |
| IMEX / additive splitting | **not supported** |

`OBSERVED` — each certified method above is exercised, on the
[C-14](#c-14-the-certification-test-population) population, by:

| Claim | Test |
|---|---|
| Forward solve realises the monolithic residual | `test_oracle_gradient.py::test_reference_forward_reproduces_package_forward` |
| Gradient equals the independent reference | `test_oracle_gradient.py::test_explicit_gradient_matches_independent_reference` |
| Gradient survives a fixed-mesh ε-sweep | `test_oracle_gradient.py::test_gradient_finite_difference_sweep` |
| Hessian operator equals the independent reference | `test_oracle_hessian.py::test_hvp_matches_independent_reference` |
| Hessian survives a fixed-mesh ε-sweep | `test_oracle_hessian.py::test_hvp_finite_difference_sweep` |

The certified list and `adjungo/optimization/interface.py::CERTIFIED_STAGE_TYPES`
must agree; adding an entry to either without the evidence above is a false
certification.

`OBSERVED` — the agreement is checked by
`tests/test_documentation.py::test_certified_families_agree_with_the_code`,
which reads this table and compares it with `CERTIFIED_STAGE_TYPES`. Measured
under [R-11](#r-11) against a 312-test baseline with 0 failures:

| Injected defect | Failing tests |
|---|---|
| Contract marks a refused family (`r > 1` multistep) as certified | 1 |
| Contract downgrades a family the code certifies | 1 |
| A fabricated family row is appended to the table | 1 |
| An archived historical report loses its superseded banner | 1 |
| An archived report is dropped from the `docs/history/` accounting table | 1 |
| An ad-hoc report reappears in the repository root | 1 |
| An internal cross-reference anchor is deleted | 1 |
| `docs/architecture.md` names a module that does not exist | 1 |
| `docs/architecture.md` omits a module that does exist | 1 |
| The `glm_opt.tex`-to-code symbol map cites a stale attribute | 1 |
| The symbol-map table is removed | 1 |
| `docs/architecture.md` stops recording IMEX as unbuilt | 1 |
| `docs/architecture.md` lists a certified family as unbuilt | 1 |

#### Additional evidence for the fully implicit family

A dense `A` is solved by one coupled Newton system per step rather than by
substitution, so three properties that are structural for the triangular routes
have to be established rather than assumed. They are carried by
`tests/test_fully_implicit.py`:

| Claim | Test |
|---|---|
| The stored factorization is of the analytic Jacobian **at the converged iterate** (C-5.4), checked against a Jacobian built by differencing the residual | `::test_cached_factorization_matches_a_differenced_jacobian` |
| The adjoint operator is the transpose of the forward Jacobian, so the forward factorization may be reused with `trans=1` | `::test_transposed_solve_is_the_adjoint_of_the_untransposed_one` |
| The stage adjoints satisfy `μ_p = h F_pᵀ (Σ_i A[i,p] μ_i + Σ_l B[l,p] λ_l)`, with the `A` coupling in the operator and **not** also in the right-hand side | `::test_adjoint_stage_solve_reuses_the_forward_factorization` |
| Routes do not cross: a coupled cache never carries per-stage factorizations, and a triangular cache never carries a coupled one | `::test_coupled_cache_carries_a_factorization_not_per_stage_ones`, `::test_triangular_methods_do_not_set_the_coupled_factorization` |

`OBSERVED` — sensitivity of that evidence, measured under the
[R-11](#r-11) procedure at a
286-test baseline with 0 failures:

| Injected defect | Failing tests |
|---|---|
| Coupled Jacobian block `(i,j)` uses `F_i` where the derivation gives `F_j` (defect B0 transposed; see [R-5](#r-5)) | 8 |
| Adjoint stage solve drops the transpose (`trans=1` → `trans=0`) | 10 |
| Adjoint right-hand side also carries the `A` coupling, double-counting it | 8 |
| Coupled Jacobian keeps only its diagonal blocks | 8 |
| Factorization taken at the Newton starting point rather than the converged iterate (C-5.4) | 8 |
| Coupled **tangent** solve drops the `h Σ_j A[i,j] G_j δu_j` forcing | 4 |
| Coupled **tangent** solve uses the transposed operator | 4 |
| Coupled **tangent** solve uses `A[j,i]` where the derivation gives `A[i,j]` | 4 |

The last three rows matter because `forward_sensitivity` has its own coupled
branch, reached only by the Hessian path. They confirm it is executed rather
than merely present: a branch that no test enters would show 0 failures for all
three.

The fourth row is worth reading carefully. Dropping the off-diagonal Jacobian
blocks turns Newton into a quasi-Newton iteration, which still converges to the
**same** stage values: the forward solve, and therefore the order study below,
remain correct. Only the derivatives are wrong, because the adjoint reuses that
factorization. A certification resting on forward accuracy alone would have
missed it entirely.

`OBSERVED` — continuous accuracy (C-4): `gauss2` attains observed order 4.0
(rates 3.989, 3.997, 3.999, 4.000 over N = 10…160) against `expm(A T) y0` for a
linear damped oscillator, in
`tests/test_fully_implicit.py::test_gauss2_observed_order_is_four`. Per
[C-2](#c-2-primary-accuracy-claim) this is a refined-mesh claim about the
continuous problem and may never be cited to excuse a fixed-mesh derivative
discrepancy. It is recorded because order 4 from two stages is the property
that distinguishes a correct Gauss solve from one that has degenerated to the
implicit midpoint rule.

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

<a id="c-7-no-silent-sentinels"></a>

<a id="c-7"></a>
## C-7 No silent sentinels — `APPROVED`

No certified path may return a zero array, a NaN, an infinity, or a default value
in place of a computed result.

A placeholder that returns zeros is a **contract violation**, not an incomplete
feature, because the caller cannot distinguish it from a correct answer. Such
code is deleted and replaced by a C-6.2 refusal until it is implemented.

NaN and infinity are permitted as *data* only where a clause explicitly defines
that output semantics. No clause currently does.

---

<a id="c-8"></a>
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

<a id="c-9"></a>
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

Implemented — `OBSERVED` — for the terminal and node terms as the `Objective`
protocol methods `dJ_dy_terminal`, `dJ_dy`, `dJ_du`, `d2J_du2`, `d2J_dy2`, and
`d2J_dy2_terminal`. `adjoint_sensitivity` and
`assemble_hessian_vector_product` raise `NotImplementedError` naming the
missing callback. The stage-quadrature row (C-9.3) is **not yet implemented**;
no `dJ_dZ` exists and the discrete objective the adjoint differentiates has no
`Z` dependence. `adjungo/validation/reference.py` documents the same gap at
`_objective_state_gradient`, so the reference and the package agree about what
is *not* yet supported as well as about what is.

---

<a id="c-10"></a>
## C-10 Control coordinates and inner product — `APPROVED`

### C-10.1 Two layers

- **Internal layer.** Derivatives are computed with respect to the stage controls
  `u ∈ ℝ^(N×s×ν)`. All of C-2 is stated in this layer.
- **Adapter layer.** The optimization adapter maps a parameter vector `θ` to
  stage controls and returns derivatives in `θ`.

### C-10.2 Affine parametrizations — `DERIVED`, implemented

For `u = P θ + q`:

```
g_θ    = Pᵀ g_u
H_θ v  = Pᵀ H_u (P v)
```

Piecewise-constant-per-step and nodal-interpolation-to-abscissae are both affine
and are therefore both expressed by `P`.

Implemented in `adjungo/optimization/parametrization.py` as
`PiecewiseConstantControl` and `NodalControl`, and reached through the
`parametrization=` argument of `scipy_interface` and `scipy_hessp`. `P` is never
formed at run time; the three operations `expand`, `push` and `pullback` are
implemented directly.

**The transpose must not be validated against itself.** A transposition error in
this layer does not raise, does not produce a nonfinite value, and does not
change the shape of the result. It returns a vector that is merely not the
gradient, so a line search still makes progress and convergence degrades
silently. The required chain of evidence is:

1. `expand` is definitional — it produces the stage controls the solver
   integrates.
2. `P` is materialised by differencing `expand` on Cartesian basis vectors.
   Because the map is affine this difference is the linear part **exactly**,
   with no truncation term to bound, so the materialised `P` inherits `expand`'s
   authority.
3. `push` is checked against that `P`.
4. `pullback` is checked against `Pᵀ`, that is against `push`.
5. The parameter-space gradient and Hessian are checked against the independent
   monolithic reference in `u`, composed with the validated transpose.

Hessian symmetry is corroboration only here, as everywhere: `Pᵀ H P` is
symmetric whenever `H` is, so it cannot detect a wrong-but-symmetric `H`.

**Direction versus point.** `push` applies the linear part only. Applying the
full affine map to a Hessian direction would add the offset `q` to a quantity
that is not a point. Both shipped maps have `q = 0`, so this error would be
dormant until the first map with a nonzero offset; the operations are therefore
kept distinct rather than merged into one that happens to be correct today.

**Evidence.** `tests/test_parametrization.py` (48 tests). Checked against four
injected defects, each measured under the R-11 procedure:

| Injected defect | Tests failed |
|---|---|
| none (baseline) | 0 |
| `pullback` averages over stages instead of summing | 7 |
| `NodalControl.pullback` swaps the interpolation weights | 12 |
| `NodalControl` ignores `c` and uses the left endpoint | 2 |
| `pullback` keeps the first stage instead of contracting | 9 |

The swapped-weights row is the important one: it is the transposition error this
layer is designed against, and it is caught by the dedicated transpose test, by
the reference-gradient test, by the reference-HVP test and by symmetry. It
affects only the `nodal` cases, confirming that the checks are specific to the
defective map rather than globally sensitive.

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

<a id="c-11"></a>
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

### C-11.4 Language level — `APPROVED`

The supported interpreter is **CPython 3.13 or later**, declared by
`requires-python` in `pyproject.toml`. The CI matrix must include that floor,
and no CI job may pin an interpreter below it.

The floor is set by the constructs the supported paths actually use, tests
included, not by a preference for breadth. `copy.replace` and the
`__replace__` hook C-15.7's restoration walk models are new in 3.13;
`typing.Self` is new in 3.11. A floor below the constructs in use is not a
wider envelope, it is a false one: the package declares an interpreter on
which it cannot be imported.

The floor also bounds the type-check evidence. mypy's verdict depends on the
stub versions resolved for the interpreter, and numpy's stubs are not stable
across its own releases — the same tree that checks clean under numpy 2.5
reports `Returning Any` under 2.4 and two further assignment errors under
2.2, which is the newest numpy an older interpreter can install. Declaring a
floor therefore fixes a stub range, and widening the floor commits the
repository to every stub difference inside it.

---

<a id="c-12"></a>
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

<a id="c-13"></a>
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
8. **The coupled-system identity for a dense `A` (M3).** The forward stage
   Jacobian has blocks

   ```
   ∂R_i/∂Z_j = δ_ij I − h A[i,j] F_j
   ```

   and the operator acting on the stage adjoints has blocks
   `δ_pi I − h A[i,p] F_pᵀ`, which is exactly the transpose of the above. A port
   must therefore reuse the **forward** factorisation with a transposed solve
   (LAPACK `trans='T'`), not assemble a second matrix. Two consequences are easy
   to get wrong and are invisible to a forward-accuracy test:

   - block `(i, j)` carries `F_j`, the Jacobian of the stage being differentiated
     *with respect to*, not `F_i`;
   - all of the `A` coupling lives in the operator. Adding an `A` term to the
     adjoint right-hand side as well — the natural slip when adapting triangular
     backward substitution — double-counts it.

   A quasi-Newton port that drops the off-diagonal blocks to save work will still
   converge to the correct stage values and still show order 4, while returning a
   wrong gradient. If the port does this deliberately it must factor the true
   Jacobian once at the converged iterate for the adjoint's use.
9. **Why the terminal Hessian contribution is assembled by a backward sweep.**
   The terminal cost contributes curvature through `∂²J/∂y²` at the final
   external stage (precedent [R-6](#r-6)). Assembling that contribution
   *directly* requires the state sensitivity `∂y^[N]/∂u` in full, which costs one
   forward tangent solve per control component — `O(N·s·ν)` solves for a single
   Hessian-vector product, making the operator more expensive than forming the
   dense Hessian by differencing.

   The implemented alternative propagates the terminal term as an initial
   condition of a **second backward sweep** using the same operator as the
   first-order adjoint with a different right-hand side. One extra sweep per
   Hessian-vector product, matrix-free, independent of `ν`.

   A port that reaches for the direct form because it is easier to derive will
   be correct and unusably slow, and the slowness will not appear until `ν`
   grows. Recorded because the cost argument is not visible from the code, which
   simply performs the cheap version.

### What a port must *not* preserve

The list above is semantics: changing any of it changes the numbers. The
coefficient-immutability machinery of [C-15.7](#c-15) is the opposite, and is
recorded here so that a port does not faithfully reproduce scaffolding.

Freezing buffers, re-freezing them in `__setstate__`, recording that the root
initialiser ran, and checking all of it at three call sites exist only because
Python permits an array to be unfrozen and an attribute to be rebound at any
moment. None of it is a statement about General Linear Methods. A C++ port
should **delete it** and obtain the same property from the type system: the
coefficients are `const` members of a problem type, so nothing can unfreeze
them and no check is needed to discover that.

The same applies to the *route decision* it guards. `affine_dynamics_verified`
compares members at runtime and answers a question that in C++ is a property
of the type — affineness is declared once, checked at compile time, cannot
change afterwards, and selects the solve path by static polymorphism rather
than by inspection. The runtime refusals of [C-16.2](#c-16) become overload
resolution or a constraint failure, and the quiet-versus-loud distinction that
C-15.7 spends so much care on largely disappears with them: what is quiet
there is a type that does not match, and what is loud is a compile error.

Preserve the *conclusions* — which problems are eligible for the affine route,
and that a retained coefficient must not change during a solve. Do not
preserve the mechanism that establishes them.

---

<a id="c-14-the-certification-test-population"></a>

<a id="c-14"></a>
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

### C-14.0 The population is not a fixed list — `APPROVED`

The table above states properties a certifying problem must have; it does not
license the belief that one problem carrying all of them certifies everything.
Precedent [R-9](#r-9) is the counterexample: `CoupledNonlinear` satisfies every
row, yet its Jacobian depends on `u`, and that single incidental fact hid a
`4.2e-05` gradient error from all 188 tests in the suite.

**Therefore:** when a code path branches on a *property of the problem* rather
than on the method, the population must contain a problem on each side of that
branch. Current instances:

| Problem | Distinguishing property | Path it certifies |
|---|---|---|
| `tests/problems.py::CoupledNonlinear` | `F` depends on `y`, `u`, and `t`; `n=3`, `ν=2` | The general path |
| `tests/problems.py::LinearTimeVarying` | `F` depends on `t` only | Time-dependent, state-independent |
| `tests/problems.py::ScalarAnchor` | Closed-form reducible | Sign and scale conventions |
| `tests/test_implicit_solvers.py::StateOnlyJacobian` | `F` depends on `y` **only**; `G` constant | Reuse and control-independence (R-9) |

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

### C-14.2 A test must be shown to fail — `APPROVED`

A test earns its place as evidence only when it has been **observed to fail
against the defect it guards**. Passing proves nothing on its own: a test that
asserts a tautology, compares a value to itself, or never reaches the code it
names passes exactly as convincingly as one that works.

Concretely, for every claim certified in this contract:

1. The defect the test guards against is **injected** into the source.
2. The suite is run and the number of failing tests is **recorded**.
3. The injection is **reverted** and the suite is run again, returning to the
   baseline.

Both runs must follow the cache-invalidation procedure of
[R-11](#r-11); a count obtained otherwise may not be quoted here, because a
stale `.pyc` can report zero failures for a defect the interpreter never ran.

The count is reported in the clause it supports, not merely in a commit message,
so that a later reader can tell whether a test population that has since been
edited still has the sensitivity originally claimed. A count that falls to zero
after a refactor is a finding.

**The load-bearing direction is a test that fails to fail.** A test that fails
when it should pass is noisy and gets fixed immediately. A test that passes when
it should fail is silent, and is counted as evidence for as long as it exists.
Every precedent in this document from [R-2](#r-2) onward is an instance of the
second kind.

`OBSERVED` — the oracle itself is protected this way:
`tests/test_oracle_gradient.py::test_oracle_detects_the_b0_stage_index_defect`
re-installs the pre-fix adjoint recursion and **requires the oracle to reject
it**, so the acceptance criterion cannot decay into a tautology while continuing
to report success.

**Status note.** This clause was practised from the beginning — every milestone
in this repository recorded injection counts — but was never written down, while
being cited as authority from three places. It was reconstructed from those
citations rather than newly decided, and states no requirement that was not
already being applied. The gap was found by
`tests/test_documentation.py::test_cited_clause_is_defined`.

---

<a id="c-15"></a>
## C-15 Factorization reuse is certified by count — `APPROVED`

### C-15.1 Reuse must be gated on a declaration, never on a probe

An implicit stage solver may reuse an LU factorization across stages, steps, or
repeated calls **only** when the caller has declared
`ProblemStructure(jacobian_constant=True)`.

It may never decide to reuse by examining the problem at run time. Precedent
[R-9](#r-9) records what that costs: a probe that evaluated `F` at
`(y_history[0], u_i, t_i)` varied the control and the time but never the state,
so a Jacobian depending on `y` alone passed it, the adjoint solved with
`(I − hγF₀)ᵀ` at every stage, and the relative gradient error against the
monolithic reference was 4.24e-05 — seven orders above the C-3 floor — while
every test in the suite passed.

The same prohibition applies to *deduction rules that amount to a probe*. Until
M6, `deduce_requirements` computed

```
can_reuse_across_stages = jacobian_constant or not jacobian_control_dependent
```

The second disjunct is false for exactly the R-9 case: a state-dependent
Jacobian is control-independent and still differs at every stage. It was inert
only because nothing consumed the flag. **Correctness of a gate that nothing
reads is not evidence that the gate is correct.**

### C-15.2 A declaration is a promise; reuse requires the fact

`ProblemStructure` is supplied by the caller and is not checked by declaring it.
Before any stored factorization is returned, the matrix it was taken from is
compared with the matrix now being asked for, **element for element**. Reuse is
returned only on exact equality. Otherwise the declaration is false for this
problem and `DeclaredStructureViolation` is raised.

This is a hard guard with no override, per C-6. The alternative is a derivative
computed from the transpose of a matrix the forward solve did not use, wrong by
an amount that shrinks with `h` — which [C-3](#c-3) forbids reading as
discretization error. The caller always has a correct alternative available at
no cost in accuracy: declare `jacobian_constant=False`.

The comparison is exact and carries no tolerance. That is not strictness; it is
the governing condition. If two stage matrices differ in the last place they
are different matrices, and reusing across them is an approximation of a
quantity [C-2](#c-2) defines exactly. [C-12](#c-12) forbids inventing a
tolerance where the condition is exact.

Verification is affordable *because* it is not free: forming the matrix costs
`O(n²)` and factoring it costs `O(n³)`, so the check is asymptotically cheaper
than the work it guards. A guard that cost more than the optimization would be
a reason to drop the optimization, not the guard.

### C-15.3 The factorization count is a certified quantity — `OBSERVED`

Reuse is the one optimization here that **no accuracy test can detect**. Working
or silently broken, the answers are bit-identical: the matrix is the same, so
its factors are the same. Only the cost moves, and a derivative test cannot see
cost.

The certified quantity is therefore the count. It is asserted **structurally**,
against `SolverRequirements.factorizations_for_solve`, never by timing.

| Family | Distinct stage matrices per step | Whole solve, `jacobian_constant=True` |
|---|---|---|
| Explicit | 0 | 0 — nothing to reuse; prediction is *not applicable*, not zero |
| SDIRK | 1 (the constant diagonal γ) | **1**, independent of `N` |
| DIRK | one per distinct nonzero `A[i,i]` | that many, independent of `N` |
| Fully implicit | 1, of size `s·n` | **1**, independent of `N` |

Without the declaration the count is **not predicted**:
`factorizations_for_solve` returns `None`. Every Newton iteration factors, and
the iteration count is a property of Newton and the initial guess, not of the
dispatch. A number there would assert something the deduction does not know.

Two consequences are asserted separately:

- The adjoint, tangent and second-order adjoint sweeps add **nothing** to the
  count. They apply the transpose via `lu_solve(..., trans=1)` on the forward
  factors. Forming and factoring a transpose explicitly would pass every
  accuracy test and move only this count.
- A constant Jacobian means `f` is affine in `y`, so every implicit stage
  equation is affine in its unknown and Newton converges in one step. There is
  no constant-Jacobian problem that needs several Newton iterations; the
  prediction is not evading a hard case.

Measured at `N = 12`, `n = 3`, `ν = 2` on `ConstantJacobianQuadraticControl`,
gradient evaluation, before and after M6:

| Method | `s` | Before | After | Predicted |
|---|---|---|---|---|
| `implicit_midpoint` | 1 | 16 | 1 | 1 |
| `implicit_trapezoid` | 2 | 16 | 1 | 1 |
| `sdirk2` | 2 | 32 | 1 | 1 |
| `sdirk3` | 3 | 48 | 1 | 1 |
| `gauss2` | 2 | 16 (size 4) | 1 (size 4) | 1 |

(The "before" figures are at `N = 8` on a 2-state LTI problem, the measurement
that motivated the work; the ratio, not the absolute number, is the point. The
factor of two over the naive `s·N` is Newton factoring both during its single
iteration and again at the converged iterate, as [C-5.4](#c-5) requires.)

### C-15.5 The count must be complete, not representative — `APPROVED`

A count taken at the store measures the right quantity only if the store is the
**sole** route to `lu_factor`. That is a claim about the code, and it must be
tested, not asserted.

`OBSERVED` — it was not, when first written. Injection M7 below restored a
single direct `scipy.linalg.lu_factor` call in `newton_solve`, at the converged
iterate, and **nothing failed**. The store still reported one factorization
because the extra one never reached it, and the answers were unchanged because
it factored the same matrix. A per-stage, per-step cost had returned while the
certified count reported the optimization working perfectly.

`tests/test_factorization_reuse.py::test_no_factorization_bypasses_the_store`
now counts at `scipy.linalg.lu_factor` and requires the two numbers to agree,
on the reusing and the non-reusing path.

### C-15.6 Injection evidence — `OBSERVED`

Under [R-11](#r-11) against a 349-test baseline with 0 failures:

| Injected defect | Failing tests |
|---|---|
| Reuse without comparing the matrix (R-9, in declarative form) | 6 |
| Reuse inferred from control-independence instead of the declaration | 1 |
| The store silently stops recording entries | 13 |
| The store keeps a view of the caller's buffer instead of a copy | 1 |
| The DIRK count claims one factorization per stage | 2 |
| Reuse enabled regardless of what the caller declared | 42 |
| Newton bypasses the store for the converged-iterate factorization | 2 |
| The prediction returns a number where the count is unpredictable | 6 |

Two of these initially failed to be detected, and both are recorded because the
reason is the same in each case: **a gate that nothing reads cannot be tested by
behaviour.**

- The control-independence deduction rule was wrong and inert, because no
  solver consumed `can_reuse_across_stages`. It is now asserted directly.
- The Newton bypass is invisible to every accuracy test and to the store's own
  count; see C-15.5.

The 42-failure result is the guard behaving correctly: enabling reuse against a
false declaration converts what R-9 recorded as a silent 4.24e-05 gradient error
into an immediate refusal.

Evidence: `tests/test_factorization_reuse.py`,
`tests/test_requirements.py::test_requirements_dirk_counts_distinct_diagonals_not_stages`.

### C-15.4 Reuse must reproduce the unreused result exactly — `APPROVED`

Enabling reuse must change **no** certified number, and the required agreement
is bit-identity, not a tolerance.

This is not a pinned float and does not conflict with
[C-11](#c-11)'s prohibition on pinning: nothing is recorded, and no two
different computations are compared. Under C-15.2 the stored matrix is
identical to the matrix that would otherwise be factored, so a deterministic
`lu_factor` yields identical factors and an identical right-hand side an
identical solution. The argument holds on any platform and under any BLAS
because it never crosses between two computations.

If this agreement ever fails by a small amount, the correct reading is **not**
roundoff. It is that some input was not identical and C-15.2's guard was
reached with something it should have rejected.

<a id="c-15-7"></a>
### C-15.7 Coefficient immutability must survive copying — `APPROVED`

`AffineDynamics.F` and `G` return coefficient buffers by identity, and the
forward tape retains them. Construction and supported copying must preserve
the coefficients' values, immutability and provenance so that the backward
sweeps read the matrices used by the forward solve. A quiet refusal of the
affine optimization is not a remedy: the general solver route retains the
same buffers. This obligation has the caller boundary stated below; it is
not a lifetime guarantee against mutation by the caller.

The detailed [failure history and evidence](docs/evidence/coefficient_immutability.md)
preserve the counterexamples, superseded approaches and measurements that
led to these rules. This clause remains the normative source. Moving the
history changes neither the supported envelope nor the certification status.

#### Coefficients and validity

The constructor copies `_M`, `_C` and `_b` into owning, non-writeable base
numeric `ndarray` buffers. A frozen view does not establish ownership;
object-dtype arrays and `ndarray` subclasses are refused because freezing
their data does not freeze their referents or additional state. The
constructor also refuses subclass array arguments rather than silently
discarding a mask or other state. Ordinary sequences may still be converted.

Within the `AffineDynamics` class gate, validity requires the union of:

- All three coefficients when the root-initialised marker is present.
- The coefficients required by inherited class readers: `F` obliges `_M`,
  `G` obliges `_C`, and `f` obliges all three.
- Every root coefficient name that resolves on the instance, regardless of
  the resolved value's type. This includes delegating overrides.

These are widenings, not interchangeable tests. A required name that is
missing, or resolves to storage outside the coefficient envelope, is refused
by name. An unrelated problem, or a valid general-route subclass outside
these obligations, must not acquire a coefficient refusal merely because
affine eligibility is false. Reusing a root coefficient name for mutable
storage in a subclass is an accepted loud over-refusal; renaming that storage
removes the collision.

The marker is read and written through the root's own instance-dictionary
descriptor. Class identity and namespace questions use the underlying
`type` descriptors. Coefficients instead use the same attribute-resolution
function in construction, restoration and checking, so slots, properties and
consistent descriptors resolve as the actual readers do. A descriptor that
does not resolve a required coefficient produces a named refusal.

Validity precedes eligibility. `require_immutable_coefficients` checks at:

| Call site | Obligation |
|---|---|
| `affine_dynamics_verified` | Check validity before any quiet eligibility refusal. |
| `GLMOptimizer.__init__` | Refuse before any solve, with explicit or deduced structure. |
| `forward_solve` | Check where buffers enter the tape, including low-level compositions. |

Specific buffer defects must be reported before absence of provenance.
Returning `False` from affine verification cannot replace a required loud
refusal under [C-7](#c-7).

#### Construction, restoration and provenance

The default copy and pickle protocols restore through `__setstate__`, which
accepts both an instance dictionary and `(instance_dict, slot_values)`.
The hooks `__setstate__`, `__getstate__`, `__deepcopy__`, `__copy__`,
`__replace__`, `__reduce__` and `__reduce_ex__` remain part of affine
eligibility under [C-16.9](#c-16); validity is enforced even when a hook makes
the affine route ineligible. An external `copyreg` reducer for the problem
class runs ahead of the default protocols and remains outside this guarantee
under [C-1](#c-1). Registrations for objects within the produced state are
inspected by the traversal below. The following rules apply together:

1. A setter may relocate the array handed to it, but may not substitute a
   different one. Construction and restoration check identity after **all**
   setters have run. Restoration records expectations for every coefficient,
   including those it leaves in place, and re-enumerates names afterwards so
   a newly appearing coefficient is not exempt. A refusal distinguishes an
   own-setter substitution, a later substitution, an unassigned substitution
   and a newly appearing name.
2. Restoration reads the finished object's resolving coefficients, not just
   keys in the arriving state. A surviving root marker requires all three;
   restoration records what it established. A coefficient not safe to retain
   is copied and frozen before assignment, and **all** copies are taken before
   the first setter runs. Values are re-read and frozen after the setters.
   Restoration must not mutate shared source storage, dispatch to a buffer's
   overridden `copy`, or erase subclass state that the validity guard must
   refuse. A replacement that cannot be stored is refused by name and reason.
3. Owning and frozen is necessary but insufficient for retaining an arriving
   coefficient. The root's state record vouches for the exact array resolved
   when state was produced. A carried coefficient is exempt from replacement
   only when its identity agrees and the state was handed through unchanged;
   rebuilding can create aliases even when memoisation preserves identity.
   A coefficient recorded as not carried keeps its exemption on either route,
   subject to the owning/frozen postcondition and the envelope below.
   The record is attached to a copied state mapping, never the source's live
   dictionary, and removed before the state is installed on the restored object.
4. "Carried" concerns the whole copying traversal: both state mappings,
   containers, instance attributes and slot descriptors. An inconclusive
   traversal withholds the exemption and attempts replacement; it does not
   automatically refuse a problem whose coefficient can be safely replaced.
   Unmodelled copying or lookup behaviour, a subclass `__new__`, and object
   arrays cannot be treated as empty state. Registrations in
   `copyreg.dispatch_table` are checked by exact type **before atomicity**;
   the standard `copyreg.pickle_complex` entry is modelled by identity.
   Inert subclasses and names written by the copy protocol itself must not
   be mistaken for new copying behaviour. Supported incidental state remains
   answerable, including repeated and mixed duplication routes.
5. Storage is not reconstruction: the traversal does not follow `ndarray.base`
   or `memoryview.obj`. Default deep copying and in-band pickling rebuild
   arrays independently. The consequences for out-of-band buffers, including
   the replacement-provider exclusion, are stated below.
6. For rooted instances, a module-level register records the exact arrays
   established by root construction or restoration. It is keyed by the array,
   with weak-reference identity confirmation, rather than by the problem.
   A copying hook may share already established frozen arrays; rebuilding
   arrays while bypassing root establishment cannot vouch for them. Unrooted
   subclasses still owe the buffer checks above, but the provenance of their
   own storage is their responsibility.

The [scalar explicit-Euler fixture](docs/evidence/coefficient_immutability.md#scalar-euler-fixture)
specifies the problem, controls, mesh, objective and closed-form displacement
used in the observations below. Its executable witnesses are in
`tests/test_coefficient_immutability.py`.

<a id="c-15-7-envelope"></a>
#### What the check is, and what it is not — `ENVELOPE`

It is a **lightweight sanity check at the points where retention happens**, not
a lifetime guarantee, and the three call sites do not change that. In a
language where any array can be unfrozen and any attribute rebound at any
moment, no runtime check is a guarantee; the guard exists to catch the defect
class that arises *without anyone intending it* — `copy.deepcopy` returning
writeable buffers, a subclass copying in `__init__`, a property or slot
displacing the frozen storage, a delegating override. That is what it is
claimed to reach, and the [historical injection table](docs/evidence/coefficient_immutability.md#historical-campaign)
is the evidence.

What it does not reach is anything a caller does deliberately after the check:
unfreezing a buffer once the tape is built (`OBSERVED`: a cached
`objective_value(U)`, then `p.M.flags.writeable = True` and a mutation, then
`gradient(U)` returns `0.7552125`), handing `adjoint_solve` a trajectory built
by hand — it takes a `Trajectory`, not a problem, so it has no `problem` to
check — or a descriptor that answers the check and the solve differently, or a raw
`ctypes` pointer taken to a frozen buffer's memory, or a coefficient setter
that unfreezes the array it is handed, derives a handle from it and freezes it
again before storing it (`OBSERVED`: on the linked scalar Euler fixture at `C = 2`, verified,
frozen and owning, with a writeable alias sharing its memory, `0.7552125`), or
a subclass that never calls `super().__init__` and aliases coefficient storage
it built itself, or a **reducer bound on the instance** — `self.__reduce_ex__ =
...`, which `copy` honours because it fetches that hook with `getattr` rather
than invoking it implicitly. The last cannot be answered where the question
arises: restoration runs *on the object the state describes*, whose instance
dictionary holds whatever the arriving state put there, and a reducer that
fabricates a coefficient omits itself from the state it builds. Rebinding the
duplication protocol per instance is the same act as rebinding the coefficient
per instance, and is excluded on the same grounds. Overriding those hooks on
the **class** is ordinary and is covered above.

Nor does it reach a subclass that **forges the record itself**, writing the
provenance key into the state it builds so that the array it substituted is the
array it claims the root resolved. The record is a private key, not a
signature, and it is kept honest by the same thing that keeps a frozen flag
honest — nothing intending to be checked defeats it by accident. Nor does it
reach a coefficient that never travels in the state *and* is manufactured by
the class each time it is read; that is the inconsistently-answering descriptor
already excluded above, seen from the restoration side. Nor does it reach a
**dispatch table set on a particular `Pickler`**: `copyreg.dispatch_table` is
module-level and can be read, whereas an instance table is chosen by the
caller doing the pickling and `__getstate__` is not told which pickler is
running. That is the instance-bound reducer again, moved to the other side of
the protocol, and `Pickler.reducer_override` and the persistent-id hooks
belong to it for the same reason: all of them are chosen by whoever runs the
pickler, and none of them is visible from the object being pickled.

**Out-of-band buffers** join that list, and are the sharpest case in it.
Protocol 5 serialises a buffer out of band only when the caller passes
`buffer_callback` to the dump, and the restored object reads whatever the
caller then passes as `buffers` to the load. Neither end is visible from the
object, and the two need not agree: a caller can hand back a buffer other
than the one produced, backed by memory it retains a writeable view of.
`OBSERVED` on the linked scalar Euler fixture at `C = 2` with a coefficient answered from
the array that exports a metadata view, and one substituted buffer: the
clone verified with `_M` frozen and owning, the retained view shared its
memory and was writeable, and a write through it in the terminal derivative
callback moved the first gradient component from `0.6781500` to `0.7552125`.
Every default route answered `0.6781500`, and so did an out-of-band round
trip handing back the buffers the dump produced.

Both sides are now executable in `tests/test_coefficient_immutability.py`:
`test_a_metadata_exporter_answers_on_every_supported_route` checks the
supported routes, and `test_a_replacement_provider_displaces_the_gradient`
reproduces the excluded replacement-provider case against the closed-form
displacement.

The exclusion accompanies the withdrawal of storage traversal. For this
witness, following `ndarray.base` records the coefficient's owner as
carried, causing restoration to copy the coefficient and sever the
receiver's alias. Without that descent, the coefficient is recorded as not
carried; the arriving owning, frozen exporter is accepted even though the
receiver retains a writeable alias. The transition is therefore from a
correct gradient to a displaced gradient.

This observation establishes a limitation of the current provenance scheme.
Plain `AffineDynamics`, included as a control, still severs the alias and
returns the correct gradient with replacement storage. The broader
replacement-storage exclusion states the guarantee this implementation
declines to provide; it does not establish that every replacement provider
is unsafe or that supporting such providers is impossible.

The boundary is mechanical, and it is deliberately drawn wider than
mismatched or hostile buffers. **Supported**: loading in the same process
from the buffer providers that dump returned, or from wrappers over that
same storage. **Excluded**: any replacement provider — copied,
reconstituted, or received over a transport — *even when its bytes, sizes
and order are exactly right*. That excludes the ordinary cross-process
workflow, where the sender forwards the bytes and the receiver wraps them in
a fresh `bytearray`, `memoryview` or shared-memory block. Nothing is wrong
with that workflow; it simply hands the restored object storage this library
cannot see and the receiver may still hold writeably, which is the state the
measurement above describes. The contract declines to promise what it cannot
check, rather than accusing the caller of anything.

These are not separate defects to be closed one at a time. The rest of this
list is the same fact about Python, and chasing them costs more than it
buys: the only construction that would close the first is copying `F` and
`G`'s result at every step, which is precisely the aliasing this design
exists to avoid.

Per [C-1](#c-1) that is the correct place to stop. The invariant belongs to a
type system, not to a runtime check, and [C-13](#c-13) records that the port
should replace this machinery rather than reproduce it.

One over-refusal is accepted rather than cured, and is recorded here so that
reopening it is a decision. Binding a reader on the *instance* —
`self.F = partial(...)` — shadows the class's method, so the MRO widening
still sees the root's `F` and obliges `_M`. A problem that previously ran on
the general route is now refused, loudly. The cure would be to subtract on
instance shadows, and that is unsound in the quiet direction for the same
reason method identity is: the shadow may read the buffer and hand it back.
The class is a declaration about what a member does; an instance attribute
is not, and this clause does not infer liveness from anything it cannot see
into. Refusing audibly is the standing trade.

The general rule this leaves: **a check that raises must not be placed behind
checks that return, and must be gated on the precondition for its own
invariant rather than on the refusals it precedes.** The second half is not
optional decoration — without it the first half converts a required quiet
refusal into a raise. Ordering is part of the guarantee, not an implementation
detail, and it is the part no accuracy test can see — every test of the guard
itself passed throughout, because each presented a problem that was otherwise
eligible.

What this does not reach, per [C-1](#c-1): a caller who builds an instance
through `object.__new__` and populates `__dict__` directly runs neither
`__init__` nor `__setstate__`, and a caller who holds a reference taken before
the buffer was frozen. The refusal above narrows the gap to those, rather than
closing it.

#### Evidence and campaign provenance

The [failure history and injection table](docs/evidence/coefficient_immutability.md#historical-campaign)
retain the reported 95/95 detections at the **822-test baseline**, with all
original failure counts. This is historical evidence, not a campaign rerun
against the current test population. The later replacement-provider witness
and its supported-route controls are named in the envelope above. The evidence
records observations; this clause defines the requirements.

---

<a id="c-16"></a>
## C-16 Structure-aware dispatch is certified by route, not by accuracy — `APPROVED`

### C-16.1 Three axes decide the route, and they may not be merged

Solver selection reads three independent inputs:

| Axis | Decides | Known from |
|---|---|---|
| Tableau structure | the shape of the solve: none, sequential, or one coupled `(s·n)` system | `GLMethod`, statically |
| Vector-field structure | whether that solve is linear, and which curvature blocks vanish | the problem representation |
| Linear algebra | how the assembled matrix is represented and factored | the assembled matrix |

Merging two of them produces a rule that is correct for the cases that
motivated it and silently wrong elsewhere. Two consequences are normative:

- **`Linearity.LINEAR` may never gate a curvature skip.** That member is
  documented as "F independent of y, u", which permits
  `f(y,u,t) = M(t)y + b(u,t)` with `b` arbitrary in `u`, so `f_uu` need not
  vanish. The in-tree counterexample is
  `tests/problems.py::ConstantJacobianQuadraticControl`, which is `f = A y +
  B(u∘u)`: constant `F`, and `F_uu_action` returning `2 diag(Bᵀv)`. A rule
  reading `LINEAR` as permission to drop curvature would make
  `assemble_hessian_vector_product` return a Gauss-Newton approximation while
  its docstring promises an exact Hessian, violating [C-2](#c-2).
- **Eligibility for a specialised factorization is a property of the assembled
  matrix, never of the tableau.** `K = I − h(A ⊗ I)blockdiag(F_j)` is not
  symmetric for a general `F`, so no tableau classification can establish that
  a Cholesky factorization applies. A route that inferred it from the tableau
  would be deciding positive-definiteness by looking at the wrong object.

### C-16.2 Where a declaration cannot be verified, structure must be constructed

[C-15.2](#c-15) accepts a caller's declaration of a constant Jacobian because
it can *verify* it: comparing two assembled matrices costs `O(n²)` and guards
an `O(n³)` factorization, so the check is asymptotically cheaper than the work
it protects.

No comparably cheap exact check exists for "the dynamics have identically zero
curvature". Evaluating `f` at sample points and inferring affineness is a
**probe**, and generalising a probe from visited points to unvisited ones is
precedent [R-9](#r-9).

Therefore the zero-curvature route is **not** opened by a boolean. It is opened
only by a representation that owns `M`, `C`, `b` and computes `f`, `F`, `G`
from them (`adjungo.core.affine.AffineDynamics`), checked by
`affine_dynamics_verified`, which confirms by object identity that every method
carrying the guarantee is still the one that class defines. A subclass that
replaces any of them is refused.

[C-17](#c-17) extends this to `TimeVaryingAffineDynamics`, whose coefficients
are caller-supplied callables. The guarantee is still constructive there, by a
different argument: the callables are invoked with `t` alone, so they cannot
depend on `y` or `u` at any point. That class does **not** carry a constant
Jacobian, and C-17.2 records why the two facts had to be separated.

`ProblemStructure.jointly_affine` remains a public field, and a caller may set
it. It is **not sufficient**: both the field and the construction check are
required at the skip site. A declaration that fails the check causes more work,
never a wrong answer — the terms are computed and the callbacks are required.

The state-affine claim *is* separately verifiable and is separately verified;
see C-16.4.

### C-16.3 The dispatch is invisible to every accuracy test — `OBSERVED`

Both optimisations in this clause are undetectable by any numerical assertion:

| Optimisation | Why no accuracy test sees it |
|---|---|
| Direct affine stage solve instead of Newton | Both routes reach the same stage values — Newton to its C-5.1 threshold, the direct solve exactly — so gradients, Hessians and observed order rates agree either way |
| Skipping identically zero curvature | The skipped terms are additions of exact zero, so the result is bit-identical |

The certified quantities are therefore the **route** and the **count**, as in
[C-15.3](#c-15). Measured on `AffineDynamics` with `n = 3`, `ν = 2`, `N = 6`,
one gradient evaluation:

| Method | `s` | Implicit stages | Newton entries before | after | Factorizations before | after |
|---|---|---|---|---|---|---|
| `implicit_midpoint` | 1 | 1 | 6 | **0** | 1 | 1 |
| `implicit_trapezoid` | 2 | 1 | 6 | **0** | 1 | 1 |
| `sdirk2` | 2 | 2 | 12 | **0** | 1 | 1 |
| `sdirk3` | 3 | 3 | 18 | **0** | 1 | 1 |
| `gauss2` | 2 | 1 coupled | 6 | **0** | 1 | 1 |

Both "before" columns are measured with `ProblemStructure(SEMILINEAR,
jacobian_constant=True, ...)`, which keeps C-15 reuse enabled and only forces
the Newton route; otherwise the comparison would credit M7 with M6's work. The
Newton counts are `N` times the number of implicit stage *solves* per step,
which is one per implicit stage for the sequential families and one coupled
solve for `gauss2`.

The factorization columns are identical, and saying so matters: **M7 is not a
factorization milestone.** C-15 already reduced these to 1. What M7 removes per
stage is one Jacobian evaluation, one store lookup, and the iteration
machinery, and with it the C-5.3 stage-failure mode, which cannot arise for a
direct solve. The saving is real and modest for the `n × n` families; it is
larger for a coupled tableau, where assembling `K` costs `s²` blocks.

`needs_newton` was computed by `deduce_requirements` and consumed by nothing
for the whole life of the project before this clause, so every implicit method
entered Newton regardless of its value. That is [C-15.1](#c-15)'s lesson
recurring: correctness of a gate that nothing reads is not evidence that the
gate is correct.

### C-16.4 The affine route must verify what it assumes — `APPROVED`

`linear_stage_solve` factors `dR/dz` at its **starting point**, not at the
value it returns. That is admissible under [C-5.4](#c-5) only because for an
affine `f` the Jacobian is independent of the state, making the two matrices
identical element for element rather than merely close.

That argument depends on `f` actually being affine, so it is checked, not
assumed. After computing `z`, the residual `R(z)` is evaluated and tested
against the same scaled C-5.1 threshold Newton would have used. For an affine
residual it is zero up to the backward error of the linear solve; otherwise
`NonAffineStageEquation` is raised.

This is a check rather than a probe: it evaluates the residual at the point the
computation actually uses and generalises nothing. Its cost is one extra
evaluation of `f`, `O(n²)` for a dense problem, guarding an `O(n³)` solve. Its
error direction is toward refusal.

### C-16.5 Completeness must be established at the boundary — `APPROVED`

A count proves only that the counted route was not taken. Two facts in this
clause are therefore asserted by **tripwire**:

- `NewtonMixin.newton_solve` is replaced with a function that raises, and the
  affine gradient must still compute. This reaches any path to Newton, not
  only the one the counter instruments.
- The three contracted-Hessian callbacks are replaced **on the instance** with
  functions that raise. Affineness is verified from the class, so the problem
  stays verified while any actual call fails.

  [C-16.9](#c-16) tightened verification to reject instance shadowing, and
  exempts exactly these three so that this tripwire keeps working. The
  exemption and this clause stand or fall together: removing it costs 13
  failures here, and closing it deliberately requires amending this clause in
  the same commit.

This requirement exists because [C-15.6](#c-15)'s injection round nearly missed
a Newton call that bypassed the factorization store: the count was taken at the
store, the bypassing call never reached it, and the answers were unchanged
because it factored the same matrix.

### C-16.6 Skipping zero curvature must reproduce the computed result exactly — `APPROVED`

Enabling the skip must change **no** number, and the required agreement is
bit-identity.

As with [C-15.4](#c-15) this is not a pinned float and does not conflict with
[C-11](#c-11): nothing is recorded, and the claim is about two computations
performed in the same test. The skipped terms are `h·F_yu[Λ]ᵀδZ` and
`h·F_uu[Λ]δu` with the contracted Hessians identically zero, so the skip
removes `x + 0.0`, which is `x` in IEEE-754 for every `x` this code produces.
The argument holds on any platform and under any BLAS.

**Objective curvature is never skipped.** An affine plant under a quadratic
cost has a perfectly nonzero Hessian; only the contribution of the *dynamics*
second derivatives vanishes.

### C-16.7 Modified Newton is not prohibited; misusing it is

An approximate Newton **iteration** matrix — lagged, preconditioned, or with
blocks dropped — converges to the *same* stage values and costs convergence
rate, not accuracy. It is permitted.

What is prohibited is allowing that approximate matrix to become the
**derivative operator**. [C-5.4](#c-5) requires the adjoint to apply the
transpose of the true Jacobian at the converged stage value, so an
implementation that lags the Newton matrix must factor the true Jacobian once
at the solution and pay for it. The defect in [C-13](#c-13) item 4 is exactly
this confusion.

This clause corrects a broader statement in `docs/architecture.md`, which
refused modified Newton outright on the grounds that it trades exactness for
cost. That is true only of the derivative solve.

### C-16.8 Injection evidence — `OBSERVED`

Baseline 442 passed, 0 failed. Each defect introduced alone, under the
[R-11](#r-11) cache-clearing procedure:

| Defect | Failures |
|---|---|
| the dispatch ignores `needs_newton` and always iterates | 10 |
| the affine solve omits its residual verification | 1 |
| the affine solve applies the Newton step with the wrong sign | 50 |
| verification accepts any `AffineDynamics`, overridden or not | 6 |
| the Hessian's curvature skip believes the declaration alone | 4 |
| the second-order adjoint's skip believes the declaration alone | 3 |
| the skip also drops the *objective* curvature term | 12 |
| `LINEAR` is read as implying zero dynamics curvature | 1 |
| the affine coefficients alias the caller's arrays | 1 |
| `jointly_affine` no longer implies `state_affine` | 1 |
| `needs_newton` ignores `state_affine` | 1 |
| a stage-solver constructor default stops being conservative | 1 |
| the dispatch class attribute stops being conservative | 1 |
| the stage factorization is taken from a perturbed matrix | 50 |

**Three of these initially went undetected, and two of the three share a
cause: a population that cannot reach the defect.**

1. *The second-order adjoint's skip.* `adjoint_sensitivity` reads `F_yy` and
   `F_yu`; `assemble_hessian_vector_product` reads `F_yu` and `F_uu`. Every
   affine fixture in the suite had zero `F_yu`, so wrongly skipping in the
   first changed no number at all. The cure is
   `tests/problems.py::BilinearStateAffine`, `f = (A + u_0 V)y + C u`, which
   has nonzero `F_yu` and zero `F_uu` — the opposite pattern to
   `ConstantJacobianQuadraticControl`. **Two skips that consume different
   blocks need a fixture per block, not a fixture per route.**

2. *`LINEAR` read as implying zero curvature.* C-16.1's counterexample was
   pinned by a test, and the test still passed, because
   `ConstantJacobianQuadraticControl` declares no `linearity` attribute and is
   therefore deduced `NONLINEAR`. The counterexample never travelled the
   `LINEAR` branch where the dangerous rule lives. **Pinning a counterexample
   is not the same as routing it through the code that could misuse it.**

3. *The conservative constructor default.* The factory always passes an
   explicit `needs_newton`, so no end-to-end test reaches the default and no
   behavioural test could. It is now asserted directly, which is
   [C-15.1](#c-15) once more: a value that nothing reads is not evidenced by
   being correct.

### C-16.9 Verification is checked at the instance, not only at the class — `APPROVED`

M6 established affineness by comparing every guaranteed member of
`type(problem)` against the root class. That reads a fact about a *class* and
applies it to an *instance*, and the two can disagree: `problem.f = ...` writes
to the instance `__dict__` and disturbs no class-level comparison. Verification
must therefore check both, and additionally that attribute lookup itself is the
root's, since every comparison reaches the problem through it.

**What this is not.** The obvious reproduction is not this defect, and mistaking
one for the other is the trap here. Writing a nonlinear `f` onto the instance
while leaving `F` the affine Jacobian gives a gradient wrong by `6.395` — and a
plain callback problem supplying the same mismatched pair is wrong by *exactly*
the same amount, on the general route, with no affine class anywhere near it.
That is the universal obligation that `F` be the Jacobian of `f`. Verification
never bore on it, and a fix justified by that number would be justified by
nothing.

**What it is.** The cost needs a *consistent* shadow: an instance carrying a
complete, self-consistent nonlinear system. The gradient is then right, because
it reads only `f`, `F` and `G` and those agree. But the route is still chosen
from the stale class-level fact, so `jointly_affine` holds and [C-16.6](#c-16)
skips a curvature term that is now genuinely nonzero. **Only the Hessian
exposes it, and it is silent.**

Measured: `f = 0.4y + u + 5y²`, `F = 0.4 + 10y`, `G = 1`,
`F_yy_action = 10v`, the other two zero, written with `object.__setattr__`
onto an `AffineDynamics([[0.4]], [[1.0]])`; `y₀ = 0.8`, `t_span = (0, 0.5)`,
`N = 2`, explicit Euler, `J = y_N²/2`, `u = ((0.3), (0.2))`, `v = ((1), (0))`.
The M6 predicate returns `Hv₀ = 1.882` against an exact `2.793`, an absolute
error of `0.911`; `Hv₁` is unchanged, so the defect is localised to the
component the curvature reaches.

**Explicit methods specifically.** On an implicit method this shadow is already
refused loudly — the stage equation is nonlinear, one linear solve cannot
satisfy it, and [C-16.4](#c-16) raises. An explicit method solves no stage
equation and so has no residual to check. The certification population is built
on `explicit_euler` for that reason.

**The curvature callbacks are exempt from the instance check, and that
exemption is load-bearing.** [C-16.5](#c-16) requires a tripwire that replaces
`F_yy_action`, `F_yu_action` and `F_uu_action` on the instance with functions
that raise and then asserts a **Hessian-vector product** still computes. (Not a
gradient: the gradient never consults these callbacks on any route, so a
gradient-based tripwire would be vacuous.) That evidence is only obtainable
while shadowing them leaves the problem verified. Closing the exemption disarms
C-16.5 — injected below, 13 failures — and must not be done without amending
it.

**The exemption is sound on the route verification selects, and that qualifier
is the whole of it.** The deduced structure sets `jointly_affine=True`, under
which C-16.6 skips those three, and a replacement that is never called cannot
change an answer. Four entry points do not consult the deduced structure and
*do* call them, so a shadow would be read there:

- `GLMOptimizer` constructed with an explicit `ProblemStructure` carrying
  `jointly_affine=False`;
- `adjungo.validation.reference.reference_hessian`, which calls all three
  unconditionally — by design, since it is the independent oracle and must
  share no dispatch with the code it checks;
- `assemble_hessian_vector_product` called directly with `structure=None`;
- `adjoint_sensitivity` called directly with `structure=None`.

None is reached by `GLMOptimizer.hessian_vector_product` on a deduced
structure, which is the certified route. The rest is [C-1](#c-1).

**Attribute lookup is part of the check, and `__dict__` is part of attribute
lookup.** The instance check reads `problem.__dict__`, which is itself an
attribute access the class being checked can answer. A subclass defining
`__dict__` as a property returning `{}` shows the check an empty mapping while
the real storage holds every shadow, and reinstates this clause's own silent
Hessian error in full — `Hv₀ = 1.882` against `2.793`, through the default
public route. The first implementation of this clause had that hole and it was
found by review before release. Verification therefore reads bindings out of
the MRO directly (`affine.py::_defined_as`) instead of through `getattr`, and
compares `__getattribute__`, `__getattr__` and `__dict__` against the root.

These are conservative refusals, not free ones: a subclass using `__getattr__`
only for unrelated metadata is refused too, and pays the general route.

Per C-1 none of this is a claim of robustness against a hostile caller. Patching
the root class itself changes both sides of every comparison and still passes.
The error direction is unchanged and one-way: a problem that fails the check
takes the general route and gets more work, never a wrong answer.

#### Injection evidence — `OBSERVED`

Baseline 596 passed, 0 failed, under the [R-11](#r-11) procedure:

| Defect | Failures |
|---|---|
| the instance shadow check is dropped entirely | 13 |
| the skipped-member exemption is removed, disarming C-16.5 | 13 |
| the exemption is widened to `f`, `F` and `G` | 9 |
| attribute lookup is no longer compared against the root | 2 |
| `__dict__` alone is no longer compared, reopening the hidden-dict bypass | 1 |
| the instance check reads the class dict instead of the instance | 61 |
| member comparison goes back through `getattr` instead of the MRO | 1 |
| `_defined_as` stops walking the MRO past the class itself | 3 |

The seventh went **undetected on the first round**, and its cure is the
lesson. `_defined_as` and `getattr` differ for a guaranteed member only when a
*metaclass* answers for the class — the same shape of defect as the `__dict__`
property one level up. Nothing in the suite exercised it, so swapping the safer
lookup for the unsafe one changed no test at all. Cured by
`test_a_metaclass_answering_for_the_class_is_refused`. **A hardening whose
only justification is a channel no test travels is not evidenced by passing.**

Evidence: `tests/test_instance_shadowing.py`.

---

<a id="c-17"></a>
## C-17 Time-varying affine coefficients keep the skip and lose the reuse — `APPROVED`

[C-16.2](#c-16) states `jointly_affine` as `f = M(t)y + C(t)u + b(t)`, with the
coefficients free to vary in time, and then opens the zero-curvature route only
to `adjungo.core.affine.AffineDynamics`, whose coefficients are constant. This
clause governs the general case, implemented as
`adjungo.core.affine.TimeVaryingAffineDynamics`.

### C-17.1 The form is constructed; the coefficients' purity is a C-1 obligation

`AffineDynamics` can be read for affineness because it owns `M`, `C` and `b`.
The time-varying class does not own them: the caller supplies three functions
whose bodies this package never sees. [C-16.2](#c-16) refuses exactly that for
the general problem protocol, on the grounds that sampling an opaque `f` and
generalising is precedent [R-9](#r-9). The guarantee here is therefore in two
parts, and they must not be run together.

**Constructed.** The *form* of `f` is owned by the class and the caller cannot
change it: `f = M(·)y + C(·)u + b(·)`, with `y` and `u` entering linearly and
exactly once, protected by the same class-identity check as the constant case.
What the caller supplies is coefficient *values*. That is a much narrower
freedom than the one C-16.2 refuses, where `jointly_affine` is asserted of an
`f` about whose form nothing whatever is known. The coefficient accessors are
part of the verified guarantee because they are the only sites that invoke the
caller's callables; a subclass intercepting one could pass `y` and the form
would no longer be owned.

**Obliged.** Given that form, the curvature blocks vanish **iff** `M`, `C` and
`b` depend on `t` alone. The class confines the *interface* — no `y` and no `u`
is ever passed to a coefficient — but confining an interface does not confine a
Python closure. A callable may capture the very control array being
differentiated and read it, and then `f` is not affine in `u` while every check
in this package still passes.

It would be false to write that a function never given `y` cannot depend on
`y`. The true statement is that it cannot depend on `y` *through this
interface*, and that is weaker.

The damage is not confined to the skipped curvature, which is why this is
stated as an obligation on the representation rather than a caveat on C-16.6:
`G` returns `C(t)`, which omits `(∂M/∂u)y` outright, so the **first**
derivative is already wrong.

[C-1](#c-1) is what makes the remainder admissible: inputs are data-only, and a
callable that reads the optimizer's own control array is not a data-only input.
But the residue is not zero and is recorded here rather than elided, because
**the failure direction is silent.** Both numbers below are exact, and
`tests/test_time_varying_affine.py::test_a_coefficient_closing_over_the_control_defeats_the_skip`
recomputes them so this clause cannot rot: `y' = m(u)y + u` with
`m = −0.2 + 0.7u` smuggled through a closure, `y₀ = 0.8`,
`t_span = (0, 0.5)`, `N = 1`, explicit Euler, `J = y₁²/2`, at `u = 0.3`. The
package differentiates the form it was handed and returns `y₁h = 0.477`; the
objective it actually computes moves at `y₁h(0.7y₀ + 1) = 0.74412`, confirmed
by a central difference at `ε = 1e-6`. Nothing raises.

No cheap exact check exists on this side, and the reason is worth stating: a
coefficient that varies only when `u` varies is **constant throughout any
single gradient evaluation**, so no consistency comparison within a solve can
observe it. That is what separates this from [C-15.2](#c-15), where the two
matrices being compared both exist inside one solve.

Whether this residue should be closed by having the representation own
coefficient *samples* on the solve mesh, rather than invoke callables, is
[C-Q6](#open-questions).

### C-17.2 Zero curvature does not imply a constant Jacobian

[C-16.1](#c-16) requires the three dispatch axes be kept separate, and this is
the case that separates two of them. `F = M(t_i)` differs between stages at
distinct abscissae, so [C-15](#c-15) reuse must be **off** while the curvature
skip stays **on**.

The fact is carried by `coefficients_constant`, a member of each root class's
guarantee rather than a caller declaration, and is read by `deduce_structure` to
set `jacobian_constant`. Before this clause that field was asserted `True` for
anything verified affine, which was correct only because the constant case was
the only one that existed — [C-15.1](#c-15)'s lesson in its other direction: a
value can be correct for the whole population that can reach it and still state
something false.

Nothing here replaces [C-15.2](#c-15). A caller may still pass an explicit
`ProblemStructure(jacobian_constant=True)`, and the store then compares the
stored matrix with the requested one element for element and raises
`DeclaredStructureViolation` at the first disagreement. That refusal is asserted
for every implicit method in the certification population, because the deduction
being right is not evidence that the backstop behind it works.

### C-17.3 Caller obligations, and what guards them

`M`, `C` and `b` must be **deterministic functions of `t`**. A coefficient that
drifted between calls would make the adjoint apply the transpose of a matrix the
forward solve did not use, which is the [C-5.4](#c-5) failure mode.

This is not checked exhaustively — that would require retaining every earlier
call — but it is not unguarded. On the affine stage route, [C-16.4](#c-16)
evaluates the residual after the solve from freshly read coefficients and tests
it against the C-5.1 threshold, so a coefficient that changed between assembling
`K` and evaluating `R` raises `NonAffineStageEquation` rather than returning an
unconverged stage value.

That guard is real but partial, and the boundary is worth naming. The `F` and
`G` stored for the backward sweeps are read on a *further* call after the
residual check (`adjungo/solvers/dirk.py:98`, `adjungo/solvers/sdirk.py:159`),
so a coefficient that drifted on that call would seed the adjoint with a matrix
the forward solve never used, undetected. Determinism is an obligation, not a
verified fact.

Two further obligations are enforced rather than documented:

- **Shape.** Every coefficient value is checked against `(n, n)`, `(n, ν)` or
  `(n,)` *at the time it is used*. Validating one sample at construction and
  generalising to every other time would be a probe.
- **Ownership.** Every coefficient value is copied and made non-writeable. A
  callable is free to return one scratch buffer on every call and then write
  into it, which would change a matrix a factorization had already been taken
  from — [C-15.6](#c-15)'s aliasing defect arriving through the problem instead
  of through the store. The copy costs `O(n²)` and guards an `O(n³)` solve, the
  ratio [C-15.2](#c-15) uses to justify its own comparison. The constant-
  coefficient class establishes the same ownership once, at construction, and
  [C-15.7](#c-15) is what keeps it through a copy.

### C-17.4 The certified quantities — `OBSERVED`

Per [C-16.3](#c-16) the route is invisible to every accuracy assertion, so route
and count are certified separately from the numbers. A fourth quantity is
certified here that has no analogue in the constant case: the **times** at which
the coefficients are read.

Measured on the fixture in `tests/test_time_varying_affine.py`: `n = 3`,
`ν = 2`, `N = 6`, `t_span = (0.2, 0.95)`, `y₀ = (0.6, −0.3, 0.2)`, controls
`make_controls(N, s, 2, seed=5, scale=0.4)`, one gradient evaluation.
`M(t) = M₀ + sin(1.7t)M₁`, `C(t) = C₀ + tC₁`, `b(t) = b₀cos(0.9t)`.

| Method | `s` | Implicit stage solves per step | Newton entries | Factorizations, `M(t)` | Factorizations, constant `M` |
|---|---|---|---|---|---|
| `rk4` | 4 | 0 | *n/a* | 0 | 0 |
| `implicit_midpoint` | 1 | 1 | **0** | 6 | 1 |
| `implicit_trapezoid` | 2 | 1 | **0** | 6 | 1 |
| `sdirk2` | 2 | 2 | **0** | 12 | 1 |
| `sdirk3` | 3 | 3 | **0** | 18 | 1 |
| `gauss2` | 2 | 1 coupled, size `s·n` | **0** | 6 | 1 |

The two factorization columns are asserted **together**, so that neither can be
vacuous: a store that wrongly reused would collapse the fifth column to 1, and
one that stopped reusing would raise the sixth to the fifth.

The count is `steps × (implicit stage solves per step)` with no Newton
multiplier. [C-15.3](#c-15) declines to predict a count without reuse precisely
because Newton iteration counts are a property of Newton and the initial guess;
on this route there is no Newton, so the count is a property of the tableau
alone and is predictable after all. `factorizations_for_solve` still returns
`None` here, and the test states the rule from the tableau rather than
tabulating the numbers.

**Times.** The coefficients are read at `t_n + c_i h` and nowhere else, asserted
as a set over a whole gradient evaluation against the abscissae the method
supplies, at 1e-12 absolute. The basis: stage times are O(1) and formed in a few
flops, so a few ULPs of O(1) is the budget, and [C-11.3](#c-11) forbids pinning
the bit pattern. The tolerance is eight orders below the spacing being resolved,
`h·minᵢ≠ⱼ|cᵢ−cⱼ| ≈ 1.8e-2` for `sdirk2` on this mesh.

This assertion exists because **every constant-coefficient fixture in the suite
is blind to it.** A defect reading `M` at the step time instead of the stage
time produces a derivative wrong by an amount that shrinks with `h`, which
[C-3](#c-3) forbids reading as discretization error and which precedents
[R-1](#r-1) and [R-2](#r-2) record being misread that way twice. This is
[C-16.8](#c-16)'s "population that cannot reach the defect" anticipated rather
than discovered.

**Numbers.** Against the independent monolithic reference
([C-14.1](#c-14) level 1) over the six methods above, relative max-norm: worst
gradient error **1.11e-16**, worst Hessian-vector error **5.55e-17**. Neither
side stops at a convergence threshold — the package solves each stage equation
directly and the reference's Newton iteration is exact in one step on an affine
residual — so the floor is the backward error of the dense LU solves and the
implicit population needs no `CONDITIONING_ALLOWANCE`. The certified tolerance
is 1e-11, set five orders above the observation to absorb stage-matrix
conditioning rather than to certify these coefficients.

A closed-form anchor ([C-14.2](#c-14)) covers the stage-time claim at the level
no implementation participates in: two explicit Euler steps on the scalar
`y' = m(t)y + c(t)u` with `J = y₂²/2`, for which
`dJ/du₀ = y₂(1 + h·m₁)h·c₀` and `dJ/du₁ = y₂h·c₁`. Two steps rather than one,
and distinct coefficient values at the two step times, so that the anchor
distinguishes `m₀` from `m₁`; a single step would be satisfied by any code that
read the coefficients at one consistent time.

### C-17.5 Injection evidence — `OBSERVED`

Baseline 596 passed, 0 failed. Each defect introduced alone, under the
[R-11](#r-11) cache-clearing procedure:

| Defect | Failures |
|---|---|
| `deduce_structure` claims a constant Jacobian for varying coefficients | 52 |
| `coefficients_constant` claims `True` for time-varying coefficients | 53 |
| `F` reads `M` at a fixed time instead of the stage abscissa | 56 |
| the coefficient value aliases the caller's buffer | 1 |
| the coefficient shape is not checked where it is used | 2 |
| verification stops guarding the constancy claim and the accessors | 9 |
| verification resolves a multi-root class by MRO instead of refusing | 1 |
| `f` drops the time-varying forcing `b(t)` | 4 |

The first three are large because a false `jacobian_constant` is caught by
[C-15.2](#c-15) as a refusal across the whole implicit population, and a
wrong-time `M` moves every oracle comparison at once. None went undetected,
which is the first injection round in this repository for which that is true;
the reason is that the population was built from the defect list rather than the
other way round, following [C-16.8](#c-16).

The list is nonetheless not a proof of coverage, and the C-17.1 obligation is
the demonstration: it is a defect no injection into this package can produce,
because the defect lives in the caller's closure. It is held by a pair of
closed-form tests instead.

Evidence: `tests/test_time_varying_affine.py`.

---

## Open questions

| ID | Question | Blocks |
|---|---|---|
| C-Q3 | Is `ν = 0` (uncontrolled trajectory) a supported configuration? | Nothing currently |
| C-Q4 | Do multistep external vectors hold `y` history or `h·f` history under C-8.1? | Any future `r > 1` work |
| C-Q5 | Characteristic state scale `y_scale` for C-5.1 — is `max(‖y₀‖_∞, 1)` adequate for problems with large transients? | Tightening C-5 |
| C-Q6 | Should `TimeVaryingAffineDynamics` own coefficient *samples* on the solve mesh instead of invoking caller callables, closing the [C-17.1](#c-17) closure-capture residue by construction? Needs a mesh-aware constructor and an index rather than a float-`t` lookup, so it changes the `Problem` protocol. | Nothing currently; removes a silent-failure obligation |
| C-Q7 | How should a **vector**-valued function of the trajectory, `c(u) ∈ ℝᵐ`, expose its Jacobian? `GLMOptimizer` differentiates one scalar objective, so `m` components means `m` adjoint sweeps. Today that costs `m` optimizer instances, each repeating the same forward solve, and the resulting rows are assembled by the caller. The decomposition in [C-9](#c-9) is already per-objective; what is missing is a route that solves forward once and sweeps a set of terminal seeds. Deciding this also decides whether the adjoint seed enters the public API. | Any example or application posing path or terminal constraints to `scipy.optimize` — `NonlinearConstraint` wants the whole Jacobian |

---

## Precedents

<a id="r-1"></a>

**R-1 — A single-ε finite-difference check cannot certify an adjoint.**
Two defects in this repository (a DIRK adjoint that never applied the transposed
stage solve, and a second-order adjoint missing its terminal term) were
attributed to time-discretization error on the basis of one-sided checks at a
single ε. Both showed ε-independent plateaus under a sweep. Superseded by C-3.3.

<a id="r-2"></a>

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

<a id="r-5"></a>

**R-5 — The B0 stage-index defect, cured and measured.** — `OBSERVED`

**Extension, recorded later: the specification carried the same defect.**
`docs/runge_kutta_opt.tex` is named in the README as authority for the
Runge-Kutta case. It was never audited while the code was being repaired. It
stated the stage adjoint as
`μ_i = h Σ_j a_ji (F_j)ᵀ μ_j + h b_i (F_i)ᵀ λ`, placing the Jacobian at the
summation index — the identical error, in the document a reimplementation
would follow. Ten equations were affected: the stage adjoint, the block
matrix `M_k`, the reduced gradient, the stage adjoint sensitivity and its
forcing, and the `H_uu`, `H_uz` and `H_uμ` Hessian blocks.

The document also contradicted itself. It claimed `M_k = A_kᵀ`, but with
forward blocks `(A_k)_ij = δ_ij I − h a_ij F_j`, the transpose is
`(A_kᵀ)_ij = δ_ij I − h a_ji (F_i)ᵀ`, whose Jacobian index is the **row**.
The document's `M_k` used the column. Assembling both for `d = 2`, `n = 3`
with distinct per-stage Jacobians: `‖A_kᵀ − M_k^{doc}‖_∞ = 4.02e-01`, while
the row-indexed form matches `A_kᵀ` exactly.

**The lesson is about audit scope, not about the derivation.** Curing a
defect in code does not cure it in the document that specifies the code, and
the repository's own correctness work did not look there for six milestones.
Documents carrying normative equations need the same guards as code:
`tests/test_documentation.py` now fails if any `Σ_j a_ji` in that document
weights a `j`-indexed derivative tensor, and asserts numerically that the
row-indexed block matrix is the forward transpose while the column-indexed
one is not.
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

<a id="r-6"></a>

**R-6 — The second-order adjoint returned zero curvature for a terminal cost.**
— `OBSERVED`
Three independent defects compounded in the Hessian path:

1. `delta_Lambda[N]` was set to `0` with a comment asserting that the terminal
   term "is handled through the gradient assembly". It is not. The derivation
   requires `δλ^[N] = J_yy^terminal(y^[N]) δy^[N]`; zero is correct only for an
   **affine** terminal cost.
2. The running `J_yy δy^[n]` term was absent, marked `# For now omitted`.
3. `F_yy_action`, `F_yu_action` and `F_uu_action` were contracted against `δZ`
   and `δu` instead of against the weighted adjoint `Λ_k`. Their `v` argument
   sums over the *equation* index `ℓ`. Since `∂²f_ℓ/∂y_a∂y_b` is symmetric in
   `(a,b)` but carries no symmetry in `ℓ`, this sums the wrong index.
4. `assemble_hessian_vector_product` used `−h` on all three constraint terms
   where differentiating `g = ∂J/∂u + h Gᵀ Λ` gives `+h` on each.

Severity, measured on the C-14.1 item 2 closed-form anchor
(`y₁ = y₀ + h u`, `J = ½y₁²`, `h = 0.37`, exact `Hv = h²v = 0.1369 v`):

| | Before | After |
|---|---|---|
| Anchor `Hv` | **0.0** | 0.1369 |

For a terminal-cost-only problem the operator returned **identically zero**,
because defect 1 removed the only path by which terminal curvature could reach
the gradient. It was nevertheless symmetric — see R-3.

Against `reference_hessian` on `CoupledNonlinear` (`n = 3`, `ν = 2`), `N = 4`,
relative error of the fully materialized operator:

| Method | `s` | After |
|---|---|---|
| `explicit_euler` | 1 | 2.44e-15 |
| `heun` | 2 | 2.20e-16 |
| `rk4` | 4 | 5.55e-17 |

Defect 3 is a direct vindication of the `n ≠ ν` requirement in C-14: with
`n = 3` and `ν = 2` the wrong contraction **raises `IndexError`**. Every
historical test used `n = ν = 1`, where contracting the wrong index is
silently shape-conformable and merely returns a wrong number.

Consequent policy, now enforced: a missing second-derivative callback raises
(C-7) rather than dropping the term. Dropping it returns a Gauss-Newton-like
operator while the public method still promises the exact Hessian.

---

### R-7 A DIRK "solve" that linearises is not a solve — `APPROVED`

**Finding.** `DIRKStageSolver.solve_stages` did not iterate. It assumed
`f = F z + G u` with `F` and `G` frozen at the incoming right-hand side and
solved the resulting linear system once. `SDIRKStageSolver` had a second,
distinct version of the same error: it solved `(I − hγF) Z_i = rhs`, which
drops the inhomogeneous term `h γ f(0, u, t)` and is correct only for a linear
*homogeneous* `f = F y`. Its Jacobian was additionally evaluated at the previous
stage value `Z[i−1]`, not at `Z_i`.

Neither solver had a transposed stage solve in its adjoint: the factorisation of
`I − h a_ii F_i` was never applied with `trans=1`, so the adjoint stage equation
was solved with the wrong operator even when the forward value was right.

**Why it survived.** Every historical implicit test used an LTI problem, for
which freezing `F` and `G` *is* exact. The one nonlinear Crank-Nicolson test was
marked `@pytest.mark.skip(reason="DIRK solver needs Newton iteration for
nonlinear problems")` — the defect was known, recorded as a skip, and then
functioned as permission to leave the public method advertising a capability it
did not have.

**Measured**, `CoupledNonlinear`, `N = 6`, relative to the monolithic reference:

| method | s | forward, before | gradient, before | gradient, after |
|---|---|---|---|---|
| `implicit_midpoint` | 1 | 5.851e-03 | 2.857e-03 | 3.580e-15 |
| `implicit_trapezoid` (CN) | 2 | 2.881e-03 | 1.873e-03 | 2.044e-15 |
| `sdirk2` | 2 | 3.052e-03 | 1.310e-03 | 6.009e-15 |
| `sdirk3` | 3 | 5.068e-03 | 3.546e-03 | 6.377e-14 |

**Consequent policy.** A skip whose reason names a missing numerical capability
is an envelope statement. Either the method family is refused at construction
under [C-6](#c-6-envelope-enforcement), or the capability is implemented. It may
not remain silently available to callers.

---

### R-8 A factorisation must be taken at the converged iterate — `APPROVED`

**Finding.** `NewtonMixin` had zero inheritors, exited its `max_iter` loop
silently, and returned the last iterate. It also factored the Jacobian
**before** the final `z += dz`, so the LU it returned belonged to the
second-to-last iterate.

**Why it matters.** The adjoint reuses that factorisation as the operator whose
transpose it solves. A factorisation at the wrong point is the wrong operator,
and the resulting gradient is the exact derivative of *no* discrete map.

**Consequent policy, now enforced.**

1. Non-convergence raises `StageSolveError` naming the stage and time. An
   unconverged stage value is a silent sentinel under
   [C-7](#c-7-no-silent-sentinels).
2. The returned factorisation is computed at the converged iterate.
3. `STAGE_NEWTON_TOL` is a named module constant and is the published basis for
   [C-3.4](#c-34-certified-tolerance-implicit-methods--derived).

---

<a id="r-9"></a>

### R-9 A reuse probe must vary every argument the matrix depends on — `APPROVED`

**Finding.** The M2 SDIRK solver tried to exploit the constant diagonal `γ` by
probing whether `F` was constant and, on agreement, publishing **one shared LU
object** for every implicit stage of the step. The probe evaluated

```
problem.F(y_history[0], u_stages[i], t_n + c[i]*h)
```

varying `u` and `t` but **never the state**. A Jacobian depending on `y` alone
passed it. The adjoint then solved every stage with `(I − hγF_0)^T` instead of
`(I − hγF_i)^T`.

**Measured.** On `f = y² + u` (`F = 2y` state-dependent, `G = 1` constant),
`N = 4`, `y_0 = 0.5`, `u ≡ 0.3`, relative to the monolithic reference:

| method | s | gradient error |
|---|---|---|
| `sdirk2` | 2 | 4.244e-05 |
| `sdirk3` | 3 | 4.260e-05 |

Seven orders above the C-3.4 certified tolerance of `1e-9`.

**Why it survived.** `CoupledNonlinear`, the C-14 workhorse, has a Jacobian that
depends on `u`, so the probe correctly refused reuse there. **The entire
188-test suite — including every oracle test at every certified method — passed
with this defect present.** Only a problem whose Jacobian depends on the state
and nothing else exposes it. This is the same shape as
[R-5](#r-5): a wrong matrix in the adjoint stage solve, invisible to a test
population that does not vary the one argument that matters.

**Cure.** The substitution was removed. Each stage publishes the factorisation
taken at its own converged iterate. The substitution bought nothing in any case:
Newton has already factored each stage by the time it converges, so sharing the
object saved no work and only risked the adjoint operator.

**Consequent policy.**

1. A runtime probe may **never** be the authority for factorisation reuse. Reuse
   requires a declared `ProblemStructure`, because constancy is a property of the
   problem, not an observation at points the stages do not visit.
2. A sufficient-condition probe must vary **every** argument its conclusion
   quantifies over, and must err toward refusing. This one erred toward
   accepting.
3. Genuine factorisation reuse — skipping the LU *work* — is deferred to M6 and
   must be gated on a declaration, not a measurement.
4. The C-14 population is extended with a state-only-Jacobian problem
   (`tests/test_implicit_solvers.py::StateOnlyJacobian`). A population in which
   every Jacobian depends on `u` cannot certify a control-independent path.

**Not in scope of this precedent (M3).** `StepCache.coupled_factorization`, added
for the fully implicit route, is *not* a shared per-stage factorisation and does
not require a structure declaration. A dense `A` produces **one** `(s·n)` linear
system for the whole step, whose blocks already carry the distinct per-stage
`F_j`. Nothing is substituted for anything: there is no second matrix that the
adjoint could have used instead. The property R-9 protects — that the matrix a
stage is solved with is the matrix taken at that stage's own converged iterate —
holds here by construction, and is checked directly by
`tests/test_fully_implicit.py::test_cached_factorization_matches_a_differenced_jacobian`.
The two routes are kept from crossing by
`::test_coupled_cache_carries_a_factorization_not_per_stage_ones` and
`::test_triangular_methods_do_not_set_the_coupled_factorization`.

### R-10 An iteration count is not a correctness discriminator — `APPROVED`

**Context.** The planned acceptance criterion for the SciPy `hessp` adapter was
"`Newton-CG` / `trust-ncg` with `hessp` converges in strictly fewer iterations
than `L-BFGS-B`". The reasoning was that the minimum-energy oscillator has
linear dynamics and a quadratic cost, so the reduced objective `J(u)` is an
exactly quadratic form, and a Newton method given the exact Hessian should
therefore beat a method that must accumulate curvature.

**Observation.** On the shipped example at `N = 20` (80 control variables),
`trust-ncg` with the exact `hessp` took **6** iterations and `L-BFGS-B` took
**5**. The criterion failed against a Hessian that is exact to rounding level.

**Why the reasoning was wrong.** The two methods do not count the same thing.
`L-BFGS-B` terminates on relative reduction of `f`; `trust-ncg` terminates on
gradient norm. One `L-BFGS-B` iteration performs a line search with multiple
function and gradient evaluations, while a trust-region iteration performs a CG
subproblem solve. The comparison measures SciPy's stopping rules and work
accounting, not the quality of the curvature information. It would also be
sensitive to the conditioning of any particular test problem, so tightening or
loosening it would amount to tuning a correctness gate against an unrelated
quantity.

**The attributable quantity.** Final stationarity distinguishes the two
decisively and for the right reason. Both methods find the same minimiser and
agree on the objective to 9 significant figures. `trust-ncg` with the exact
Hessian leaves `‖∇J‖_∞ ≈ 1e-17`; `L-BFGS-B` stops at `≈ 1e-9`, limited by its
own curvature-approximation error. A Hessian that was merely *close* to exact
could not reach rounding-level stationarity, so this check has the discriminating
power the iteration count was assumed to have.

**Rule.** An acceptance criterion for a derivative claim must be a property of
the derivative: agreement with an independently assembled reference, or the
stationarity achieved by a method that consumes it. Iteration counts, wall-clock
times, and evaluation counts across *different* algorithms may be reported as
observations. They may not be assertions in a correctness gate.

This does not prohibit performance regression tests. It requires that they be
named and reported as performance, never as evidence for C-2.

**Evidence.** `tests/test_examples.py::test_exact_hessian_reaches_a_sharper_stationary_point`.

<a id="r-11"></a>

### R-11 A defect-injection check must invalidate the bytecode cache — `APPROVED`

**Why this clause exists.** Much of this repository's evidence has the form
"this test fails against the defective code and passes against the cure"
(C-14.2). That procedure requires that the interpreter actually execute the
code just written to disk. On CPython it sometimes does not.

**Mechanism.** A `.pyc` file records exactly two facts about its source: the
source modification time **as a whole number of seconds**, and the source size
in bytes. If a rewritten source file has the same size and the same integer
mtime as the one the cache was built from, the validator treats the cache as
current and the interpreter runs the **previous** bytecode. An inject-run-restore
cycle completes here in well under one second, and a defect injected by
exchanging two characters — `F[i]` for `F[j]`, `self._w[:, 0]` for
`self._w[:, 1]`, `sum` for a same-length alternative — preserves the size
exactly. Both conditions are met routinely.

Demonstrated directly: a module whose body was changed from `return 111` to
`return 222` within the same second continued to return `111` in a fresh
interpreter; the same edit made more than one second later returned the new
value. Header inspection confirmed the stored mtime and size were identical to
the source's.

**Observed here.** During U-M1.7 the nodal `pullback` was restored from a
same-size defect injection. The source on disk was correct, `inspect.getsource`
printed the correct body, and a hand-evaluation of the same expressions gave the
correct result — while the method itself returned the defective answer and 12
tests failed. The source was not at fault and no cure was warranted.

**The failure direction that matters.** The case above is loud: correct code
appears to fail, and the discrepancy forces investigation. The dangerous case is
the mirror image. Inject a defect, observe that the suite still passes, and
conclude that the tests are insensitive to it — when in fact the interpreter
never ran the defect. That conclusion would be used to justify writing a *weaker*
test, or to dismiss a real finding as unreachable. A stale cache can therefore
manufacture false evidence for exactly the claims C-14.2 exists to protect.

**Required procedure.** Any measurement of the form "N tests fail against the
injected defect" must, for each injection *and each restoration*:

1. remove every `__pycache__` directory in the tree, excluding `.venv`, and
2. run with `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`.

A count obtained without both steps is not evidence and may not be quoted in
this document. Sleeping between edits is **not** an acceptable substitute: it
depends on filesystem timestamp resolution and on the edit happening to cross a
second boundary.

**Standing consequence.** The B0 stage-index cure (R-5) was originally verified
with `git stash`, which rewrites files in place and is subject to the same
hazard. It was therefore re-verified under this procedure. Re-injecting the
historical defective form

```
mu[i] = h F_i^T (B[:, i] @ lambda_ext)
for j > i:  mu[i] += h A[j, i] F_j^T mu[j]
```

fails 22 tests, spanning the gradient oracle, the Hessian oracle, the
finite-difference sweeps, duality, the parametrization layer and the shipped
example. Every failure is on a method with `s > 1` (`heun`, `rk4`); no
`explicit_euler` case fails, because its coupling sum is empty and the defect is
unreachable there. That is the pattern the derivation predicts, so R-5 stands on
re-verified evidence.

**Evidence.** The clause is procedural and is enforced by review of how a count
was produced, not by a test. The B0 re-verification above is its first
application.

**Addendum.** Applying this clause exposed eight `.pyc` files tracked in the
repository (`adjungo/**/__pycache__/*.cpython-312.pyc`), committed before
`.gitignore` covered them; `.gitignore` does not untrack files already in the
index. They were compiled for CPython 3.12 while the pinned environment runs
3.14, so they were inert for current work, but a contributor on 3.12 could have
received cached bytecode for a source file they then edited — R-11's hazard made
durable and distributable. They have been removed from the index.

**Second addendum — the harness must not silence the result it reads.**
`pyproject.toml` already sets `addopts = "-q --strict-markers"`. Passing `-q`
again on the command line therefore yields `-q -q`, and at that verbosity pytest
prints the progress line and **no summary line at all**: no `N passed`, no
`N failed`. An injection harness that scrapes the output for `failed` then reads
zero for every injection and reports that the suite is insensitive to all of
them.

This was observed here while certifying [C-17](#c-17): eight injections, all
genuinely detected, all reported as `0 failing`, with a baseline that looked
clean because it was equally unparseable. The failure direction is the dangerous
one this clause is about — quiet, plausible, and arguing for weaker tests.

Two requirements follow. Use the command in `AGENTS.md` **as written**, without
adding `-q`. And have the harness cross-check the parsed count against the
process exit status, refusing to report a number when a nonzero exit yields no
parsed failures or a zero exit yields some; the discrepancy is what caught this.

### R-12 An unrun check is not a check — `APPROVED`

**Why this clause exists.** [C-14.2](#c-14) treats "fails against the defect,
passes against the cure" as evidence. That form assumes the check runs. A
check that exists, is correct, and is never executed contributes nothing, and
is more dangerous than a missing one because the repository looks guarded.

**Mechanism.** Two instances were found together, on the same push.

`requires-python` declared `>=3.10` while `tests/test_coefficient_immutability.py`
imported `typing.Self` (3.11) and referenced `copy.replace` (3.13) at module
scope. Both are collection-time failures, not skips: on 3.10 the import
raises, on 3.12 the attribute lookup does. The CI matrix named `3.10` and
`3.12` and would have caught either on the day it was written. It had not run,
because twenty-one commits had accumulated locally unpushed. The declaration
and the evidence for it were both present; only the execution was missing.

The second instance is the same shape one level down. The workflow step
guarding bare-`pytest` collection compared `tail -1` of two collection runs.
`pyproject.toml` already supplies `-q`, so the step's own `-q` raised the
quiet level past the summary line and both sides reduced to a blank line: the
step compared `""` with `""` for its entire life. Verified insensitive at 833
collected against 819. The step ran on every push and checked nothing, which
is why the first instance survived even the pushes that did occur.

**What follows.** A guard's first evidence is that it fails when the property
it names is false; [C-14.2](#c-14) already requires this, and the repaired
step was falsified before being kept. Beyond that, local verification does not
substitute for the matrix. A single development interpreter cannot observe a
floor violation, and this one could not: CPython 3.14 runs every construct
above without complaint. Push before the local history grows past what CI has
seen, or the matrix is a description of intent rather than a measurement.
