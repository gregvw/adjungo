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
                       Jacobian constancy, control dependence, and the two
                       affineness axes state_affine and jointly_affine.
    affine.py          AffineDynamics: y' = M y + C u + b, and
                       TimeVaryingAffineDynamics: y' = M(t)y + C(t)u + b(t),
                       as representations rather than claims. Affineness is
                       established by construction, so it needs no verification
                       the way a declaration does. The two differ on
                       coefficients_constant, which decides C-15 reuse and is
                       part of the guarantee rather than a declaration.
    objective.py       Objective protocol: terminal and running cost, their
                       first derivatives, and their second-derivative actions.
    method.py          GLMethod: tableaux A, U, B, V, abscissae c; StageType
                       and PropType classification of a tableau, decided by
                       exact zeros and exact equality. An optional declared
                       stage type is checked against that structure at
                       construction (C-8.3).
    requirements.py    deduce_requirements(method, structure, state_dim):
                       decides, from the declaration alone, what the solver
                       must be able to do.
    plan.py            DiscretizationPlan: the time nodes and the method used
                       on each step (C-18.1). It is the discrete problem's
                       identity, so it is frozen, it is carried on the
                       trajectory, and the backward sweep reads it by step
                       index rather than re-deriving it. DiscretizationPlan
                       .uniform reproduces the single-method, equal-step case
                       and gives every step exactly the same h, which C-15
                       reuse depends on.

  algebra/         how linear systems are represented and solved
    protocols.py       LinearAlgebraBackend protocol.
    dense.py           The one implementation: NumPy/SciPy dense LU.
    operators.py       A LinearOperator wrapper. Exported from nowhere and
                       called from nowhere: it anticipates the matrix-free
                       route listed as not built below. Present as a sketch,
                       not as a working seam.

  solvers/         one step's stage equations
    base.py            StageSolver protocol and the per-step cache.
    explicit.py        A strictly lower triangular: forward substitution.
    dirk.py            A lower triangular: one n-by-n factorization per stage.
    sdirk.py           A lower triangular with constant diagonal: one n-by-n
                       factorization per step, shared by all stages.
    implicit.py        A dense: one (s*n)-by-(s*n) coupled Newton solve.
    newton.py          Newton iteration with explicit convergence criteria,
                       and StageDispatchMixin, which chooses between it and
                       the affine route from requirements.needs_newton.
    linear_stage.py    linear_stage_solve: one exact solve for a stage
                       equation affine in its unknown, verified by testing
                       the residual at the value it returns.
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
| Partitioned methods (PRK, Nystrom) | Not built, and not certified. Conventions, domain, the paired symplectic condition and the conjugate exchange are `APPROVED` in `NUMERICS.md` C-8.4; certification cases in C-14.5, also `APPROVED`. Both rule how such a family must be expressed and certified; neither delivers one. No representation exists, so there is nothing to hand to `GLMOptimizer`; C-6.2 would refuse one if there were. Sequenced after the discretization plan, which is now delivered, so that stage storage was refactored once rather than twice — not because a partitioned step needs packed storage; both target methods fit the present rectangular arrays. |
| Mesh adaptation, and control parametrization on a varying stage count | Not built. **Prescribed** non-uniform steps and per-step method selection *are* built, as `core/plan.py`; `NUMERICS.md` C-18's first increment covers `r = 1` families at fixed state dimension and is acceptance-tested in `tests/test_discretization_plan.py`. What remains unbuilt is the outer **adaptation policy** that would construct a plan from a trajectory, and multistep startup, which still waits on C-Q4. Two present limits, both loud under C-7: the C-10 adapters bake one `method.s`, so a plan whose stage count varies is refused a control parametrization; and `Objective.evaluate` is specified on the rectangular `(N, s, ν)` layout, which such a plan does not have, so it has gradients and Hessian-vector products but no scalar objective value and therefore no line search. |
| Matrix-free linear algebra | Not built. `algebra/protocols.py` is the seam it would enter through; `algebra/operators.py` holds an unused `LinearOperator` sketch. |
| Sparse linear algebra | Not built. Same seam. |
| Reuse for a *varying* Jacobian (modified Newton, lagged Jacobian) | Not built. The refusal is narrower than it first appears and is stated precisely below. |
| Sparse or specialised factorization of `K` | Not built. Eligibility is a property of the **assembled matrix**, never of the tableau: `K = I - h(A (x) I)blockdiag(F_j)` is not symmetric for a general `F`, so no tableau classification can establish that Cholesky applies. |
| Nonlinear-in-control affine coefficients | Not built. `f = M(t, u)y + b(t, u)` is `state_affine` but not `jointly_affine`, so it keeps the direct stage solve and loses the curvature skip. `tests/problems.py::ConstantJacobianQuadraticControl` is the in-tree instance; no representation class owns the case. |
| Checkpointing | Not built. Storage is `O(N s n)`, everything retained. |
| Generic scalar type | A C++ concern, not a Python one. See below. |

---

## For a C++ reimplementation

This Python package is a pilot study. `NUMERICS.md` C-13 is the normative list
of what must carry forward; it currently has twelve items. The thirteen below are
that list restated for an implementer, together with the implementation
lessons that produced it — a longer list, not a competing one. Where the two
differ, C-13 governs.

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

11. **Keep tableau structure, vector-field structure, and linear algebra as
    three separate inputs.** The tableau fixes the *shape* of the solve and is
    known statically. The vector field decides whether that solve is *linear*.
    The linear algebra decides how the assembled matrix is represented and
    factored. Merging any two of them produces a rule that is right for the
    cases that motivated it and silently wrong elsewhere. Two concrete
    instances: `Linearity.LINEAR` classifies the state Jacobian and says
    nothing about `f_uu`, so it must never gate a curvature skip; and Cholesky
    eligibility is a property of the assembled `K`, which is not symmetric for
    a general `F`, so no tableau classification can establish it.

12. **Establish structure by construction where verification is not
    available.** Item 10 accepts a declaration because it can check one:
    comparing two assembled matrices is `O(n^2)` against an `O(n^3)`
    factorization. No comparably cheap check exists for "the dynamics have
    zero curvature", and sampling `f` to infer it is the R-9 mistake. So the
    zero-curvature route is opened only by an object that *owns* `M`, `C`, `b`
    and computes `f`, `F`, `G` from them, never by a caller-set flag. A C++
    reimplementation can enforce this far better than Python can, by making
    the coefficient-owning type `final`.

    Ownership has to be established of the *instance*, not just the type.
    Comparing the members of `type(problem)` against the root class leaves
    `problem.f = ...` verified, and NUMERICS.md C-16.9 records what that was
    worth: a consistently shadowed nonlinear system keeps the gradient exact,
    so only the Hessian moves, by `0.911` on the fixture there, in silence.
    The check now also compares the instance `__dict__` and the attribute
    lookup hooks. A `final` type in C++ removes the class half of this
    outright; the instance half has no C++ analogue, since a member function
    cannot be rebound per object.

    Ownership also has to survive *copying*, and has to be true of the
    instance rather than assumed from the constructor. The coefficient buffers
    are frozen at construction so that `F` can hand back `self._M` itself and
    the tape may alias it; `copy.deepcopy` returned them writeable, because
    NumPy's own `ndarray.__deepcopy__` does, and a subclass calling
    `self._M = self._M.copy()` after `super().__init__` unfroze them too.
    NUMERICS.md C-15.7 records the cure — re-freeze in `__setstate__`, and
    check the buffers at the verification boundary — together with the reason
    that check *raises* where every other refusal in that function returns
    `False`: refusing merely selects the general route, which retains what `F`
    returns in the same way, so for this one defect there is no route that
    gives a better answer. That asymmetry also fixes *where* the check runs:
    a check that raises may not sit behind checks that return, so it is
    answered before the eligibility comparisons rather than after them.
    Ordered last, it was reachable only by problems that passed all three
    eligibility comparisons — so the subclass that had unfrozen its buffers by
    overriding a copy hook was refused quietly for that override and never
    reached it.

    Hoisting a raise, though, widens what it refuses, and immutability is a
    *conditional* invariant: it matters only because `AffineDynamics.F` and
    `G` return `self._M` and `self._C` by identity and `f` reads all three to
    build its result. An instance holding none of that storage — one that
    never ran the root initialiser — has nothing the tape could alias, and
    gating on `isinstance` alone made two such valid general-route problems
    raise where C-16.2 requires a quiet refusal. So the check carries its own
    precondition: class membership *and* a flag `AffineDynamics.__init__`
    records once it has frozen all three buffers.

    Not, as first written, whether `f`, `F` and `G` still resolve to the
    root's own definitions. An override may delegate — `return super().F(...)`
    is three different function objects reading every buffer — so method
    identity cannot witness what a method reads, and that precondition let a
    displaced gradient through in silence. Nor whether the buffers are
    present: asking evaluates descriptors on classes that never had them, and
    misreads a subclass reusing one of those names as a root instance taken
    apart. So `__init__` records that it ran.

    That record is necessary and is not sufficient, which took a further
    round to see. "The initialiser did not run, so nothing is aliased" holds
    only of a subclass that also replaces the members handing a buffer back.
    One that assigns `self._M`, `self._C` and `self._b` itself and inherits
    `f`, `F` and `G` — what a subclass looks like when its author builds the
    coefficients from a file or a mesh — runs the root's `return self._M`
    over writeable storage, verified as affine, and displaced the gradient by
    the full amount C-15.7 measures. So the method-identity question returns
    beside the flag rather than in place of it, where it is sound because it
    can only *add* instances to the checked set, and where it also says which
    buffers are live: `F` owes `_M`, `G` owes `_C`, `f` owes all three.

    That still left one shape, and it took another round to see: a subclass
    that declines the initialiser *and* delegates all three readers through
    `super()`. No marker, no reader of its own, an empty live set, and
    `super().F(...)` handing the writeable `_M` to the tape — the full
    displacement, silently. Each half of that shape is ordinary alone, which
    is the whole difficulty. So a second widening runs beside the first: any
    root coefficient name that resolves *at all* is live, which is the same
    question `__setstate__` already asks of a restored object. Resolution,
    not type — obliging only names holding an `ndarray` left the same
    delegating shape holding three mutable `csr_array` coefficients outside
    both widenings, returning the same displacement, quiet. Whether the
    storage is usable is decided afterwards by a refusal that raises;
    deciding it inside the widening is skipping under another name. What
    identity adds beside it is a name that does not resolve at all: a
    subclass inheriting `F` while holding no `_M` is refused by name at the
    route decision instead of raising `AttributeError` inside the first step.

    The cost is over-refusal, and it is paid by one shape: a subclass that
    replaces all three readers and reuses one of the root's buffer names for
    writeable storage of its own. A loud refusal of a working problem is the
    right side of that trade — the refusal names the attribute that collided,
    renaming it lifts it, and restoration already charged this shape the same
    price on every copy. A rule that binds on copying but not on checking is
    not one rule.

    The buffers are read by plain attribute access, because that is how `F`
    reads them. Reading the instance dictionary instead — even through the
    root's own `__dict__` descriptor, chosen to bypass a subclass's attribute
    machinery — misses a `__slots__` buffer entirely and, worse, reads storage
    that a `_M` property may have displaced: a frozen decoy in the dictionary
    while `F` returns the writeable array. A more suspicious read is not a
    more faithful one. The *flag* is read the opposite way, through that same
    descriptor, because unlike a coefficient it is not something a subclass is
    entitled to answer for — and it is *written* through that descriptor too,
    since a guarded read establishes nothing if a subclass can divert the
    write elsewhere and leave the check concluding the root never ran.

    `__setstate__` resolves storage by the same rule — literally the same
    function, after two spellings of one question drifted apart: the check
    absorbed any exception a coefficient descriptor raised while restoration
    absorbed only `AttributeError`, so a subclass the optimizer accepts could
    not be copied at all, and in the other direction restoration obliged only
    names holding an `ndarray`, so a delegating subclass with `csr_array`
    coefficients the guard refuses round-tripped quietly into a writeable
    copy. Restoration freezes what resolves; the guard obliges what must, the
    difference being names an inherited reader obliges that resolve to
    nothing, which restoration cannot conjure and the route decision refuses
    by name. Every read goes through the one function, the guard's own last
    one included — it restated the rule as a bare `getattr` and let a
    refusing descriptor's exception out in place of that refusal, which is
    how a shared rule stops being shared. So a `__slots__`
    subclass that verifies can also be copied and pickled. It freezes in a
    second pass over re-read values, because a property setter may store a
    copy of its argument or rewrite another coefficient, leaving the freeze on
    an object already discarded. It also freezes the replacement *before* the
    setter sees it, which is a separate claim: that second pass reaches only
    what the object finally resolves to, and `writeable` does not propagate to
    views already made, so a setter deriving a handle from a writeable
    replacement kept a writeable alias of the coefficient's memory behind a
    frozen owner — and because no ordering reaches a setter that allocates
    storage of its own, a coefficient must additionally *be* the array the
    root froze, checked by identity after assignment in the constructor and
    in restoration alike, after all three names rather than after each,
    because one coefficient's setter may rewrite another — and for every
    name, not only the ones restoration replaced, since a coefficient that
    arrives already frozen and owning is skipped but is still one another
    setter can rebuild, and for names that were *absent* when restoration
    began, since an unrooted instance can gain one from a setter that rebuilds
    a coefficient the state did not carry. A setter may relocate what it is
    handed; it may not substitute, and the refusal separates whose setter did
    it — a setter that faithfully stored its own argument can still be holding
    a substitute a later setter left it. Nor is any coefficient left writeable
    while a setter can still run: `deepcopy` and protocol 4 rebuild one owning
    but writeable, which used to be skipped on the reasoning that a rooted
    instance's arrays were frozen already, and it then stayed writeable long
    enough for another coefficient's setter to take a view of it, so anything
    not already at the postcondition is now replaced by a frozen copy first,
    and every copy is taken before the *first* setter runs rather than one
    name at a time: interleaving them left the later names holding those
    writeable arrays while an earlier setter ran, and one writing into a
    not-yet-copied `_C` had the change carried into the replacement, so a
    `deepcopy` of a problem built at `C = 2` verified, differentiated exactly,
    and did so for `C = 3`.
    So is anything *at* it, when the subclass produced the state — satisfying
    the postcondition is not the same as having been put there by the
    constructor, since a `__getstate__` can allocate a coefficient, keep a
    writeable view of it, freeze it and return both, and identity then
    confirms itself because the fabricated array is the only one restoration
    ever saw. So a coefficient at the postcondition is left where it is only
    when the root can vouch for the array itself: `__getstate__` records,
    under a private key in the state it returns, the array each coefficient
    resolved to — or `None` for a name the state does not carry — and
    restoration lifts that record out first and exempts a buffer only when the
    array that arrived *is* the one recorded. `deepcopy` and `pickle` memoise
    by identity, so that correspondence is carried across the honest routes
    rather than asserted; a substitution breaks it, including one made by
    taking the root's own state and replacing an entry.

    Identity says where an array came from, not what was done to it on the
    way, so the exemption also requires a state that was handed on unchanged.
    A copy arrives writeable and is frozen only afterwards, so anything else
    copied in the same traversal may keep a view of it — and memoisation
    carries the record across for the aliased copy too, so identity agrees.
    A plain `AffineDynamics` carrying an ordinary attribute that rebuilds a
    flattened view of `M` in its `__deepcopy__` verified after `copy.deepcopy`
    with a writeable alias of its frozen coefficient. The record therefore
    carries a marker that `copy.copy` passes through and every rebuilding
    route replaces; a rebuilt state has its coefficients replaced, severing
    any such view, which costs nothing because that state is the traversal's
    own. A coefficient the state does not carry keeps its exemption either
    way, having never been copied — where "does not carry" is reachability
    through the whole produced state, dictionary and slots, through
    containers and through objects, with a view's owner reachable through the
    view. Membership in the instance dictionary is a weaker statement, and a
    coefficient answered from inside an attribute object or held in a slot
    fell through it and came back aliased.

    An object that redefines its own copying makes the walk inconclusive, and
    an inconclusive walk treats every coefficient as reachable. The walk
    predicts a traversal from the graph as it stands; a hook may build
    something that graph does not contain. An *empty* companion whose
    `__deepcopy__` allocates the coefficient from storage outside the state
    and takes a writeable view of it before freezing it is the witness, and
    overrides nothing on the problem class for verification to object to.
    Scope had to match the traversal's, and eight hiding places each defeated
    it alone: a container subclass's own attributes (not just its contents);
    slots read from member descriptors rather than by replaying `__slots__`,
    whose legal bare-string form iterates one character at a time; a
    class-level `__getattribute__` answering a request for `__deepcopy__`
    with a hook written down nowhere; atomicity decided by `isinstance`, so
    that an `int` subclass carrying an instance dictionary was skipped; an
    inexact subclass of any modelled type *that declares anything of its
    own*, which needs no listed hook at all, because NumPy calls an
    `ndarray` subclass's `__array_finalize__` while building the copy and a
    `dict` subclass is rebuilt blank and repopulated through its own
    `__setitem__` — the question being whether it declares any name, not
    which, since a subclass declaring none is rebuilt exactly as its base; `__new__`, which every rebuilding route
    calls before it puts any state in; an object-dtype array, which is an
    exact `ndarray` holding Python objects NumPy copies one by one; and a
    mapping's own `keys` and `values`, read instead through `dict`'s, though
    the subclass rule now subsumes that one. Class namespaces are read
    through `type`'s own descriptor so a metaclass cannot answer for them.

    None of that reaches a hook that never calls restoration at all, and
    ineligibility for the affine route is not safety: `F` returns `_M` by
    identity and the general route reads it the same way. A clone assembled
    by a plausible hand-written `__deepcopy__` — copy the state, keep a
    handle, freeze the buffers — is owning and frozen and holds a writeable
    alias, because coefficients arrive writeable and the flag is not part of
    what a copy carries. Its final state is indistinguishable from an honest
    one, so validity stops inspecting the arrays and asks instead for
    positive evidence: the root records the exact arrays it established at
    the end of its own initialisation and restoration, and a coefficient must
    be one of them. The register is module-level so no state can carry or
    forge it, checked last so it cannot mask a precise diagnosis, and applied
    only to rooted instances so a subclass that reads none of the root's
    buffers is not refused for never having used it. It records the *array*
    rather than the problem holding it, since a frozen owning array cannot
    acquire a writeable alias afterwards and is therefore as sound to read
    from a second object as from the first.

    One trap is worth naming on its own, because it makes a class namespace
    something other than a property of the class: `copyreg` caches
    `__slotnames__` into a class the first time an instance is copied or
    pickled. A rule reading a namespace consequently answered differently
    before and after the first duplication, accepting a problem and then
    refusing that same problem's next round trip.

    Storage is not reconstruction, and the walk no longer follows what an
    array shares memory with. Following `ndarray.base` and `memoryview.obj`
    rested on the theory that a view travels with whatever owns its memory;
    a test now pins the opposite, since default `deepcopy` and the in-band
    pickle protocols rebuild each array independently and serialise no
    `.base` relationship, and a shallow copy hands back the same objects. A
    faithful out-of-band load borrows the buffers it is handed and so can
    preserve sharing, but it still rebuilds no exporter and calls no hooks
    of one, which is the part the walk depends on. The claim is about
    sharing rather than ownership: in band, protocol 5 returns a contiguous
    array as a non-owning view of an incidental one it just made. The
    chain cost answers both ways: it recorded owners no copy would rebuild,
    and it judged what it found there by hooks no copy would call — an
    ordinary `np.frombuffer(array("d", ...))` carried as metadata reaches an
    `array.array` and cost a problem with an unassignable coefficient its
    answer. The earlier protocol-5 symptom, where the *second* round trip of
    a problem the first had answered for was refused, was the same mistake
    from the other side; modelling `memoryview` cured the symptom and is
    withdrawn with the chain, since `complete` decides nothing on the one
    route that can carry a memoryview at all.

    Reading namespaces is not the whole of the answer either. A reducer
    registered through `copyreg.pickle` lives in a module-level table that
    `deepcopy` and `pickle` consult by exact type, is in no namespace, and
    reaches the modelled types precisely because their own hooks are
    skipped — one registered for `memoryview` gave a verified deep copy
    holding a writeable alias of its frozen coefficient and the full
    C-15.7 displacement. An exact type in `copyreg.dispatch_table` is
    inconclusive before any namespace is read — and before atomicity is
    read, since `deepcopy` answers a `complex` from its own dispatch table
    while every pickle protocol consults the registration, so a walk asking
    about atomicity first read the deep copy correctly and the round trip
    not at all. `copyreg` is not empty at import, so what keeps an ordinary
    complex number answerable is not its atomicity but that the walk models
    that one registration, comparing it with `copyreg.pickle_complex` by
    identity. The registration is asked at one site, of every object the
    walk reaches; a second copy inside the namespace scan was removed once
    the campaign reported it as a defect no test could detect. A
    per-`Pickler` dispatch table, `reducer_override`, replacement
    out-of-band buffers supplied to the loader — any provider backed by
    storage other than the storage the dump returned, even a byte-for-byte
    faithful copy received over a transport, since loading from the original
    providers, or from wrappers over that same storage, is supported and
    asserted — and persistent-id hooks stay outside the envelope: all are
    chosen by whoever runs the pickler and none is visible from the object.

    Discriminating a constructor slot from a written `__new__` by type was
    tried first and is refuted: `__new__` can be assigned after the class
    exists, and a bound built-in assigned there is indistinguishable from
    the slot every C type carries while handing back an object prepared in
    advance — a companion holding a writeable view of the coefficient,
    verified after `copy.deepcopy`, displacing the gradient by the full
    C-15.7 amount. How a hook is spelled is not what it does.

    That was invisible because a route does not hand back what it was
    given, and every duplication test built its case from a fresh source:
    protocol 5's own output is the only input carrying a `memoryview`. The
    evidence now chains each route three deep from its own output and mixes
    kinds.

    Inferring the producer from the class was tried first and failed in both
    directions. Asking which copy-protocol hooks the subclass overrides was
    too narrow — `__copy__`, `__deepcopy__` and `__replace__` replace what the
    route *does*, not the restoration it performs, and each may call
    `__setstate__` with a state it chose, as may an overridden `__setstate__`
    delegating to `super()`, or a class overriding nothing and answering the
    `getattr` of `__reduce_ex__` through `__getattribute__`. It was also too
    wide, which is worse: a class defining only `__replace__`, which
    `copy.copy` never calls, lost the exemption, so its setter ran against
    state shared with the source and rewrote the source's own `_C` — copying
    the problem moved the gradient of the object being copied. Which hooks a
    route consults is a property of the route, and a static list sees only the
    class. The exemption does not turn on whether the attribute is settable
    either — excusing an `AttributeError` reopens the same hole behind a
    read-only property — and leaving a coefficient alone is the reason an
    expectation is still recorded for every name rather than only the replaced
    ones.
    This closes
    substitution, not sabotage: a setter that unfreezes its argument before
    deriving a handle, and an unrooted subclass aliasing storage it built
    itself, stay in the excluded class C-15.7 names. And it
    decides *which* buffers to freeze by
    asking the restored object, not the state it arrived in: three rules that
    read the state — the marker, all three names, any name — were each
    defeated by a subclass shaping what it serialised, the last by delivering
    `_M` under a different name through a property setter. So any coefficient
    name that resolves to an array on the finished instance is frozen — and
    asking the object is not a passive read, so a coefficient rebuilt lazily
    on first access is built *there*, copied and frozen, rather than appearing
    writeable later; a name that never resolves at all is refused by name at
    the route decision. And without the marker a still-writeable buffer is copied
    first, since `copy.copy` shares the array and freezing in place reached
    back into the original — while one already owning and already frozen is
    left where it is, the freeze being a no-op with nothing to reach back to,
    and replacing it having broken every round trip of a subclass answering
    for a coefficient through a read-only property; the copy is made with `np.array(buffer, copy=True, subok=True)`
    rather than the buffer's own `copy`, which an `ndarray` subclass may
    define and one defined as returning `self` — while `subok=False` went too
    far the other way, silently turning a `MaskedArray` into a base array and
    so turning a buffer the check must refuse into one that passes.
    Restoration preserves faithfully; the guard refuses loudly.

    Two further readings had to be taken through `type`. `cls.__mro__` and
    `klass.__dict__` are attribute accesses on a class object, so a metaclass
    answers them, and one reporting a forged MRO or a forged class dictionary
    turned the liveness widening off and let the full displacement through.
    They are read through `type.__dict__["__mro__"]` and
    `type.__dict__["__dict__"]`. A metaclass is to a class what a `__dict__`
    property is to an instance; wherever a check reads state through a path
    chosen to be unforgeable, it must resolve that path on the type itself.

    Finally, `writeable = False` seals the `ndarray` it is set on and nothing
    that array carries or points at. An `object`-dtype array holds references
    whose referents stay mutable; an `ndarray` *subclass* holds state of its
    own, and a `MaskedArray`'s mask changed what a verified, frozen, owning
    coefficient meant in the middle of a solve. Deep-freezing arbitrary
    Python objects is not an operation that exists, and what a subclass keeps
    cannot be enumerated, so both are refused by name instead: `hasobject`,
    and a type that is not `ndarray` exactly. The root initialiser refuses an
    `ndarray` subclass *argument* for the same reason rather than converting
    it, since converting would discard a mask and hand back a different
    matrix. Coefficients are base numeric arrays everywhere; a buffer that is
    not one is named, never normalised. This costs nothing the certified
    numeric envelope wanted, and keeps the guard from certifying an
    immutability it cannot deliver.

    Three call sites, for three different reasons:
    `affine_dynamics_verified` answers validity before eligibility;
    `GLMOptimizer.__init__` refuses before any work is done, on both the
    explicit-structure and the deduced-structure paths; and `forward_solve`
    is the retention boundary — every step stores what `F` and `G` returned,
    so checking there covers every composition that *produces* a trajectory
    rather than every caller that consumes one. It remains a lightweight
    sanity check rather than a guarantee — a caller who unfreezes after the
    solve, or who hands `adjoint_solve` a trajectory built by hand, is outside
    it, as anything is in a language with runtime mutability. C-15.7 states
    that envelope and C-13 records that a C++ port should delete this
    machinery rather than reproduce it: `const` coefficients cannot be
    unfrozen, and affineness becomes a compile-time property of the type that
    selects the solve path by static polymorphism instead of by inspection.

13. **Differentiate the discretization you executed.** The C-18 plan clause is
    `OPEN` and nothing is implemented, but the port obligation is already clear
    and belongs beside the others. Once a step's size and coefficients can
    differ from its neighbours', both sweeps must read one immutable plan by
    step index. Deterministic lookup is fine — duplicating coefficients per
    step is not required. What is forbidden is asking a selection or adaptation
    policy a second time. C++ sharpens the temptation: recomputing a small
    lookup is cheaper than storing one per step, and it looks like an
    optimization.

    A mismatched backward plan need not differentiate the executed objective,
    and need not return the gradient of anything: two Euler steps of `y' = u`
    at length 1 with `J = y(2)²/2`, differentiated backward at lengths
    `(1, 2)`, give `(u₁+u₂, 2(u₁+u₂))`, whose Jacobian `[[1,1],[2,2]]` is
    asymmetric. Symmetry and duality corroborate; shared errors can pass them,
    so neither establishes this property — see NUMERICS.md C-14.1.

    The same paragraph governs a second temptation: sharing the packed stage
    offsets between the stepping code and the independent reference. They
    compute the same thing, which is exactly why they may not compute it
    once. NUMERICS.md C-18.5 states this; the tier-1 oracle's whole value is
    that it shares no code with what it checks.

### Modified Newton: the refusal is narrower than it looks

The "not built" table above refuses lagged and modified Jacobians. The reason
is often misstated, including in earlier drafts of this document, so it is
worth being exact.

An approximate **iteration** matrix does not cost accuracy. Newton with a stale
or preconditioned Jacobian converges to the *same* stage values, because the
stage equation being solved is unchanged; only the convergence rate differs.
Used that way, modified Newton and preconditioning are legitimate, and an
implementation is free to add them.

What is prohibited is letting the approximate iteration matrix silently
*become* the derivative operator. Clause C-5.4 requires the adjoint to apply
the transpose of the true Jacobian at the converged stage value. An
implementation that lags the Newton matrix must therefore factor the true
Jacobian once at the solution for the derivative sweeps to use, and pay for it.
The defect recorded in item 4 above is exactly this confusion: dropping the
off-diagonal blocks gives a quasi-Newton iteration that converges correctly and
then hands its own approximate factorization to the adjoint.

The same distinction sets the solve tolerance requirement. The stage values may
be obtained to a tolerance; the derivative solves must use the true operator.

Ergonomics that need not carry forward: the Python `Problem` protocol requires
a general callback problem to supply three second-derivative callbacks even
when they return zero. A problem expressed as `adjungo.core.affine`'s
`AffineDynamics` or `TimeVaryingAffineDynamics` no longer needs them, because
the terms are known to vanish and are skipped; a C++ interface should make that
the default for every structurally known zero.

`TimeVaryingAffineDynamics` is also where the two structural facts come apart,
and a reimplementation must keep them apart. Zero curvature opens the skip;
a constant Jacobian opens factorization reuse. The time-varying case has the
first and not the second, and NUMERICS.md C-17.2 records that the deduction
originally asserted the second for anything verified affine -- correct only
because the constant case was then the only one that existed.

It is also the first place in this package where part of a guarantee is a
caller obligation rather than a constructed fact. The class owns the *form*
`M(.)y + C(.)u + b(.)`, but the coefficients are caller-supplied callables,
and invoking them with `t` alone confines the interface and not a closure over
the control. NUMERICS.md C-17.1 carries the argument and the failure numbers.
A C++ port has a real option Python does not: take coefficient callables as
stateless function pointers, or take sampled arrays on the solve mesh, either
of which closes the residue by construction (C-Q6).
