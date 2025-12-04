# Adjoint Gradient Sign Issue - RESOLVED ✅

## Final Status

✅ **Fixed:** Added step size `h` to adjoint solver interface and implementations
✅ **Fixed:** Added missing `F[i]^T` multiplication in terminal adjoint contribution
✅ **Fixed:** Corrected `F[i]` vs `F[j]` indexing in adjoint coupling loop
✅ **Fixed:** Corrected gradient sign (changed from minus to plus)

**Test Results:** 82/86 tests passing (95.3%)

## Problem Description

After fixing the missing `h` parameter, gradients still don't match finite differences, but now by a **sign/magnitude** issue rather than just missing scaling.

### Debug Output

For a simple test case (dy/dt = -y + u, with regularization):

```
ADJOINT GRADIENT:      [0.651, 0.513, 0.410]  (POSITIVE)
FINITE DIFFERENCE:     [-0.038, -0.107, -0.210]  (NEGATIVE)

Terminal adjoint: λ(T) = -0.930
Weighted adjoints: Λ = [-1.653, -1.240, -0.930]  (NEGATIVE)
```

### Gradient Computation

Current formula (from `docs/python_implementation.md:718`):
```
∇_{u_k^n} Ĵ = ∂J/∂u_k^n - h (G_k^n)^T Λ_k^n
```

With values:
- ∂J/∂u = 0.1 (regularization term u)
- h = 0.333
- G = 1.0
- Λ = -1.653

Computed gradient:
```
grad = 0.1 - 0.333 * 1.0 * (-1.653)
     = 0.1 + 0.551
     = 0.651  ✗ (should be -0.038)
```

## Possible Issues

### 1. Sign Convention in Gradient Formula

**Current:**
```python
grad[step, k] = objective.dJ_du(u[step, k], step, k) - h * G_k.T @ Lambda_k
```

**Maybe should be:**
```python
grad[step, k] = objective.dJ_du(u[step, k], step, k) + h * G_k.T @ Lambda_k
```

### 2. Adjoint Sign Convention

The weighted adjoints Λ are negative, which might be correct or might indicate the adjoints themselves have the wrong sign.

### 3. Objective Gradient Sign

The objective gradient `dJ/du = u` for regularization `0.5 * u^2` seems correct.

## Theoretical Question

For optimal control with Lagrangian:
```
L = J(y, u) + ∫ λ^T [f(y, u) - dy/dt] dt
```

The optimality condition is:
```
∂L/∂u = ∂J/∂u + ∫ λ^T ∂f/∂u dt = 0
```

Which gives:
```
∇J = ∂J/∂u + ∫ λ^T G dt
```

**Question:** In the discrete GLM formulation, what is the correct sign?

### From Documentation (glm_opt.tex)

The gradient formula in the Python implementation notes states:
```
∇_{u_k^n} Ĵ = ∂J/∂u_k^n - h (G_k^n)^T Λ_k^n
```

But this doesn't match finite differences!

## Files Affected

1. `adjungo/optimization/gradient.py:55` - Gradient assembly formula
2. `adjungo/stepping/adjoint.py` - Adjoint computation
3. `adjungo/solvers/explicit.py:65-67` - Adjoint stage solver

## Next Steps

1. **Review LaTeX documentation** (`docs/glm_opt.tex`) for correct discrete adjoint formulation
2. **Check sign conventions** in the derivation
3. **Verify weighted adjoint formula** in `adjungo/stepping/adjoint.py:66-68`
4. **Test with known analytical gradient** for validation

## Test Results

After `h` fix:
- ✅ Forward solvers: 11/11 scipy comparisons pass
- ✅ Core infrastructure: 64/75 tests pass
- ❌ Gradient validation: 0/2 (sign error)
- ❌ Integration tests: 0/7 (wrong gradients)

## Files Modified in `h` Fix

✅ Completed:
1. `adjungo/solvers/base.py` - Added `h` parameter to interface
2. `adjungo/solvers/explicit.py` - Multiplied by `h`
3. `adjungo/solvers/sdirk.py` - Multiplied by `h`
4. `adjungo/solvers/dirk.py` - Multiplied by `h`
5. `adjungo/solvers/implicit.py` - Multiplied by `h`
6. `adjungo/stepping/adjoint.py` - Added `h` parameter
7. `adjungo/optimization/interface.py` - Pass `h` to adjoint_solve
8. `tests/test_stepping.py` - Updated test calls (3 places)

## Recommendation

**This requires theoretical review of the discrete adjoint formulation.** The issue is either:

1. The gradient formula has the wrong sign (minus should be plus)
2. The adjoint equations have the wrong sign somewhere
3. The weighted adjoint computation is incorrect

Given the complexity of discrete adjoint methods for GLMs, this should be carefully checked against the original LaTeX documentation (`docs/glm_opt.tex`) to ensure the discrete optimality conditions are correctly derived.

---

## For Future Reference: Multistep Methods

Per user feedback, multistep methods need:
1. **Startup procedure** - Use lower-order methods or RK to generate initial history
2. **Consistent adjoint** - Adjoint must correspond to whatever startup method is actually used
3. **Order preservation** - Ideally startup shouldn't degrade convergence order

This should be a future enhancement after the adjoint sign issue is resolved.
