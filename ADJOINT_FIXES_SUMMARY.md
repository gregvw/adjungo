# Adjoint Gradient Fixes Summary

## Issues Discovered and Fixed

### 1. Missing Step Size `h` in Adjoint Solver (✅ Fixed)

**Problem:** The adjoint stage solver interface didn't include the step size `h` parameter, causing incorrect gradient magnitudes.

**Fix:** Added `h` parameter to:
- `adjungo/solvers/base.py:StageSolver.solve_adjoint_stages()` interface
- `adjungo/solvers/explicit.py:65,67` - Added `h *` multipliers
- `adjungo/solvers/sdirk.py:99,101` - Added `h *` multipliers
- `adjungo/solvers/dirk.py:71,73` - Added `h *` multipliers
- `adjungo/stepping/adjoint.py:62` - Pass `h` to solver
- `adjungo/optimization/interface.py:207` - Pass `h` from optimizer

### 2. Missing Jacobian `F[i]^T` in Terminal Contribution (✅ Fixed)

**Problem:** The terminal adjoint contribution was missing the Jacobian multiplication.

**Formula:** `μ_i = h * F_i^T * B * λ + h * Σ_j a_{ji} * F_j^T * μ_j`

**Fix:** Changed in all solvers (explicit, SDIRK, DIRK):
```python
# BEFORE:
mu[i] = h * (B[:, i].T @ lambda_ext)

# AFTER:
mu[i] = h * cache.F[i].T @ (B[:, i] @ lambda_ext)
```

**Files:**
- `adjungo/solvers/explicit.py:65`
- `adjungo/solvers/sdirk.py:99`
- `adjungo/solvers/dirk.py:71`

### 3. Wrong Jacobian Index in Adjoint Loop (✅ Fixed)

**Problem:** Using `F[i]^T` when the formula requires `F[j]^T` in the coupling terms.

**Formula:** `h * Σ_j a_{ji} * F_j^T * μ_j` (uses F at stage j, not i)

**Fix:** Changed in all solvers:
```python
# BEFORE:
mu[i] += h * A[j, i] * cache.F[i].T @ mu[j]

# AFTER:
mu[i] += h * A[j, i] * cache.F[j].T @ mu[j]
```

**Files:**
- `adjungo/solvers/explicit.py:67`
- `adjungo/solvers/sdirk.py:101`
- `adjungo/solvers/dirk.py:73`

### 4. Wrong Sign in Gradient Formula (✅ Fixed) **[CRITICAL]**

**Problem:** The gradient formula had a MINUS sign when it should have a PLUS sign.

**Theoretical Basis:** For the Lagrangian L = J + λ^T(constraint), the gradient is:
```
∇J = ∂J/∂u + λ^T * ∂f/∂u
```

In discrete form with weighted adjoints:
```
∇J = ∂J/∂u + h * G^T * Λ
```

**Fix:** Changed in gradient assembly:
```python
# BEFORE:
grad[step, k] -= h * G_k.T @ Lambda_k

# AFTER:
grad[step, k] += h * G_k.T @ Lambda_k
```

**File:** `adjungo/optimization/gradient.py:55`

**Note:** This contradicts the formula in `docs/glm_opt.tex` and `docs/python_implementation.md` which state:
```
∇J = ∂J/∂u - h * G^T * Λ
```

The LaTeX documentation appears to have a sign error in the discrete gradient formula.

### 5. Test Updates (✅ Fixed)

Updated test mocks and calls to include `h` parameter:
- `tests/test_solvers.py:83,122` - Pass `h` to adjoint solver calls
- `tests/test_optimizer.py:66` - Add `h` parameter to fake_adjoint mock

## Validation

### Test Results

**Before fixes:** 11/86 tests passing (only scipy comparison tests)
**After fixes:** **82/86 tests passing** (95.3%)

Remaining 4 failures are multistep method structural issues (BDF2, Adams-Bashforth, Adams-Moulton) that require proper startup procedures (noted as future work by user).

### Gradient Validation Tests

All gradient validation tests now pass:
- ✅ `test_gradient_finite_difference_explicit_euler` - Validates against finite differences
- ✅ `test_gradient_finite_difference_rk4` - RK4 gradient correctness
- ✅ `test_gradient_zero_for_optimal_control` - Zero gradient at optimum
- ✅ `test_gradient_different_controls_different_gradients` - Gradient variation
- ✅ `test_scipy_interface` - Scipy optimizer interface
- ✅ `test_gradient_shape_consistency` - Array shape validation

### Analytical Test Case

Created `test_simple.py` with ultra-simple problem (`dy/dt = u`) that can be solved analytically:

```
Problem: dy/dt = u, y(0) = 0, J = 0.5*y(T)^2 + 0.5*u^2
Analytical: dJ/du = u*(h^2 + 1) = 2.0
Finite Difference: dJ/du ≈ 2.0 ✓
Adjoint Method: dJ/du = 2.0 ✓
```

## Summary of Changes

**Files Modified:**
1. `adjungo/solvers/base.py` - Interface signature
2. `adjungo/solvers/explicit.py` - Lines 65, 67
3. `adjungo/solvers/sdirk.py` - Lines 99, 101
4. `adjungo/solvers/dirk.py` - Lines 71, 73
5. `adjungo/stepping/adjoint.py` - Line 62
6. `adjungo/optimization/interface.py` - Line 207
7. `adjungo/optimization/gradient.py` - Line 55 (**sign change**)
8. `tests/test_solvers.py` - Lines 83, 122
9. `tests/test_optimizer.py` - Line 66
10. `tests/test_stepping.py` - Lines 134, 164, 227 (already fixed in session context)

**Documentation Issues:**
- `docs/glm_opt.tex` - Gradient formula has wrong sign
- `docs/python_implementation.md:718` - Gradient formula has wrong sign

These should be corrected to:
```
∇_{u_k^n} Ĵ = ∂J/∂u_k^n + h (G_k^n)^T Λ_k^n
```

## Testing Recommendations

1. ✅ Run gradient validation tests: `pytest tests/test_gradient.py -v`
2. ✅ Run integration tests: `pytest tests/test_integration.py -v`
3. ✅ Run scipy comparison: `pytest tests/test_scipy_comparison.py -v`
4. ✅ Run full suite: `pytest tests/ -v`

All recommended tests pass (82/86 total, excluding known multistep issues).
