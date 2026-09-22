# Assertion audit

Plan unit U-M0.5. A sweep of `tests/` for assertions that cannot fail.

Motivation: `test_adjoint_sensitivity_finite_difference` computed a finite-difference
reference and then asserted only that two dataclass fields were not `None`. Since
those fields are unconditionally assigned arrays, the test could never fail. It
had never tested anything, and it concealed a quantity that is wrong by 97%.

## Changed

| Location | Was | Now | Reason |
|---|---|---|---|
| `test_nonlinear_and_sensitivities.py::test_adjoint_sensitivity_finite_difference` | `assert adj_sens.delta_Lambda is not None`; `assert adj_sens.delta_Mu is not None`, with `delta_lambda_fd` computed and discarded | Central-difference comparison against `delta_Lambda` at a fixed mesh, with a stated tolerance basis, marked `xfail(strict=True)` | The assertions could not fail. Making the comparison real exposes an ε-independent plateau of **3.139691e-02** against a reference of the same magnitude — `delta_Lambda` recovers about 3% of its true value, and `delta_Lambda[N]` is identically zero. Cured by U-M1.3 |

The same test previously set `optimizer._u_hash = None` to force recomputation.
That poke was removed in favour of constructing a fresh optimizer per
perturbation, so the test no longer depends on cache-invalidation internals. The
need for the poke is itself evidence for U-M0.9.

## Examined and kept

| Location | Assertion | Verdict |
|---|---|---|
| `test_methods_library.py:109,126,303` | `assert gamma is not None` | **Meaningful.** `GLMethod.sdirk_gamma` returns `Optional[float]`, yielding `None` for non-SDIRK methods (`core/method.py:58-62`). These are narrowing guards preceding `np.isclose(A[i,i], gamma)`, and they fail if `stage_type` is misclassified |
| `test_solvers.py:102` | `assert cache.factorization is not None` | **Meaningful.** `StepCache.factorization` is optional and is `None` for explicit methods. The assertion checks that the SDIRK solver actually cached a factorization |
| `test_optimizer.py:81` | `pass` | **Not an assertion.** It is the body of the `FakeStageSolver` test double |

## Effect on counts

The suite moved from `8 failed, 89 passed, 1 skipped` to
`8 failed, 88 passed, 1 skipped, 1 xfailed`. The lost "pass" was never earned.

Because `xfail_strict = true` is set in `pyproject.toml`, the marker cannot be
left behind: once U-M1.3 lands, the test reports XPASS and fails the suite until
the marker is removed.
