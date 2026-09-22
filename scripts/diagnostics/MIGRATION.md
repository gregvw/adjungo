# Diagnostic script migration checklist

These scripts were quarantined from the repository root, where they were named
`test_*.py` and were picked up by bare `pytest` collection. One of them
(`test_cn_debug.py`) raised at import, which broke collection entirely.

They are **not tests**. They print; they do not assert. They are excluded from
collection by `testpaths = ["tests"]` in `pyproject.toml`.

## Policy

A script is deleted **only once its useful cases exist in `tests/` as
assertions**. Until then it is retained as evidence. Record the disposition of
each below as it is migrated.

Per `AGENTS.md`, new exploratory work goes here, not in the repository root, and
is expected to be temporary.

## A structural observation worth preserving

**Thirteen of these fourteen scripts use `explicit_euler`.** The exception,
`test_cn_debug.py`, uses `implicit_trapezoid` (s = 2) but never reaches a
derivative check.

`explicit_euler` has `s = 1`. At `s = 1` the adjoint stage-coupling loop is
empty, so the stage-index defect recorded as precedent R-2 in `NUMERICS.md`
cannot manifest. Months of Hessian debugging were conducted on the single
configuration where the dominant gradient defect is structurally invisible.

This is the direct motivation for the C-14 certification population, and it is
the reason migrated cases must be re-expressed with `s > 1` and a state-varying
Jacobian rather than transcribed as-is.

## Checklist

| Script | Demonstrates | Disposition | Done |
|---|---|---|---|
| `test_hessian_symmetry.py` | Dense Hessian symmetry, and the `d²J/du²` contribution | Migrate as the C-14.1 item-5 structural check. **Must be accompanied by a value check** — symmetry held to 8.7e-19 while the operator was wrong by 3.8e-3 (precedent R-3) | ☐ |
| `test_terminal_hessian_fix.py` | Adding the terminal `J_yy δy^[N]` term removes the HVP error | Migrate as the regression test for that term (`sensitivity.py:181`) | ☐ |
| `test_quadrature_weighted_objective.py` | Effect of quadrature weighting on the objective and its Hessian | Migrate as the C-9.3 stage-quadrature regression, pinning where `h` and `w` enter | ☐ |
| `test_hessian_convergence.py` | Fixed-mesh FD ε-sweep of the Hessian | Fold into the shared ε-sweep harness (C-3.3). Do not migrate as a standalone test | ☐ |
| `test_hessian_asymptotic_error.py` | A non-zero HVP error persisting as ε → 0 | Fold into the same harness. **This script recorded the plateau that proves the HVP is wrong, and the conclusion drawn at the time — discretization error — was incorrect.** Preserve the number, discard the interpretation (precedent R-1) | ☐ |
| `test_hessian_discretization_convergence.py` | HVP error under **mesh** refinement | Do **not** migrate as a derivative test. This conflates C-2 with C-4. Retain only if a genuine C-4 claim needs it | ☐ |
| `test_hessian_element_convergence.py` | Per-element FD convergence of the Hessian | Subsumed by the ε-sweep harness applied elementwise. Likely deletable without migration | ☐ |
| `test_hessian_per_timestep_error.py` | Which time steps carry the worst HVP error | Diagnostic only, no invariant. Delete once the HVP is certified | ☐ |
| `test_hessian_components.py` | Component-by-component breakdown of the Hessian assembly | Diagnostic only. Delete once the HVP is certified | ☐ |
| `test_debug_adjoint_sens.py` | Adjoint-sensitivity intermediate quantities | Diagnostic only. Delete once the HVP is certified | ☐ |
| `debug_adjoint.py` | Adjoint gradient mismatch on `explicit_euler` | **Stale.** Investigated a mismatch at `s = 1`, where the gradient is in fact exact (verified: central-FD ε-sweep converges at order 2 to a ~2e-12 floor). The mismatch it chased was finite-difference noise. Delete; the finding is recorded here | ☐ |
| `test_cn_debug.py` | Crank–Nicolson stage solve | **Broken.** Raises `ValueError` at import; this is what broke bare `pytest` collection. Its subject — the CN/DIRK path — is covered properly by the M2 work. Delete; the finding is recorded here | ☐ |
| `test_simple.py` | One step, one stage, `dy/dt = u` | Closest existing thing to the C-14.1 item-2 closed-form anchor. Supersede with the derived anchor `y₁ = y₀ + hu`, `J = ½y₁²` ⟹ `g = h y₁`, `Hv = h²v`, which has a known answer rather than a printed one | ☐ |
| `test_sensitivity_demo.py` | Forward sensitivity as an implicit-function derivative | Pedagogical. Promote to `examples/` if it is worth keeping, not to `tests/`. Otherwise delete | ☐ |
