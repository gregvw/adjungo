# Diagnostic script migration record

These scripts were quarantined from the repository root, where they were named
`test_*.py` and were picked up by bare `pytest` collection. One of them
(`test_cn_debug.py`) raised at import, which broke collection entirely.

They are **not tests**. They print; they do not assert. They are excluded from
collection by `testpaths = ["tests"]` in `pyproject.toml`, a property now held by
`tests/test_documentation.py::test_bare_collection_and_scoped_collection_agree`.

## Policy

A script is deleted **only once its useful cases exist in `tests/` as
assertions**. Until then it is retained as evidence.

Per `AGENTS.md`, new exploratory work goes here, not in the repository root, and
is expected to be temporary.

## A structural observation worth preserving

**Thirteen of the original fourteen scripts used `explicit_euler`.** The
exception, `test_cn_debug.py`, used `implicit_trapezoid` (s = 2) but never
reached a derivative check.

`explicit_euler` has `s = 1`. At `s = 1` the adjoint stage-coupling loop is
empty, so the stage-index defect recorded as precedent R-2 in `NUMERICS.md`
cannot manifest. Months of Hessian debugging were conducted on the single
configuration where the dominant gradient defect is structurally invisible.

This is the direct motivation for the C-14 certification population, and it is
why the discharges below are recorded against tests that exercise the C-14
methods with `s > 1`, rather than against transcriptions of the scripts.

## Discharged

Each row names the assertion that now carries the requirement. These scripts
have been deleted; this table is the record of where each one went.

| Script | Requirement | Now asserted by |
|---|---|---|
| `test_simple.py` | One step, one stage, `dy/dt = u` | Superseded by the derived anchor the checklist asked for — `y₁ = y₀ + hu`, `J = ½y₁²` ⟹ `g = h y₁`, `Hv = h²v` — in `test_oracle_gradient.py::test_reference_gradient_matches_closed_form_anchor`, `::test_reference_hessian_matches_closed_form_anchor`, `::test_package_gradient_matches_closed_form_anchor` and `test_oracle_hessian.py::test_package_hvp_matches_closed_form_anchor`. A known answer rather than a printed one. |
| `test_hessian_symmetry.py` | Dense Hessian symmetry | `test_oracle_hessian.py::test_hvp_is_symmetric`, with the value check the checklist required alongside it in `::test_hvp_matches_independent_reference`. Its docstring states that symmetry is corroboration only (precedent R-3). |
| `test_terminal_hessian_fix.py` | The terminal `J_yy δy^[N]` term | `test_oracle_hessian.py::test_hvp_matches_independent_reference`, whose reference assembles that term from the monolithic Lagrangian, and `::test_hvp_refuses_without_objective_second_derivatives`, which pins that dropping the callback refuses instead of silently returning a Gauss-Newton operator. |
| `test_hessian_convergence.py` | Fixed-mesh FD ε-sweep of the Hessian | `test_oracle_hessian.py::test_hvp_finite_difference_sweep` (C-3.3), over the C-14 methods. |
| `test_hessian_asymptotic_error.py` | A non-zero HVP error persisting as ε → 0 | The same sweep, which fails on exactly this signature and names it: "an eps-independent plateau indicates a wrong operator, not round-off". The conclusion drawn when the script was written — discretization error — was wrong, and is preserved as precedent R-1. |
| `test_hessian_element_convergence.py` | Per-element FD convergence | Subsumed by the same sweep, which takes the maximum over components. |
| `test_hessian_discretization_convergence.py` | HVP error under **mesh** refinement | Deliberately not migrated as a derivative test: it conflates C-2 with C-4. A genuine C-4 order study exists instead in `test_fully_implicit.py::test_gauss2_observed_order_is_four`, against `expm(A T) y0`. |
| `test_hessian_per_timestep_error.py` | Which steps carry the worst HVP error | Diagnostic only, no invariant. The stated condition for deletion was HVP certification, and C-6.1 now certifies the HVP for every certified family. |
| `test_hessian_components.py` | Component-by-component Hessian assembly | As above. |
| `test_debug_adjoint_sens.py` | Adjoint-sensitivity intermediates | As above. |
| `debug_adjoint.py` | Adjoint gradient mismatch on `explicit_euler` | **Stale.** It investigated a mismatch at `s = 1`, where the gradient is exact; what it chased was finite-difference noise. The finding is recorded here, which is all that was required of it. |
| `test_cn_debug.py` | Crank–Nicolson stage solve | **Broken.** It raised `ValueError` at import, which is what broke bare collection. Its subject is covered by the M2 DIRK work. |

## Retained

| Script | Why it cannot be discharged yet |
|---|---|
| `test_quadrature_weighted_objective.py` | Stage quadrature is **C-9.3, not yet implemented**. `adjungo/validation/reference.py` records that `dJ/dZ = 0` until it lands. There is no delivered behaviour to regress against, so this case cannot become an assertion yet. Migrate it as the C-9.3 regression when the clause is implemented, pinning where `h` and `w` enter. |
| `test_sensitivity_demo.py` | Pedagogical: forward sensitivity as an implicit-function derivative. It belongs in `examples/`, not `tests/`. Promote it when the implicit-route examples are written, or delete it then. |
