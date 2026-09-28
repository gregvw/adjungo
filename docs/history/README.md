# Historical reports — archive and accounting checklist

The original 17 reports in this directory were written during earlier development and kept
in the repository root; `ASSERTION_AUDIT.md` joined them later, from `tests/`,
where it was referenced from nothing. They are **superseded**. Each now carries
a banner saying so.

This file is the accounting required before any of them may be deleted. For
every report it records where its durable content now lives, and — just as
importantly — which of its claims were deliberately **not** promoted, and why.
Nothing here is normative. The normative documents are
[`NUMERICS.md`](../../NUMERICS.md), [`README.md`](../../README.md),
[`AGENTS.md`](../../AGENTS.md) and [`docs/glm_opt.tex`](../glm_opt.tex).

## The standard applied

A claim was promoted into the contract only if it is **reproducible**. For a
derivation that means the derivation is written out. For a measured number it
means the report states the problem, the initial state, the control, the time
span, the mesh, and the error norm, so that the number can be recomputed and
would fail if the code regressed.

Most measured numbers in these reports do **not** meet that standard. They were
written as progress notes, and typically give a figure such as "gradient error
improved from 9e-4 to 6.7e-4" without saying on what problem, at what mesh, in
what norm. Such a figure cannot be rechecked, cannot fail, and therefore cannot
support a certification. Promoting it would create the appearance of evidence
where there is none, which is the specific failure
[C-14.2](../../NUMERICS.md#c-14-the-certification-test-population) exists to
prevent.

Those numbers are **not deleted** — they remain here, in context, as the record
of what was observed at the time. They are simply not cited as authority.

The second reason a claim was not promoted is that it has been **replaced by
stronger evidence**. Several reports establish a result by agreement with SciPy
or with a hand-computed scalar case. The repository now validates against an
independently assembled monolithic reference
([`adjungo/validation/reference.py`](../../adjungo/validation/reference.py)),
which forms `dJ/du = J_u − J_w (dR/dw)⁻¹ dR/du` by a dense solve and shares no
code with the stepping and adjoint implementations. Where the two overlap, the
current evidence is the stronger and the historical result is redundant rather
than lost.

## Per-report accounting

| Report | Durable content | Where it lives now | Deliberately not promoted |
|---|---|---|---|
| `WARM_START_2026_09_24.md` | Travel checkpoint at `5c19cc8`: Git custody, verified runtime, 822-test baseline, closure reproducers, hashes and resume commands. | Current numerical claims remain in `NUMERICS.md` C-15.7 and `tests/test_coefficient_immutability.py`; dated evidence and checksums accompany this snapshot in `checkpoints/2026-09-24/`. | The checkpoint is not a full independent review, injection-campaign rerun, policy approval or authorization to begin parked work. |
| `ADJOINT_FIXES_SUMMARY.md` | The stage adjoint carries the step size `h`; omitting it from the terminal and coupling terms is a defect, not a scaling choice. | `NUMERICS.md` C-8.1 (placement of `h`) and C-13 item 1. | A scalar anchor `dy/dt = u`, `J = ½y(T)² + ½u²` with derivative `2.0`. The anchor idea was promoted; this instance's indexing is not fully stated. Closed-form anchors now live in `tests/`. |
| `ADJOINT_SENSITIVITY_IMPLEMENTED.md` | The second-order adjoint is a second backward sweep with the same operator and a different right-hand side. | `NUMERICS.md` R-6; implemented in `adjungo/stepping/sensitivity.py`. | Hessian finite-difference discrepancies of `6.7e-4`–`3e-3`: no random direction, objective, mesh or tolerance recorded. |
| `ADJOINT_SIGN_ISSUE.md` | The sign conflict between the Lagrangian as stated and the terminal/recursion/gradient relations derived from it. | **Fixed in `docs/glm_opt.tex`** — the Lagrangian now adjoins constraints with a minus sign, and a "Sign convention (corrected)" paragraph explains why `Aᴴμ = Bᴴλ` is invariant under the flip while the relations carrying `∂J` are not. Matches `adjungo/stepping/adjoint.py` and `adjungo/optimization/gradient.py`. | Diagnostic vectors for `dy/dt = −y + u`, `h = 0.333`: the objective and initial state are not fully stated. |
| `ASSERTION_AUDIT.md` | An assertion that computes a reference and then checks only `is not None` has never tested anything; this repository shipped one, and it concealed a `delta_Lambda` wrong by 97%. | The rule is in [`AGENTS.md`](../../AGENTS.md) ("Assertions must be able to fail") and `NUMERICS.md` C-14.2, which requires a test to be shown failing against a broken implementation. The cured test is `tests/test_nonlinear_and_sensitivities.py::test_adjoint_sensitivity_finite_difference`. | The suite counts (`8 failed, 89 passed, 1 skipped`) and the `xfail(strict=True)` bookkeeping, which describe a 98-test tree that no longer exists. Held in `tests/` rather than here until 2026-09-28, referenced from nothing. |
| `BUG_REPORT.md` | A catalogue of early defects. | The ones with an identified mechanism are `NUMERICS.md` R-1 … R-6. | The observation that an integration objective stayed at exactly `5.0` across a gradient step. No root cause was ever established and no reproducer survives. Recorded here as a debugging episode only; see the note below. |
| `CODEX_VS_CLAUDE_COMPARISON.md` | The recommendation to merge adjoint work with multistep work. | Superseded by a stronger policy: `r > 1` is refused outright under `NUMERICS.md` C-6.2, not deferred. | Nothing durable beyond that. |
| `FORWARD_SENSITIVITY_IMPLEMENTED.md` | The tangent recursion uses `F_j` — the Jacobian of the stage being differentiated *with respect to* — where the adjoint recursion uses `F_k` factored out of the whole weighted sum. | `NUMERICS.md` C-13 item 2, which states the asymmetry explicitly. | Agreement figures without a stated mesh. |
| `HESSIAN_BUG_SUMMARY.md` | The terminal cost contributes curvature that a Gauss–Newton-shaped assembly drops entirely. | `NUMERICS.md` R-6. | `‖H_terminal‖ = 0.022` and the component values: no complete objective specified. |
| `LTIC_CRANK_NICOLSON_PROGRESS.md` | Crank–Nicolson is order 2, and a single-mesh tolerance is not a test of that. | `tests/test_ltic_crank_nicolson.py::test_crank_nicolson_lti_observed_order`, which measures the rate over five meshes; `NUMERICS.md` C-4. | "Energy drift below 1% over 100 periods." No invariant, norm or horizon is defined well enough to test, and energy behaviour is not a certified claim. |
| `NONLINEAR_AND_SENSITIVITY_STATUS.md` | Progress narration. | — | Quadratic-drag optimisation reaching "error < 0.5": no setup. |
| `python_implementation.md` | The module layout, which the delivered package follows closely. Written *before* implementation as a design proposal; its Python blocks are sketches, most with `...` bodies, under the package name `glm_opt`. | Superseded by [`docs/architecture.md`](../architecture.md), which documents the delivered structure, the four sweeps, the `glm_opt.tex`-to-code symbol map, and an explicit list of what is *not* built. | A factorization-count table (`Explicit 0`, `SDIRK+linear 1`, `DIRK+linear s`, ...). The counts are plausible and are restated as a claim in `docs/adjoint_sensitivity_insight.md`, but nothing measures them. Milestone M6 adds an instrumented counter; until then they are not evidence. Also proposed `solvers/imex.py` and AD-based derivatives, neither of which exists. |
| `SCIPY_VALIDATION.md` | Forward solvers were checked against SciPy and against analytical solutions; RK4 showed approximately fourth order over `N = 10, 20, 40, 80`. | Superseded. Order is now measured by rate over a refinement sequence against a closed-form `expm` solution in `tests/test_ltic_crank_nicolson.py` and `tests/test_fully_implicit.py`. | External agreement with SciPy as a *validation precedent*. SciPy solves the continuous problem adaptively; it is not an oracle for this repository's fixed-mesh discrete derivative (`NUMERICS.md` C-2). Retaining it as authority would invite exactly the confusion C-2 forbids. |
| `SENSITIVITY_COMPLETE_STATUS.md` | Progress narration. | — | Scalar validation `0.03139691` against itself; Hessian discrepancy `2–3e-3` without setup. |
| `SESSION_FINAL_SUMMARY.md` | Index of the other reports. | — | Nothing unique. |
| `SESSION_SUMMARY_HESSIAN_FIXES.md` | Narration of the Hessian work. | Mechanism in `NUMERICS.md` R-6. | The `15.67×` improvement figure: no baseline definition. |
| `TERMINAL_HESSIAN_BUG_SOLVED.md` | (a) The terminal-Hessian mechanism. (b) **A cost argument**: assembling the terminal contribution directly needs `O(N·s·ν)` forward sensitivity solves, one per control component, whereas a single modified backward sweep gives the same operator matrix-free. | (a) `NUMERICS.md` R-6. (b) **Promoted** to `NUMERICS.md` C-13 item 9, because it is guidance a C++ reimplementer needs and would otherwise have to rediscover. | The `6.404827e-3` fixed-perturbation plateau and the `11.70% → 24.14%` growth figures: the mechanism they demonstrate (an ε-independent plateau indicates a systematic error, not finite-difference noise) *is* promoted, in `NUMERICS.md` under the ε-sweep discussion. The numbers themselves lack setup. |
| `TEST_RESULTS_SUMMARY.md` | The isolation result: forward solvers agreed with references while gradients did not, localising the defect to the adjoint. | The reasoning pattern is `NUMERICS.md` C-14.1 — corroboration cannot substitute for an independent oracle. | The 11-test population, which no longer exists. |
| `TEST_REVIEW.md` | Recommendations to add invalid-input tests, Newton-non-convergence tests, and factorization-reuse tests. | All three now exist and are required: `NUMERICS.md` C-5.3 (non-convergence raises with a locator), C-6.2 (hard refusals), R-9 (factorisation reuse must be gated on a declaration, never a runtime probe). | A separate normative clause on test design. The substance is already binding through the clauses above; a second statement would only create two places to keep in agreement. |

### Note on the "objective stuck at 5.0" report

`BUG_REPORT.md` records that `objective_value` returned exactly `5.0` before and
after a gradient step. This is **not** promoted to a precedent. A precedent
states a mechanism that a future reader can guard against; no mechanism was ever
identified here, and two plausible explanations offered at the time — a stale
forward cache and a non-functioning objective evaluation — were never
distinguished. The current tree has a cache-invalidation test
(`tests/test_cache_and_structure.py`) and an objective that is validated against
the monolithic reference, so if either hypothesis were true today it would fail
a test. Preserving the episode as a precedent would assert a cause that was
never established.

## Before deleting anything

These files may be removed only when every row above is still accurate against
the tree at that time. The check is not "is the report old" but "is its durable
content still recorded somewhere that a reader will find". If a clause cited
above is ever removed or rewritten, this table must be re-verified first.
