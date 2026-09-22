# AGENTS.md

Operating instructions for AI agents and new contributors working on Adjungo.

## Read first

**[`NUMERICS.md`](NUMERICS.md) is normative.** It holds the domain contract:
accuracy claims (C-2, C-3), the supported-method envelope (C-6), the GLM
convention (C-8), the objective decomposition (C-9), control coordinates (C-10),
and the certification test population (C-14).

Every accuracy, envelope, or degeneracy claim in code, tests, or documentation
must cite a clause that exists. If no clause covers your situation, propose an
amendment — do not invent a local convention.

## Environment

This repository uses a local virtual environment at `.venv/` (gitignored).

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

**Always invoke `.venv/bin/python` explicitly.** Do not assume an activated
shell; each command runs in a fresh process.

## Commands

| Purpose | Command |
|---|---|
| Full test suite | `.venv/bin/python -m pytest` |
| Single file | `.venv/bin/python -m pytest tests/test_glm_core.py` |
| Lint | `.venv/bin/python -m ruff check adjungo tests examples` |
| Lint, autofix | `.venv/bin/python -m ruff check --fix adjungo tests examples` |
| Type check | `.venv/bin/python -m mypy adjungo` |
| Run the example | `.venv/bin/python examples/minimum_energy_oscillator.py` |

The full suite runs in well under a second, so **the full suite is the fast
tier**. There is no reason to select a subset of tests; run all of them.

### Defect injection

When measuring how many tests fail against a deliberately broken version of the
code -- the standard evidence for a cure here -- use:

```bash
find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider
```

Clear the cache **before the injected run and again before the restored run**.
CPython validates a `.pyc` against the source's size and its modification time
*truncated to whole seconds*. A one-character swap such as `F[i]` to `F[j]`
changes neither, and an inject-run-restore cycle here takes far less than a
second, so the interpreter will silently execute the previous bytecode. This has
already produced a false result in this repository; see NUMERICS.md R-11 for the
mechanism and for why the quiet failure direction is the dangerous one.

`pytest` collection is configured in `pyproject.toml` under
`[tool.pytest.ini_options]` with `testpaths = ["tests"]`. Bare `pytest` and
`pytest tests/` must report identical counts.

## Repository rules

### No `test_*.py` at the repository root

Root-level `test_*.py` files are not tests. They previously broke bare `pytest`
collection at import time. Exploratory scripts belong in
`scripts/diagnostics/`, which is excluded from collection.

A diagnostic script is a **temporary** artifact. When it demonstrates something
real, migrate the case into `tests/` as an assertion and delete the script. See
`scripts/diagnostics/MIGRATION.md`.

### Derivative tests hold the mesh fixed

Per C-2 and C-4, these are two different claims and must live in two differently
named tests:

- **Derivative exactness (C-2)** — fixed mesh, vary ε, expect order 2 and no
  plateau. Never refine the mesh in one of these.
- **Order of accuracy (C-4)** — refine the mesh, compare against a closed-form
  or manufactured solution.

A derivative discrepancy is **never** explained by time-discretization error.
If you find yourself writing that sentence in a commit message, you have found a
bug. This repository made that mistake twice; see precedents R-1 and R-2 in
`NUMERICS.md`.

### Never pin bit-exact floating-point output

Per C-11.3. Development machines here link **Apple Accelerate**; CI links
**OpenBLAS**. LU-based reductions differ in the last ULPs between them.

Every numeric tolerance carries its **basis** in a comment beside it — either a
bound derived under C-3, or a stated relative/ULP budget. A tolerance with no
stated basis certifies a machine, not a method.

When adding or changing a linear-algebra route, check C-11.2: certified paths use
LU only (`lu_factor`/`lu_solve`/`gesv`). Introducing an SVD-based route requires
amending that clause.

### No silent sentinels

Per C-7. Do not return zeros, NaN, or a default array in place of a computed
result. If a path is unimplemented, raise `NotImplementedError` at construction
per C-6.2 so the caller cannot mistake a placeholder for an answer.

### Assertions must be able to fail

An assertion that computes a reference and then checks only `is not None` has
never tested anything. This repository shipped one. When you write a test,
confirm it fails when the code under test is broken.

## Certification status

`NUMERICS.md` C-6.1 holds the authoritative table of which method families are
certified. **Do not describe a family as working in the README or docstrings
before its milestone completes.** Advertised capability exceeding delivered
capability is the defect class this contract exists to prevent.

## Oracle hierarchy

When validating a derivative, prefer stronger oracles (C-14.1):

1. Independently assembled discrete reference (shares no code with
   `adjungo/stepping/`)
2. Closed-form anchors
3. Fixed-mesh ε-sweep
4. Duality test — **corroboration only**; it can pass when tangent and adjoint
   share a mistake
5. Hessian symmetry — necessary, never sufficient

## Style

- Comment only what needs clarification, particularly the *why* behind a
  numerical choice. Do not narrate what the code plainly does.
- Where a formula implements an equation from `docs/glm_opt.tex`, cite it.
- Prefer auditable, directly readable formulations. Per C-1 this code is a
  reference for a future C++ port; Python-specific cleverness costs more than it
  saves.
