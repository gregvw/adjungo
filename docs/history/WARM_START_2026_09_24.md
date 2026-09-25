> **SUPERSEDED as authority:** the live `NUMERICS.md`, `AGENTS.md`, implementation and tests govern. This dated checkpoint preserves restart context and evidence; compare it with the live tree before acting.

# Warm start — 2026-09-24

Created for travel on 2026-09-24, approximately 16:48 MDT (America/Denver).
Start here after interruption. This is a checkpoint, not a new certification
or an instruction to continue implementation.

## Live state and custody

- Repository: `/Users/greg.von.winckel/Projects/github/adjungo`.
- Branch: `main`; HEAD: `5c19cc8676d1b39435d6bfeee51671f3a67ce517`.
- Local `origin/main`: `4b749628268f29fd554081c98dfd236dc4663990`.
  The branch was 21 commits ahead of that local tracking ref; no fetch was
  performed, so this is not a statement about the remote's current state.
- One worktree, at the repository path above. Working tree and index were
  **clean before this checkpoint**. There was no implementation WIP to save.
- This checkpoint adds this document, the accompanying evidence and checksum
  manifest, and its row in `docs/history/README.md`. Those checkpoint files
  are deliberately left uncommitted. No implementation files were changed.
- No commit, push, stash, reset, cleanup, merge or new scientific-policy
  approval was performed for this request. No task depends on a running
  Codex subagent or a temporary diagnostic script.
- The saved source is already in the local Git commits. No recovery patch is
  needed. Do not reset to the older review base or reapply an old WIP patch.

## What changed since the last review message

The preceding chat review concerned five uncommitted files on `dd535b4`,
at an 805-test baseline. **That is no longer the live tree.** Two commits
were discovered while making this checkpoint:

| Commit | Recorded change |
|---|---|
| `6caf0e7fc3a0851f8f018d1797c0b41a252ec295` | Close four copy-traversal seams in the coefficient-immutability walk. |
| `5c19cc8676d1b39435d6bfeee51671f3a67ce517` | Withdraw the storage traversal and draw the out-of-band boundary. |

In the current implementation, the provenance walk no longer follows
`ndarray.base` or `memoryview.obj`; `memoryview` modelling was also removed.
The distinction is reconstruction reachability versus shared storage.
The previous `np.frombuffer(array("d", ...))` over-refusal is now covered by
`test_metadata_over_an_external_buffer_costs_no_answer`.

The live C-15.7 now records an out-of-band pickle boundary: same-process
loads using the providers returned by the dump, or wrappers over that same
storage, are supported; replacement storage, even byte-for-byte faithful
cross-process transport, is excluded. Read the actual clause before reviewing
this boundary. This checkpoint records the committed policy; it does not
independently approve or fully review that amendment.

## Evidence and its limits

Codex freshly ran these commands on HEAD `5c19cc8` during checkpoint creation:

| Command | Observed result |
|---|---|
| `.venv/bin/python -m pytest` | 822 passed |
| `.venv/bin/python -m ruff check adjungo tests examples` | Clean |
| `.venv/bin/python -m mypy adjungo` | Clean, 41 source files |
| `.venv/bin/python examples/minimum_energy_oscillator.py` | Ran successfully and converged |
| `git diff --check` | Clean before checkpoint writes |

Outputs, environment and the targeted closure observations are preserved in
[`verification.txt`](checkpoints/2026-09-24/verification.txt).
A second full-suite run after adding these checkpoint documents also passed
all 822 tests.

Runtime: Python 3.14.3, NumPy 2.5.3, SciPy 1.18.1, macOS 26.7 arm64,
NumPy linked to Apple Accelerate. The interpreter and imported package were
verified to belong to this checkout.

The contract and latest commit report **95/95 injected defects detected at
an 822-test baseline**. Codex did **not** rerun that campaign. The earlier
93/93-at-805 and 90/90-at-793 reports are historical, not current evidence.
The two newer commits have not received a full independent review in this
checkpoint task.

### Reproduced closure of the last two findings

Both the external-buffer metadata case and three chained protocol-5 round
trips followed by deepcopy now succeed and match the closed-form gradient.
The exact setup is C-15.7's scalar explicit-Euler fixture:
`M=0.4`, `C=1`, `b=0`, `y0=0.8`, two steps on `(0, 0.5)`,
controls `(0.3, 0.2)`, terminal objective `J=0.5*y2**2`.
The absolute comparison budget is `EXACT_TOL=1e-13`, justified beside the
fixture under C-3. No mesh refinement or finite-difference oracle is involved.

Run from the repository root:

```sh
.venv/bin/python - <<'PY'
from array import array
import copy
import pickle
import numpy as np
from tests.test_coefficient_immutability import (
    AnswersWithAFrozenBuffer, OverwritesCoefficientMidSolve,
    _gradient, _closed_form, EXACT_TOL,
)

def check(problem):
    np.testing.assert_allclose(
        _gradient(problem, OverwritesCoefficientMidSolve(None)),
        _closed_form()[0], rtol=0, atol=EXACT_TOL,
    )

p = AnswersWithAFrozenBuffer()
p.metadata = np.frombuffer(array("d", [0.0]), dtype=np.float64)
check(p)
check(copy.deepcopy(p))
for protocol in (4, 5):
    check(pickle.loads(pickle.dumps(p, protocol=protocol)))

p = AnswersWithAFrozenBuffer()
for _ in range(3):
    p = pickle.loads(pickle.dumps(p, protocol=5))
    check(p)
check(copy.deepcopy(p))
print("Both earlier findings remain closed on this tree.")
PY
```

These are targeted closure checks, not a blanket assurance about all copy
protocols or the expanded caller envelope.

## Exact warm-start commands

```sh
cd /Users/greg.von.winckel/Projects/github/adjungo
git status --short --branch
git rev-parse HEAD
git worktree list
shasum -a 256 -c docs/history/checkpoints/2026-09-24/SHA256SUMS
.venv/bin/python - <<'PY'
from pathlib import Path
import sys
import adjungo
root = Path.cwd().resolve()
assert Path(adjungo.__file__).resolve() == root / "adjungo" / "__init__.py"
print(sys.executable)
print(adjungo.__file__)
PY
.venv/bin/python -m pytest
.venv/bin/python -m ruff check adjungo tests examples
.venv/bin/python -m mypy adjungo
.venv/bin/python examples/minimum_energy_oscillator.py
```

The checksum manifest covers all 117 pre-checkpoint tracked files plus this
checkpoint and its saved verification output, with the archive index hashed
after its update. A mismatch means the tree changed: inspect the diff rather
than restoring files automatically. The manifest excludes itself and does not
capture `.venv`; use the recorded versions and package-origin check.

Read `AGENTS.md` and the relevant live contract clauses before resuming.
If doing defect injection later, obey R-11 and the exact cache-clearing
instructions in `AGENTS.md`. Do not add another `-q`; pytest already sets
one and over-quiet output previously broke campaign count parsing.

## Next work and parked questions

1. Reconcile HEAD, status and hashes with this checkpoint. Preserve any newer
   work and identify its owner before changing files.
2. If resuming independent review, inspect `git diff dd535b4..5c19cc8`,
   especially C-15.7, the copy/registry implementation and its tests. Confirm
   the scope and authority of the latest out-of-band boundary rather than
   treating passing tests as policy approval.
3. Do not reopen the two preceding findings as still failing without rerunning
   their reproducers: they passed during this checkpoint.
4. No implementation milestone was requested for the restart. Await the
   user's direction before changing numerical semantics or extending the
   caller envelope.

The contract still lists C-Q3 (`nu=0`), C-Q4 (multistep external vectors),
C-Q5 (`y_scale`) and C-Q6 (sampled time-varying coefficients) as open.
They are parked; checkpoint creation authorizes none of that work.
C-1's reference/pilot scope and C-13's C++ port guidance remain controlling.
