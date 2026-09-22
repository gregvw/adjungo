"""Retained but uncertified method tableaux.

Nothing in this subpackage carries an accuracy claim. These definitions are kept
for future work (NUMERICS.md C-6.3) and are unreachable from ``GLMOptimizer``,
which refuses them under C-6.2.

- :mod:`multistep` — linear multistep methods. These need ``r > 1``, for which
  Adjungo has no starting procedure. See C-Q4 for the open representability
  question.
- :mod:`imex` — additive/IMEX pairs. The adjoint of an additive splitting is not
  implemented at all.

Retention is not endorsement. Importing from here and passing the result to the
optimizer raises ``NotImplementedError``.
"""

__all__: list[str] = []
