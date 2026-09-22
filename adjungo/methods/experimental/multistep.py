"""Linear multistep method tableaux.

UNCERTIFIED AND UNREACHABLE. Retained under NUMERICS.md C-6.3 for future work.
``GLMOptimizer`` refuses ``r > 1`` at construction under C-6.2, and Adjungo has
no starting procedure for multistep methods.

Status of the definitions below:

- :func:`bdf2`, :func:`bdf3` satisfy the C-8.2 shape invariants and construct
  successfully. They are refused by the optimizer solely because ``r > 1``.
  **Their ``V`` and ``B`` output rows were wrong** and have been corrected; see
  each docstring for the observed error. They remain uncertified: no test in
  this repository asserts their accuracy, per C-6.3.
- :func:`adams_bashforth2`, :func:`adams_moulton2` are **malformed**: they store
  per-history coefficients in the rows of ``A``, giving ``A`` shape ``(1, 2)``.
  Since ``s`` is read from ``A.shape[0]``, these silently reported ``s = 1``
  while supplying two abscissae. They now raise ``ValueError`` at construction.

  The Adams family *is* representable under C-8 if the external vector carries
  ``h·f`` history rather than a second state. For AB2, with external vector
  ``[y_n, h·f_{n-1}]``::

      s = 1, r = 2
      A = [[0]]
      U = [[1, 0]]
      B = [[3/2], [1]]
      V = [[1, -1/2], [0, 0]]

  This is recorded rather than adopted: it is open question C-Q4 whether
  multistep external vectors should hold state or derivative history, and that
  decision belongs with whoever implements the starting procedure.
"""

import numpy as np

from adjungo.core.method import GLMethod


def bdf2() -> GLMethod:
    """2-step BDF method (2nd order, A-stable).

    y_{n+1} = 4/3 y_n - 1/3 y_{n-1} + 2/3 h f_{n+1}, external vector
    [y_n, y_{n-1}].

    The update coefficients belong in the first row of ``V`` and ``B``, because
    under C-8.1 the output is ``y^[n] = V y^[n-1] + h (B f_stages)``. They were
    previously ``V[0,:] = [0, 1]`` and ``B[0,0] = 1``, which computes
    ``y_{n-1} + h f`` instead. Integrating ``y' = -y`` to t=1 at h=0.05 returned
    0.5970 against an exact 0.3679, a 62% error.
    """
    A = np.array([[2.0/3.0]])
    U = np.array([[4.0/3.0, -1.0/3.0]])  # Coefficients for [y_n, y_{n-1}]
    B = np.array([[2.0/3.0], [0.0]])     # Output: [y_{n+1}, y_n]
    V = np.array([[4.0/3.0, -1.0/3.0], [1.0, 0.0]])
    c = np.array([1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def bdf3() -> GLMethod:
    """3-step BDF method (3rd order).

    y_{n+1} = 18/11 y_n - 9/11 y_{n-1} + 2/11 y_{n-2} + 6/11 h f_{n+1}.

    Carried the same defect as :func:`bdf2`, plus a mis-ordered history shift:
    rows 1 and 2 of ``V`` were swapped, so the external vector was permuted
    rather than shifted. Integrating ``y' = -y`` to t=1 at h=0.02 returned
    0.7623 against an exact 0.3679.
    """
    A = np.array([[6.0/11.0]])
    U = np.array([[18.0/11.0, -9.0/11.0, 2.0/11.0]])
    B = np.array([[6.0/11.0], [0.0], [0.0]])
    V = np.array([
        [18.0/11.0, -9.0/11.0, 2.0/11.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    c = np.array([1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def adams_bashforth2() -> GLMethod:
    """2-step Adams-Bashforth method (2nd order)."""
    # y_{n+1} = y_n + h/2 (3 f_n - f_{n-1})
    A = np.array([[0.0, 0.0]])
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[3.0/2.0, -1.0/2.0]])
    V = np.array([[0.0, 1.0], [1.0, 0.0]])
    c = np.array([1.0, 0.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def adams_moulton2() -> GLMethod:
    """2-step Adams-Moulton method (3rd order, implicit)."""
    # y_{n+1} = y_n + h/12 (5 f_{n+1} + 8 f_n - f_{n-1})
    A = np.array([[5.0/12.0, 0.0]])
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[8.0/12.0, -1.0/12.0]])
    V = np.array([[0.0, 1.0], [1.0, 0.0]])
    c = np.array([1.0, 0.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)
