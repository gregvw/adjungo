"""Linear multistep method tableaux."""

import numpy as np
from adjungo.core.method import GLMethod


def bdf2() -> GLMethod:
    """2-step BDF method (2nd order, A-stable)."""
    # For BDF2: y_{n+1} = 4/3 y_n - 1/3 y_{n-1} + 2/3 h f_{n+1}
    # Single stage with diagonal A = [[2/3]]
    A = np.array([[2.0/3.0]])
    U = np.array([[4.0/3.0, -1.0/3.0]])  # Coefficients for [y_n, y_{n-1}]
    B = np.array([[1.0], [0.0]])  # Output: [y_{n+1}, y_n]
    V = np.array([[0.0, 1.0], [1.0, 0.0]])  # Shift matrix
    c = np.array([1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def bdf3() -> GLMethod:
    """3-step BDF method (3rd order)."""
    # y_{n+1} = 18/11 y_n - 9/11 y_{n-1} + 2/11 y_{n-2} + 6/11 h f_{n+1}
    A = np.array([[6.0/11.0]])
    U = np.array([[18.0/11.0, -9.0/11.0, 2.0/11.0]])
    B = np.array([[1.0], [0.0], [0.0]])
    V = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    c = np.array([1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def adams_bashforth2() -> GLMethod:
    """2-step Adams-Bashforth method (2nd order)."""
    # y_{n+1} = y_n + h/2 (3 f_n - f_{n-1})
    # Two explicit stages evaluate f_n and f_{n-1}
    A = np.zeros((2, 2))
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([
        [3.0 / 2.0, -1.0 / 2.0],
        [0.0, 0.0],
    ])
    V = np.array([[0.0, 1.0], [1.0, 0.0]])
    # c-values reflect evaluations at t_n and t_{n-1}
    c = np.array([0.0, -1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def adams_moulton2() -> GLMethod:
    """2-step Adams-Moulton method (3rd order, implicit)."""
    # y_{n+1} = y_n + h/12 (5 f_{n+1} + 8 f_n - f_{n-1})
    # Three stages: implicit stage for f_{n+1}, explicit stages for f_n and f_{n-1}
    A = np.array([
        [5.0 / 12.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    U = np.array([
        [1.0, 0.0],  # y_n baseline for implicit solve
        [1.0, 0.0],  # y_n for f_n
        [0.0, 1.0],  # y_{n-1} for f_{n-1}
    ])
    B = np.array([
        [5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0],
        [0.0, 0.0, 0.0],
    ])
    V = np.array([[0.0, 1.0], [1.0, 0.0]])
    # c-values at t_{n+1}, t_n, t_{n-1}
    c = np.array([1.0, 0.0, -1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)
