"""General Linear Method utilities and custom tableaux."""

import numpy as np
from adjungo.core.method import GLMethod


def create_custom_glm(
    A: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    V: np.ndarray,
    c: np.ndarray,
) -> GLMethod:
    """
    Create a custom GLM from tableaux.

    Args:
        A: Stage coefficient matrix (s, s)
        U: History to stages matrix (s, r)
        B: Stages to output matrix (r, s)
        V: History propagation matrix (r, r)
        c: Abscissae vector (s,)

    Returns:
        GLMethod instance
    """
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def validate_glm(method: GLMethod, order: int) -> bool:
    """
    Validate GLM order conditions (simplified).

    Args:
        method: GLM to validate
        order: Expected order of accuracy

    Returns:
        True if method satisfies basic consistency conditions
    """
    # Basic consistency: B @ e = V @ e (where e is vector of ones)
    e_s = np.ones(method.s)
    e_r = np.ones(method.r)

    lhs = method.B @ e_s
    rhs = method.V @ e_r

    return np.allclose(lhs, rhs)
