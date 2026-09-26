"""General Linear Method utilities and custom tableaux."""

import numpy as np

from adjungo.core.method import GLMethod, StageType


def create_custom_glm(
    A: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    V: np.ndarray,
    c: np.ndarray,
    *,
    declared_stage_type: StageType | None = None,
) -> GLMethod:
    """
    Create a custom GLM from tableaux.

    Declaring the stage type is recommended (NUMERICS.md C-8.3). It is checked
    against the exact structure of ``A``, so a coefficient that moves the
    tableau out of the intended class -- a stray entry above the diagonal, two
    SDIRK diagonal entries typed to different precision -- is reported at
    construction instead of being integrated as a different method.

    Args:
        A: Stage coefficient matrix (s, s)
        U: History to stages matrix (s, r)
        B: Stages to output matrix (r, s)
        V: History propagation matrix (r, r)
        c: Abscissae vector (s,)
        declared_stage_type: The intended class, or ``None`` to route by the
            exact structure alone.

    Returns:
        GLMethod instance

    Raises:
        TableauDeclarationError: If ``A`` is not in the declared class.
    """
    return GLMethod(
        A=A, U=U, B=B, V=V, c=c, declared_stage_type=declared_stage_type
    )


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
