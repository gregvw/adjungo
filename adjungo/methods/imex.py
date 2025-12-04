"""IMEX (Implicit-Explicit) method pairs."""

import numpy as np
from dataclasses import dataclass
from adjungo.core.method import GLMethod
from adjungo.methods.runge_kutta import rk4, sdirk2


@dataclass
class IMEXMethod:
    """Additive method: f = f_E + f_I with separate tableaux."""

    explicit: GLMethod   # A^E strictly lower triangular
    implicit: GLMethod   # A^I with structure as classified

    # Shared components
    U: np.ndarray
    V: np.ndarray
    B_E: np.ndarray
    B_I: np.ndarray


def imex_ark2() -> IMEXMethod:
    """2nd-order IMEX ARK pair."""
    gamma = (2.0 - np.sqrt(2.0)) / 2.0

    # Explicit tableau (forward Euler on first stage, trapezoidal on second)
    A_E = np.array([
        [0.0, 0.0],
        [gamma, 0.0],
    ])

    # Implicit tableau (SDIRK)
    A_I = np.array([
        [gamma, 0.0],
        [1.0 - gamma, gamma],
    ])

    U = np.array([[1.0], [1.0]])
    V = np.array([[1.0]])
    B_E = np.array([[1.0 - gamma, gamma]])
    B_I = np.array([[1.0 - gamma, gamma]])
    c = np.array([gamma, 1.0])

    explicit = GLMethod(A=A_E, U=U, B=B_E, V=V, c=c)
    implicit = GLMethod(A=A_I, U=U, B=B_I, V=V, c=c)

    return IMEXMethod(
        explicit=explicit,
        implicit=implicit,
        U=U,
        V=V,
        B_E=B_E,
        B_I=B_I,
    )


def imex_ark3() -> IMEXMethod:
    """3rd-order IMEX ARK pair (SSP-ARK(3,3,2))."""
    gamma = 0.4358665215

    A_E = np.array([
        [0.0, 0.0, 0.0],
        [gamma, 0.0, 0.0],
        [(1.0 - 2.0*gamma)/2.0, gamma, 0.0],
    ])

    A_I = np.array([
        [gamma, 0.0, 0.0],
        [0.0, gamma, 0.0],
        [(1.0 - 2.0*gamma)/2.0, (1.0 - 2.0*gamma)/2.0, gamma],
    ])

    U = np.array([[1.0], [1.0], [1.0]])
    V = np.array([[1.0]])
    B_E = np.array([[(1.0 - 2.0*gamma)/2.0, (1.0 - 2.0*gamma)/2.0, gamma]])
    B_I = np.array([[(1.0 - 2.0*gamma)/2.0, (1.0 - 2.0*gamma)/2.0, gamma]])
    c = np.array([gamma, gamma, 1.0 - gamma])

    explicit = GLMethod(A=A_E, U=U, B=B_E, V=V, c=c)
    implicit = GLMethod(A=A_I, U=U, B=B_I, V=V, c=c)

    return IMEXMethod(
        explicit=explicit,
        implicit=implicit,
        U=U,
        V=V,
        B_E=B_E,
        B_I=B_I,
    )
