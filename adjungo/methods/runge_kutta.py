"""Standard Runge-Kutta method tableaux."""

import numpy as np
from adjungo.core.method import GLMethod


def explicit_euler() -> GLMethod:
    """Forward Euler method (1st order)."""
    A = np.array([[0.0]])
    U = np.array([[1.0]])
    B = np.array([[1.0]])
    V = np.array([[1.0]])
    c = np.array([0.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def rk4() -> GLMethod:
    """Classic 4th-order Runge-Kutta method."""
    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    U = np.array([[1.0], [1.0], [1.0], [1.0]])
    B = np.array([[1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]])
    V = np.array([[1.0]])
    c = np.array([0.0, 0.5, 0.5, 1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def heun() -> GLMethod:
    """Heun's method (2nd order)."""
    A = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    U = np.array([[1.0], [1.0]])
    B = np.array([[0.5, 0.5]])
    V = np.array([[1.0]])
    c = np.array([0.0, 1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def implicit_midpoint() -> GLMethod:
    """Implicit midpoint rule (2nd order, symplectic)."""
    A = np.array([[0.5]])
    U = np.array([[1.0]])
    B = np.array([[1.0]])
    V = np.array([[1.0]])
    c = np.array([0.5])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def gauss2() -> GLMethod:
    """2-stage Gauss-Legendre method (4th order)."""
    sqrt3 = np.sqrt(3.0)
    A = np.array([
        [0.25, 0.25 - sqrt3/6.0],
        [0.25 + sqrt3/6.0, 0.25],
    ])
    U = np.array([[1.0], [1.0]])
    B = np.array([[0.5, 0.5]])
    V = np.array([[1.0]])
    c = np.array([0.5 - sqrt3/6.0, 0.5 + sqrt3/6.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def sdirk2() -> GLMethod:
    """2-stage SDIRK method (2nd order, L-stable)."""
    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    A = np.array([
        [gamma, 0.0],
        [1.0 - gamma, gamma],
    ])
    U = np.array([[1.0], [1.0]])
    B = np.array([[1.0 - gamma, gamma]])
    V = np.array([[1.0]])
    c = np.array([gamma, 1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)


def sdirk3() -> GLMethod:
    """3-stage SDIRK method (3rd order)."""
    gamma = 0.4358665215
    A = np.array([
        [gamma, 0.0, 0.0],
        [(1.0 - gamma)/2.0, gamma, 0.0],
        [(-6.0*gamma**2 + 16.0*gamma - 1.0)/4.0,
         (6.0*gamma**2 - 20.0*gamma + 5.0)/4.0, gamma],
    ])
    U = np.array([[1.0], [1.0], [1.0]])
    B = np.array([[A[2, 0], A[2, 1], A[2, 2]]])
    V = np.array([[1.0]])
    c = np.array([gamma, (1.0 + gamma)/2.0, 1.0])
    return GLMethod(A=A, U=U, B=B, V=V, c=c)
