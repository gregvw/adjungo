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


def implicit_trapezoid() -> GLMethod:
    """
    Implicit trapezoidal rule / Crank-Nicolson method (2nd order, A-stable).

    This is the DIRK(2,2) method:
        y_{n+1} = y_n + h/2 * [f(y_n, u_n, t_n) + f(y_{n+1}, u_{n+1}, t_{n+1})]

    The method has two stages:
        - Stage 1: Explicit evaluation at y_n (c_1 = 0)
        - Stage 2: Implicit evaluation at y_{n+1} (c_2 = 1)

    Properties:
        - A-stable (unconditionally stable)
        - 2nd order accurate
        - Symplectic for Hamiltonian systems
        - Excellent for stiff problems and long-time integration
    """
    A = np.array([
        [0.0, 0.0],      # Stage 1: explicit (Z_1 = y_n)
        [0.5, 0.5],      # Stage 2: implicit (Z_2 involves f(Z_2))
    ])
    U = np.array([[1.0], [1.0]])
    B = np.array([[0.5, 0.5]])  # Equal weights (trapezoidal)
    V = np.array([[1.0]])
    c = np.array([0.0, 1.0])    # Evaluate at t_n and t_{n+1}
    return GLMethod(A=A, U=U, B=B, V=V, c=c)
