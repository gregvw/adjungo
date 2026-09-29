"""The C-14.5 certification problem for a partitioned family.

A [C-8.4](NUMERICS.md) partitioned method needs a problem inside its domain to
be certified on, and that problem has to exist before the method does, or the
first evidence for the method will be produced by whatever the method happens
to compute. This module supplies it. **It depends on no partitioned code**, and
on nothing added for the discretization plan: every method certified under
C-6.1 today can already integrate it, so the problem itself can be validated
before the family it exists to certify is written.

The model
---------

A neutral atom in the lowest band of a one-dimensional-per-axis optical
lattice, held in a shaken, anharmonic trap whose stiffness and lateral force
are the controls::

    H(q, p, u, t) = T(p) + V(q, u, t)

    T(p)     = J_1 (1 - cos p_1) + J_2 (1 - cos p_2)
    V(q,u,t) = (kappa + u_2^2)/2 * ||q - d(t)||^2
               + alpha/4 * (q_1^2 + q_2^2)^2
               - u_1 q_1
    d(t)     = (delta sin(Omega t), 0)

``T`` is the tight-binding band energy, so the group velocity
``dT/dp = J sin(p)`` saturates and reverses rather than growing linearly: the
effective mass depends on the momentum. ``kappa + u_2^2`` is a trap stiffness
set by a laser power, which is quadratic in a field amplitude; ``u_1`` is a
uniform force from a field gradient. ``d(t)`` shakes the trap centre.

Why these terms and not simpler ones
------------------------------------

Every term earns its place against [C-14](NUMERICS.md)'s requirement that the
population exercise, *at the same time*, every structural feature whose absence
can hide a derivative defect. The historical case is finding B0: a wrong stage
index in the adjoint stage solve was invisible for every problem then in the
suite, because each had either ``s = 1`` or a constant Jacobian.

* **``T`` is nonlinear.** With the usual ``T = ||p||^2 / 2`` the block
  ``df^q/dp`` is the identity, and a wrong index in the ``A^q`` stage
  recursion acts on a constant Jacobian and cancels. That is finding B0
  again, in the partition C-8.4 adds. A partitioned certification problem
  with a quadratic kinetic energy tests half of what it appears to.
* **``V`` is quartic in ``q``**, so ``df^p/dq`` varies with the state and the
  contracted Hessian ``F_yy`` is nonzero in the ``p`` block as well as the
  ``q`` block.
* **The stiffness is ``u_2^2``**, not ``u_2``. A control entering linearly
  gives ``F_uu = 0``, and a trap whose centre alone is controlled gives
  ``F_yu = 0`` too. Here ``d2f^p/du_2^2 = -2(q - d)`` and
  ``d2f^p/dq du_2 = -2 u_2 I`` are both nonzero, and both depend on the
  state.
* **``u_1`` enters linearly** and ``u_2`` does not, so a routine that
  contracted over the wrong control component could not stay hidden.
* **``n_q = n_p = 2``, ``n = 4``, ``nu = 2``.** ``n != nu``, and neither is
  ``1``, so a transposed block or a contraction over the wrong index changes
  a shape rather than surviving.
* **``t`` appears explicitly**, through ``d(t)``, and ``J_1 != J_2``,
  ``delta != 0``: a component swap between the two axes changes the answer.

What separability looks like in the Jacobian
--------------------------------------------

``f^q`` depends only on ``p`` and ``f^p`` only on ``(q, u, t)``, so the state
Jacobian is **block anti-diagonal**::

    F = [[0,         df^q/dp],
         [df^p/dq,   0      ]]

and ``df^q/du = 0``. That is the structure a partitioned method exploits, and
:func:`tests.test_prk_problem` asserts it directly rather than inferring it
from the model. A problem that failed it would be outside C-8.4's domain while
still looking like a Hamiltonian system.

The callbacks below are written out by hand. ``tests/test_prk_problem.py``
checks all six against an independent symbolic differentiation of ``f``, and
checks ``f`` itself against an independent symbolic ``J grad H`` -- two paths to
each quantity, neither reading the other.

The objective is :class:`tests.problems.FullCostObjective` at ``nx = 4``,
``nu = 2``, with :data:`TRANSPORT_TARGET`. It already carries a terminal cost,
a running state cost and a running control cost with pairwise-distinct
diagonal weights, which is what C-14 asks of the population, and it is already
exercised by the certified families. A second objective written for this
problem alone would be one more piece of unvalidated arithmetic between the
model and the evidence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["TRANSPORT_TARGET", "ShakenLatticeTrap"]


class ShakenLatticeTrap:
    """Separable-Hamiltonian certification problem for C-8.4 / C-14.5.

    State is ``y = (q_1, q_2, p_1, p_2)``; control is ``u = (u_1, u_2)``.

    Args:
        J: Band hopping energies ``(J_1, J_2)``, deliberately unequal.
        kappa: Baseline trap stiffness.
        alpha: Quartic anharmonicity.
        delta: Amplitude of the trap-centre shaking.
        omega_drive: Angular frequency of the shaking.
    """

    state_dim = 4
    control_dim = 2

    #: Index of the first momentum component. The state splits as
    #: ``y[:n_q]`` and ``y[n_q:]``; no partitioned method object exists yet
    #: (C-6.1 lists the family as not supported), so the split is recorded
    #: here as plain data rather than promised as an API.
    n_q = 2

    def __init__(
        self,
        J: tuple[float, float] = (1.0, 0.6),
        kappa: float = 0.8,
        alpha: float = 0.35,
        delta: float = 0.25,
        omega_drive: float = 1.7,
    ) -> None:
        self.J = np.array(J, dtype=float)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.omega_drive = float(omega_drive)

    # -- the model -----------------------------------------------------

    def centre(self, t: float) -> NDArray:
        """``d(t)``, the commanded trap centre."""
        return np.array([self.delta * np.sin(self.omega_drive * t), 0.0])

    def kinetic(self, p: NDArray) -> float:
        """``T(p)``, the tight-binding band energy."""
        return float(np.sum(self.J * (1.0 - np.cos(p))))

    def potential(self, q: NDArray, u: NDArray, t: float) -> float:
        """``V(q, u, t)``."""
        r = q - self.centre(t)
        return float(
            0.5 * (self.kappa + u[1] ** 2) * (r @ r)
            + 0.25 * self.alpha * (q @ q) ** 2
            - u[0] * q[0]
        )

    def hamiltonian(self, y: NDArray, u: NDArray, t: float) -> float:
        """``H = T(p) + V(q, u, t)``.

        Supplied for the energy-behaviour studies a symplectic method invites.
        Those need their own evidence: per C-18.6, preserving the symplectic
        form implies nothing about energy here, both because the Hamiltonian
        is non-autonomous and because the control does work on the system.
        """
        return self.kinetic(y[self.n_q :]) + self.potential(y[: self.n_q], u, t)

    # -- Problem protocol ----------------------------------------------

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``f = (dH/dp, -dH/dq)``."""
        q, p = y[:2], y[2:]
        r = q - self.centre(t)
        f_q = self.J * np.sin(p)
        f_p = -(self.kappa + u[1] ** 2) * r - self.alpha * (q @ q) * q
        f_p[0] += u[0]
        return np.concatenate([f_q, f_p])

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``df/dy``, block anti-diagonal because ``H`` is separable."""
        q, p = y[:2], y[2:]
        out = np.zeros((4, 4))
        out[0, 2] = self.J[0] * np.cos(p[0])
        out[1, 3] = self.J[1] * np.cos(p[1])

        # d/dq of -alpha (q.q) q is -alpha[(q.q) I + 2 q q^T].
        stiffness = self.kappa + u[1] ** 2
        out[2:, :2] = -stiffness * np.eye(2) - self.alpha * (
            (q @ q) * np.eye(2) + 2.0 * np.outer(q, q)
        )
        return out

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``df/du``. Only the ``p`` block is nonzero: ``V`` carries ``u``."""
        q = y[:2]
        r = q - self.centre(t)
        out = np.zeros((4, 2))
        out[2, 0] = 1.0
        out[2:, 1] = -2.0 * u[1] * r
        return out

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2f_l/dy dy``, shape ``(4, 4)``."""
        q, p = y[:2], y[2:]
        out = np.zeros((4, 4))

        # Kinetic block: f^q_d = J_d sin(p_d), so the only second derivative
        # is d2/dp_d^2 = -J_d sin(p_d), diagonal in the momentum block.
        out[2, 2] = -v[0] * self.J[0] * np.sin(p[0])
        out[3, 3] = -v[1] * self.J[1] * np.sin(p[1])

        # Potential block: f^p = ... - alpha (q.q) q, whose second derivative
        # contracted with w = v[2:] is
        #   -alpha [ 2 (q.w) I + 2 q w^T + 2 w q^T ].
        w = v[2:]
        out[:2, :2] = -2.0 * self.alpha * (
            (q @ w) * np.eye(2) + np.outer(q, w) + np.outer(w, q)
        )
        return out

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2f_l/dy du``, shape ``(4, 2)``.

        Nonzero only through the controlled stiffness: ``u_1`` enters ``f``
        additively and ``u_2`` multiplies ``q``.
        """
        out = np.zeros((4, 2))
        out[:2, 1] = -2.0 * u[1] * v[2:]
        return out

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        """``sum_l v_l d2f_l/du du``, shape ``(2, 2)``.

        ``d2f^p/du_2^2 = -2 (q - d(t))``; every other second derivative in
        ``u`` vanishes, because ``u_1`` enters linearly.
        """
        q = y[:2]
        r = q - self.centre(t)
        out = np.zeros((2, 2))
        out[1, 1] = float(-2.0 * (v[2:] @ r))
        return out


#: Terminal target for the transport problem: the atom displaced and brought
#: to rest. Paired with ``FullCostObjective(nx=4, nu=2, y_target=...)``.
TRANSPORT_TARGET = np.array([1.0, -0.5, 0.0, 0.0])
