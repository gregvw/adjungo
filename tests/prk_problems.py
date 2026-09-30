"""The C-14.5 certification problem for a partitioned family.

A [C-8.4](NUMERICS.md) partitioned method needs a problem inside its domain to
be certified on, and that problem has to exist before the method does, or the
first evidence for the method will be produced by whatever the method happens
to compute. This module supplies it, and was written and validated before
`symplectic_euler` and `verlet` were. **It depends on no partitioned code**,
and on nothing added for the discretization plan: every other certified family
can already integrate it, so a future disagreement can be attributed to the
partitioned route rather than to the fixture.

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
  ``d2f^p/dq du_2 = -2 u_2 I`` are both nonzero. The first varies with the
  state; the second varies with the control and is constant in the state,
  which is all the term needs to be to catch a contraction over the wrong
  index.
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

__all__ = [
    "EXCITATION_MODES",
    "TRANSPORT_TARGET",
    "DrivenOscillator",
    "ShakenLatticeTrap",
    "TerminalExcitation",
    "excitation",
    "oracle_terminal_map",
]


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
    #: ``y[:n_q]`` and ``y[n_q:]``. The partition is the *problem's* to
    #: declare, never the tableau's to guess: ``state_dim // 2`` is right for
    #: every canonical Hamiltonian and wrong in silence for anything else,
    #: which is the shape of answer C-7 forbids.
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

    def velocity(self, p: NDArray, u: NDArray, t: float) -> NDArray:
        """``dH/dp = dT/dp``, the group velocity. Reads ``p`` alone."""
        return self.J * np.sin(p)

    def force(self, q: NDArray, u: NDArray, t: float) -> NDArray:
        """``-dH/dq = -dV/dq``. Reads ``(q, u, t)`` alone."""
        r = q - self.centre(t)
        out = -(self.kappa + u[1] ** 2) * r - self.alpha * (q @ q) * q
        out[0] += u[0]
        return out

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``f = (dH/dp, -dH/dq)``, assembled from the two halves.

        Written this way so that :class:`SplitShakenLatticeTrap` can declare
        C-8.4's split route over exactly the same arithmetic, which is what
        lets the two routes be compared bit for bit rather than to a
        tolerance.
        """
        return np.concatenate(
            [
                self.velocity(y[self.n_q :], u, t),
                self.force(y[: self.n_q], u, t),
            ]
        )

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


class SplitShakenLatticeTrap(ShakenLatticeTrap):
    """:class:`ShakenLatticeTrap` declaring C-8.4's split right-hand side.

    The same dynamics reached by the other route. Because ``f_q`` and ``f_p``
    are the very functions the inherited ``f`` concatenates, the two routes
    perform identical arithmetic on identical inputs, so their trajectories,
    gradients and Hessian-vector products must agree *exactly* -- a
    tolerance would hide a route that quietly computed something else.

    Nothing here needs the split: ``sin`` and a polynomial are entire, so the
    whole-vector route's totality hypothesis holds by construction. The
    fixture exists to certify that declaring the split changes no answer. For
    a problem the whole-vector route genuinely cannot serve, see
    ``tests/test_partitioned_methods.py::_MovingLogTrap``.
    """

    def f_q(self, p: NDArray, u: NDArray, t: float) -> NDArray:
        return self.velocity(p, u, t)

    def f_p(self, q: NDArray, u: NDArray, t: float) -> NDArray:
        return self.force(q, u, t)


#: Terminal target for the transport problem: the atom displaced and brought
#: to rest. Paired with ``FullCostObjective(nx=4, nu=2, y_target=...)``.
TRANSPORT_TARGET = np.array([1.0, -0.5, 0.0, 0.0])


class DrivenOscillator:
    """``q' = v``, ``v' = -(q - x0(t))``: the C-14.5 refinement fixture.

    A second, deliberately *simple* separable Hamiltonian, for the continuous
    accuracy claim alone. ``H = p²/2 + (q - x₀)²/2`` at ``ω = 1``, driven by
    the control through the potential minimum.

    Why a second fixture, and why this one
    --------------------------------------

    :class:`ShakenLatticeTrap` is built to make derivative defects visible,
    which is why its kinetic energy is not quadratic. Requirement 3 of C-14.5
    asks a different question -- whether the discrete terminal state converges
    to the continuous one at the method's order -- and answering it needs a
    problem with a *closed-form* continuous solution. This one has one::

        z = q + i v  ⟹  z' = -i z + i x₀,  z(0) = 0
        z(T) = i e^{-iT} ∫₀ᵀ e^{it} x₀(t) dt

    so the oracle is an integral, not another discretization. The two fixtures
    are therefore not redundant: neither can answer the other's question. The
    quadratic kinetic energy that would hide an ``A^q`` index defect here is
    harmless, because the claim measured here is not a derivative.

    ``f^q = v`` reads only ``p``, ``f^p`` only ``(q, u)``, and ``∂f^q/∂u = 0``,
    so the problem is inside C-8.4's domain.
    """

    state_dim = 2
    control_dim = 1
    n_q = 1

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([y[1], -(y[0] - u[0])])

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[0.0, 1.0], [-1.0, 0.0]])

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return np.array([[0.0], [1.0]])

    def F_yy_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((2, 2))

    def F_yu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((2, 1))

    def F_uu_action(
        self, y: NDArray, u: NDArray, t: float, v: NDArray
    ) -> NDArray:
        return np.zeros((1, 1))


class TerminalExcitation:
    """``J = ‖(q(T) - 1, v(T))‖² / 2``: displace the oscillator and stop it."""

    target = np.array([1.0, 0.0])

    def evaluate(self, trajectory, u: NDArray) -> float:
        d = trajectory.Y[-1, 0] - self.target
        return 0.5 * float(d @ d)

    def dJ_dy_terminal(self, y: NDArray) -> NDArray:
        g = np.zeros_like(y)
        g[0] = y[0] - self.target
        return g

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros_like(y)

    def dJ_du(self, u: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros_like(u)

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((y.shape[-1], y.shape[-1]))

    def d2J_du2(self, u: NDArray, step: int, stage: int) -> NDArray:
        return np.zeros((len(u), len(u)))


#: Number of sine modes in the excitation basis of C-14.5 requirement 3.
EXCITATION_MODES = 8


def excitation(theta: NDArray, T: float):
    """``x₀(t) = t/T + Σ_k θ_k sin(kπt/T)``, the drive of C-14.5.

    A bare ramp plus a fixed sine basis. The ramp alone carries the target
    displacement; the modes are what a fit has to work with.
    """
    theta = np.asarray(theta, dtype=float)

    def x0(t: float) -> float:
        modes = np.sin(np.arange(1, theta.size + 1) * np.pi * t / T)
        return float(t / T + theta @ modes)

    return x0


def oracle_terminal_map(T: float) -> tuple[NDArray, NDArray]:
    """Exact ``(R, d)`` with continuous terminal residual ``r = R θ + d``.

    From ``z(T) = i e^{-iT} ∫₀ᵀ e^{it} x₀(t) dt`` with ``r = z(T) - 1``, whose
    real and imaginary parts are the position and velocity residuals. The two
    integrals are elementary::

        ∫₀ᵀ e^{it} (t/T) dt      = [e^{iT}(1 - iT) - 1] / T
        ∫₀ᵀ e^{it} sin(ω t) dt   = [E(1+ω) - E(1-ω)] / 2i,
                                   E(a) = (e^{iaT} - 1) / ia

    and are written out rather than quadratured, so the oracle carries no
    discretization of its own. ``ω = kπ/T`` never equals ``1`` for the
    horizons in use, so ``E(1-ω)`` has no removable singularity here; a
    horizon that made ``kπ/T = 1`` would need the limit form and is refused.

    The oracle's standing rests on the integral satisfying the stated ODE and
    initial condition, which is a derivation, not a measurement. The
    independent checks in the test module -- adaptive quadrature, and
    refinement of ``gauss2`` toward it -- corroborate the arithmetic; per
    C-14.5 neither proves it nor bounds the residual error.
    """
    k = np.arange(1, EXCITATION_MODES + 1)
    omega = k * np.pi / T
    if np.any(np.isclose(omega, 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError(
            f"horizon T = {T!r} puts an excitation mode exactly on "
            f"resonance with the oscillator; the closed form below divides "
            f"by 1 - kπ/T."
        )

    def z(integral):
        return 1j * np.exp(-1j * T) * integral

    def E(a):
        return (np.exp(1j * a * T) - 1.0) / (1j * a)

    d_c = z((np.exp(1j * T) * (1.0 - 1j * T) - 1.0) / T) - 1.0
    cols = z((E(1.0 + omega) - E(1.0 - omega)) / 2j)
    return (
        np.array([cols.real, cols.imag]),
        np.array([d_c.real, d_c.imag]),
    )
