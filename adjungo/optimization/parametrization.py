"""Affine control parametrizations: the C-10 adapter layer.

Adjungo computes derivatives with respect to the *stage controls*
``u ∈ ℝ^(N×s×ν)`` -- one control value per step, per stage, per component.
That is the layer in which C-2's exactness claim is stated, and it is the
layer the solvers and adjoints use.

It is rarely the layer a user wants to optimise in. Stage controls are tied
to the tableau: change from ``heun`` to ``rk4`` and the number of unknowns
changes, even though the physical control being sought did not. They also
allow the control to jump discontinuously between stages within one step,
which is usually an artefact rather than a modelling choice.

This module supplies the C-10.1 *adapter layer*: a map from a parameter
vector ``θ`` to stage controls, together with the two derivative transforms
that map derivatives back. For an affine map ``u = P θ + q`` (C-10.2):

    g_θ    = Pᵀ g_u
    H_θ v  = Pᵀ H_u (P v)

Both shipped maps are affine with ``q = 0``:

- :class:`PiecewiseConstantControl` -- one value per step, held across all
  stages of that step.
- :class:`NodalControl` -- values at the ``N + 1`` step nodes, linearly
  interpolated to the stage abscissae ``t_n + c_i h``.

Nothing here is matrix-based at run time; ``P`` is never formed. The
transforms are implemented directly, and :meth:`AffineControlParametrization.matrix`
exists so that tests can materialise ``P`` and check that
:meth:`~AffineControlParametrization.pullback` really is its transpose. A
transposition error in this layer would otherwise be invisible: it produces
a descent-like direction that is simply not the gradient, and the optimizer
would converge slowly rather than fail.

Nonlinear maps are specified in C-10.3 and are **not** implemented. They
need an extra curvature term ``Σ_i (g_u)_i ∂²m_i/∂θ²``; omitting it silently
yields a Gauss-Newton operator, which C-2 forbids presenting as the exact
Hessian.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AffineControlParametrization",
    "NodalControl",
    "PiecewiseConstantControl",
]


class AffineControlParametrization(ABC):
    """An affine map ``u = P θ + q`` from parameters to stage controls.

    Subclasses implement three operations. They must be mutually consistent,
    and :func:`tests.test_parametrization` checks that they are:

    ``expand(θ)``
        The map itself. This is definitional: it produces the stage controls
        the solver actually integrates, so whatever it does *is* the
        discrete problem being differentiated.
    ``push(v)``
        The linear part applied to a direction, ``P v``. Equal to
        ``expand(θ + v) - expand(θ)`` for any ``θ``, exactly, because the
        map is affine.
    ``pullback(g)``
        The transpose ``Pᵀ g``. Defined by ``⟨P v, g⟩ = ⟨v, Pᵀ g⟩`` for all
        ``v`` and ``g``, in the unweighted Euclidean product of the flat
        arrays -- C-10.4's coordinate convention, not a Riesz
        representative under an ``h``-weighted product.

    Attributes:
        n_steps: Number of steps ``N``.
        n_stages: Number of stages ``s`` in the tableau.
        control_dim: Number of control components ``ν``.
    """

    def __init__(self, n_steps: int, n_stages: int, control_dim: int) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1; got {n_steps}")
        if n_stages < 1:
            raise ValueError(f"n_stages must be >= 1; got {n_stages}")
        if control_dim < 1:
            raise ValueError(f"control_dim must be >= 1; got {control_dim}")

        self.n_steps = int(n_steps)
        self.n_stages = int(n_stages)
        self.control_dim = int(control_dim)

    @property
    def stage_shape(self) -> tuple[int, int, int]:
        """Shape ``(N, s, ν)`` of the stage-control array."""
        return (self.n_steps, self.n_stages, self.control_dim)

    @property
    def n_stage_controls(self) -> int:
        """Total number of stage controls, ``N·s·ν``."""
        return self.n_steps * self.n_stages * self.control_dim

    @property
    @abstractmethod
    def parameter_shape(self) -> tuple[int, ...]:
        """Natural (unflattened) shape of the parameter array."""

    @property
    def n_parameters(self) -> int:
        """Total number of parameters, the length of the flat ``θ``."""
        return int(np.prod(self.parameter_shape))

    @abstractmethod
    def expand(self, theta: NDArray) -> NDArray:
        """Map parameters to stage controls, returning shape ``(N, s, ν)``."""

    @abstractmethod
    def push(self, v: NDArray) -> NDArray:
        """Apply the linear part to a parameter-space direction: ``P v``."""

    @abstractmethod
    def pullback(self, g: NDArray) -> NDArray:
        """Apply the transpose to a stage-control covector: ``Pᵀ g``."""

    def _check_parameter_shape(self, theta: NDArray) -> NDArray:
        arr = np.asarray(theta, dtype=float)
        if arr.size != self.n_parameters:
            raise ValueError(
                f"{type(self).__name__} expects {self.n_parameters} "
                f"parameters (shape {self.parameter_shape}); got an array "
                f"of size {arr.size} with shape {arr.shape}."
            )
        return arr.reshape(self.parameter_shape)

    def _check_stage_shape(self, g: NDArray) -> NDArray:
        arr = np.asarray(g, dtype=float)
        if arr.size != self.n_stage_controls:
            raise ValueError(
                f"{type(self).__name__} expects a stage-control array of "
                f"size {self.n_stage_controls} (shape {self.stage_shape}); "
                f"got size {arr.size} with shape {arr.shape}."
            )
        return arr.reshape(self.stage_shape)

    def matrix(self) -> NDArray:
        """Materialise ``P`` as a dense ``(N·s·ν, n_parameters)`` array.

        Built from :meth:`push` applied to each Cartesian basis vector. This
        is for testing, documentation and small-scale inspection. It is not
        used by any derivative path, and it costs ``n_parameters`` calls, so
        it should not be placed inside an optimisation loop.
        """
        columns = np.empty((self.n_stage_controls, self.n_parameters))
        basis = np.zeros(self.n_parameters)
        for j in range(self.n_parameters):
            basis[j] = 1.0
            columns[:, j] = self.push(basis).ravel()
            basis[j] = 0.0
        return columns


class PiecewiseConstantControl(AffineControlParametrization):
    """One control value per step, held constant across that step's stages.

    ``θ`` has shape ``(N, ν)`` and ``u[n, i, :] = θ[n, :]`` for every stage
    ``i``. This is the natural parametrization for a control that a physical
    actuator holds over each step, and it removes the tableau's stage count
    from the optimisation problem: the number of unknowns is ``N·ν``
    whichever method is used.

    The transpose sums over the stage axis. That is not a choice of
    convention but a consequence: parameter ``θ[n, c]`` influences ``s``
    distinct stage controls, so its derivative collects ``s`` contributions.
    Averaging instead of summing -- an easy substitution, since the *forward*
    map broadcasts -- would scale the gradient by ``1/s`` and break C-10.4.
    """

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (self.n_steps, self.control_dim)

    def expand(self, theta: NDArray) -> NDArray:
        arr = self._check_parameter_shape(theta)
        return np.broadcast_to(
            arr[:, None, :], self.stage_shape
        ).copy()

    def push(self, v: NDArray) -> NDArray:
        return self.expand(v)

    def pullback(self, g: NDArray) -> NDArray:
        arr = self._check_stage_shape(g)
        return arr.sum(axis=1)


class NodalControl(AffineControlParametrization):
    """Node values linearly interpolated to the stage abscissae.

    ``θ`` has shape ``(N + 1, ν)``: one value at each step boundary
    ``t_0, …, t_N``. Within step ``n`` the control is the linear
    interpolant between ``θ[n]`` and ``θ[n + 1]``, sampled at the stage
    abscissae ``c_i``::

        u[n, i, :] = (1 - c_i) · θ[n, :] + c_i · θ[n + 1, :]

    This yields a control that is continuous across step boundaries, which
    piecewise-constant parametrization cannot represent, at the cost of
    ``ν`` extra unknowns.

    The abscissae are the tableau's own ``c`` vector, captured at
    construction. They are *not* re-read from the method later: the
    interpolation weights and the stage times the solver evaluates must come
    from the same tableau, or the control seen by ``f`` is sampled at
    different instants than the parametrization believes.

    Note:
        ``c_i`` outside ``[0, 1]`` is permitted and gives linear
        extrapolation rather than interpolation. No certified tableau in
        this repository has such an abscissa, but the formula is stated for
        the general case rather than silently assuming the common one.
    """

    def __init__(
        self,
        n_steps: int,
        control_dim: int,
        c: NDArray,
    ) -> None:
        c_arr = np.asarray(c, dtype=float).ravel()
        super().__init__(n_steps, c_arr.size, control_dim)

        if not np.all(np.isfinite(c_arr)):
            raise ValueError(
                f"stage abscissae must all be finite; got c = {c_arr}"
            )

        self.c = c_arr
        # Interpolation weights, shape (s, 2): left node, right node.
        self._w = np.stack([1.0 - c_arr, c_arr], axis=1)

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (self.n_steps + 1, self.control_dim)

    def expand(self, theta: NDArray) -> NDArray:
        arr = self._check_parameter_shape(theta)
        left = arr[:-1]
        right = arr[1:]
        return np.asarray(
            self._w[None, :, 0, None] * left[:, None, :]
            + self._w[None, :, 1, None] * right[:, None, :],
            dtype=float,
        )

    def push(self, v: NDArray) -> NDArray:
        return self.expand(v)

    def pullback(self, g: NDArray) -> NDArray:
        arr = self._check_stage_shape(g)
        out = np.zeros(self.parameter_shape)
        # Each interior node receives from the step on its left and the step
        # on its right. The two statements are sequential accumulations onto
        # overlapping slices, so interior nodes collect both contributions.
        out[:-1] += np.einsum("s,nsc->nc", self._w[:, 0], arr)
        out[1:] += np.einsum("s,nsc->nc", self._w[:, 1], arr)
        return out
