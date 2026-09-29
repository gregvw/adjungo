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

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from adjungo.core.plan import DiscretizationPlan

__all__ = [
    "AffineControlParametrization",
    "NodalControl",
    "PiecewiseConstantControl",
]


def _as_per_step(c: NDArray | Sequence[NDArray], n_steps: int) -> list:
    """Resolve an abscissa argument into one entry per step.

    A single vector means "these abscissae at every step". It is recognised
    by being one-dimensional, so a ragged sequence of per-step vectors and a
    rectangular one are both accepted without the caller having to say which
    it meant.
    """
    if isinstance(c, np.ndarray) and c.ndim == 1:
        return [c] * n_steps
    if isinstance(c, np.ndarray):
        return list(c)
    rows = list(c)
    if rows and np.ndim(rows[0]) == 0:
        return [np.asarray(rows, dtype=float)] * n_steps
    return rows


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

    def __init__(
        self,
        n_steps: int,
        n_stages: int | Sequence[int],
        control_dim: int,
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1; got {n_steps}")
        if control_dim < 1:
            raise ValueError(f"control_dim must be >= 1; got {control_dim}")

        if np.ndim(n_stages) == 0:
            counts = (int(cast(int, n_stages)),) * int(n_steps)
        else:
            counts = tuple(int(s) for s in cast(Sequence[int], n_stages))
            if len(counts) != int(n_steps):
                raise ValueError(
                    f"n_stages was given per step, so it needs one entry for "
                    f"each of the {n_steps} steps; got {len(counts)}."
                )
        for step, s in enumerate(counts):
            if s < 1:
                raise ValueError(
                    f"n_stages must be >= 1; step {step} has {s}"
                )

        self.n_steps = int(n_steps)
        self.control_dim = int(control_dim)
        #: Stages per step. A plan may change method between steps, so this
        #: is a sequence rather than one number (C-18.5).
        self.stage_counts = counts
        self.stage_offsets = (0, *itertools.accumulate(counts))

    @property
    def n_stages(self) -> int:
        """The common stage count, when there is one.

        Refused rather than defaulted when the steps disagree: any single
        number named ``n_stages`` would be a property of one step reported as
        a property of the whole map (C-7).
        """
        return self._uniform_stage_count("n_stages")

    def _uniform_stage_count(self, what: str) -> int:
        first = self.stage_counts[0]
        if any(s != first for s in self.stage_counts):
            raise NotImplementedError(
                f"{type(self).__name__} has no single {what}: its steps "
                f"carry stage counts {self.stage_counts}. Use the packed "
                f"stage-control layout, shape {self.packed_shape} "
                f"(NUMERICS.md C-18.5)."
            )
        return first

    @property
    def total_stages(self) -> int:
        """``Σ_n s_n``, the number of stages in the whole plan."""
        return self.stage_offsets[-1]

    @property
    def packed_shape(self) -> tuple[int, int]:
        """Shape ``(Σ_n s_n, ν)`` of the packed stage-control array."""
        return (self.total_stages, self.control_dim)

    @property
    def stage_shape(self) -> tuple[int, int, int]:
        """Shape ``(N, s, ν)``, when the stage count does not vary.

        ``(N, s, ν)`` and the packed ``(N·s, ν)`` are the same buffer
        reshaped, so this remains the natural description of a plan whose
        steps share a stage count. It is refused when they do not; see
        :attr:`packed_shape`.
        """
        return (self.n_steps, self._uniform_stage_count("stage shape"),
                self.control_dim)

    @property
    def stage_abscissae(self) -> tuple[NDArray, ...] | None:
        """The abscissae this map samples at, one array per step.

        ``None`` declares that the map does not sample at abscissae at all,
        as :class:`PiecewiseConstantControl` does not: its stage controls are
        the same value whatever ``c`` is, so there is nothing for a plan to
        disagree with. That is a declaration, not a missing answer.

        A map that *does* use abscissae must report the ones it used, so that
        :class:`~adjungo.optimization.interface.GLMOptimizer` can check them
        against the plan's own. Before C-18 there was one tableau and the
        agreement was structural; a plan may now change method between steps,
        and a map built against one step's abscissae samples every other step
        at the wrong instants while producing an array of exactly the right
        shape.
        """
        return None

    @property
    def n_stage_controls(self) -> int:
        """Total number of stage controls, ``ν·Σ_n s_n``."""
        return self.total_stages * self.control_dim

    @property
    @abstractmethod
    def parameter_shape(self) -> tuple[int, ...]:
        """Natural (unflattened) shape of the parameter array."""

    @property
    def n_parameters(self) -> int:
        """Total number of parameters, the length of the flat ``θ``."""
        return int(np.prod(self.parameter_shape))

    @abstractmethod
    def _expand_packed(self, theta: NDArray) -> NDArray:
        """Map parameters to packed stage controls, shape ``(Σ s_n, ν)``."""

    @abstractmethod
    def _pullback_packed(self, g: NDArray) -> NDArray:
        """Apply the transpose to a packed stage-control covector."""

    def expand(self, theta: NDArray) -> NDArray:
        """Map parameters to stage controls.

        Returns ``(N, s, ν)`` when every step carries the same stage count,
        and the packed ``(Σ_n s_n, ν)`` when they differ. The two are the
        same buffer reshaped, so this is a presentation choice, not a second
        code path.
        """
        return self._to_natural_layout(self._expand_packed(theta))

    def push(self, v: NDArray) -> NDArray:
        """Apply the linear part to a parameter-space direction: ``P v``."""
        return self.expand(v)

    def pullback(self, g: NDArray) -> NDArray:
        """Apply the transpose to a stage-control covector: ``Pᵀ g``."""
        return self._pullback_packed(self._check_stage_shape(g))

    def _to_natural_layout(self, packed: NDArray) -> NDArray:
        first = self.stage_counts[0]
        if any(s != first for s in self.stage_counts):
            return packed
        return packed.reshape(self.n_steps, first, self.control_dim)

    def _blocks(self, packed: NDArray) -> list[NDArray]:
        """Views of ``packed``, one per step."""
        off = self.stage_offsets
        return [packed[off[n]:off[n + 1]] for n in range(self.n_steps)]

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
                f"size {self.n_stage_controls} (packed shape "
                f"{self.packed_shape}); got size {arr.size} with shape "
                f"{arr.shape}."
            )
        return arr.reshape(self.packed_shape)

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

    def _expand_packed(self, theta: NDArray) -> NDArray:
        arr = self._check_parameter_shape(theta)
        return np.repeat(arr, self.stage_counts, axis=0)

    def _pullback_packed(self, g: NDArray) -> NDArray:
        return np.add.reduceat(g, self.stage_offsets[:-1], axis=0)


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

    The abscissae are captured at construction and are **not** re-read from
    the methods later: the interpolation weights and the stage times the
    solver evaluates must come from the same tableau, or the control seen by
    ``f`` is sampled at different instants than the parametrization believes.

    Since C-18 a plan may carry a different method at each step, so ``c`` is
    per step. Passing a single abscissa vector still means "these abscissae
    at every step", which is correct for a single-method plan and wrong for
    any other; :meth:`from_plan` takes them from the plan itself and is the
    way to get this right without restating it. A map built against one
    step's abscissae is not caught by shape alone -- ``explicit_euler`` and
    ``implicit_midpoint`` both have ``s = 1`` -- so
    :class:`~adjungo.optimization.interface.GLMOptimizer` compares
    :attr:`stage_abscissae` against the plan's, element for element. Measured
    on ``y' = u``, ``J = y(1)²/2``, nodes ``(0, ½, 1)``, Euler then midpoint,
    ``θ = (0, ½, 1)``: sampling both steps at Euler's ``c = 0`` gives stage
    controls ``(0, ½)`` and ``J = 0.03125`` where the plan's own abscissae
    give ``(0, ¾)`` and ``J = 0.0703125``.

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
        c: NDArray | Sequence[NDArray],
    ) -> None:
        arrays = [np.asarray(row, dtype=float).ravel() for row in _as_per_step(
            c, n_steps
        )]
        super().__init__(n_steps, [row.size for row in arrays], control_dim)

        for step, row in enumerate(arrays):
            if not np.all(np.isfinite(row)):
                raise ValueError(
                    f"stage abscissae must all be finite; step {step} has "
                    f"c = {row}"
                )
            row.flags.writeable = False

        self._c = tuple(arrays)
        #: Interpolation weights per step, shape ``(s_n, 2)``: left, right.
        self._w = tuple(
            np.stack([1.0 - row, row], axis=1) for row in arrays
        )

    @classmethod
    def from_plan(
        cls, plan: DiscretizationPlan, control_dim: int
    ) -> NodalControl:
        """Interpolate to the abscissae each step's own method declares."""
        return cls(
            n_steps=plan.N,
            control_dim=control_dim,
            c=[method.c for method in plan.methods],
        )

    @property
    def c(self) -> NDArray:
        """The abscissae, when every step samples the same ones.

        Refused when they differ: a single ``c`` would name one step's
        abscissae as the whole map's (C-7). Use :attr:`stage_abscissae`.
        """
        first = self._c[0]
        if any(
            row.shape != first.shape or not np.array_equal(row, first)
            for row in self._c
        ):
            raise NotImplementedError(
                "NodalControl has no single abscissa vector: its steps "
                "sample at different abscissae. Use stage_abscissae."
            )
        return first

    @property
    def stage_abscissae(self) -> tuple[NDArray, ...]:
        return self._c

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (self.n_steps + 1, self.control_dim)

    def _expand_packed(self, theta: NDArray) -> NDArray:
        arr = self._check_parameter_shape(theta)
        out = np.empty(self.packed_shape)
        for n, block in enumerate(self._blocks(out)):
            w = self._w[n]
            block[...] = (
                w[:, 0, None] * arr[n][None, :]
                + w[:, 1, None] * arr[n + 1][None, :]
            )
        return out

    def _pullback_packed(self, g: NDArray) -> NDArray:
        out = np.zeros(self.parameter_shape)
        # Each interior node receives from the step on its left and the step
        # on its right. The two accumulations overlap on interior nodes, so
        # those collect both contributions.
        for n, block in enumerate(self._blocks(g)):
            w = self._w[n]
            out[n] += w[:, 0] @ block
            out[n + 1] += w[:, 1] @ block
        return out
