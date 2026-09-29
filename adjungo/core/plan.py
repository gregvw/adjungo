"""The discretization plan (NUMERICS.md C-18).

A plan is the time nodes together with a method for each step. It is caller
supplied data, frozen for the duration of a solve, and it is what the exactness
claim of C-2 is stated relative to: adjungo returns the exact derivatives of
``J_𝒟`` with ``𝒟`` held fixed.

The uniform single-method case is a plan like any other, built by
:meth:`DiscretizationPlan.uniform`. There is no second storage mode and no
second code path for it; see C-18.5.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from adjungo.core.method import GLMethod

#: Tolerance basis for :func:`_check_step_sizes_match_nodes`. ``nodes[n]`` and
#: ``nodes[n] + h_n`` are two floating-point values of the same real number,
#: each carrying the rounding of its own construction, and ``h_n`` itself may
#: be a rounded quotient that is multiplied up to ``N`` times to reach the far
#: node. The admissible disagreement therefore scales with
#: ``|t_n| + |t_{n+1}| + N|h_n|``. The constant 16 is a margin over the worst
#: observed ratio of 0.91, measured over ``t_0 in {0, 1, -3, 100}``, spans
#: ``{0.1, 1, 2, 5, 13.7}`` and ``N`` up to 1000 for the equally spaced
#: construction in :meth:`DiscretizationPlan.uniform`. It is a consistency
#: guard against unrelated arrays, not an accuracy claim.
_NODE_STEP_ULP_BUDGET = 16.0


def _check_step_sizes_match_nodes(nodes: NDArray, h: NDArray) -> None:
    """Refuse step sizes that do not describe the given nodes.

    Explicit ``step_sizes`` exist to hold an exactly uniform ``h`` that node
    differences cannot represent, not to let a caller pair a mesh with the
    step lengths of a different one. Accepting that would give the forward
    sweep one discretization and every error estimate built on ``nodes``
    another.
    """
    eps = float(np.finfo(float).eps)
    drift = np.abs(nodes[:-1] + h - nodes[1:])
    scale = (
        np.abs(nodes[:-1]) + np.abs(nodes[1:]) + nodes[:-1].size * np.abs(h)
    )
    budget = _NODE_STEP_ULP_BUDGET * eps * scale
    bad = np.flatnonzero(drift > budget)
    if bad.size:
        n = int(bad[0])
        raise ValueError(
            f"DiscretizationPlan.step_sizes[{n}] = {h[n]!r} does not carry "
            f"nodes[{n}] = {nodes[n]!r} to nodes[{n + 1}] = "
            f"{nodes[n + 1]!r}; the discrepancy {drift[n]!r} exceeds the "
            f"rounding budget {budget[n]!r} (NUMERICS.md C-18.1)."
        )


def _frozen_tableau(method: GLMethod) -> GLMethod:
    """A copy of ``method`` whose coefficient buffers cannot be written.

    A plan is the record of what was executed (C-18.2), and the trajectory it
    produced is interpreted against it. Retaining the caller's ``GLMethod``
    does not give that: ``GLMethod`` is a frozen dataclass, so ``m.B = ...``
    is refused, but its fields are ordinary writable arrays and ``m.B[0, 0] =
    2.0`` is not. That write reaches the plan through the alias and changes
    the discretization out from under a trajectory already recorded against
    it -- the same shape of failure as C-15.7's coefficient aliasing, reached
    through the tableau rather than through the problem. Measured on
    ``y' = u``, ``J = y(1)²/2`` with two Euler steps and controls ``(1, 3)``:
    writing ``B[0,0] = 2`` after the solve moved a recorded ``J = 2.0`` to
    ``6.125`` with no refusal anywhere.

    Copying at construction closes the ordinary aliasing path, which is the
    one an ordinary caller reaches. It is not a claim of immunity to a caller
    who sets ``writeable`` back to ``True``; that is outside the trust model
    recorded in C-15.7.
    """
    frozen = copy.copy(method)
    for name in ("A", "U", "B", "V", "c"):
        buffer = np.array(getattr(method, name), dtype=float)
        buffer.flags.writeable = False
        object.__setattr__(frozen, name, buffer)
    return frozen


@dataclass(frozen=True)
class DiscretizationPlan:
    """Time nodes and a per-step method.

    Args:
        nodes: ``(N+1,)`` strictly monotonic time nodes. Step ``n`` spans
            ``[nodes[n], nodes[n+1]]`` and carries ``h_n = nodes[n+1] -
            nodes[n]``, per C-18.1.
        methods: One :class:`GLMethod` per step, length ``N``.

    Monotonicity is required in either direction: a decreasing node sequence
    integrates backwards with negative ``h_n``, which the scalar-``h`` code
    this replaces also admitted. A repeated node is refused -- a zero-length
    step contributes nothing to the state and divides by zero in every error
    estimate built on it.
    """

    nodes: NDArray
    methods: tuple[GLMethod, ...]

    #: Optional explicit step lengths, one per step. When omitted these are
    #: ``np.diff(nodes)``, which is the definition and what a caller-supplied
    #: mesh uses. They are supplied explicitly for exactly one reason: an
    #: equally spaced node array does **not** have equal floating-point
    #: differences. ``np.diff(np.linspace(0, 1, 11))`` spans two values one
    #: ulp apart. The scalar-``h`` code this replaces used one exact
    #: ``h = (t_N - t_0)/N`` for every step, and the difference is not
    #: cosmetic: an implicit stage matrix ``I - h a_ii F`` built from a
    #: step size that varies in its last bit is a *different matrix* at every
    #: step, so the C-15.1 element-for-element reuse comparison fails and the
    #: solve silently refactorizes once per step. Measured on the certified
    #: population: 5 factorizations over 12 steps where 1 is predicted.
    #:
    #: ``t_n`` and ``h_n`` are therefore both part of the executed
    #: discretization, and for a uniform mesh they cannot both be exact.
    #: This is the same pair the scalar-``h`` code carried.
    step_sizes: NDArray | None = None

    #: Derived and cached at construction. Recomputing them per step would be
    #: cheap, but the point of C-18.2 is that both sweeps read *one* immutable
    #: description of what was executed.
    h: NDArray = field(init=False, repr=False)
    stage_offsets: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # A copy, always. ``asarray`` would hand back the caller's own array
        # for the common float64 input, and the freeze below would then make
        # *their* object read-only as a side effect of constructing a plan.
        nodes = np.array(self.nodes, dtype=float)
        if nodes.ndim != 1:
            raise ValueError(
                f"DiscretizationPlan.nodes must be one-dimensional, got shape "
                f"{nodes.shape} (NUMERICS.md C-18.1)."
            )
        if nodes.size < 2:
            raise ValueError(
                f"DiscretizationPlan needs at least two nodes (one step), got "
                f"{nodes.size} (NUMERICS.md C-12)."
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError(
                "DiscretizationPlan.nodes must all be finite (NUMERICS.md "
                "C-12)."
            )

        methods = tuple(self.methods)
        if len(methods) != nodes.size - 1:
            raise ValueError(
                f"DiscretizationPlan has {nodes.size - 1} steps but "
                f"{len(methods)} methods; one method per step is required "
                f"(NUMERICS.md C-18.1)."
            )
        for step, method in enumerate(methods):
            if not isinstance(method, GLMethod):
                raise TypeError(
                    f"DiscretizationPlan.methods[{step}] must be a GLMethod, "
                    f"got {type(method).__name__}."
                )
            if method.r != 1:
                raise NotImplementedError(
                    f"Methods with r > 1 external stages are not supported "
                    f"(got r={method.r} at step {step}). Adjungo has no "
                    f"starting procedure for multistep methods: they need a "
                    f"defined history representation and a certified starter "
                    f"(NUMERICS.md C-6.2, C-18.3, open question C-Q4)."
                )

        if self.step_sizes is None:
            h = np.diff(nodes)
        else:
            h = np.array(self.step_sizes, dtype=float)
            if h.shape != (nodes.size - 1,):
                raise ValueError(
                    f"DiscretizationPlan.step_sizes must have one entry per "
                    f"step ({nodes.size - 1}), got shape {h.shape}."
                )
            if not np.all(np.isfinite(h)):
                raise ValueError(
                    "DiscretizationPlan.step_sizes must all be finite "
                    "(NUMERICS.md C-12)."
                )
            _check_step_sizes_match_nodes(nodes, h)

        if np.any(h == 0.0):
            bad = int(np.flatnonzero(h == 0.0)[0])
            raise ValueError(
                f"every step must have nonzero extent; step {bad} has zero "
                f"length (nodes[{bad}] == nodes[{bad + 1}] == "
                f"{nodes[bad]!r}). Nodes must be strictly monotonic "
                f"(NUMERICS.md C-12)."
            )
        if not (np.all(h > 0.0) or np.all(h < 0.0)):
            raise ValueError(
                "DiscretizationPlan.nodes must be strictly monotonic; got a "
                "sequence that changes direction. Integrating forwards over "
                "some steps and backwards over others is not a mesh "
                "(NUMERICS.md C-18.1)."
            )

        counts = np.array([m.s for m in methods], dtype=np.intp)
        offsets = np.zeros(len(methods) + 1, dtype=np.intp)
        np.cumsum(counts, out=offsets[1:])

        # Keyed by identity, not by value. Two steps given the *same* method
        # object must keep sharing one snapshot: GLMOptimizer keys its stage
        # solvers -- and therefore its C-15.1 factorization store -- on
        # ``id(method)``, so handing out a separate copy per step would give
        # every step its own solver and silently refactorize once per step.
        # Two equal-valued but distinct objects already got separate solvers
        # before this snapshot existed, and still do.
        snapshots: dict[int, GLMethod] = {}
        frozen_methods = tuple(
            snapshots.setdefault(id(m), _frozen_tableau(m)) for m in methods
        )

        for array in (nodes, h, offsets):
            array.flags.writeable = False

        object.__setattr__(self, "step_sizes", None)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "methods", frozen_methods)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "stage_offsets", offsets)

    @classmethod
    def uniform(
        cls, t_span: tuple[float, float], N: int, method: GLMethod
    ) -> DiscretizationPlan:
        """The degenerate plan: ``N`` equal steps of one method.

        This is what every solve did before C-18, and it remains the common
        case. ``np.linspace`` is used rather than ``t0 + n*h`` so that the
        final node is exactly ``t_span[1]``.
        """
        if N < 1:
            raise ValueError(
                f"N must be at least 1, got {N} (NUMERICS.md C-12)."
            )
        t0, t1 = float(t_span[0]), float(t_span[1])
        h = (t1 - t0) / N
        nodes = t0 + np.arange(N + 1, dtype=float) * h
        # The final node is the caller's ``t1`` exactly. ``t0 + N*h`` need not
        # reproduce it, and a horizon that is not the one the caller asked for
        # is a silently different problem.
        nodes[-1] = t1
        return cls(
            nodes=nodes,
            methods=(method,) * N,
            step_sizes=np.full(N, h),
        )

    @property
    def N(self) -> int:
        """Number of steps."""
        return len(self.methods)

    @property
    def t_span(self) -> tuple[float, float]:
        """``(first node, last node)``."""
        return (float(self.nodes[0]), float(self.nodes[-1]))

    @property
    def total_stages(self) -> int:
        """``Σ_n s_n`` -- the packed length of a stage-indexed array."""
        return int(self.stage_offsets[-1])

    def method_at(self, step: int) -> GLMethod:
        """The method executing step ``step``."""
        return self.methods[step]

    def t(self, step: int) -> float:
        """``t_n``, the time at the start of step ``step``."""
        return float(self.nodes[step])

    def step_size(self, step: int) -> float:
        """``h_n = t_{n+1} - t_n``."""
        return float(self.h[step])

    def s(self, step: int) -> int:
        """Stage count of step ``step``."""
        return self.methods[step].s

    def stage_times(self, step: int) -> NDArray:
        """``t_n + c^(n)_j h_n`` for every stage ``j`` of step ``step``."""
        method = self.methods[step]
        times: NDArray = self.nodes[step] + method.c * self.h[step]
        return times

    def stages(self, packed: NDArray, step: int) -> NDArray:
        """The ``(s_n, ...)`` block of a packed stage array, as a **view**.

        Writing through the returned array writes into ``packed``. That is
        deliberate: the stepping code fills stage storage in place.
        """
        start = self.stage_offsets[step]
        return packed[start:start + self.methods[step].s]

    def pack(self, arr: NDArray, name: str = "array") -> NDArray:
        """Normalize a stage-indexed array to packed ``(Σ s_n, ...)`` form.

        A plan whose steps share a stage count may also present its stage
        storage rectangularly as ``(N, s, ...)``, which is the shape every
        caller predating C-18 uses. Accepting it here, once, keeps the
        packed layout an internal representation rather than a second code
        path (C-18.5). The reshape is a view for contiguous input.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim >= 3:
            s = self.uniform_stage_count
            if s is None:
                raise ValueError(
                    f"{name} was given as {a.shape}, but this plan's stage "
                    f"count varies across steps, so no rectangular shape "
                    f"describes it; supply "
                    f"({self.total_stages}, ...) (NUMERICS.md C-18.5)."
                )
            if a.shape[:2] != (self.N, s):
                raise ValueError(
                    f"{name} has shape {a.shape}, expected leading "
                    f"({self.N}, {s})."
                )
            return a.reshape(self.total_stages, *a.shape[2:])
        if a.ndim < 1 or a.shape[0] != self.total_stages:
            raise ValueError(
                f"{name} has shape {a.shape}, expected a packed array of "
                f"leading length {self.total_stages}."
            )
        return a

    @property
    def uniform_stage_count(self) -> int | None:
        """``s`` when every step shares it, else ``None``.

        This answers exactly one question: may an array packed as
        ``(total_stages, ...)`` also be *viewed* as ``(N, s, ...)``, for the
        callers and tests that predate the plan. It never selects a storage
        mode -- there is only one -- and it says nothing about whether the
        mesh is regular, which is a separate property that no caller needs.
        """
        counts = {m.s for m in self.methods}
        return counts.pop() if len(counts) == 1 else None
