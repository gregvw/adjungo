"""Optimization interface for external optimizers."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from adjungo.core.method import GLMethod, StageType
from adjungo.core.objective import Objective
from adjungo.core.problem import Linearity, Problem, ProblemStructure
from adjungo.core.requirements import deduce_requirements
from adjungo.optimization.gradient import assemble_gradient
from adjungo.optimization.hessian import assemble_hessian_vector_product
from adjungo.solvers.factory import create_stage_solver
from adjungo.stepping.adjoint import AdjointTrajectory, adjoint_solve
from adjungo.stepping.forward import forward_solve
from adjungo.stepping.sensitivity import (
    adjoint_sensitivity,
    forward_sensitivity,
)
from adjungo.stepping.trajectory import Trajectory

#: Method families certified by a completed milestone (NUMERICS.md C-6.1).
#: Adding an entry here is a certification claim and requires the milestone's
#: acceptance evidence.
CERTIFIED_STAGE_TYPES = frozenset(
    {StageType.EXPLICIT, StageType.DIRK, StageType.SDIRK}
)


def _enforce_envelope(
    method: GLMethod, t_span: tuple[float, float], N: int
) -> None:
    """Refuse unsupported configurations at construction (C-6.2, C-12).

    These are hard guards with no override. Proceeding into an uncertified path
    and reporting a plausible-looking number is the failure mode this contract
    exists to prevent.
    """
    if N < 1:
        raise ValueError(f"N must be at least 1, got {N} (NUMERICS.md C-12).")

    if t_span[1] == t_span[0]:
        raise ValueError(
            f"t_span must have nonzero extent, got {t_span} (NUMERICS.md C-12)."
        )

    if method.r > 1:
        raise NotImplementedError(
            f"Methods with r > 1 external stages are not supported (got "
            f"r={method.r}). Adjungo has no starting procedure for multistep "
            f"methods; see NUMERICS.md C-6.2 and open question C-Q4."
        )

    if method.stage_type not in CERTIFIED_STAGE_TYPES:
        raise NotImplementedError(
            f"Method family {method.stage_type.name} is not certified "
            f"(NUMERICS.md C-6.1). A dense stage matrix A requires a coupled "
            f"Newton solve, scheduled for milestone M3. Certified families: "
            f"{', '.join(sorted(t.name for t in CERTIFIED_STAGE_TYPES))}."
        )



class GLMOptimizer:
    """
    Provides J(u), ∇J(u), H(u)·v to outer optimizer.
    """

    def __init__(
        self,
        problem: Problem,
        objective: Objective,
        method: GLMethod,
        t_span: tuple[float, float],
        N: int,
        y0: NDArray,
        problem_structure: ProblemStructure | None = None,
    ):
        """
        Initialize GLM optimizer.

        Args:
            problem: Problem specification
            objective: Objective function
            method: GLM tableau
            t_span: Time interval (t0, tf)
            N: Number of time steps
            y0: Initial state
            problem_structure: Optional problem structure (deduced if not provided)
        """
        self.problem = problem
        self.objective = objective
        self.method = method
        self.t_span = t_span
        self.N = N
        self.y0 = y0

        _enforce_envelope(method, t_span, N)

        self.h = (t_span[1] - t_span[0]) / N

        # Deduce problem structure if not provided
        if problem_structure is None:
            problem_structure = self._deduce_problem_structure()

        # Deduce requirements and create appropriate solver
        self.requirements = deduce_requirements(
            method, problem_structure, problem.state_dim
        )
        self.stage_solver = create_stage_solver(
            method,
            self.requirements,
            problem_structure,
            y_scale=max(float(np.max(np.abs(self.y0))), 1.0),
        )

        # Cached trajectory (invalidated when u changes)
        self._trajectory: Trajectory | None = None
        self._adjoint: AdjointTrajectory | None = None
        self._u_cached: NDArray | None = None

    def _deduce_problem_structure(self) -> ProblemStructure:
        """Deduce problem structure from the problem specification.

        This previously returned a hard-coded ``NONLINEAR`` /
        ``has_second_derivatives=False`` regardless of the problem, so the flag
        described nothing and a problem supplying second derivatives was
        reported as not supplying them.

        What can honestly be deduced:

        - ``has_second_derivatives`` is detected by the presence of all three
          contracted-Hessian callbacks. Partial support is treated as absent,
          because the second-order path needs all three.
        - ``linearity`` cannot be deduced from an opaque callback protocol
          without evaluating it, so an undeclared problem is assumed
          ``NONLINEAR``. That is the conservative choice: it forces Jacobian
          re-evaluation and disables reuse. A problem that knows it is linear
          declares so via ``problem.linearity`` or by passing an explicit
          ``problem_structure``.
        """
        declared = getattr(self.problem, "linearity", None)
        linearity = declared if isinstance(declared, Linearity) else (
            Linearity.NONLINEAR
        )

        second_derivative_hooks = ("F_yy_action", "F_yu_action", "F_uu_action")
        has_second_derivatives = all(
            callable(getattr(self.problem, name, None))
            for name in second_derivative_hooks
        )

        # Only a problem that declares linearity may claim a constant Jacobian.
        jacobian_constant = linearity is Linearity.LINEAR
        jacobian_control_dependent = linearity not in (
            Linearity.LINEAR,
            Linearity.SEMILINEAR,
        )

        return ProblemStructure(
            linearity=linearity,
            jacobian_constant=jacobian_constant,
            jacobian_control_dependent=jacobian_control_dependent,
            has_second_derivatives=has_second_derivatives,
        )

    def objective_value(self, u: NDArray) -> float:
        """
        J(u) - runs forward solve if needed.

        Args:
            u: Control array (N, s, ν)

        Returns:
            Objective value
        """
        self._ensure_forward(u)
        assert self._trajectory is not None
        return self.objective.evaluate(self._trajectory, u)

    def trajectory(self, u: NDArray) -> Trajectory:
        """Return the forward trajectory at ``u``.

        Without this a caller can optimize but cannot inspect the state the
        optimal control produces, which makes the result unusable for
        anything but reporting the objective value.

        The returned ``Y`` and ``Z`` are **copies**. The optimizer caches the
        trajectory keyed on ``u``, so handing out the live arrays would let a
        caller mutate the state that every subsequent gradient and Hessian is
        computed from while the cache key still matched: the derivatives
        would then belong to a trajectory that no forward solve ever
        produced.

        ``caches`` is shared rather than copied. It holds LU factorization
        objects that are not meaningfully copyable, and it is an internal
        record of the solve rather than caller-facing data.

        Args:
            u: Control array (N, s, ν)

        Returns:
            The trajectory, with ``Y`` of shape ``(N+1, r, n)`` and ``Z`` of
            shape ``(N, s, n)``.
        """
        self._ensure_forward(u)
        assert self._trajectory is not None
        return Trajectory(
            Y=self._trajectory.Y.copy(),
            Z=self._trajectory.Z.copy(),
            caches=self._trajectory.caches,
        )

    def gradient(self, u: NDArray) -> NDArray:
        """
        ∇J(u) - runs forward + adjoint if needed.

        Args:
            u: Control array (N, s, ν)

        Returns:
            Gradient (N, s, ν)
        """
        self._ensure_adjoint(u)
        assert self._trajectory is not None
        assert self._adjoint is not None
        return assemble_gradient(
            self._trajectory,
            self._adjoint,
            u,
            self.objective,
            self.method,
            self.problem,
            self.h,
        )

    def hessian_vector_product(self, u: NDArray, v: NDArray) -> NDArray:
        """
        [∇²J(u)]v - full second-order computation.

        Args:
            u: Control array (N, s, ν)
            v: Direction vector (N, s, ν)

        Returns:
            Hessian-vector product (N, s, ν)
        """
        self._ensure_adjoint(u)
        assert self._trajectory is not None
        assert self._adjoint is not None

        # Forward sensitivity: δy, δZ from δu = v
        sensitivity = forward_sensitivity(
            self._trajectory, v, self.method, self.stage_solver, self.problem, self.h
        )

        # Backward adjoint sensitivity: δλ, δμ
        adj_sensitivity = adjoint_sensitivity(
            self._trajectory,
            self._adjoint,
            sensitivity,
            u,
            v,
            self.method,
            self.stage_solver,
            self.problem,
            self.h,
            self.t_span[0],
            self.objective,
        )

        return assemble_hessian_vector_product(
            self._trajectory,
            self._adjoint,
            sensitivity,
            adj_sensitivity,
            u,
            v,
            self.objective,
            self.method,
            self.problem,
            self.h,
            self.t_span[0],
        )

    def scipy_interface(self) -> tuple[Callable, Callable]:
        """
        Returns (fun, jac) for scipy.optimize.minimize.

        Returns:
            fun: Objective function taking flat array
            jac: Gradient function taking flat array

        See also:
            :meth:`scipy_hessp` for the Hessian-vector callable accepted by
            ``trust-ncg``, ``trust-krylov`` and ``Newton-CG``.
        """
        def fun(u_flat: NDArray) -> float:
            u = u_flat.reshape(self.N, self.method.s, self.problem.control_dim)
            return self.objective_value(u)

        def jac(u_flat: NDArray) -> NDArray:
            u = u_flat.reshape(self.N, self.method.s, self.problem.control_dim)
            return self.gradient(u).ravel()

        return fun, jac

    def scipy_hessp(self) -> Callable[[NDArray, NDArray], NDArray]:
        """Return ``hessp(x, p)`` for SciPy's Hessian-free Newton methods.

        SciPy passes the current point and the direction as separate flat
        arrays and expects a flat array back. Both are reshaped to the
        ``(N, s, ν)`` control layout, which is the only layout in which the
        Hessian operator is defined: ``ravel`` order is
        :ref:`C-10 <numerics>`'s coordinate convention and a caller that
        flattens differently gets a different operator.

        The direction is **not** cached. Only ``u`` selects the trajectory
        and adjoint; ``p`` is a free argument of the resulting linear
        operator, so repeated calls at fixed ``u`` with different ``p`` reuse
        the forward and adjoint solves. This is what makes a Krylov method
        affordable here: one nonlinear solve per outer iteration, one
        second-order adjoint sweep per inner product.

        Raises:
            NotImplementedError: If the objective or problem does not supply
                the second derivatives the exact Hessian requires. The
                operator is never silently replaced by a Gauss-Newton
                approximation; see NUMERICS.md C-7.
        """
        shape = (self.N, self.method.s, self.problem.control_dim)

        def hessp(u_flat: NDArray, p_flat: NDArray) -> NDArray:
            u = np.asarray(u_flat, dtype=float).reshape(shape)
            p = np.asarray(p_flat, dtype=float).reshape(shape)
            return self.hessian_vector_product(u, p).ravel()

        return hessp

    def _ensure_forward(self, u: NDArray) -> None:
        """Run forward solve if not cached or u changed.

        The cache key is a defensive copy of ``u`` compared with
        ``np.array_equal``, not a hash. A hash collision would silently return
        another point's trajectory, and every derivative check downstream would
        then be comparing quantities evaluated at different controls. Copying
        also means a caller mutating ``u`` in place afterwards cannot corrupt
        the key. The arrays are (N, s, nu) and small, so this is cheap; per C-1
        correctness outranks micro-optimization in a reference implementation.
        """
        if self._trajectory is None or not self._cache_matches(u):
            self._trajectory = forward_solve(
                self.y0,
                u,
                self.t_span,
                self.N,
                self.problem,
                self.method,
                self.stage_solver,
            )
            self._u_cached = u.copy()
            self._adjoint = None  # Invalidate adjoint

    def _cache_matches(self, u: NDArray) -> bool:
        """True when ``u`` is exactly the control the cache was built from."""
        cached = self._u_cached
        return (
            cached is not None
            and cached.shape == u.shape
            and bool(np.array_equal(cached, u))
        )

    def _ensure_adjoint(self, u: NDArray) -> None:
        """Run forward and adjoint solve if not cached or u changed."""
        self._ensure_forward(u)
        if self._adjoint is None:
            assert self._trajectory is not None
            self._adjoint = adjoint_solve(
                self._trajectory, self.objective, self.method, self.stage_solver, self.h
            )
