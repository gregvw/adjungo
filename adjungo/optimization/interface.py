"""Optimization interface for external optimizers."""

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from adjungo.core.affine import (
    affine_dynamics_verified,
    require_immutable_coefficients,
)
from adjungo.core.method import GLMethod, StageType
from adjungo.core.objective import Objective
from adjungo.core.plan import DiscretizationPlan
from adjungo.core.problem import Linearity, Problem, ProblemStructure
from adjungo.core.requirements import (
    SolverRequirements,
    deduce_requirements,
)
from adjungo.optimization.gradient import assemble_gradient
from adjungo.optimization.hessian import assemble_hessian_vector_product
from adjungo.optimization.parametrization import (
    AffineControlParametrization,
)
from adjungo.solvers.base import StageSolver
from adjungo.solvers.factory import create_stage_solver
from adjungo.stepping.adjoint import AdjointTrajectory, adjoint_solve
from adjungo.stepping.forward import forward_solve
from adjungo.stepping.sensitivity import (
    adjoint_sensitivity,
    forward_sensitivity,
)
from adjungo.stepping.trajectory import Trajectory

_T = TypeVar("_T")

#: Method families certified by a completed milestone (NUMERICS.md C-6.1).
#: Adding an entry here is a certification claim and requires the milestone's
#: acceptance evidence.
CERTIFIED_STAGE_TYPES = frozenset(
    {
        StageType.EXPLICIT,
        StageType.DIRK,
        StageType.SDIRK,
        StageType.IMPLICIT,
    }
)


def _enforce_envelope(plan: DiscretizationPlan) -> None:
    """Refuse unsupported configurations at construction (C-6.2, C-12).

    These are hard guards with no override. Proceeding into an uncertified path
    and reporting a plausible-looking number is the failure mode this contract
    exists to prevent.

    Every method in the plan is checked, not just the first. A plan is a
    sequence of independent method choices (C-18.1), so a single uncertified
    step anywhere in it puts the whole solve outside the envelope.

    Mesh validity and ``r > 1`` are **not** checked here. A plan cannot be
    built that violates them: :class:`DiscretizationPlan` refuses a
    degenerate mesh and a multistep method in its own constructor, which is
    the earlier and more complete boundary because it also covers callers
    who reach ``forward_solve`` without an optimizer. Repeating those checks
    here would add branches no test can reach.
    """
    for step, method in enumerate(plan.methods):
        where = "" if plan.N == 1 else f" at step {step}"
        if method.stage_type not in CERTIFIED_STAGE_TYPES:
            raise NotImplementedError(
                f"Method family {method.stage_type.name} is not certified"
                f"{where} (NUMERICS.md C-6.1). A dense stage matrix A requires "
                f"a coupled Newton solve, scheduled for milestone M3. "
                f"Certified families: "
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
        method: GLMethod | None = None,
        t_span: tuple[float, float] | None = None,
        N: int | None = None,
        y0: NDArray | None = None,
        problem_structure: ProblemStructure | None = None,
        *,
        plan: DiscretizationPlan | None = None,
    ):
        """
        Initialize GLM optimizer.

        The discretization is either given directly as a
        :class:`DiscretizationPlan`, or built from the uniform single-method
        arguments, which remain the convenience form for the common case
        (C-18.1). Supplying both is refused rather than resolved by
        precedence: a caller who passes a plan *and* an ``N`` has two
        different meshes in mind, and silently honouring one of them would
        return the exact derivatives of a discrete objective the caller did
        not ask for.

        Args:
            problem: Problem specification
            objective: Objective function
            method: GLM tableau (uniform form)
            t_span: Time interval (t0, tf) (uniform form)
            N: Number of time steps (uniform form)
            y0: Initial state
            problem_structure: Optional problem structure (deduced if not provided)
            plan: Explicit discretization plan, in place of (method, t_span, N)
        """
        uniform_args = (method, t_span, N)
        if plan is None:
            missing = [
                name
                for name, value in zip(
                    ("method", "t_span", "N"), uniform_args, strict=True
                )
                if value is None
            ]
            if missing:
                raise TypeError(
                    f"GLMOptimizer needs either plan= or all of method, "
                    f"t_span, N; missing {', '.join(missing)}."
                )
            assert method is not None and t_span is not None and N is not None
            plan = DiscretizationPlan.uniform(t_span, N, method)
        elif any(value is not None for value in uniform_args):
            supplied = [
                name
                for name, value in zip(
                    ("method", "t_span", "N"), uniform_args, strict=True
                )
                if value is not None
            ]
            raise TypeError(
                f"GLMOptimizer takes either plan= or (method, t_span, N), not "
                f"both; got plan= together with {', '.join(supplied)}."
            )
        if y0 is None:
            raise TypeError("GLMOptimizer requires y0.")

        self.problem = problem
        self.objective = objective
        self._plan = plan
        self.y0 = y0

        _enforce_envelope(plan)

        # Coefficient validity is a property of the problem, not of the route,
        # so it is answered here rather than inside the deduction below. A
        # caller who supplies an explicit ProblemStructure never reaches
        # affine_dynamics_verified, and a rewritable coefficient displaces the
        # general route by exactly the amount it displaces the affine one
        # (C-15.7) -- so on that path the wrong gradient was returned in
        # silence. Measured at 0.0770625 against an exact 0.6781500. Repeated
        # in forward_solve, which is the boundary the invariant is really
        # about; this one refuses before any work is done rather than at the
        # first step of the first solve.
        require_immutable_coefficients(problem)

        # Deduce problem structure if not provided
        if problem_structure is None:
            problem_structure = self._deduce_problem_structure()
        #: Retained because the second-order path reads it: a problem whose
        #: dynamics are affine by construction has identically zero
        #: dynamics-curvature blocks, and the terms they would contribute are
        #: skipped rather than computed as additions of zero.
        self.problem_structure = problem_structure

        # One route per step. The route is decided by the step's method
        # (C-18.1), so a plan that changes method mid-solve changes route with
        # it. Distinct methods are keyed by identity rather than by value:
        # GLMethod holds ndarrays, so ``==`` is not a usable key, and two
        # equal-valued tableaux getting separate solvers costs only
        # construction.
        y_scale = max(float(np.max(np.abs(self.y0))), 1.0)
        requirements_by_method: dict[int, SolverRequirements] = {}
        solvers_by_method: dict[int, StageSolver] = {}
        for step_method in plan.methods:
            key = id(step_method)
            if key in solvers_by_method:
                continue
            requirements = deduce_requirements(
                step_method, problem_structure, problem.state_dim
            )
            requirements_by_method[key] = requirements
            solvers_by_method[key] = create_stage_solver(
                step_method,
                requirements,
                problem_structure,
                y_scale=y_scale,
            )
        self.step_requirements = tuple(
            requirements_by_method[id(m)] for m in plan.methods
        )
        #: One stage solver per step; shared objects when a method repeats.
        self.stage_solvers = tuple(
            solvers_by_method[id(m)] for m in plan.methods
        )

        # Cached trajectory (invalidated when u changes)
        self._trajectory: Trajectory | None = None
        self._adjoint: AdjointTrajectory | None = None
        self._u_cached: NDArray | None = None

    @property
    def plan(self) -> DiscretizationPlan:
        """The discretization this optimizer differentiates.

        Read-only. C-18.2 requires the backward calculation to differentiate
        the objective the forward sweep executed, and almost everything this
        object holds is derived from the plan: the cached trajectory and
        adjoint, the per-step stage solvers and their C-15.1 factorization
        stores, and the per-step ``SolverRequirements``. Rebinding the
        attribute changes none of them.

        Measured before this was refused, on ``y' = u``, ``J = y(1)²/2`` with
        two Euler steps and stage controls ``(1, 3)``: moving the interior
        node from ``½`` to ``¼`` and re-evaluating at the *same* controls
        returned the previous plan's ``J = 2.0`` instead of ``3.125``,
        because the cache compares controls only. The gradient and
        Hessian-vector product were stale with it.

        Use :meth:`with_plan` to change discretization. Rebuilding is the
        honest cost: a setter would have to rebuild the solvers and discard
        every cache, which is the constructor.
        """
        return self._plan

    @plan.setter
    def plan(self, value: DiscretizationPlan) -> None:
        raise AttributeError(
            "GLMOptimizer.plan is read-only. Rebinding it would leave the "
            "cached trajectory, adjoint, stage solvers and factorization "
            "stores describing the previous discretization, and those are "
            "keyed on the controls alone -- a re-evaluation at unchanged "
            "controls would return the old plan's answer. Use "
            "optimizer.with_plan(new_plan) (NUMERICS.md C-18.2)."
        )

    def with_plan(
        self, plan: DiscretizationPlan, **overrides: object
    ) -> "GLMOptimizer":
        """A new optimizer for ``plan``, over the same problem and objective.

        This is the supported way to re-discretize: an adaptation policy
        producing ``𝒟_k`` from the current trajectory hands the new plan
        here and gets an object with no state carried over from ``𝒟_{k-1}``.
        Control *parameters* are the caller's to transfer; stage controls are
        not transferable in general, since a new plan may have a different
        number of them.
        """
        kwargs: dict[str, object] = {
            "problem": self.problem,
            "objective": self.objective,
            "y0": self.y0,
            "problem_structure": self.problem_structure,
        }
        kwargs.update(overrides)
        return type(self)(plan=plan, **kwargs)  # type: ignore[arg-type]

    @property
    def N(self) -> int:
        """Number of steps in the plan."""
        return self.plan.N

    @property
    def t_span(self) -> tuple[float, float]:
        """``(t_0, t_N)`` -- the plan's first and last nodes."""
        return self.plan.t_span

    def _single(self, values: tuple[_T, ...], what: str) -> _T:
        """The one distinct entry of a per-step tuple, or a refusal.

        Returning the first step's value for a plan that does not have one
        value would name a property of step 0 after the whole solve (C-7).
        """
        distinct = {id(v): v for v in values}
        if len(distinct) != 1:
            raise ValueError(
                f"this optimizer's plan does not have a single {what}: it "
                f"varies across the {self.plan.N} steps. Index the per-step "
                f"sequence instead (NUMERICS.md C-18.1)."
            )
        return next(iter(distinct.values()))

    @property
    def method(self) -> GLMethod:
        """The method, when the plan executes exactly one."""
        return self._single(self.plan.methods, "method")

    @property
    def h(self) -> float:
        """The step size, when every step shares it.

        Equality is exact, and ``DiscretizationPlan.uniform`` gives every
        step the identical ``h`` so that this succeeds. A caller that wants
        a representative size for a genuinely unequal mesh is asking a
        different question and should read ``plan.h``.
        """
        h = self.plan.h
        if h.size > 1 and not bool(np.all(h == h[0])):
            raise ValueError(
                f"this optimizer's plan does not have a single step size: "
                f"h ranges over [{h.min()!r}, {h.max()!r}]. Read plan.h, or "
                f"plan.step_size(n) (NUMERICS.md C-18.1)."
            )
        return float(h[0])

    @property
    def stage_solver(self) -> StageSolver:
        """The stage solver, when the plan uses exactly one."""
        return self._single(self.stage_solvers, "stage solver")

    @property
    def requirements(self) -> SolverRequirements:
        """The solver requirements, when the plan uses exactly one method."""
        return self._single(self.step_requirements, "set of requirements")

    def _pack(self, u: NDArray, what: str = "control") -> NDArray:
        """Accept a rectangular ``(N, s, nu)`` or packed ``(Σ s_n, nu)`` array.

        Stage-indexed storage is packed (C-18.5). A plan with one stage count
        throughout can also be *viewed* rectangularly, and that is the shape
        every caller predating the plan uses, so it is accepted and reshaped
        at the boundary rather than supported by a second code path inside.
        """
        arr = np.asarray(u, dtype=float)
        nu = self.problem.control_dim
        packed = (self.plan.total_stages, nu)
        if arr.ndim == 3:
            s = self.plan.uniform_stage_count
            if s is None:
                raise ValueError(
                    f"{what} was given as {arr.shape}, but this plan's stage "
                    f"count varies across steps, so no rectangular shape "
                    f"describes it. Supply the packed shape {packed} "
                    f"(NUMERICS.md C-18.5)."
                )
            expected = (self.plan.N, s, nu)
            if arr.shape != expected:
                raise ValueError(
                    f"{what} has shape {arr.shape}, expected {expected}."
                )
            return arr.reshape(packed)
        if arr.ndim == 2:
            if arr.shape != packed:
                raise ValueError(
                    f"{what} has shape {arr.shape}, expected {packed}."
                )
            return arr
        raise ValueError(
            f"{what} must be a packed ({packed[0]}, {nu}) array or, for a "
            f"plan with one stage count, an (N, s, {nu}) array; got "
            f"{arr.ndim} dimensions."
        )

    def _unpack_like(self, packed: NDArray, like: NDArray) -> NDArray:
        """Return ``packed`` in the shape the caller's ``like`` array used."""
        return packed.reshape(np.shape(like)) if np.ndim(like) == 3 else packed

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
        - ``jacobian_constant`` is **not** deduced from ``linearity``. A
          linearity declaration classifies how ``F`` depends on the state and
          the control, not on time, so it cannot stand in for the C-15.1
          declaration that licenses reuse. That declaration is made by
          passing ``ProblemStructure(jacobian_constant=True)``; the only
          deduced source is a verified affine class (C-17.2).
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

        # Affineness is established by construction, never by declaration: the
        # problem must *be* an unmodified affine root class, which computes f,
        # F and G from coefficients it owns or from callables it invokes with
        # t alone. There is no cheap exact check that an opaque callback has
        # zero curvature -- sampling f would be a probe, and generalising a
        # probe from visited to unvisited points is precedent R-9 -- so no flag
        # is offered for it. See adjungo.core.affine.
        #
        # Zero curvature and a constant Jacobian are separate facts (C-16.1),
        # and the second does not follow from the first: AffineDynamics has
        # both, TimeVaryingAffineDynamics has only the first, because
        # F = M(t_i) differs between stages at distinct abscissae. Asserting
        # jacobian_constant unconditionally here would hand a time-varying
        # problem to the C-15 reuse path, where C-15.2's element-for-element
        # comparison would refuse it -- correctly, but only after the route had
        # already been chosen wrongly. The fact is read from the verified
        # class, so it is no more a declaration than the affineness is.
        if affine_dynamics_verified(self.problem):
            return ProblemStructure(
                linearity=Linearity.LINEAR,
                jacobian_constant=self.problem.coefficients_constant,
                jacobian_control_dependent=False,
                has_second_derivatives=True,
                state_affine=True,
                jointly_affine=True,
            )

        # A constant Jacobian is never deduced from a linearity declaration.
        # LINEAR says F is independent of y and u; it says nothing about t,
        # and F = M(t) satisfies it (C-16.1). Reading it as constancy made the
        # C-15.1 declaration on the caller's behalf, and a time-varying LINEAR
        # problem was routed to reuse and then refused by the C-15.2 guard.
        # Constancy reaches reuse only as an explicit
        # ProblemStructure(jacobian_constant=True) or from a verified affine
        # class's coefficients_constant (C-17.2), handled above.
        jacobian_constant = False
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
        """``J(u)`` -- runs the forward solve if needed.

        ``Objective.evaluate`` is specified on the rectangular ``(N, s, ν)``
        stage-control layout, and a plan whose stage count varies across
        steps has no such layout (C-18.5). On such a plan the objective is
        asked for :meth:`~adjungo.core.objective.PackedObjective.\
evaluate_packed` instead, which takes the packed ``(Σ_n s_n, ν)`` array and
        reads each step's block through ``trajectory.plan.stages``.

        An objective that implements neither is **refused**, not handed the
        packed array in place of the rectangular one. That substitution would
        not fail: the objective would index stage ``(n, k)`` out of whatever
        step happens to lie at packed row ``n`` and return a number (C-7).

        Derivatives never needed this. ``dJ_du`` and ``d2J_du2`` are called
        one stage at a time with ``(step, stage)`` alongside, so they carry
        no layout assumption, and ``gradient`` and ``hessian_vector_product``
        work on any plan C-18.1 admits.

        Args:
            u: Stage controls, rectangular ``(N, s, ν)`` or packed
                ``(Σ_n s_n, ν)``

        Returns:
            Objective value
        """
        self._ensure_forward(u)
        assert self._trajectory is not None
        packed = self._pack(u)

        s = self.plan.uniform_stage_count
        if s is not None:
            return self.objective.evaluate(
                self._trajectory,
                packed.reshape(self.N, s, self.problem.control_dim),
            )

        evaluate_packed = getattr(self.objective, "evaluate_packed", None)
        if evaluate_packed is None:
            raise NotImplementedError(
                f"{type(self.objective).__name__} does not implement "
                f"evaluate_packed, so it has no value on a plan whose stage "
                f"count varies across steps (stage counts "
                f"{tuple(m.s for m in self.plan.methods)}). Objective."
                f"evaluate takes the rectangular (N, s, nu) layout, which "
                f"does not describe this plan, and the packed array is not "
                f"substituted for it because an objective would index it as "
                f"if it were rectangular and return a number. Gradients and "
                f"Hessian-vector products need nothing added: their "
                f"objective callbacks are per-stage (NUMERICS.md C-7, "
                f"C-18.5, C-18.7)."
            )
        return float(evaluate_packed(self._trajectory, packed))

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
            plan=self.plan,
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
        grad = assemble_gradient(
            self._trajectory,
            self._adjoint,
            self._pack(u),
            self.objective,
            self.problem,
        )
        return self._unpack_like(grad, u)

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

        u_packed = self._pack(u)
        v_packed = self._pack(v, "direction")

        # Forward sensitivity: δy, δZ from δu = v
        sensitivity = forward_sensitivity(
            self._trajectory, v_packed, self.stage_solvers, self.problem
        )

        # Backward adjoint sensitivity: δλ, δμ
        adj_sensitivity = adjoint_sensitivity(
            self._trajectory,
            self._adjoint,
            sensitivity,
            u_packed,
            v_packed,
            self.stage_solvers,
            self.problem,
            self.objective,
            structure=self.problem_structure,
        )

        hvp = assemble_hessian_vector_product(
            self._trajectory,
            self._adjoint,
            sensitivity,
            adj_sensitivity,
            u_packed,
            v_packed,
            self.objective,
            self.problem,
            structure=self.problem_structure,
        )
        return self._unpack_like(hvp, u)

    def scipy_interface(
        self,
        parametrization: AffineControlParametrization | None = None,
    ) -> tuple[Callable, Callable]:
        """Return ``(fun, jac)`` for :func:`scipy.optimize.minimize`.

        Args:
            parametrization: Optional C-10.1 adapter layer. When ``None``
                (the default) the optimisation variables are the stage
                controls themselves, flattened in ``ravel`` order. When
                given, the variables are that map's parameters ``θ``, and
                the returned gradient is ``Pᵀ g_u``.

        Returns:
            ``fun`` evaluating the objective at a flat variable vector, and
            ``jac`` returning the gradient in the same flat coordinates.

        The gradient is the **coordinate** derivative in the flat variable
        vector, per C-10.4 -- the quantity ``∂J/∂x_i`` for which
        ``J(x + δ) ≈ J(x) + Σ_i (∂J/∂x_i) δ_i``. It is not a Riesz
        representative under an ``h``-weighted inner product, which would be
        natural for this problem class and would cause
        ``scipy.optimize.minimize`` to take silently wrong steps rather than
        to raise.

        See also:
            :meth:`scipy_hessp` for the Hessian-vector callable accepted by
            ``trust-ncg``, ``trust-krylov`` and ``Newton-CG``. Pass it the
            same ``parametrization``, or the two operators describe
            different variables.
        """
        expand, pullback = self._coordinate_maps(parametrization)

        def fun(x_flat: NDArray) -> float:
            return self.objective_value(expand(x_flat))

        def jac(x_flat: NDArray) -> NDArray:
            return pullback(self.gradient(expand(x_flat))).ravel()

        return fun, jac

    def _coordinate_maps(
        self,
        parametrization: AffineControlParametrization | None,
    ) -> tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
        """Resolve a parametrization into ``(expand, pullback)`` callables.

        Both are captured once, here, and the returned closures consume only
        what is captured. A caller that mutates the parametrization object
        afterwards does not change the operator SciPy is already driving,
        which is the same ownership rule the stage solvers follow for
        ``y_scale``.
        """
        shape = (self.plan.total_stages, self.problem.control_dim)

        if parametrization is None:
            def expand(x_flat: NDArray) -> NDArray:
                return np.asarray(x_flat, dtype=float).reshape(shape)

            def pullback(g: NDArray) -> NDArray:
                return g

            return expand, pullback

        self._check_parametrization(parametrization)
        p = parametrization
        return p.expand, p.pullback

    def _check_parametrization(
        self, parametrization: AffineControlParametrization
    ) -> None:
        """Refuse a parametrization built for a different discretisation.

        Three things must agree, and shape alone establishes none of them.
        A map whose stage-control size disagrees would otherwise fail deep
        inside a reshape, or -- worse, when the sizes happen to coincide --
        succeed while silently permuting the controls.

        The abscissae are the addition C-18 forces. Before it there was one
        tableau for the whole solve, so a map holding "the" abscissae could
        not disagree with the method being integrated. A plan may now change
        method between steps at an unchanged stage count -- ``explicit_euler``
        and ``implicit_midpoint`` both have ``s = 1`` -- so a map built
        against one step's ``c`` produces a correctly shaped array sampled at
        the wrong instants. That is C-10.4's coordinate convention silently
        violated, not a representable alternative: measured on ``y' = u``,
        ``J = y(1)²/2`` with nodes ``(0, ½, 1)``, Euler then midpoint, and
        ``θ = (0, ½, 1)``, it returned ``J = 0.03125`` against the plan's own
        ``0.0703125``, with a correspondingly displaced gradient.

        A map reporting :attr:`~adjungo.optimization.parametrization.\
AffineControlParametrization.stage_abscissae` of ``None`` declares that it
        does not sample at abscissae at all, so there is nothing to compare;
        that is the piecewise-constant case.
        """
        p = parametrization
        nu = self.problem.control_dim
        plan_counts = tuple(m.s for m in self.plan.methods)
        if (
            p.n_steps != self.plan.N
            or p.stage_counts != plan_counts
            or p.control_dim != nu
        ):
            raise ValueError(
                f"{type(p).__name__} produces stage controls for "
                f"{p.n_steps} steps with stage counts {p.stage_counts} and "
                f"control_dim {p.control_dim}, but this optimizer integrates "
                f"{self.plan.N} steps with stage counts {plan_counts} and "
                f"control_dim {nu}."
            )

        declared = p.stage_abscissae
        if declared is None:
            return
        for step, (c_map, method) in enumerate(
            zip(declared, self.plan.methods, strict=True)
        ):
            # Element for element, as in C-15.1's factorization comparison.
            # A tolerance here would accept a map sampling at instants the
            # solver never evaluates, which is a different discrete problem
            # rather than a rounding difference.
            if not np.array_equal(c_map, method.c):
                raise ValueError(
                    f"{type(p).__name__} samples step {step} at abscissae "
                    f"{np.asarray(c_map)}, but that step's method evaluates "
                    f"its stages at {np.asarray(method.c)}. The control "
                    f"would be sampled at instants the solver never visits "
                    f"(NUMERICS.md C-10.4, C-18.4). NodalControl.from_plan "
                    f"takes the abscissae from the plan."
                )

    def scipy_hessp(
        self,
        parametrization: AffineControlParametrization | None = None,
    ) -> Callable[[NDArray, NDArray], NDArray]:
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

        Args:
            parametrization: Optional C-10.1 adapter layer, which must be
                the same one passed to :meth:`scipy_interface`. The operator
                becomes ``H_θ v = Pᵀ H_u (P v)`` per C-10.2.

        Raises:
            NotImplementedError: If the objective or problem does not supply
                the second derivatives the exact Hessian requires. The
                operator is never silently replaced by a Gauss-Newton
                approximation; see NUMERICS.md C-7.
        """
        expand, pullback = self._coordinate_maps(parametrization)

        if parametrization is None:
            push = expand
        else:
            push = parametrization.push

        def hessp(x_flat: NDArray, v_flat: NDArray) -> NDArray:
            x = expand(x_flat)
            # The direction is pushed with the *linear* part only. Using
            # ``expand`` here would add the affine offset ``q`` to a
            # direction, which is not a direction. Both shipped maps have
            # ``q = 0``, so the error would be dormant until the first map
            # with an offset -- hence the separate operation rather than a
            # reuse that happens to work today.
            v = push(np.asarray(v_flat, dtype=float))
            return pullback(self.hessian_vector_product(x, v)).ravel()

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
                self._pack(u),
                self.plan,
                self.problem,
                self.stage_solvers,
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
                self._trajectory, self.objective, self.stage_solvers
            )
