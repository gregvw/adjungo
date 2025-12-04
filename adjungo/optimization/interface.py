"""Optimization interface for external optimizers."""

from typing import Optional, Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from adjungo.core.problem import Problem, ProblemStructure, Linearity
from adjungo.core.objective import Objective
from adjungo.core.method import GLMethod
from adjungo.core.requirements import deduce_requirements
from adjungo.solvers.factory import create_stage_solver
from adjungo.stepping.trajectory import Trajectory
from adjungo.stepping.forward import forward_solve
from adjungo.stepping.adjoint import AdjointTrajectory, adjoint_solve
from adjungo.stepping.sensitivity import (
    forward_sensitivity,
    adjoint_sensitivity,
)
from adjungo.optimization.gradient import assemble_gradient
from adjungo.optimization.hessian import assemble_hessian_vector_product


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
        problem_structure: Optional[ProblemStructure] = None,
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
        self.h = (t_span[1] - t_span[0]) / N

        # Deduce problem structure if not provided
        if problem_structure is None:
            problem_structure = self._deduce_problem_structure()

        # Deduce requirements and create appropriate solver
        self.requirements = deduce_requirements(
            method, problem_structure, problem.state_dim
        )
        self.stage_solver = create_stage_solver(
            method, self.requirements, problem_structure
        )

        # Cached trajectory (invalidated when u changes)
        self._trajectory: Optional[Trajectory] = None
        self._adjoint: Optional[AdjointTrajectory] = None
        self._u_hash: Optional[int] = None

    def _deduce_problem_structure(self) -> ProblemStructure:
        """Deduce problem structure from problem specification."""
        # Simplified deduction - user should provide for better performance
        return ProblemStructure(
            linearity=Linearity.NONLINEAR,
            jacobian_constant=False,
            jacobian_control_dependent=True,
            has_second_derivatives=False,
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
            v,
            self.method,
            self.stage_solver,
            self.problem,
            self.h,
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
        )

    def scipy_interface(self) -> Tuple[Callable, Callable]:
        """
        Returns (fun, jac) for scipy.optimize.minimize.

        Returns:
            fun: Objective function taking flat array
            jac: Gradient function taking flat array
        """
        def fun(u_flat: NDArray) -> float:
            u = u_flat.reshape(self.N, self.method.s, self.problem.control_dim)
            return self.objective_value(u)

        def jac(u_flat: NDArray) -> NDArray:
            u = u_flat.reshape(self.N, self.method.s, self.problem.control_dim)
            return self.gradient(u).ravel()

        return fun, jac

    def _ensure_forward(self, u: NDArray) -> None:
        """Run forward solve if not cached or u changed."""
        u_hash = hash(u.tobytes())
        if self._trajectory is None or self._u_hash != u_hash:
            self._trajectory = forward_solve(
                self.y0,
                u,
                self.t_span,
                self.N,
                self.problem,
                self.method,
                self.stage_solver,
            )
            self._u_hash = u_hash
            self._adjoint = None  # Invalidate adjoint

    def _ensure_adjoint(self, u: NDArray) -> None:
        """Run forward and adjoint solve if not cached or u changed."""
        self._ensure_forward(u)
        if self._adjoint is None:
            assert self._trajectory is not None
            self._adjoint = adjoint_solve(
                self._trajectory, self.objective, self.method, self.stage_solver, self.h
            )
