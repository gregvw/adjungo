"""Tests for solver requirements deduction."""

import numpy as np
import pytest

from adjungo.core.requirements import deduce_requirements, SolverRequirements
from adjungo.core.problem import ProblemStructure, Linearity
from adjungo.core.method import GLMethod, StageType
from adjungo.methods.runge_kutta import explicit_euler, rk4, sdirk2, implicit_midpoint
from adjungo.solvers.factory import create_stage_solver
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.solvers.sdirk import SDIRKStageSolver
from adjungo.solvers.dirk import DIRKStageSolver
from adjungo.solvers.implicit import ImplicitStageSolver


def test_requirements_explicit_method():
    """Test requirements for explicit methods."""
    method = rk4()
    problem = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=False,
    )

    req = deduce_requirements(method, problem, state_dim=5)

    assert req.needs_newton is False
    assert req.newton_system_size == 0
    assert req.factorizations_per_step == 0
    assert req.can_reuse_across_stages is False
    assert req.can_reuse_across_steps is False
    assert req.store_jacobians is True
    assert req.store_stage_values is True
    assert req.trajectory_vectors_per_step == method.s + method.r


def test_requirements_sdirk_linear():
    """Test requirements for SDIRK with linear problem."""
    method = sdirk2()
    problem = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=False,
    )

    req = deduce_requirements(method, problem, state_dim=5)

    assert req.needs_newton is False  # Linear problem
    assert req.factorizations_per_step == 1  # Can reuse
    assert req.factorization_size == 5
    assert req.can_reuse_across_stages is True
    assert req.can_reuse_across_steps is True  # Constant Jacobian


def test_requirements_sdirk_nonlinear():
    """Test requirements for SDIRK with nonlinear problem."""
    method = sdirk2()
    problem = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=False,
    )

    req = deduce_requirements(method, problem, state_dim=5)

    assert req.needs_newton is True  # Nonlinear problem
    assert req.newton_system_size == 5
    assert req.factorization_size == 5
    # Cannot reuse across stages if control-dependent
    assert req.can_reuse_across_stages is False
    assert req.can_reuse_across_steps is False


def test_requirements_sdirk_constant_jacobian():
    """Test SDIRK with constant Jacobian can reuse factorization."""
    method = sdirk2()
    problem = ProblemStructure(
        linearity=Linearity.SEMILINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=False,
    )

    req = deduce_requirements(method, problem, state_dim=5)

    assert req.can_reuse_across_stages is True
    assert req.can_reuse_across_steps is True
    assert req.factorizations_per_step == 1


def test_requirements_dirk():
    """Test requirements for DIRK methods."""
    # Create a DIRK method (varying diagonal)
    A = np.array([[0.3, 0.0], [0.2, 0.5]])
    U = np.array([[1.0], [1.0]])
    B = np.array([[0.5, 0.5]])
    V = np.array([[1.0]])
    c = np.array([0.3, 0.7])

    method = GLMethod(A=A, U=U, B=B, V=V, c=c)
    assert method.stage_type == StageType.DIRK

    problem = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=False,
    )

    req = deduce_requirements(method, problem, state_dim=5)

    assert req.needs_newton is True
    assert req.newton_system_size == 5
    assert req.factorizations_per_step == method.s  # One per stage
    assert req.can_reuse_across_stages is False


def test_requirements_implicit():
    """Test requirements for fully implicit methods."""
    method = implicit_midpoint()
    problem = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=False,
    )

    req = deduce_requirements(method, problem, state_dim=5)

    assert req.needs_newton is True
    assert req.newton_system_size == 5 * method.s  # Full system
    assert req.factorizations_per_step == 1
    assert req.factorization_size == 5 * method.s


def test_factory_explicit_method():
    """Test factory creates explicit solver for explicit methods."""
    method = rk4()
    problem = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=False,
    )
    req = deduce_requirements(method, problem, state_dim=5)

    solver = create_stage_solver(method, req, problem)
    assert isinstance(solver, ExplicitStageSolver)


def test_factory_sdirk_method():
    """Test factory creates SDIRK solver for SDIRK methods."""
    method = sdirk2()
    problem = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=False,
    )
    req = deduce_requirements(method, problem, state_dim=5)

    solver = create_stage_solver(method, req, problem)
    assert isinstance(solver, SDIRKStageSolver)


def test_factory_dirk_method():
    """Test factory creates DIRK solver for DIRK methods."""
    A = np.array([[0.3, 0.0], [0.2, 0.5]])
    U = np.array([[1.0], [1.0]])
    B = np.array([[0.5, 0.5]])
    V = np.array([[1.0]])
    c = np.array([0.3, 0.7])

    method = GLMethod(A=A, U=U, B=B, V=V, c=c)
    problem = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=False,
    )
    req = deduce_requirements(method, problem, state_dim=5)

    solver = create_stage_solver(method, req, problem)
    assert isinstance(solver, DIRKStageSolver)


def test_factory_implicit_method():
    """Test factory creates implicit solver for fully implicit methods."""
    # Create a fully implicit method
    A = np.array([[0.25, 0.25 - np.sqrt(3)/6], [0.25 + np.sqrt(3)/6, 0.25]])
    U = np.array([[1.0], [1.0]])
    B = np.array([[0.5, 0.5]])
    V = np.array([[1.0]])
    c = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])

    method = GLMethod(A=A, U=U, B=B, V=V, c=c)
    assert method.stage_type == StageType.IMPLICIT

    problem = ProblemStructure(
        linearity=Linearity.NONLINEAR,
        jacobian_constant=False,
        jacobian_control_dependent=True,
        has_second_derivatives=False,
    )
    req = deduce_requirements(method, problem, state_dim=5)

    solver = create_stage_solver(method, req, problem)
    assert isinstance(solver, ImplicitStageSolver)


def test_problem_structure_linearity_classification():
    """Test different linearity classifications."""
    linearity_types = [
        Linearity.LINEAR,
        Linearity.BILINEAR,
        Linearity.QUASILINEAR,
        Linearity.SEMILINEAR,
        Linearity.NONLINEAR,
    ]

    method = sdirk2()

    for lin_type in linearity_types:
        problem = ProblemStructure(
            linearity=lin_type,
            jacobian_constant=False,
            jacobian_control_dependent=True,
            has_second_derivatives=False,
        )

        req = deduce_requirements(method, problem, state_dim=5)

        # Newton needed only for nonlinear/semilinear
        if lin_type in [Linearity.NONLINEAR, Linearity.SEMILINEAR]:
            assert req.needs_newton is True
        else:
            assert req.needs_newton is False


def test_trajectory_storage_requirements():
    """Test trajectory storage requirements are computed correctly."""
    for method_func in [explicit_euler, rk4, sdirk2]:
        method = method_func()
        problem = ProblemStructure(
            linearity=Linearity.LINEAR,
            jacobian_constant=True,
            jacobian_control_dependent=False,
            has_second_derivatives=False,
        )

        req = deduce_requirements(method, problem, state_dim=5)

        # Should store s internal + r external stages per step
        expected = method.s + method.r
        assert req.trajectory_vectors_per_step == expected
