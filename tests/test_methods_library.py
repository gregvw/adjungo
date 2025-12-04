"""Tests for standard method tableaux library."""

import numpy as np
import pytest

from adjungo.methods.runge_kutta import (
    explicit_euler,
    rk4,
    heun,
    implicit_midpoint,
    gauss2,
    sdirk2,
    sdirk3,
)
from adjungo.methods.multistep import bdf2, bdf3, adams_bashforth2, adams_moulton2
from adjungo.methods.glm import create_custom_glm, validate_glm
from adjungo.methods.imex import imex_ark2, imex_ark3
from adjungo.core.method import StageType, PropType


def test_explicit_euler_structure():
    """Test explicit Euler method structure."""
    method = explicit_euler()

    assert method.s == 1
    assert method.r == 1
    assert method.stage_type == StageType.EXPLICIT
    assert method.prop_type == PropType.IDENTITY
    assert np.allclose(method.A, 0.0)
    assert np.allclose(method.c, 0.0)


def test_rk4_structure():
    """Test RK4 method structure."""
    method = rk4()

    assert method.s == 4
    assert method.r == 1
    assert method.stage_type == StageType.EXPLICIT
    assert method.prop_type == PropType.IDENTITY

    # Check c values
    expected_c = np.array([0.0, 0.5, 0.5, 1.0])
    assert np.allclose(method.c, expected_c)

    # Check weights sum to 1
    assert np.isclose(np.sum(method.B), 1.0)


def test_rk4_butcher_tableau():
    """Test RK4 Butcher tableau values."""
    method = rk4()

    # Check A is strictly lower triangular
    assert np.allclose(method.A, np.tril(method.A, -1))

    # Check specific RK4 coefficients
    assert np.isclose(method.A[1, 0], 0.5)
    assert np.isclose(method.A[2, 1], 0.5)
    assert np.isclose(method.A[3, 2], 1.0)

    # Check weights
    expected_b = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
    assert np.allclose(method.B[0], expected_b)


def test_heun_structure():
    """Test Heun's method (explicit 2nd order)."""
    method = heun()

    assert method.s == 2
    assert method.stage_type == StageType.EXPLICIT

    # Check weights
    expected_b = np.array([0.5, 0.5])
    assert np.allclose(method.B[0], expected_b)


def test_implicit_midpoint_structure():
    """Test implicit midpoint rule."""
    method = implicit_midpoint()

    assert method.s == 1
    assert method.stage_type == StageType.SDIRK
    assert np.isclose(method.sdirk_gamma, 0.5)


def test_gauss2_structure():
    """Test 2-stage Gauss-Legendre method."""
    method = gauss2()

    assert method.s == 2
    assert method.stage_type == StageType.IMPLICIT

    # Gauss methods are fully implicit and symmetric
    # Check symmetry
    assert np.isclose(method.c[0] + method.c[1], 1.0)


def test_sdirk2_structure():
    """Test SDIRK2 method."""
    method = sdirk2()

    assert method.s == 2
    assert method.stage_type == StageType.SDIRK

    # Check constant diagonal
    gamma = method.sdirk_gamma
    assert gamma is not None
    assert np.isclose(method.A[0, 0], gamma)
    assert np.isclose(method.A[1, 1], gamma)

    # Check c values
    assert np.isclose(method.c[0], gamma)


def test_sdirk3_structure():
    """Test SDIRK3 method."""
    method = sdirk3()

    assert method.s == 3
    assert method.stage_type == StageType.SDIRK

    # Check constant diagonal
    gamma = method.sdirk_gamma
    assert gamma is not None
    for i in range(3):
        assert np.isclose(method.A[i, i], gamma)


def test_bdf2_structure():
    """Test BDF2 multistep method."""
    method = bdf2()

    assert method.s == 1  # Single stage
    assert method.r == 2  # Two-step method

    # Check shift matrix
    assert method.prop_type == PropType.SHIFT


def test_bdf3_structure():
    """Test BDF3 multistep method."""
    method = bdf3()

    assert method.s == 1
    assert method.r == 3  # Three-step method

    # Check that it's implicit
    assert not np.isclose(method.A[0, 0], 0.0)


def test_adams_bashforth2_structure():
    """Test Adams-Bashforth 2-step method."""
    method = adams_bashforth2()

    assert method.s == 2  # Two stages for history evaluation
    assert method.r == 2  # Two-step method
    assert method.stage_type == StageType.EXPLICIT


def test_adams_moulton2_structure():
    """Test Adams-Moulton 2-step method."""
    method = adams_moulton2()

    assert method.s == 2
    assert method.r == 2

    # Adams-Moulton is implicit
    assert not np.isclose(method.A[0, 0], 0.0)


def test_create_custom_glm():
    """Test creating custom GLM."""
    A = np.array([[0.0]])
    U = np.array([[1.0]])
    B = np.array([[1.0]])
    V = np.array([[1.0]])
    c = np.array([0.0])

    method = create_custom_glm(A, U, B, V, c)

    assert method.s == 1
    assert method.r == 1
    assert np.array_equal(method.A, A)


def test_validate_glm_consistent():
    """Test GLM validation for consistent methods."""
    # Forward Euler should be consistent
    method = explicit_euler()
    assert validate_glm(method, order=1)

    # RK4 should be consistent
    method = rk4()
    assert validate_glm(method, order=4)


def test_validate_glm_inconsistent():
    """Test GLM validation catches inconsistent methods."""
    # Create inconsistent method (B @ e ≠ V @ e)
    A = np.array([[0.0]])
    U = np.array([[1.0]])
    B = np.array([[2.0]])  # Wrong!
    V = np.array([[1.0]])
    c = np.array([0.0])

    method = create_custom_glm(A, U, B, V, c)

    # Should fail consistency check
    assert not validate_glm(method, order=1)


def test_imex_ark2_structure():
    """Test IMEX ARK2 pair."""
    imex = imex_ark2()

    assert imex.explicit.s == 2
    assert imex.implicit.s == 2

    # Explicit part should be explicit
    assert imex.explicit.stage_type == StageType.EXPLICIT

    # Implicit part should be SDIRK
    assert imex.implicit.stage_type == StageType.SDIRK

    # Shared components should match
    assert np.array_equal(imex.U, imex.explicit.U)
    assert np.array_equal(imex.U, imex.implicit.U)


def test_imex_ark3_structure():
    """Test IMEX ARK3 pair."""
    imex = imex_ark3()

    assert imex.explicit.s == 3
    assert imex.implicit.s == 3

    assert imex.explicit.stage_type == StageType.EXPLICIT
    assert imex.implicit.stage_type == StageType.SDIRK


def test_rk_methods_order_conditions():
    """Test that RK methods satisfy basic order conditions."""
    methods = [
        (explicit_euler(), 1),
        (heun(), 2),
        (rk4(), 4),
    ]

    for method, order in methods:
        # Basic consistency: sum of weights = 1
        assert np.isclose(np.sum(method.B), 1.0)

        # For order >= 2: B @ c = 1/2
        if order >= 2:
            bc = method.B @ method.c
            assert np.isclose(bc, 0.5)


def test_method_immutability():
    """Test that methods are immutable (frozen dataclass)."""
    method = rk4()

    with pytest.raises(Exception):  # FrozenInstanceError or similar
        method.s = 10


def test_all_methods_have_correct_shapes():
    """Test that all methods have consistent array shapes."""
    methods = [
        explicit_euler(),
        rk4(),
        heun(),
        implicit_midpoint(),
        gauss2(),
        sdirk2(),
        sdirk3(),
        bdf2(),
        bdf3(),
        adams_bashforth2(),
        adams_moulton2(),
    ]

    for method in methods:
        s, r = method.s, method.r

        # Check shapes
        assert method.A.shape == (s, s), f"A shape mismatch for {method}"
        assert method.U.shape == (s, r), f"U shape mismatch for {method}"
        assert method.B.shape == (r, s), f"B shape mismatch for {method}"
        assert method.V.shape == (r, r), f"V shape mismatch for {method}"
        assert method.c.shape == (s,), f"c shape mismatch for {method}"


def test_sdirk_gamma_extraction():
    """Test that SDIRK gamma is correctly extracted."""
    methods = [sdirk2(), sdirk3(), implicit_midpoint()]

    for method in methods:
        assert method.stage_type == StageType.SDIRK
        gamma = method.sdirk_gamma
        assert gamma is not None

        # Check all diagonal elements equal gamma
        for i in range(method.s):
            assert np.isclose(method.A[i, i], gamma)


def test_explicit_stage_indices():
    """Test explicit stage index detection."""
    # RK4: all stages are explicit
    method = rk4()
    assert len(method.explicit_stage_indices) == 4

    # SDIRK2: no explicit stages (all have diagonal)
    method = sdirk2()
    assert len(method.explicit_stage_indices) == 0

    # Could create DIRK with some explicit stages
    A = np.array([[0.0, 0.0], [0.5, 0.3]])
    U = np.array([[1.0], [1.0]])
    B = np.array([[0.5, 0.5]])
    V = np.array([[1.0]])
    c = np.array([0.0, 0.8])

    method = create_custom_glm(A, U, B, V, c)
    assert 0 in method.explicit_stage_indices  # First stage is explicit
    assert 1 not in method.explicit_stage_indices  # Second is implicit
