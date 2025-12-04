"""Tests for utility functions."""

import numpy as np
import pytest

from adjungo.utils.kronecker import (
    kronecker_eye,
    eye_kronecker,
    block_matvec,
    block_solve,
)


def test_kronecker_eye_basic():
    """Test A ⊗ I_n computation."""
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    n = 3

    result = kronecker_eye(A, n)

    # Should be (2*3, 2*3) = (6, 6)
    assert result.shape == (6, 6)

    # Check block structure
    expected = np.block([
        [2.0 * np.eye(3), 1.0 * np.eye(3)],
        [0.0 * np.eye(3), 3.0 * np.eye(3)],
    ])

    assert np.allclose(result, expected)


def test_kronecker_eye_identity():
    """Test A ⊗ I with A = I."""
    A = np.eye(3)
    n = 2

    result = kronecker_eye(A, n)

    # Should be block diagonal with identity blocks
    expected = np.eye(6)
    assert np.allclose(result, expected)


def test_eye_kronecker_basic():
    """Test I_n ⊗ B computation."""
    n = 2
    B = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = eye_kronecker(n, B)

    # Should be (2*2, 2*2) = (4, 4)
    assert result.shape == (4, 4)

    # Check block structure
    expected = np.block([
        [B, np.zeros_like(B)],
        [np.zeros_like(B), B],
    ])

    assert np.allclose(result, expected)


def test_block_matvec_basic():
    """Test (A ⊗ I_n) @ x computation."""
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    s, n = 2, 3

    x = np.arange(s * n, dtype=float)  # [0, 1, 2, 3, 4, 5]

    result = block_matvec(A, x, s, n)

    # Compare with explicit Kronecker product
    A_kron = kronecker_eye(A, n)
    expected = A_kron @ x

    assert np.allclose(result, expected)


def test_block_matvec_with_matrix_input():
    """Test block_matvec with matrix-shaped input."""
    A = np.array([[1.0, 0.5], [0.0, 2.0]])
    s, n = 2, 4

    x = np.random.randn(s, n)

    result = block_matvec(A, x.ravel(), s, n)

    # Expected result
    expected = np.zeros((s, n))
    for i in range(s):
        for j in range(s):
            expected[i] += A[i, j] * x[j]

    assert np.allclose(result, expected.ravel())


def test_block_solve_lower_triangular():
    """Test block solve with lower triangular A."""
    A = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 1.5, 1.0]])
    s, n = 3, 2

    b = np.random.randn(s * n)

    x = block_solve(A, b, s, n)

    # Verify solution
    residual = block_matvec(A, x, s, n) - b
    assert np.linalg.norm(residual) < 1e-10


def test_block_solve_upper_triangular():
    """Test block solve with upper triangular A."""
    A = np.array([[2.0, 1.0, 0.5], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]])
    s, n = 3, 2

    b = np.random.randn(s * n)

    x = block_solve(A, b, s, n)

    # Verify solution
    residual = block_matvec(A, x, s, n) - b
    assert np.linalg.norm(residual) < 1e-10


def test_block_solve_diagonal():
    """Test block solve with diagonal A."""
    A = np.diag([2.0, 3.0, 1.0])
    s, n = 3, 4

    b = np.random.randn(s * n)

    x = block_solve(A, b, s, n)

    # Verify solution
    residual = block_matvec(A, x, s, n) - b
    assert np.linalg.norm(residual) < 1e-10


def test_block_solve_dense():
    """Test block solve with dense A."""
    A = np.random.randn(3, 3)
    s, n = 3, 2

    b = np.random.randn(s * n)

    x = block_solve(A, b, s, n)

    # Verify solution
    residual = block_matvec(A, x, s, n) - b
    assert np.linalg.norm(residual) < 1e-10


def test_kronecker_eye_scalar_multiplication():
    """Test that scalar A gives correct block structure."""
    A = np.array([[5.0]])
    n = 3

    result = kronecker_eye(A, n)

    expected = 5.0 * np.eye(3)
    assert np.allclose(result, expected)


def test_block_operations_consistency():
    """Test that block operations are consistent with explicit Kronecker."""
    A = np.random.randn(4, 4)
    s, n = 4, 3

    # Generate random vector
    x = np.random.randn(s * n)

    # Block matvec
    y_block = block_matvec(A, x, s, n)

    # Explicit Kronecker
    A_kron = kronecker_eye(A, n)
    y_kron = A_kron @ x

    assert np.allclose(y_block, y_kron)


def test_kronecker_properties():
    """Test mathematical properties of Kronecker products."""
    A = np.random.randn(3, 3)
    n = 2

    K = kronecker_eye(A, n)

    # Test shape
    assert K.shape == (3 * n, 3 * n)

    # Test rank (if A is full rank, so is A ⊗ I)
    if np.linalg.matrix_rank(A) == 3:
        assert np.linalg.matrix_rank(K) == 6

    # Test determinant: det(A ⊗ I_n) = det(A)^n
    det_A = np.linalg.det(A)
    det_K = np.linalg.det(K)
    assert np.isclose(det_K, det_A ** n)
