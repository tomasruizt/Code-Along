from numpy.linalg import LinAlgError
import numpy as np
from numba import jit
import pytest


def test_cholseky_1d():
    A = np.array([[4.0]])
    expected = np.array([[2.0]])
    R = np.zeros_like(A)
    cholesky(A, R)
    assert np.allclose(R, expected), R


def test_cholesky_2d():
    A = np.array([[4.0, 0], [0, 9]])
    expected = np.array([[2, 0], [0, 3]])
    R = np.zeros_like(A)
    cholesky(A, R)
    assert np.allclose(R, expected), R


def test_cholesky_3d():
    expected = np.array([[3.0, 1, 1], [0, 2, 1], [0, 0, 4]])
    assert np.linalg.det(expected) > 0
    A = expected.T @ expected
    R = np.zeros_like(A)
    cholesky(A, R)
    assert np.allclose(R, expected), R


def test_cholesky_4d():
    expected, A = some_4d_matrix()
    R = np.zeros_like(A)
    cholesky(A, R)
    assert np.allclose(R, expected), R


def some_4d_matrix():
    expected = np.array([[3.0, 1, 1, 0], [0, 2, 1, 1], [0, 0, 4, 1], [0, 0, 0, 5]])
    assert np.linalg.det(expected) > 0
    A = expected.T @ expected
    return expected, A


def test_cholesky_outer_product():
    expected, A = some_4d_matrix()
    cholesky_op(A)
    assert np.allclose(A, expected)


def test_forward_substition():
    """Solve Ax = b, where A is lower triangular"""
    A = np.array([[1.0, 0, 0], [2, 3, 0], [4, 5, 6]])
    expected_x = np.array([7.0, 8.0, 9.0])
    b = A @ expected_x
    forward_substitution(A, b)
    x = b
    assert np.allclose(x, expected_x)


@jit(nopython=True)
def forward_substitution(A, b):
    """Solves Ax = b, for lower triangular matrix A. Fills x in place of b."""
    n = len(b)
    for idx in range(n):
        for prev in range(idx):
            b[idx] -= A[idx, prev] * b[prev]
        b[idx] /= A[idx, idx]


@jit(nopython=True)
def cholesky_op(A: np.ndarray):
    """
    Calculate the Cholesky decomposition RᵀR = A (in-place)
    Where R is upper triangular (n x n)
    In the outer product form
    """
    # set zeros to make sure we don't acces the lower triangle
    n = A.shape[0]
    for row_idx in range(1, n):  # skip first row
        for col_idx in range(row_idx):
            A[row_idx, col_idx] = 0

    for it_idx in range(n):
        A[it_idx, it_idx] = np.sqrt(A[it_idx, it_idx])
        for col_idx in range(it_idx + 1, n):
            A[it_idx, col_idx] /= A[it_idx, it_idx]
        for row_idx in range(it_idx + 1, n):
            for col_idx in range(row_idx, n):
                A[row_idx, col_idx] -= A[it_idx, row_idx] * A[it_idx, col_idx]


def test_failure():
    singular_matrix = np.array([[0.0, 2.0], [1.0, 2.0]])

    # check that numpy raises an exception
    with pytest.raises(LinAlgError):
        np.linalg.cholesky(singular_matrix)

    # check that my implementation raises an exception
    with pytest.raises(LinAlgError):
        R = np.zeros_like(singular_matrix)
        cholesky(singular_matrix, R)


@jit(nopython=True)
def cholesky(A, R):
    """
    Implements the Cholesky decomposition of a positive definite matrix A
    In inner product form
    Puts the result in R such that RᵀR = A
    """
    n = A.shape[0]
    for i in range(n):
        R[i, i] = np.sqrt(A[i, i] - R[:i, i] @ R[:i, i])
        if R[i, i] == 0:
            raise LinAlgError("Matrix is not positive definite")
        for j in range(i + 1, n):
            R[i, j] = (A[i, j] - R[:i, i] @ R[:i, j]) / R[i, i]
