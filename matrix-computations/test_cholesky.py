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
    expected = np.array([[3.0, 1, 1, 0], [0, 2, 1, 1], [0, 0, 4, 1], [0, 0, 0, 5]])
    assert np.linalg.det(expected) > 0
    A = expected.T @ expected
    R = np.zeros_like(A)
    cholesky(A, R)
    assert np.allclose(R, expected), R


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
    Puts the result in R such that RᵀR = A
    """
    n = A.shape[0]
    for i in range(n):
        R[i, i] = np.sqrt(A[i, i] - R[:i, i] @ R[:i, i])
        if R[i, i] == 0:
            raise LinAlgError("Matrix is not positive definite")
        for j in range(i + 1, n):
            R[i, j] = (A[i, j] - R[:i, i] @ R[:i, j]) / R[i, i]
