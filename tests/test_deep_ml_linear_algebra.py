import numpy as np
import pytest

from nlp_research.deep_ml.linear_algebra import (
    calculate_covariance_matrix,
    matrix_dot_vector,
    solve_jacobi,
    svd_2x2_singular_values,
)


@pytest.mark.parametrize(
    'a, b, expected',
    [
        ([[1, 2, 3], [4, 5, 6]], [1, 2, 3], [14, 32]),
        ([[1, 2, 3], [4, 5, 6]], [1, 2], -1),
    ],
)
def test_matrix_dot_vector(a, b, expected):
    assert matrix_dot_vector(a, b) == expected


@pytest.mark.parametrize(
    'vectors, expected',
    [
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_calculate_covariance_matrix(vectors, expected):
    assert np.allclose(calculate_covariance_matrix(vectors), expected)


@pytest.mark.parametrize(
    'A, b, n, expected',
    [
        (np.array([[4, 1], [1, 3]]), np.array([1, 2]), 10, [0.0909, 0.6364]),
    ],
)
def test_solve_jacobi(A, b, n, expected):
    assert solve_jacobi(A, b, n) == expected


@pytest.mark.parametrize(
    'A, expected',
    [
        (
            np.array([[2, 1], [1, 2]]),
            (
                np.array([[-0.70710678, -0.70710678], [-0.70710678, 0.70710678]]),
                np.array([3.0, 1.0]),
                np.array([[-0.70710678, -0.70710678], [-0.70710678, 0.70710678]]),
            ),
        )
    ],
)
def test_svd_2x2_singular_values(A, expected):
    u_expected, s_expected, v_t_expected = expected
    u, s, v_t = svd_2x2_singular_values(A)
    assert np.allclose(u, u_expected)
    assert np.allclose(s, s_expected)
    assert np.allclose(v_t, v_t_expected)
