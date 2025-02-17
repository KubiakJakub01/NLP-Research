import numpy as np
import pytest

from nlp_research.deep_ml.linear_algebra import (
    calculate_covariance_matrix,
    feature_scaling,
    linear_regression_gradient_descent,
    linear_regression_normal_equation,
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
    'A',
    [
        (np.array([[2, 1], [1, 2]])),
        (np.array([[1, 2], [2, 1]])),
        (np.array([[1, 0], [0, 1]])),
    ],
)
def test_svd_2x2_singular_values(A):
    _, s_expected, _ = np.linalg.svd(A)
    _, s, _ = svd_2x2_singular_values(A)
    assert np.allclose(s, s_expected)


@pytest.mark.parametrize(
    'X, y, expected',
    [
        (np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), [0.0, 1.0]),
    ],
)
def test_linear_regression_normal_equation(X, y, expected):
    assert np.allclose(linear_regression_normal_equation(X, y), expected)


@pytest.mark.parametrize(
    'X, y, alpha, iterations, expected',
    [
        (np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000, [0.1107, 0.9513]),
    ],
)
def test_linear_regression_gradient_descent(X, y, alpha, iterations, expected):
    assert np.allclose(linear_regression_gradient_descent(X, y, alpha, iterations), expected)


@pytest.mark.parametrize(
    'X, expected',
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            (
                np.array([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]),
                np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            ),
        )
    ],
)
def test_feature_scaling(X, expected):
    expected_standardized_data, expected_normalized_data = expected
    standardized_data, normalized_data = feature_scaling(X)
    assert np.allclose(standardized_data, expected_standardized_data)
    assert np.allclose(normalized_data, expected_normalized_data)
