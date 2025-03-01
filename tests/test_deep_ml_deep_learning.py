import numpy as np
import pytest

from nlp_research.deep_ml import (
    feature_scaling,
    linear_regression_gradient_descent,
    linear_regression_normal_equation,
)


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
