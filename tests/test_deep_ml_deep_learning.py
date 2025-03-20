import numpy as np
import pytest

from nlp_research.deep_ml import (
    adaboost_fit,
    adaboost_predict,
    feature_scaling,
    linear_regression_gradient_descent,
    linear_regression_normal_equation,
    rnn_forward,
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


@pytest.mark.parametrize(
    'input_sequence, initial_hidden_state, Wx, Wh, b, expected',
    [
        (
            np.array([[1.0], [2.0], [3.0]]),
            np.array([0.0]),
            np.array([[0.5]]),
            np.array([[0.8]]),
            np.array([0.0]),
            [0.9759],
        ),
        (
            np.array([[0.5], [0.1], [-0.2]]),
            np.array([0.0]),
            np.array([[1.0]]),
            np.array([[0.5]]),
            np.array([0.1]),
            [0.118],
        ),
    ],
)
def test_rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b, expected):
    assert np.allclose(rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b), expected)


@pytest.mark.parametrize(
    'X, y, n_clf, expected_len, expected_feature_idx',
    [
        (
            np.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
            np.array([1, 1, -1, -1]),
            3,
            3,
            0,  # Feature index 0 should be selected
        ),
    ],
)
def test_adaboost_fit(X, y, n_clf, expected_len, expected_feature_idx):
    classifiers = adaboost_fit(X, y, n_clf)
    assert len(classifiers) == expected_len
    # Verify that the first feature is chosen (in our test case)
    assert classifiers[0][0] == expected_feature_idx


@pytest.mark.parametrize(
    'X, y, n_clf',
    [
        (
            np.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
            np.array([1, 1, -1, -1]),
            3,
        ),
    ],
)
def test_adaboost_predict(X, y, n_clf):
    classifiers = adaboost_fit(X, y, n_clf)
    predictions = adaboost_predict(X, classifiers)
    # Ensure predictions have the same shape as y
    assert predictions.shape == y.shape
    # Check that predictions are either -1 or 1
    assert np.all(np.isin(predictions, [-1, 1]))
    # For this simple dataset, we should achieve perfect accuracy
    assert np.all(predictions == y)
